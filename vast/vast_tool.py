#!/usr/bin/env python3
"""Local helper for the vast/*.sh scripts (stdlib only, runs on macOS/Linux python3).

  rank     rank `vastai search offers --raw` output by estimated TOTAL cost per run
  status   print actual_status from `vastai show instance --raw` (stdin), or "gone"
  new-id   print the new instance id from `vastai create instance --raw` (stdin)
  get      print one key of a JSON file
  summary  write output_vast/<id>/run_summary.{md,json}: measured phases vs. estimate
"""
import argparse
import ast
import json
import re
import sys
import time


def _load_any(text):
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None


def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------------------- rank
def estimate(o, a):
    """Return a dict with the estimate for offer `o`, or None if it must be skipped."""
    inet_down = _num(o.get("inet_down"), 0.0)
    if inet_down <= 0:
        return None
    down_cost = _num(o.get("inet_down_cost"), 0.0)
    if a.max_inet_cost is not None and down_cost > a.max_inet_cost:
        return None

    dph_total = _num(o.get("dph_total"))
    dph_base = _num(o.get("dph_base"))
    if dph_total is not None and dph_base is not None and dph_total >= dph_base:
        storage_h = dph_total - dph_base
    elif o.get("storage_cost") is not None:  # $/GB/month
        storage_h = _num(o.get("storage_cost"), 0.0) * a.disk / 730.0
    else:
        storage_h = 0.0

    bid = None
    if a.interruptible:
        min_bid = _num(o.get("min_bid"))
        if not min_bid:
            return None
        bid = round(min_bid * a.bid_mult, 4)
        price_h = bid + storage_h
    else:
        if dph_total is None:
            return None
        price_h = dph_total  # searched with --storage <disk>, so storage is included

    dl_h = a.download_gb * 8000.0 / (inet_down * a.net_eff) / 3600.0
    hours = dl_h + a.setup_hours + a.bench_hours
    bandwidth = a.download_gb * down_cost
    return {
        "offer_id": o.get("id") or o.get("ask_contract_id"),
        "machine_id": o.get("machine_id"),
        "gpu_name": o.get("gpu_name"),
        "num_gpus": o.get("num_gpus"),
        "gpu_ram_gb": round(_num(o.get("gpu_ram"), 0) / 1000.0, 1),
        "compute_cap": o.get("compute_cap"),
        "cuda_max_good": o.get("cuda_max_good"),
        "geolocation": o.get("geolocation"),
        "reliability": _num(o.get("reliability2", o.get("reliability")), 0.0),
        "inet_down": inet_down,
        "inet_down_cost": down_cost,
        "interruptible": bool(a.interruptible),
        "bid_price": bid,
        "price_h": round(price_h, 4),
        "storage_h": round(storage_h, 4),
        "est_download_h": round(dl_h, 3),
        "est_hours": round(hours, 3),
        "est_bandwidth_usd": round(bandwidth, 3),
        "est_total_usd": round(price_h * hours + bandwidth, 3),
    }


def cmd_rank(a):
    with open(a.offers) as f:
        data = _load_any(f.read())
    offers = data.get("offers", []) if isinstance(data, dict) else (data or [])
    ranked = [e for e in (estimate(o, a) for o in offers) if e]
    ranked.sort(key=lambda e: (e["est_total_usd"], -e["reliability"]))
    if not ranked:
        print("No offers left after filtering (price data, bandwidth cost, bid).", file=sys.stderr)
        return 2

    kind = "interruptible bid" if a.interruptible else "on-demand"
    print(f"\nTop offers by estimated total run cost ({kind}; {a.download_gb:g} GB download, "
          f"{a.setup_hours:g} h setup + {a.bench_hours:g} h bench, net efficiency {a.net_eff:g}):", file=sys.stderr)
    print(f"{'offer':>10}  {'gpu':<16}{'$/h':>7}{'Mb/s':>7}{'$/GB':>7}{'dl h':>6}{'hours':>6}{'est $':>7}  where",
          file=sys.stderr)
    for e in ranked[:a.top]:
        print(f"{e['offer_id']:>10}  {str(e['gpu_name'])[:15]:<16}{e['price_h']:>7.3f}{e['inet_down']:>7.0f}"
              f"{e['inet_down_cost']:>7.4f}{e['est_download_h']:>6.2f}{e['est_hours']:>6.2f}"
              f"{e['est_total_usd']:>7.2f}  {e['geolocation'] or ''}", file=sys.stderr)

    chosen = ranked[0]
    if a.offer_id:
        match = [e for e in ranked if str(e["offer_id"]) == str(a.offer_id)]
        if not match:
            print(f"OFFER_ID={a.offer_id} is not among the filtered offers.", file=sys.stderr)
            return 3
        chosen = match[0]
    with open(a.out, "w") as f:
        json.dump(chosen, f, indent=1)
    return 0


# ----------------------------------------------------------------------------- small helpers
def cmd_status(_a):
    d = _load_any(sys.stdin.read())
    if d is None:                      # CLI/API error or empty output: don't conclude anything
        print("unknown")
        return 0
    if isinstance(d, dict) and "instances" in d:
        d = d["instances"]
    if isinstance(d, list):
        d = d[0] if d else None
    if not isinstance(d, dict) or not d:
        print("gone")                  # explicit {"instances": null} / empty list
    else:
        print(d.get("actual_status") or "unknown")
    return 0


def cmd_new_id(_a):
    text = sys.stdin.read()
    d = _load_any(text)
    if isinstance(d, dict) and d.get("new_contract"):
        print(d["new_contract"])
        return 0
    m = re.search(r"new_contract['\"]?\s*[:=]\s*(\d+)", text)
    if m:
        print(m.group(1))
        return 0
    print(text.strip(), file=sys.stderr)
    return 1


def cmd_get(a):
    with open(a.file) as f:
        v = json.load(f).get(a.key)
    print("" if v is None else v)
    return 0


# ----------------------------------------------------------------------------- summary
def _read_marks(path):
    """'<epoch> <event> [k=v ...]' lines -> {event: (epoch, {k: v})} keeping the LAST occurrence."""
    marks, counts = {}, {}
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2 or not parts[0].isdigit():
                    continue
                kv = dict(p.split("=", 1) for p in parts[2:] if "=" in p)
                marks[parts[1]] = (int(parts[0]), kv)
                counts[parts[1]] = counts.get(parts[1], 0) + 1
    except OSError:
        pass
    return marks, counts


def _dur(m, a, b):
    if a in m and b in m and m[b][0] >= m[a][0]:
        return m[b][0] - m[a][0]
    return None


def cmd_summary(a):
    with open(a.meta) as f:
        meta = json.load(f)
    local, _ = _read_marks(a.events)
    remote, rcount = _read_marks(a.timings)
    now = int(time.time())
    created = local.get("created", (meta.get("created_at", now), {}))[0]
    billed_h = (now - created) / 3600.0

    dl_s = _dur(remote, "download_start", "download_done")
    dl_bytes = _num(remote.get("download_done", (0, {}))[1].get("bytes"))
    eff_mbps = dl_bytes * 8 / 1e6 / dl_s if (dl_s and dl_bytes) else None
    dl_gb = dl_bytes / 1e9 if dl_bytes else meta.get("download_gb")
    actual_usd = meta["price_h"] * billed_h + (dl_gb or 0) * (meta.get("inet_down_cost") or 0)

    phases = [
        ("Instance created → setup started (image pull, boot)", _dur({**local, **remote}, "created", "setup_start")),
        ("Packages", _dur(remote, "setup_start", "packages_done")),
        ("Engine start", _dur(remote, "packages_done", "engine_ready")),
        ("Model download", dl_s),
        ("Model load + smoke test", _dur(remote, "download_done", "ready")),
        ("Benchmark", _dur(remote, "bench_start", "bench_done")),
    ]
    res = {
        "instance_id": meta.get("instance_id"), "mode": meta.get("mode"), "preset": meta.get("preset"),
        "gpu_name": meta.get("gpu_name"), "interruptible": meta.get("interruptible"),
        "price_h": meta.get("price_h"), "est_hours": meta.get("est_hours"), "est_total_usd": meta.get("est_total_usd"),
        "billed_hours_so_far": round(billed_h, 3), "actual_usd_estimate": round(actual_usd, 3),
        "download_bytes": dl_bytes, "download_seconds": dl_s,
        "advertised_inet_down_mbps": meta.get("inet_down"),
        "effective_inet_down_mbps": round(eff_mbps, 1) if eff_mbps else None,
        "measured_net_efficiency": round(eff_mbps / meta["inet_down"], 3) if (eff_mbps and meta.get("inet_down")) else None,
        "setup_runs": rcount.get("setup_start", 0), "outcome": a.outcome,
        "phases_seconds": {k: v for k, v in phases},
    }
    with open(f"{a.out_dir}/run_summary.json", "w") as f:
        json.dump(res, f, indent=1)

    def fm(s):
        return "N/A" if s is None else f"{s / 60:.1f} min"
    lines = [f"# vast.ai run summary — instance {res['instance_id']}", "",
             f"- Mode: `{res['mode']}` / preset `{res['preset']}` on **{res['gpu_name']}**"
             f" ({'interruptible' if res['interruptible'] else 'on-demand'}, ${res['price_h']:.3f}/h)",
             f"- Outcome: **{a.outcome}**",
             f"- Estimated: {res['est_hours']} h → ${res['est_total_usd']}",
             f"- Actual (billed time so far × price + download): {res['billed_hours_so_far']} h → "
             f"**${res['actual_usd_estimate']}**",
             f"- Download: {fm(dl_s)}, {round(dl_bytes / 1e9, 1) if dl_bytes else 'N/A'} GB, effective "
             f"{res['effective_inet_down_mbps']} Mb/s of {res['advertised_inet_down_mbps']} advertised "
             f"(NET_EFFICIENCY ≈ {res['measured_net_efficiency']})",
             f"- Setup runs: {res['setup_runs']} (more than 1 = instance was preempted and resumed)", "",
             "| Phase | Duration |", "| :--- | ---: |"]
    lines += [f"| {k} | {fm(v)} |" for k, v in phases]
    lines += ["", "Use the measured numbers to tune `NET_EFFICIENCY`, `SETUP_HOURS` and `BENCH_HOURS_*` in vast/.env."]
    with open(f"{a.out_dir}/run_summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rank")
    r.add_argument("--offers", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--disk", type=float, required=True)
    r.add_argument("--download-gb", type=float, required=True)
    r.add_argument("--setup-hours", type=float, required=True)
    r.add_argument("--bench-hours", type=float, required=True)
    r.add_argument("--net-eff", type=float, default=0.5)
    r.add_argument("--max-inet-cost", type=float, default=None)
    r.add_argument("--interruptible", type=int, default=0)
    r.add_argument("--bid-mult", type=float, default=1.25)
    r.add_argument("--offer-id", default="")
    r.add_argument("--top", type=int, default=10)

    sub.add_parser("status")
    sub.add_parser("new-id")
    g = sub.add_parser("get")
    g.add_argument("file")
    g.add_argument("key")

    s = sub.add_parser("summary")
    s.add_argument("--meta", required=True)
    s.add_argument("--events", required=True)
    s.add_argument("--timings", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--outcome", default="unknown")

    a = p.parse_args()
    return {"rank": cmd_rank, "status": cmd_status, "new-id": cmd_new_id,
            "get": cmd_get, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
