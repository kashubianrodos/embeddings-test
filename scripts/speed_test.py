"""Detailed speed test: per-text latency statistics for each model.

Unlike benchmark.py (which only reports totals), this measures each
embedding call individually and reports mean / median / p95 / min / max
latency and throughput, after a warmup round to exclude model-load time.
"""
import os
import statistics
import time
from datetime import datetime

from common import (MODELS, check_model_availability, client, die,
                    ensure_output_dir, load_samples)

WARMUP_CALLS = 3


def percentile(sorted_vals, p):
    k = (len(sorted_vals) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def time_each(model_name, texts):
    """Embed texts one by one, returning per-call latencies in ms."""
    latencies = []
    for text in texts:
        start = time.perf_counter()
        response = client.embeddings(model=model_name, prompt=text)
        latencies.append((time.perf_counter() - start) * 1000)
        if "embedding" not in response:
            raise RuntimeError(f"Missing 'embedding' key for '{text[:30]}...'")
    return latencies


def run_model(model_name, texts):
    ok, err = check_model_availability(model_name)
    if not ok:
        print(f"❌ Skipping {model_name}: {err}")
        return {"status": "error", "error": err}

    try:
        print(f"✅ {model_name}: warmup ({WARMUP_CALLS} calls)...")
        time_each(model_name, texts[:WARMUP_CALLS])

        print(f"✅ {model_name}: timing {len(texts)} texts...")
        lat = time_each(model_name, texts)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    s = sorted(lat)
    total_s = sum(lat) / 1000
    return {
        "status": "ok",
        "n": len(lat),
        "mean_ms": statistics.mean(lat),
        "median_ms": statistics.median(lat),
        "stdev_ms": statistics.stdev(lat) if len(lat) > 1 else 0.0,
        "p95_ms": percentile(s, 95),
        "min_ms": s[0],
        "max_ms": s[-1],
        "total_s": total_s,
        "throughput_tps": len(lat) / total_s if total_s else 0.0,
    }


def main():
    texts_en, texts_pl = load_samples()
    texts = texts_en + texts_pl
    if not texts:
        die("No sample texts loaded.")

    started = datetime.now()
    results = {}
    for cfg in MODELS:
        print(f"\n--- {cfg['name']} ---")
        results[cfg["name"]] = run_model(cfg["name"], texts)

    rows = [
        ("Status", lambda r: "✅ ok" if r["status"] == "ok" else "❌ error"),
        ("Texts timed", lambda r: str(r.get("n", "N/A"))),
        ("Mean latency (ms)", lambda r: f"{r['mean_ms']:.1f}" if r["status"] == "ok" else "N/A"),
        ("Median latency (ms)", lambda r: f"{r['median_ms']:.1f}" if r["status"] == "ok" else "N/A"),
        ("Std dev (ms)", lambda r: f"{r['stdev_ms']:.1f}" if r["status"] == "ok" else "N/A"),
        ("P95 latency (ms)", lambda r: f"{r['p95_ms']:.1f}" if r["status"] == "ok" else "N/A"),
        ("Min latency (ms)", lambda r: f"{r['min_ms']:.1f}" if r["status"] == "ok" else "N/A"),
        ("Max latency (ms)", lambda r: f"{r['max_ms']:.1f}" if r["status"] == "ok" else "N/A"),
        ("Total time (s)", lambda r: f"{r['total_s']:.3f}" if r["status"] == "ok" else "N/A"),
        ("Throughput (texts/s)", lambda r: f"{r['throughput_tps']:.2f}" if r["status"] == "ok" else "N/A"),
    ]
    lines = [
        "# Embedding Model Speed Test Report",
        "",
        f"Generated: {started:%Y-%m-%d %H:%M:%S} — {len(texts)} texts (50 EN + 50 PL) per model, "
        f"after {WARMUP_CALLS} warmup calls",
        "",
        "| Metric | " + " | ".join(c["name"] for c in MODELS) + " |",
        "| :--- | " + " | ".join(":---:" for _ in MODELS) + " |",
    ]
    for label, fn in rows:
        lines.append(f"| **{label}** | " + " | ".join(fn(results[c["name"]]) for c in MODELS) + " |")

    errors = [f"- **{n}**: {r['error']}" for n, r in results.items() if r["status"] != "ok"]
    if errors:
        lines += ["", "## Errors", *errors]

    out_dir = ensure_output_dir()
    report_path = os.path.join(out_dir, f"speed_report_{started:%Y%m%d_%H%M%S}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n✅ Report written to: {report_path}")


if __name__ == "__main__":
    main()
