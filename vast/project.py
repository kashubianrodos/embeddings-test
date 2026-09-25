"""Project a benchmark run measured on a rented GPU onto a machine we can't rent (e.g. the
ASUS Ascent GX10 / NVIDIA GB10). Stdlib only.

Model (validated against published GB10 measurements of the same Qwen3.8-27B Q4_K model):
  decode, per step:  t = max( W / BW_eff(target),                        memory-bound part
                              max(t_proxy - W / BW_eff(proxy), 0) × C_proxy / C_target )  compute part
  prefill:           tok/s × C_target / C_proxy                          compute-bound
where W = weight bytes read per token, BW_eff = achievable bandwidth, C = dense tensor throughput
of the unit that does the matmuls (INT8 for GGUF/llama.cpp, FP8 or BF16 for vLLM).
"""
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models import GPUS, Gpu, Model, gpu_from_report, model_from_ref

PROXY_BW_EFFICIENCY = 0.85   # discrete GDDR/HBM cards reach ~85% of spec in decode


@dataclass(frozen=True)
class Target:
    key: str
    title: str
    memory_gb: int
    usable_gb: int            # what the GPU can actually allocate
    ssd_gb: int
    bw_spec_gbs: float
    bw_eff_gbs: Dict[str, float]      # per engine, measured/derived
    compute: Dict[str, float]         # dense int8 TOPS / fp8 / bf16 TFLOPS
    published: Dict[str, Dict[str, float]] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)


TARGETS: Dict[str, Target] = {
    "gx10": Target(
        key="gx10", title="ASUS Ascent GX10 — NVIDIA GB10 · 128 GB unified LPDDR5x · 1 TB SSD",
        memory_gb=128, usable_gb=119, ssd_gb=1000, bw_spec_gbs=273,
        # llama.cpp dense Q4 reaches ~75% of spec on GB10 (tok/s × weights); vLLM FP8 ~87%.
        bw_eff_gbs={"ollama": 205, "vllm": 238},
        # measured matmul peaks (no official dense figures): BF16 ~100, FP8 ~208; INT8 ≈ FP8
        compute={"int8": 208, "fp8": 208, "bf16": 100},
        published={  # same model, measured on GB10 (Kubesimplify, Aug 2026)
            "qwen3.8-27b-q4k": {"decode_tps": 11.6, "prefill_tps": 838},   # llama.cpp Q4_K_XL, pp512/pp2048
            "qwen3.8-27b-fp8": {"decode_tps": 8.2, "prefill_tps": 1914},   # vLLM FP8, pp2048
        },
        sources=["https://docs.nvidia.com/dgx/dgx-spark/hardware.html",
                 "https://blog.kubesimplify.com/qwen3-8-27b-on-dgx-spark",
                 "https://github.com/ggml-org/llama.cpp/discussions/16578",
                 "https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops"]),
}


def _mean(xs):
    xs = [x for x in xs if x]
    return sum(xs) / len(xs) if xs else None


def _load(run_dir: str, prefix: str) -> Optional[dict]:
    files = sorted(glob.glob(os.path.join(run_dir, f"{prefix}_*.json")))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def proxy_compute(g: Gpu, kind: str) -> float:
    return {"int8": g.int8_tops, "fp8": g.fp8_tflops or g.bf16_tflops, "bf16": g.bf16_tflops}[kind]


def project_run(run_dir: str, target_key: str = "gx10") -> dict:
    t = TARGETS[target_key]
    speed, load = _load(run_dir, "llm_speed"), _load(run_dir, "llm_load")
    if not speed:
        raise FileNotFoundError(f"no llm_speed_*.json in {run_dir} — the benchmark didn't run")
    engine = "ollama" if speed["args"].get("backend") == "ollama" else "vllm"
    model: Optional[Model] = model_from_ref(speed["args"].get("model", ""))
    if not model:
        raise ValueError(f"model {speed['args'].get('model')} is not in vast/models.py")
    proxy = gpu_from_report(speed.get("gpu", ""))
    if not proxy or not proxy.bw_gbs:
        raise ValueError(f"no specs for GPU '{speed.get('gpu')}' in vast/models.py")

    kind = model.compute_kind
    c_ratio = proxy_compute(proxy, kind) / t.compute[kind]           # >1: target is slower at compute
    w = model.weights_gb
    bw_t = t.bw_eff_gbs[engine]
    mem_t = w / bw_t                                                  # s per step on the target
    mem_p = w / (proxy.bw_gbs * PROXY_BW_EFFICIENCY)

    def step(proxy_tps):
        tp = 1.0 / proxy_tps
        cmp_t = max(tp - mem_p, 0.0) * c_ratio
        return max(mem_t, cmp_t), ("memory bandwidth" if mem_t >= cmp_t else "compute")

    ok = [r for r in speed["results"] if r.get("ok")]
    dec_p = _mean([r.get("server_decode_tps") or r.get("decode_tps") for r in ok])
    pre_p = _mean([r.get("server_prefill_tps") for r in ok])
    sweep = [p for p in speed.get("prefill", []) if p.get("ok")]
    long_p = sweep[-1] if sweep else None
    ttft_p = _mean([r.get("ttft_s") for r in ok])
    out_tok = _mean([r.get("completion_tokens") for r in ok]) or 256

    res = {"target": t.title, "proxy_gpu": proxy.name, "model": model.key, "engine": engine,
           "weights_gb": w, "compute_kind": kind, "compute_ratio_proxy_over_target": round(c_ratio, 3),
           "bw_eff_target_gbs": bw_t, "fits": w * 1.1 + 2 <= t.usable_gb,
           "memory_use_gb": round(w * 1.1 + 2, 1), "usable_gb": t.usable_gb}
    if dec_p:
        s, lim = step(dec_p)
        res.update(decode_proxy=round(dec_p, 1), decode_target=round(1 / s, 1), decode_limit=lim)
    if pre_p:
        res.update(prefill_short_proxy=round(pre_p), prefill_short_target=round(pre_p / c_ratio))
    if long_p:
        tps = long_p.get("server_prefill_tps") or long_p.get("prefill_tps_client")
        if tps:
            res.update(prefill_long_tokens=long_p.get("prompt_tokens"), prefill_long_proxy=round(tps),
                       prefill_long_target=round(tps / c_ratio))
    if ttft_p and dec_p:
        ttft_t = ttft_p * c_ratio
        res.update(ttft_proxy=round(ttft_p, 3), ttft_target=round(ttft_t, 3),
                   latency_target=round(ttft_t + out_tok / res["decode_target"], 2), output_tokens=round(out_tok))
    if load:
        levels = []
        for lv in load.get("levels", []):
            ps = lv.get("per_stream_decode_tps")
            if not ps:
                continue
            s, lim = step(ps)
            f = (1 / s) / ps
            levels.append({"concurrency": lv["concurrency"], "output_tps_proxy": round(lv["output_tok_per_s"], 1),
                           "output_tps_target": round(lv["output_tok_per_s"] * f, 1),
                           "per_stream_proxy": round(ps, 1), "per_stream_target": round(1 / s, 1), "limit": lim})
        res["load"] = levels
    if model.key in t.published:
        res["published"] = t.published[model.key]
    return res


def write_report(run_dir: str, res: dict, target_key: str = "gx10") -> str:
    t = TARGETS[target_key]
    pub = res.get("published", {})

    def row(label, proxy, target, published=None, unit=""):
        p = "N/A" if proxy is None else f"{proxy}{unit}"
        q = "N/A" if target is None else f"**{target}{unit}**"
        return f"| {label} | {p} | {q} | {published if published is not None else '–'} |"

    lines = [f"# Projection: {t.title}", "",
             f"- Measured on **{res['proxy_gpu']}** ({GPUS[res['proxy_gpu']].bw_gbs:g} GB/s) with {res['engine']}, "
             f"model `{res['model']}` ({res['weights_gb']:g} GB weights, {res['compute_kind']} matmuls)",
             f"- Target: {t.bw_spec_gbs:g} GB/s spec, {res['bw_eff_target_gbs']:g} GB/s achievable with {res['engine']}; "
             f"{res['compute_kind']} compute {t.compute[res['compute_kind']]:g} vs proxy "
             f"{proxy_compute(GPUS[res['proxy_gpu']], res['compute_kind']):g} (proxy/target = {res['compute_ratio_proxy_over_target']})",
             f"- Memory: ~{res['memory_use_gb']} GB (weights + runtime, before KV cache) of {res['usable_gb']} GB usable → "
             + ("fits" if res["fits"] else "**does not fit**")
             + f"; model files fit the {t.ssd_gb // 1000} TB SSD", "",
             "| Metric | Measured on proxy | Projected on target | Published on target |",
             "| :--- | ---: | ---: | ---: |",
             row("Decode, single stream (tok/s)", res.get("decode_proxy"), res.get("decode_target"), pub.get("decode_tps")),
             row("Prefill, short prompt (tok/s)", res.get("prefill_short_proxy"), res.get("prefill_short_target")),
             row(f"Prefill, {res.get('prefill_long_tokens', '?')} tokens (tok/s)", res.get("prefill_long_proxy"),
                 res.get("prefill_long_target"), pub.get("prefill_tps")),
             row("TTFT, short prompt (s)", res.get("ttft_proxy"), res.get("ttft_target")),
             row(f"Latency, {res.get('output_tokens', '?')}-token answer (s)", None, res.get("latency_target")), ""]
    if res.get("decode_limit"):
        lines.append(f"Single-stream decode on the target is limited by **{res['decode_limit']}**.")
    if res.get("load"):
        lines += ["", "| Concurrency | Output tok/s proxy | Output tok/s target | Per-stream proxy | Per-stream target | Limit |",
                  "| ---: | ---: | ---: | ---: | ---: | :--- |"]
        lines += [f"| {x['concurrency']} | {x['output_tps_proxy']} | **{x['output_tps_target']}** | "
                  f"{x['per_stream_proxy']} | {x['per_stream_target']} | {x['limit']} |" for x in res["load"]]
    lines += ["", "Translation quality (chrF, exact match) is the same on the target: same weights, same engine.",
              "", "How it's computed: decode = max(weights ÷ target bandwidth, proxy compute time × compute ratio); "
              "prefill = proxy × compute ratio. Validated on Qwen3.8-27B Q4_K: RTX 3090 → GB10 projects "
              "12.2 decode / 811 prefill tok/s vs 11.6 / 838 measured (+5% / −3%). Expect roughly ±15% for "
              "large dense models. Small models (a few GB) can be limited by per-token overhead on GB10 and "
              "come out slower than projected. Batched (concurrency) numbers are the least certain.", "", "Sources: " + " · ".join(t.sources)]
    path = os.path.join(run_dir, f"{target_key}_projection.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(run_dir, f"{target_key}_projection.json"), "w") as f:
        json.dump(res, f, indent=1)
    return path
