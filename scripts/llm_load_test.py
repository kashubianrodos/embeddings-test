"""Concurrency / throughput test for a generative LLM.

For each concurrency level, fires N requests with that many in flight and
reports aggregate output throughput (tok/s), requests/s, and TTFT / latency
percentiles. This is the number that matters for serving many users.

Note for Ollama: parallel requests are capped by OLLAMA_NUM_PARALLEL on the
server; anything above it is queued (visible as rising TTFT).

Examples:
  python scripts/llm_load_test.py --concurrency 1,2,4,8
  python scripts/llm_load_test.py --backend openai --concurrency 1,8,32,64 --max-tokens 512
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from llm_common import (add_common_args, build_prompt, client_from_args, die, fmt,
                        header_lines, load_samples, summarize, write_outputs)


def run_level(client, prompts, concurrency, max_tokens):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda pr: client.generate(pr, max_tokens), prompts))
    wall = time.perf_counter() - start

    good = [r for r in results if r.ok]
    out_tokens = sum(r.completion_tokens or 0 for r in good)
    in_tokens = sum(r.prompt_tokens or 0 for r in good)
    ttft, lat = summarize([r.ttft_s for r in good]), summarize([r.total_s for r in good])
    dec = summarize([r.decode_tps for r in good])
    return {
        "concurrency": concurrency, "requests": len(results), "ok": len(good),
        "wall_s": wall, "req_per_s": len(good) / wall,
        "output_tok_per_s": out_tokens / wall, "total_tok_per_s": (out_tokens + in_tokens) / wall,
        "ttft_p50": ttft and ttft["median"], "ttft_p95": ttft and ttft["p95"],
        "lat_p50": lat and lat["median"], "lat_p95": lat and lat["p95"],
        "per_stream_decode_tps": dec and dec["mean"],
        "errors": [r.error for r in results if not r.ok][:5],
        "results": [r.to_dict() for r in results],
    }


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--concurrency", default="1,2,4,8")
    p.add_argument("--requests-per-level", type=int, default=0,
                   help="Default: max(8, 3 x concurrency)")
    p.add_argument("--task", default="explain", choices=["explain", "translate_en_pl", "translate_pl_en"])
    p.add_argument("--max-tokens", type=int, default=256)
    args = p.parse_args()

    client = client_from_args(args)
    ok, err = client.check()
    if not ok:
        die(err)

    en, pl = load_samples()
    print("Warmup...")
    if not client.generate(build_prompt(args.task, en[0], pl[0]), 16).ok:
        die("Warmup request failed")

    levels, offset = [], 1
    for c in [int(x) for x in args.concurrency.split(",") if x.strip()]:
        n = args.requests_per_level or max(8, 3 * c)
        # different sentences per level so prefix caches don't flatter later levels
        prompts = [build_prompt(args.task, en[(offset + i) % len(en)], pl[(offset + i) % len(pl)])
                   for i in range(n)]
        offset += n
        print(f"\n--- concurrency {c}: {n} requests ---")
        lv = run_level(client, prompts, c, args.max_tokens)
        levels.append(lv)
        print(f"ok={lv['ok']}/{n} wall={lv['wall_s']:.1f}s out={lv['output_tok_per_s']:.1f} tok/s "
              f"req/s={lv['req_per_s']:.2f} ttft_p95={fmt(lv['ttft_p95'])}s")

    lines = header_lines("LLM Load Test (concurrency sweep)", args,
                         f"Task: `{args.task}`, max_tokens={args.max_tokens}")
    lines += ["| Concurrency | OK / sent | Output tok/s | Total tok/s | Req/s | Per-stream decode tok/s "
              "| TTFT p50 (s) | TTFT p95 (s) | Latency p50 (s) | Latency p95 (s) |",
              "| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for lv in levels:
        lines.append(f"| {lv['concurrency']} | {lv['ok']}/{lv['requests']} | {lv['output_tok_per_s']:.1f} | "
                     f"{lv['total_tok_per_s']:.1f} | {lv['req_per_s']:.2f} | {fmt(lv['per_stream_decode_tps'], '.1f')} | "
                     f"{fmt(lv['ttft_p50'])} | {fmt(lv['ttft_p95'])} | {fmt(lv['lat_p50'])} | {fmt(lv['lat_p95'])} |")
    errs = [f"- c={lv['concurrency']}: {e}" for lv in levels for e in lv["errors"]]
    if errs:
        lines += ["", "## Errors (first 5 per level)", *errs]

    write_outputs("llm_load", args, lines, {"levels": levels})


if __name__ == "__main__":
    main()
