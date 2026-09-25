"""Single-stream speed test for a generative LLM.

Part 1 – generation: sends N prompts one at a time (after warmup) and reports
TTFT, end-to-end latency, output tokens and decode speed (tokens/s).

Part 2 – prefill sweep (optional, --prefill-sizes): sends long prompts of
roughly the requested token sizes and reports prompt-processing speed.

Examples:
  python scripts/llm_speed_test.py                          # Ollama defaults
  python scripts/llm_speed_test.py --backend openai --n 30  # vLLM
  python scripts/llm_speed_test.py --prefill-sizes 1024,4096,8192 --num-ctx 16384
"""
import argparse
import uuid

from llm_common import (add_common_args, build_prompt, client_from_args, die, fmt,
                        header_lines, load_samples, summarize, write_outputs)


def long_prompt(target_tokens, en, pl):
    # ~3.5 chars/token for mixed EN/PL text; the real count is taken from the server.
    # A random nonce at the start defeats prefix caching between runs.
    parts, chars, i = [f"[run {uuid.uuid4().hex}]"], 0, 0
    while chars < target_tokens * 3.5:
        s = f"{en[i % len(en)]} {pl[i % len(pl)]}"
        parts.append(s)
        chars += len(s) + 1
        i += 1
    return "\n".join(parts) + "\n\nSummarize the text above in one sentence."


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--n", type=int, default=20, help="Number of timed prompts")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--task", default="explain", choices=["explain", "translate_en_pl", "translate_pl_en"])
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--prefill-sizes", default="", help="Comma list, e.g. 512,2048,8192")
    args = p.parse_args()

    client = client_from_args(args)
    ok, err = client.check()
    if not ok:
        die(err)

    en, pl = load_samples()
    prompts = [build_prompt(args.task, en[i % len(en)], pl[i % len(pl)])
               for i in range(args.warmup + args.n)]

    print(f"Warmup ({args.warmup})...")
    for pr in prompts[:args.warmup]:
        r = client.generate(pr, args.max_tokens)
        if not r.ok:
            die(f"Warmup failed: {r.error}")

    results = []
    for i, pr in enumerate(prompts[args.warmup:], 1):
        r = client.generate(pr, args.max_tokens)
        results.append(r)
        print(f"[{i}/{args.n}] ok={r.ok} ttft={fmt(r.ttft_s, '.3f')}s total={r.total_s:.2f}s "
              f"out={r.completion_tokens} decode={fmt(r.decode_tps, '.1f')} tok/s"
              + (f" ERROR: {r.error}" if not r.ok else ""))

    good = [r for r in results if r.ok]
    metrics = [
        ("TTFT (s)", summarize([r.ttft_s for r in good]), ".3f"),
        ("End-to-end latency (s)", summarize([r.total_s for r in good]), ".2f"),
        ("Prompt tokens", summarize([r.prompt_tokens for r in good]), ".0f"),
        ("Output tokens", summarize([r.completion_tokens for r in good]), ".0f"),
        ("Decode speed, client (tok/s)", summarize([r.decode_tps for r in good]), ".1f"),
        ("Decode speed, server (tok/s)", summarize([r.server_decode_tps for r in good]), ".1f"),
        ("Prefill speed, server (tok/s)", summarize([r.server_prefill_tps for r in good]), ".0f"),
    ]
    lines = header_lines("LLM Speed Test (single stream)", args,
                         f"Task: `{args.task}`, max_tokens={args.max_tokens}, "
                         f"{len(good)}/{args.n} ok after {args.warmup} warmup")
    lines += ["| Metric | Mean | Median | P95 | Min | Max | Std dev |",
              "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, s, spec in metrics:
        if s is None:
            lines.append(f"| **{label}** | N/A | N/A | N/A | N/A | N/A | N/A |")
        else:
            lines.append(f"| **{label}** | " + " | ".join(
                format(s[k], spec) for k in ("mean", "median", "p95", "min", "max", "stdev")) + " |")
    if any(r.tokens_approx for r in good):
        lines.append("\n> ⚠️ Server did not return token usage; counts are approximated from stream chunks.")

    prefill = []
    sizes = [int(x) for x in args.prefill_sizes.split(",") if x.strip()]
    if sizes:
        print("\n--- Prefill sweep ---")
        for size in sizes:
            r = client.generate(long_prompt(size, en, pl), max_tokens=32)
            tps = (r.prompt_tokens / r.ttft_s) if (r.ok and r.prompt_tokens and r.ttft_s) else None
            prefill.append({"target": size, **r.to_dict(), "prefill_tps_client": tps})
            print(f"target={size} ok={r.ok} prompt_tokens={r.prompt_tokens} ttft={fmt(r.ttft_s, '.2f')}s"
                  + (f" ERROR: {r.error}" if not r.ok else ""))
        lines += ["", "## Prefill sweep", "",
                  "| Target tokens | Actual prompt tokens | TTFT (s) | Prefill, client (tok/s) | "
                  "Prefill, server (tok/s) | Status |",
                  "| ---: | ---: | ---: | ---: | ---: | :---: |"]
        for x in prefill:
            lines.append(f"| {x['target']} | {x['prompt_tokens'] or 'N/A'} | {fmt(x['ttft_s'])} | "
                         f"{fmt(x['prefill_tps_client'], '.0f')} | {fmt(x['server_prefill_tps'], '.0f')} | "
                         f"{'✅' if x['ok'] else '❌ ' + x['error'][:60]} |")

    errors = [f"- {r.error}" for r in results if not r.ok]
    if errors:
        lines += ["", "## Errors", *errors]

    write_outputs("llm_speed", args, lines,
                  {"results": [r.to_dict() for r in results], "prefill": prefill})


if __name__ == "__main__":
    main()
