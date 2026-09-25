"""Translation-quality sanity check using the repo's parallel EN/PL pairs.

The embedding tests use the pairs for cross-lingual retrieval; for a generative
model we use them as a translation benchmark instead: EN→PL and PL→EN, scored
against the reference with chrF (character n-gram F-score, the metric used by
WMT; implemented here without dependencies, closely following sacreBLEU's
chrF defaults: char n=6, beta=2, whitespace ignored).

Useful to confirm that a quantization / backend / abliteration did not
degrade the model, and to compare runs. Every output is saved for review.

  python scripts/llm_quality_test.py --n 50
"""
import argparse
from collections import Counter

from llm_common import (add_common_args, build_prompt, client_from_args, die, fmt,
                        header_lines, load_samples, strip_thinking, write_outputs)

CHAR_ORDER, BETA = 6, 2.0


def _ngram_stats(hyp, ref):
    hyp, ref = hyp.replace(" ", ""), ref.replace(" ", "")
    stats = []
    for n in range(1, CHAR_ORDER + 1):
        h = Counter(hyp[i:i + n] for i in range(len(hyp) - n + 1))
        r = Counter(ref[i:i + n] for i in range(len(ref) - n + 1))
        stats.append((sum((h & r).values()), sum(h.values()), sum(r.values())))
    return stats


def _chrf_from_stats(stats):
    precs, recs = [], []
    for match, hyp_n, ref_n in stats:
        if hyp_n and ref_n:
            precs.append(match / hyp_n)
            recs.append(match / ref_n)
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + r == 0:
        return 0.0
    b2 = BETA ** 2
    return 100 * (1 + b2) * p * r / (b2 * p + r)


def chrf_corpus(hyps, refs):
    total = [[0, 0, 0] for _ in range(CHAR_ORDER)]
    for h, r in zip(hyps, refs):
        for i, s in enumerate(_ngram_stats(h, r)):
            for j in range(3):
                total[i][j] += s[j]
    return _chrf_from_stats(total)


def normalize(s):
    return " ".join(s.lower().strip().strip('"„”“\'').rstrip(".!?").split())


def run_direction(client, task, sources, refs, max_tokens):
    items = []
    for i, (src, ref) in enumerate(zip(sources, refs), 1):
        en, pl = (src, ref) if task == "translate_en_pl" else (ref, src)
        r = client.generate(build_prompt(task, en, pl), max_tokens=max_tokens, temperature=0.0)
        hyp = strip_thinking(r.text) if r.ok else ""
        items.append({"source": src, "reference": ref, "hypothesis": hyp, "ok": r.ok, "error": r.error,
                      "raw_had_think": "<think>" in r.text or "</think>" in r.text,
                      "chrf": _chrf_from_stats(_ngram_stats(hyp, ref)) if hyp else 0.0,
                      "latency_s": r.total_s})
        print(f"[{task} {i}/{len(sources)}] chrF={items[-1]['chrf']:.1f}  {hyp[:70]!r}")
    good = [x for x in items if x["ok"]]
    return {
        "items": items,
        "ok": len(good),
        "corpus_chrf": chrf_corpus([x["hypothesis"] for x in good], [x["reference"] for x in good]) if good else None,
        "mean_chrf": sum(x["chrf"] for x in good) / len(good) if good else None,
        "exact_match": sum(normalize(x["hypothesis"]) == normalize(x["reference"]) for x in good) / len(good)
        if good else None,
        "empty": sum(1 for x in good if not x["hypothesis"]),
        "think_leaks": sum(1 for x in good if x["raw_had_think"]),
        "mean_latency_s": sum(x["latency_s"] for x in good) / len(good) if good else None,
    }


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--n", type=int, default=50, help="Number of pairs per direction (max 500)")
    p.add_argument("--max-tokens", type=int, default=160)
    args = p.parse_args()

    client = client_from_args(args)
    ok, err = client.check()
    if not ok:
        die(err)

    en, pl = load_samples()
    en, pl = en[:args.n], pl[:args.n]
    res = {"en_pl": run_direction(client, "translate_en_pl", en, pl, args.max_tokens),
           "pl_en": run_direction(client, "translate_pl_en", pl, en, args.max_tokens)}

    lines = header_lines("LLM Translation Quality (EN↔PL, chrF)", args,
                         f"{len(en)} pairs per direction, temperature 0")
    lines += ["| Metric | EN→PL | PL→EN |", "| :--- | ---: | ---: |"]
    rows = [("Successful requests", "ok", "d"), ("Corpus chrF (0–100)", "corpus_chrf", ".1f"),
            ("Mean sentence chrF", "mean_chrf", ".1f"), ("Exact match (normalized)", "exact_match", ".1%"),
            ("Empty outputs", "empty", "d"), ("Outputs with <think> leak", "think_leaks", "d"),
            ("Mean latency (s)", "mean_latency_s", ".2f")]
    for label, key, spec in rows:
        lines.append(f"| **{label}** | {fmt(res['en_pl'][key], spec)} | {fmt(res['pl_en'][key], spec)} |")

    worst = sorted(res["en_pl"]["items"], key=lambda x: x["chrf"])[:5]
    lines += ["", "## Lowest-scoring EN→PL outputs (review manually)", "",
              "| chrF | Source | Reference | Output |", "| ---: | :--- | :--- | :--- |"]
    for x in worst:
        cells = [x["source"], x["reference"], x["hypothesis"] or "(empty)"]
        lines.append(f"| {x['chrf']:.1f} | " + " | ".join(c.replace("|", "\\|").replace("\n", " ")[:120]
                                                            for c in cells) + " |")

    write_outputs("llm_quality", args, lines, res)


if __name__ == "__main__":
    main()
