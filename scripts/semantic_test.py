"""Semantic accuracy test: cross-lingual EN↔PL retrieval.

The 50 Polish samples are translations of the 50 English samples.
A good multilingual embedding model should, for each English sentence,
rank its Polish translation as the nearest Polish vector (and vice versa).

Metrics per model:
- top-1 retrieval accuracy EN→PL and PL→EN
- mean cosine similarity of correct pairs (higher = better)
- mean cosine similarity of random non-pairs (lower = better separation)
- separation margin = pair mean − non-pair mean
"""
import os
import statistics
from datetime import datetime

from common import (MODELS, check_model_availability, cosine_similarity,
                    embed_texts, ensure_output_dir, load_samples)


def retrieval_accuracy(queries, docs):
    """Top-1 accuracy: queries[i] should be closest to docs[i]."""
    hits = 0
    for i, q in enumerate(queries):
        sims = [cosine_similarity(q, d) for d in docs]
        if max(range(len(sims)), key=sims.__getitem__) == i:
            hits += 1
    return hits / len(queries)


def evaluate(model_name, texts_en, texts_pl):
    ok, err = check_model_availability(model_name)
    if not ok:
        print(f"❌ Skipping {model_name}: {err}")
        return None

    print(f"✅ {model_name}: embedding and evaluating...")
    vec_en, _ = embed_texts(model_name, texts_en)
    vec_pl, _ = embed_texts(model_name, texts_pl)

    n = len(vec_en)
    pair_sims = [cosine_similarity(vec_en[i], vec_pl[i]) for i in range(n)]
    nonpair_sims = [cosine_similarity(vec_en[i], vec_pl[(i + 7) % n]) for i in range(n)]

    return {
        "acc_en_pl": retrieval_accuracy(vec_en, vec_pl),
        "acc_pl_en": retrieval_accuracy(vec_pl, vec_en),
        "pair_mean": statistics.mean(pair_sims),
        "nonpair_mean": statistics.mean(nonpair_sims),
        "margin": statistics.mean(pair_sims) - statistics.mean(nonpair_sims),
        "dimension": len(vec_en[0]),
    }


def main():
    texts_en, texts_pl = load_samples()
    started = datetime.now()
    results = {}

    for cfg in MODELS:
        print(f"\n--- {cfg['name']} ---")
        results[cfg["name"]] = evaluate(cfg["name"], texts_en, texts_pl)

    lines = [
        "# Semantic Accuracy Report (cross-lingual EN↔PL)",
        "",
        f"Generated: {started:%Y-%m-%d %H:%M:%S} — 50 parallel sentence pairs",
        "",
        "| Metric | " + " | ".join(c["name"] for c in MODELS) + " |",
        "| :--- | " + " | ".join(":---:" for _ in MODELS) + " |",
    ]
    metrics = [
        ("Vector dimension", "dimension", "{}"),
        ("Top-1 accuracy EN→PL", "acc_en_pl", "{:.1%}"),
        ("Top-1 accuracy PL→EN", "acc_pl_en", "{:.1%}"),
        ("Mean cosine, true pairs", "pair_mean", "{:.4f}"),
        ("Mean cosine, non-pairs", "nonpair_mean", "{:.4f}"),
        ("Separation margin", "margin", "{:.4f}"),
    ]
    for label, key, fmt in metrics:
        row = [fmt.format(results[c["name"]][key]) if results[c["name"]] else "N/A" for c in MODELS]
        lines.append(f"| **{label}** | " + " | ".join(row) + " |")

    out_dir = ensure_output_dir()
    path = os.path.join(out_dir, f"semantic_report_{started:%Y%m%d_%H%M%S}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\n✅ Report written to: {path}")


if __name__ == "__main__":
    main()
