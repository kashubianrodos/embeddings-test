"""Speed / size / dimension benchmark of the three embedding models.

Embeds 50 English + 50 Polish samples per model, times each language
separately, and writes a Markdown comparison report plus raw JSON vectors.
"""
import json
import os
from datetime import datetime

from common import (MODELS, check_model_availability, embed_texts,
                    ensure_output_dir, load_samples)


def run_model(model_name, texts_en, texts_pl):
    ok, err = check_model_availability(model_name)
    if not ok:
        print(f"❌ Skipping {model_name}: {err}")
        return {"status": "error", "error": err}

    print(f"✅ {model_name}: embedding {len(texts_en)} EN + {len(texts_pl)} PL texts...")
    try:
        vec_en, dur_en = embed_texts(model_name, texts_en)
        vec_pl, dur_pl = embed_texts(model_name, texts_pl)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    return {
        "status": "ok",
        "dimension": len(vec_en[0]),
        "duration_en_s": dur_en,
        "duration_pl_s": dur_pl,
        "duration_total_s": dur_en + dur_pl,
        "vectors_en": vec_en,
        "vectors_pl": vec_pl,
    }


def main():
    texts_en, texts_pl = load_samples()
    started = datetime.now()
    results = {}

    for cfg in MODELS:
        print(f"\n--- {cfg['name']} ---")
        results[cfg["name"]] = run_model(cfg["name"], texts_en, texts_pl)

    out_dir = ensure_output_dir()
    stamp = started.strftime("%Y%m%d_%H%M%S")

    # Markdown report
    rows = [
        ("Status", lambda c, r: "✅ ok" if r["status"] == "ok" else "❌ error"),
        ("Texts embedded", lambda c, r: str(len(texts_en) + len(texts_pl)) if r["status"] == "ok" else "N/A"),
        ("Vector dimension", lambda c, r: str(r.get("dimension", "N/A"))),
        ("Approx. model size (MB)", lambda c, r: str(c["size_approx_mb"])),
        ("Duration EN, 50 texts (s)", lambda c, r: f"{r['duration_en_s']:.3f}" if r["status"] == "ok" else "N/A"),
        ("Duration PL, 50 texts (s)", lambda c, r: f"{r['duration_pl_s']:.3f}" if r["status"] == "ok" else "N/A"),
        ("Total duration (s)", lambda c, r: f"{r['duration_total_s']:.3f}" if r["status"] == "ok" else "N/A"),
    ]
    lines = [
        "# Embedding Model Benchmark Report",
        "",
        f"Generated: {started:%Y-%m-%d %H:%M:%S} — 50 English + 50 Polish samples per model",
        "",
        "| Metric | " + " | ".join(c["name"] for c in MODELS) + " |",
        "| :--- | " + " | ".join(":---:" for _ in MODELS) + " |",
    ]
    for label, fn in rows:
        lines.append(f"| **{label}** | " + " | ".join(fn(c, results[c["name"]]) for c in MODELS) + " |")

    errors = [f"- **{n}**: {r['error']}" for n, r in results.items() if r["status"] != "ok"]
    if errors:
        lines += ["", "## Errors", *errors]

    report_path = os.path.join(out_dir, f"benchmark_report_{stamp}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n✅ Report written to: {report_path}")

    # Raw vectors (for semantic_test.py or later analysis)
    raw_path = os.path.join(out_dir, f"embeddings_{stamp}.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": started.isoformat(),
            "texts_en": texts_en,
            "texts_pl": texts_pl,
            "models": {
                n: {k: v for k, v in r.items() if k != "status"}
                for n, r in results.items() if r["status"] == "ok"
            },
        }, f)
    print(f"✅ Raw embeddings written to: {raw_path}")


if __name__ == "__main__":
    main()
