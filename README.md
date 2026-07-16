# embeddings-test

Benchmark and semantic-accuracy tests for local Ollama embedding models, based on
[Decoding AI's Inner Language: How to Test Your Embedding Models](https://dev.to/aairom/decoding-ais-inner-language-how-to-test-your-embedding-models-126).

## Models tested

| Model | Approx. size |
| :--- | :--- |
| `snowflake-arctic-embed2` | ~1.2 GB |
| `granite-embedding:278m` | ~278 MB |
| `paraphrase-multilingual` | ~563 MB |

## Dataset

`data/samples.json` — 50 parallel English/Polish sentence pairs (100 texts total).
Because the Polish sentences are translations of the English ones, the same data
supports both a speed benchmark and a cross-lingual semantic accuracy test.

## Tests

1. **`scripts/benchmark.py`** — the three pillars from the article: latency
   (per language and total), model size, and vector dimension. Writes a Markdown
   report and raw JSON vectors to `output/`.
2. **`scripts/semantic_test.py`** — semantic accuracy via cross-lingual retrieval:
   for each EN sentence, is its PL translation the nearest PL vector (top-1
   accuracy, both directions)? Also reports mean cosine similarity of true pairs
   vs. non-pairs and the separation margin.

## Usage

Requires [Ollama](https://ollama.com) running locally (`http://localhost:11434`).

```bash
./run_all.sh
```

Or manually:

```bash
ollama pull snowflake-arctic-embed2
ollama pull granite-embedding:278m
ollama pull paraphrase-multilingual

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python scripts/benchmark.py
python scripts/semantic_test.py
```

Reports land in `output/` as timestamped Markdown files.

## Interpreting results

- **Speed**: lower total duration = better for real-time / high-throughput use.
- **Dimension**: higher usually means more semantic nuance, at a cost.
- **Top-1 accuracy**: how well the model aligns Polish and English meaning —
  the key metric for multilingual RAG over Polish content.
- **Separation margin**: larger margin = clearer distinction between related
  and unrelated texts, which improves retrieval thresholding.
