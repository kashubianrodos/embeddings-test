#!/usr/bin/env bash
# Pull models and run both tests against a local Ollama server.
set -euo pipefail
cd "$(dirname "$0")"

MODELS=("snowflake-arctic-embed2" "granite-embedding:278m" "paraphrase-multilingual")

echo "== Checking Ollama =="
ollama --version >/dev/null || { echo "Ollama not installed: https://ollama.com"; exit 1; }

echo "== Pulling models =="
for m in "${MODELS[@]}"; do
  ollama pull "$m"
done

echo "== Setting up Python env =="
if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "== Running benchmark =="
python scripts/benchmark.py

echo "== Running semantic test =="
python scripts/semantic_test.py

echo "== Done. Reports in ./output =="
