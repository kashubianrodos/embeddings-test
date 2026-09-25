#!/usr/bin/env bash
# Runs ON the vast.ai instance. Installs Ollama, pulls the GGUF model, prepares Python.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${LLM_MODEL:-hf.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF:Q4_K}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"    # max concurrent requests per model
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
export OLLAMA_FLASH_ATTENTION=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=1h
export OLLAMA_MODELS="${OLLAMA_MODELS:-/workspace/ollama-models}"

echo "== Packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl ca-certificates git python3 python3-venv zstd pciutils procps >/dev/null

echo "== Ollama"
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
if ! pgrep -x ollama >/dev/null; then
  nohup ollama serve > /workspace/ollama.log 2>&1 &
fi
for _ in $(seq 60); do curl -sf localhost:11434/api/version >/dev/null && break; sleep 2; done
ollama --version

echo "== Pulling $MODEL (this is the slow part)"
ollama pull "$MODEL"

echo "== Python env"
[ -d venv ] || python3 -m venv venv
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -q -r requirements-llm.txt

cat > /workspace/bench.env <<ENV
export LLM_BACKEND=ollama
export LLM_MODEL='$MODEL'
export LLM_NUM_CTX=$OLLAMA_CONTEXT_LENGTH
ENV

echo "== Smoke test"
LLM_MODEL="$MODEL" ./venv/bin/python -c '
import os, sys; sys.path.insert(0, "scripts")
from llm_common import LLMClient
r = LLMClient("ollama", os.environ["LLM_MODEL"]).generate("Say hello in Polish.", max_tokens=32)
print("ok:", r.ok, repr(r.text), r.error)'
echo "READY"
