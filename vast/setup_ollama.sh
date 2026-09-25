#!/usr/bin/env bash
# Runs ON the instance (vast base image, started by onstart.sh → bootstrap.sh).
# Starts Ollama, pulls the GGUF, prepares Python, smoke-tests, then touches /workspace/READY.
# Idempotent: after a preemption/resume it reuses the downloaded model.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=vast/remote_lib.sh
. vast/remote_lib.sh
setup_guard
mark setup_start mode=ollama

MODEL="${LLM_MODEL:?LLM_MODEL not set}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-f16}"
export OLLAMA_FLASH_ATTENTION=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=1h
export OLLAMA_MODELS="${OLLAMA_MODELS:-$W/ollama-models}"
export OLLAMA_HOST=127.0.0.1:11434

export DEBIAN_FRONTEND=noninteractive
NEED=""
for c in curl zstd pgrep; do command -v $c >/dev/null || NEED=1; done
python3 -c 'import venv, ensurepip' 2>/dev/null || NEED=1
if [ -n "$NEED" ]; then
  apt-get update -qq || echo "apt-get update had errors; trying install anyway"
  apt-get install -y -qq curl ca-certificates git python3 python3-venv procps pciutils zstd >/dev/null
fi
mark packages_done

# Pinned Ollama (OLLAMA_VERSION). Skipped if the right version is already there
# (e.g. on resume after preemption).
WANT="${OLLAMA_VERSION:-0.34.4}"
HAVE=$(ollama --version 2>/dev/null | awk '{print $NF}' | tail -n1 || true)
if [ "$HAVE" != "$WANT" ]; then
  mark engine_download_start version="$WANT"
  curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="$WANT" sh
fi
if ! pgrep -x ollama >/dev/null; then
  nohup ollama serve >> "$W/ollama.log" 2>&1 &
fi
for _ in $(seq 60); do curl -sf localhost:11434/api/version >/dev/null && break; sleep 2; done
curl -sf localhost:11434/api/version >/dev/null || { tail -n 50 "$W/ollama.log"; exit 1; }
mark engine_ready version="$(ollama --version 2>/dev/null | awk '{print $NF}' | tail -n1)"

mark download_start
ollama pull "$MODEL"
mark download_done bytes="$(dir_bytes "$OLLAMA_MODELS")"

# The name Ollama lists can differ in case/format from what we pulled; use the listed one.
NAME=$(ollama list | awk 'NR>1{print $1}' | grep -iF "${MODEL##*:}" | head -n1 || true)
[ -n "$NAME" ] || NAME=$(ollama list | awk 'NR>1{print $1}' | grep -iF "${MODEL%:*}" | head -n1 || true)
[ -n "$NAME" ] || { echo "Pulled model not found in 'ollama list'"; ollama list; exit 1; }
echo "Model as listed by Ollama: $NAME"

[ -d venv ] || python3 -m venv venv
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -q -r requirements-llm.txt

cat > "$W/bench.env" <<ENV
export LLM_BACKEND=ollama
export LLM_MODEL='$NAME'
export LLM_BASE_URL=http://localhost:11434
export LLM_NUM_CTX=$OLLAMA_CONTEXT_LENGTH
export LLM_PYTHON=$PWD/venv/bin/python
export OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL
ENV

# Fail fast (DESIGN.md Q12): no automatic fallback to another engine or model.
echo "== Smoke test"
if ! LLM_MODEL="$NAME" ./venv/bin/python - <<'PY'
import os, sys
sys.path.insert(0, "scripts")
from llm_common import LLMClient
r = LLMClient("ollama", os.environ["LLM_MODEL"], timeout=600).generate("Say hello in Polish.", max_tokens=32)
print("ok:", r.ok, repr(r.text), r.error)
sys.exit(0 if r.ok and r.text.strip() else 1)
PY
then
  echo "Smoke test failed. Last Ollama log lines:"; tail -n 40 "$W/ollama.log"
  echo "Retry manually with another tag, e.g.  LLM_MODEL=huihui_ai/Qwen3.8-abliterated ./vast/bench.sh ollama"
  exit 1
fi
ollama ps || true
mark ready
touch "$W/READY"
echo "READY"
