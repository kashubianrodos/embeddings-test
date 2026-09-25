#!/usr/bin/env bash
# Runs ON the instance (image vastai/vllm:<pinned>, started by onstart.sh).
# Downloads the weights (timed separately), starts an OpenAI-compatible vLLM server,
# smoke-tests it, then touches /workspace/READY. Idempotent across preemption/resume.
#
# The vastai/vllm image only auto-starts vLLM when VLLM_MODEL is set; we never set it,
# so its supervisor stays idle and this script owns the server (own port, own log).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=vast/remote_lib.sh
. vast/remote_lib.sh
setup_guard
mark setup_start mode=vllm

MODEL="${LLM_MODEL:?LLM_MODEL not set}"
REV="${LLM_REVISION:-main}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.92}"
TP="${VLLM_TP:-$(nvidia-smi -L | wc -l)}"
EXTRA="${VLLM_EXTRA_ARGS:---language-model-only --kv-cache-dtype fp8 --reasoning-parser qwen3}"
PORT=8011
export HF_HOME="${HF_HOME:-$W/hf}"

if [ -x /venv/main/bin/python ]; then PY=/venv/main/bin/python; else PY=$(command -v python3); fi
if [ -x /venv/main/bin/vllm ]; then VLLM=/venv/main/bin/vllm; else VLLM=$(command -v vllm || true); fi
[ -n "$VLLM" ] || { echo "vllm not found in this image — VLLM_IMAGE must be a vastai/vllm (or vllm-openai) image"; exit 1; }
command -v curl >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl procps >/dev/null; }
"$PY" -c 'import requests' 2>/dev/null || "$PY" -m pip install -q -r requirements-llm.txt \
  || uv pip install --python "$PY" -q -r requirements-llm.txt
mark packages_done
"$PY" -c "import vllm; print('vLLM', vllm.__version__)"
mark engine_ready version="$("$PY" -c 'import vllm; print(vllm.__version__)')"

# Separate download step so the summary can report real download throughput.
mark download_start
MODEL="$MODEL" REV="$REV" "$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(os.environ["MODEL"], revision=os.environ["REV"],
                      ignore_patterns=["*.gguf", "*.bin", "*.pt", "*.pth", "*.onnx", "original/*"])
print("weights in", p)
PY
mark download_done bytes="$(dir_bytes "$HF_HOME")"

if ! pgrep -f "vllm serve" >/dev/null; then
  # shellcheck disable=SC2086
  nohup "$VLLM" serve "$MODEL" --revision "$REV" --served-model-name "$MODEL" \
    --max-model-len "$MAX_LEN" --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$MEM_UTIL" --port "$PORT" $EXTRA >> "$W/vllm.log" 2>&1 &
fi
for i in $(seq 180); do   # up to 30 min for load + compile (weights are already local)
  curl -sf "localhost:$PORT/v1/models" >/dev/null && break
  pgrep -f "vllm serve" >/dev/null || { echo "vLLM exited:"; tail -n 60 "$W/vllm.log"; exit 1; }
  [ $((i % 6)) -eq 0 ] && echo "... vLLM still loading ($((i / 6)) min)"
  sleep 10
done
curl -sf "localhost:$PORT/v1/models" >/dev/null || { echo "vLLM not up after 30 min"; tail -n 60 "$W/vllm.log"; exit 1; }

cat > "$W/bench.env" <<ENV
export LLM_BACKEND=openai
export LLM_MODEL='$MODEL'
export LLM_BASE_URL=http://localhost:$PORT
export LLM_PYTHON=$PY
ENV

echo "== Smoke test"
LLM_MODEL="$MODEL" LLM_BASE_URL="http://localhost:$PORT" "$PY" - <<'PY'
import os, sys
sys.path.insert(0, "scripts")
from llm_common import LLMClient
r = LLMClient("openai", os.environ["LLM_MODEL"], os.environ["LLM_BASE_URL"], timeout=600).generate(
    "Say hello in Polish.", max_tokens=32)
print("ok:", r.ok, repr(r.text), r.error)
sys.exit(0 if r.ok and r.text.strip() else 1)
PY
mark ready
touch "$W/READY"
echo "READY"
