#!/usr/bin/env bash
# Runs ON the vast.ai instance (vllm/vllm-openai image). Starts an OpenAI-compatible server.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${LLM_MODEL:-huihui-ai/Huihui-Qwen3.8-27B-abliterated}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
TP="${VLLM_TP:-$(nvidia-smi -L | wc -l)}"
EXTRA="${VLLM_EXTRA_ARGS:---reasoning-parser qwen3}"
export HF_HOME="${HF_HOME:-/workspace/hf}"

echo "== Packages"
export DEBIAN_FRONTEND=noninteractive
(apt-get update -qq && apt-get install -y -qq curl procps >/dev/null) || true
python3 -m pip install -q -r requirements-llm.txt || python3 -m pip install -q --break-system-packages -r requirements-llm.txt
python3 -c "import vllm; print('vLLM', vllm.__version__)"

echo "== Starting vLLM (downloads ~56 GB on first start)"
if ! pgrep -f "vllm serve" >/dev/null; then
  # shellcheck disable=SC2086
  nohup vllm serve "$MODEL" --served-model-name "$MODEL" \
    --max-model-len "$MAX_LEN" --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.92 --port 8000 $EXTRA > /workspace/vllm.log 2>&1 &
fi

for i in $(seq 360); do   # up to 60 min
  curl -sf localhost:8000/v1/models >/dev/null && break
  pgrep -f "vllm serve" >/dev/null || { echo "vLLM exited:"; tail -n 50 /workspace/vllm.log; exit 1; }
  [ $((i % 6)) -eq 0 ] && echo "... still loading ($((i/6)) min)"; sleep 10
done

cat > /workspace/bench.env <<ENV
export LLM_BACKEND=openai
export LLM_MODEL='$MODEL'
ENV
echo "READY"
