#!/usr/bin/env bash
# Runs ON the instance. Executes all LLM tests while logging GPU usage.
#   bash vast/run_llm_bench.sh            # full run
#   QUICK=1 bash vast/run_llm_bench.sh    # smoke-size run
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f /workspace/bench.env ] && source /workspace/bench.env
export LLM_BACKEND="${LLM_BACKEND:-ollama}"
PY=python3; [ -x venv/bin/python ] && PY=venv/bin/python

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-' | tr -cd '[:alnum:]-')
export LLM_TAG="${LLM_TAG:-${GPU_NAME}-${LLM_BACKEND}}"
mkdir -p output
STAMP=$(date +%Y%m%d_%H%M%S)

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 1 > "output/gpu_log_${LLM_TAG}_${STAMP}.csv" &
MON=$!; trap 'kill $MON 2>/dev/null || true' EXIT

if [ "${QUICK:-0}" = "1" ]; then
  N=5; QN=10; CONC="1,2"; PREFILL="1024"
else
  N=20; QN=50
  if [ "$LLM_BACKEND" = "ollama" ]; then CONC="1,2,4"; PREFILL="1024,4096"
  else CONC="1,4,8,16,32"; PREFILL="1024,4096,12000"; fi
fi

echo "== Speed test";   $PY scripts/llm_speed_test.py  --n "$N" --prefill-sizes "$PREFILL"
echo "== Load test";    $PY scripts/llm_load_test.py   --concurrency "$CONC"
echo "== Quality test"; $PY scripts/llm_quality_test.py --n "$QN"
echo "== Done. Reports in $(pwd)/output"
