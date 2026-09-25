#!/usr/bin/env bash
# Runs ON the instance after setup (bench.sh starts it with nohup). Executes all LLM tests
# while logging GPU usage, and leaves BENCH_DONE / BENCH_FAILED for the local orchestrator.
#   bash vast/run_llm_bench.sh            # full run
#   QUICK=1 bash vast/run_llm_bench.sh    # smoke-size run
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=vast/remote_lib.sh
. vast/remote_lib.sh
[ -f "$W/bench.env" ] || { echo "No $W/bench.env — setup has not finished"; exit 1; }
# shellcheck disable=SC1091
. "$W/bench.env"

rm -f "$W/BENCH_DONE" "$W/BENCH_FAILED"
echo $$ > "$W/BENCH_RUNNING"
COMPLETED=0
trap_signals
# Success only if the last line was reached ($? is unreliable after a signal). On a signal
# (container stopping) leave BENCH_RUNNING stale: bench.sh sees a dead pid and reruns once.
# shellcheck disable=SC2154  # rc is assigned inside the trap
trap 'rc=$?; kill "${MON:-}" 2>/dev/null || true
      if [ "$COMPLETED" = 1 ]; then rm -f "$W/BENCH_RUNNING"; mark bench_done; touch "$W/BENCH_DONE"
      elif [ "$SIGNALED" = 0 ]; then rm -f "$W/BENCH_RUNNING"; mark bench_failed rc=$rc; touch "$W/BENCH_FAILED"
      else mark bench_interrupted; fi' EXIT
mark bench_start quick="${QUICK:-0}"

export LLM_BACKEND="${LLM_BACKEND:-ollama}"
PY="${LLM_PYTHON:-python3}"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-' | tr -cd '[:alnum:]-')
export LLM_TAG="${LLM_TAG:-${GPU}-${BENCH_LABEL:-$LLM_BACKEND}}"
mkdir -p output
STAMP=$(date +%Y%m%d_%H%M%S)

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 1 > "output/gpu_log_${LLM_TAG}_${STAMP}.csv" &
MON=$!

if [ "${QUICK:-0}" = "1" ]; then
  N=5; QN=10; CONC="1,2"; PREFILL="1024"
else
  N=20; QN=50
  if [ "$LLM_BACKEND" = "ollama" ]; then
    P="${OLLAMA_NUM_PARALLEL:-4}"; CONC="1,2,$P,$((P * 2))"; PREFILL="1024,4096"   # 2×P shows queueing
  else
    CONC="1,4,8,16,32"; PREFILL="1024,4096,12000"
  fi
fi

echo "== Speed test";   "$PY" scripts/llm_speed_test.py   --n "$N" --prefill-sizes "$PREFILL"; mark speed_done
echo "== Load test";    "$PY" scripts/llm_load_test.py    --concurrency "$CONC";            mark load_done
echo "== Quality test"; "$PY" scripts/llm_quality_test.py --n "$QN";                        mark quality_done
echo "== Done. Reports in $(pwd)/output"
COMPLETED=1
