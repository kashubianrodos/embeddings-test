#!/usr/bin/env bash
# One command, whole run, always cleaned up:
#   rent cheapest run → wait for READY → benchmark → fetch logs+reports → DESTROY
#
#   ./vast/bench.sh ollama|vllm [--interruptible|--on-demand] [--quick] [--yes]
#
# The instance is destroyed on every exit path (success, failure, Ctrl-C, MAX_HOURS,
# preempted longer than OUTBID_WAIT_MIN) — always after trying to fetch the logs.
set -euo pipefail
# shellcheck source=vast/common.sh
. "$(dirname "$0")/common.sh"

MODE_ARG="${1:-}"
[ -n "$MODE_ARG" ] || die "usage: $0 ollama|vllm [--interruptible|--on-demand] [--quick] [--yes]"
resolve_config "$MODE_ARG"
POLL=${POLL_SECONDS:-30}

# ---- preflight: the instance clones REPO_URL at your local HEAD, so it must be pushed
if [ -n "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=no -- scripts vast data requirements-llm.txt)" ]; then
  log "⚠️  Uncommitted changes in scripts/ vast/ data/ — the instance will NOT see them."
fi
if git -C "$ROOT_DIR" rev-parse '@{u}' >/dev/null 2>&1; then
  AHEAD=$(git -C "$ROOT_DIR" rev-list --count '@{u}..HEAD')
  [ "$AHEAD" = 0 ] || die "Local branch is $AHEAD commit(s) ahead of its upstream. git push first — the instance clones $REPO_URL."
fi

[ -f "$STATE_ID" ] && die "An instance is already tracked in $STATE_ID ($(cat "$STATE_ID")). Destroy it first: ./vast/destroy.sh"
OUTCOME="interrupted"; ID=""
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  [ -n "$ID" ] || { [ -f "$STATE_ID" ] && ID=$(cat "$STATE_ID"); }
  if [ -n "$ID" ]; then
    log "Cleaning up instance $ID (outcome: $OUTCOME) — fetching logs, then destroying"
    OUTCOME="$OUTCOME" "$VAST_DIR/fetch_results.sh" "$ID"
    "$VAST_DIR/destroy.sh" -y "$ID"
  fi
  exit $rc
}
# Armed BEFORE renting: if launch.sh fails after the instance exists, it is still destroyed.
trap cleanup EXIT
trap 'OUTCOME=interrupted; exit 130' INT TERM

OUTCOME="launch_failed"
"$VAST_DIR/launch.sh" "$@"
[ -f "$STATE_ID" ] || { OUTCOME="not_rented"; exit 0; }   # user answered "no"
ID=$(cat "$STATE_ID")
OUTCOME="interrupted"
DEADLINE=$(( $(date +%s) + $(secs_from_hours "$MAX_HOURS") ))

remote_state() {
  # shellcheck disable=SC2016  # expands on the instance, not here
  remote "$ID" 'W=/workspace
    if   [ -f $W/SETUP_FAILED ]; then echo SETUP_FAILED
    elif [ -f $W/BENCH_DONE ]; then echo BENCH_DONE
    elif [ -f $W/BENCH_FAILED ]; then echo BENCH_FAILED
    elif [ -f $W/BENCH_RUNNING ] && kill -0 "$(cat $W/BENCH_RUNNING)" 2>/dev/null; then echo BENCH_RUNNING
    elif [ -f $W/READY ]; then echo READY
    else echo SETUP; fi' 2>/dev/null || echo UNREACHABLE
}

BENCH_STARTS=0; DOWN_SINCE=""; LAST=""
while :; do
  NOW=$(date +%s)
  if [ "$NOW" -ge "$DEADLINE" ]; then OUTCOME="max_hours"; die "MAX_HOURS=$MAX_HOURS reached."; fi

  ST=$(instance_status "$ID")
  case "$ST" in
    gone) OUTCOME="instance_gone"; die "Instance $ID disappeared." ;;
    running) DOWN_SINCE="" ;;
    stopped|exited|offline)
      # interruptible instances go to 'stopped' when outbid; storage keeps billing
      [ -n "$DOWN_SINCE" ] || { DOWN_SINCE=$NOW; event preempted status="$ST"; log "Instance is $ST (outbid/preempted?) — waiting up to ${OUTBID_WAIT_MIN} min"; }
      if [ $(( NOW - DOWN_SINCE )) -ge $(( OUTBID_WAIT_MIN * 60 )) ]; then
        OUTCOME="preempted"; die "Instance stayed $ST for ${OUTBID_WAIT_MIN} min."
      fi
      sleep "$POLL"; continue ;;
    *) ;;  # loading / created / unknown: image still pulling
  esac

  RS=SETUP; [ "$ST" = running ] && RS=$(remote_state)
  [ "$RS" != "$LAST" ] && { log "instance=$ST remote=$RS"; LAST=$RS; }
  case "$RS" in
    SETUP_FAILED)
      OUTCOME="setup_failed"
      remote "$ID" 'tail -n 25 /workspace/setup.log /workspace/onstart.log' 2>/dev/null || true
      die "Setup failed on the instance (logs will be in output_vast/$ID/)." ;;
    READY)
      [ "$BENCH_STARTS" = 0 ] && event ready
      if [ "$BENCH_STARTS" -ge 2 ]; then OUTCOME="bench_interrupted"; die "Benchmark was interrupted twice."; fi
      BENCH_STARTS=$((BENCH_STARTS + 1))
      log "Starting benchmark (attempt $BENCH_STARTS)"
      event bench_start attempt="$BENCH_STARTS"
      remote "$ID" 'cd /workspace/embeddings-test && nohup bash vast/run_llm_bench.sh >> /workspace/bench.log 2>&1 < /dev/null &' \
        || log "Could not start the benchmark over SSH; will retry"
      ;;
    BENCH_DONE)   event bench_done; OUTCOME="success"; log "Benchmark finished."; break ;;
    BENCH_FAILED)
      OUTCOME="bench_failed"
      remote "$ID" 'tail -n 30 /workspace/bench.log' 2>/dev/null || true
      die "Benchmark failed (logs will be in output_vast/$ID/)." ;;
    *) ;;  # SETUP / BENCH_RUNNING / UNREACHABLE: keep waiting
  esac
  sleep "$POLL"
done
# EXIT trap: fetch results → run_summary.md → destroy
