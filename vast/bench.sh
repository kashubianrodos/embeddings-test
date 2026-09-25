#!/usr/bin/env bash
# One command, whole run, always cleaned up:
#   rent cheapest run → wait for READY → benchmark → fetch logs+reports → DESTROY
#
#   ./vast/bench.sh ollama|vllm [--preset NAME] [--interruptible|--on-demand] [--quick] [--yes]
#     ./vast/bench.sh ollama --preset bielik-1.5b     # smallest Bielik instead of Qwen3.8-27B
#
# The instance is destroyed on every exit path (success, failure, Ctrl-C, MAX_HOURS, container
# exited, no SSH for SSH_WAIT_MIN, preempted longer than OUTBID_WAIT_MIN) — always after trying
# to fetch the logs, and nothing in cleanup can block (every step is time-limited, no prompts).
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
# The instance clones anonymously: check REPO_URL is readable without credentials and has HEAD.
HEAD_SHA=$(git -C "$ROOT_DIR" rev-parse HEAD)
REMOTE_REFS=$(GIT_TERMINAL_PROMPT=0 run_limited 30 git -c credential.helper= ls-remote "$REPO_URL" 2>/dev/null) \
  || die "Cannot read $REPO_URL anonymously (private repo?). The instance must be able to git clone it."
if ! printf '%s\n' "$REMOTE_REFS" | grep -q "^$HEAD_SHA"; then
  log "⚠️  Local HEAD ${HEAD_SHA:0:7} is not a branch tip on $REPO_URL — the checkout on the instance may fail if it was never pushed."
fi
# SSH needs one of YOUR keys in the vast account (vast copies account keys into new instances).
# A run without it is paid for but can never be driven — so this is a hard stop.
PRE=$(mktemp -d)
run_limited 30 vast show ssh-keys --raw > "$PRE/account" 2>&1
ssh-add -L > "$PRE/agent" 2>/dev/null || true
SSHCHK_RC=0; SSHCHK=$(python3 "$TOOL" ssh-check --account "$PRE/account" --agent "$PRE/agent") || SSHCHK_RC=$?
rm -rf "$PRE"
case "$SSHCHK_RC" in
  0) log "SSH key: $SSHCHK" ;;
  4) log "⚠️  SSH key check skipped — $SSHCHK" ;;
  *) printf '%s\n' "$SSHCHK" >&2
     K=""; for f in "$HOME"/.ssh/id_ed25519.pub "$HOME"/.ssh/id_*.pub; do [ -f "$f" ] && { K=$f; break; }; done
     die "Add your public key to vast first:  vastai create ssh-key \"\$(cat ${K:-~/.ssh/id_ed25519.pub})\"   (no key? ssh-keygen -t ed25519)" ;;
esac

[ -f "$STATE_ID" ] && die "An instance is already tracked in $STATE_ID ($(cat "$STATE_ID")). Destroy it first: ./vast/destroy.sh"
OUTCOME="interrupted"; ID=""
# fetch logs + destroy the current instance (never blocks, never aborts)
cleanup_instance() {
  [ -n "$ID" ] || { [ -f "$STATE_ID" ] && ID=$(cat "$STATE_ID"); }
  [ -n "$ID" ] || return 0
  log "Cleaning up instance $ID (outcome: $OUTCOME) — fetching logs (max ${FETCH_TIMEOUT}s), then destroying"
  OUTCOME="$OUTCOME" run_limited "$FETCH_TIMEOUT" "$VAST_DIR/fetch_results.sh" "$ID" || log "Fetching logs failed or timed out — destroying anyway"
  run_limited 240 "$VAST_DIR/destroy.sh" -y "$ID" || printf '\n🚨 Destroy did not complete for instance %s — check https://cloud.vast.ai/instances/ NOW.\n' "$ID" >&2
  ID=""
}
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  cleanup_instance
  exit $rc
}
# Armed BEFORE renting: if launch.sh fails after the instance exists, it is still destroyed.
trap cleanup EXIT
trap 'OUTCOME=interrupted; exit 130' INT TERM

DEADLINE=$(( $(date +%s) + $(secs_from_hours "$MAX_HOURS") ))   # for the whole run, all hosts

remote_state() {
  # shellcheck disable=SC2016  # expands on the instance, not here
  run_limited 60 remote "$ID" 'W=/workspace
    if   [ -f $W/SETUP_FAILED ]; then echo SETUP_FAILED
    elif [ -f $W/BENCH_DONE ]; then echo BENCH_DONE
    elif [ -f $W/BENCH_FAILED ]; then echo BENCH_FAILED
    elif [ -f $W/BENCH_RUNNING ] && kill -0 "$(cat $W/BENCH_RUNNING)" 2>/dev/null; then echo BENCH_RUNNING
    elif [ -f $W/READY ]; then echo READY
    else echo SETUP; fi' 2>/dev/null || echo UNREACHABLE
}

show_vast_logs() {  # why did the container die / never get SSH? (API, works without SSH)
  local msg; msg=$(instance_msg "$ID")
  [ -n "$msg" ] && log "vast status_msg: $msg"
  log "Last container log lines (vastai logs $ID):"
  run_limited 45 vast logs "$ID" --tail 25 2>&1 | sed 's/^/    /' >&2 || true
}

# One host: rent → setup → benchmark. Sets RETRY=1 (and returns) when the host is too slow
# to be worth keeping; anything else fatal goes through die → EXIT trap.
run_attempt() {
OUTCOME="launch_failed"; RETRY=0
"$VAST_DIR/launch.sh" "$@"
[ -f "$STATE_ID" ] || { OUTCOME="not_rented"; exit 0; }   # user answered "no"
ID=$(cat "$STATE_ID")
OUTCOME="interrupted"
BENCH_STARTS=0; DOWN_SINCE=""; NOSSH_SINCE=""; LOADING_SINCE=""; LAST=""; LAST_ST=""
LOAD_WAIT_MIN=${LOAD_WAIT_MIN:-20}
while :; do
  NOW=$(date +%s)
  if [ "$NOW" -ge "$DEADLINE" ]; then OUTCOME="max_hours"; die "MAX_HOURS=$MAX_HOURS reached."; fi

  J=$(vast show instance "$ID" --raw 2>&1 </dev/null)
  ST=$(printf '%s' "$J" | python3 "$TOOL" status)
  INTENDED=$(printf '%s' "$J" | python3 "$TOOL" status --field intended_status)
  if [ "$ST/$INTENDED" != "$LAST_ST" ]; then
    MSG=$(printf '%s' "$J" | python3 "$TOOL" status --field status_msg)
    log "instance status: $ST (intended: ${INTENDED:-?})${MSG:+ — $MSG}"; LAST_ST="$ST/$INTENDED"
  fi
  # vast stopped it (outbid, or an on-demand renter took the GPU): intended_status=stopped.
  # Only an exit while vast still intends it to run is a crash.
  if [ "$ST" = exited ] && [ "$INTENDED" != running ]; then ST=stopped; fi
  case "$ST" in
    gone) OUTCOME="instance_gone"; die "Instance $ID disappeared." ;;
    running) DOWN_SINCE=""; LOADING_SINCE="" ;;
    exited)
      # the container itself died (bad image / entrypoint / host problem) — not an outbid
      OUTCOME="container_exited"; event container_exited intended="$INTENDED"; show_vast_logs
      die "The container exited on its own (not a preemption). See output_vast/$ID/container.log and daemon.log." ;;
    stopped|offline)
      # interruptible instances go to 'stopped' when outbid; storage keeps billing
      [ -n "$DOWN_SINCE" ] || { DOWN_SINCE=$NOW; event preempted status="$ST" intended="$INTENDED"
        log "Preempted: vast stopped the instance (outbid, or an on-demand renter took the GPU) — waiting up to ${OUTBID_WAIT_MIN} min for it to resume"; }
      if [ $(( NOW - DOWN_SINCE )) -ge $(( OUTBID_WAIT_MIN * 60 )) ]; then
        OUTCOME="preempted"; die "Instance stayed $ST for ${OUTBID_WAIT_MIN} min."
      fi
      sleep "$POLL"; continue ;;
    *)  # loading / created / unknown: image still pulling
      [ -n "$LOADING_SINCE" ] || LOADING_SINCE=$NOW
      if [ $(( NOW - LOADING_SINCE )) -ge $(( LOAD_WAIT_MIN * 60 )) ]; then
        OUTCOME="stuck_loading"; show_vast_logs; die "Instance not running after ${LOAD_WAIT_MIN} min (status: $ST)."
      fi ;;
  esac

  RS=SETUP; [ "$ST" = running ] && RS=$(remote_state)
  [ "$RS" != "$LAST" ] && { log "instance=$ST remote=$RS"; LAST=$RS; }
  if [ "$RS" = UNREACHABLE ]; then
    [ -n "$NOSSH_SINCE" ] || NOSSH_SINCE=$NOW
    if [ $(( NOW - NOSSH_SINCE )) -ge $(( SSH_WAIT_MIN * 60 )) ]; then
      OUTCOME="ssh_unreachable"; show_vast_logs
      die "Running but no SSH for ${SSH_WAIT_MIN} min. Is your SSH key in the vast account (vastai show ssh-keys)?"
    fi
  else
    NOSSH_SINCE=""
  fi
  case "$RS" in
    SETUP_FAILED)
      REASON=$(run_limited 30 remote "$ID" 'cat /workspace/SETUP_FAILED' 2>/dev/null | head -n1 || true)
      run_limited 60 remote "$ID" 'tail -n 25 /workspace/setup.log /workspace/onstart.log' 2>/dev/null || true
      if [ "$REASON" = slow_network ]; then
        OUTCOME="slow_network"; event slow_network; RETRY=1; return 0
      fi
      OUTCOME="setup_failed"
      die "Setup failed on the instance (logs will be in output_vast/$ID/)." ;;
    READY)
      [ "$BENCH_STARTS" = 0 ] && event ready
      if [ "$BENCH_STARTS" -ge 2 ]; then OUTCOME="bench_interrupted"; die "Benchmark was interrupted twice."; fi
      BENCH_STARTS=$((BENCH_STARTS + 1))
      log "Starting benchmark (attempt $BENCH_STARTS)"
      event bench_start attempt="$BENCH_STARTS"
      run_limited 60 remote "$ID" 'cd /workspace/embeddings-test && nohup bash vast/run_llm_bench.sh >> /workspace/bench.log 2>&1 < /dev/null &' \
        || log "Could not start the benchmark over SSH; will retry"
      ;;
    BENCH_DONE)   event bench_done; OUTCOME="success"; log "Benchmark finished."; return 0 ;;
    BENCH_FAILED)
      OUTCOME="bench_failed"
      run_limited 60 remote "$ID" 'tail -n 30 /workspace/bench.log' 2>/dev/null || true
      die "Benchmark failed (logs will be in output_vast/$ID/)." ;;
    *) ;;  # SETUP / BENCH_RUNNING / UNREACHABLE: keep waiting
  esac
  sleep "$POLL"
done
}

HOST=1
while :; do
  run_attempt "$@"
  [ "$RETRY" = 1 ] || break
  # slow host: keep its logs, destroy it, never pick it again, try the next cheapest offer
  M=$(python3 "$TOOL" get "$STATE_META" machine_id 2>/dev/null || true)
  log "Host $HOST (machine ${M:-?}) is too slow to download the model — replacing it"
  cleanup_instance
  if [ "$HOST" -gt "$HOST_RETRIES" ]; then
    OUTCOME="slow_network"; die "Tried $HOST hosts, all too slow (MIN_NET_MBPS=$MIN_NET_MBPS). Try another REGION or lower MIN_NET_MBPS."
  fi
  [ -n "$M" ] && EXCLUDE_MACHINES="${EXCLUDE_MACHINES:+$EXCLUDE_MACHINES,}$M"
  export EXCLUDE_MACHINES
  HOST=$((HOST + 1))
  log "Retrying on another host ($HOST of $((HOST_RETRIES + 1))), excluding machines: $EXCLUDE_MACHINES"
done
# EXIT trap: fetch results → run_summary.md → destroy
