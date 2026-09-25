#!/usr/bin/env bash
# Copy reports + logs from the instance into ./output_vast/<instance-id>/ and write
# run_summary.md (measured phases vs. estimate). Safe to call on a failed run, and it never
# prompts: SSH runs in BatchMode, every step is time-limited, and vast's container/daemon
# logs are always saved via the API (they explain a container that died or never got SSH).
#   ./vast/fetch_results.sh [INSTANCE_ID]
set -uo pipefail
# shellcheck source=vast/common.sh
. "$(dirname "$0")/common.sh"
ID=$(current_id "${1:-}")
DEST="$ROOT_DIR/output_vast/$ID"; mkdir -p "$DEST"

# 1. vast API logs — work even when the container is dead or SSH never came up
run_limited 45 vast logs "$ID" --tail 1000 > "$DEST/container.log" 2>&1
run_limited 45 vast logs "$ID" --tail 300 --daemon-logs > "$DEST/daemon.log" 2>&1
run_limited 30 vast show instance "$ID" --raw > "$DEST/instance_final.json" 2>&1

# 2. files from /workspace over SSH (only if the container is up)
if [ "$(instance_status "$ID")" = running ] && ssh_target "$ID"; then
  # shellcheck disable=SC2086
  run_limited 120 scp $SSH_OPTS -P "$SSH_PORT" -r "$SSH_USERHOST:/workspace/embeddings-test/output/*" "$DEST/" 2>/dev/null \
    || log "No reports on the instance (benchmark did not run?)"
  # logs + timing marks; never remote.env (it holds HF_TOKEN)
  # shellcheck disable=SC2086
  run_limited 60 scp $SSH_OPTS -P "$SSH_PORT" "$SSH_USERHOST:/workspace/*.log" "$SSH_USERHOST:/workspace/bench.env" "$DEST/" 2>/dev/null \
    || log "Could not copy /workspace logs over SSH"
else
  log "Instance not reachable over SSH — only vast's container/daemon logs were saved"
fi

[ -f "$STATE_EVENTS" ] && cp "$STATE_EVENTS" "$DEST/local_events.log"
[ -f "$STATE_META" ] && cp "$STATE_META" "$DEST/instance_meta.json"
if [ -f "$STATE_META" ]; then
  python3 "$TOOL" summary --meta "$STATE_META" --events "$STATE_EVENTS" --timings "$DEST/timings.log" \
    --out-dir "$DEST" --outcome "${OUTCOME:-manual}" >/dev/null || log "Could not write run_summary.md"
fi
log "Results in $DEST ($(find "$DEST" -type f | wc -l | tr -d ' ') files)"
[ -f "$DEST/run_summary.md" ] && cat "$DEST/run_summary.md" >&2
exit 0
