#!/usr/bin/env bash
# Copy reports + logs from the instance into ./output_vast/<instance-id>/ and write
# run_summary.md (measured phases vs. estimate). Safe to call on a failed run.
#   ./vast/fetch_results.sh [INSTANCE_ID]
set -uo pipefail
# shellcheck source=vast/common.sh
. "$(dirname "$0")/common.sh"
ID=$(current_id "${1:-}")
DEST="$ROOT_DIR/output_vast/$ID"; mkdir -p "$DEST"
OK=0

if ssh_target "$ID"; then
  # shellcheck disable=SC2086
  scp $SSH_OPTS -P "$SSH_PORT" -r "$SSH_USERHOST:/workspace/embeddings-test/output/*" "$DEST/" 2>/dev/null && OK=1
  # logs + timing marks; never remote.env (it holds HF_TOKEN)
  # shellcheck disable=SC2086
  scp $SSH_OPTS -P "$SSH_PORT" "$SSH_USERHOST:/workspace/*.log" "$SSH_USERHOST:/workspace/bench.env" "$DEST/" 2>/dev/null && OK=1
fi
if [ "$OK" = 0 ]; then
  log "scp failed (instance not running?) — trying vastai copy via the host"
  vast copy "C.$ID:/workspace/embeddings-test/output/" "local:$DEST/" >/dev/null 2>&1 || true
  for f in setup.log onstart.log bench.log timings.log ollama.log vllm.log; do
    vast copy "C.$ID:/workspace/$f" "local:$DEST/$f" >/dev/null 2>&1 || true
  done
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
