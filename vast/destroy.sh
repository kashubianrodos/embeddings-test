#!/usr/bin/env bash
# Destroy the instance (billing stops; its disk is deleted — fetch results first).
#   ./vast/destroy.sh [-y] [INSTANCE_ID]
# Success is verified with `show instance` — the CLI's exit code is not reliable.
set -uo pipefail
# shellcheck source=vast/common.sh
. "$(dirname "$0")/common.sh"
YES=0; [ "${1:-}" = "-y" ] && { YES=1; shift; }
ID=$(current_id "${1:-}")

if [ "$YES" != 1 ]; then
  read -r -p "Destroy instance $ID? [y/N] " ans; [[ "$ans" =~ ^[Yy]$ ]] || exit 0
fi

for attempt in 1 2 3; do
  run_limited 60 vast destroy instance "$ID" -y --raw >/dev/null 2>&1
  for _ in 1 2 3 4 5 6; do
    if [ "$(instance_status "$ID")" = gone ]; then
      [ "$(cat "$STATE_ID" 2>/dev/null)" = "$ID" ] && rm -f "$STATE_ID" "$STATE_MODE" "$STATE_META" "$STATE_EVENTS"
      log "✅ Instance $ID destroyed — billing stopped."
      exit 0
    fi
    sleep 5
  done
  log "Instance $ID still listed after destroy attempt $attempt"
done
printf '\n🚨 COULD NOT CONFIRM DESTROY of instance %s — it may still be billing.\n   Check: vastai show instances   /   https://cloud.vast.ai/instances/\n' "$ID" >&2
exit 1
