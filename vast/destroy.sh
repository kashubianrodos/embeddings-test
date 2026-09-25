#!/usr/bin/env bash
# Destroy the instance (billing stops; its disk is deleted — fetch results first).
#   ./vast/destroy.sh [-y] [-q] [INSTANCE_ID]
#     -y  don't ask      -q  quiet (no per-check debug lines)
# Success is verified, not assumed (the CLI exits 0 even on API errors). The instance counts as
# gone when EITHER `show instance` says so (null / 404) OR it is missing from `show instances`.
# Each attempt prints what vast actually answered, so a "still listed" can be diagnosed.
# VAST_DEBUG=1 additionally logs every vastai call with full output to vast/.debug.log.
set -uo pipefail
# shellcheck source=vast/common.sh
. "$(dirname "$0")/common.sh"
YES=0; QUIET=0
while [ $# -gt 0 ]; do
  case "$1" in -y) YES=1 ;; -q) QUIET=1 ;; *) break ;; esac
  shift
done
ID=$(current_id "${1:-}")

short() { printf '%s' "$1" | tr '\n' ' ' | tr -s ' ' | cut -c1-"${2:-220}"; }
dbg() { [ "$QUIET" = 1 ] || log "  · $*"; }

clear_state() {
  if [ "$(cat "$STATE_ID" 2>/dev/null)" = "$ID" ]; then
    rm -f "$STATE_ID" "$STATE_MODE" "$STATE_META" "$STATE_EVENTS"
    dbg "cleared local state ($STATE_ID)"
  fi
}

# One verification round: prints both views, returns 0 if gone.
check_gone() {
  local show list st st_why in in_why
  show=$(run_limited 30 vast show instance "$ID" --raw 2>&1)
  st_why=$(printf '%s' "$show" | python3 "$TOOL" status --why)
  st=$(printf '%s' "$st_why" | cut -f1)
  list=$(run_limited 30 vast show instances --raw 2>&1)
  in_why=$(printf '%s' "$list" | python3 "$TOOL" in-list --why "$ID")
  in=$(printf '%s' "$in_why" | cut -f1)
  dbg "show instance $ID → $st ($(printf '%s' "$st_why" | cut -f2-))"
  dbg "show instances  → $in ($(printf '%s' "$in_why" | cut -f2-))"
  if [ "$st" = unknown ] && [ "$in" = unknown ]; then
    dbg "raw show instance: $(short "$show")"
    dbg "raw show instances: $(short "$list")"
  fi
  case "$st_why$in_why" in
    *"API error 401"*|*"API error 403"*)
      die "vast rejected the API key (401/403) — can't check or destroy anything. API key from: $(api_key_source)" ;;
  esac
  [ "$st" = gone ] || [ "$in" = absent ]
}

dbg "vastai $(vastai --version 2>&1 | head -n1); API key from: $(api_key_source)"

if check_gone; then
  log "✅ Instance $ID does not exist (already destroyed) — nothing to do."
  clear_state; exit 0
fi

if [ "$YES" != 1 ]; then
  read -r -p "Destroy instance $ID? [y/N] " ans; [[ "$ans" =~ ^[Yy]$ ]] || exit 0
fi

for attempt in 1 2 3; do
  OUT=$(run_limited 60 vast destroy instance "$ID" -y --raw 2>&1); RC=$?
  case "$OUT" in
    *"unrecognized arguments"*|*"unrecognized argument"*)   # old CLI without -y: answer the prompt
      dbg "this vastai has no -y; retrying with piped confirmation (pip install -U 'vastai>=1.8.1' to fix)"
      OUT=$(printf 'y\n' | vast destroy instance "$ID" --raw 2>&1); RC=$? ;;
  esac
  dbg "destroy attempt $attempt → rc=$RC: $(short "$OUT")"
  for _ in 1 2 3 4 5 6; do
    if check_gone; then
      log "✅ Instance $ID destroyed — billing stopped."
      clear_state; exit 0
    fi
    sleep 5
  done
  log "Instance $ID still listed after destroy attempt $attempt"
done
printf '\n🚨 COULD NOT CONFIRM DESTROY of instance %s — it may still be billing.\n   Check: vastai show instances   /   https://cloud.vast.ai/instances/\n   If the console shows nothing, clear local state: rm vast/.instance_*   (and send me the lines above)\n' "$ID" >&2
exit 1
