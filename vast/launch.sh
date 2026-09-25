#!/usr/bin/env bash
# Run on YOUR machine. Finds the offer with the lowest estimated TOTAL cost per run,
# checks it against the budget, and rents it. Setup then starts by itself (onstart.sh).
# Normally called by ./vast/bench.sh; run it directly only for step-by-step debugging.
#
#   ./vast/launch.sh ollama|vllm [--interruptible|--on-demand] [--yes] [--quick]
#
# All knobs: vast/.env.example.  Why: DESIGN.md.
set -euo pipefail
# shellcheck source=vast/common.sh
. "$(dirname "$0")/common.sh"

MODE_ARG="${1:-}"; shift || true
while [ $# -gt 0 ]; do
  case "$1" in
    --interruptible) INTERRUPTIBLE=1 ;;
    --on-demand)     INTERRUPTIBLE=0 ;;
    --yes|-y)        ASSUME_YES=1 ;;
    --quick)         QUICK=1 ;;
    *) die "unknown option $1 (usage: $0 ollama|vllm [--interruptible|--on-demand] [--yes] [--quick])" ;;
  esac
  shift
done
resolve_config "$MODE_ARG"

need vastai "Install the CLI locally: pip install -r requirements-vast.txt, then: vastai set api-key <KEY>"
need python3 "Needed for offer ranking."
[ -f "$STATE_ID" ] && die "An instance is already tracked in $STATE_ID ($(cat "$STATE_ID")). Destroy it first: ./vast/destroy.sh"

REPO_COMMIT=$(git -C "$ROOT_DIR" rev-parse HEAD)

KIND="on-demand"; [ "$INTERRUPTIBLE" = 1 ] && KIND="interruptible"
log "Mode $MODE/$PRESET, $KIND, image $IMAGE, disk ${DISK_GB} GB, model $LLM_MODEL"
log "Region: $REGION${REGION_CODES:+ ($REGION_CODES)}"
log "Query: $QUERY"

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
TYPE_ARGS=""; [ "$INTERRUPTIBLE" = 1 ] && TYPE_ARGS="--type bid"
# shellcheck disable=SC2086
vast search offers "$QUERY" $TYPE_ARGS --storage "$DISK_GB" -o 'dph' --limit 200 --raw > "$TMP/offers.json" \
  || die "vastai search offers failed"

python3 "$TOOL" rank --offers "$TMP/offers.json" --out "$TMP/chosen.json" \
  --disk "$DISK_GB" --download-gb "$DOWNLOAD_GB" --setup-hours "$SETUP_HOURS" --bench-hours "$BENCH_HOURS" \
  --net-eff "$NET_EFFICIENCY" --max-inet-cost "$MAX_INET_DOWN_COST" \
  --interruptible "$INTERRUPTIBLE" --bid-mult "$BID_MULTIPLIER" --offer-id "$OFFER_ID" \
  || die "No usable offer. Relax the filters in vast/.env (REGION=any, EXTRA_QUERY, MIN_INET_DOWN_*, MAX_INET_DOWN_COST, GPU_NAME)."

get() { python3 "$TOOL" get "$TMP/chosen.json" "$1"; }
OFFER=$(get offer_id); EST=$(get est_total_usd); PRICE=$(get price_h); BID=$(get bid_price)
OVER=$(awk -v e="$EST" -v m="$MAX_RUN_USD" 'BEGIN{print (e > m) ? 1 : 0}')
[ "$OVER" = 1 ] && die "Cheapest run is estimated at \$$EST > MAX_RUN_USD \$$MAX_RUN_USD. Raise the cap or relax filters."

echo >&2
log "Chosen offer $OFFER: $(get gpu_name), \$$PRICE/h ($KIND${BID:+, bid \$$BID}), est. $(get est_hours) h → \$$EST (cap \$$MAX_RUN_USD)"
if [ "$ASSUME_YES" != 1 ]; then
  read -r -p "Rent offer $OFFER? [y/N] " ans; [[ "$ans" =~ ^[Yy]$ ]] || exit 0
fi

ENV_ARGS="-e BENCH_ENV_B64=$(remote_env_b64)"
BID_ARGS=""; [ "$INTERRUPTIBLE" = 1 ] && BID_ARGS="--bid_price $BID"
# shellcheck disable=SC2086
RESP=$(vast create instance "$OFFER" --image "$IMAGE" --disk "$DISK_GB" --ssh --direct \
  --label "bench-$BENCH_LABEL" --env "$ENV_ARGS" --onstart "$VAST_DIR/onstart.sh" $BID_ARGS --raw) \
  || die "vastai create instance failed: $RESP"
INSTANCE_ID=$(printf '%s' "$RESP" | python3 "$TOOL" new-id) || die "Could not read the new instance id from: $RESP"

echo "$INSTANCE_ID" > "$STATE_ID"
echo "$MODE" > "$STATE_MODE"
: > "$STATE_EVENTS"; event created id="$INSTANCE_ID"
python3 - "$TMP/chosen.json" "$STATE_META" <<PY
import json, sys, time
m = json.load(open(sys.argv[1]))
m.update(instance_id=int("$INSTANCE_ID"), mode="$MODE", preset="$PRESET", image="$IMAGE", disk_gb=$DISK_GB,
         download_gb=$DOWNLOAD_GB, model="$LLM_MODEL", repo_commit="$REPO_COMMIT", created_at=int(time.time()))
json.dump(m, open(sys.argv[2], "w"), indent=1)
PY

cat >&2 <<MSG

✅ Instance $INSTANCE_ID created ($MODE/$PRESET, $KIND). Billing has started.
   Recommended: let ./vast/bench.sh drive it. Manual steps:
   ./vast/ssh.sh 'tail -f /workspace/setup.log'      # wait for READY
   ./vast/ssh.sh 'bash /workspace/embeddings-test/vast/run_llm_bench.sh'
   ./vast/fetch_results.sh && ./vast/destroy.sh      # stop paying!
MSG
