#!/usr/bin/env bash
# Copy reports from the instance into ./output_vast/<instance-id>/
set -euo pipefail
cd "$(dirname "$0")"
ID="${INSTANCE_ID:-$(cat .instance_id)}"
URL=$(vastai ssh-url "$ID"); HOSTPORT="${URL#ssh://}"
USERHOST="${HOSTPORT%:*}"; PORT="${HOSTPORT##*:}"
DEST="../output_vast/$ID"; mkdir -p "$DEST"
scp -o StrictHostKeyChecking=accept-new -P "$PORT" -r "$USERHOST:/workspace/embeddings-test/output/*" "$DEST/"
scp -o StrictHostKeyChecking=accept-new -P "$PORT" "$USERHOST:/workspace/*.log" "$DEST/" 2>/dev/null || true
echo "✅ Results in $(cd "$DEST" && pwd)"
