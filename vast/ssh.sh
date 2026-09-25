#!/usr/bin/env bash
# Run a command (or open a shell) on the rented instance: ./vast/ssh.sh ['cmd']
set -euo pipefail
cd "$(dirname "$0")"
ID="${INSTANCE_ID:-$(cat .instance_id)}"
URL=$(vastai ssh-url "$ID")                 # ssh://root@HOST:PORT
HOSTPORT="${URL#ssh://}"; USERHOST="${HOSTPORT%:*}"; PORT="${HOSTPORT##*:}"
exec ssh -o StrictHostKeyChecking=accept-new -p "$PORT" "$USERHOST" "$@"
