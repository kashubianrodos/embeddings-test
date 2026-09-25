#!/usr/bin/env bash
# Run a command (or open a shell) on the rented instance: ./vast/ssh.sh ['cmd']
set -euo pipefail
# shellcheck source=vast/common.sh
. "$(dirname "$0")/common.sh"
ID=$(current_id "${INSTANCE_ID:-}")
ssh_target "$ID" || die "No SSH endpoint for instance $ID yet (still loading?). Try again shortly."
# shellcheck disable=SC2086
exec ssh $SSH_OPTS -p "$SSH_PORT" "$SSH_USERHOST" "$@"
