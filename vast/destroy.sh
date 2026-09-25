#!/usr/bin/env bash
# Destroy the instance (billing stops; disk is deleted — fetch results first).
set -euo pipefail
cd "$(dirname "$0")"
ID="${INSTANCE_ID:-$(cat .instance_id)}"
read -r -p "Destroy instance $ID? [y/N] " ans; [[ "$ans" =~ ^[Yy]$ ]] || exit 0
vastai destroy instance "$ID" && rm -f .instance_id .instance_mode
