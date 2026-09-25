#!/bin/bash
# Instance bootstrap, passed to `vastai create instance --onstart`. Runs as root on EVERY
# container start (also after an interruptible instance is resumed). Keep it short: vast
# stores it as the onstart command.
# shellcheck disable=SC2015  # `a && b || fail` is intended: fail if either step fails All config arrives in ONE env var, BENCH_ENV_B64.
mkdir -p /workspace && cd /workspace || exit 1
exec >> /workspace/onstart.log 2>&1
echo "== onstart $(date -u +%FT%TZ)"
rm -f /workspace/READY /workspace/SETUP_FAILED
fail() { echo "ONSTART FAILED: $*"; echo "$(date +%s) setup_failed onstart" >> /workspace/timings.log; touch /workspace/SETUP_FAILED; exit 1; }

B="${BENCH_ENV_B64:-}"
[ -n "$B" ] || B=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^BENCH_ENV_B64=//p')
[ -n "$B" ] || fail "BENCH_ENV_B64 not set"
B=$(printf %s "$B" | tr '_-' '/+'); while [ $(( ${#B} % 4 )) -ne 0 ]; do B="$B="; done
printf %s "$B" | base64 -d > /workspace/remote.env && chmod 600 /workspace/remote.env || fail "cannot decode BENCH_ENV_B64"
. /workspace/remote.env

command -v git >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git ca-certificates >/dev/null; } || fail "cannot install git"
R=/workspace/embeddings-test
[ -d "$R/.git" ] || git clone -q -b "$REPO_BRANCH" "$REPO_URL" "$R" || fail "git clone $REPO_URL ($REPO_BRANCH)"
git -C "$R" fetch -q origin "$REPO_BRANCH" && git -C "$R" checkout -q -f "${REPO_COMMIT:-origin/$REPO_BRANCH}" || fail "checkout ${REPO_COMMIT:-$REPO_BRANCH} (pushed?)"

nohup bash "$R/vast/setup_$BENCH_MODE.sh" >> /workspace/setup.log 2>&1 &
echo "setup started (pid $!)"
