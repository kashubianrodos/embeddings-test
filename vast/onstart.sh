#!/bin/bash
# Instance bootstrap, passed to `vastai create instance --onstart`. vast runs it as root on
# EVERY container start (also after an interruptible instance is resumed), as part of its
# SSH-mode entrypoint. So the top level must never exit/exec/cd or block: all work happens
# in a detached background script, and a failure there can only leave SETUP_FAILED — it can
# never take the container (and SSH) down with it. All config arrives in BENCH_ENV_B64.
mkdir -p /workspace
cat > /workspace/bootstrap.sh <<'BOOT'
#!/bin/bash
cd /workspace || exit 1
echo "== bootstrap $(date -u +%FT%TZ)"
rm -f /workspace/READY /workspace/SETUP_FAILED
fail() { echo "BOOTSTRAP FAILED: $*"; echo "$(date +%s) setup_failed bootstrap" >> /workspace/timings.log; touch /workspace/SETUP_FAILED; exit 1; }

B="${BENCH_ENV_B64:-}"
[ -n "$B" ] || B=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^BENCH_ENV_B64=//p')
[ -n "$B" ] || fail "BENCH_ENV_B64 not set"
B=$(printf %s "$B" | tr '_-' '/+'); while [ $(( ${#B} % 4 )) -ne 0 ]; do B="$B="; done
printf %s "$B" | base64 -d > /workspace/remote.env || fail "cannot decode BENCH_ENV_B64"
chmod 600 /workspace/remote.env
. /workspace/remote.env

if ! command -v git >/dev/null; then
  apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git ca-certificates >/dev/null || fail "cannot install git"
fi
R=/workspace/embeddings-test
export GIT_TERMINAL_PROMPT=0
[ -d "$R/.git" ] || git clone -q -b "$REPO_BRANCH" "$REPO_URL" "$R" || fail "git clone $REPO_URL ($REPO_BRANCH)"
git -C "$R" fetch -q origin "$REPO_BRANCH" || fail "git fetch"
git -C "$R" checkout -q -f "${REPO_COMMIT:-origin/$REPO_BRANCH}" || fail "checkout ${REPO_COMMIT:-$REPO_BRANCH} (pushed?)"
echo "code at $(git -C "$R" rev-parse --short HEAD); starting setup_$BENCH_MODE.sh"
exec bash "$R/vast/setup_$BENCH_MODE.sh" >> /workspace/setup.log 2>&1
BOOT
nohup bash /workspace/bootstrap.sh >> /workspace/onstart.log 2>&1 < /dev/null &
echo "bench bootstrap started (log: /workspace/onstart.log)"
