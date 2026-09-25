#!/usr/bin/env bash
# Sourced ON the instance by setup_*.sh and run_llm_bench.sh.
# Loads the config forwarded by the local launcher and provides timing marks.
W=/workspace
# shellcheck disable=SC1091
[ -f "$W/remote.env" ] && . "$W/remote.env"
export HF_TOKEN="${HF_TOKEN:-}"

# mark <event> [k=v ...] -> /workspace/timings.log (fetched and summarised locally)
mark() { echo "$(date +%s) $*" >> "$W/timings.log"; echo "== [$(date -u +%T)] $*"; }

# A signal means the container is stopping (preemption / docker stop), NOT that the step
# failed: the step reruns after resume, so no failure marker is written in that case.
SIGNALED=0
on_signal() { SIGNALED=1; echo "== received $1 (container stopping?)"; exit 143; }
trap_signals() { trap 'on_signal TERM' TERM; trap 'on_signal INT' INT; trap 'on_signal HUP' HUP; }

# For setup scripts: a real failure leaves SETUP_FAILED so the local orchestrator stops early.
setup_guard() {
  rm -f "$W/READY" "$W/SETUP_FAILED"
  trap_signals
  # shellcheck disable=SC2154  # rc is assigned inside the trap
  trap 'rc=$?; if [ "$SIGNALED" = 0 ] && [ ! -f "$W/READY" ]; then mark setup_failed rc=$rc; touch "$W/SETUP_FAILED"; fi' EXIT
}

dir_bytes() { du -sb "$1" 2>/dev/null | cut -f1; }
