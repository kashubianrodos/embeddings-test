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

# fail_reason <word> <message>: a failure the local orchestrator can act on. "slow_network"
# makes bench.sh destroy this host and retry on another one.
fail_reason() { echo "$1" > "$W/SETUP_FAILED"; mark setup_failed reason="$1"; echo "SETUP FAILED ($1): $2"; exit 1; }

# net_probe <url>: real download speed in Mb/s (first PROBE_MB MB, max 60 s). Advertised
# inet_down means little — a host that claimed 1342 Mb/s delivered ~10 Mb/s from HF.
net_probe() {
  local url="$1" mb="${PROBE_MB:-200}" auth=() bps
  [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
  bps=$(curl -sL "${auth[@]}" -r "0-$((mb * 1048576 - 1))" --max-time 60 -o /dev/null \
        -w '%{speed_download}' "$url" 2>/dev/null || echo 0)
  awk -v b="${bps:-0}" 'BEGIN{printf "%d", b * 8 / 1e6}'
}

# check_network <url> <label>: probe and fail with slow_network below MIN_NET_MBPS.
check_network() {
  command -v curl >/dev/null || { echo "(no network probe: curl missing)"; return 0; }
  local mbps; mbps=$(net_probe "$1")
  mark net_probe mbps="$mbps" target="$2"
  if [ "$mbps" -lt "${MIN_NET_MBPS:-150}" ]; then
    fail_reason slow_network "only $mbps Mb/s from $2 (MIN_NET_MBPS=${MIN_NET_MBPS:-150})"
  fi
  echo "network ok: $mbps Mb/s from $2"
}

# download_watchdog <dir> <pid>: in the background, kill <pid> if <dir> grows slower than
# MIN_PULL_MBPS over a WATCH_MIN-minute window (after the first window). Writes the reason.
download_watchdog() {
  local dir="$1" pid="$2" win=$(( ${WATCH_MIN:-5} * 60 )) min="${MIN_PULL_MBPS:-80}"
  (
    local b0 b1 mbps
    b0=$(dir_bytes "$dir"); b0=${b0:-0}
    while kill -0 "$pid" 2>/dev/null; do
      sleep "$win"
      kill -0 "$pid" 2>/dev/null || break
      b1=$(dir_bytes "$dir"); b1=${b1:-0}
      mbps=$(( (b1 - b0) * 8 / win / 1000000 ))
      echo "== watchdog: $mbps Mb/s over the last ${WATCH_MIN:-5} min"
      # only while files are still downloading (Ollama: *-partial*, HF: *.incomplete) —
      # the final checksum/verify phase writes nothing and must not count as a stall
      [ -n "$(find "$dir" \( -name '*-partial*' -o -name '*.incomplete' \) -print -quit 2>/dev/null)" ] || { b0=$b1; continue; }
      if [ "$mbps" -lt "$min" ]; then
        echo "slow_network" > "$W/SETUP_FAILED"; mark setup_failed reason=slow_network mbps="$mbps"
        echo "== watchdog: download too slow ($mbps < MIN_PULL_MBPS=$min Mb/s) — stopping"
        kill "$pid" 2>/dev/null; break
      fi
      b0=$b1
    done
  ) &
}

# quiet_progress: turn \r progress bars into one line per ~30 s (setup.log was 5 MB of bars)
quiet_progress() {
  local last=-100 line
  tr '\r' '\n' | while IFS= read -r line; do
    case "$line" in
      *%*|*pulling*|*"#"*) [ $((SECONDS - last)) -lt 30 ] && continue; last=$SECONDS ;;
    esac
    printf '%s\n' "$line"
  done
}
