#!/usr/bin/env bash
# Shared helpers for the LOCAL vast.ai scripts (launch/bench/ssh/fetch/destroy).
# Sourced, not executed. Written for bash 3.2 (macOS default): no associative arrays,
# no ${var,,}, no `timeout`, no GNU-only flags.
# shellcheck disable=SC2034  # variables are consumed by the scripts that source this file

VAST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$VAST_DIR")"
STATE_ID="$VAST_DIR/.instance_id"
STATE_MODE="$VAST_DIR/.instance_mode"
STATE_META="$VAST_DIR/.instance_meta.json"
STATE_EVENTS="$VAST_DIR/.instance_events"
TOOL="$VAST_DIR/vast_tool.py"

log()  { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }
die()  { printf '\n❌ %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "$1 not found. $2"; }
event() { printf '%s %s\n' "$(date +%s)" "$*" >> "$STATE_EVENTS"; }
secs_from_hours() { awk -v h="$1" 'BEGIN{printf "%d", h*3600}'; }

# Load KEY=VALUE lines from a file without overriding variables already set in the
# environment (so `MAX_HOURS=1 ./vast/bench.sh` beats vast/.env).
load_env_file() {
  local f="$1" line key
  [ -f "$f" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    case "$line" in ''|'#'*) continue ;; esac
    line="${line#export }"
    key="${line%%=*}"
    case "$key" in ''|*[!A-Za-z0-9_]*) continue ;; esac
    if [ -z "${!key+x}" ]; then eval "export $line"; fi
  done < "$f"
}

# vast/.env + secret aliases. Runs whenever common.sh is sourced, so standalone scripts
# (destroy.sh, fetch_results.sh, ssh.sh) authenticate exactly like bench.sh does.
load_local_env() {
  load_env_file "$VAST_DIR/.env"
  # secrets: accept the names used in vast/.env as well as the canonical ones
  if [ -z "${VAST_API_KEY:-}" ] && [ -n "${VAST_AI_API_KEY:-}" ]; then export VAST_API_KEY="$VAST_AI_API_KEY"; fi
  if [ -z "${HF_TOKEN:-}" ] && [ -n "${HF_KEY:-}" ]; then export HF_TOKEN="$HF_KEY"; fi
  return 0
}

api_key_source() {
  if [ -n "${VAST_API_KEY:-}" ]; then echo "VAST_API_KEY env (vast/.env or shell)"
  elif [ -s "${XDG_CONFIG_HOME:-$HOME/.config}/vastai/vast_api_key" ]; then echo "$HOME/.config/vastai/vast_api_key"
  elif [ -s "$HOME/.vast_api_key" ]; then echo "$HOME/.vast_api_key (legacy)"
  else echo "NONE — set VAST_AI_API_KEY in vast/.env or run: vastai set api-key <KEY>"; fi
}

# resolve_config <ollama|vllm>: every knob gets a value; nothing is hidden in other scripts.
resolve_config() {
  MODE="$1"
  case "$MODE" in ollama|vllm) ;; *) die "usage: mode must be 'ollama' or 'vllm' (got '$MODE')" ;; esac
  load_local_env

  REPO_URL="${REPO_URL:-https://github.com/kashubianrodos/embeddings-test.git}"
  REPO_BRANCH="${REPO_BRANCH:-$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"

  BID_MULTIPLIER="${BID_MULTIPLIER:-1.25}"
  OUTBID_WAIT_MIN="${OUTBID_WAIT_MIN:-15}"
  MAX_HOURS="${MAX_HOURS:-3}"
  QUICK="${QUICK:-0}"
  ASSUME_YES="${ASSUME_YES:-0}"
  NET_EFFICIENCY="${NET_EFFICIENCY:-0.5}"
  SETUP_HOURS="${SETUP_HOURS:-0.1}"
  MAX_INET_DOWN_COST="${MAX_INET_DOWN_COST:-0.02}"
  MIN_RELIABILITY="${MIN_RELIABILITY:-0.98}"
  OFFER_ID="${OFFER_ID:-}"
  EXTRA_QUERY="${EXTRA_QUERY:-}"
  GPU_NAME="${GPU_NAME:-}"
  SSH_WAIT_MIN="${SSH_WAIT_MIN:-10}"
  FETCH_TIMEOUT="${FETCH_TIMEOUT:-180}"

  if [ "$MODE" = ollama ]; then
    INTERRUPTIBLE="${INTERRUPTIBLE:-1}"
    MAX_RUN_USD="${MAX_RUN_USD_OLLAMA:-3}"
    BENCH_HOURS="${BENCH_HOURS_OLLAMA:-0.75}"
    MIN_INET_DOWN="${MIN_INET_DOWN_OLLAMA:-500}"
    # vast's own base image: built for SSH launch mode (sshd, onstart). Ollama is installed
    # at a pinned version by setup_ollama.sh (ollama/ollama failed in SSH mode — DESIGN.md Q7).
    IMAGE="${OLLAMA_IMAGE:-vastai/base-image:stock-ubuntu24.04-py312-2026-09-07}"
    OLLAMA_VERSION="${OLLAMA_VERSION:-0.34.4}"
    CUDA_MIN="${OLLAMA_CUDA_MIN:-12.4}"
    DISK_GB="${OLLAMA_DISK_GB:-40}"
    DOWNLOAD_GB="${OLLAMA_DOWNLOAD_GB:-22}"
    LLM_MODEL="${LLM_MODEL:-${OLLAMA_LLM_MODEL:-hf.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF:Huihui-Qwen3.8-27B-abliterated-Q4_K.gguf}}"
    LLM_REVISION=""
    OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"
    OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
    OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-f16}"
    if [ -n "$GPU_NAME" ]; then GPU_FILTER="gpu_name=$GPU_NAME"
    else GPU_FILTER="gpu_name in [${OLLAMA_GPU_NAMES:-RTX_3090,RTX_4090}]"; fi
    PRESET="q4k"
  else
    INTERRUPTIBLE="${INTERRUPTIBLE:-0}"
    MAX_RUN_USD="${MAX_RUN_USD_VLLM:-6}"
    BENCH_HOURS="${BENCH_HOURS_VLLM:-1.5}"
    MIN_INET_DOWN="${MIN_INET_DOWN_VLLM:-1000}"
    IMAGE="${VLLM_IMAGE:-vastai/vllm:v0.29.0-cuda-12.9}"
    CUDA_MIN="${VLLM_CUDA_MIN:-12.9}"
    PRESET="${VLLM_PRESET:-fp8}"
    case "$PRESET" in
      fp8)
        LLM_MODEL="${LLM_MODEL:-${VLLM_LLM_MODEL:-leoncca/Qwen3.8-27B-Huihui-Mixed-FP8}}"
        LLM_REVISION="${VLLM_LLM_REVISION:-3ea006fab18c94e486f446a874c15e25a02a0bd1}"
        GPU_FILTER="${VLLM_GPU_QUERY:-gpu_ram>=45 compute_cap>=890}"
        DISK_GB="${VLLM_DISK_GB:-90}"
        DOWNLOAD_GB="${VLLM_DOWNLOAD_GB:-44}" ;;
      bf16)
        LLM_MODEL="${LLM_MODEL:-${VLLM_LLM_MODEL:-huihui-ai/Huihui-Qwen3.8-27B-abliterated}}"
        LLM_REVISION="${VLLM_LLM_REVISION:-main}"
        GPU_FILTER="${VLLM_GPU_QUERY:-gpu_ram>=79 compute_cap>=800}"
        DISK_GB="${VLLM_DISK_GB:-130}"
        DOWNLOAD_GB="${VLLM_DOWNLOAD_GB:-72}" ;;
      *) die "VLLM_PRESET must be fp8 or bf16 (got '$PRESET')" ;;
    esac
    [ -n "$GPU_NAME" ] && GPU_FILTER="$GPU_FILTER gpu_name=$GPU_NAME"
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
    VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.92}"
    VLLM_TP="${VLLM_TP:-}"
    VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---language-model-only --kv-cache-dtype fp8 --reasoning-parser qwen3}"
  fi
  BENCH_MODE="$MODE"
  BENCH_LABEL="$MODE-$PRESET"

  QUERY="num_gpus=1 $GPU_FILTER reliability>$MIN_RELIABILITY inet_down>=$MIN_INET_DOWN disk_space>=$DISK_GB cuda_vers>=$CUDA_MIN rentable=true"
  [ -n "$EXTRA_QUERY" ] && QUERY="$QUERY $EXTRA_QUERY"
  return 0
}

# Everything the instance needs, as `export K='v'` lines, base64url-encoded into ONE env var.
# One opaque value avoids vast's -e parsing/quoting rules; onstart.sh decodes it to /workspace/remote.env.
remote_env_b64() {
  local k v
  {
    for k in BENCH_MODE BENCH_LABEL LLM_MODEL LLM_REVISION REPO_URL REPO_BRANCH REPO_COMMIT QUICK HF_TOKEN \
             OLLAMA_VERSION OLLAMA_NUM_PARALLEL OLLAMA_CONTEXT_LENGTH OLLAMA_KV_CACHE_TYPE \
             VLLM_MAX_MODEL_LEN VLLM_GPU_MEM_UTIL VLLM_TP VLLM_EXTRA_ARGS; do
      v="${!k-}"
      [ -n "$v" ] || continue
      v=$(printf '%s' "$v" | sed "s/'/'\\\\''/g")
      printf "export %s='%s'\n" "$k" "$v"
    done
  } | base64 | tr -d '\n=' | tr '+/' '-_'
}

# vast: the vastai CLI. With VAST_DEBUG=1 every call, its exit code and (truncated) output go
# to vast/.debug.log. BENCH_ENV_B64 is redacted there because it carries HF_TOKEN.
vast() {
  if [ "${VAST_DEBUG:-0}" != 1 ]; then command vastai "$@"; return; fi
  local out err rc t0
  out=$(mktemp); err=$(mktemp); t0=$(date +%s)
  command vastai "$@" >"$out" 2>"$err"; rc=$?
  {
    printf '\n[%s] $ vastai %s\n    rc=%s  %ss\n' "$(date '+%F %T')" "$*" "$rc" "$(( $(date +%s) - t0 ))" \
      | sed 's/BENCH_ENV_B64=[^ ]*/BENCH_ENV_B64=<redacted>/'
    printf -- '--- stdout ---\n'; head -c 3000 "$out"; printf '\n--- stderr ---\n'; head -c 2000 "$err"; printf '\n'
  } >> "$VAST_DIR/.debug.log"
  cat "$out"; cat "$err" >&2; rm -f "$out" "$err"
  return $rc
}

current_id() {
  if [ -n "${1:-}" ]; then echo "$1"; return; fi
  [ -f "$STATE_ID" ] || die "No instance id given and $STATE_ID not found."
  cat "$STATE_ID"
}

instance_status() {  # prints actual_status, "gone" (positive evidence only), or "unknown"
  vast show instance "$1" --raw 2>&1 </dev/null | python3 "$TOOL" status
}
# instance_gone <id>: 0 if the instance no longer exists. Two independent checks: the
# single-instance lookup (null / 404) OR absence from the full `show instances` list.
instance_gone() {
  [ "$(instance_status "$1")" = gone ] && return 0
  [ "$(vast show instances --raw 2>&1 </dev/null | python3 "$TOOL" in-list "$1")" = absent ]
}
instance_msg() {     # vast's status_msg (why a container exited / is loading)
  vast show instance "$1" --raw 2>/dev/null </dev/null | python3 "$TOOL" status --field status_msg
}

# run_limited <seconds> <cmd...>: run cmd, kill it after <seconds>. bash 3.2 has no `timeout`.
# Nothing in cleanup may block forever (a password prompt once hung the whole run).
run_limited() {
  local secs="$1"; shift
  "$@" </dev/null &
  local pid=$!
  ( sleep "$secs"; kill -TERM "$pid" 2>/dev/null; sleep 5; kill -KILL "$pid" 2>/dev/null ) >/dev/null 2>&1 &
  local watchdog=$!
  wait "$pid"; local rc=$?
  kill "$watchdog" 2>/dev/null; wait "$watchdog" 2>/dev/null
  return $rc
}

# SSH to ephemeral vast hosts: IPs/ports get reused, so don't pollute known_hosts.
# BatchMode=yes: never prompt for a password — fail instead.
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"

ssh_target() {  # sets SSH_USERHOST, SSH_PORT
  local url hp
  url=$(vast ssh-url "$1" 2>/dev/null </dev/null) || return 1
  case "$url" in ssh://*) ;; *) return 1 ;; esac
  hp="${url#ssh://}"
  SSH_USERHOST="${hp%:*}"
  SSH_PORT="${hp##*:}"
}

remote() {  # remote <id> <command string>
  local id="$1"; shift
  ssh_target "$id" || return 255
  # shellcheck disable=SC2086
  ssh $SSH_OPTS -p "$SSH_PORT" "$SSH_USERHOST" "$@"
}

load_local_env
