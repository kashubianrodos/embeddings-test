#!/usr/bin/env bash
# Run on YOUR machine. Rents a vast.ai GPU and bootstraps the benchmark server.
#
#   pip install vastai && vastai set api-key <KEY>     # once
#   ./vast/launch.sh ollama      # GGUF Q4_K on a >=24 GB GPU (cheap, ~17 GB weights)
#   ./vast/launch.sh vllm        # BF16 safetensors on an 80 GB GPU (~56 GB weights)
#
# Overridable env: GPU_QUERY, IMAGE, DISK_GB, LLM_MODEL, REPO_URL, REPO_BRANCH, HF_TOKEN
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-ollama}"
REPO_URL="${REPO_URL:-https://github.com/kashubianrodos/embeddings-test.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"

case "$MODE" in
  ollama)
    GPU_QUERY="${GPU_QUERY:-gpu_ram>=24 num_gpus=1 reliability>0.98 inet_down>=500 disk_space>=80 cuda_vers>=12.4 rentable=true}"
    IMAGE="${IMAGE:-ubuntu:24.04}"      # Ollama ships its own CUDA runtime
    DISK_GB="${DISK_GB:-80}" ;;
  vllm)
    GPU_QUERY="${GPU_QUERY:-gpu_ram>=79 num_gpus=1 reliability>0.98 inet_down>=1000 disk_space>=150 cuda_vers>=12.8 rentable=true}"
    IMAGE="${IMAGE:-vllm/vllm-openai:latest}"
    DISK_GB="${DISK_GB:-150}" ;;
  *) echo "usage: $0 [ollama|vllm]"; exit 1 ;;
esac

command -v vastai >/dev/null || { echo "Install the CLI: pip install vastai"; exit 1; }

echo "== Searching offers: $GPU_QUERY"
vastai search offers "$GPU_QUERY" -o 'dph' | head -n 11

OFFER_ID="${OFFER_ID:-$(vastai search offers "$GPU_QUERY" -o 'dph' --raw \
  | python3 -c 'import json,sys; o=json.load(sys.stdin); print(o[0]["id"] if o else "")')}"
[ -n "$OFFER_ID" ] || { echo "No offers match. Relax GPU_QUERY."; exit 1; }
read -r -p "Rent offer $OFFER_ID? [y/N] " ans; [[ "$ans" =~ ^[Yy]$ ]] || exit 0

ENV_ARGS="-e BENCH_MODE=$MODE"
[ -n "${LLM_MODEL:-}" ] && ENV_ARGS="$ENV_ARGS -e LLM_MODEL=$LLM_MODEL"
[ -n "${HF_TOKEN:-}" ]  && ENV_ARGS="$ENV_ARGS -e HF_TOKEN=$HF_TOKEN"

# The entrypoint of vllm images is replaced by vast's SSH launcher; onstart does the setup.
ONSTART="(command -v git || (apt-get update -qq && apt-get install -y -qq git)) \
&& (git clone -b $REPO_BRANCH $REPO_URL /workspace/embeddings-test || git -C /workspace/embeddings-test pull) \
&& bash /workspace/embeddings-test/vast/setup_$MODE.sh > /workspace/setup.log 2>&1"

RESP=$(vastai create instance "$OFFER_ID" --image "$IMAGE" --disk "$DISK_GB" \
  --ssh --direct --env "$ENV_ARGS" --onstart-cmd "$ONSTART" --raw)
echo "$RESP"
INSTANCE_ID=$(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["new_contract"])')
echo "$INSTANCE_ID" > .instance_id
echo "$MODE" > .instance_mode

cat <<MSG

✅ Instance $INSTANCE_ID created ($MODE). Next:
   vastai show instance $INSTANCE_ID           # wait for status 'running'
   ./vast/ssh.sh 'tail -f /workspace/setup.log' # watch model download (Ctrl-C when "READY")
   ./vast/ssh.sh 'bash /workspace/embeddings-test/vast/run_llm_bench.sh'
   ./vast/fetch_results.sh
   ./vast/destroy.sh                           # stop paying!
MSG
