# LLM performance test on vast.ai — Huihui-Qwen3.8-27B-abliterated

Extends this repo (built for Ollama *embedding* models) with tests for a
**generative** 27B model. The EN/PL pairs in `data/samples.json` are reused as
prompts and as a translation-quality reference.

| Script | Measures |
| :--- | :--- |
| `scripts/llm_speed_test.py` | Single stream: TTFT, latency, decode tok/s, optional prefill sweep |
| `scripts/llm_load_test.py` | Concurrency sweep: aggregate tok/s, req/s, TTFT/latency p50/p95 |
| `scripts/llm_quality_test.py` | EN↔PL translation chrF (matches sacreBLEU), exact match, `<think>` leaks |
| `vast/*.sh` | Rent a GPU, set up Ollama or vLLM, run everything, copy results, destroy |

All scripts write Markdown + JSON to `output/` and share flags:
`--backend ollama|openai --model ... --base-url ... --think --num-ctx ... --tag ...`
(`openai` = any OpenAI-compatible server: vLLM, llama.cpp `llama-server`, SGLang).

## Two ways to run the model

| Mode | Weights | Size | GPU | Good for |
| :--- | :--- | :--- | :--- | :--- |
| `ollama` | `hf.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF:Q4_K` | ~17 GB | 1× ≥24 GB (4090/3090/L4/A5000); 48 GB for longer context | cheap, single-user latency |
| `vllm` | `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (BF16) | ~56 GB | 1× 80 GB (A100/H100) or 2× 48 GB with `VLLM_TP=2` | real throughput under concurrency |

Other quants: change `LLM_MODEL`, e.g. `...-GGUF:Q8_0` (tag = suffix of the `.gguf` file name).

## Workflow

```bash
# 0. commit/push this folder to the repo (the instance clones REPO_URL)
pip install vastai && vastai set api-key <YOUR_KEY>

# 1. rent + bootstrap (shows the cheapest matching offers, asks before renting)
./vast/launch.sh ollama            # or: ./vast/launch.sh vllm
#    HF_TOKEN=hf_xxx ./vast/launch.sh vllm      # faster, fewer rate limits on HF

# 2. wait for setup ("READY" at the end)
vastai show instance $(cat vast/.instance_id)
./vast/ssh.sh 'tail -f /workspace/setup.log'

# 3. benchmark (QUICK=1 for a 2-minute smoke run)
./vast/ssh.sh 'bash /workspace/embeddings-test/vast/run_llm_bench.sh'

# 4. copy reports + GPU logs locally, then STOP BILLING
./vast/fetch_results.sh            # -> output_vast/<instance-id>/
./vast/destroy.sh
```

Pick a specific offer: `OFFER_ID=123456 ./vast/launch.sh vllm`.
Change hardware: `GPU_QUERY='gpu_name=H100_SXM num_gpus=1 reliability>0.99' ./vast/launch.sh vllm`.

## Manual runs (on the instance or any machine with a server)

```bash
source /workspace/bench.env                      # sets backend/model
python scripts/llm_speed_test.py --n 30 --max-tokens 512 --prefill-sizes 1024,4096,8192
python scripts/llm_load_test.py  --concurrency 1,8,32,64 --requests-per-level 96
python scripts/llm_quality_test.py --n 200       # dataset has 500 pairs
```

## Reading the numbers

- **TTFT** — prefill + queueing. Rising TTFT at higher concurrency = server saturated.
- **Decode tok/s (single stream)** — what one user feels. Client-side value includes network
  jitter; Ollama also reports the exact server-side value.
- **Output tok/s at concurrency N** — serving capacity. On Ollama it plateaus at
  `OLLAMA_NUM_PARALLEL` (default here 4); vLLM keeps scaling until KV cache runs out.
- **chrF** — compare runs (quant vs BF16, backend vs backend), not an absolute grade.
- `gpu_log_*.csv` — utilization, VRAM, power per second during the whole run.

## Knobs

| Env var | Default | Where |
| :--- | :--- | :--- |
| `OLLAMA_NUM_PARALLEL` | 4 | setup_ollama.sh — each slot reserves KV cache |
| `OLLAMA_CONTEXT_LENGTH` | 8192 | setup_ollama.sh |
| `VLLM_MAX_MODEL_LEN` | 16384 | setup_vllm.sh |
| `VLLM_TP` | #GPUs | setup_vllm.sh |
| `VLLM_EXTRA_ARGS` | `--reasoning-parser qwen3` | setup_vllm.sh |
| `QUICK` | 0 | run_llm_bench.sh |

Thinking mode is **off** by default (`think=false` for Ollama,
`chat_template_kwargs.enable_thinking=false` for vLLM) so token counts measure the
answer, not hidden reasoning. Add `--think` to benchmark reasoning mode.

## Troubleshooting

- **`unknown model architecture` / load error** — Qwen3.8 is new; the backend needs a
  release that supports it. Update Ollama (`curl -fsSL https://ollama.com/install.sh | sh`)
  or use a newer vLLM image (`IMAGE=vllm/vllm-openai:nightly`). Fallback: llama.cpp's
  `llama-server -hf huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF:Q4_K --port 8000`
  and run the scripts with `--backend openai --model <name from /v1/models>`.
- **vLLM OOM** — lower `VLLM_MAX_MODEL_LEN`, or use 2 GPUs with `VLLM_TP=2`.
- **Ollama slow / partial CPU offload** — `ollama ps` should show `100% GPU`; otherwise
  lower `OLLAMA_NUM_PARALLEL`/`OLLAMA_CONTEXT_LENGTH` or rent more VRAM.
