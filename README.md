# embeddings-test

Two benchmark suites in one repo:

1. **Embedding models (local Ollama)**: speed and EN↔PL semantic accuracy of small embedding models.
2. **Generative LLM on rented vast.ai GPUs**: speed, load and EN↔PL translation quality of
   `huihui-ai/Huihui-Qwen3.8-27B-abliterated`. It runs as a single command that rents the cheapest
   suitable machine and always destroys it at the end.

Both use the same dataset: `data/samples.json` holds 500 parallel English/Polish sentence pairs.
The embedding tests use the pairs for cross-lingual retrieval. The LLM tests use them as prompts and as the
translation reference.

`DESIGN.md` explains why the vast.ai part is built the way it is. This README covers how to run it.

---

## Part 1: embedding models (local)

Based on [Decoding AI's Inner Language: How to Test Your Embedding Models](https://dev.to/aairom/decoding-ais-inner-language-how-to-test-your-embedding-models-126).

| Model | Approx. size |
| :--- | :--- |
| `snowflake-arctic-embed2` | ~1.2 GB |
| `granite-embedding:278m` | ~278 MB |
| `paraphrase-multilingual` | ~563 MB |

1. **`scripts/benchmark.py`** runs the three pillars from the article: latency (per language and total),
   model size, and vector dimension. It writes a Markdown report and the raw JSON vectors to `output/`.
2. **`scripts/semantic_test.py`** tests cross-lingual retrieval. For each EN sentence, it checks whether the PL
   translation is the nearest PL vector (top-1 accuracy, both directions). It also reports the mean cosine similarity of
   true pairs vs. non-pairs, and the separation margin.
3. **`scripts/speed_test.py`** times each embedding call individually (after warmup). It reports mean, median, p95,
   min and max latency, plus throughput in texts/s.

This part requires [Ollama](https://ollama.com) running locally at `http://localhost:11434`.

```bash
./run_all.sh
```

Or manually:

```bash
ollama pull snowflake-arctic-embed2
ollama pull granite-embedding:278m
ollama pull paraphrase-multilingual

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python scripts/benchmark.py
python scripts/semantic_test.py
python scripts/speed_test.py
```

Reports land in `output/` as timestamped Markdown files.

How to read the results:

- **Speed**: a lower total duration is better for real-time or high-throughput use.
- **Dimension**: a higher dimension usually captures more semantic nuance, at a cost.
- **Top-1 accuracy**: how well the model aligns Polish and English meaning. This is the key metric for
  multilingual RAG over Polish content.
- **Separation margin**: a larger margin means a clearer distinction between related and unrelated texts, which makes
  retrieval thresholds easier to set.

---

## Part 2: generative LLM on vast.ai

### What it measures

| Script | Measures |
| :--- | :--- |
| `scripts/llm_speed_test.py` | Single stream: TTFT, latency, decode tok/s, prefill sweep |
| `scripts/llm_load_test.py` | Concurrency sweep: aggregate tok/s, req/s, TTFT and latency p50/p95 |
| `scripts/llm_quality_test.py` | EN↔PL translation chrF (matches sacreBLEU), exact match, `<think>` leaks |

There are two modes. Run them as two separate, cheap runs:

| Mode | Answers | Weights | Machine the ranker picks from | Default billing | Rough cost / run* |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ollama` | "How does it feel for one user?" and "Did quantization or abliteration hurt quality?" | GGUF Q4_K, 16.8 GB | 1× RTX 3090 or RTX 4090 (24 GB) | on-demand | ~$0.2–0.5 |
| `vllm` | "How many users can one GPU serve?" | FP8, 28 GB (`VLLM_PRESET=fp8`) | 1× 48 GB Ada or Hopper (L40S, L40, RTX 6000 Ada, H100) | on-demand | ~$1–1.5 |
| `ollama --preset bielik-1.5b` | Same questions, for the smallest Polish model: [Bielik-1.5B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-1.5B-v3.0-Instruct) | GGUF Q8_0, 1.7 GB | any ≥10 GB card | on-demand | ~$0.05–0.15 |
| `vllm` + `VLLM_PRESET=bf16` | Full-precision reference only | BF16, 56 GB | 1× 80 GB (A100, H100) | on-demand | ~$2–4 |

\*These are estimates from September 2026 market prices. Every run prints its real estimate before renting and its
actual cost afterwards (`run_summary.md`).

### One-time setup (on your machine)

1. Create a vast.ai account, add credit, and add **the SSH public key of this machine**:
   `vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"` (or Account → SSH Keys).
   `bench.sh` compares your local keys with the account's keys and refuses to rent if none match.
2. Install the CLI and store your API key:
   ```bash
   pip install -r requirements-vast.txt
   vastai set api-key <YOUR_KEY>          # or put VAST_AI_API_KEY=... in vast/.env
   ```
3. Create your config. The defaults are sensible, so the file can stay empty:
   ```bash
   cp vast/.env.example vast/.env         # vast/.env is git-ignored. Secrets go here only.
   ```
   Optionally set `HF_KEY` (a Hugging Face token) in it for faster, unthrottled downloads.
4. **Push your commits.** The instance runs `git clone` on `REPO_URL` and checks out your local `HEAD`.
   `bench.sh` refuses to start if you're ahead of the upstream branch. The repo must be public, or `REPO_URL` must
   contain a token.

### Run

```bash
./vast/bench.sh ollama                  # cheapest 3090/4090 (on-demand), full benchmark
./vast/bench.sh ollama --quick          # ~2-minute smoke run: check the pipeline first
./vast/bench.sh vllm                    # FP8 on a 48 GB Ada/Hopper card, on-demand
VLLM_PRESET=bf16 ./vast/bench.sh vllm   # BF16 reference on 80 GB
./vast/bench.sh ollama --preset bielik-1.5b   # smallest Bielik (1.5B, Q8_0) for comparison
```

Flags: `--interruptible` bids for a cheaper machine that vast may stop at any time (see Troubleshooting); `--on-demand` is the default. `--quick` runs the smoke-size
benchmark. `--yes` skips the "Rent offer …?" prompt.

What happens:

1. **Rank.** The script searches offers that pass the filters (GPU, CUDA, reliability, download speed, disk,
   bandwidth price). It prints the top 10 **by estimated total cost of the whole run**, not by hourly price
   (formula below), and picks the cheapest.
2. **Guard.** If that estimate exceeds `MAX_RUN_USD_OLLAMA` ($3) or `MAX_RUN_USD_VLLM` ($6), it refuses to rent.
3. **Rent.** It creates the instance with a pinned image. All settings travel in a single env var, and the
   instance bootstraps itself with `vast/onstart.sh` and then `vast/setup_<mode>.sh`.
4. **Wait for READY.** It polls every 30 s and prints vast's status message whenever the status changes.
   The run stops immediately (fetch logs, destroy) when:
   - the setup script fails, for example because the model doesn't load. There is no automatic fallback.
   - the container **exits** on its own. That is a crash, not a preemption.
   - the machine is running but has no SSH for `SSH_WAIT_MIN` (10 min).
   - it is still loading after `LOAD_WAIT_MIN` (20 min).

   One case is retried instead of stopping: a **host that is too slow**. Setup measures the real
   download speed from Hugging Face before installing anything, then watches the model download.
   Below `MIN_NET_MBPS` (150) at the start, or `MIN_PULL_MBPS` (80) during the download, the host is
   destroyed and the next cheapest offer is rented instead (up to `HOST_RETRIES`=2 times), never the same
   machine again.
5. **Benchmark.** `vast/run_llm_bench.sh` runs on the instance under `nohup`, so a dropped SSH connection doesn't
   kill it. If an interruptible instance is preempted mid-run, the script waits up to `OUTBID_WAIT_MIN` (5 min) for
   it to resume, then reruns the benchmark once.
6. **Always:** it fetches logs and reports into `output_vast/<instance-id>/`, writes `run_summary.md`, and **destroys
   the instance**. Fetching is capped at `FETCH_TIMEOUT` (180 s) and never prompts, so it can't block the destroy.
   vast's container and daemon logs come from the API, so they arrive even when SSH never worked. It then verifies the instance is gone and prints a loud warning if it can't confirm that. This
   happens on success, on failure, on Ctrl-C, when `MAX_HOURS` (3 h) is reached, and after a preemption that doesn't
   resume.

### Results

`output_vast/<instance-id>/` contains:

| File | Content |
| :--- | :--- |
| `llm_speed_*.md/json`, `llm_load_*.md/json`, `llm_quality_*.md/json` | Benchmark reports. File names carry the GPU and mode, e.g. `RTX-4090-ollama-q4k` |
| `gpu_log_*.csv` | Per-second GPU utilization, VRAM, power and temperature |
| `run_summary.md/json` | Estimated vs. actual cost, phase durations, measured download speed |
| `container.log`, `daemon.log`, `instance_final.json` | vast's own view of the run (from the API): why a container died or never got SSH |
| `setup.log`, `onstart.log`, `bench.log`, `ollama.log`/`vllm.log`, `timings.log`, `local_events.log` | Everything needed to debug a failed run (only if SSH worked) |

Use `run_summary.md` to tune the cost model: it reports the measured `NET_EFFICIENCY` and phase durations. Put
better values for `NET_EFFICIENCY`, `SETUP_HOURS` and `BENCH_HOURS_*` into `vast/.env`.

### How "cheapest" is computed

```
estimated run cost = $/h × (download_GB × 8000 / (inet_down_Mbps × NET_EFFICIENCY) / 3600 + SETUP_HOURS + BENCH_HOURS)
                   + download_GB × inet_down_cost ($/GB)
```

The `$/h` figure depends on the billing type:

- **On-demand:** `dph_total`, searched with your real disk size (`--storage`), so storage is included.
- **Interruptible:** `min_bid × BID_MULTIPLIER` (1.25), plus storage.

A cheap host with a slow or expensive link can lose to a slightly pricier one that downloads 20–44 GB much faster.

### Configuration

Every setting lives in `vast/.env.example`, with its default and a comment. The most useful ones:

| Setting | Default | Effect |
| :--- | :--- | :--- |
| `REGION` | `europe` | Where the machine may be: `europe` (EU + UK, NO, CH, IS, LI, Western Balkans, MD, UA), `eu` (EU-27), `any`, or a list like `DE,PL,NL` |
| `GPU_NAME` / `VLLM_GPU_NAME` | – | Pin one card (e.g. `RTX_4090` / `L40S`) so speed numbers are comparable across runs |
| `INTERRUPTIBLE` | `0` | `1` = interruptible bid instead of on-demand |
| `MAX_HOURS` / `MAX_RUN_USD_*` | `3` / `$3`, `$6` | Hard limits |
| `OLLAMA_PRESET` | `q4k` | `bielik-1.5b` = speakleash Bielik-1.5B-v3.0-Instruct Q8_0 (CLI: `--preset bielik-1.5b`) |
| `VLLM_PRESET` | `fp8` | `bf16` = 80 GB reference run (CLI: `--preset bf16`) |
| `OLLAMA_NUM_PARALLEL` / `OLLAMA_CONTEXT_LENGTH` | `4` / `8192` | Ollama concurrency slots and context |
| `VLLM_MAX_MODEL_LEN` / `VLLM_EXTRA_ARGS` | `16384` / `--language-model-only --kv-cache-dtype fp8 --reasoning-parser qwen3` | vLLM server |
| `EXTRA_QUERY` | – | Extra vast filters, e.g. `geolocation in [DE,NL,PL]` |
| `OFFER_ID` | – | Force a specific offer (it must still pass the filters) |

A value exported in your shell overrides `vast/.env`, e.g. `MAX_HOURS=1 GPU_NAME=RTX_4090 ./vast/bench.sh ollama`.

### Step by step (debugging)

`bench.sh` is the supported path. The individual steps still work on their own:

```bash
./vast/launch.sh ollama --interruptible     # rank + rent only
./vast/ssh.sh 'tail -f /workspace/setup.log'
./vast/ssh.sh 'bash /workspace/embeddings-test/vast/run_llm_bench.sh'
./vast/fetch_results.sh                     # -> output_vast/<id>/
./vast/destroy.sh                           # STOP BILLING (asks; -y to skip)
```

If you rent this way, **nothing destroys the instance for you**.

### Manual runs against any server

The test scripts work against any Ollama server or OpenAI-compatible server (vLLM, llama.cpp `llama-server`,
SGLang):

```bash
pip install -r requirements-llm.txt
python scripts/llm_speed_test.py   --backend ollama --model <name> --n 30 --prefill-sizes 1024,4096,8192
python scripts/llm_load_test.py    --backend openai --base-url http://host:8000 --model <served-name> --concurrency 1,8,32
python scripts/llm_quality_test.py --n 200          # the dataset has 500 pairs
```

Shared flags: `--backend ollama|openai --model … --base-url … --think --num-ctx … --tag …`.

### Reading the numbers

- **TTFT** is prefill plus queueing. If TTFT rises at higher concurrency, the server is saturated.
- **Decode tok/s (single stream)** is what one user feels. The client-side value includes network jitter; Ollama also
  reports the exact server-side value.
- **Output tok/s at concurrency N** is serving capacity.
  - On Ollama it plateaus at `OLLAMA_NUM_PARALLEL`. The sweep deliberately goes to 2× that value to show queueing.
  - vLLM keeps scaling until the KV cache runs out.
- **chrF** is for comparing runs (quantization vs. BF16, backend vs. backend). It is not an absolute grade.
- Thinking mode is **off** by default, so token counts measure the answer rather than hidden reasoning. Add `--think`
  to benchmark reasoning mode.

### Troubleshooting

- **"none of your local SSH keys is in your vast account"** / **"has NO SSH keys"**: add the key it
  lists as local: `vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"`. The fingerprints printed match
  `ssh-keygen -lf ~/.ssh/id_ed25519.pub` and the `Failed publickey … SHA256:…` lines in `container.log`.
- **"Preempted: vast stopped the instance"**: only with `--interruptible`. Someone outbid you or rented
  the GPU on-demand. The run waits `OUTBID_WAIT_MIN`, then destroys. Rerun, or use the on-demand default.
- **`⚠️ vast/.env:N: …` warnings**: that line isn't a single `KEY=value` (explanatory text, or two values).
  Put `#` in front of it. Quote values that contain spaces.
- **"An instance is already tracked"**: a previous run didn't finish cleanup. `./vast/destroy.sh -y <id>` destroys
  it (if it still exists), verifies it's gone, and clears the local state.
- **"The container exited on its own"** (vast still intended it to run): read `output_vast/<id>/container.log` and `daemon.log`. Common causes are an
  image that isn't compatible with vast's SSH launch mode, or a host problem. Retry and the ranker picks the next host,
  or exclude the host with `EXTRA_QUERY='machine_id!=<id>'`.
- **"no SSH for 10 min"**: check that your key is registered with `vastai show ssh-keys`.
- **"ahead of its upstream"**: run `git push`. The instance can only run pushed code.
- **"Host … is too slow" / "Tried N hosts, all too slow"**: the machine's real link to Hugging Face was far
  below its advertised speed. It is replaced automatically. If every host in your `REGION` is slow, try
  another region or lower `MIN_NET_MBPS`.
- **"No usable offer"**: loosen the filters. Try `REGION=any` (or a wider list),  `MAX_INET_DOWN_COST`, `MIN_INET_DOWN_*`, `EXTRA_QUERY`, a different
  `GPU_NAME`, or `--on-demand`.
- **"Cheapest run is estimated at … > MAX_RUN_USD"**: raise the cap in `vast/.env` or relax the filters.
- **Ollama setup failed on the smoke test** (for example `unknown model architecture`, or a problem with the vision
  file `mmproj`): the pinned Ollama version couldn't load this GGUF. Retry with the Ollama-library build
  (`LLM_MODEL=huihui_ai/Qwen3.8-abliterated ./vast/bench.sh ollama`) or a newer Ollama (`OLLAMA_VERSION=<newer> …`).
- **vLLM fails to load the FP8 checkpoint**: the third-party checkpoint has only been validated on another vLLM fork.
  Check `vllm.log`, try a newer `VLLM_IMAGE`, or use `VLLM_PRESET=bf16`.
- **vLLM out of memory**: lower `VLLM_MAX_MODEL_LEN` or `VLLM_GPU_MEM_UTIL`.
- **Ollama slow or partly offloaded to CPU**: `ollama.log` shows the layer split. Lower `OLLAMA_NUM_PARALLEL` or
  `OLLAMA_CONTEXT_LENGTH`, or set `OLLAMA_KV_CACHE_TYPE=q8_0`.
- **Debugging any vast call**: `VAST_DEBUG=1 ./vast/bench.sh …` (or `destroy.sh`, `fetch_results.sh`) logs every
  `vastai` call with its exit code and output to `vast/.debug.log`. The env blob holding `HF_TOKEN` is redacted.
- **`destroy.sh` says "still listed"**: each check prints what vast answered for both `show instance` and
  `show instances`. `API error 401` means the script has no valid API key. It reads `VAST_AI_API_KEY` from
  `vast/.env`, or the key saved by `vastai set api-key`.
- **"COULD NOT CONFIRM DESTROY"** or **"Destroy did not complete"**: check `vastai show instances` or the web console right away. Storage bills even
  while an instance is stopped.
