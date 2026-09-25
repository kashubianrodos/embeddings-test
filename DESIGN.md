# Design: generative LLM benchmark on vast.ai

The README covers how to run it. This document covers why it is built this way. The decisions come from a design
review on 2026-09-25. The facts they rely on were checked on that date; see "Verified facts and known risks".

## 1. Goals

The benchmark answers three questions about `huihui-ai/Huihui-Qwen3.8-27B-abliterated`:

| # | Question | How it is answered |
| :- | :--- | :--- |
| a | How does it feel for a single user on cheap hardware? | `ollama` run: single-stream TTFT and decode tok/s |
| b | How many users can one GPU serve? | `vllm` run: concurrency sweep with aggregate tok/s and p95 latency |
| c | Did quantization or abliteration hurt quality? | `ollama` run: EN↔PL chrF against the 500 reference pairs |

**Decision (Q1): two separate cheap runs instead of one big one.** Ollama with Q4_K on a 24 GB consumer card covers
(a) and (c). vLLM is rented only when (b) matters. One run on an expensive card would pay H100 prices for questions
a 3090 can answer.

**Optimisation target (Q2): lowest total cost per finished run.** This counts download time, setup and bandwidth, not
just the hourly price. Faster results and a lower hourly price are secondary.

## 2. Architecture

```
your machine (bash 3.2 + python3 stdlib + vastai CLI)             rented instance (root, Linux)
─────────────────────────────────────────────────                 ─────────────────────────────────────
vast/.env ──► common.sh: resolve_config(mode)                       onstart.sh (every container start)
                  │                                                   ├─ decode BENCH_ENV_B64 → /workspace/remote.env
bench.sh ──► launch.sh                                                ├─ git clone REPO_URL, checkout REPO_COMMIT
   │           ├─ vastai search offers (--storage, --type bid)        └─ nohup setup_<mode>.sh
   │           ├─ vast_tool.py rank  → cheapest *run*                        ├─ install/verify engine
   │           ├─ MAX_RUN_USD guard, confirm                                 ├─ download (timed) ─► timings.log
   │           └─ vastai create instance --onstart onstart.sh ─────────►     ├─ start server, smoke test
   │                  --env "-e BENCH_ENV_B64=…"                             └─ touch READY  (or SETUP_FAILED)
   ├─ poll: vastai show instance + ssh marker check
   ├─ READY → ssh: nohup run_llm_bench.sh ───────────────────────────►  run_llm_bench.sh
   │                                                                     ├─ BENCH_RUNNING (pid)
   ├─ BENCH_DONE / FAILED / timeout / preempted / Ctrl-C                 ├─ speed, load, quality tests + GPU log
   └─ EXIT trap (always):                                                └─ BENCH_DONE  (or BENCH_FAILED)
        fetch_results.sh → output_vast/<id>/ + run_summary.md
        destroy.sh -y → verify gone
```

**The local side** is bash 3.2 plus stdlib `python3`, so it runs on stock macOS. It uses no `timeout`, no associative
arrays and no GNU-only flags.

**The instance side** coordinates only through files in `/workspace`:

- `READY`, `SETUP_FAILED`, `BENCH_RUNNING`, `BENCH_DONE` and `BENCH_FAILED` are the state markers.
- `timings.log` records the phases.

There is no daemon and no open port apart from SSH. Each poll is one SSH call that prints the current state.

## 3. Decisions

### Machines and weights

| # | Decision | Why | Rejected |
| :- | :--- | :--- | :--- |
| Q5 | vLLM default is the **FP8** checkpoint `leoncca/Qwen3.8-27B-Huihui-Mixed-FP8` (28 GB) on a **48 GB Ada or Hopper** card. | FP8 on Ada or Hopper is the realistic way to serve this model in production, at about a quarter of the H100 price (L40S ≈ $0.48/h on-demand). | BF16 on 80 GB is kept as a reference preset (Q13), never the default. AWQ on a 4090 is too far from production to plan capacity with. |
| Q14 | The vLLM query requires `compute_cap>=890`, and the FP8 revision is pinned. | Without it the ranker would pick an Ampere A6000. Ampere has no FP8 compute, so vLLM falls back to a slower weight-only path and the numbers stop meaning anything. The third-party checkpoint can also change under you. | A plain `gpu_ram>=48` filter. |
| – | Ollama is allowed on `RTX_3090` or `RTX_4090`. `GPU_NAME` pins one card (Q15). | Quality (c) doesn't depend on the GPU. Speed (a) differs by 1.5–2×, so runs you want to compare should be pinned. Reports always carry the GPU in their name. | Any 24 GB card. Cards like A5000 and L4 are slow or rarely cheaper. |
| – | Ollama keeps `OLLAMA_NUM_PARALLEL=4` and a context of 8192 on 24 GB. | This model keeps a KV cache in only 16 of its 64 layers (the other 48 use linear attention). KV is about 64 KiB/token in BF16, so 4 slots × 8k needs about 2 GB on top of the 16.8 GB of weights. | Dropping to 2 slots. That isn't needed. |

### Model source and images

| # | Decision | Why | Rejected |
| :- | :--- | :--- | :--- |
| Q6 | Ollama pulls the Hugging Face GGUF with the **full filename** as the tag. | The quality comparison needs a known quantization. The repo contains Q4_K, Q4_K_L and UD-DW-Q4_K_M, so the short tag `:Q4_K` is ambiguous. | The Ollama library tag, whose quantization isn't obvious. It remains a documented manual retry. |
| Q12 | **Fail fast** when the model won't load: the smoke test fails, logs are fetched, and the instance is destroyed. | A silent fallback to another engine or model would produce numbers for something you didn't ask to test. | Automatic fallback to llama.cpp. |
| Q7 (revised after first real run) | Images are pinned. Ollama runs on `vastai/base-image:stock-ubuntu24.04-py312-2026-09-07` with Ollama **0.34.4** installed by the setup script (`OLLAMA_VERSION`). vLLM runs on `vastai/vllm:v0.29.0-cuda-12.9`. | The first run on `ollama/ollama:0.34.4` never got SSH and the container **exited** after about 2 minutes. vast's docs say SSH mode replaces the entrypoint and "requires that your docker image is compatible with typical ssh daemon setup". Both chosen images are vast's own SSH-mode images, and the Ollama version stays pinned. The install adds about 2 GB and roughly a minute. The CUDA 12.9 vLLM variant runs on more hosts (driver ≥575) than 13.0 (≥580). | `ollama/ollama` as the base image: it failed in SSH mode, though the exact cause couldn't be recovered because no logs were fetched. vast's "Open WebUI + Ollama" template: it puts auth in front of the API. `vllm/vllm-openai:latest`: may need CUDA 13. |
| – | Nothing the `vastai/vllm` image reads (`VLLM_MODEL`, `VLLM_ARGS`) is set. `setup_vllm.sh` starts `vllm serve` itself on port 8011. | The image's supervisor stays idle when `VLLM_MODEL` is unset. Owning the process gives us one log, one port with no vast proxy auth in front of it, and control over flags and revision. | Letting the template start vLLM. That adds an auth proxy and moves the log somewhere else. |
| – | The vLLM weights are downloaded as a separate step before `vllm serve`. | The run summary can then report real download throughput, which tunes the cost model. | Letting `vllm serve` download them, which mixes the download into load time. |

### Cost and billing

| # | Decision | Why | Rejected |
| :- | :--- | :--- | :--- |
| Q8 | Offers are ranked by **estimated total run cost** (section 4) by `vast/vast_tool.py rank`, called from `launch.sh`. | `-o dph` alone ranks by a price that counts only 5 GB of storage and ignores download time and bandwidth price. | Sorting by `dph`, which was the previous behaviour. |
| – | `--simulate gx10` / `./vastbench project`: run on a rentable card, then project onto the ASUS Ascent GX10 (GB10). Decode = max(weights ÷ GB10 achievable bandwidth, proxy compute time × compute ratio). Prefill scales by dense tensor throughput of the unit doing the matmuls: INT8 for GGUF, FP8/BF16 for vLLM. Specs live in `vast/models.py` (GPUs) and `vast/project.py` (target). | GB10 hosts on vast.ai are rare and arm64 (our images are amd64). A projection from any card costs nothing extra and works for past runs too. Checked with a real RTX 3090 run: projected 12.2 / 811 tok/s vs published GB10 11.6 / 838 for the same Qwen3.8-27B Q4_K (+5% / −3%). Pure spec-ratio bandwidth scaling would have said 5.9 tok/s, because the 3090 isn't bandwidth-bound on this hybrid model. | Renting a real GB10 (availability, arm64 images). Renting a bandwidth look-alike (L4, 300 GB/s): offered as a marked card choice, not forced, because it limits the offers. |
| – | `./vastbench` (Typer) is the front end. It holds the model and card registry (`vast/models.py`), asks for model, cards and disk with defaults, and validates them (VRAM, compute capability, disk ≥ download + 10 GB). The run itself is still `bench.sh`, driven through `RUN_*` env vars that win over presets and `vast/.env`. | One place to pick a model or cards, with the defaults visible. Wrong combinations fail before renting. The orchestration and cleanup logic that has been through real runs stays unchanged. Ctrl-C reaches `bench.sh`: the CLI ignores SIGINT and waits for the script's cleanup. | Rewriting the orchestration in Python: it would re-open every failure mode already fixed in bash. |
| – | `OLLAMA_PRESET=bielik-1.5b`: the smallest generative Bielik, [speakleash/Bielik-1.5B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-1.5B-v3.0-Instruct) (Apache-2.0, llama architecture, 8k context), as the Q8_0 GGUF (1.7 GB, repo commit `99d72cc`) on any ≥10 GB card. | A Polish-native baseline for the EN↔PL quality test and the speed numbers, at a few cents per run. Q8_0 is the smallest published quantization and is practically lossless. Bielik-Guard 0.1B/0.5B are classifiers, not chat models. | The 4.5B/7B/11B Bielik models: all bigger. fp16: twice the download for no measurable quality gain. |
| – | A host is judged by **measured** throughput, not the advertised one: a probe before setup (`MIN_NET_MBPS`=150) and a watchdog during the download (`MIN_PULL_MBPS`=80). A slow host is replaced automatically (`HOST_RETRIES`=2). | Run 4: a host advertising 1342 Mb/s delivered about 10 Mb/s from HF. Replacing it after 2 minutes costs cents; waiting cost an hour. | Trusting `inet_down`; failing the whole run on a slow host. |
| – | Offers are limited to Europe by default (`REGION=europe`: 2-letter codes from vast's `geolocation`; `eu`, `any` or an explicit list also work). | You asked for European hosts: European hosts mean lower latency to a European workstation for SSH and result fetches, and keep the machine in European jurisdictions. The ranker still picks the cheapest run within the region. | Worldwide search, the previous behaviour, which picked US and Asian hosts. |
| Q3 (revised after two real runs) | `--interruptible` / `--on-demand` flags, **on-demand by default for both modes**. Originally Ollama defaulted to interruptible. Both real runs were then stopped by vast within 2.5–4.5 min. The second time, the bid ($0.129) was still above the current `min_bid` ($0.093), so an on-demand renter probably took the GPU. For a job of about 1 h, on-demand costs about $0.05 more and isn't lost halfway. | Interruptible is about 60% cheaper. Preemption costs a re-download: minutes for Ollama's 17 GB, more for vLLM's 28–56 GB, where on-demand is safer. | Always on-demand, as before. |
| Q9 | The bid is `min_bid × 1.25` (`BID_MULTIPLIER`). | A bid just above the floor gets outbid within minutes; 1.25× buys some stability for little money. | A fixed dollar bid. |
| Q16 | Budget guards: `MAX_RUN_USD` (ollama $3, vllm $6) is checked **before** renting. `MAX_HOURS=3` is a hard wall-clock stop. | This caps the damage from a bad estimate or a hung setup. | No caps. |

### Run lifecycle, configuration and docs

| # | Decision | Why | Rejected |
| :- | :--- | :--- | :--- |
| Q10 | `vast/bench.sh` is the documented entry point and **always** fetches logs, then destroys the instance. | Forgetting `destroy.sh` is the most expensive failure. The trap is armed *before* renting, so even a failure inside `launch.sh` after the instance exists is cleaned up. | Manual steps as the default. They remain available for debugging. |
| – | If a preempted instance doesn't resume within `OUTBID_WAIT_MIN` (5 min; a GPU taken by an on-demand renter rarely comes back soon), fetch and destroy. After a resume, rerun the benchmark at most once. | Storage bills while an instance is stopped. Setup is idempotent: the model is already on disk, so a resume costs minutes. | Waiting forever, or never retrying. |
| Q4 | `vast/.env.example` is committed and `vast/.env` is git-ignored. **Every** setting is resolved in one place (`common.sh`) and forwarded to the instance. | Previously `OLLAMA_NUM_PARALLEL`, `VLLM_MAX_MODEL_LEN` and similar were documented as settings but never reached the instance. `launch.sh` only forwarded 3 variables. | Env vars spread across three scripts. |
| – | Settings are forwarded as **one** base64url blob, `-e BENCH_ENV_B64=…`, which onstart decodes to `/workspace/remote.env` (mode 600). | vast's `-e` parser has its own quoting rules. Values like `--language-model-only --kv-cache-dtype fp8` would need escaping. One opaque token can't be mangled. | Many `-e K=V` pairs. |
| – | The instance checks out your exact local `HEAD` (`REPO_COMMIT`). `bench.sh` refuses to start if the branch is ahead of its upstream. | Results must correspond to known code. Without the check, an unpushed change silently doesn't run. | Cloning `main`. |
| Q11 | `README.md` holds all instructions and `DESIGN.md` holds the reasoning. `LLM_BENCHMARK.md` was folded into the README and deleted. | Three overlapping docs drift apart. | Keeping `LLM_BENCHMARK.md`. |

## 4. Cost model

```
hours     = download_GB × 8000 / (inet_down_Mbps × NET_EFFICIENCY) / 3600 + SETUP_HOURS + BENCH_HOURS
price_h   = dph_total                        (on-demand; search uses --storage DISK_GB so storage is included)
          = min_bid × BID_MULTIPLIER + storage_h   (interruptible)
est_total = price_h × hours + download_GB × inet_down_cost
```

| Parameter | Ollama | vLLM FP8 | vLLM BF16 | Source |
| :--- | ---: | ---: | ---: | :--- |
| `DOWNLOAD_GB` (weights + image) | 20 | 44 | 72 | HF file sizes; image sizes estimated |
| `DISK_GB` | 40 | 90 | 130 | download + about 20–60 GB headroom |
| `BENCH_HOURS` | 0.75 | 1.5 | 1.5 | estimate; tune from `run_summary.md` |
| `SETUP_HOURS` | 0.1 | 0.1 | 0.1 | estimate |
| `NET_EFFICIENCY` | 0.5 | 0.5 | 0.5 | conservative guess; measured each run |
| Search filters | `inet_down>=500` | `inet_down>=1000` | `inet_down>=1000` | `inet_down_cost<=$0.02/GB`, `reliability>0.98` |

`storage_h` is `dph_total − dph_base` when the offer reports both. Otherwise it is `storage_cost × disk / 730`.

The estimate is deliberately simple and is **calibrated from real runs**. Every run writes the measured download
speed (and hence NET_EFFICIENCY), phase durations and actual cost to `run_summary.md`.

## 5. Failure handling

| Situation | Detected by | Result |
| :--- | :--- | :--- |
| Container exits while vast still intends it to run (`exited` + `intended_status=running`) | `show instance` | Show `status_msg` + `vastai logs`, fetch, destroy. **Not** treated as a preemption (the first real run lost 15 min waiting on this) |
| Running, but no SSH for `SSH_WAIT_MIN` | poll | Show vast logs, fetch, destroy (usually a missing SSH key) |
| Stuck `loading` for `LOAD_WAIT_MIN` | `show instance` | Show vast logs, fetch, destroy |
| No offer passes the filters | the ranker returns nothing | Refuse to rent; suggest which filters to relax |
| Estimate > `MAX_RUN_USD` | `launch.sh` | Refuse to rent |
| Clone or checkout fails (unpushed commit, private repo) | `onstart.sh` → `SETUP_FAILED` | Fetch logs, destroy |
| Model won't load (e.g. Ollama `qwen35` or mmproj issue, FP8 kernel) | smoke test → `SETUP_FAILED` | Fetch logs, destroy, print the retry hint |
| SSH connection drops | the benchmark runs under `nohup` | Polling continues |
| Outbid or preempted (`stopped`/`offline`, or `exited` with `intended_status=stopped`) | `show instance` | Wait `OUTBID_WAIT_MIN`, then resume (rerun the benchmark once) or fetch and destroy |
| Hang anywhere | `MAX_HOURS` | Fetch whatever exists, destroy |
| Ctrl-C | INT trap | Fetch, destroy |
| Destroy not confirmed | `show instance` still lists the instance after 3 attempts | Loud warning with the instance id |

**Cleanup can never block.** In the first real run, the log-fetch fallback (`vastai copy`, which uses rsync to the
host) prompted for a password and hung before the destroy, so the instance had to be stopped by hand. Now:

- that fallback is gone;
- SSH runs with `BatchMode=yes`;
- every cleanup step runs under a watchdog (`run_limited`, since bash 3.2 has no `timeout`);
- the whole fetch is capped at `FETCH_TIMEOUT`, and the destroy follows in any case;
- vast's container and daemon logs are fetched over the HTTPS API (`vastai logs`), which works without SSH.

`onstart.sh` runs as part of vast's SSH-mode entrypoint, so its top level no longer does anything that could exit,
exec or block. It writes `bootstrap.sh` and starts it detached. A failure there can only leave `SETUP_FAILED`; it
can't take the container or sshd down with it.

**Lessons from real runs 2–3:**

- `exited` alone doesn't mean a crash. vast reports a preempted interruptible instance as `exited` with
  `intended_status=stopped`, so the classification now reads both fields.
- An SSH key missing from the account wastes a paid run, so it is now a hard pre-rent check. It compares the
  key material of `~/.ssh/*.pub` and `ssh-add -L` with `vastai show ssh-keys` and prints fingerprints in the
  same `SHA256:` form sshd logs.
- `vast/.env` is parsed, not `eval`'d. Explanatory text or `a / b` lines produce a warning and are ignored,
  instead of breaking bash.
- `run_limited` must not use a bare `wait` under `set -e`: a killed or failed command would silently exit the
  caller.

**Lesson from real run 4** (machine 78080, Yunnan CN, before `REGION` existed): the host advertised
1342 Mb/s. It delivered about 35 Mb/s for the Docker image and the Ollama install, and 0.1–3 MB/s from
Hugging Face. `ollama pull` gave up after 62 min at 11% ("max retries exceeded: EOF") and the run cost $0.23
for nothing. The advertised `inet_down` can't be trusted, so setup now measures the real link first
(`net_probe`: 200 MB range request to the model file on HF), then guards the download (`download_watchdog`:
rate over 5-min windows while partial files exist). A slow host fails with the reason `slow_network`.
`bench.sh` then fetches its logs, destroys it, adds its `machine_id` to `EXCLUDE_MACHINES`, and rents the next
cheapest offer, up to `HOST_RETRIES` times inside the same `MAX_HOURS` budget. Progress bars are thinned to one
line every 30 s; `setup.log` had been 5 MB.

Two more checks now run before renting: `REPO_URL` must be readable anonymously (`git ls-remote`), and your account
must have an SSH key.

The CLI's exit codes are not trusted (`vastai destroy` exits 0 on API errors). Success is decided by re-reading
`show instance`. If that read fails, the status is `unknown`, never `gone`.

## 6. Security

- The vast API key stays on your machine, in `vast/.env` or `~/.config/vastai/vast_api_key`.
- `HF_TOKEN`/`HF_KEY` is forwarded inside the env blob. Anyone with access to your vast account can see it in the
  instance config. On the instance it lives in `/workspace/remote.env` (0600), which is **not** fetched back.
  Use a read-only token.
- SSH to instances skips host-key checking (`UserKnownHostsFile=/dev/null`). vast reuses IPs and ports, so a pinned
  known_hosts file would keep breaking. The hosts are throwaway, and nothing secret is sent over the connection.

## 7. Verified facts and known risks (as of 2026-09-25)

These were verified:

- **Model and checkpoints:**
  - The model is `qwen3_5` hybrid: 64 layers, 16 of them full attention (4 KV heads × 256), native context 262k.
  - The GGUF repo (commit `3f101cd`) contains `Huihui-Qwen3.8-27B-abliterated-Q4_K.gguf` (16.8 GB) and
    `mmproj-model-bf16.gguf`.
  - The FP8 repo (commit `3ea006f`) uses native FP8 e4m3 with 128×128 blocks. It keeps attention, vision and MTP in
    BF16, and its base model is the huihui BF16 model.
- **vastai CLI (≥1.8.1):**
  - `--type bid`, `--storage`, `--onstart FILE`, `--bid_price` (per machine), `destroy -y` and
    `compute_cap` (×100) all exist.
  - `gpu_name in [A,B]` needs spaces around `in`.
  - `inet_down` is in Mb/s and `inet_down_cost` in $/GB.
- **vastai/vllm image:** its venv is `/venv/main`, and vLLM only auto-starts when `VLLM_MODEL` is set.

Not yet confirmed; each first run checks these:

1. Why `ollama/ollama` exited in SSH mode on the first real run. The next failure of this kind will ship
   `container.log` and `daemon.log`.
2. Whether Ollama 0.34.4 loads this GGUF when the repo also has an mmproj file. Earlier versions failed (ollama#14730,
   fixed in the 0.30 line according to third-party reports).
3. Whether upstream vLLM 0.29 loads the leoncca FP8 checkpoint. Its card only mentions a V100 fork. The format is
   standard block FP8, so it probably works.
4. Whether `--language-model-only` is accepted by v0.29 for `qwen3_5`. It is documented in the Qwen3.5 recipe.
5. The exact meaning of `min_bid` and `dph_total` for bid offers. The cost math assumes `min_bid` is the machine's
   GPU price and adds storage on top. The measured cost in `run_summary.md` will show if this is off.
6. The vast.ai prices quoted in the README come from third-party trackers, not live listings.

## 8. Possible next steps

- Calibrate the `BENCH_HOURS_*` and `NET_EFFICIENCY` defaults after a few real runs.
- An `ollama` preset for a second quantization (e.g. `Q8_0` on 48 GB) to measure quantization loss directly.
- `--speculative-config` (MTP) for vLLM once the baseline numbers exist.
- Run both modes back-to-back from one command, if the two-step flow proves annoying.
