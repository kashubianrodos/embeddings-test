"""Shared helpers for generative-LLM performance tests.

Two backends are supported:

- ``ollama`` : native Ollama ``/api/chat`` (also returns server-side token timings)
- ``openai`` : any OpenAI-compatible server (vLLM, llama.cpp ``llama-server``, SGLang)

Only ``requests`` is required (see requirements-llm.txt).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_FILE = os.path.join(ROOT, "data", "samples.json")
OUTPUT_DIR = os.path.join(ROOT, "output")

DEFAULT_OLLAMA_MODEL = "hf.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF:Huihui-Qwen3.8-27B-abliterated-Q4_K.gguf"
DEFAULT_OPENAI_MODEL = "huihui-ai/Huihui-Qwen3.8-27B-abliterated"
DEFAULT_URLS = {"ollama": "http://localhost:11434", "openai": "http://localhost:8000"}

MAX_RETRIES = 30
RETRY_DELAY = 5  # seconds (the server may still be loading a 27B model)


# --------------------------------------------------------------------------- data
def load_samples():
    with open(DATA_FILE, encoding="utf-8") as f:
        pairs = json.load(f)["pairs"]
    return [p["en"] for p in pairs], [p["pl"] for p in pairs]


PROMPTS = {
    "explain": 'Write a short paragraph (about 150 words) expanding on this statement: "{en}"',
    "translate_en_pl": "Translate the following English sentence into Polish. "
                       "Reply with the translation only, no comments.\n\n{en}",
    "translate_pl_en": "Translate the following Polish sentence into English. "
                       "Reply with the translation only, no comments.\n\n{pl}",
}


def build_prompt(task, en, pl):
    return PROMPTS[task].format(en=en, pl=pl)


# --------------------------------------------------------------------------- results
@dataclass
class GenResult:
    ok: bool
    text: str = ""
    ttft_s: float | None = None          # time to first token (content or reasoning)
    total_s: float = 0.0                 # wall-clock request latency
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_approx: bool = False          # True if token count came from chunk counting
    server_prefill_tps: float | None = None  # Ollama only
    server_decode_tps: float | None = None   # Ollama only
    error: str = ""

    @property
    def decode_tps(self):
        """Client-side decode speed: tokens after the first / time after the first."""
        n, t = self.completion_tokens, self.ttft_s
        if n and n > 1 and t is not None and self.total_s > t:
            return (n - 1) / (self.total_s - t)
        return None

    def to_dict(self):
        d = asdict(self)
        d["decode_tps"] = self.decode_tps
        return d


THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_thinking(text):
    text = THINK_RE.sub("", text)
    if "</think>" in text:  # opening tag was in the template, not in the output
        text = text.split("</think>", 1)[1]
    return text.strip()


# --------------------------------------------------------------------------- client
class LLMClient:
    def __init__(self, backend, model, base_url=None, think=False, timeout=900,
                 num_ctx=None, api_key=None):
        if backend not in DEFAULT_URLS:
            raise ValueError(f"Unknown backend '{backend}'")
        self.backend = backend
        self.model = model
        self.base_url = (base_url or DEFAULT_URLS[backend]).rstrip("/")
        self.think = think
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._local = threading.local()
        self._send_think = True  # disabled automatically if Ollama rejects the field

    @property
    def session(self):
        if not hasattr(self._local, "s"):
            self._local.s = requests.Session()
        return self._local.s

    # ---- health ------------------------------------------------------------
    def check(self):
        """Wait for the server and confirm the model is served. Returns (ok, msg)."""
        last = ""
        for attempt in range(MAX_RETRIES):
            try:
                if self.backend == "ollama":
                    r = self.session.get(f"{self.base_url}/api/tags", timeout=10)
                    r.raise_for_status()
                    names = [m.get("model") or m.get("name") for m in r.json().get("models", [])]
                    if self.model in names or f"{self.model}:latest" in names:
                        return True, ""
                    return False, (f"Model '{self.model}' not pulled. Run: ollama pull {self.model}\n"
                                   f"Available: {', '.join(names) or '(none)'}")
                r = self.session.get(f"{self.base_url}/v1/models", headers=self._headers(), timeout=10)
                r.raise_for_status()
                names = [m["id"] for m in r.json().get("data", [])]
                if self.model in names:
                    return True, ""
                return False, f"Model '{self.model}' not served. Available: {', '.join(names)}"
            except Exception as e:  # server not up yet
                last = str(e)
                print(f"[server] attempt {attempt + 1}/{MAX_RETRIES} not ready ({last[:80]}), "
                      f"retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
        return False, f"Server at {self.base_url} not reachable: {last}"

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    # ---- generation --------------------------------------------------------
    def generate(self, prompt, max_tokens=256, temperature=0.0, seed=42):
        start = time.perf_counter()
        try:
            if self.backend == "ollama":
                res = self._gen_ollama(prompt, max_tokens, temperature, seed, start)
            else:
                res = self._gen_openai(prompt, max_tokens, temperature, seed, start)
        except Exception as e:
            return GenResult(ok=False, total_s=time.perf_counter() - start, error=str(e)[:500])
        res.total_s = time.perf_counter() - start
        return res

    def _gen_ollama(self, prompt, max_tokens, temperature, seed, start):
        options = {"num_predict": max_tokens, "temperature": temperature, "seed": seed}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        payload = {"model": self.model, "stream": True, "keep_alive": "1h", "options": options,
                   "messages": [{"role": "user", "content": prompt}]}
        if self._send_think:
            payload["think"] = self.think

        r = self.session.post(f"{self.base_url}/api/chat", json=payload, stream=True, timeout=self.timeout)
        if r.status_code >= 400 and "think" in r.text.lower() and self._send_think:
            self._send_think = False  # model/template has no thinking switch
            return self._gen_ollama(prompt, max_tokens, temperature, seed, start)
        r.raise_for_status()

        res, text = GenResult(ok=True), []
        for line in r.iter_lines():
            if not line:
                continue
            obj = json.loads(line)
            if "error" in obj:
                raise RuntimeError(obj["error"])
            msg = obj.get("message") or {}
            piece = msg.get("content") or ""
            if (piece or msg.get("thinking")) and res.ttft_s is None:
                res.ttft_s = time.perf_counter() - start
            text.append(piece)
            if obj.get("done"):
                res.prompt_tokens = obj.get("prompt_eval_count")
                res.completion_tokens = obj.get("eval_count")
                ped, ed = obj.get("prompt_eval_duration"), obj.get("eval_duration")
                if ped and res.prompt_tokens:
                    res.server_prefill_tps = res.prompt_tokens / (ped / 1e9)
                if ed and res.completion_tokens:
                    res.server_decode_tps = res.completion_tokens / (ed / 1e9)
        res.text = "".join(text)
        return res

    def _gen_openai(self, prompt, max_tokens, temperature, seed, start):
        payload = {"model": self.model, "stream": True, "max_tokens": max_tokens,
                   "temperature": temperature, "seed": seed,
                   "stream_options": {"include_usage": True},
                   "messages": [{"role": "user", "content": prompt}],
                   # Qwen3.x thinking switch (vLLM / SGLang / llama.cpp); ignored elsewhere
                   "chat_template_kwargs": {"enable_thinking": self.think}}
        r = self.session.post(f"{self.base_url}/v1/chat/completions", json=payload,
                              headers=self._headers(), stream=True, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")

        res, text, chunks = GenResult(ok=True), [], 0
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", "replace")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                res.prompt_tokens = obj["usage"].get("prompt_tokens")
                res.completion_tokens = obj["usage"].get("completion_tokens")
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if piece or reasoning:
                    chunks += 1
                    if res.ttft_s is None:
                        res.ttft_s = time.perf_counter() - start
                text.append(piece)
        if res.completion_tokens is None:
            res.completion_tokens, res.tokens_approx = chunks, True
        res.text = "".join(text)
        return res


# --------------------------------------------------------------------------- cli
def add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--backend", choices=["ollama", "openai"],
                   default=os.environ.get("LLM_BACKEND", "ollama"))
    p.add_argument("--model", default=os.environ.get("LLM_MODEL"),
                   help="Model name as the server knows it (defaults depend on backend)")
    p.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"))
    p.add_argument("--think", action="store_true", help="Enable Qwen3 thinking mode (off by default)")
    p.add_argument("--num-ctx", type=int, default=int(os.environ.get("LLM_NUM_CTX", "0")) or None,
                   help="Ollama context window (num_ctx)")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--tag", default=os.environ.get("LLM_TAG", ""),
                   help="Label added to report names, e.g. 'rtx4090-q4k'")
    return p


def client_from_args(args):
    model = args.model or (DEFAULT_OLLAMA_MODEL if args.backend == "ollama" else DEFAULT_OPENAI_MODEL)
    args.model = model
    return LLMClient(args.backend, model, args.base_url, think=args.think,
                     timeout=args.timeout, num_ctx=args.num_ctx)


# --------------------------------------------------------------------------- stats / reports
def percentile(values, p):
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"n": len(v), "mean": statistics.mean(v), "median": statistics.median(v),
            "stdev": statistics.stdev(v) if len(v) > 1 else 0.0,
            "p95": percentile(v, 95), "min": min(v), "max": max(v)}


def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return out.replace("\n", "; ") or "unknown"
    except Exception:
        return "unknown (nvidia-smi not available)"


def header_lines(title, args, extra=""):
    return [f"# {title}", "",
            f"- Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"- Backend: `{args.backend}` @ `{args.base_url or DEFAULT_URLS[args.backend]}`",
            f"- Model: `{args.model}`",
            f"- Thinking: {'on' if args.think else 'off'}",
            f"- GPU: {gpu_info()}",
            *([f"- {extra}"] if extra else []), ""]


def fmt(v, spec=".2f"):
    return "N/A" if v is None else format(v, spec)


def write_outputs(prefix, args, lines, raw):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    md = os.path.join(OUTPUT_DIR, f"{prefix}{tag}_{stamp}.md")
    js = os.path.join(OUTPUT_DIR, f"{prefix}{tag}_{stamp}.json")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(js, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "gpu": gpu_info(), **raw}, f, ensure_ascii=False, indent=1)
    print("\n" + "\n".join(lines))
    print(f"\n✅ Report: {md}\n✅ Raw data: {js}")


def die(msg):
    print(f"\n❌ {msg}", file=sys.stderr)
    sys.exit(1)
