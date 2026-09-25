"""Model and GPU registry for the vast.ai benchmark CLI (vast/cli.py).

Add a model by appending to MODELS. Every field has a meaning in the cost model or the
offer search; sizes come from the Hugging Face repos (checked 2026-09-25).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Gpu:
    name: str          # vast gpu_name, spaces written as "_" (the CLI query converts them)
    vram_gb: int
    compute_cap: int   # CUDA compute capability × 100 (Ada = 890)


# Cards that show up on vast.ai often enough to be worth offering. VRAM is the common variant.
GPUS: Dict[str, Gpu] = {g.name: g for g in [
    Gpu("RTX_3060", 12, 860), Gpu("RTX_3080", 10, 860), Gpu("RTX_3090", 24, 860),
    Gpu("RTX_4070", 12, 890), Gpu("RTX_4080", 16, 890), Gpu("RTX_4090", 24, 890),
    Gpu("RTX_5090", 32, 1200),
    Gpu("RTX_A4000", 16, 860), Gpu("RTX_A5000", 24, 860), Gpu("RTX_A6000", 48, 860),
    Gpu("L4", 24, 890), Gpu("L40", 48, 890), Gpu("L40S", 48, 890), Gpu("RTX_6000Ada", 48, 890),
    Gpu("A100_PCIE", 80, 800), Gpu("A100_SXM4", 80, 800),
    Gpu("H100_PCIE", 80, 900), Gpu("H100_SXM", 80, 900),
]}


@dataclass(frozen=True)
class Model:
    key: str                    # CLI name, also the report tag (ollama-<key> / vllm-<key>)
    title: str
    engine: str                 # "ollama" or "vllm"
    ref: str                    # ollama: hf.co/<repo>:<file>.gguf   vllm: <repo>
    weights_gb: float
    min_vram_gb: int            # weights + KV cache for the default slots/context, with headroom
    disk_gb: int                # default disk
    download_gb: float          # weights + image + engine, for the cost estimate
    bench_hours: float          # estimate for the full benchmark
    default_gpus: List[str] = field(default_factory=list)  # empty = any compatible card
    min_compute: int = 0        # e.g. 890 for native FP8
    vllm_preset: str = ""       # vllm only: fp8 / bf16 (image flags, revision)
    revision: str = ""
    note: str = ""

    @property
    def mode(self) -> str:
        return self.engine

    def compatible_gpus(self) -> List[Gpu]:
        return [g for g in GPUS.values()
                if g.vram_gb >= self.min_vram_gb and g.compute_cap >= self.min_compute]

    def base_filter(self) -> str:
        """Search filter when no specific card is chosen."""
        f = f"gpu_ram>={max(self.min_vram_gb - 1, 1)}"
        if self.min_compute:
            f += f" compute_cap>={self.min_compute}"
        return f


MODELS: List[Model] = [
    Model("qwen3.8-27b-q4k", "Huihui-Qwen3.8-27B-abliterated · GGUF Q4_K", "ollama",
          "hf.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF:Huihui-Qwen3.8-27B-abliterated-Q4_K.gguf",
          weights_gb=16.8, min_vram_gb=24, disk_gb=40, download_gb=22, bench_hours=0.75,
          default_gpus=["RTX_3090", "RTX_4090"], note="default"),
    Model("bielik-11b-q4km", "Bielik-11B-v3.0-Instruct · GGUF Q4_K_M", "ollama",
          "hf.co/speakleash/Bielik-11B-v3.0-Instruct-GGUF:Bielik-11B-v3.0-Instruct.Q4_K_M.gguf",
          weights_gb=6.7, min_vram_gb=16, disk_gb=30, download_gb=11, bench_hours=0.6,
          note="largest Polish model; KV for 4×8k slots needs ~7 GB more"),
    Model("bielik-4.5b-q8", "Bielik-4.5B-v3.0-Instruct · GGUF Q8_0", "ollama",
          "hf.co/speakleash/Bielik-4.5B-v3.0-Instruct-GGUF:Bielik-4.5B-v3.0-Instruct.Q8_0.gguf",
          weights_gb=5.1, min_vram_gb=12, disk_gb=25, download_gb=9, bench_hours=0.5),
    Model("bielik-1.5b-q8", "Bielik-1.5B-v3.0-Instruct · GGUF Q8_0", "ollama",
          "hf.co/speakleash/Bielik-1.5B-v3.0-Instruct-GGUF:Bielik-1.5B-v3.0-Instruct.Q8_0.gguf",
          weights_gb=1.7, min_vram_gb=10, disk_gb=25, download_gb=5, bench_hours=0.4,
          note="smallest Bielik"),
    Model("qwen3.8-27b-fp8", "Huihui-Qwen3.8-27B Mixed-FP8 (leoncca) · vLLM", "vllm",
          "leoncca/Qwen3.8-27B-Huihui-Mixed-FP8",
          weights_gb=27.8, min_vram_gb=46, disk_gb=90, download_gb=44, bench_hours=1.5,
          min_compute=890, vllm_preset="fp8", revision="3ea006fab18c94e486f446a874c15e25a02a0bd1",
          note="serving capacity; native FP8 needs Ada/Hopper"),
    Model("qwen3.8-27b-bf16", "Huihui-Qwen3.8-27B-abliterated BF16 · vLLM", "vllm",
          "huihui-ai/Huihui-Qwen3.8-27B-abliterated",
          weights_gb=55.6, min_vram_gb=80, disk_gb=130, download_gb=72, bench_hours=1.5,
          min_compute=800, vllm_preset="bf16", revision="main", note="full-precision reference"),
]
BY_KEY: Dict[str, Model] = {m.key: m for m in MODELS}
DEFAULT_MODEL = MODELS[0].key


def find_model(name: str) -> Optional[Model]:
    """Exact key, list number (1-based) or a unique prefix/substring."""
    name = name.strip().lower()
    if name.isdigit() and 1 <= int(name) <= len(MODELS):
        return MODELS[int(name) - 1]
    if name in BY_KEY:
        return BY_KEY[name]
    hits = [m for m in MODELS if m.key.startswith(name)] or [m for m in MODELS if name in m.key]
    return hits[0] if len(hits) == 1 else None


def normalize_gpu(name: str) -> str:
    """'4090', 'rtx 4090', 'RTX_4090' → 'RTX_4090' when known; otherwise the input with '_'."""
    n = name.strip().upper().replace(" ", "_").replace("-", "_")
    if n in GPUS:
        return n
    for key in GPUS:
        if key.endswith("_" + n) or key.replace("_", "") == n.replace("_", ""):
            return key
    return n
