"""Shared helpers for embedding tests with local Ollama."""
import json
import math
import os
import sys
import time

import ollama

MODELS = [
    {"name": "snowflake-arctic-embed2", "size_approx_mb": 1160},
    {"name": "granite-embedding:278m", "size_approx_mb": 278},
    {"name": "paraphrase-multilingual", "size_approx_mb": 563},
]

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "samples.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

client = ollama.Client()


def load_samples():
    with open(DATA_FILE, encoding="utf-8") as f:
        pairs = json.load(f)["pairs"]
    en = [p["en"] for p in pairs]
    pl = [p["pl"] for p in pairs]
    return en, pl


def check_model_availability(model_name):
    """Check model presence with retry logic. Returns (ok, error_msg)."""
    for attempt in range(MAX_RETRIES):
        try:
            available = [m.model for m in client.list().get("models", [])]
            # match with or without ':latest' tag
            if model_name in available or f"{model_name}:latest" in available:
                return True, ""
            if available:
                return False, (
                    f"Model '{model_name}' not found. "
                    f"Run: ollama pull {model_name}\n"
                    f"Available: {', '.join(available)}"
                )
            raise ConnectionError("Ollama returned an empty model list.")
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"[{model_name}] attempt {attempt + 1}/{MAX_RETRIES} failed, retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                return False, f"Failed after {MAX_RETRIES} attempts: {e}"
    return False, "Model could not be verified."


def embed_texts(model_name, texts):
    """Embed a list of texts. Returns (vectors, duration_seconds)."""
    vectors = []
    start = time.time()
    for text in texts:
        response = client.embeddings(model=model_name, prompt=text)
        if "embedding" not in response:
            raise RuntimeError(f"Missing 'embedding' key for '{text[:30]}...'")
        vectors.append(response["embedding"])
    return vectors, time.time() - start


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def die(msg):
    print(f"\n❌ {msg}", file=sys.stderr)
    sys.exit(1)
