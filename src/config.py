# src/config.py

import os
from dotenv import load_dotenv

# Resolve project root (the "freefire-vlm-analyzer" folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env if present (mainly on the VM)
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)


def get_paths() -> dict:
    """
    Returns important folder paths as an absolute-path dictionary.
    Works both on Windows and Linux.
    """
    data_dir = os.path.join(BASE_DIR, "data")

    return {
        "base_dir": BASE_DIR,
        "data_dir": data_dir,
        "videos_raw": os.path.join(data_dir, "videos_raw"),
        "frames": os.path.join(data_dir, "frames"),
        "metadata": os.path.join(data_dir, "metadata"),
        "outputs_raw": os.path.join(data_dir, "outputs", "raw_model"),
        "outputs_structured": os.path.join(data_dir, "outputs", "structured"),
    }


def get_model_endpoint() -> dict:
    """
    Reads model / server info from environment variables (or defaults).

    On the VM, we will create a .env file with:
      VLLM_BASE_URL
      VLLM_API_KEY
      VLLM_MODEL_NAME
    """
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("VLLM_API_KEY", "EMPTY")
    model_name = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct")

    return {
        "base_url": base_url,
        "api_key": api_key,
        "model_name": model_name,
    }
