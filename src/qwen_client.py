# src/qwen_client.py

from typing import Optional
from openai import OpenAI
from .config import get_model_endpoint


def get_client() -> tuple[OpenAI, str]:
    """
    Creates an OpenAI-compatible client that talks to the vLLM server.

    The vLLM server will be started on the VM and expose an OpenAI-style
    HTTP API. The base URL, API key, and model name are read from env vars
    via get_model_endpoint().
    """
    cfg = get_model_endpoint()

    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
    )

    return client, cfg["model_name"]


def analyze_frame(
    image_path: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """
    Sends a single image frame + prompt to the Qwen3-VL-2B-Instruct model
    (running behind vLLM's OpenAI-compatible server).

    Parameters
    ----------
    image_path : str
        Absolute or relative path to a local image file (e.g., .jpg).
    user_prompt : str
        The user-facing prompt describing what analysis we want.
    system_prompt : Optional[str]
        Optional system-level instructions (e.g., style, JSON-only output).
    temperature : float
        Sampling temperature for the model (0.0–1.0).
    max_tokens : int
        Maximum number of tokens to generate.

    Returns
    -------
    str
        The content string returned by the model (expected to be JSON, based
        on how we design the prompts).
    """
    client, model_name = get_client()

    messages = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    # For vision models, we send a message that includes both text and image.
    # vLLM's OpenAI-compatible API expects "input_image" with an "image_url".
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "input_image",
                    "image_url": {
                        # vLLM will load the image file from disk when given as file://
                        "url": f"file://{image_path}"
                    },
                },
            ],
        }
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # We expect a single choice with a single message
    return response.choices[0].message.content or ""
