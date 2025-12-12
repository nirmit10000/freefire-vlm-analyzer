"""
This file talks to the AI vision model (Qwen3-VL-2B-Instruct).
It loads the model and runs it on images.
"""

import os
from pathlib import Path
from PIL import Image

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class QwenLocalClient:
    """
    A simple wrapper to use the Qwen AI model.

    What it does:
    1. Load the AI model from Hugging Face
    2. Send images to the model with prompts
    3. Get back text responses (hopefully JSON!)
    """

    from typing import Optional

    def __init__(self, model_name="Qwen/Qwen3-VL-2B-Instruct", device: Optional[str] = None):

        """
        Initialize and load the model.

        Args:
            model_name: Which model to use (default is Qwen3-VL-2B)
            device: optional device string ("mps", "cpu", "cuda"); if None, autodetect
        """
        print("[MODEL] Loading Qwen3-VL-2B-Instruct...")

        # Determine device: explicit param -> env var -> autodetect
        if device is None:
            dev_env = os.environ.get("PREFERRED_DEVICE", None)
            if dev_env:
                device = dev_env
            else:
                if torch.cuda.is_available():
                    device = "cuda"
                elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

        self.device = torch.device(device)
        self.device_str = device  # keep the original string for messages

        # Print chosen device info
        if self.device_str == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"[MODEL] Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            except Exception:
                print(f"[MODEL] Using GPU: {gpu_name}")
        elif self.device_str == "mps":
            print("[MODEL] Using Apple MPS backend (Apple Silicon GPU)")
        else:
            print("[MODEL] Using CPU (will be slower)")

        # Load the processor (handles image + text formatting)
        print("[MODEL] Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,  # Qwen requires this
        )

        # Load the actual AI model
        print("[MODEL] Loading model (takes 1-2 min first time)...")

        # dtype: use float16 on CUDA, float32 otherwise (MPS uses float32)
        dtype = torch.float16 if self.device_str == "cuda" else torch.float32

        # For CUDA we can use device_map="auto" to shard; for CPU/MPS load normally then move
        if self.device_str == "cuda":
            # let HF distribute the model across available CUDA devices if possible
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # load into CPU first then move to the selected device (MPS or CPU)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=None,
                trust_remote_code=True,
            )
            try:
                # move model to chosen device (works for 'mps' and 'cpu')
                self.model = self.model.to(self.device)
            except Exception:
                # if move fails, keep model as-is and hope HF handles device internally
                print("[MODEL] Warning: failed to move model to device automatically.")

        # Set to evaluation mode (not training)
        self.model.eval()

        print(f"[MODEL] ✓ Model loaded successfully on {self.device_str}")
        self.model_name = model_name

    def run_inference(self, image_path, prompt, max_tokens=512, temperature=0.1):
        """
        Analyze one image with the AI model.

        Args:
            image_path: Path to image file (e.g., "data/frames/video/frame_00000.jpg")
            prompt: What to ask the AI (from prompts.py)
            max_tokens: Maximum response length (512 is good for JSON)
            temperature: Randomness (0.1 = very predictable, 1.0 = creative)

        Returns:
            str: AI's response (should be JSON text)
        """
        # Validate image exists
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Create conversation messages (like a chat)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a game analyst. Respond ONLY with valid JSON. No markdown.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]

        # Convert messages to model format (chat text)
        chat_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Prepare inputs (text + image) for model as PyTorch tensors
        inputs = self.processor(
            text=[chat_text],
            images=[image],
            return_tensors="pt",
        )

        # --- CRITICAL: move all tensors in `inputs` to the same device as the model ---
        # The processor may return a ProcessorOutput (dict-like). Move every tensor to self.device.
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                try:
                    inputs[k] = v.to(self.device)
                except Exception:
                    # fallback: leave as-is if move fails for some unexpected reason
                    inputs[k] = v
        # ------------------------------------------------------------------------------

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
            )

        # Move outputs to CPU for decoding (tokenizer expects CPU tensors or numpy)
        if isinstance(generated_ids, torch.Tensor):
            generated_ids_cpu = generated_ids.cpu()
        else:
            # If HF returns a different structure, try to handle common cases:
            try:
                # if it's a list/tuple of tensors
                if isinstance(generated_ids, (list, tuple)) and len(generated_ids) > 0 and isinstance(generated_ids[0], torch.Tensor):
                    generated_ids_cpu = type(generated_ids)([g.cpu() if isinstance(g, torch.Tensor) else g for g in generated_ids])
                else:
                    generated_ids_cpu = generated_ids
            except Exception:
                generated_ids_cpu = generated_ids

        # Convert model output back to text
        output_texts = self.processor.batch_decode(
            generated_ids_cpu,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Return first output (trim whitespace)
        output = output_texts[0].strip() if output_texts else ""
        return output

    def cleanup(self):
        """
        Free GPU memory after we're done.
        Call this when finished processing all images.
        """
        if hasattr(self, "model"):
            try:
                del self.model
            except Exception:
                pass
        if hasattr(self, "processor"):
            try:
                del self.processor
            except Exception:
                pass

        # Clear caches where available
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                # torch.mps.empty_cache may not exist on older builds; guard it
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
        except Exception:
            pass

        print("[MODEL] ✓ Cleaned up, GPU memory freed")
