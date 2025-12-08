"""
This file talks to the AI vision model (Qwen3-VL-2B-Instruct).
It loads the model and runs it on images.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class QwenLocalClient:
    """
    A simple wrapper to use the Qwen AI model.
    
    What it does:
    1. Load the AI model from Hugging Face
    2. Send images to the model with prompts
    3. Get back text responses (hopefully JSON!)
    """
    
    def __init__(self, model_name="Qwen/Qwen3-VL-2B-Instruct"):
        """
        Initialize and load the model.
        
        This happens when you create QwenLocalClient():
        - Downloads model if first time (~4GB download)
        - Loads model into GPU or CPU
        - Gets ready to analyze images
        
        Args:
            model_name: Which model to use (default is Qwen3-VL-2B)
        """
        print("[MODEL] Loading Qwen3-VL-2B-Instruct...")
        
        # Check if we have a GPU (much faster!) or CPU only (slower)
        if torch.cuda.is_available():
            self.device = "cuda"  # GPU
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[MODEL] Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            self.device = "cpu"  # CPU
            print("[MODEL] No GPU found. Using CPU (will be SLOW!)")
        
        # Load the processor (handles image + text formatting)
        print("[MODEL] Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,  # Qwen needs this
        )
        
        # Load the actual AI model
        print("[MODEL] Loading model (takes 1-2 min first time)...")
        
        # Use float16 on GPU to save memory, float32 on CPU
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        
        # Move model to device (GPU or CPU)
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        # Set to evaluation mode (not training)
        self.model.eval()
        
        print(f"[MODEL] ✓ Model loaded successfully on {self.device}")
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
        
        Example:
            client = QwenLocalClient()
            response = client.run_inference("frame.jpg", "Analyze this image")
            # response = '{"gameplay_summary": "Player in lobby..."}'
        """
        # Check image exists
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Create conversation messages (like a chat)
        messages = [
            {
                "role": "system",  # System message sets AI behavior
                "content": [
                    {
                        "type": "text",
                        "text": "You are a game analyst. Respond ONLY with valid JSON. No markdown.",
                    }
                ],
            },
            {
                "role": "user",  # User message has image + question
                "content": [
                    {
                        "type": "image",
                        "image": image,  # The actual image
                    },
                    {
                        "type": "text",
                        "text": prompt,  # What we're asking
                    },
                ],
            },
        ]
        
        # Convert messages to model format
        chat_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        # Prepare inputs (text + image) for model
        inputs = self.processor(
            text=[chat_text],
            images=[image],
            return_tensors="pt",  # PyTorch tensors
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():  # Don't compute gradients (we're not training)
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
            )
        
        # Convert model output back to text
        output_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        # Return first output
        output = output_texts[0].strip()
        return output
    
    
    def cleanup(self):
        """
        Free GPU memory after we're done.
        Call this when finished processing all images.
        """
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[MODEL] ✓ Cleaned up, GPU memory freed")