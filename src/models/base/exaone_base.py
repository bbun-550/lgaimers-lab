import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Optional

class ExaoneBase:
    def __init__(self, model_name: str = "LGAI-EXAONE/EXAONE-4.0-1.2B", device: str = "auto"):
        """
        Initialize EXAONE Base Model.
        
        Args:
            model_name (str): Hugging Face model identifier
            device (str): Device to load model on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Handle MPS device for macOS
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        print(f"Loading model on device: {device}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        # Ensure model is in eval mode by default for generation
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        """
        Generate text based on code snippet provided.
        """
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        # Move inputs to the same device as model
        if hasattr(self.model, "device"):
            input_ids = input_ids.to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            
        # Decode output
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Test the implementation
    exaone = ExaoneBase()
    
    test_prompts = [
        "Explain how wonderful you are",
        "Explica lo increíble que eres",
        "너가 얼마나 대단한지 설명해 봐"
    ]
    
    for p in test_prompts:
        print(f"User: {p}")
        response = exaone.generate(p)
        print(f"EXAONE: {response}\n")
