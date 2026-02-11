import torch
from typing import List, Dict, Optional
import asyncio
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import numpy as np

logger = logging.getLogger(__name__)


class QuantizedInference:
    def __init__(self, model_name: str, quantization: str = "4bit"):
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
    
    def generate(self, prompt: str, max_length: int = 1000, temperature: float = 0.7) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class SpeculativeDecoding:
    def __init__(self, target_model, draft_model, num_draft_tokens: int = 4):
        self.target_model = target_model
        self.draft_model = draft_model
        self.num_draft_tokens = num_draft_tokens
    
    def generate(self, prompt: str, max_length: int = 1000) -> str:
        inputs = self.target_model.tokenizer(prompt, return_tensors="pt")
        
        generated = inputs['input_ids'].clone()
        
        while generated.size(1) < max_length:
            draft_outputs = self.draft_model.model.generate(
                generated,
                max_new_tokens=self.num_draft_tokens,
                do_sample=False
            )
            
            draft_tokens = draft_outputs[0, generated.size(1):]
            
            accepted = []
            for i, token in enumerate(draft_tokens):
                target_logits = self.target_model.model(generated).logits
                target_probs = torch.softmax(target_logits[0, -1], dim=0)
                draft_probs = torch.softmax(self.draft_model.model(generated).logits[0, -1], dim=0)
                
                acceptance_prob = min(1.0, target_probs[token] / (draft_probs[token] + 1e-8))
                
                if torch.rand(1) < acceptance_prob:
                    accepted.append(token.item())
                    generated = torch.cat([generated, token.unsqueeze(0).unsqueeze(0)], dim=1)
                else:
                    break
            
            if len(accepted) < len(draft_tokens):
                new_token = self.target_model.model.generate(
                    generated,
                    max_new_tokens=1,
                    do_sample=True
                )[0, -1:]
                generated = torch.cat([generated, new_token.unsqueeze(0)], dim=1)
        
        return self.target_model.tokenizer.decode(generated[0], skip_special_tokens=True)


class FlashAttentionInference:
    def __init__(self, model_name: str):
        try:
            from flash_attn import flash_attn_func
            self.use_flash = True
        except:
            self.use_flash = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def generate(self, prompt: str, max_length: int = 1000) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                use_cache=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
