import torch
from typing import List, Dict, Optional
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import time

logger = logging.getLogger(__name__)


class InferenceOptimizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def optimize_with_torch_compile(self):
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode='reduce-overhead')
            logger.info("Model compiled with torch.compile")
    
    def optimize_with_kv_cache(self):
        if hasattr(self.model, 'generate'):
            self.use_cache = True
            logger.info("KV cache enabled")
    
    def batch_generate(self, prompts: List[str], batch_size: int = 4, max_length: int = 1000) -> List[str]:
        all_outputs = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=self.use_cache if hasattr(self, 'use_cache') else True
                )
            
            for j, output in enumerate(outputs):
                generated = self.tokenizer.decode(
                    output[len(inputs['input_ids'][j]):],
                    skip_special_tokens=True
                )
                all_outputs.append(generated)
        
        return all_outputs
    
    def measure_latency(self, prompt: str, num_runs: int = 10) -> Dict:
        latencies = []
        
        for _ in range(num_runs):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=100,
                    do_sample=False
                )
            end_time = time.time()
            
            latencies.append(end_time - start_time)
        
        return {
            'mean_latency': np.mean(latencies),
            'std_latency': np.std(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies)
        }
