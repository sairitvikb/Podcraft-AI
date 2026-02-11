import torch
from typing import Dict, List, Optional
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import gc

logger = logging.getLogger(__name__)


class LLMModelManager:
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.tokenizers = {}
        self.current_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, model_name: str, model_id: str = "default",
                  quantization: Optional[str] = None) -> None:
        logger.info(f"Loading model: {model_name} as {model_id}")
        
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
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.device.type == "cuda" else None,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        )
        
        self.models[model_id] = model
        self.tokenizers[model_id] = tokenizer
        
        if self.current_model is None:
            self.current_model = model_id
        
        logger.info(f"Model {model_id} loaded successfully")
    
    def switch_model(self, model_id: str) -> None:
        if model_id in self.models:
            self.current_model = model_id
            logger.info(f"Switched to model: {model_id}")
        else:
            logger.warning(f"Model {model_id} not found")
    
    def unload_model(self, model_id: str) -> None:
        if model_id in self.models:
            del self.models[model_id]
            del self.tokenizers[model_id]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if self.current_model == model_id:
                self.current_model = list(self.models.keys())[0] if self.models else None
            
            logger.info(f"Unloaded model: {model_id}")
    
    def get_model(self, model_id: Optional[str] = None):
        model_id = model_id or self.current_model
        return self.models.get(model_id)
    
    def get_tokenizer(self, model_id: Optional[str] = None):
        model_id = model_id or self.current_model
        return self.tokenizers.get(model_id)


class ModelEnsemble:
    def __init__(self, models: List, tokenizers: List, fusion_method: str = "weighted"):
        self.models = models
        self.tokenizers = tokenizers
        self.fusion_method = fusion_method
        self.weights = [1.0 / len(models)] * len(models)
    
    def generate_ensemble(self, prompt: str, max_length: int = 1000,
                         temperature: float = 0.7) -> str:
        all_outputs = []
        
        for model, tokenizer in zip(self.models, self.tokenizers):
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            all_outputs.append(text[len(prompt):].strip())
        
        if self.fusion_method == "weighted":
            return self._weighted_fusion(all_outputs)
        elif self.fusion_method == "majority_vote":
            return self._majority_vote_fusion(all_outputs)
        else:
            return all_outputs[0]
    
    def _weighted_fusion(self, outputs: List[str]) -> str:
        sentences = []
        for output in outputs:
            sentences.extend(output.split('.'))
        
        return '. '.join(sentences[:len(sentences)//2])
    
    def _majority_vote_fusion(self, outputs: List[str]) -> str:
        return outputs[0]
