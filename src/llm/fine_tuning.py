import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from typing import Dict, List, Optional
import logging
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

logger = logging.getLogger(__name__)


class LLMFineTuner:
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 use_lora: bool = True,
                 use_4bit: bool = True):
        self.model_name = model_name
        self.use_lora = use_lora
        self.use_4bit = use_4bit
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if use_lora:
            self.model = prepare_model_for_kbit_training(self.model)
            
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(self.model, lora_config)
    
    def prepare_dataset(self, texts: List[str], max_length: int = 512):
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=max_length,
                padding='max_length'
            )
        
        from datasets import Dataset
        dataset = Dataset.from_dict({'text': texts})
        tokenized = dataset.map(tokenize_function, batched=True)
        
        return tokenized
    
    def train(self, train_dataset, val_dataset, output_dir: str = "./results"):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        trainer.train()
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer
