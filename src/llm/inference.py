import asyncio
from typing import List, Dict, Optional
import openai
from anthropic import Anthropic
import logging
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class OptimizedLLMInference:
    def __init__(self,
                 provider: str = "openai",
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 batch_size: int = 5,
                 max_concurrent: int = 3):
        self.provider = provider
        self.model = model
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        
        if provider == "openai":
            self.client = openai.OpenAI(api_key=api_key)
        elif provider == "anthropic":
            self.client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def generate_batch_async(self,
                                   prompts: List[str],
                                   optimized_params: Optional[Dict] = None) -> List[str]:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def generate_one(prompt: str) -> str:
            async with semaphore:
                return await self._generate_async(prompt, optimized_params)
        
        tasks = [generate_one(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def _generate_async(self,
                             prompt: str,
                             optimized_params: Optional[Dict] = None) -> str:
        if optimized_params is None:
            optimized_params = {}
        
        if self.provider == "openai":
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=optimized_params.get("max_tokens", 5000),
                temperature=optimized_params.get("temperature", 0.7),
                top_p=optimized_params.get("top_p", 0.9),
                frequency_penalty=optimized_params.get("frequency_penalty", 0.3),
                presence_penalty=optimized_params.get("presence_penalty", 0.3)
            )
            return response.choices[0].message.content
        else:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=optimized_params.get("max_tokens", 5000),
                temperature=optimized_params.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
    
    def generate_batch_sync(self,
                           prompts: List[str],
                           optimized_params: Optional[Dict] = None) -> List[str]:
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = [
                executor.submit(self._generate_sync, prompt, optimized_params)
                for prompt in prompts
            ]
            results = [future.result() for future in futures]
        
        elapsed = time.time() - start_time
        logger.info(f"Generated {len(prompts)} batches in {elapsed:.2f}s "
                   f"({elapsed/len(prompts):.2f}s per batch)")
        
        return results
    
    def _generate_sync(self,
                      prompt: str,
                      optimized_params: Optional[Dict] = None) -> str:
        if optimized_params is None:
            optimized_params = {}
        
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=optimized_params.get("max_tokens", 5000),
                temperature=optimized_params.get("temperature", 0.7),
                top_p=optimized_params.get("top_p", 0.9),
                frequency_penalty=optimized_params.get("frequency_penalty", 0.3),
                presence_penalty=optimized_params.get("presence_penalty", 0.3)
            )
            return response.choices[0].message.content
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=optimized_params.get("max_tokens", 5000),
                temperature=optimized_params.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
    
    def process_newsletter(self,
                          newsletter_content: str,
                          chunks: List[Dict],
                          prompt_engineer,
                          use_async: bool = True) -> str:
        prompts = prompt_engineer.create_batch_prompts(chunks, self.batch_size)
        optimized = [prompt_engineer.optimize_prompt(p) for p in prompts]
        
        start_time = time.time()
        
        if use_async:
            results = asyncio.run(self.generate_batch_async(
                prompts,
                optimized[0] if optimized else None
            ))
        else:
            results = self.generate_batch_sync(
                prompts,
                optimized[0] if optimized else None
            )
        
        elapsed = time.time() - start_time
        logger.info(f"Generated podcast script in {elapsed:.2f}s "
                   f"(~{len(' '.join(results))} words)")
        
        return "\n\n".join(results)
