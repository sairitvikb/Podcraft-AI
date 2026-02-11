from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PromptEngineer:
    def __init__(self):
        self.system_prompts = {
            "podcast_host": """You are an engaging podcast host with expertise in technology, business, and current events. 
Your style is conversational, insightful, and accessible. You make complex topics understandable and interesting.""",
            
            "newsletter_summarizer": """You are an expert at summarizing newsletters into engaging podcast scripts. 
You maintain the key insights while making the content conversational and podcast-friendly.""",
            
            "podcast_script": """Create a podcast script that:
1. Opens with an engaging hook
2. Summarizes key points from the newsletter
3. Provides context and analysis
4. Includes natural transitions
5. Ends with a thoughtful conclusion
6. Uses conversational language suitable for audio"""
        }
    
    def create_podcast_prompt(self,
                             newsletter_content: str,
                             chunks: Optional[List[Dict]] = None,
                             style: str = "conversational") -> str:
        base_prompt = f"""{self.system_prompts['podcast_host']}

{self.system_prompts['podcast_script']}

Newsletter Content:
{newsletter_content}

Style: {style}
Target Length: ~5,000 words (30-minute podcast)
Format: Natural conversation with smooth transitions

Generate a complete podcast script:"""
        
        if chunks:
            context = "\n\n".join([f"Section {i+1}:\n{chunk['text']}" for i, chunk in enumerate(chunks[:5])])
            base_prompt += f"\n\nRelevant Context:\n{context}"
        
        return base_prompt
    
    def create_batch_prompts(self,
                            chunks: List[Dict],
                            batch_size: int = 5) -> List[str]:
        prompts = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            context = "\n\n".join([f"Section {j+1}:\n{chunk['text']}" for j, chunk in enumerate(batch)])
            
            prompt = f"""{self.system_prompts['newsletter_summarizer']}

Newsletter Sections:
{context}

Create a podcast segment that:
- Summarizes these sections naturally
- Maintains narrative flow
- Uses conversational language
- Connects ideas smoothly

Generate the segment:"""
            
            prompts.append(prompt)
        
        return prompts
    
    def optimize_prompt(self,
                       base_prompt: str,
                       max_tokens: int = 5000,
                       temperature: float = 0.7) -> Dict:
        optimized = {
            "prompt": base_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.3
        }
        
        if len(base_prompt) > 3000:
            optimized["temperature"] = 0.6
            optimized["max_tokens"] = min(max_tokens, 4000)
        
        return optimized
