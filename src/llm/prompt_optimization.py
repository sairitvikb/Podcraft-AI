import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class PromptOptimizer:
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.prompt_templates = {}
    
    def optimize(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        if context:
            for key, value in context.items():
                prompt = prompt.replace(f"{{{key}}}", str(value))
        return prompt
    
    def register_template(self, name: str, template: str) -> None:
        self.prompt_templates[name] = template
    
    def get_template(self, name: str) -> str:
        return self.prompt_templates.get(name, "")


class FewShotPromptBuilder:
    def __init__(self, max_examples: int = 5):
        self.max_examples = max_examples
        self.examples: List[Dict[str, str]] = []
    
    def add_example(self, input_text: str, output_text: str) -> None:
        self.examples.append({"input": input_text, "output": output_text})
        if len(self.examples) > self.max_examples:
            self.examples = self.examples[-self.max_examples:]
    
    def build_prompt(self, query: str, task_description: str = "") -> str:
        prompt_parts = []
        if task_description:
            prompt_parts.append(task_description)
        for ex in self.examples:
            prompt_parts.append(f"Input: {ex['input']}")
            prompt_parts.append(f"Output: {ex['output']}")
        prompt_parts.append(f"Input: {query}")
        prompt_parts.append("Output:")
        return "\n\n".join(prompt_parts)
    
    def clear_examples(self) -> None:
        self.examples = []
