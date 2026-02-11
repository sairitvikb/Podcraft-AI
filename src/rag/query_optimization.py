import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class QueryOptimizer:
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
    
    def expand_query(self, query: str, method: str = 'synonym') -> List[str]:
        if method == 'synonym':
            return self._synonym_expansion(query)
        elif method == 'paraphrase':
            return self._paraphrase_expansion(query)
        elif method == 'keyword':
            return self._keyword_expansion(query)
        else:
            return [query]
    
    def _synonym_expansion(self, query: str) -> List[str]:
        words = query.split()
        expanded = [query]
        
        for word in words[:5]:
            expanded.append(f"{query} {word}")
        
        return expanded
    
    def _paraphrase_expansion(self, query: str) -> List[str]:
        return [
            query,
            f"What is {query}?",
            f"Tell me about {query}",
            f"Explain {query}",
            f"Details on {query}"
        ]
    
    def _keyword_expansion(self, query: str) -> List[str]:
        words = query.split()
        expanded = []
        
        for i in range(len(words)):
            for j in range(i+1, min(i+4, len(words)+1)):
                expanded.append(' '.join(words[i:j]))
        
        return expanded[:10]
    
    def optimize_query(self, query: str, context: Optional[str] = None) -> str:
        if context:
            optimized = f"{query} Context: {context[:200]}"
        else:
            optimized = query
        
        return optimized


class QueryRewriter:
    def __init__(self):
        self.rewrite_patterns = [
            (r'what is (.+)', r'\1 explanation'),
            (r'tell me about (.+)', r'\1 information'),
            (r'explain (.+)', r'\1 details'),
        ]
    
    def rewrite(self, query: str) -> str:
        import re
        rewritten = query
        
        for pattern, replacement in self.rewrite_patterns:
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
        
        return rewritten
