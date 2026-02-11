import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics import precision_score, recall_score, f1_score
import logging

logger = logging.getLogger(__name__)


class RAGEvaluator:
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_retrieval(self, 
                          retrieved_docs: List[Dict],
                          relevant_doc_ids: List[str],
                          top_k: int = 5) -> Dict:
        retrieved_ids = [doc['id'] for doc in retrieved_docs[:top_k]]
        
        relevant_retrieved = set(retrieved_ids) & set(relevant_doc_ids)
        
        precision = len(relevant_retrieved) / max(len(retrieved_ids), 1)
        recall = len(relevant_retrieved) / max(len(relevant_doc_ids), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'relevant_retrieved': len(relevant_retrieved),
            'total_retrieved': len(retrieved_ids),
            'total_relevant': len(relevant_doc_ids)
        }
    
    def evaluate_generation(self, 
                            generated_text: str,
                            reference_text: str) -> Dict:
        from nltk.translate.bleu_score import sentence_bleu
        from rouge_score import rouge_scorer
        
        generated_words = generated_text.lower().split()
        reference_words = reference_text.lower().split()
        
        bleu = sentence_bleu([reference_words], generated_words)
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference_text, generated_text)
        
        return {
            'bleu': bleu,
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure
        }
    
    def calculate_diversity(self, generated_texts: List[str]) -> float:
        unique_ngrams = set()
        total_ngrams = 0
        
        for text in generated_texts:
            words = text.lower().split()
            for i in range(len(words) - 1):
                bigram = (words[i], words[i+1])
                unique_ngrams.add(bigram)
                total_ngrams += 1
        
        diversity = len(unique_ngrams) / max(total_ngrams, 1)
        return diversity
