import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import logging
from collections import defaultdict
import hashlib
import pickle

logger = logging.getLogger(__name__)


class CrossEncoderReranker(nn.Module):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        super().__init__()
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
    
    def forward(self, query: str, documents: List[str]) -> torch.Tensor:
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        return torch.tensor(scores)


class ColBERTReranker:
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def encode_query(self, query: str) -> torch.Tensor:
        tokens = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state
    
    def encode_document(self, document: str) -> torch.Tensor:
        tokens = self.tokenizer(document, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state
    
    def score(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> float:
        scores = torch.max(torch.sum(query_emb.unsqueeze(1) * doc_emb.unsqueeze(0), dim=-1), dim=-1)[0]
        return scores.sum().item()


class LearnedSparseRetrieval:
    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def encode(self, text: str) -> Dict[str, float]:
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        logits = outputs.logits
        relu_log = torch.log(1 + torch.relu(logits))
        weighted = torch.max(relu_log, dim=1)[0]
        
        sparse_vector = {}
        for idx, weight in enumerate(weighted[0]):
            if weight > 0:
                token = self.tokenizer.convert_ids_to_tokens([idx])[0]
                sparse_vector[token] = weight.item()
        
        return sparse_vector


class MultiVectorRetrieval:
    def __init__(self, embedding_model: SentenceTransformer, num_vectors: int = 5):
        self.embedding_model = embedding_model
        self.num_vectors = num_vectors
    
    def encode_multi_vector(self, text: str) -> List[np.ndarray]:
        sentences = text.split('. ')
        if len(sentences) < self.num_vectors:
            sentences = [text] * self.num_vectors
        
        vectors = []
        for sentence in sentences[:self.num_vectors]:
            vector = self.embedding_model.encode(sentence, show_progress_bar=False)
            vectors.append(vector)
        
        return vectors
    
    def search(self, query: str, document_vectors: List[List[np.ndarray]]) -> List[float]:
        query_vectors = self.encode_multi_vector(query)
        
        scores = []
        for doc_vectors in document_vectors:
            max_score = 0.0
            for q_vec in query_vectors:
                for d_vec in doc_vectors:
                    score = np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec))
                    max_score = max(max_score, score)
            scores.append(max_score)
        
        return scores


class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        self.k = k
    
    def fuse(self, ranked_lists: List[List[Dict]]) -> List[Dict]:
        scores = defaultdict(float)
        
        for ranked_list in ranked_lists:
            for rank, item in enumerate(ranked_list, 1):
                doc_id = item.get('id', str(item))
                scores[doc_id] += 1.0 / (self.k + rank)
        
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [{'id': doc_id, 'score': score} for doc_id, score in fused]


class QueryExpansion:
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
    
    def expand_with_synonyms(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = self.embedding_model.encode(query, show_progress_bar=False)
        
        expanded_queries = [query]
        
        words = query.split()
        for word in words[:5]:
            expanded_queries.append(f"{query} {word}")
        
        return expanded_queries[:top_k]
    
    def expand_with_paraphrase(self, query: str) -> List[str]:
        expanded = [
            query,
            f"What is {query}?",
            f"Tell me about {query}",
            f"Explain {query}"
        ]
        return expanded


class AdaptiveRetrieval:
    def __init__(self, vector_store, reranker, query_expander):
        self.vector_store = vector_store
        self.reranker = reranker
        self.query_expander = query_expander
        self.retrieval_history = []
    
    def retrieve(self, query: str, top_k: int = 5, 
                adaptive: bool = True) -> List[Dict]:
        if adaptive:
            expanded_queries = self.query_expander.expand_with_synonyms(query)
            
            all_results = []
            for eq in expanded_queries:
                results = self.vector_store.search(eq, top_k=top_k * 2)
                all_results.extend(results)
            
            seen_ids = set()
            unique_results = []
            for result in all_results:
                if result['id'] not in seen_ids:
                    unique_results.append(result)
                    seen_ids.add(result['id'])
            
            if self.reranker:
                reranked = self.reranker.forward(query, [r['text'] for r in unique_results])
                scored_results = [
                    {**r, 'rerank_score': float(score)}
                    for r, score in zip(unique_results, reranked)
                ]
                scored_results.sort(key=lambda x: x['rerank_score'], reverse=True)
                return scored_results[:top_k]
            
            return unique_results[:top_k]
        else:
            return self.vector_store.search(query, top_k=top_k)


class HybridRetrieval:
    def __init__(self, dense_retriever, sparse_retriever, fusion_method: str = "rrf"):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion_method = fusion_method
        self.rrf = ReciprocalRankFusion() if fusion_method == "rrf" else None
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        dense_results = self.dense_retriever.search(query, top_k=top_k * 2)
        sparse_results = self.sparse_retriever.search(query, top_k=top_k * 2)
        
        if self.fusion_method == "rrf":
            fused = self.rrf.fuse([dense_results, sparse_results])
            return fused[:top_k]
        else:
            combined = {}
            for result in dense_results:
                doc_id = result.get('id', str(result))
                combined[doc_id] = combined.get(doc_id, 0) + result.get('score', 0) * 0.7
            
            for result in sparse_results:
                doc_id = result.get('id', str(result))
                combined[doc_id] = combined.get(doc_id, 0) + result.get('score', 0) * 0.3
            
            sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
            return [{'id': doc_id, 'score': score} for doc_id, score in sorted_results[:top_k]]


class ContextualCompression:
    def __init__(self, llm_model):
        self.llm_model = llm_model
    
    def compress(self, documents: List[str], query: str) -> List[str]:
        compressed = []
        
        for doc in documents:
            prompt = f"""Given the following document and query, extract only the relevant information.

Query: {query}

Document: {doc}

Relevant information:"""
            
            compressed_doc = self.llm_model.generate(prompt, max_length=200)
            compressed.append(compressed_doc)
        
        return compressed


class MetadataFiltering:
    def __init__(self, metadata_fields: List[str]):
        self.metadata_fields = metadata_fields
    
    def filter(self, documents: List[Dict], filters: Dict) -> List[Dict]:
        filtered = []
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            match = True
            
            for field, value in filters.items():
                if field in metadata:
                    if isinstance(value, list):
                        if metadata[field] not in value:
                            match = False
                            break
                    else:
                        if metadata[field] != value:
                            match = False
                            break
            
            if match:
                filtered.append(doc)
        
        return filtered
