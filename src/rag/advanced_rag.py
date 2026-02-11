from typing import List, Dict, Optional
import numpy as np
from .vector_store import AdvancedVectorStore
from .semantic_chunking import SemanticChunker
from .retrieval import AdvancedRAGRetriever
import logging

logger = logging.getLogger(__name__)


class AdaptiveRAG:
    def __init__(self, vector_store, chunker, retriever, llm_inference):
        self.vector_store = vector_store
        self.chunker = chunker
        self.retriever = retriever
        self.llm_inference = llm_inference
        self.retrieval_history = []
    
    def adaptive_retrieve(self, query: str, initial_k: int = 5) -> List[Dict]:
        retrieved = self.retriever.retrieve(query, top_k=initial_k)
        
        if len(retrieved) == 0:
            expanded_queries = self.retriever.query_expansion(query)
            retrieved = self.retriever.multi_query_retrieval(expanded_queries, top_k=initial_k)
        
        confidence_scores = [doc.get('hybrid_score', doc.get('rerank_score', 0.5)) for doc in retrieved]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        if avg_confidence < 0.6:
            retrieved = self.retriever.retrieve(query, top_k=initial_k * 2)
            retrieved = retrieved[:initial_k]
        
        return retrieved
    
    def iterative_refinement(self, query: str, max_iterations: int = 3) -> str:
        context = ""
        
        for iteration in range(max_iterations):
            if iteration == 0:
                retrieved = self.adaptive_retrieve(query)
            else:
                enhanced_query = f"{query} Based on: {context[:500]}"
                retrieved = self.adaptive_retrieve(enhanced_query)
            
            new_context = "\n\n".join([doc['text'] for doc in retrieved])
            
            if len(new_context) > len(context):
                context = new_context
            else:
                break
        
        return context


class GraphRAG:
    def __init__(self, vector_store, chunker):
        self.vector_store = vector_store
        self.chunker = chunker
        self.entity_graph = {}
    
    def build_entity_graph(self, documents: List[Dict]):
        import spacy
        nlp = spacy.load("en_core_web_sm")
        
        for doc in documents:
            text = doc.get('text', '')
            doc_nlp = nlp(text)
            
            entities = [ent.text for ent in doc_nlp.ents]
            
            for i, entity1 in enumerate(entities):
                if entity1 not in self.entity_graph:
                    self.entity_graph[entity1] = set()
                
                for entity2 in entities[i+1:]:
                    self.entity_graph[entity1].add(entity2)
                    if entity2 not in self.entity_graph:
                        self.entity_graph[entity2] = set()
                    self.entity_graph[entity2].add(entity1)
    
    def retrieve_by_entities(self, query_entities: List[str], top_k: int = 5) -> List[Dict]:
        related_entities = set()
        
        for entity in query_entities:
            if entity in self.entity_graph:
                related_entities.update(self.entity_graph[entity])
        
        all_entities = set(query_entities) | related_entities
        
        query = " ".join(list(all_entities)[:10])
        
        return self.vector_store.search(query, top_k=top_k)


class MultiVectorRAG:
    def __init__(self, vector_stores: List[AdvancedVectorStore], fusion_method: str = "weighted"):
        self.vector_stores = vector_stores
        self.fusion_method = fusion_method
        self.weights = [1.0 / len(vector_stores)] * len(vector_stores)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        all_results = []
        
        for store in self.vector_stores:
            results = store.search(query, top_k=top_k * 2)
            all_results.extend(results)
        
        seen_ids = set()
        fused_results = []
        
        for result in all_results:
            if result['id'] not in seen_ids:
                fused_results.append(result)
                seen_ids.add(result['id'])
        
        fused_results.sort(
            key=lambda x: x.get('hybrid_score', x.get('distance', 1.0)),
            reverse=True
        )
        
        return fused_results[:top_k]
