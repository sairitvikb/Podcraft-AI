from typing import List, Dict, Optional
import numpy as np
from .vector_store import AdvancedVectorStore, Reranker
from .semantic_chunking import SemanticChunker
import logging

logger = logging.getLogger(__name__)


class AdvancedRAGRetriever:
    def __init__(self,
                 vector_store: AdvancedVectorStore,
                 chunker: SemanticChunker,
                 reranker: Optional[Reranker] = None,
                 retrieval_strategy: str = "hybrid"):
        self.vector_store = vector_store
        self.chunker = chunker
        self.reranker = reranker
        self.retrieval_strategy = retrieval_strategy
    
    def retrieve(self, query: str, top_k: int = 5,
                use_reranking: bool = True) -> List[Dict]:
        logger.info(f"Retrieving documents for query: {query[:50]}...")
        
        if self.retrieval_strategy == "hybrid":
            retrieved = self.vector_store.hybrid_search(query, top_k=top_k * 2)
        elif self.retrieval_strategy == "semantic":
            retrieved = self.vector_store.search(query, top_k=top_k * 2)
        else:
            retrieved = self.vector_store._keyword_search(query, top_k=top_k * 2)
        
        if use_reranking and self.reranker:
            retrieved = self.reranker.rerank(query, retrieved, top_k=top_k)
        else:
            retrieved = retrieved[:top_k]
        
        logger.info(f"Retrieved {len(retrieved)} documents")
        return retrieved
    
    def retrieve_with_context(self, query: str, context: Optional[str] = None,
                             top_k: int = 5) -> Dict:
        if context:
            enhanced_query = f"{query} Context: {context}"
        else:
            enhanced_query = query
        
        retrieved = self.retrieve(enhanced_query, top_k=top_k)
        
        context_text = "\n\n".join([doc['text'] for doc in retrieved])
        
        return {
            'query': query,
            'retrieved_documents': retrieved,
            'context': context_text,
            'num_documents': len(retrieved)
        }
    
    def multi_query_retrieval(self, queries: List[str], top_k: int = 5) -> List[Dict]:
        all_retrieved = []
        seen_ids = set()
        
        for query in queries:
            retrieved = self.retrieve(query, top_k=top_k)
            
            for doc in retrieved:
                if doc['id'] not in seen_ids:
                    all_retrieved.append(doc)
                    seen_ids.add(doc['id'])
        
        all_retrieved.sort(key=lambda x: x.get('hybrid_score', x.get('rerank_score', 0)), reverse=True)
        
        return all_retrieved[:top_k]
    
    def query_expansion(self, query: str) -> List[str]:
        expanded_queries = [query]
        
        query_words = query.split()
        
        if len(query_words) > 1:
            expanded_queries.append(" ".join(query_words[:len(query_words)//2]))
            expanded_queries.append(" ".join(query_words[len(query_words)//2:]))
        
        return expanded_queries


class RAGPipeline:
    def __init__(self,
                 vector_store: AdvancedVectorStore,
                 chunker: SemanticChunker,
                 retriever: AdvancedRAGRetriever,
                 llm_inference):
        self.vector_store = vector_store
        self.chunker = chunker
        self.retriever = retriever
        self.llm_inference = llm_inference
    
    def process_newsletter(self, newsletter_text: str) -> Dict:
        logger.info("Processing newsletter through RAG pipeline...")
        
        chunks = self.chunker.chunk_text(newsletter_text, preserve_semantics=True)
        
        chunk_docs = [
            {
                'id': f"chunk_{i}",
                'text': chunk['text'],
                'metadata': {
                    'chunk_index': chunk['index'],
                    'num_sentences': chunk.get('num_sentences', 0)
                }
            }
            for i, chunk in enumerate(chunks)
        ]
        
        self.vector_store.add_documents(chunk_docs)
        
        query = "Summarize the main topics and key insights"
        retrieved = self.retriever.retrieve(query, top_k=5)
        
        context = "\n\n".join([doc['text'] for doc in retrieved])
        
        return {
            'chunks': chunks,
            'retrieved_context': context,
            'retrieved_documents': retrieved
        }
    
    def generate_with_rag(self, query: str, newsletter_text: Optional[str] = None) -> str:
        if newsletter_text:
            self.process_newsletter(newsletter_text)
        
        retrieval_result = self.retriever.retrieve_with_context(query)
        
        prompt = f"""Based on the following context, answer the query.

Context:
{retrieval_result['context']}

Query: {query}

Answer:"""
        
        response = self.llm_inference._generate_sync(prompt)
        
        return response
