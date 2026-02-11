import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import logging
import pickle
import hashlib

logger = logging.getLogger(__name__)


class AdvancedVectorStore:
    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 collection_name: str = "newsletters",
                 persist_directory: Optional[str] = None):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.collection_name = collection_name
        
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(name=collection_name)
        
        self.cache = {}
        logger.info(f"Vector store initialized: {collection_name}")
    
    def add_documents(self, documents: List[Dict], batch_size: int = 100):
        logger.info(f"Adding {len(documents)} documents to vector store...")
        
        ids = []
        texts = []
        embeddings = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            doc_id = doc.get('id', f"doc_{i}")
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            embedding = self.embedding_model.encode(text, show_progress_bar=False)
            
            ids.append(doc_id)
            texts.append(text)
            embeddings.append(embedding.tolist())
            metadatas.append(metadata)
            
            if len(ids) >= batch_size:
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                ids, texts, embeddings, metadatas = [], [], [], []
        
        if ids:
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def search(self, query: str, top_k: int = 5, 
              filter_dict: Optional[Dict] = None) -> List[Dict]:
        query_embedding = self.embedding_model.encode(query, show_progress_bar=False)
        
        where_clause = filter_dict if filter_dict else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause
        )
        
        retrieved_docs = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                retrieved_docs.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return retrieved_docs
    
    def hybrid_search(self, query: str, top_k: int = 5,
                     keyword_weight: float = 0.3,
                     semantic_weight: float = 0.7) -> List[Dict]:
        semantic_results = self.search(query, top_k=top_k * 2)
        
        keyword_results = self._keyword_search(query, top_k=top_k * 2)
        
        combined_scores = {}
        
        for doc in semantic_results:
            doc_id = doc['id']
            semantic_score = 1.0 - (doc.get('distance', 1.0) if doc.get('distance') else 1.0)
            combined_scores[doc_id] = semantic_weight * semantic_score
        
        for doc in keyword_results:
            doc_id = doc['id']
            keyword_score = doc.get('keyword_score', 0.0)
            if doc_id in combined_scores:
                combined_scores[doc_id] += keyword_weight * keyword_score
            else:
                combined_scores[doc_id] = keyword_weight * keyword_score
        
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        seen_ids = set()
        
        for doc_id, score in sorted_docs[:top_k]:
            if doc_id not in seen_ids:
                doc = next((d for d in semantic_results + keyword_results if d['id'] == doc_id), None)
                if doc:
                    doc['hybrid_score'] = score
                    final_results.append(doc)
                    seen_ids.add(doc_id)
        
        return final_results
    
    def _keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_words = set(query.lower().split())
        
        all_docs = self.collection.get()
        
        scored_docs = []
        for i, doc_text in enumerate(all_docs['documents']):
            doc_words = set(doc_text.lower().split())
            intersection = query_words.intersection(doc_words)
            score = len(intersection) / max(len(query_words), 1)
            
            scored_docs.append({
                'id': all_docs['ids'][i],
                'text': doc_text,
                'metadata': all_docs['metadatas'][i],
                'keyword_score': score
            })
        
        scored_docs.sort(key=lambda x: x['keyword_score'], reverse=True)
        return scored_docs[:top_k]
    
    def update_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None):
        embedding = self.embedding_model.encode(text, show_progress_bar=False)
        
        self.collection.update(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding.tolist()],
            metadatas=[metadata] if metadata else [{}]
        )
    
    def delete_document(self, doc_id: str):
        self.collection.delete(ids=[doc_id])


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        logger.info(f"Reranker initialized: {model_name}")
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        pairs = [[query, doc['text']] for doc in documents]
        
        scores = self.model.predict(pairs)
        
        scored_docs = [
            {**doc, 'rerank_score': float(score)}
            for doc, score in zip(documents, scores)
        ]
        
        scored_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return scored_docs[:top_k]
