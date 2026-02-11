from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


class SemanticChunker:
    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 similarity_threshold: float = 0.7):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_text(self, text: str, preserve_semantics: bool = True) -> List[Dict]:
        if preserve_semantics:
            return self._semantic_chunking(text)
        else:
            chunks = self.text_splitter.split_text(text)
            return [{"text": chunk, "index": i} for i, chunk in enumerate(chunks)]
    
    def _semantic_chunking(self, text: str) -> List[Dict]:
        sentences = text.split('. ')
        if len(sentences) < 2:
            return [{"text": text, "index": 0, "embedding": None}]
        
        embeddings = self.model.encode(sentences, show_progress_bar=False)
        
        chunks = []
        current_chunk = []
        current_embeddings = []
        chunk_index = 0
        
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            if len(current_chunk) == 0:
                current_chunk.append(sentence)
                current_embeddings.append(embedding)
            else:
                avg_embedding = np.mean(current_embeddings, axis=0)
                similarity = np.dot(embedding, avg_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(avg_embedding) + 1e-8
                )
                
                if similarity >= self.similarity_threshold and len(' '.join(current_chunk)) < self.chunk_size:
                    current_chunk.append(sentence)
                    current_embeddings.append(embedding)
                else:
                    chunk_text = '. '.join(current_chunk)
                    if chunk_text:
                        chunks.append({
                            "text": chunk_text,
                            "index": chunk_index,
                            "embedding": np.mean(current_embeddings, axis=0),
                            "num_sentences": len(current_chunk)
                        })
                        chunk_index += 1
                    
                    current_chunk = [sentence]
                    current_embeddings = [embedding]
        
        if current_chunk:
            chunk_text = '. '.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "index": chunk_index,
                "embedding": np.mean(current_embeddings, axis=0),
                "num_sentences": len(current_chunk)
            })
        
        logger.info(f"Created {len(chunks)} semantic chunks from text")
        return chunks
    
    def merge_similar_chunks(self, chunks: List[Dict]) -> List[Dict]:
        if len(chunks) <= 1:
            return chunks
        
        merged = []
        current = chunks[0]
        
        for next_chunk in chunks[1:]:
            if current.get('embedding') is not None and next_chunk.get('embedding') is not None:
                similarity = np.dot(current['embedding'], next_chunk['embedding']) / (
                    np.linalg.norm(current['embedding']) * np.linalg.norm(next_chunk['embedding']) + 1e-8
                )
                
                if similarity > self.similarity_threshold:
                    current['text'] += ' ' + next_chunk['text']
                    current['num_sentences'] += next_chunk.get('num_sentences', 0)
                    current['embedding'] = (current['embedding'] + next_chunk['embedding']) / 2
                else:
                    merged.append(current)
                    current = next_chunk
            else:
                merged.append(current)
                current = next_chunk
        
        merged.append(current)
        return merged
