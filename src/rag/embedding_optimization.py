import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import torch
import logging

logger = logging.getLogger(__name__)


class EmbeddingOptimizer:
    def __init__(self, model: SentenceTransformer):
        self.model = model
    
    def optimize_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def reduce_dimensionality(self, embeddings: np.ndarray, target_dim: int = 128, method: str = 'pca') -> np.ndarray:
        if method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=target_dim)
            reduced = pca.fit_transform(embeddings)
        elif method == 'umap':
            from umap import UMAP
            umap = UMAP(n_components=target_dim, random_state=42)
            reduced = umap.fit_transform(embeddings)
        else:
            reduced = embeddings[:, :target_dim]
        
        return reduced
    
    def quantize_embeddings(self, embeddings: np.ndarray, bits: int = 8) -> np.ndarray:
        min_val = np.min(embeddings)
        max_val = np.max(embeddings)
        
        scale = (2 ** bits - 1) / (max_val - min_val)
        quantized = np.round((embeddings - min_val) * scale).astype(np.uint8)
        
        dequantized = quantized.astype(float) / scale + min_val
        
        return dequantized
