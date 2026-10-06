import math
from typing import List, Optional, Union
import time
import os
from app.multilingual.model_manager import model_manager

class EmbeddingService:
    """
    Service for generating semantic embeddings using multilingual models.
    """
    def __init__(self):
        self.default_model = os.getenv("TRUSTLENS_MULTILINGUAL_MODEL", "xlm-roberta")

    def _validate_embedding(self, embedding: List[float], dimension: int) -> str:
        if not embedding:
            return "invalid"
        if len(embedding) != dimension:
            return "invalid"
        for val in embedding:
            if not isinstance(val, (int, float)):
                return "invalid"
            if math.isnan(val) or math.isinf(val):
                return "invalid"
        return "valid"

    def encode(self, text: str, model_name: Optional[str] = None) -> dict:
        """
        Encode a single text and return metadata.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        model_to_use = model_name or self.default_model
        model = model_manager.get_model(model_to_use)
        
        if not model:
            raise RuntimeError(f"Model {model_to_use} is unavailable.")

        start_time = time.time()
        embeddings = model.encode([text])
        latency = time.time() - start_time

        embedding = embeddings[0]
        dimension = model.embedding_dimension
        quality = self._validate_embedding(embedding, dimension)
        
        return {
            "embedding": embedding,
            "embedding_dimension": dimension,
            "model_name": model_to_use,
            "embedding_quality": quality,
            "latency_seconds": latency,
            "device": str(model.device)
        }

    def encode_batch(self, texts: List[str], model_name: Optional[str] = None) -> List[dict]:
        """
        Encode a batch of texts.
        """
        if not texts:
            return []

        model_to_use = model_name or self.default_model
        model = model_manager.get_model(model_to_use)
        
        if not model:
            raise RuntimeError(f"Model {model_to_use} is unavailable.")

        # Filter empty
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return []

        start_time = time.time()
        embeddings = model.encode(valid_texts)
        latency = time.time() - start_time

        dimension = model.embedding_dimension
        
        results = []
        for emb in embeddings:
            quality = self._validate_embedding(emb, dimension)
            results.append({
                "embedding": emb,
                "embedding_dimension": dimension,
                "model_name": model_to_use,
                "embedding_quality": quality,
                "batch_latency_seconds": latency,
                "device": str(model.device)
            })
            
        return results

embedding_service = EmbeddingService()
