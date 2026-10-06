"""
Log Vectorizer & Embeddings Generator.
Part of Member 1 (Data & Preprocessing) deliverables.

Converts parsed log messages into dense semantic embeddings using
sentence-transformers (all-MiniLM-L6-v2) or an ultra-fast TF-IDF fallback.
Emits records strictly adhering to Schema Contract A.1: PreprocessedLogRecord.
"""

from typing import Dict, List, Optional, Union
import numpy as np

from src.schemas import PreprocessedLogRecord


class LogVectorizer:
    """
    Transforms text log templates/messages into numerical feature vectors.
    Includes template-level embedding caching for blazing fast batch processing.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_sentence_transformers: bool = True,
        vector_dim: int = 384,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.vector_dim = vector_dim
        self.use_sentence_transformers = use_sentence_transformers
        self.embedding_cache: Dict[str, List[float]] = {}
        self.model = None

        if self.use_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[Vectorizer] Loading SentenceTransformer model '{model_name}'...")
                if hasattr(self.model, "get_embedding_dimension"):
                    self.vector_dim = self.model.get_embedding_dimension()
                else:
                    self.vector_dim = self.model.get_sentence_embedding_dimension()
                print(f"[Vectorizer] Model loaded successfully. Dimension: {self.vector_dim}")
            except Exception as e:
                print(f"[Vectorizer] Warning: Failed to load sentence-transformers ({e}).")
                print("[Vectorizer] Falling back to fast deterministic TF-IDF / hash vectorizer.")
                self.use_sentence_transformers = False

    def _fallback_embed(self, text: str) -> List[float]:
        """Deterministic fast feature hashing for offline testing without neural network."""
        import hashlib
        vec = np.zeros(self.vector_dim, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vec.tolist()
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.vector_dim
            sign = 1.0 if (h & 1) else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return [float(x) for x in vec]

    def encode_text(self, text: str) -> List[float]:
        """Encodes a single text string into a float vector with cache lookup."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        if self.use_sentence_transformers and self.model is not None:
            embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            vector = [float(x) for x in embedding]
        else:
            vector = self._fallback_embed(text)

        self.embedding_cache[text] = vector
        return vector

    def encode_batch(self, texts: List[str], batch_size: int = 128) -> List[List[float]]:
        """Encodes a batch of texts, leveraging template caching where possible."""
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for idx, t in enumerate(texts):
            if t in self.embedding_cache:
                results[idx] = self.embedding_cache[t]
            else:
                uncached_indices.append(idx)
                uncached_texts.append(t)

        if uncached_texts:
            if self.use_sentence_transformers and self.model is not None:
                encoded = self.model.encode(
                    uncached_texts,
                    batch_size=batch_size,
                    show_progress_bar=len(uncached_texts) > 500,
                    convert_to_numpy=True
                )
                for orig_idx, text, emb in zip(uncached_indices, uncached_texts, encoded):
                    vec = [float(x) for x in emb]
                    self.embedding_cache[text] = vec
                    results[orig_idx] = vec
            else:
                for orig_idx, text in zip(uncached_indices, uncached_texts):
                    vec = self._fallback_embed(text)
                    self.embedding_cache[text] = vec
                    results[orig_idx] = vec

        return [r for r in results if r is not None]

    def create_preprocessed_record(
        self,
        log_id: str,
        timestamp: str,
        raw_text: str,
        source: str,
        text_to_embed: str,
        metrics: Optional[dict] = None
    ) -> PreprocessedLogRecord:
        """
        Creates and validates a Schema Contract A.1 PreprocessedLogRecord instance.
        """
        embedding = self.encode_text(text_to_embed)
        return PreprocessedLogRecord(
            log_id=log_id,
            timestamp=timestamp,
            raw_text=raw_text,
            source=source,
            embedding=embedding,
            metrics=metrics
        )
