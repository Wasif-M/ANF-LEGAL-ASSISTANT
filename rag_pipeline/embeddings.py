from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

# Use the model already cached under models_cache/ and never hit the network at
# load time. Without this, fastembed tries HuggingFace first; on a flaky/offline
# link that stalls for minutes of retries before falling back to the local copy.
# setdefault keeps any value the environment already set.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastembed import TextEmbedding

from .config import EmbeddingConfig


def _get_onnx_providers() -> list[str]:
    """Return CPU-only ONNX provider list. GPU support removed in this build."""
    return ['CPUExecutionProvider']


@dataclass(slots=True)
class Embedder:
    config: EmbeddingConfig
    _model_instance: TextEmbedding | None = field(default=None, init=False, repr=False)

    def _model(self) -> TextEmbedding:
        if self._model_instance is None:
            # Configure providers (CPU-only for this build)
            providers = _get_onnx_providers()
            
            self._model_instance = TextEmbedding(
                model_name=self.config.model_name,
                providers=providers,
                max_length=512,  # Ensure consistent max length
                cache_dir="models_cache",
            )
        return self._model_instance

    @property
    def dimension(self) -> int:
        return int(self._model().embedding_size)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        vectors: list[np.ndarray] = []
        batch_size = max(1, self.config.batch_size)
        index = 0

        while index < len(texts):
            batch = texts[index : index + batch_size]
            try:
                print(f"[embedder] embedding documents {index + 1}-{index + len(batch)} of {len(texts)}")
                batch_vectors = list(self._model().passage_embed(batch))
                vectors.extend(batch_vectors)
                index += len(batch)
            except Exception:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)

        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vector = next(self._model().query_embed(text))
        return np.asarray(vector, dtype=np.float32)