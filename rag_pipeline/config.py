from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _default_supported_extensions() -> set[str]:
    return {
        ".pdf",
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".yaml",
        ".yml",
        ".log",
        ".rst",
        ".ini",
        ".toml",
        ".xml",
        ".html",
        ".htm",
    }


@dataclass(slots=True)
class ChunkingConfig:
    max_chars: int = 2000  # Increased for legal documents with longer definitions
    overlap_chars: int = 250  # Increased to preserve legal context
    min_chars: int = 100  # Lowered: some legal sections are very short (e.g., "20B. Confiscation.")


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = "BAAI/bge-base-en-v1.5"
    use_query_passage_prefix: bool = False
    batch_size: int = 16  # Reasonable default for CPU
    use_gpu: bool = False  # GPU acceleration disabled by default


@dataclass(slots=True)
class RetrievalConfig:
    dense_top_k: int = 20   # Increased from 12 → 20 for better coverage
    hybrid_top_k: int = 15  # Increased from 8 → 15 for cross-document results
    rerank_top_k: int = 10  # Increased from 5 → 10 for multi-document queries
    rrf_k: int = 60
    bm25_weight: float = 0.55   # Increased from 0.45 — keyword matches matter more for legal section lookups
    dense_weight: float = 0.45  # Decreased from 0.55 — balances with BM25
    reranker_model: str | None = None


@dataclass(slots=True)
class StorageConfig:
    collection_prefix: str = "rag"
    local_path: Path = field(default_factory=lambda: Path("qdrant_data"))


@dataclass(slots=True)
class PipelineConfig:
    data_dir: Path = field(default_factory=lambda: Path("data"))
    fallback_dir: Path = field(default_factory=lambda: Path("."))
    supported_extensions: set[str] = field(default_factory=_default_supported_extensions)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    custom_collection_name: str | None = None  # Override collection name if provided

    def collection_name(self) -> str:
        # If custom collection name is set, use it
        if self.custom_collection_name:
            return self.custom_collection_name
        
        # Otherwise, generate from embedding model
        safe_model = "".join(ch.lower() if ch.isalnum() else "_" for ch in self.embedding.model_name)
        safe_model = "_".join(part for part in safe_model.split("_") if part)
        return f"{self.storage.collection_prefix}_{safe_model}"[:100]