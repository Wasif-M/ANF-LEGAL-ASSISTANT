from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chunking import chunk_document
from .config import PipelineConfig
from .embeddings import Embedder
from .loaders import discover_documents, load_document
from .retrieval import HybridRetriever
from .storage import QdrantChunkStore
from .utils import chunk_metadata


@dataclass(slots=True)
class IndexedDocumentSummary:
    document_count: int
    chunk_count: int
    source_files: list[str]


class RAGPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.embedder = Embedder(self.config.embedding)
        self.store = QdrantChunkStore(self.config.collection_name(), self.config.storage.local_path)
        self.retriever = HybridRetriever(
            store=self.store,
            embedder=self.embedder,
            dense_top_k=self.config.retrieval.dense_top_k,
            hybrid_top_k=self.config.retrieval.hybrid_top_k,
            rerank_top_k=self.config.retrieval.rerank_top_k,
            rrf_k=self.config.retrieval.rrf_k,
            bm25_weight=self.config.retrieval.bm25_weight,
            dense_weight=self.config.retrieval.dense_weight,
            reranker_model=self.config.retrieval.reranker_model,
        )

    @property
    def qdrant_client(self):
        """Expose Qdrant client for direct access."""
        return self.store.client

    def discover_sources(self) -> list[Path]:
        data_sources = discover_documents([self.config.data_dir], self.config.supported_extensions)
        if data_sources:
            return data_sources
        return discover_documents([self.config.fallback_dir], self.config.supported_extensions)

    def ingest(self, force_reindex: bool = False) -> IndexedDocumentSummary:
        """Ingest documents into the vector store.
        
        Args:
            force_reindex: If True, re-index all files even if already in collection.
                          If False, only index new/changed files (incremental indexing).
        """
        sources = self.discover_sources()
        print(f"Found {len(sources)} total documents to process.")
        
        # Get already-indexed files from the store
        existing_chunks = self.store.scroll_all() if not force_reindex else []
        indexed_sources = {chunk.get("source_path") for chunk in existing_chunks}
        
        if indexed_sources and not force_reindex:
            print(f"Already indexed {len(indexed_sources)} files.")
        
        # Filter to only new/changed files
        sources_to_index = [s for s in sources if s.resolve().as_posix() not in indexed_sources]
        
        if not sources_to_index and not force_reindex:
            # No new files to index; return summary of existing state
            print(f"✓ No new files to index. {len(indexed_sources)} files already indexed.")
            return IndexedDocumentSummary(
                document_count=len(sources),
                chunk_count=len(existing_chunks),
                source_files=[],
            )
        
        # If force_reindex, process all sources; otherwise only process new ones
        sources_to_process = sources if force_reindex else sources_to_index
        
        if sources_to_process:
            print(f"Processing {len(sources_to_process)} file(s) for embedding...")
        
        documents = [load_document(path) for path in sources_to_process]
        chunks = []
        for document in documents:
            doc_chunks = chunk_document(document, self.config.chunking)
            chunks.extend(doc_chunks)
            print(f"  [doc] {document.title}: {len(doc_chunks)} chunks")

        if not chunks:
            if force_reindex:
                self.store.reset(self.embedder.dimension)
                self.retriever._refresh_indexes()
            return IndexedDocumentSummary(
                document_count=len(documents),
                chunk_count=0,
                source_files=[path.as_posix() for path in sources_to_process],
            )

        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        
        # If force reindexing, reset the collection; otherwise, ensure it exists for incremental upsert
        if force_reindex:
            self.store.reset(vectors.shape[1])
        else:
            self.store.ensure_collection_exists(vectors.shape[1])
        
        self.store.upsert(chunks, vectors)
        self.retriever._refresh_indexes()
        
        return IndexedDocumentSummary(
            document_count=len(documents),
            chunk_count=len(chunks),
            source_files=[path.as_posix() for path in sources_to_process],
        )

    def build_context(
        self, query: str, max_chars: int = 15000, expansion_terms: str | None = None
    ) -> tuple[str, list[dict], list]:
        """Build context from retrieved results.

        Returns tuple of (formatted_context, sources_list, retrieved_chunks)
        """
        results = self.retriever.search(query, expansion_terms=expansion_terms)
        context_blocks: list[str] = []
        sources: list[dict] = []
        total_chars = 0

        for rank, result in enumerate(results, start=1):
            chunk = result.chunk
            metadata = chunk_metadata(chunk)
            doc_title = metadata.get("title", chunk.source_path)
            section_num = metadata.get("section_number", "")
            
            block = (
                f"[{rank}] Source: {doc_title}\n"
                f"File: {chunk.source_path}\n"
                f"Section: {' > '.join(chunk.section_path) if chunk.section_path else 'Document'}\n"
            )
            if section_num:
                block += f"Section Number: {section_num}\n"
            block += f"Text: {chunk.text}"
            
            if total_chars + len(block) > max_chars:
                break
            context_blocks.append(block)
            total_chars += len(block)
            sources.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.chunk_id,
                    "source_path": chunk.source_path,
                    "section_path": list(chunk.section_path),
                    "section_number": section_num,
                    "document_title": doc_title,
                    "dense_score": result.dense_score,
                    "lexical_score": result.lexical_score,
                    "fused_score": result.fused_score,
                    "rerank_score": result.rerank_score,
                }
            )

        return "\n\n---\n\n".join(context_blocks), sources, results

    def build_prompt(
        self,
        query: str,
        max_chars: int = 15000,
        use_legal_format: bool = True,
        expansion_terms: str | None = None,
    ) -> tuple[str, list[dict]]:
        """Build prompt with optional legal formatting.

        Args:
            query: User's question
            max_chars: Maximum context length
            use_legal_format: Whether to use dynamic legal response format
            expansion_terms: Optional statutory-vocabulary terms to widen retrieval

        Returns:
            Tuple of (formatted_prompt, sources)
        """
        from .prompts import build_legal_prompt, get_system_prompt, classify_query_intent

        context, sources, retrieved_chunks = self.build_context(
            query, max_chars=max_chars, expansion_terms=expansion_terms
        )
        
        if use_legal_format:
            # Classify query intent for dynamic template selection
            query_intent = classify_query_intent(query)
            
            # Use dynamic legal prompt with cross-references and intent-aware template
            prompt = build_legal_prompt(
                question=query,
                context=context,
                retrieved_chunks=retrieved_chunks,
                query_intent=query_intent,
            )
            # Add system context
            system_prompt = get_system_prompt()
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            # Use simple format
            full_prompt = (
                "You are a precise retrieval-augmented assistant.\n"
                "Answer only from the provided context. If the context is insufficient, say so clearly.\n\n"
                f"Question:\n{query}\n\n"
                f"Context:\n{context}\n\n"
                "Answer:\n"
            )
        
        return full_prompt, sources

    def answer(self, query: str, llm: Callable[[str], str] | None = None) -> dict:
        prompt, sources = self.build_prompt(query)
        answer = llm(prompt) if llm else None
        return {"query": query, "prompt": prompt, "answer": answer, "sources": sources}