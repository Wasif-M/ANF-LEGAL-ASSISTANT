from __future__ import annotations

import argparse
from pathlib import Path

from rag_pipeline import PipelineConfig, RAGPipeline


def parse_extensions(value: str) -> set[str]:
    result: set[str] = set()
    for item in value.split(","):
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        result.add(ext)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index heterogeneous documents into a local Qdrant collection.")
    parser.add_argument("--data-dir", default="data", help="Primary directory containing source documents.")
    parser.add_argument("--fallback-dir", default=".", help="Fallback directory to scan if data-dir is empty.")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5", help="FastEmbed embedding model.")
    parser.add_argument("--embed-batch-size", type=int, default=16, help="Embedding batch size (lower uses less RAM).")
    parser.add_argument("--local-path", default="qdrant_data", help="Qdrant local storage path.")
    parser.add_argument(
        "--extensions",
        default="",
        help=(
            "Comma-separated extensions to index, e.g. .pdf,.md,.csv. "
            "If omitted, uses defaults from PipelineConfig."
        ),
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force re-indexing of all files, even if already in the collection. "
             "By DEFAULT, only new/changed files are indexed (INCREMENTAL mode). "
             "Use this flag only when you want to rebuild the entire index from scratch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        data_dir=Path(args.data_dir),
        fallback_dir=Path(args.fallback_dir),
    )
    config.embedding.model_name = args.model
    config.embedding.batch_size = max(1, args.embed_batch_size)
    config.storage.local_path = Path(args.local_path)
    if args.extensions.strip():
        config.supported_extensions = parse_extensions(args.extensions)

    pipeline = RAGPipeline(config)
    
    # Log the indexing mode
    if args.force_reindex:
        print("[REINDEX] FORCE REINDEX MODE: Re-indexing ALL files (including already processed ones)...")
    else:
        print("[INCREMENTAL] INCREMENTAL MODE (default): Only new/changed files will be indexed.")
    
    summary = pipeline.ingest(force_reindex=args.force_reindex)
    
    if args.force_reindex:
        print(f"\n[OK] Force re-indexed {summary.document_count} documents into {summary.chunk_count} chunks.")
    else:
        print(f"\n[OK] Indexed {summary.document_count} new/changed documents into {summary.chunk_count} chunks.")
    
    if summary.source_files:
        print("\nProcessed files:")
        for source_file in summary.source_files:
            print(f"  - {source_file}")


if __name__ == "__main__":
    main()