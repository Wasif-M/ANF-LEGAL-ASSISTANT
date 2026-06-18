# Heterogeneous Document RAG

This project builds a local RAG pipeline for documents with mixed and inconsistent structure, using Python and a local Qdrant store.

## What it does

- Detects hierarchical structure when the document exposes it: chapter/section labels, numbered sections, alphabetic subsections, and roman numeral sublevels.
- Falls back to recursive chunking when the formatting is noisy or unstructured.
- Retrieves with a hybrid dense + BM25 strategy, with an optional reranking hook you can plug in later.
- Persists vectors in local Qdrant storage at `qdrant_data/`.

## Recommended strategy

For highly variable legal or policy documents, the best practical setup is:

1. Structure-aware chunking first. Preserve heading paths in chunk metadata so answers keep section context.
2. Recursive fallback chunking for sections that are too long or badly formatted.
3. Hybrid retrieval. Dense embeddings capture semantics, BM25 preserves exact legal wording, and reranking improves final precision.
4. Use a strong embedding model. Good local choices with the current setup are `BAAI/bge-base-en-v1.5`, `BAAI/bge-small-en-v1.5`, or `intfloat/multilingual-e5-large` when multilingual coverage matters.
5. Store both payload text and section metadata in Qdrant so citations stay traceable.

## Install

```bash
pip install -r requirements.txt
```

## Index documents

```bash
python ingest.py --data-dir data --fallback-dir .
```

```bash
python ingest.py --embed-batch-size 8
```

If `data/` is empty, the pipeline scans the fallback directory and indexes any PDF, TXT, or MD documents it finds.

By default, the pipeline indexes these extensions: `.pdf`, `.txt`, `.md`, `.csv`, `.json`, `.yaml`, `.yml`, `.log`, `.rst`, `.ini`, `.toml`, `.xml`, `.html`, `.htm`.

To override extensions for a run:

```bash
python ingest.py --data-dir data --fallback-dir . --extensions .pdf,.md,.csv,.json
```

## Ask questions

```bash
python query.py "What are the main legal provisions about appeals?"
```

You can pass the same `--extensions` option on `query.py` when you need a custom file-type set.

## Run API

Start the FastAPI server (from the project root):

```bash
uvicorn api:app --reload --port 8000
```

Then POST JSON to `http://localhost:8000/query` with `question` (and optional `max_chars`). The endpoint returns the assembled prompt and the top chunks with scores and metadata.

## Notes on chunking

- Main headings are treated as level 1 sections.
- Numeric subsections like `1`, `1.1`, and `1.1.1` are assigned deeper levels automatically.
- Alphabetic and roman numeral sections are recognized as lower levels when the format looks legal or formal.
- If a section is still too long, it is split recursively by paragraphs, then sentences, then words.

## Notes on retrieval

- Dense retrieval finds semantic matches.
- BM25 improves exact-match recall for citations, statutes, and terms of art.
- Reranking is optional but recommended once the corpus is large enough.