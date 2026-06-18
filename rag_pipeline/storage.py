from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from .models import DocumentChunk


class QdrantChunkStore:
    def __init__(self, collection_name: str, local_path: Path) -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(path=str(local_path))

    def reset(self, vector_size: int) -> None:
        # Explicitly delete then create. `recreate_collection` does NOT reliably
        # clear an existing collection in qdrant-local mode: a smaller rebuild only
        # overwrites the low chunk-index point IDs and leaves the previous build's
        # higher-index points orphaned, mixing stale (and duplicate) data into the
        # "fresh" index. Deleting first guarantees a truly clean rebuild.
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
        )

    def ensure_collection_exists(self, vector_size: int) -> None:
        """Create the collection if it doesn't already exist."""
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            # Collection doesn't exist, create it
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
            )

    def upsert(self, chunks: list[DocumentChunk], vectors: np.ndarray) -> None:
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            payload = asdict(chunk)
            payload["section_path"] = list(chunk.section_path)
            points.append(
                rest.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                    vector=vector.tolist(),
                    payload=payload,
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def scroll_all(self) -> list[dict]:
        try:
            items: list[dict] = []
            next_offset = None
            while True:
                points, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=256,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                items.extend(point.payload or {} for point in points)
                if next_offset is None:
                    break
            return items
        except Exception:
            return []

    def search(self, query_vector: np.ndarray, limit: int) -> list[rest.ScoredPoint]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return list(response.points)