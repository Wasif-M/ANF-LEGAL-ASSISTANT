from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re
import logging

import numpy as np
from rank_bm25 import BM25Okapi

from .document_catalog import DocumentCatalog
from .embeddings import Embedder
from .models import DocumentChunk, RetrievedChunk
from .prompts import INTENT_COMPARISON, classify_query_intent
from .storage import QdrantChunkStore
from .text_normalization import tokenize_for_search
from .utils import (
    chunk_metadata,
    chunk_text_matches_section_number,
    normalize_doc_blob,
    normalize_section_number,
    section_number_variants,
)

logger = logging.getLogger(__name__)

# ─── Shared section number pattern (must match hierarchy.py) ───
_SEC_NUM = r"\d+(?:\.\d+)*(?:[A-Z]{1,3})?(?:[-_][A-Za-z]{1,6})?(?:\([a-z\d]+\))*"

# Long queries (e.g. pasted statutes): avoid grabbing unrelated section numbers from the body
_LONG_QUERY_CHAR_THRESHOLD = 520
_QUERY_TAIL_WINDOW = 520

# Above this length a query is treated as a scenario/prose question, not a terse
# section lookup, so keyword-less number matching ("19", "20B") is disabled.
_TERSE_QUERY_CHAR_MAX = 90

# A query that asks for "(the) section/provision/article … of/under … <Act>" is
# DIRECTED at a named instrument even when it is long and carries no section number
# ("quote the section of the Control of Narcotic Substances Act under which I may
# appeal"). That framing — a provision-seeking noun bound to the act by of/under/in —
# distinguishes a deliberate lookup from a scenario that merely mentions an Act in
# passing, so document targeting is honoured for it regardless of length.
_DOC_DIRECTED_RE = re.compile(
    r"\b(?:sections?|provisions?|articles?|clauses?|sub-?sections?|rules?|"
    r"chapters?|parts?|schedules?)\b\s+(?:no\.?\s+|number\s+)?(?:of|under|in|from)\b",
    re.IGNORECASE,
)

# Words that carry no TOPIC signal when ranking sections WITHIN a targeted document:
# provision-type nouns, instrument-type words, question/polite filler, bare grammar.
# Stripping them (plus the Act's own name tokens) leaves the substantive terms — e.g.
# "appeal", "order", "court", "forfeiture" — that should pick the right section. Note
# "order" is deliberately NOT here: it doubles as an instrument-type word but a query
# about a judicial order genuinely needs it as a topic term.
_TOPIC_STOPWORDS = frozenset(
    {
        "the", "of", "under", "which", "may", "any", "for", "me", "you", "your",
        "quote", "tell", "what", "whats", "please", "give", "show", "cite", "about",
        "section", "sections", "provision", "provisions", "article", "articles",
        "clause", "clauses", "rule", "rules", "chapter", "chapters", "part", "parts",
        "schedule", "schedules", "sub", "subsection", "subsections",
        "act", "acts", "code", "ordinance", "law", "laws",
        "is", "are", "in", "on", "to", "by", "and", "or", "an", "that", "this",
        "from", "with", "shall", "can", "could", "would", "should", "must",
    }
)


def _tokenize(text: str) -> list[str]:
    return tokenize_for_search(text)


@dataclass(slots=True)
class HybridRetriever:
    store: QdrantChunkStore
    embedder: Embedder
    dense_top_k: int = 20
    hybrid_top_k: int = 15
    rerank_top_k: int = 10
    rrf_k: int = 60
    bm25_weight: float = 0.55
    dense_weight: float = 0.45
    reranker_model: str | None = None
    _chunks: list[DocumentChunk] = None  # type: ignore[assignment]
    _bm25: BM25Okapi | None = None
    _catalog: DocumentCatalog | None = None

    def __post_init__(self) -> None:
        self._chunks = []
        self._bm25 = None
        self._catalog = None
        self._refresh_indexes()

    def _refresh_indexes(self) -> None:
        payloads = self.store.scroll_all()
        self._chunks = [self._payload_to_chunk(payload) for payload in payloads]
        if self._chunks:
            tokenized = [_tokenize(chunk.text) for chunk in self._chunks]
            self._bm25 = BM25Okapi(tokenized)
            self._catalog = DocumentCatalog.from_chunks(self._chunks)
        else:
            self._bm25 = None
            self._catalog = None

    def _payload_to_chunk(self, payload: dict[str, Any]) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=str(payload.get("chunk_id", "")),
            document_id=str(payload.get("document_id", "")),
            source_path=str(payload.get("source_path", "")),
            text=str(payload.get("text", "")),
            section_path=tuple(payload.get("section_path") or ()),
            chunk_index=int(payload.get("chunk_index", 0)),
            start_char=int(payload.get("start_char", 0)),
            end_char=int(payload.get("end_char", 0)),
            metadata=dict(payload.get("metadata") or {}),
        )

    # ─── Core search methods ───

    def _dense_search(self, query: str, top_k: int | None = None) -> list[tuple[DocumentChunk, float]]:
        limit = top_k if top_k is not None else self.dense_top_k
        query_vector = self.embedder.embed_query(query)
        hits = self.store.search(query_vector, limit)
        results: list[tuple[DocumentChunk, float]] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append((self._payload_to_chunk(payload), float(hit.score)))
        return results

    def _lexical_search(self, query: str, top_k: int | None = None) -> list[tuple[DocumentChunk, float]]:
        limit = top_k if top_k is not None else self.dense_top_k
        if not self._bm25 or not self._chunks:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indices = np.argsort(scores)[::-1][: limit]
        return [(self._chunks[index], float(scores[index])) for index in ranked_indices if scores[index] > 0]

    def _fuse(
        self,
        dense_hits: list[tuple[DocumentChunk, float]],
        lexical_hits: list[tuple[DocumentChunk, float]],
        hybrid_top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        cap = hybrid_top_k if hybrid_top_k is not None else self.hybrid_top_k
        merged: dict[str, RetrievedChunk] = {}
        for rank, (chunk, score) in enumerate(dense_hits, start=1):
            entry = merged.setdefault(chunk.chunk_id, RetrievedChunk(chunk=chunk))
            entry.dense_score = score
            entry.fused_score += self.dense_weight / (self.rrf_k + rank)
        for rank, (chunk, score) in enumerate(lexical_hits, start=1):
            entry = merged.setdefault(chunk.chunk_id, RetrievedChunk(chunk=chunk))
            entry.lexical_score = score
            entry.fused_score += self.bm25_weight / (self.rrf_k + rank)
        ranked = sorted(merged.values(), key=lambda item: item.fused_score, reverse=True)
        logger.info(f"Fused (BM25={self.bm25_weight:.0%}, Dense={self.dense_weight:.0%}):")
        for rank, item in enumerate(ranked[:5], start=1):
            logger.info(f"   {rank}. fused={item.fused_score:.6f} (dense={item.dense_score:.4f}, lex={item.lexical_score:.1f})")
        return ranked[:cap]

    def _document_blob_for_match(self, chunk: DocumentChunk) -> str:
        meta = chunk_metadata(chunk)
        stem = Path(chunk.source_path).stem
        display = meta.get("display_title") or meta.get("title", "")
        aliases = " ".join(meta.get("search_aliases") or [])
        blob = f"{display} {aliases} {meta.get('file_name', '')} {stem}"
        return normalize_doc_blob(blob)

    @staticmethod
    def _canonical_doc_keywords(canonical: str) -> list[str]:
        normalized = normalize_doc_blob(canonical)
        return [w for w in re.findall(r"[a-z0-9]+", normalized) if len(w) > 2]

    def _document_name_matches_canonical(self, canonical: str, chunk: DocumentChunk) -> bool:
        """True if chunk's path/title matches the canonical act name (hyphen/space tolerant, token-based)."""
        words = self._canonical_doc_keywords(canonical)
        if not words:
            return False
        blob = self._document_blob_for_match(chunk)
        hits = sum(1 for w in words if w in blob)
        need = max(2, (2 * len(words) + 2) // 3)
        return hits >= need

    def _fallback_chunks_for_named_act_section(
        self, canonical: str, target_section: str
    ) -> list[RetrievedChunk]:
        """Full-corpus scan when fusion pool has no chunks from the named Act (BM25 + vector pool miss)."""
        found: list[RetrievedChunk] = []
        if not self._chunks:
            return found
        for chunk in self._chunks:
            if not self._document_name_matches_canonical(canonical, chunk):
                continue
            metadata = chunk_metadata(chunk)
            sn = metadata.get("section_number")
            if sn and normalize_section_number(str(sn)) == target_section:
                found.append(RetrievedChunk(chunk=chunk, fused_score=5_000.0, lexical_score=100.0, dense_score=0.0))
                continue
            if chunk_text_matches_section_number(chunk.text, target_section):
                found.append(RetrievedChunk(chunk=chunk, fused_score=500.0, lexical_score=10.0, dense_score=0.0))
        found.sort(key=lambda x: x.fused_score, reverse=True)
        return found[: max(self.rerank_top_k, self.hybrid_top_k)]

    def _salient_topic_tokens(self, query: str, canonical: str, expansion_terms: str | None) -> list[str]:
        """Substantive query terms for ranking sections WITHIN a targeted document:
        drop the Act's own name tokens (they appear on every chunk and bias toward
        title/preamble pages) and generic provision/filler words, leaving the topic
        ("appeal", "order", "court", …). Expansion terms add statutory synonyms."""
        doc_tokens = set(self._canonical_doc_keywords(canonical))
        text = f"{query} {expansion_terms or ''}"
        tokens: list[str] = []
        for tok in re.findall(r"[a-z]+", text.lower()):
            if len(tok) <= 2 or tok in _TOPIC_STOPWORDS or tok in doc_tokens:
                continue
            tokens.append(tok)
        return tokens

    def _recover_document_chunks_by_topic(
        self, canonical: str, query: str, expansion_terms: str | None
    ) -> list[RetrievedChunk]:
        """Full-document scan for the section that best matches the query TOPIC, used
        when a query is directed at a named Act but pins no section number (e.g. "the
        section of the CNS Act under which I may appeal"). The right provision's body
        often never enters the capped dense/BM25 pool, so we rank the whole target
        document by its salient topic terms and inject the top matches. A section whose
        HEADING carries a topic term ("48 Appeal") is favoured over a body that merely
        mentions it in passing."""
        if not self._bm25 or not self._chunks:
            return []
        topic_tokens = self._salient_topic_tokens(query, canonical, expansion_terms)
        if not topic_tokens:
            return []
        topic_set = set(topic_tokens)
        scores = self._bm25.get_scores(topic_tokens)
        scored: list[tuple[DocumentChunk, float]] = []
        for idx, chunk in enumerate(self._chunks):
            base = float(scores[idx])
            if base <= 0:
                continue
            if not self._document_name_matches_canonical(canonical, chunk):
                continue
            # Heading match is a strong signal the section IS about the topic, not just
            # mentioning it — "48 Appeal" should beat a body that references appeals.
            heading = " ".join(str(s) for s in chunk.section_path).lower()
            if any(t in heading for t in topic_set):
                base *= 2.0
            scored.append((chunk, base))
        if not scored:
            return []
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[0][1]
        recovered: list[RetrievedChunk] = []
        # Scale into a band (≈20–50) that dominates the tiny post-fusion scores of the
        # title-page stubs that survived filtering, while preserving topic ordering.
        for chunk, sc in scored[: max(self.rerank_top_k, self.hybrid_top_k)]:
            recovered.append(
                RetrievedChunk(chunk=chunk, fused_score=20.0 + 30.0 * (sc / top), lexical_score=sc, dense_score=0.0)
            )
        logger.info(
            f"Topic recovery for '{canonical}' [{' '.join(topic_tokens[:6])}]: "
            f"injected {len(recovered)} chunk(s); top heading "
            f"'{' > '.join(recovered[0].chunk.section_path)[:60]}'"
        )
        return recovered

    def _filter_candidates_by_named_instrument(
        self,
        candidates: list[RetrievedChunk],
        canonical: str,
        query_section: str | None,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return candidates
        filtered = [c for c in candidates if self._document_name_matches_canonical(canonical, c.chunk)]
        if filtered:
            logger.info(f"Named instrument filter '{canonical}': {len(candidates)} → {len(filtered)} candidates")
            # Guarantee the actual provision wins: the body chunk for the target
            # section may not have made the dense/BM25 top pool (a contents-page stub,
            # or a *different* section that merely mentions "154" in its text, can
            # outscore it). Run a full-corpus scan for this document+section and apply
            # its authoritative scores — UPGRADING chunks already in the pool (they may
            # be sitting at a near-zero fusion score) and adding any that are missing.
            if query_section:
                by_id = {c.chunk.chunk_id: c for c in filtered}
                for rec in self._fallback_chunks_for_named_act_section(canonical, query_section):
                    existing = by_id.get(rec.chunk.chunk_id)
                    if existing is not None:
                        existing.fused_score = max(existing.fused_score, rec.fused_score)
                    else:
                        filtered.append(rec)
                        by_id[rec.chunk.chunk_id] = rec
            filtered.sort(key=lambda x: x.fused_score, reverse=True)
            return filtered
        if query_section:
            recovered = self._fallback_chunks_for_named_act_section(canonical, query_section)
            if recovered:
                logger.info(
                    f"Named instrument '{canonical}' absent from fusion top pool; recovered {len(recovered)} chunk(s) via scan"
                )
                return recovered
        logger.warning(f"No candidates matched named instrument '{canonical}'; keeping unrestricted pool")
        return candidates

    def _query_tokens_for_doc_match(self, query: str) -> set[str]:
        return {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}

    def _document_query_token_overlap(self, query: str, chunk: DocumentChunk) -> float:
        """Generic overlap between query tokens and document title / filename (no act-specific rules)."""
        q_tokens = self._query_tokens_for_doc_match(query)
        if not q_tokens:
            return 0.0
        meta = chunk_metadata(chunk)
        stem = Path(chunk.source_path).stem
        blob = f"{meta.get('title', '')} {stem}"
        blob = blob.lower().replace("_", " ").replace("-", " ")
        d_tokens = {t for t in re.findall(r"[a-z0-9]+", blob) if len(t) > 2}
        if not d_tokens:
            return 0.0
        inter = q_tokens & d_tokens
        if not inter:
            return 0.0
        return min(1.0, len(inter) / max(3.0, len(q_tokens) ** 0.5))

    def _apply_document_query_affinity(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """When the same section number exists in several laws, prefer chunks whose title/path matches query words."""
        for cand in candidates:
            overlap = self._document_query_token_overlap(query, cand.chunk)
            if overlap > 0:
                cand.fused_score *= 1.0 + 0.55 * overlap
        candidates.sort(key=lambda x: x.fused_score, reverse=True)
        return candidates

    def _query_window_for_loose_section_patterns(self, query: str) -> str:
        """For long pasted queries, only scan the tail for implicit section numbers."""
        if len(query) <= _LONG_QUERY_CHAR_THRESHOLD:
            return query
        return query[-_QUERY_TAIL_WINDOW:]

    def _expand_query_for_dense_section(self, query: str, section: str | None) -> str:
        """Append neutral anchors for dense embedding search (headings often use '72.' not the user's phrasing)."""
        if not section:
            return query
        return f"{query}\nsection {section}\narticle {section}\n{section}."

    @staticmethod
    def _augment_query(text: str, expansion_terms: str | None) -> str:
        """Append generic query-understanding terms (statutory synonyms / anchors) to
        the text used for embedding + BM25 only. The original `query` is still used for
        section/document extraction, so terse-query heuristics are unaffected."""
        if not expansion_terms:
            return text
        return f"{text}\n{expansion_terms}"

    # ─── Main search orchestration ───

    def search(self, query: str, expansion_terms: str | None = None) -> list[RetrievedChunk]:
        # Extract section number from query if present
        query_section = self._extract_section_from_query(query)
        # Extract document context from query if present
        query_document = self._extract_document_context(query)

        # Document targeting is only trustworthy for concise, direct questions. In a
        # long scenario question an Act/agency name (e.g. "ANF", "ANF Act 1997",
        # "Punjab police rules 1975") is usually mentioned in passing — hard-filtering
        # to that one document drops the other laws the scenario actually needs. So
        # only honour document targeting when the query is terse OR pins an explicit
        # section number.
        is_terse = len(query.strip()) <= _TERSE_QUERY_CHAR_MAX
        is_doc_directed = bool(query_document) and self._query_is_document_directed(query)
        target_document = query_document if (is_terse or query_section or is_doc_directed) else None
        if query_document and not target_document:
            logger.info(f"   Ignoring document target '{query_document}' for long scenario query")
        elif is_doc_directed and not is_terse and not query_section:
            logger.info(f"   Honouring document target '{query_document}': query is directed at a named instrument")

        logger.info(f"🔎 SEARCH INITIATED")
        logger.info(f"   Query: {query}")
        logger.info(f"   Target section: {query_section}")
        logger.info(f"   Target document: {target_document}")
        query_document = target_document

        # A clause-level reference ("9c", "9(c)", "14(1)(a)") is stored in the index
        # as its PARENT section; remap to the stored form so the lookup doesn't
        # come back empty. The exact form is tried first, so a genuinely distinct
        # lettered section ("20B") is never collapsed to its base number.
        if query_section:
            resolved_section = self._resolve_section_variant(query_section, query_document)
            if resolved_section != query_section:
                logger.info(
                    f"   Section '{query_section}' not stored; using stored variant '{resolved_section}'"
                )
                query_section = resolved_section
        
        dense_query = self._augment_query(
            self._expand_query_for_dense_section(query, query_section), expansion_terms
        )
        if expansion_terms:
            logger.info(f"   Query expansion applied: {expansion_terms[:160]}")
        widen_dense = self.dense_top_k
        widen_lex = self.dense_top_k
        widen_hybrid = self.hybrid_top_k
        if query_section and query_document and classify_query_intent(query) != INTENT_COMPARISON:
            widen_dense = max(widen_dense, 48)
            widen_lex = max(widen_lex, 48)
            widen_hybrid = max(widen_hybrid, 36)
        elif query_section and not query_document:
            widen_dense = max(widen_dense, 36)
            widen_lex = max(widen_lex, 36)

        dense_hits = self._dense_search(dense_query, top_k=widen_dense)
        logger.info(f"   Dense retrieval: {len(dense_hits)} candidates")
        
        base_lexical = self._expand_query_for_dense_section(query, query_section) if query_section else query
        lexical_query = self._augment_query(base_lexical, expansion_terms)
        lexical_hits = self._lexical_search(lexical_query, top_k=widen_lex)
        logger.info(f"   Lexical retrieval: {len(lexical_hits)} candidates")
        
        candidates = self._fuse(dense_hits, lexical_hits, hybrid_top_k=widen_hybrid)
        logger.info(f"   Fused: {len(candidates)} candidates")
        
        # Log initial ranking
        logger.info(f"   Initial ranking (before filtering):")
        for i, cand in enumerate(candidates[:5], 1):
            metadata = chunk_metadata(cand.chunk)
            logger.info(f"      {i}. Score={cand.fused_score:.4f}, Doc: {metadata.get('title', 'N/A')}, Text: {cand.chunk.text[:60]}...")
        
        if query_document and classify_query_intent(query) != INTENT_COMPARISON:
            candidates = self._filter_candidates_by_named_instrument(candidates, query_document, query_section)
        
        # Apply SOFT document boosting when a specific document is targeted
        if query_document:
            logger.info(f"🔍 APPLYING DOCUMENT BOOST for {query_document}")
            candidates = self._boost_document_matches(candidates, query_document)
            candidates = self._boost_related_group_documents(candidates, query_document)

        # Targeted document but NO section number: the right provision's body is often
        # missing from the capped pool (title/preamble pages echo the Act name in the
        # query and crowd it out). Recover it by ranking the whole target document on
        # the query's salient TOPIC terms and injecting the best matches.
        if query_document and not query_section and classify_query_intent(query) != INTENT_COMPARISON:
            recovered = self._recover_document_chunks_by_topic(query_document, query, expansion_terms)
            if recovered:
                by_id = {c.chunk.chunk_id: c for c in candidates}
                for rec in recovered:
                    existing = by_id.get(rec.chunk.chunk_id)
                    if existing is not None:
                        existing.fused_score = max(existing.fused_score, rec.fused_score)
                    else:
                        candidates.append(rec)
                        by_id[rec.chunk.chunk_id] = rec
                candidates.sort(key=lambda x: x.fused_score, reverse=True)

        # Apply section-aware filtering and boosting if a specific section is targeted
        if query_section:
            logger.info(f"🔍 APPLYING SECTION FILTERING for section {query_section}")
            
            # Filter by keyword to prioritize chunks containing the section number
            candidates = self._filter_by_section_keyword(candidates, query_section)
            
            logger.info(f"🚀 APPLYING SECTION BOOST")
            candidates = self._boost_section_matches(candidates, query_section)
            
            # Apply parent-child boosting
            logger.info(f"🚀 APPLYING PARENT-CHILD BOOST")
            candidates = self._boost_parent_child_matches(candidates, query_section)
            
            # Apply amendment-aware boosting
            logger.info(f"🚀 APPLYING AMENDMENT-AWARE BOOST")
            candidates = self._boost_amendment_matches(candidates, query_section)
            
            logger.info(f"   Final ranking (after all filtering & boosting):")
            for i, cand in enumerate(candidates[:5], 1):
                metadata = chunk_metadata(cand.chunk)
                logger.info(f"      {i}. Score={cand.fused_score:.4f}, Doc: {metadata.get('title', 'N/A')}")
            
            candidates = self._apply_document_query_affinity(query, candidates)
        else:
            logger.info(f"⚠️  No section number found in query")
        
        want_multi_doc = (not query_section) or classify_query_intent(query) == INTENT_COMPARISON
        if want_multi_doc:
            candidates = self._ensure_multi_document_coverage(candidates, self.rerank_top_k)
        else:
            logger.info("Skipping multi-document coverage (section-specific lookup; use comparison query for forced breadth)")
        
        result = candidates[: self.rerank_top_k]
        logger.info(f"✅ Returning top {len(result)} results")
        return result

    # ─── Document context extraction ───

    @staticmethod
    def _query_is_document_directed(query: str) -> bool:
        """True when the query explicitly asks for a provision OF/UNDER a named Act
        (e.g. "the section of the CNS Act under which …"), as opposed to a scenario
        that mentions an Act only in passing. Used to honour document targeting for
        such queries even when they are long and carry no section number."""
        return bool(_DOC_DIRECTED_RE.search(query))

    def _extract_document_context(self, query: str) -> str | None:
        """Match query text against aliases discovered at ingest time."""
        if not self._catalog:
            return None
        matched = self._catalog.match_query(query)
        if matched:
            logger.debug(f"   Extracted document context: {matched}")
        return matched

    def _boost_related_group_documents(
        self, candidates: list[RetrievedChunk], target_doc: str
    ) -> list[RetrievedChunk]:
        """Soft-boost volumes and amendment siblings in the same document group."""
        if not self._catalog or not target_doc:
            return candidates
        related_ids = set(self._catalog.related_document_ids(target_doc))
        if not related_ids:
            return candidates
        boosted = 0
        for candidate in candidates:
            if candidate.chunk.document_id in related_ids:
                candidate.fused_score *= 1.35
                boosted += 1
        if boosted:
            logger.info(f"Boosted {boosted} chunk(s) from related volumes/amendments of '{target_doc}'")
            candidates.sort(key=lambda x: x.fused_score, reverse=True)
        return candidates
    
    def _boost_document_matches(self, candidates: list[RetrievedChunk], target_doc: str) -> list[RetrievedChunk]:
        """Soft-boost candidates from the target document instead of hard filtering.
        
        This ensures cross-document chunks still appear in results but target document
        chunks are ranked higher.
        """
        if not target_doc:
            return candidates
        
        boosted_count = 0
        
        for candidate in candidates:
            if self._document_name_matches_canonical(target_doc, candidate.chunk):
                old_score = candidate.fused_score
                candidate.fused_score *= 2.0  # 2x boost for target document
                boosted_count += 1
                logger.debug(f"   ✅ Document boost: {old_score:.4f} → {candidate.fused_score:.4f}")
        
        if boosted_count:
            logger.info(f"✅ Boosted {boosted_count} chunk(s) from target document: {target_doc}")
        
        # Re-sort by boosted scores
        candidates.sort(key=lambda x: x.fused_score, reverse=True)
        return candidates

    # ─── Multi-document coverage guarantee ───
    
    def _ensure_multi_document_coverage(self, candidates: list[RetrievedChunk], max_results: int) -> list[RetrievedChunk]:
        """Ensure results include chunks from ALL relevant documents, not just the top-scored one.
        
        Algorithm:
        1. First pass: take best chunk per document (guarantees coverage)
        2. Second pass: fill remaining slots with highest-scored chunks
        """
        if not candidates:
            return candidates
        
        # Track per-document best chunks
        seen_docs: dict[str, list[RetrievedChunk]] = {}
        for cand in candidates:
            doc_id = cand.chunk.document_id
            if doc_id not in seen_docs:
                seen_docs[doc_id] = []
            seen_docs[doc_id].append(cand)
        
        # If only one document, no coverage issue
        if len(seen_docs) <= 1:
            return candidates[:max_results]

        # First: take best chunk from each document — but only for documents that
        # are actually competitive. Reserving a slot for every document in the pool
        # (even weak, off-topic ones) pushes out the strong chunks of the correct
        # document and makes the model cite the wrong file. Require a document's best
        # chunk to score within a fraction of the overall top score to earn a slot.
        top_score = max((c.fused_score for c in candidates), default=0.0)
        coverage_floor = top_score * 0.45
        final: list[RetrievedChunk] = []
        used_ids: set[str] = set()

        for doc_id, doc_chunks in seen_docs.items():
            best = doc_chunks[0]  # Already sorted by score
            if best.fused_score < coverage_floor:
                continue
            final.append(best)
            used_ids.add(best.chunk.chunk_id)
        
        # Second: fill remaining slots with highest-scored unused chunks
        remaining = max_results - len(final)
        if remaining > 0:
            for cand in candidates:
                if cand.chunk.chunk_id not in used_ids:
                    final.append(cand)
                    used_ids.add(cand.chunk.chunk_id)
                    remaining -= 1
                    if remaining <= 0:
                        break
        
        # Re-sort final list by score
        final.sort(key=lambda x: x.fused_score, reverse=True)
        
        doc_count = len(set(c.chunk.document_id for c in final[:max_results]))
        logger.info(f"📚 Multi-document coverage: {doc_count} document(s) in top {min(max_results, len(final))} results")
        
        return final[:max_results]

    # ─── Parent-child boosting ───

    def _boost_parent_child_matches(self, candidates: list[RetrievedChunk], query_section: str) -> list[RetrievedChunk]:
        """Boost parent section when querying subsection, and vice versa.
        
        Handles parent-child relationships:
        - Query "45" → boost both "45" and "45(1)", "45(2)"
        - Query "45(1)" → boost parent "45" and siblings "45(2)"
        """
        logger.info(f"🔍 Applying parent-child boosting")
        
        for candidate in candidates:
            metadata = chunk_metadata(candidate.chunk)
            section_num = metadata.get("section_number")
            parent_section = metadata.get("parent_section")
            section_hierarchy = metadata.get("section_hierarchy", [])
            
            if not section_num:
                continue
            
            # If querying parent section, boost subsections
            if query_section == parent_section or query_section in section_hierarchy:
                old_score = candidate.fused_score
                candidate.fused_score *= 1.8  # 80% boost for parent-child match
                logger.debug(f"   ✅ Parent-child boost for {section_num}: {old_score:.4f} → {candidate.fused_score:.4f}")
            
            # If querying subsection, boost parent sections
            if section_num.startswith(query_section) and section_num != query_section:
                old_score = candidate.fused_score
                candidate.fused_score *= 1.5  # 50% boost for hierarchical match
                logger.debug(f"   ✅ Hierarchical boost for {section_num}: {old_score:.4f} → {candidate.fused_score:.4f}")
        
        return candidates
    
    def _boost_amendment_matches(self, candidates: list[RetrievedChunk], query_section: str) -> list[RetrievedChunk]:
        """Boost sections that reference amendments or are amendments.
        
        If querying a section that has amendments, include amendment info.
        If querying an amendment, return original section + amendments.
        """
        logger.info(f"🔍 Applying amendment-aware boosting")
        
        for candidate in candidates:
            metadata = chunk_metadata(candidate.chunk)
            section_num = metadata.get("section_number")
            amendments_detected = metadata.get("amendments_detected", False)
            amendment_references = metadata.get("amendment_references", [])
            
            if not section_num:
                continue
            
            # Boost if this section contains amendments
            if amendments_detected:
                old_score = candidate.fused_score
                candidate.fused_score *= 1.6  # 60% boost for sections with amendments
                logger.debug(f"   ✅ Amendment boost for {section_num}: {old_score:.4f} → {candidate.fused_score:.4f}")
            
            # Boost if this section references the queried section
            if query_section in amendment_references:
                old_score = candidate.fused_score
                candidate.fused_score *= 1.4  # 40% boost
                logger.debug(f"   ✅ Amendment reference boost for {section_num}: {old_score:.4f} → {candidate.fused_score:.4f}")
        
        return candidates
    
    # ─── Section number handling ───

    def _corpus_has_section(self, target_section: str, canonical: str | None) -> bool:
        """True if any chunk (optionally restricted to the named act) stores this
        section number in its metadata, or carries it in its text."""
        for chunk in self._chunks:
            if canonical and not self._document_name_matches_canonical(canonical, chunk):
                continue
            sn = chunk_metadata(chunk).get("section_number")
            if sn and normalize_section_number(str(sn)) == target_section:
                return True
        return False

    def _resolve_section_variant(self, query_section: str, canonical: str | None) -> str:
        """Map a clause-level reference ("9(c)", "14(1)(a)") to the section form
        actually stored in the index, scoped to the named act when one was given."""
        for variant in section_number_variants(query_section):
            if self._corpus_has_section(variant, canonical):
                return variant
        return query_section

    @staticmethod
    def _looks_like_year(section: str) -> bool:
        """A bare 4-digit number in the 1500-2099 range is a statute year, not a section."""
        return bool(re.fullmatch(r"\d{4}", section)) and 1500 <= int(section) <= 2099

    def _extract_section_from_query(self, query: str) -> str | None:
        """Extract the target section/article/rule number from a query.

        Two safe sources only:
        1. An explicit "section/article/rule/sec/§ <num>" reference (the last one, so
           pasted statute text can't steal the number from the user's real question).
        2. A *pure reference* query like "20B" or "337-I" — i.e. once polite filler
           ("what is", "tell me about", ...) is stripped, the whole query is just the
           number.

        Everything else returns None. In particular we must NOT scan free prose for
        loose numbers: scenario questions are full of incidental numbers
        ("register no 19", "46 days", "12 kg", "8 packets") and statute years
        ("rule 1975", "Order 1984") that are not the target section.
        """
        # A stray space around the hyphenated letter-suffix ("411 -A", "411- A",
        # "411 - A") is a typo, not a token boundary — but it breaks _SEC_NUM
        # (whose suffix part allows no space), so the section would silently fail
        # to extract and all section targeting/boosting would be skipped. Glue it
        # back to the canonical "411-A". Suffix capped at 3 letters so we never
        # join a following word ("1979 - Article", "5 - crpc").
        query = re.sub(r"(?<=\d)\s*-\s*([A-Za-z]{1,3})\b", r"-\1", query)

        def _clean(num: str) -> str | None:
            num = num.strip()
            if self._looks_like_year(num):
                return None
            if re.fullmatch(r"\d", num) and int(num) <= 2:  # "1"/"2" are too noisy alone
                return None
            return normalize_section_number(num)

        # 1. explicit keyword reference — reject when a unit word follows (e.g.
        #    "rule for 46 days", "section of 12 kg") which signals an incidental number.
        unit = r"(?:days?|day|kg|kgs|grams?|gram|gm|gms|rupees?|rs|packets?|persons?|years?|months?|weeks?|hours?|witnesses?)"
        # tolerate filler/typo tokens between the keyword and the number:
        # "section no. 9", "section number 9", "section o 14(1)(a)" (typo for "of")
        filler = r"(?:(?:no|num|number|o|of)\.?\s+)?"
        explicit_patterns = [
            rf"(?:section|article|rule|sec)\s+{filler}({_SEC_NUM})(?!\s+{unit}\b)",
            rf"\bu/s\s*\.?\s*({_SEC_NUM})",
            rf"§\s*({_SEC_NUM})",
        ]
        last_explicit: str | None = None
        for pattern in explicit_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                cleaned = _clean(match.group(1))
                if cleaned:
                    last_explicit = cleaned
        if last_explicit:
            logger.debug(f"   Extracted section (explicit): {last_explicit}")
            return last_explicit

        # 2. pure-reference query: strip leading polite filler, then the remainder
        #    must be just the section token.
        stripped = re.sub(
            r"^(?:what(?:\s+is|'s|s)?|whats|tell\s+me\s+about|explain|define|describe|give\s+me|show\s+me|about|regarding)\s+",
            "",
            query.strip(),
            flags=re.IGNORECASE,
        ).strip()
        m = re.fullmatch(rf"(?:section|article|rule|sec)?\s*({_SEC_NUM})[.?!]?", stripped, re.IGNORECASE)
        if m:
            cleaned = _clean(m.group(1))
            if cleaned:
                logger.debug(f"   Extracted section (pure reference): {cleaned}")
                return cleaned

        # 3. leading "<number> of <document>" form: "339-A of code of criminal procedure",
        #    "175 of crpc". Only for TERSE lookup queries (a long scenario sentence that
        #    happens to start with a number — "3 of the accused were …" — must not match),
        #    with the number at the very START (after polite filler) followed by "of <word>".
        if len(query.strip()) <= _TERSE_QUERY_CHAR_MAX:
            m = re.match(rf"^(?:section|article|rule|sec)?\s*({_SEC_NUM})\s+of\s+[A-Za-z]", stripped, re.IGNORECASE)
            if m:
                cleaned = _clean(m.group(1))
                if cleaned:
                    logger.debug(f"   Extracted section (leading 'N of <doc>'): {cleaned}")
                    return cleaned

        logger.debug(f"   No section number found in query: {query}")
        return None
    
    def _filter_by_section_keyword(self, candidates: list[RetrievedChunk], target_section: str) -> list[RetrievedChunk]:
        """Prioritize chunks whose text or metadata clearly contains the target section/article."""
        exact_matches: list[RetrievedChunk] = []
        other_matches: list[RetrievedChunk] = []

        for candidate in candidates:
            metadata = chunk_metadata(candidate.chunk)
            found = False

            if chunk_text_matches_section_number(candidate.chunk.text, target_section):
                logger.info(f"   ✅ TEXT PATTERN MATCH for section {target_section} in {candidate.chunk.chunk_id}")
                exact_matches.append(candidate)
                found = True

            if not found:
                stored_section = metadata.get("section_number", "")
                if stored_section:
                    normalized_stored = normalize_section_number(str(stored_section))
                    if target_section == normalized_stored:
                        logger.info(f"   ✅ METADATA MATCH: section_number={stored_section} in {candidate.chunk.chunk_id}")
                        exact_matches.append(candidate)
                        found = True

            if not found:
                section_path = metadata.get("section_path", candidate.chunk.section_path)
                if isinstance(section_path, tuple):
                    section_iter = section_path
                else:
                    section_iter = section_path or []
                for sp in section_iter:
                    head = str(sp).strip()
                    if not head:
                        continue
                    m = re.match(rf"(?i)(?:section|article|rule)\s+({_SEC_NUM})\b", head)
                    if m and normalize_section_number(m.group(1)) == target_section:
                        logger.info(f"   ✅ PATH HEADING MATCH: '{sp}' in {candidate.chunk.chunk_id}")
                        exact_matches.append(candidate)
                        found = True
                        break
                    m2 = re.match(rf"^({_SEC_NUM})\s", head)
                    if m2 and normalize_section_number(m2.group(1)) == target_section:
                        logger.info(f"   ✅ PATH NUMERIC MATCH: '{sp}' in {candidate.chunk.chunk_id}")
                        exact_matches.append(candidate)
                        found = True
                        break

            if not found:
                other_matches.append(candidate)

        if exact_matches:
            logger.info(f"✅ Found {len(exact_matches)} chunk(s) with exact section keyword/metadata match")
            exact_matches.sort(key=lambda x: x.fused_score, reverse=True)
            return exact_matches + other_matches
        logger.warning(f"⚠️  No chunks contain keyword for section {target_section}")
        return candidates

    def _boost_section_matches(self, candidates: list[RetrievedChunk], target_section: str) -> list[RetrievedChunk]:
        """Boost scores of chunks that match the target section number."""
        boosted = []
        found_match = False
        
        logger.info(f"🔍 Boosting for target section: {target_section}")
        
        for candidate in candidates:
            # Check if chunk has section metadata
            metadata = chunk_metadata(candidate.chunk)
            
            # Debug: Log what metadata we have
            logger.debug(f"   Chunk {candidate.chunk.chunk_id}:")
            logger.debug(f"      amendment_target_sections: {metadata.get('amendment_target_sections', [])}")
            logger.debug(f"      amendment_sections: {metadata.get('amendment_sections', [])}")
            logger.debug(f"      section_number: {metadata.get('section_number')}")
            
            matched = False
            boost_factor = 1.0
            
            # 1. Check for exact amendment target section match (highest priority)
            amendment_targets = metadata.get("amendment_target_sections", [])
            if target_section in amendment_targets:
                logger.info(f"   ✅ AMENDMENT TARGET MATCH: Section {target_section} in {candidate.chunk.chunk_id}")
                old_score = candidate.fused_score
                boost_factor = 3.0
                candidate.fused_score *= boost_factor
                logger.info(f"      Score: {old_score:.4f} → {candidate.fused_score:.4f} ({boost_factor}x boost)")
                boosted.append((candidate, True, boost_factor))
                found_match = True
                matched = True
                continue
            
            # 2. Check for section_number metadata (primary match for regular sections)
            section_num = metadata.get("section_number")
            if section_num:
                # Normalize stored section number for comparison
                normalized_stored = normalize_section_number(str(section_num))
                if target_section == normalized_stored:
                    logger.info(f"   ✅ SECTION NUMBER MATCH: Section {target_section} in {candidate.chunk.chunk_id}")
                    old_score = candidate.fused_score
                    boost_factor = 3.0
                    candidate.fused_score *= boost_factor
                    logger.info(f"      Score: {old_score:.4f} → {candidate.fused_score:.4f} ({boost_factor}x boost)")
                    boosted.append((candidate, True, boost_factor))
                    found_match = True
                    matched = True
                    continue
            
            # 3. Check for amendment sections (medium priority)
            amendment_sections = metadata.get("amendment_sections", [])
            for amend in amendment_sections:
                amend_section = amend.get("section") if isinstance(amend, dict) else amend
                normalized_amend = normalize_section_number(str(amend_section))
                if target_section == normalized_amend:
                    logger.info(f"   ✅ AMENDMENT MATCH: Section {target_section} in {candidate.chunk.chunk_id}")
                    old_score = candidate.fused_score
                    boost_factor = 2.5
                    candidate.fused_score *= boost_factor
                    logger.info(f"      Score: {old_score:.4f} → {candidate.fused_score:.4f} ({boost_factor}x boost)")
                    boosted.append((candidate, True, boost_factor))
                    found_match = True
                    matched = True
                    break
            
            if not matched:
                logger.debug(f"   ⚠️  No section match for {target_section}")
                boosted.append((candidate, False, 1.0))
        
        if not found_match:
            logger.warning(f"⚠️  No exact section matches found for section {target_section}")
            logger.warning(f"   This suggests metadata not properly extracted/stored")

        # Among chunks that matched the SAME section, prefer the one carrying the
        # actual provision text over a table-of-contents / heading stub. Many docs
        # list a section in their contents page (just the heading) and again in the
        # body (with the real text); both match the number, so without this the
        # stub can outrank the provision and the answer comes back "not found".
        for candidate, was_matched, _ in boosted:
            if not was_matched:
                continue
            body = candidate.chunk.text
            if len(body) >= 250 or re.search(r"\(\d+\)|\bshall\b|\bmay\b|\bmeans\b", body, re.IGNORECASE):
                candidate.fused_score *= 1.4

        # Re-sort by boosted score
        boosted.sort(key=lambda x: x[0].fused_score, reverse=True)
        return [item[0] for item in boosted]