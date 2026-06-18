# A Section-Aware Hybrid Retrieval-Augmented Generation System for Heterogeneous Pakistani Legal Corpora

**A systems paper on robust statute lookup, citation fidelity, and hallucination control over noisy PDF law.**

---

## Abstract

We present the design, implementation, and empirical hardening of a retrieval-augmented generation (RAG) system for question answering over a heterogeneous corpus of Pakistani primary legislation — penal, procedural, evidentiary, narcotics-control, anti-money-laundering, civil-service, and police-rules instruments. The corpus consists of 19 source PDFs that vary widely in structure, drafting convention (sections vs. articles vs. rules), amendment history, and OCR/extraction quality. We describe an ingestion-to-response pipeline built on a CPU-only stack: `pypdf` extraction with statute-specific text repair, a hierarchy-aware recursive chunker that preserves section boundaries, dense embeddings from `BAAI/bge-base-en-v1.5` served through `fastembed` (ONNX), a local Qdrant vector store, and a hybrid retriever fusing dense similarity with BM25 lexical scores via Reciprocal Rank Fusion (RRF) augmented by section- and document-aware boosting. Generation is performed by an instruction-tuned LLM under intent-specific prompt templates with strict grounding constraints. The system indexes 10,676 chunks. Rather than reporting a single end-to-end accuracy number, we document a *diagnostic, case-driven evaluation*: a sequence of real failure modes uncovered during deployment — answer truncation, clause collapse, section-number relabeling/hallucination, spelling-sensitive document matching, hyphen/case section-number mismatches, page-footer phantom sections, and cross-reference number theft — and the targeted fixes that resolved each, verified live against the running service. We argue that for high-stakes legal QA, *citation fidelity and honest abstention* matter more than fluency, and that the dominant error source is not the model but the ingestion and retrieval substrate operating over noisy, inconsistently formatted statutory text.

---

## 1. Introduction

Legal question answering is an attractive application of RAG: the answers must be grounded in authoritative text, the corpus is finite and curated, and users (lawyers, law-enforcement officers, students) ask both *lookup* questions ("what is Section 5 of the Anti-Money-Laundering Act?") and *analytical* ones ("what is the punishment for contravention of Section 7?"). But legal corpora — particularly digitised Pakistani statutes — are unusually hostile to naïve RAG:

1. **Heterogeneous structure.** Different instruments number their provisions as *Sections* (Pakistan Penal Code), *Articles* (Qanun-e-Shahadat Order, 1984), or *Rules* (Punjab Police Rules), with suffixed numbers (`35-A`, `302B`, `337-III`), decimal numbering, and deeply nested sub-clauses.
2. **Amendment layering.** A base Act, its first amendment, and a later amendment Act may all coexist in the corpus and amend overlapping sections.
3. **Extraction noise.** PDF text extraction injects mid-word spaces ("St ate", "th e"), run-on tokens, page footers ("3 | Page"), dotted-leader tables of contents, and arbitrary line wrapping.
4. **High cost of error.** A confidently wrong citation, or a fabricated provision, is worse than an honest "not found."

This paper reports a complete, working system and — more importantly — a candid account of *why first-cut RAG fails on this data and how each failure was diagnosed and fixed*. The contributions are:

- A **section-aware ingestion and chunking** method that survives statute-specific PDF noise and preserves provision boundaries and numbering.
- A **hybrid retriever** combining dense and lexical signals with legal-specific section/document boosting and a full-corpus fallback that guarantees the correct provision surfaces for a named-Act + section query.
- A **typo- and format-tolerant addressing layer**: fuzzy document-name matching and a canonical section-number normaliser so that `35A`, `35-A`, `35a`, and `35-a` all resolve identically, and "qanun"/"qanuan"/"qanoon" all resolve to the same instrument.
- A **grounding-first generation protocol** that quotes provisions verbatim and in full, refuses to relabel one section's text as another, and abstains when the requested provision is absent.
- A **diagnostic evaluation methodology** for legal RAG centred on citation fidelity and honest abstention, with reproducible query→response observations.

---

## 2. Related Work (Brief)

The system synthesises well-established components rather than inventing new model architectures. Dense passage retrieval with bi-encoders, BM25 lexical retrieval, and their late fusion via Reciprocal Rank Fusion (Cormack et al., 2009) are standard. The BGE family of embedding models (Xiao et al., 2023) provides strong general-purpose English retrieval embeddings. RAG (Lewis et al., 2020) frames the generate-from-retrieved-context paradigm. Our work is positioned as an *engineering and evaluation* study: the novelty is in the legal-domain ingestion/retrieval hardening and the grounding protocol, not in new neural components.

---

## 3. Corpus and Data Characterisation

The corpus comprises **19 documents** spanning seven legal domains. All are English-language Pakistani primary or subordinate legislation, supplied as PDFs of varying provenance and scan quality. After ingestion the corpus yields **10,676 chunks**. Table 1 lists the documents and their chunk counts.

**Table 1. Corpus composition (post-ingestion chunk counts).**

| # | Document | Domain | Chunks |
|---|----------|--------|-------:|
| 1 | ANF Act 1997 | Narcotics enforcement | 43 |
| 2 | Anti-Money-Laundering Act 2010 (amended up to Sep. 2020) | AML/CFT | 207 |
| 3 | Civil Servants Act, 1973 | Civil service | 160 |
| 4 | Code of Criminal Procedure 1898 | Procedure | 1,414 |
| 5 | Control of Narcotic Substances (First Amendment) Act 2020 | Narcotics (amendment) | 10 |
| 6 | Control of Narcotic Substances Act 1997 | Narcotics (base) | 102 |
| 7 | Control Substances (Regular Drugs of Abuse) | Narcotics | 92 |
| 8 | Control of Narcotic Substances Amendment Act 2022 | Narcotics (amendment) | 104 |
| 9 | Dangerous Drugs Act 1930 | Narcotics | 67 |
| 10 | Disposal of Vehicles and Other Articles | Procedure | 25 |
| 11 | Inter-Agency Force Task | Policy | 6 |
| 12 | Pakistan Penal Code | Criminal | 832 |
| 13 | Punjab Police Efficiency and Discipline Rules 1975 | Police rules | 980 |
| 14 | Punjab Police Rules I | Police rules | 2,973 |
| 15 | Punjab Police Rules II | Police rules | 1,218 |
| 16 | Punjab Police Rules III | Police rules | 1,765 |
| 17 | Qanun-e-Shahadat Order 1984 | Evidence | 306 |
| 18 | The National Anti-Narcotics Policy 2019 | Policy | 334 |
| 19 | The Prevention of Smuggling Act, 1977 | Customs/smuggling | 38 |
| | **Total** | | **10,676** |

**Heterogeneity dimensions that drive design.**

- **Numbering vocabulary:** *Section* (PPC, CrPC, AML), *Article* (Qanun-e-Shahadat), *Rule* (Punjab Police Rules). A user asking for "Section 6" of an Article-numbered instrument must still be served Article 6.
- **Suffix conventions:** `35-A`/`35-B`/`35-C` (Dangerous Drugs Act), `302B`/`337-III` (PPC), decimal `3.1.4` and `Rule 12.14` (police rules).
- **Amendment groups:** the four Control-of-Narcotic-Substances documents (base 1997, First Amendment 2020, Amendment 2022, plus a related "regular drugs of abuse" instrument) form one subject group with overlapping amended sections.
- **Extraction artefacts:** page footers rendered as `"3 | P a g e"`, dotted-leader TOC lines, mid-word spacing, and headings whose own number is immediately followed by a *cross-reference* to another section (e.g. `"13. Punishment for contravention of Section 7 ..."`).

These properties — not model capability — are the principal source of failure, and they motivate every component below.

---

## 4. System Architecture

The pipeline is a classic RAG dataflow with legal-specific hardening at each stage:

```
PDF/TXT sources
   │  pypdf extraction + statute text repair (Section 5)
   ▼
Section-aware recursive chunking (Section 6)  ──►  DocumentChunk + metadata
   │  fastembed BAAI/bge-base-en-v1.5 (768-d, ONNX, CPU)   (Section 7)
   ▼
Qdrant local vector store (cosine)            (Section 8)
   ▲                                   ┌───────────────────────────────┐
   │  dense top-k                      │  Query                        │
   └───────────────  Hybrid Retriever ◄┤  • section-number extraction  │  (Section 9)
        BM25 lexical top-k             │  • document-name match (fuzzy)│
        RRF fusion + boosting          └───────────────────────────────┘
   │  top-10 context
   ▼
Intent-aware prompt + grounding rules  ──►  LLM (streaming)  ──►  SSE response  (Section 10)
```

The service is a **FastAPI** application (`api.py`) exposing `/chat` (Server-Sent-Events streaming answer), `/query` (retrieval inspection), and `/health`. The retrieval/index stack is CPU-only and dependency-light (no PyTorch): embeddings run through `fastembed`'s ONNX runtime, lexical retrieval uses `rank_bm25`, and the vector store is an embedded Qdrant instance persisted on local disk. Generation calls an external instruction-tuned LLM (`gpt-4o-mini` by default) and streams tokens to the client.

---

## 5. Data Ingestion and Text Normalisation

Ingestion (`loaders.py`) discovers documents recursively, extracts text page-by-page with `pypdf`, and applies a normalisation cascade designed for statutory PDFs:

1. **Control-character and newline hygiene:** strip NULs, normalise CRLF→LF, de-hyphenate words split across line breaks (`-\n` before a lowercase letter), and collapse 3+ blank lines.
2. **Legal annotation removal:** strip footnote/amendment citations ("vide amendments … dated …"), bracketed cross-reference notes ("[see section 2 (xxvi)]"), and inline footnote markers appended to words (e.g. `held51` → `held`). The footnote-marker regex deliberately avoids `\w` before digits, because digits are word-characters and a careless pattern corrupts section headings like `103.`.
3. **PDF spacing repair (`fix_pdf_spacing`):** a line-by-line repair of run-on and over-split tokens. Critically, this step operates **per line so that newlines survive** — an earlier implementation joined on all whitespace and flattened every document to a single line, which destroyed the line structure the chunker relies on for heading detection.

Each document is wrapped in a `SourceDocument` carrying derived metadata: a humanised title, a `document_type` (base act / amendment / rules / policy) inferred from folder and title cues, a stable `document_group_id` (a subject key that collapses a base Act and its amendments into one group), search aliases, and — for amendments — an `amends_group_id` pointing at the base instrument. This metadata is what later enables amendment-aware retrieval and group boosting.

---

## 6. Section-Aware Chunking

Generic fixed-window chunking is inadequate for statutes: it severs provisions mid-clause and discards the section numbering that users actually query by. Our chunker (`chunking.py`, `hierarchy.py`) is **hierarchy-driven**.

### 6.1 Heading detection

`detect_heading` classifies each line as a heading or body. It recognises, in priority order: explicit `Section/Article/Rule/Schedule N` headings; `Chapter/Part/Schedule` labels; numeric headings (`13. …`, `3.1.4 …`, `337-A. …`); alphabetic and roman sub-headings; and ALL-CAPS / title-case standalone headings. Two guards reject *false* headings that otherwise poison the section index:

- **Table-of-contents guard (`_TOC_LINE`):** lines with dotted leaders or trailing page numbers ("79. Proof of execution …… 35") are not headings; otherwise every section is detected twice (once in the contents page, once in the body), producing phantom duplicate sections.
- **Page-footer guard (`_PAGE_MARKER`):** lines like `"3 | Page"` or `"Page 12"` (the extractor often spaces "Page" into `"P a g e"`) are rejected, so the leading page number is not mistaken for a section number. *(Added during hardening — Section 11.6.)*

### 6.2 Buffering and recursive splitting

The document is scanned line by line. On each heading the current buffer is *emitted* as a chunk (or chunks) tagged with the active section path, and the heading stack is updated by level. Body text accumulates until the next heading. When an emitted provision exceeds the maximum chunk size it is **recursively split** — paragraphs → sentences → words — with overlap, so long provisions (e.g. the AML Act's National Executive Committee section with sub-clauses (a)–(h)) are divided at natural boundaries rather than mid-sentence.

### 6.3 Section-number extraction

Each chunk records a `section_number` extracted from its heading. The extraction order is significant (and was a source of a serious bug, Section 11.7): **the heading's own leading number takes priority over any `Section X` cross-reference appearing later in the heading text.** The final order is (1) `Section/Article/Rule N` at the *start* of the heading, (2) a leading bare number, (3) a `Section X` reference anywhere as a last resort. Without this ordering, a heading such as `"13. Punishment for contravention of Section 7"` is mis-numbered as 7.

### 6.4 Chunking configuration

| Parameter | Value | Rationale |
|-----------|------:|-----------|
| `max_chars` | 2,000 | Accommodates long legal definitions while bounding context cost |
| `overlap_chars` | 250 | Preserves cross-boundary legal context |
| `min_chars` | 100 | Allows very short provisions (e.g. "20B. Confiscation.") |

Each `DocumentChunk` carries: `chunk_id`, `document_id`, `source_path`, the chunk text (prefixed with its `SECTION:` path), `section_path`, `section_number`, parent-section and hierarchy metadata, detected amendment markers, and all document-level metadata (title, type, group id, aliases).

---

## 7. Embedding

Dense embeddings are produced by **`BAAI/bge-base-en-v1.5`** (768-dimensional) served through `fastembed`'s ONNX runtime with the `CPUExecutionProvider` — there is no GPU or PyTorch dependency, which keeps the system deployable on commodity hardware. The model is loaded lazily and capped at a 512-token maximum length.

A key detail is the **asymmetric document/query embedding**: passages are embedded with `passage_embed` and queries with `query_embed`, matching BGE's training convention. Document embedding is batched (default batch size 16) with automatic batch-size back-off on failure. Embeddings are L2-comparable under cosine distance, which the vector store uses directly.

---

## 8. Vector Store

Vectors and full chunk payloads are persisted in an **embedded Qdrant** instance (`qdrant_data/`, cosine distance). Two implementation choices proved important:

- **True reset on full reindex.** `recreate_collection` does *not* reliably clear an existing collection in Qdrant's local mode — a smaller rebuild overwrites only the low point-IDs and orphans higher-index points from the previous build, silently mixing stale and duplicate vectors into a "fresh" index. The store therefore *deletes then creates* the collection to guarantee a clean rebuild.
- **Deterministic point IDs.** Each chunk's point ID is a UUID5 derived from its `chunk_id`, making upserts idempotent and incremental indexing safe.

The retriever also loads **all** payloads at startup (`scroll_all`) to build the in-memory BM25 index and the document catalog; vectors are not loaded into memory (search is delegated to Qdrant).

A single-writer caveat applies: Qdrant local mode permits only one client to hold the on-disk lock, so the indexing job and the API server cannot run concurrently. Operationally, a full `--force-reindex` requires stopping the API process first.

---

## 9. Hybrid Retrieval

Retrieval (`retrieval.py`) is the heart of the system. A query passes through the following stages.

### 9.1 Query analysis

- **Section-number extraction.** Two safe sources only: an explicit `section/article/rule/§ N` reference (the *last* one in the query, so pasted statute text cannot steal the number from the user's actual question), or a pure-reference query that *is* just a number after polite filler is stripped ("what is 20B" → `20B`). Bare years (1500–2099) and the noisy tokens "1"/"2" are rejected. Free-prose numbers ("12 kg", "register no 19") are deliberately *not* harvested.
- **Document-name extraction.** The query is matched against a `DocumentCatalog` built from ingested titles/aliases (Section 9.5).

Document targeting is honoured only for terse queries (≤ 90 chars) or queries that pin an explicit section number; in long scenario questions an Act name mentioned in passing must not hard-filter away the other instruments the scenario needs.

### 9.2 Dual retrieval

- **Dense.** The query (optionally expanded with neutral section anchors such as `"section 6 / article 6 / 6."` so dense search can match headings phrased differently from the user) is embedded and searched in Qdrant for the top-*k* (widened to as many as 48 candidates for section+document lookups).
- **Lexical.** A BM25-Okapi index over tokenised chunk text returns the top-*k* lexical matches. BM25 is essential for exact section-number and legal-term matching, where dense similarity alone is unreliable.

### 9.3 Reciprocal Rank Fusion

Dense and lexical rankings are fused with weighted RRF:

> fused(c) = Σ_r  w_r / (k + rank_r(c))

with `k = 60`, dense weight `0.45`, and BM25 weight `0.55`. The lexical weight is set slightly higher than the dense weight because keyword matches matter more for statute lookups than for open-domain QA.

### 9.4 Legal-specific boosting and the named-Act fallback

After fusion, when a document and/or section is targeted, a cascade of soft multiplicative boosts re-ranks candidates:

- **Document boost** (×2.0) for chunks from the named instrument, plus a group boost (×1.35) for sibling volumes and amendments.
- **Section keyword/metadata filtering** that promotes chunks whose text, `section_number` metadata, or section-path heading match the target number.
- **Section boost** (×3.0) for an exact `section_number` match; **amendment-target** (×3.0) and **amendment-section** (×2.5) boosts for amended provisions; **provision-quality** boost (×1.4) that prefers chunks carrying real provision text (containing `(1)`, "shall", "may", "means", or ≥ 250 chars) over table-of-contents/heading stubs sharing the same number.
- **Parent–child** (×1.8/×1.5) and **amendment-aware** (×1.6/×1.4) boosts for hierarchical and amendment relationships.

Crucially, a **full-corpus fallback** (`_fallback_chunks_for_named_act_section`) runs whenever a named Act + section is queried: it scans *all* chunks for the document+section, assigns authoritative scores (5,000 for a metadata match, 500 for a text match), and upgrades or inserts the provision even if dense+BM25 fusion never surfaced it. This guarantees that "Section N of <Act>" returns the real provision rather than a near-miss that merely mentions the number.

### 9.5 Document catalog and fuzzy addressing

The `DocumentCatalog` resolves an instrument name from free text in three stages: (1) verbatim alias substring; (2) distinctive-token overlap against title tokens; (3) acronym match (so "anti narcotics force" → a file identified by "anf"). To tolerate the spelling variation real users produce, stage 2 was extended (Section 11.5) with **fuzzy token matching** (Levenshtein-style ratio ≥ 0.78 via `difflib`) and **unique-token acceptance**: a query token that matches a title token *owned by exactly one document in the corpus* (e.g. "shahadat", "laundering", "narcotic") identifies that document on its own. A small block-list of generic structural words ("procedure", "evidence", "court", "code", …) is excluded from unique-token acceptance so topical queries are not hard-routed to a single instrument.

### 9.6 Multi-document coverage

For non-section, non-comparison queries the retriever guarantees coverage across documents (best chunk per competitive document, then highest-scored fill), with a coverage floor at 0.45× the top score so weak off-topic documents cannot steal result slots. For section-specific lookups this breadth is deliberately suppressed in favour of precision.

The retriever returns the top-10 chunks.

---

## 10. Generation and Response

### 10.1 Intent-aware prompting

A lightweight classifier (`classify_query_intent`) routes the query to one of six prompt templates — simple lookup, explanation, comparison, procedural, penalty, cross-reference — each with a tailored answer structure. For pasted-statute queries the classifier prefers the simple-lookup template even when penalty/explanation trigger words appear inside the pasted text.

### 10.2 Grounding and citation-fidelity rules

The system prompt and templates enforce a grounding-first contract specific to legal QA:

- **Verbatim, complete quotation.** Provisions are quoted exactly; when a provision has enumerated sub-clauses ((a),(b),(c)… or (1),(2),(3)…), *every* sub-clause is reproduced — never truncated with "…" or "etc.".
- **No relabeling.** Each excerpt carries its own number; the model must not present a different section's text under the requested number. A document "Article 6" satisfies a request for "Section 6" *only when the number matches*.
- **Trust-then-abstain.** If an excerpt's `Section Number` equals the requested number, the model answers from it (even when the excerpt sits under a generic parent heading like "COMMENTS" or cross-references other sections). If *no* excerpt matches, the model abstains with an explicit "the provided documents do not contain Section N of <Act>."
- **Amendment awareness.** When the excerpts include both a base Act and an amending Act of the same group, an "amendment notice" instructs the model to answer with the current amended text and reference the prior provision.

### 10.3 Context construction and rendering

The top-10 chunks are formatted with source, file, section path, and `Section Number` headers. Chunk text is passed through `reflow_provision_text`, which **rejoins PDF-wrapped lines into continuous sentences while preserving a line break before each enumerated clause** — this lets the front-end render one clause per line (via `remark-breaks`) without shattering sentences mid-line. The answer is streamed token-by-token over SSE (`max_tokens = 4000`, temperature 0.3) to the React client, which renders Markdown with section headings, an accent-bordered block-quote for the quoted provision, a bold citation, and a significance note.

---

## 11. Experiments: Diagnostic Evaluation and Iterative Hardening

### 11.1 Methodology and rationale

For a legal assistant, the meaningful evaluation questions are: *Does a section lookup return the correct provision? Is the quoted text complete and faithful? Does the system abstain rather than fabricate when the provision is absent? Is it robust to the spelling and formatting variation real users produce?* We therefore adopt a **diagnostic, case-driven evaluation**: probe queries spanning the corpus, observe the live `/chat` and `/query` responses, localise each failure to a specific pipeline stage, fix it, and re-verify. This is more informative for a high-stakes lookup system than a single aggregate score, because each failure class has a distinct cause and a distinct user-visible consequence (a truncated quote, a wrong citation, a fabricated provision). Every result below was observed against the running service.

Each subsection states the **symptom**, the **root cause** (located by offline re-chunking of the source PDF and by inspecting retrieved chunks via `/query`), the **fix**, and the **verification**.

### 11.2 Answer truncation

**Symptom.** Long provisions (e.g. AML Act §5, National Executive Committee) cut off mid-text. **Initial hypothesis (wrong).** The output token cap. The streamed generation was limited to 1,500 tokens; raising it to 4,000 did *not* resolve it. **True cause.** The simple-lookup template instructed a 2–4 sentence answer and "avoid elaboration," so the model deliberately abbreviated the clause list with "…". The clean, well-formed trailing "…" was the tell that this was a *prompt* behaviour, not a hard token cut (which truncates mid-word). **Fix.** A global "complete quotes" rule plus a template rewrite requiring all sub-clauses; the 4,000-token cap is retained because a full multi-clause quote genuinely needs the headroom. **Verification.** AML §5 returns clauses (a)–(h) in full.

*Lesson: distinguish model-behaviour truncation from infrastructure truncation by the shape of the cut.*

### 11.3 Clause collapse and sentence shattering (rendering)

**Symptom.** Enumerated clauses ran together on one line; after a line-break fix, sentences broke mid-line. **Cause.** Markdown collapses single newlines to spaces (clauses merge); enabling `remark-breaks` then turned *every* PDF line-wrap into a hard break (sentences shatter). **Fix.** `reflow_provision_text` normalises the context so newlines occur *only* at clause boundaries; `remark-breaks` then renders exactly one clause per line with intact sentences. **Verification.** AML §5 renders (a)–(e)… each on its own line, each a continuous sentence.

### 11.4 Section relabeling / hallucination

**Symptom.** "What is section 5 of *qanuan* e shahadat?" returned a fabricated "competency of witnesses" §5 with invented sub-clauses. **Cause.** The misspelling "qanuan" failed document matching; with no document target, a general section-5 search surfaced a *different* provision (Article 3, "Who may testify"), which the model relabeled as "Section 5." Retrieval for the *correctly spelled* query was already correct. **Fix.** A system-prompt "never relabel/renumber" rule plus a "verify the section number before answering" step; the model now abstains safely instead of fabricating. **Verification.** The typo query returns an explicit "not found"; correctly spelled lookups are unaffected.

*Lesson: for legal QA, honest abstention is a feature; the grounding contract must forbid number substitution explicitly.*

### 11.5 Spelling-sensitive document matching

**Symptom.** Only the exact spelling "Qanun-e-Shahadat" resolved; "qanuan", "qanoon", "quanun", "kanun", "shahdat" failed. **Cause.** Document matching required near-exact distinctive-token overlap; one corrupted token dropped coverage below threshold. **Fix.** Fuzzy token matching (`difflib` ratio ≥ 0.78) plus unique-token acceptance ("shahadat" alone identifies the instrument), with a generic-word block-list to prevent over-routing. **Verification.** Nine spelling variants of "qanun e shahadat" all resolve to the Qanun-e-Shahadat Order; Penal Code, CrPC, ANF, AML, CNSA still match; generic queries ("bail procedure", "rules of evidence") correctly return no forced document.

### 11.6 Page-footer phantom sections

**Symptom.** Dangerous Drugs Act §3/§10 returned "not found" while title queries worked. **Cause.** Page footers extracted as `"3 | P a g e"` / `"10 | P a g e"` were detected as numeric headings, creating phantom "section 3/10" chunks full of unrelated text that hijacked the number lookup. **Fix.** A `_PAGE_MARKER` guard in `detect_heading` rejects "N | Page" and "Page N" lines. **Verification (post-reindex).** Phantom footer chunks = 0; §3 maps cleanly to the real "Calculation of percentages in liquid preparations." A complementary prompt fix stopped the model over-abstaining when the (correct) excerpt sat under a "COMMENTS" parent heading.

### 11.7 Cross-reference number theft

**Symptom.** "Section 13 of dangerous drug act" → "not found," though the heading `"13. Punishment for contravention of Section 7"` plainly exists. **Cause.** The section-number extractor matched `Section X` *anywhere* in the heading first, so it stored the cross-referenced **7** instead of the heading's own **13**. This affected every punishment section (§10→4, §11→5, §13→7, §14→8, §19→9), rendering those numbers unlookup-able. **Fix.** Reorder extraction to prefer the heading's leading number over a mid-text cross-reference. **Verification (post-reindex).** §10/§11/§12/§13/§14/§19 each resolve to their own provision; §3/§35-A and other documents unaffected; absent §999 still abstains.

### 11.8 Hyphen/case section-number normalisation

**Symptom.** "Section 35A" → "not found," but "Section 35-A" worked. **Cause.** `normalize_section_number` produced *different* canonical forms for equivalent inputs (`35A`≠`35-A`), and lowercase was not unified. **Fix.** A single canonicaliser making separator and case irrelevant: `35A` = `35-A` = `35a` = `35-a` = `35 A` → `35A` (single-letter suffix joined; multi-letter hyphenated, e.g. `337-II`); applied to both the query and the stored value before comparison. **Verification.** All four spellings of §35-A return "Power of the Court to freeze assets."

### 11.9 Summary of evaluation outcomes

**Table 2. Representative probe queries after hardening (live `/chat`).**

| Query | Outcome |
|-------|---------|
| AML Act §5 | Full National Executive Committee provision, clauses (a)–(h) |
| Qanun-e-Shahadat §5 / §6 / §11 / §22 (by number) | Correct Articles; verbatim, complete |
| 9 spelling variants of "qanun e shahadat" | All resolve to the correct instrument |
| Dangerous Drugs Act §3, §10, §11, §13, §19 | Correct provisions (post-fix) |
| Dangerous Drugs Act §35A / §35-a / §35a / §35-A | All → §35-A "freezing of assets" |
| Section that does not exist (e.g. §999) | Honest abstention — no fabrication |
| Generic topical query ("bail procedure") | Broad retrieval, no spurious hard-routing |

---

## 12. Results and Discussion

The hardened system reliably (i) resolves section/article/rule lookups across heterogeneous numbering and spelling, (ii) reproduces provisions verbatim and in full, (iii) abstains rather than fabricating when a provision is absent, and (iv) routes queries to the correct instrument despite user misspellings.

The dominant finding is that **the ingestion and retrieval substrate, not the language model, governs answer quality in legal RAG.** Six of the seven failure classes above were caused by PDF extraction noise or addressing logic (page footers, cross-reference number theft, hyphen/case mismatch, spelling-sensitive matching, relabeling, clause/sentence formatting); only the truncation issue touched the generation layer, and even that was a *prompt* behaviour rather than a model limitation. A second finding is that **two of the fixes required re-indexing** (page-footer guard, cross-reference extraction) because they alter stored chunk metadata, whereas prompt and normalisation fixes took effect immediately — a useful operational distinction when triaging legal-RAG defects: *metadata bugs are index-time; behaviour bugs are query-time.*

The hybrid design is justified empirically: exact section-number and legal-term matching depend on BM25, while paraphrastic and conceptual matching depend on dense embeddings; RRF lets each contribute without one dominating. The named-Act full-corpus fallback is what converts "usually retrieves the right section" into "deterministically retrieves the right section when the Act and number are both specified" — the single most valuable retrieval feature for a lookup-heavy legal workload.

---

## 13. Limitations

- **No formal IR benchmark.** Evaluation is diagnostic and case-based, not a labelled precision/recall/MRR study. A gold set of (question, correct-provision) pairs would allow quantitative comparison of retrieval variants.
- **Residual extraction noise.** Some chunks retain mid-word spacing ("St ate", "th e") from the source PDFs; the model usually normalises these when quoting, but the underlying text is imperfect.
- **Hard cases in the worst-scanned files.** The Dangerous Drugs Act PDF contains pseudo-headings ("Military Court.") and footers mid-section; while §3/§10/§11/§13/§19 now resolve, fully robust heading detection for the most degraded documents remains open. A few year tokens ("1857", "1930") are stored as section numbers from mis-detected headings; they are harmless because the query parser rejects four-digit years.
- **Single-writer vector store.** Qdrant local mode precludes concurrent indexing and serving; a server-mode deployment would remove this and enable online updates.
- **No reranker.** A cross-encoder reranking stage is configured but disabled; it is the most obvious lever for further precision gains.
- **English-only.** Urdu-language statutes and queries are out of scope of the current embedding model.

---

## 14. Future Work

1. **Cross-encoder reranking** (e.g. a `fastembed` `TextCrossEncoder`) over the fused top-k for precision on ambiguous lookups.
2. **A labelled evaluation set** of section-lookup and analytical questions with gold provisions, enabling quantitative ablations (dense-only vs. BM25-only vs. hybrid; with/without fallback; with/without boosting).
3. **Structure-aware ingestion** using PDF layout signals (font size, position) rather than pure text heuristics, to eliminate the residual heading-detection failures.
4. **Server-mode Qdrant** with incremental, online indexing and payload-filtered search to push document/section filters into the vector store.
5. **Provision-graph linking** that materialises cross-references ("for the purposes of Section 2") as edges, enabling multi-hop legal questions.
6. **Bilingual support** for Urdu legal text.

---

## 15. Conclusion

We built and hardened a section-aware hybrid RAG system for a heterogeneous, noisy corpus of Pakistani legislation. The system spans ingestion with statute-specific text repair, hierarchy-preserving chunking, CPU-only BGE embeddings, an embedded Qdrant store, a hybrid dense+BM25 retriever with legal-specific boosting and a deterministic named-Act fallback, fuzzy and format-tolerant addressing, and a grounding-first generation protocol that quotes provisions in full and abstains rather than fabricating. Our central, transferable lesson is that high-stakes legal QA succeeds or fails at the *data and retrieval substrate*: the model is rarely the bottleneck, but PDF noise, inconsistent numbering, and lenient addressing are. Treating citation fidelity and honest abstention as first-class requirements — and diagnosing failures by the stage that produced them (index-time vs. query-time) — yields a system that answers section lookups correctly across the corpus while refusing to invent law it does not have.

---

## References

1. P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," *NeurIPS*, 2020.
2. S. Xiao, Z. Liu, P. Zhang, N. Muennighoff, "C-Pack / BGE: Packed Resources for General Chinese and English Embeddings," 2023.
3. G. V. Cormack, C. L. A. Clarke, S. Büttcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods," *SIGIR*, 2009.
4. S. Robertson, H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," *Foundations and Trends in Information Retrieval*, 2009.
5. Qdrant: Vector Similarity Search Engine. https://qdrant.tech
6. FastEmbed: Lightweight, fast embedding generation. https://github.com/qdrant/fastembed

---

### Appendix A. Key Configuration

| Component | Setting |
|-----------|---------|
| Embedding model | `BAAI/bge-base-en-v1.5`, 768-d, ONNX CPU, max length 512 |
| Query/passage prefixing | asymmetric (`query_embed` / `passage_embed`) |
| Vector store | Qdrant local, cosine distance, UUID5 point IDs |
| Chunking | max 2,000 chars, overlap 250, min 100; recursive paragraph→sentence→word |
| Dense top-k | 20 (widened to ≤ 48 for section+document lookups) |
| Hybrid/RRF | k = 60; BM25 weight 0.55; dense weight 0.45 |
| Rerank top-k | 10 |
| Reranker | configured but disabled |
| Generation | instruction-tuned LLM (default `gpt-4o-mini`), temperature 0.3, max 4,000 tokens, SSE streaming |
| Corpus | 19 documents, 10,676 chunks |

### Appendix B. Reproducibility Notes

- Full reindex: stop the API server (release the Qdrant on-disk lock), run `python ingest.py --force-reindex`, restart the server. Index-time fixes (chunking/heading/metadata) require this; query-time fixes (prompts, normalisation, retrieval logic, document catalog) take effect on process reload.
- Inspection endpoints: `/query` returns retrieved chunks with scores and section paths for any question; `/chat` streams the grounded answer.
