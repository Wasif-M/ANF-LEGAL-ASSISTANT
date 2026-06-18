"""Shared helpers for chunk metadata, section numbers, and display formatting."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def chunk_metadata(chunk: Any) -> dict[str, Any]:
    """Return metadata dict for a DocumentChunk or similar object."""
    if hasattr(chunk, "metadata"):
        return chunk.metadata or {}
    return {}


def normalize_section_number(section: str) -> str:
    """Canonicalize a section/article number so equivalent spellings compare equal.

    The separator (space, hyphen, underscore, or none) and letter-case must NOT change
    the canonical form, otherwise "35A" misses the stored "35-A" and the lookup fails.

    - '35A', '35-A', '35a', '35 a', '35_a'  → '35A'   (single-letter suffix: joined)
    - '337I', '337 II', '337-iii'           → '337I', '337-II', '337-III'  (multi-letter: hyphen)
    - '161', '3.1.4', '45(1)(a)'            → unchanged
    """
    if not section:
        return section

    s = str(section).strip()
    # number (optionally dotted) + optional alphabetic suffix, any/no separator
    match = re.match(r"^(\d+(?:\.\d+)*)[\s_-]*([A-Za-z]{1,6})$", s)
    if match:
        base, suffix = match.group(1), match.group(2).upper()
        return f"{base}{suffix}" if len(suffix) == 1 else f"{base}-{suffix}"
    return s


def section_number_variants(section: str) -> list[str]:
    """Progressively broader equivalents of a section reference, most specific first.

    Users cite a clause/sub-section many ways ("9c", "9(c)", "14(1)(a)") while the
    index stores the PARENT section ("9", "14") whose text contains the clause.
    Try the exact form first (so a genuinely distinct lettered section like "20B"
    still wins), then the clause-paren spelling, then the parent numbers.

    - "9C"       → ["9C", "9(c)", "9"]
    - "9(c)"     → ["9(c)", "9C", "9"]
    - "14(1)(a)" → ["14(1)(a)", "14(1)", "14"]
    - "161"      → ["161"]
    """
    variants: list[str] = []

    def add(v: str) -> None:
        if v and v not in variants:
            variants.append(v)

    s = str(section).strip()
    add(s)
    # letter-suffix form: "9C" / "20-AB" → clause spelling "9(c)" (single letter
    # only) and the bare parent number
    m = re.fullmatch(r"(\d+(?:\.\d+)*)-?([A-Za-z]{1,3})", s)
    if m:
        base, suffix = m.group(1), m.group(2)
        if len(suffix) == 1:
            add(f"{base}({suffix.lower()})")
        add(base)
    # paren-clause form: peel trailing "(…)" groups one at a time
    current = s
    while True:
        m = re.fullmatch(r"(.+)\(([A-Za-z\d]+)\)", current)
        if not m:
            break
        parent, clause = m.group(1), m.group(2)
        # "9(c)" is also spelled "9C" when the clause is a single letter on a bare number
        if len(clause) == 1 and clause.isalpha() and re.fullmatch(r"\d+(?:\.\d+)*", parent):
            add(f"{parent}{clause.upper()}")
        add(parent)
        current = parent
    return variants


def normalize_doc_blob(text: str) -> str:
    """Normalize titles/paths for token overlap (hyphen/underscore tolerant)."""
    return re.sub(r"[-_]+", " ", text.lower())


def chunk_text_matches_section_number(text: str, target_section: str) -> bool:
    """Boundary-safe match for headings like '72. Punishment' and 'Section 6'."""
    if not target_section or not text:
        return False
    esc = re.escape(target_section)
    patterns = [
        rf"(?i)\b(?:section|article|rule)\s+{esc}(?:[\s\.\:\-—\(]|$)",
        rf"(?i)§\s*{esc}\b",
        rf"(?i)(?:^|[\n\r])\s*{esc}\s*[\).\:\-—]\s+\S",
        rf"(?i)(?:^|[\n\r])[^\n]{{0,6}}{esc}\s*[\).\:\-—]\s+\S",
        rf"(?i)(?:^|[\n\r])\s*{esc}(?:[\.\:\-]\s|\s)",
        rf"(?i)\b{esc}\s*[\.\:\-—]\s",
        rf"(?i)\bsection:\s*[^\n]{{0,120}}{re.escape(target_section)}\b",
    ]
    for pat in patterns:
        if re.search(pat, text, re.MULTILINE):
            return True
    return False


# A line that begins a new enumerated clause / sub-paragraph of a provision.
# These are the ONLY places a quoted provision should break onto a new line;
# every other newline in a PDF excerpt is just arbitrary line-wrapping.
_CLAUSE_START = re.compile(
    r"^\(?(?:[a-z]|[ivxlcdm]{1,5}|\d{1,3}[A-Z]?)\)"      # (a) (i) (1) (2A)
    r"|^(?:Explanation|Provided|Illustration)s?\b",        # common sub-paras
    re.IGNORECASE,
)


def reflow_provision_text(text: str) -> str:
    """Rejoin PDF-wrapped lines so a quoted provision reads as continuous sentences,
    while keeping each enumerated clause ((a), (1), (i)…) on its own line.

    PDFs wrap a single sentence across several lines; when the model quotes that text
    verbatim and the UI renders newlines as breaks (remark-breaks), the sentence
    shatters mid-line. Normalising the excerpt up front fixes the quote at the source.
    """
    if not text:
        return text
    out_paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in paragraph.split("\n") if ln.strip()]
        if not lines:
            continue
        merged: list[str] = []
        for line in lines:
            if not merged or _CLAUSE_START.match(line):
                merged.append(line)
            else:
                merged[-1] = f"{merged[-1]} {line}"
        out_paragraphs.append("\n".join(merged))
    return "\n\n".join(out_paragraphs)


def clean_markdown_formatting(text: str) -> str:
    """Remove markdown symbols while preserving structure."""
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\n\n\n+", "\n\n", text)
    return text.strip()


def build_section_references_from_chunks(candidates: list) -> list[dict]:
    """Build UI section citations from retrieved chunk metadata."""
    references: list[dict] = []
    seen: set[str] = set()

    for cand in candidates:
        chunk = cand.chunk if hasattr(cand, "chunk") else cand
        meta = chunk_metadata(chunk)
        section_num = meta.get("section_number")
        if not section_num:
            continue

        ref_number = normalize_section_number(str(section_num))
        doc_title = meta.get("title") or Path(chunk.source_path).stem.replace("_", " ").title()
        ref_type = "Section"
        full_display = f"{doc_title} {ref_type} {ref_number}"
        ref_key = full_display.lower()

        if ref_key in seen:
            continue
        seen.add(ref_key)

        fused = getattr(cand, "fused_score", None) or 0.0
        relevance = min(0.99, max(0.5, 0.5 + fused * 0.01)) if fused else 0.85

        references.append(
            {
                "type": ref_type,
                "number": ref_number,
                "law": doc_title,
                "full_reference": full_display,
                "relevance": relevance,
            }
        )

    return references
