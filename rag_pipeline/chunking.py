from __future__ import annotations

import re
from dataclasses import dataclass

from .config import ChunkingConfig
from .hierarchy import detect_heading, section_path_label
from .models import DocumentChunk, SourceDocument
from .text_normalization import fix_pdf_spacing

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_AMENDMENT_PATTERNS = [
    r"as amended by(?:\s+(?:Act|Section|the))?(?:\s+)(\S+\s+\d+[\w\(\)]*)",
    r"(?:amended|modified|substituted|replaced)\s+(?:by|with|through)(?:\s+(?:Act|Section|the))?(?:\s+)(\S+\s+\d+[\w\(\)]*)",
    r"now(?:\s+)?(?:reads?|provides?|states?):",
    r"(?:Sub-section|Clause|Paragraph)\s+(?:\d+[A-Za-z]*)\s+(?:inserted|added|substituted|omitted)",
]

_AMENDMENT_REGEX = re.compile("|".join(f"({p})" for p in _AMENDMENT_PATTERNS), re.IGNORECASE)


@dataclass(slots=True)
class _HeadingState:
    level: int
    title: str


def _merge_split_section_number_lines(text: str) -> str:
    """Join PDF lines where the section number is alone on one line and the title is on the next.

    Many extractors emit:
        103.
        Exclusion of evidence...
    which would otherwise miss numeric heading detection.
    """
    lines = text.splitlines()
    if not lines:
        return text

    # First pass: rejoin a section number whose alphabetic suffix was split onto the
    # next line, e.g. PPC's hurt provisions extract as:
    #     337-
    #     A.
    #     Punishment of shajjah :
    # -> "337-A." so the 337-A … 337-N block keeps its section numbers.
    dangling_num = re.compile(r"^(\d+(?:\.\d+)*)-$")
    suffix_only = re.compile(r"^([A-Za-z]{1,3})([.\):]?)\s*(.*)$")
    joined: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i].strip()
        dm = dangling_num.match(cur)
        if dm and i + 1 < len(lines):
            sm = suffix_only.match(lines[i + 1].strip())
            if sm and sm.group(2):  # suffix letter followed by . ) or :
                rest = sm.group(3).strip()
                head = f"{dm.group(1)}-{sm.group(1).upper()}."
                joined.append(f"{head} {rest}".strip() if rest else head)
                i += 2
                continue
        joined.append(lines[i])
        i += 1
    lines = joined

    num_only = re.compile(
        rf"^(?P<num>\d+(?:\.\d+)*(?:[A-Z]{{1,3}})?(?:[-_][A-Za-z]{{1,6}})?(?:\([a-z\d]+\))*)\s*[\.\)\:\-—]?\s*$"
    )
    merged: list[str] = []
    index = 0
    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        match = num_only.match(stripped)
        if match and index + 1 < len(lines):
            nxt = lines[index + 1].strip()
            if (
                len(nxt) >= 6
                and not num_only.match(nxt)
                and detect_heading(nxt) is None
                and re.match(r"^[A-Za-z(\"\u201c]", nxt)
            ):
                num = match.group("num")
                merged.append(f"{num}. {nxt}")
                index += 2
                continue
        merged.append(raw)
        index += 1
    return "\n".join(merged)


def _normalize_block(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        cleaned_lines.append(stripped)
    normalized = "\n".join(cleaned_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _split_by_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    return parts if parts else [text.strip()]


def _split_by_words(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current_words: list[str] = []
    current_length = 0

    for word in words:
        tentative_length = current_length + len(word) + (1 if current_words else 0)
        if tentative_length > max_chars and current_words:
            chunk = " ".join(current_words).strip()
            if chunk:
                chunks.append(chunk)
            if overlap_chars > 0:
                overlap_words: list[str] = []
                overlap_length = 0
                for previous_word in reversed(current_words):
                    needed = len(previous_word) + (1 if overlap_words else 0)
                    if overlap_length + needed > overlap_chars:
                        break
                    overlap_words.insert(0, previous_word)
                    overlap_length += needed
                current_words = overlap_words[:] if overlap_words else []
                current_length = len(" ".join(current_words))
            else:
                current_words = []
                current_length = 0
        current_words.append(word)
        current_length += len(word) + (1 if len(current_words) > 1 else 0)

    tail = " ".join(current_words).strip()
    if tail:
        chunks.append(tail)
    return chunks


def _detect_amendments(text: str) -> list[dict]:
    """Detect amendment patterns in text.
    Returns list of amendment info dicts with detected amendment references.
    """
    amendments = []
    if not text:
        return amendments
    
    # Check for amendment indicators
    for pattern in _AMENDMENT_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            amendment_text = match.group(0)
            # Extract potential section number from amendment text
            section_match = re.search(r"(\d+(?:\.\d+)*(?:[-_][A-Za-z]+)?(?:\([a-z\d]+\))*)", amendment_text)
            if section_match:
                amendments.append({
                    "text": amendment_text,
                    "section": section_match.group(1),
                    "pattern": pattern,
                })
    
    return amendments


def _extract_parent_section(section_path: tuple[str, ...]) -> str | None:
    """Extract parent section from section path.
    For hierarchical sections like "45 > 45(1) > 45(1)(a)", 
    returns the immediate parent section.
    """
    if len(section_path) <= 1:
        return None
    # Return the first section in the path as the parent
    return section_path[0] if section_path else None


def _get_subsection_depth(title: str) -> int:
    """Calculate subsection depth based on parentheses/brackets.
    45 → depth 0
    45(1) → depth 1
    45(1)(a) → depth 2
    """
    depth = title.count("(")
    return depth


def _get_section_hierarchy(section_path: tuple[str, ...]) -> list[str]:
    """Build hierarchical list of ancestor sections.
    Returns all ancestor sections from root to current.
    """
    if not section_path:
        return []
    return list(section_path)


def _recursive_split(text: str, config: ChunkingConfig) -> list[str]:
    """Recursively split text using hierarchical strategy: paragraphs → sentences → words.
    
    Args:
        text: Text to split
        config: Chunking configuration
    
    Returns:
        List of text chunks
    """
    text = _normalize_block(text)
    if not text:
        return []
    if len(text) <= config.max_chars:
        return [text]

    paragraph_chunks: list[str] = []
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    current: list[str] = []

    def flush_current() -> None:
        if current:
            paragraph_chunks.append("\n\n".join(current).strip())
            current.clear()

    for paragraph in paragraphs:
        if len(paragraph) > config.max_chars:
            flush_current()
            sentence_chunks = _split_by_sentences(paragraph)
            running: list[str] = []
            for sentence in sentence_chunks:
                if len(sentence) > config.max_chars:
                    flush_current()
                    paragraph_chunks.extend(_split_by_words(sentence, config.max_chars, config.overlap_chars))
                    running = []
                    continue
                tentative = " ".join(running + [sentence]).strip()
                if len(tentative) > config.max_chars and running:
                    paragraph_chunks.append(" ".join(running).strip())
                    running = [sentence]
                else:
                    running.append(sentence)
            if running:
                paragraph_chunks.append(" ".join(running).strip())
        else:
            tentative = "\n\n".join(current + [paragraph]).strip()
            if len(tentative) > config.max_chars and current:
                flush_current()
            current.append(paragraph)

    flush_current()

    if not paragraph_chunks:
        paragraph_chunks = [text]

    final_chunks: list[str] = []
    for chunk in paragraph_chunks:
        if len(chunk) <= config.max_chars:
            final_chunks.append(chunk)
        else:
            final_chunks.extend(_split_by_words(chunk, config.max_chars, config.overlap_chars))
    return [chunk for chunk in final_chunks if chunk.strip()]


def chunk_document(document: SourceDocument, config: ChunkingConfig) -> list[DocumentChunk]:
    text = fix_pdf_spacing(_normalize_block(document.text))
    text = _merge_split_section_number_lines(text)
    if not text:
        return []

    lines = text.splitlines()
    chunks: list[DocumentChunk] = []
    section_stack: list[_HeadingState] = []
    buffer_lines: list[str] = []
    chunk_index = 0

    def current_path() -> tuple[str, ...]:
        return tuple(item.title for item in section_stack)

    def _extract_section_number(title: str) -> str | None:
        """Extract section number from title with all formats:
        - Simple: '161', '373', '127'
        - Decimal: '3.1.4', '21.6', '22.20'
        - With suffix: '337-I', '337-J', '20B', '4A', '302B'
        - Hyphenated roman: '337-II', '337-III'
        - With subsections: '45(1)', '45(1)(a)'
        - Rules: 'Rule 12.14'
        """
        # Comprehensive section number pattern
        _SEC_NUM = r"\d+(?:\.\d+)*(?:[A-Z]{1,3})?(?:[-_][A-Za-z]{1,6})?(?:\([a-z\d]+\))*"

        # The heading's OWN number comes first in the line; a "Section X" appearing
        # later is a cross-reference, not this section's number. So check the START of
        # the heading before falling back to a mid-title "Section X". Otherwise a
        # heading like "13. Punishment for contravention of Section 7" is mis-numbered
        # as 7 instead of 13.
        # 1. Explicit "Section/Article/Rule X" at the start (e.g. "Section 5. Title")
        match = re.match(rf"^\s*(?:Section|Article|Rule)\s+({_SEC_NUM})", title, re.IGNORECASE)
        if match:
            return match.group(1)
        # 2. Leading bare number: "13.", "161.", "337-I.", "20B."
        match = re.match(rf"^\s*({_SEC_NUM})", title)
        if match:
            return match.group(1)
        # 3. Fallback: a "Section/Article/Rule X" reference anywhere (headings that
        #    don't lead with their own number)
        match = re.search(rf"(?:Section|Article|Rule)\s+({_SEC_NUM})", title, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def emit_buffer() -> None:
        nonlocal chunk_index, buffer_lines
        body = _normalize_block("\n".join(buffer_lines))
        buffer_lines = []
        if not body:
            return

        path = current_path()
        prefix = section_path_label(path)
        section_num = None
        primary_heading = path[0] if path else ""
        if path:
            for segment in reversed(path):
                extracted = _extract_section_number(segment)
                if extracted:
                    section_num = extracted
                    primary_heading = segment
                    break
        
        # NEW: Detect amendments in the section
        amendments = _detect_amendments(body)
        
        # NEW: Extract parent section and hierarchy
        parent_section = _extract_parent_section(path)
        section_hierarchy = _get_section_hierarchy(path)
        subsection_depth = _get_subsection_depth(primary_heading) if path else 0
        
        split_pieces = _recursive_split(body, config)
        for piece in split_pieces:
            chunk_text = f"SECTION: {prefix}\n\n{piece}" if path else piece
            # Combine document-level metadata with chunk-level metadata
            metadata = {
                "title": document.title,
                "section_path": list(path),
                "section_label": prefix,
            }
            if section_num:
                metadata["section_number"] = section_num
            
            # NEW: Add parent-child metadata
            if parent_section:
                metadata["parent_section"] = parent_section
                metadata["section_hierarchy"] = section_hierarchy
                metadata["subsection_depth"] = subsection_depth
            
            # NEW: Add amendment metadata
            if amendments:
                metadata["amendments_detected"] = True
                metadata["amendments"] = amendments
                # Extract referenced section numbers from amendments
                amendment_sections = [amend.get("section") for amend in amendments if amend.get("section")]
                if amendment_sections:
                    metadata["amendment_references"] = amendment_sections
            else:
                metadata["amendments_detected"] = False
            
            # Add document-level metadata (for legal docs: type, act_number, etc.)
            if document.metadata:
                metadata.update(document.metadata)
            
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.document_id}::{chunk_index:05d}",
                    document_id=document.document_id,
                    source_path=document.source_path.as_posix(),
                    text=chunk_text,
                    section_path=path,
                    chunk_index=chunk_index,
                    start_char=0,
                    end_char=len(piece),
                    metadata=metadata,
                )
            )
            chunk_index += 1

    for line in lines:
        heading = detect_heading(line)
        if heading:
            emit_buffer()
            while section_stack and section_stack[-1].level >= heading.level:
                section_stack.pop()
            section_stack.append(_HeadingState(level=heading.level, title=heading.title))
            continue
        buffer_lines.append(line)

    emit_buffer()

    if not chunks:
        for piece in _recursive_split(text, config):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.document_id}::{chunk_index:05d}",
                    document_id=document.document_id,
                    source_path=document.source_path.as_posix(),
                    text=piece,
                    section_path=(),
                    chunk_index=chunk_index,
                    start_char=0,
                    end_char=len(piece),
                    metadata={"title": document.title},
                )
            )
            chunk_index += 1

    return chunks