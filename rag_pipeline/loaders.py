from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from .document_catalog import (
    generate_aliases,
    humanize_stem,
    infer_document_type,
    infer_group_id,
    load_sidecar_metadata,
)
from .models import SourceDocument
from .text_normalization import fix_pdf_spacing

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _normalize_extensions(extensions: Iterable[str] | None) -> set[str]:
    if not extensions:
        return set(SUPPORTED_EXTENSIONS)
    normalized: set[str] = set()
    for ext in extensions:
        cleaned = ext.strip().lower()
        if not cleaned:
            continue
        if not cleaned.startswith("."):
            cleaned = f".{cleaned}"
        normalized.add(cleaned)
    return normalized or set(SUPPORTED_EXTENSIONS)


def _is_sidecar(path: Path) -> bool:
    """`<file>.meta.json` files carry metadata for a source document; they must never
    be ingested as content themselves."""
    return path.name.lower().endswith(".meta.json")


def discover_documents(search_paths: Iterable[Path], extensions: Iterable[str] | None = None) -> list[Path]:
    allowed_extensions = _normalize_extensions(extensions)
    documents: list[Path] = []
    for root in search_paths:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() in allowed_extensions and not _is_sidecar(root):
            documents.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in allowed_extensions and not _is_sidecar(path):
                documents.append(path)
    unique_documents = sorted({path.resolve() for path in documents})
    return unique_documents


def _remove_legal_annotations(text: str) -> str:
    """Remove legal footnotes and amendment citations that clutter chunks."""
    # Remove footnote references like "vide amendments in the AML Act-official Gazette Notification no. F.22(50)/2020-Legis dated 24-Sep-2020"
    text = re.sub(r"vide amendments[^\n]*(?:Legis|dated)[^\n]*", "", text, flags=re.IGNORECASE)
    # Remove inline footnote markers after letters (e.g. "held51" or "word32").
    # Do NOT use \w before digits: digits are "word" chars and would match "103." as (\w="1")\d{2} → corrupts section headings.
    text = re.sub(r"([A-Za-z])([1-9]\d{0,2})([\s.\)\],:;\-—])", r"\1\3", text)
    # Remove bracketed footnote references like "[see section 2 (xxvi)]"
    text = re.sub(r"\[see[^\]]*\]", "", text, flags=re.IGNORECASE)
    return text


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _remove_legal_annotations(text)
    text = fix_pdf_spacing(text)
    return text.strip()


def _extract_pdf_pages(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(_normalize_text(page_text))
    return pages


def _extract_plain_text(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [_normalize_text(text)]


def load_document(path: Path) -> SourceDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = _extract_pdf_pages(path)
    else:
        pages = _extract_plain_text(path)

    pages = [page for page in pages if page.strip()]
    text = "\n\n".join(pages).strip()
    sidecar = load_sidecar_metadata(path)
    title = str(sidecar.get("title") or humanize_stem(path.stem))
    group_id = infer_group_id(path.stem, sidecar)
    doc_type = infer_document_type(path.as_posix(), title, sidecar)
    aliases = generate_aliases(title, path.stem, sidecar)
    metadata: dict = {
        "page_count": len(pages),
        "file_name": path.name,
        "display_title": title,
        "document_type": doc_type,
        "document_group_id": group_id,
        "search_aliases": aliases,
    }
    if sidecar.get("amends") or sidecar.get("amends_group_id"):
        metadata["amends_group_id"] = str(sidecar.get("amends") or sidecar.get("amends_group_id"))
    elif doc_type == "amendment":
        base_stem = re.sub(
            r"(?:first\s+)?amendment.*",
            "",
            path.stem,
            flags=re.IGNORECASE,
        ).strip("-_ ")
        if base_stem:
            metadata["amends_group_id"] = infer_group_id(base_stem, {})
    return SourceDocument(
        document_id=path.resolve().as_posix(),
        source_path=path.resolve(),
        title=title,
        pages=pages,
        text=text,
        metadata=metadata,
    )