"""Generic PDF text repair and search tokenization (no act-specific rules)."""

from __future__ import annotations

import re

# Common function words (len >= 3) — used only to split long run-on tokens, not normal words.
_COMMON_WORDS: tuple[str, ...] = (
    "the",
    "of",
    "and",
    "or",
    "in",
    "to",
    "by",
    "as",
    "is",
    "be",
    "at",
    "on",
    "for",
    "with",
    "from",
    "not",
    "no",
    "any",
    "may",
    "shall",
    "such",
    "when",
    "who",
    "that",
    "this",
    "his",
    "its",
    "have",
    "has",
    "had",
    "been",
    "under",
    "within",
    "without",
    "save",
    "unless",
    "where",
    "which",
    "their",
    "there",
    "into",
    "upon",
    "against",
    "between",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "other",
    "than",
    "each",
    "every",
    "all",
    "both",
    "either",
    "neither",
    "if",
    "so",
    "but",
    "can",
    "will",
    "would",
    "should",
    "could",
    "must",
    "act",
    "section",
    "article",
    "rule",
    "clause",
    "chapter",
    "part",
    "schedule",
    "order",
    "force",
    "member",
    "members",
    "officer",
    "officers",
    "rank",
    "power",
    "powers",
    "duty",
    "duties",
    "right",
    "rights",
    "person",
    "court",
    "police",
    "director",
    "general",
    "inspector",
    "sub",
    "government",
    "punjab",
    "pakistan",
    "purpose",
    "provision",
    "provisions",
)

_SPLIT_WORDS: tuple[str, ...] = tuple(w for w in _COMMON_WORDS if len(w) >= 3)
_SORTED_SPLIT_WORDS: tuple[str, ...] = tuple(sorted(_SPLIT_WORDS, key=len, reverse=True))

_RUNON_TOKEN_RE = re.compile(r"^[A-Za-z]{20,}$")
_CAMEL_SPLIT_RE = re.compile(r"([a-z])([A-Z])")
_ACRONYM_CHUNK_RE = re.compile(r"[A-Z][a-z]+|[a-z]+|[A-Z]+")


def _segment_runon_word(word: str) -> list[str] | None:
    """Segment a collapsed PDF token into KNOWN words only.

    Returns the decomposition only when the whole token is built entirely from
    dictionary words (e.g. "thecourtshall" -> the/court/shall). If any leftover
    cannot be matched against a known word it returns None, leaving the token
    untouched. The previous version fell back to splitting on single characters,
    which shattered ordinary long words ("magistrates" -> "m a g i s t r a t es")
    and corrupted the corpus.
    """
    if not word:
        return []

    for candidate in _SORTED_SPLIT_WORDS:
        if word.startswith(candidate):
            rest = _segment_runon_word(word[len(candidate) :])
            if rest is not None:
                return [candidate, *rest]

    return None


def _repair_runon_token(token: str) -> str:
    """Split long single tokens where PDF spacing collapsed (not normal short words)."""
    if " " in token:
        return token

    lead = trail = ""
    core = token
    while core and not core[0].isalpha():
        lead += core[0]
        core = core[1:]
    while core and not core[-1].isalpha():
        trail = core[-1] + trail
        core = core[:-1]
    if len(core) < 12:
        return token

    lowered = core.lower()
    segments = _segment_runon_word(lowered)
    if not segments or len(segments) <= 1:
        return token
    # Reject low-confidence splits that leave very short fragments — these are
    # almost always normal words mis-segmented, not genuine collapsed run-ons.
    if any(len(seg) < 3 for seg in segments):
        return token

    repaired = " ".join(segments)
    repaired = re.sub(r" {2,}", " ", repaired)
    return f"{lead}{repaired}{trail}"


def fix_pdf_spacing(text: str) -> str:
    """Repair collapsed PDF spacing: camelCase splits and run-on token repair."""
    if not text:
        return text

    text = _CAMEL_SPLIT_RE.sub(r"\1 \2", text)
    text = re.sub(r"([,;:])(?=\S)", r"\1 ", text)
    text = re.sub(r"(?<=\S)([,;:])", r" \1", text)
    # Repair run-on tokens line by line. Splitting on the whole string with
    # str.split() would discard every newline and collapse the document into a
    # single line, which breaks line-based heading detection in chunking.
    repaired_lines = [
        " ".join(_repair_runon_token(part) for part in line.split())
        for line in text.split("\n")
    ]
    text = "\n".join(repaired_lines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _split_runon_token(token: str) -> list[str]:
    parts = _ACRONYM_CHUNK_RE.findall(token)
    return [p.lower() for p in parts if len(p) > 1]


def tokenize_for_search(text: str) -> list[str]:
    """Tokenize text for BM25 with PDF repair and run-on protection."""
    text = fix_pdf_spacing(text)
    tokens: list[str] = []
    for raw in text.split():
        token = raw.strip()
        if not token:
            continue
        if _RUNON_TOKEN_RE.match(token):
            tokens.extend(_split_runon_token(token))
        else:
            tokens.append(token.lower())
    return tokens
