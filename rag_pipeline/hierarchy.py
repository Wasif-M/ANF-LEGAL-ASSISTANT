from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class HeadingMatch:
    level: int
    title: str

_SECTION_NUM_PATTERN = r"\d+(?:\.\d+)*(?:[A-Z]{1,3})?(?:[-_][A-Za-z]{1,6})?(?:\([a-z\d]+\))*"

_NUMERIC_HEADING = re.compile(
    rf"^(?P<num>{_SECTION_NUM_PATTERN})[\).:;\-—]?\s+(?P<title>.+\S)"
)

_ALPHA_HEADING = re.compile(
    r"^(?P<num>[a-z])[\).:-]?\s+(?P<title>.+\S)", re.IGNORECASE
)

_ROMAN_HEADING = re.compile(
    r"^(?P<num>[ivxlcdm]+)[\).:-]?\s+(?P<title>.+\S)", re.IGNORECASE
)

_LABELED_HEADING = re.compile(
    r"^(chapter|part|schedule)\b[:\s-]*(?P<title>.+\S)?$", re.IGNORECASE
)

# Table-of-contents entries look like "79. Proof of execution ........ 35" or
# "85. Public documents        36". They must NOT be treated as section headings,
# otherwise every section number is detected twice (once in the TOC, once in the
# body) producing phantom/duplicate sections and weak fragment chunks.
_TOC_LINE = re.compile(r"(?:\.{3,}|\s{3,}\d{1,4})\s*\d{0,4}\s*$")

# Page-footer / running-header artifacts such as "3 | Page", "10 | P a g e" (the PDF
# extractor often spaces out "Page"), or "Page 12". Left unchecked, the leading number
# is mis-read as a section number, creating phantom sections that hijack number lookups.
_PAGE_MARKER = re.compile(
    r"^\d{1,4}\s*\|\s*p\s*a\s*g\s*e\b"   # "3 | Page", "10 | P a g e"
    r"|^p\s*a\s*g\s*e\s+\d{1,4}\b",       # "Page 12"
    re.IGNORECASE,
)

# Legal section/article/rule heading — handles all formats:
#   Section 4A, Article 20B, Rule 12.14, Schedule II, Section 337-III
_LEGAL_SECTION = re.compile(
    rf"^(?:Section|Article|Schedule|Rule)\s+(?P<num>{_SECTION_NUM_PATTERN})\s*[.\)\-—]?\s*(?P<title>.+\S)?",
    re.IGNORECASE,
)


def _is_running_text_after_number(after_num: str) -> bool:
    """True when what follows a leading section number reads as a wrapped sentence or
    cross-reference, not a heading title — so it must NOT be treated as a heading.

    A genuine heading has a delimiter (". ) : ; - —") right after the number, OR a
    Title-Case title. Running text instead continues with a lowercase word, possibly after
    a stray comma the PDF emitted mid-sentence:
        "492 , and includes any person ..."        (enumeration continuation)
        "Section 30 , may pass any sentence ..."   (cross-reference continuation)
        "402-B , the Provincial Government ..."     (provision body that wrapped)
    A comma followed by a Title-Case word is KEPT, so a real heading whose '.' was OCR'd as
    ',' survives — e.g. "265-N , Place of holding sittings : (1) ...".
    """
    if re.match(r"\s*[.\)\:;\-—]", after_num):
        return False
    rest = after_num.lstrip()
    if rest.startswith(","):
        rest = rest[1:].lstrip()
    return rest[:1].islower()


def _looks_like_title(line: str) -> bool:
    if len(line) < 4 or len(line) > 90:
        return False
    # Trailing punctuation marks running text, not a heading. Dashes matter most:
    # a wrapped provision like "… etc: No one\nShall---\n(a) …" emits "Shall---",
    # which would otherwise become a LEVEL-1 heading that wipes the real section
    # heading off the path stack (CNS Act 1997 s.8 lost its number this way, and
    # every following section was mis-parented under "Shall---").
    if line.endswith((".", ";", ",", "-", "—", "–")):
        return False
    words = line.split()
    if len(words) > 12:
        return False
    alpha_words = [word for word in words if any(char.isalpha() for char in word)]
    if not alpha_words:
        return False
    uppercase_ratio = sum(word.isupper() for word in alpha_words) / len(alpha_words)
    title_case_ratio = sum(word[:1].isupper() for word in alpha_words) / len(alpha_words)
    return uppercase_ratio >= 0.7 or title_case_ratio >= 0.85


def detect_heading(line: str) -> HeadingMatch | None:
    text = line.strip()
    if not text:
        return None

    # 0a. Strip leading insertion/quote markers that wrap amended provisions in legal
    #     PDFs, e.g. "[93-A. Sending of summons ...]" or “126-A. ...”. The bracket/quote
    #     otherwise hides the section number, so inserted sections (93-A, 126-A, …) are
    #     never detected and their text is absorbed into the previous section.
    text = text.lstrip("[]“”‘’\"'� ").strip()
    if not text:
        return None

    # 0. Skip table-of-contents / index entries (dotted leaders or trailing page no.)
    if _TOC_LINE.search(text):
        return None

    # 0b. Skip page-footer / running-header lines ("3 | Page", "Page 12") so their
    #     leading number isn't mistaken for a section number.
    if _PAGE_MARKER.search(text):
        return None

    # 1. Try explicit legal section/article/rule pattern first (highest priority)
    legal_section = _LEGAL_SECTION.match(text)
    if legal_section:
        num = legal_section.group("num")
        title = legal_section.group("title")
        # Reject cross-reference continuation lines. When a provision's sentence wraps so
        # the new line begins with a cross-reference — "Section 30 , may pass any sentence
        # ...", "Section 98 , Section 99-A or Section 100." — it is running text, not a
        # heading. Treating it as one splits the real provision and spawns a phantom section
        # (e.g. a wrapped line stole section 34's body, leaving "34. Higher powers of certain"
        # with no chunk at all).
        if _is_running_text_after_number(text[legal_section.start("num") + len(num):]):
            return None
        depth = num.count(".") + 1
        lower = text.lower()
        if lower.startswith("article"):
            label = "Article"
        elif lower.startswith("rule"):
            label = "Rule"
        elif lower.startswith("schedule"):
            label = "Schedule"
        else:
            label = "Section"
        if not title or not title.strip():
            full_title = f"{label} {num}"
        else:
            full_title = f"{label} {num} {title.strip()}"
        return HeadingMatch(level=1 + depth, title=full_title)

    # 2. Try labeled headings (Chapter, Part, Schedule)
    labeled = _LABELED_HEADING.match(text)
    if labeled:
        title = labeled.group("title") or labeled.group(1)
        return HeadingMatch(level=1, title=title.strip())

    # 3. Try numeric headings — most common in Pakistani legal docs
    numeric = _NUMERIC_HEADING.match(text)
    if numeric:
        num = numeric.group("num")
        title = numeric.group("title").strip()
        # Reject wrapped sentence / cross-reference lines: a BARE number (no heading
        # delimiter such as . ) : ; - after it) followed by a lowercase word is
        # mid-sentence text — "339 a person who...", "467 knowing it to be...",
        # "128 or Section 130", "211 of the Pakistan Penal Code", or after a stray
        # comma "402-B , the Provincial Government ..." — not a heading. Such false
        # headings create phantom sections and, worse, pre-empt the real suffixed
        # heading that follows them (e.g. a wrapped "339 ..." stealing 339-A).
        if _is_running_text_after_number(text[len(num):]):
            return None
        depth = num.count(".") + 1
        # Include section number in title for proper extraction later
        full_title = f"{num} {title}" if title else num
        return HeadingMatch(level=1 + depth, title=full_title)

    # 4. Alphabetic subheadings (a), b), etc.)
    alpha = _ALPHA_HEADING.match(text)
    if alpha and len(alpha.group("num")) == 1:
        return HeadingMatch(level=4, title=alpha.group("title").strip())

    # 5. Roman numeral subheadings (i, ii, iii, etc.)
    roman = _ROMAN_HEADING.match(text)
    if roman and len(roman.group("num")) <= 6:
        return HeadingMatch(level=5, title=roman.group("title").strip())

    # 6. Title-case or ALL-CAPS lines (e.g., "PRELIMINARY", "GENERAL")
    if _looks_like_title(text):
        return HeadingMatch(level=1, title=text)

    return None


def section_path_label(section_path: tuple[str, ...]) -> str:
    if not section_path:
        return "Document"
    return " > ".join(section_path)