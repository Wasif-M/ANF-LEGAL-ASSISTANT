"""Generic document catalog built from ingested metadata (no hardcoded act lists)."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .models import DocumentChunk
from .utils import chunk_metadata

_STOPWORDS = frozenset(
    {"the", "of", "and", "in", "to", "a", "an", "for", "on", "by", "no", "act", "order"}
)

# Title words that are unique to one document but too GENERIC to identify it on their
# own (a query mentioning them topically must not be hard-routed to that document).
# Distinctive subject terms — "shahadat", "laundering", "narcotic", "qanun" — are not
# here, so they still act as decisive single-token identifiers.
_GENERIC_TITLE_WORDS = frozenset(
    {
        "procedure", "evidence", "court", "courts", "rule", "rules", "code",
        "criminal", "civil", "police", "general", "order", "law", "laws",
        "national", "federal", "provincial", "government", "ordinance",
        "authority", "council", "control", "substances", "force",
    }
)

# Edition / amendment / temporal words that appear in stored titles but never help
# identify which document the user means. Dropping them keeps a long official title
# like "Anti Money Laundering Act 2010 Amended Upto Sep. 2020" matchable by its core
# name ("anti money laundering act").
_DOC_NOISE = frozenset(
    {
        "amended", "amendment", "amendments", "upto", "repealed", "substituted",
        "first", "second", "third", "fourth", "fifth", "vol", "volume", "part",
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "july", "aug", "sep",
        "sept", "oct", "nov", "dec", "january", "february", "march", "april",
        "june", "august", "september", "october", "november", "december",
    }
)

_TYPE_FROM_FOLDER = {
    "amendments": "amendment",
    "amendment": "amendment",
    "base_acts": "base_act",
    "acts": "base_act",
    "rules": "rules",
    "policy": "policy",
    "reference": "reference",
}

# Well-known short forms for Pakistani statutes whose conventional abbreviation is
# NOT the plain initials of the title (e.g. "CrPC" for the Code of Criminal
# Procedure). Each maps to distinctive title tokens that must ALL belong to the
# matching document, so an entry is inert unless such a document actually exists
# in the corpus (no hardcoded dependency on any specific file set).
_KNOWN_ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "crpc": ("code", "criminal", "procedure"),
    "cnsa": ("control", "narcotic", "substances"),
    "cns": ("control", "narcotic", "substances"),
    "ppc": ("pakistan", "penal", "code"),
    "qso": ("qanun", "shahadat"),
    "amla": ("anti", "money", "laundering"),
    "aml": ("anti", "money", "laundering"),
    "anf": ("anti", "narcotics", "force"),
    "ppr": ("punjab", "police", "rules"),
    "dda": ("dangerous", "drugs"),
}

# Instrument-type words: contribute the trailing letter of the long acronym form
# ("CNSA", "QSO") but are dropped from the short form ("CNS").
_INSTRUMENT_TYPE_WORDS = frozenset({"act", "order", "code", "rules", "ordinance", "policy"})

_VOLUME_SUFFIX_RE = re.compile(
    r"[-_\s](?:part|vol(?:ume)?)[-_\s]?\d+|[-_\s](?:I{1,3}|IV|VI{0,3}|IX|X{1,3})$",
    re.IGNORECASE,
)
_AMENDMENT_MARKERS_RE = re.compile(
    r"(?:first\s+)?amendment|amended\s+upto|amendment\s+act",
    re.IGNORECASE,
)
_YEAR_SUFFIX_RE = re.compile(r"[-_\s]?\d{4}$")


@dataclass(slots=True)
class DocumentRecord:
    document_id: str
    title: str
    aliases: list[str] = field(default_factory=list)
    document_group_id: str = ""
    document_type: str = "unknown"
    amends_group_id: str | None = None
    source_paths: list[str] = field(default_factory=list)


def load_sidecar_metadata(path: Path) -> dict[str, Any]:
    """Load optional ``<file>.meta.json`` sidecar next to a source file."""
    candidates = [
        path.with_name(f"{path.name}.meta.json"),
        path.with_suffix(".meta.json"),
    ]
    for sidecar in candidates:
        if sidecar.is_file():
            try:
                return json.loads(sidecar.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
    return {}


def humanize_stem(stem: str) -> str:
    """Turn ``anti-narcotics-force-act-1997`` into a readable title."""
    text = re.sub(r"[-_]+", " ", stem)
    text = re.sub(r"\s+", " ", text).strip()
    return text.title() if text else stem


def infer_document_type(source_path: str, title: str, sidecar: dict[str, Any]) -> str:
    if sidecar.get("document_type"):
        return str(sidecar["document_type"])
    path_lower = source_path.replace("\\", "/").lower()
    for folder, doc_type in _TYPE_FROM_FOLDER.items():
        if f"/{folder}/" in path_lower or path_lower.startswith(f"{folder}/"):
            return doc_type
    title_lower = title.lower()
    if _AMENDMENT_MARKERS_RE.search(title_lower):
        return "amendment"
    if "rules" in title_lower:
        return "rules"
    if "policy" in title_lower:
        return "policy"
    if "act" in title_lower or "code" in title_lower or "ordinance" in title_lower:
        return "base_act"
    return "unknown"


# Tokens that don't identify the *subject* of an instrument — edition/amendment
# words, document-type words, connectors, ordinals, roman numerals. Stripping them
# collapses a base Act and its amendments to one subject key so they group together.
_GROUP_NOISE = frozenset(
    {
        "act", "acts", "ordinance", "code", "rules", "rule", "order", "policy",
        "regulation", "regulations", "law", "amendment", "amendments", "amended",
        "upto", "repealed", "substituted", "first", "second", "third", "fourth",
        "fifth", "part", "vol", "volume", "no", "of", "the", "and", "for", "on",
        "to", "in", "by", "an", "a",
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "july", "aug", "sep",
        "sept", "oct", "nov", "dec",
    }
)
_ROMAN_RE = re.compile(r"^(?=[ivxlcdm]+$)(?:i{1,3}|iv|v|vi{0,3}|ix|x{1,3})$", re.IGNORECASE)


def infer_group_id(stem: str, sidecar: dict[str, Any]) -> str:
    """Derive a stable *subject* key shared by a base Act and its amendments/volumes.

    e.g. "Control of Narcotic Substances Act 1997", "Control of Narcotic Substances
    (First Amendment) Act 2020" and "Control-of-Narcotic-Substances-Amendment-Act-
    2022.ACT-NO.-XX-OF-2022" all collapse to "control-narcotic-substances".
    """
    if sidecar.get("document_group_id"):
        return str(sidecar["document_group_id"]).lower().strip()
    cleaned = re.sub(r"\([^)]*\)", " ", stem)          # drop parentheticals
    cleaned = re.sub(r"[^a-zA-Z]+", " ", cleaned)        # letters only (kills years, XX, dots)
    tokens = [t.lower() for t in cleaned.split()]
    core = [t for t in tokens if len(t) > 1 and t not in _GROUP_NOISE and not _ROMAN_RE.match(t)]
    key = "-".join(core)
    return key or re.sub(r"[-_\s]+", "-", stem).strip("-").lower()


def _acronym_candidates(text: str) -> set[str]:
    """Initial-letter acronyms of a title, with and without the instrument-type word.

    "Control of Narcotic Substances Act 1997" → {"cns", "cnsa"}. Years, roman
    numerals, and edition noise ("Amended Upto Sep 2020", "First Amendment") never
    contribute letters — so a base Act and its amendments share the same acronyms.
    """
    words = re.findall(r"[a-z]+", text.lower())  # letters only: drops years/numbers
    def keep(w: str, with_type: bool) -> bool:
        if len(w) <= 1 or w in _DOC_NOISE or _ROMAN_RE.match(w):
            return False
        if w in _INSTRUMENT_TYPE_WORDS:
            return with_type
        return w not in _STOPWORDS
    out: set[str] = set()
    for with_type in (False, True):
        seq = [w for w in words if keep(w, with_type)]
        if len(seq) >= 2:
            acronym = "".join(w[0] for w in seq)
            if 3 <= len(acronym) <= 8:
                out.add(acronym)
    return out


def generate_aliases(title: str, stem: str, sidecar: dict[str, Any]) -> list[str]:
    aliases: set[str] = set()
    if sidecar.get("title"):
        aliases.add(str(sidecar["title"]))
    if sidecar.get("short_names"):
        for name in sidecar["short_names"]:
            if name:
                aliases.add(str(name))
    aliases.add(title)
    humanized = humanize_stem(stem)
    if humanized:
        aliases.add(humanized)
    for acronym in _acronym_candidates(humanized or title):
        aliases.add(acronym.upper())
    return sorted(aliases, key=len, reverse=True)


@dataclass
class DocumentCatalog:
    """Index of documents and searchable aliases, built from stored chunks."""

    records: dict[str, DocumentRecord] = field(default_factory=dict)
    _alias_entries: list[tuple[str, str]] = field(default_factory=list)
    _acronyms: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_chunks(cls, chunks: list[DocumentChunk]) -> DocumentCatalog:
        catalog = cls()
        grouped: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "title": "",
                "aliases": set(),
                "paths": set(),
                "type": "unknown",
                "group_id": "",
                "amends": None,
            }
        )

        for chunk in chunks:
            meta = chunk_metadata(chunk)
            doc_id = chunk.document_id
            stem = Path(chunk.source_path).stem
            entry = grouped[doc_id]
            title = meta.get("display_title") or meta.get("title") or humanize_stem(stem)
            entry["title"] = str(title)
            entry["paths"].add(chunk.source_path)
            entry["type"] = meta.get("document_type") or entry["type"]
            entry["group_id"] = meta.get("document_group_id") or entry["group_id"]
            if meta.get("amends_group_id"):
                entry["amends"] = meta["amends_group_id"]
            entry["aliases"].add(str(title))
            entry["aliases"].add(humanize_stem(stem))
            for alias in meta.get("search_aliases") or []:
                entry["aliases"].add(str(alias))

        group_members: dict[str, list[str]] = defaultdict(list)
        for doc_id, entry in grouped.items():
            group_id = entry["group_id"] or doc_id
            group_members[group_id].append(doc_id)

        for doc_id, entry in grouped.items():
            group_id = entry["group_id"] or doc_id
            record = DocumentRecord(
                document_id=doc_id,
                title=entry["title"],
                aliases=sorted(entry["aliases"], key=len, reverse=True),
                document_group_id=group_id,
                document_type=entry["type"],
                amends_group_id=entry["amends"],
                source_paths=sorted(entry["paths"]),
            )
            catalog.records[doc_id] = record

        alias_entries: list[tuple[str, str]] = []
        for doc_id, record in catalog.records.items():
            for alias in record.aliases:
                alias_entries.append((alias.lower(), doc_id))
        alias_entries.sort(key=lambda item: len(item[0]), reverse=True)
        catalog._alias_entries = alias_entries

        # Acronyms derived at build time (from title/stem/aliases) so short-form
        # queries like "cns act" resolve even when the stored search_aliases were
        # written by an older ingest that never generated them.
        for doc_id, record in catalog.records.items():
            acrs = _acronym_candidates(record.title)
            for alias in record.aliases:
                acrs |= _acronym_candidates(alias)
            for path in record.source_paths:
                acrs |= _acronym_candidates(humanize_stem(Path(path).stem))
            catalog._acronyms[doc_id] = acrs
        return catalog

    @staticmethod
    def _distinctive_tokens(text: str) -> set[str]:
        """Significant tokens of a title/query: drop stopwords, years, and short noise."""
        norm = re.sub(r"[-_]+", " ", text.lower())
        tokens = re.findall(r"[a-z0-9]+", norm)
        out: set[str] = set()
        for t in tokens:
            if len(t) <= 2 or t in _STOPWORDS or t in _DOC_NOISE:
                continue
            if t.isdigit():  # years / section numbers are not document identifiers
                continue
            out.add(t)
        return out

    @staticmethod
    def _query_acronyms(query: str) -> set[str]:
        """Acronyms formed by runs of 2-5 consecutive significant words in the query.

        Lets "anti narcotics force act" map to a document whose identifier is the
        acronym "anf" (e.g. a file named ANF-Act-1997.pdf)."""
        words = [
            w for w in re.findall(r"[a-z]+", re.sub(r"[-_]+", " ", query.lower()))
            if w not in _STOPWORDS and len(w) > 1
        ]
        acronyms: set[str] = set()
        for size in range(2, 6):
            for i in range(len(words) - size + 1):
                acronyms.add("".join(w[0] for w in words[i : i + size]))
        return acronyms

    @staticmethod
    def _token_matches(q_tok: str, t_tok: str) -> bool:
        """Exact, or close enough to be a spelling variant/typo of a title token.

        Tolerant matching is what lets misspellings like "qanuan"/"qanoon" still
        resolve to "qanun". Gated on length and a high similarity ratio so unrelated
        short words don't collide.
        """
        if q_tok == t_tok:
            return True
        if len(t_tok) < 4 or len(q_tok) < 4 or abs(len(q_tok) - len(t_tok)) > 3:
            return False
        return SequenceMatcher(None, q_tok, t_tok).ratio() >= 0.78

    @classmethod
    def _fuzzy_overlap(cls, q_tokens: set[str], ref_tokens: set[str]) -> set[str]:
        """Subset of ref_tokens that some query token matches exactly or fuzzily."""
        covered: set[str] = set()
        for t in ref_tokens:
            if any(cls._token_matches(q, t) for q in q_tokens):
                covered.add(t)
        return covered

    def _token_document_counts(self) -> dict[str, int]:
        """How many documents each distinctive token (title+aliases) belongs to.

        A token owned by exactly one document (e.g. "shahadat", "laundering",
        "narcotic") is a strong identifier on its own — matching it is enough.
        """
        counts: dict[str, set[str]] = defaultdict(set)
        for doc_id, record in self.records.items():
            toks = self._distinctive_tokens(record.title)
            for alias in record.aliases:
                toks |= self._distinctive_tokens(alias)
            for t in toks:
                counts[t].add(doc_id)
        return {t: len(docs) for t, docs in counts.items()}

    def _match_abbreviation(self, query: str) -> str | None:
        """Resolve short-form references like "cns act", "CrPC", "C.N.S.A", "ppc".

        A query word (dotted forms collapsed, any case) matches a document when it
        equals one of the document's generated acronyms, or it is a well-known
        abbreviation whose expansion tokens all belong to that document's title or
        aliases. When several documents share the acronym (a base Act and its
        amendments), the base Act wins; retrieval later soft-boosts the whole group.
        """
        compact = re.sub(r"\.", "", query.lower())  # "c.n.s." → "cns"
        short_words = {w for w in re.findall(r"[a-z]+", compact) if 3 <= len(w) <= 8}
        if not short_words:
            return None
        matched: list[DocumentRecord] = []
        for doc_id, record in self.records.items():
            acronyms = self._acronyms.get(doc_id, set())
            doc_tokens: set[str] | None = None  # built lazily, only for known-abbrev hits
            for word in short_words:
                if word in acronyms:
                    matched.append(record)
                    break
                expansion = _KNOWN_ABBREVIATIONS.get(word)
                if not expansion:
                    continue
                if doc_tokens is None:
                    doc_tokens = self._distinctive_tokens(record.title)
                    for alias in record.aliases:
                        doc_tokens |= self._distinctive_tokens(alias)
                if all(t in doc_tokens for t in expansion):
                    matched.append(record)
                    break
        if not matched:
            return None
        best = min(
            matched,
            key=lambda rec: (0 if rec.document_type == "base_act" else 1, len(rec.title)),
        )
        return best.title

    def match_query(self, query: str) -> str | None:
        """Return display title of the best-matching document mentioned in *query*.

        Stages, in priority order:
        1. A full alias appears verbatim in the query (hyphen/space tolerant;
           short aliases must match as whole words).
        2. Abbreviation match: a query word equals a document acronym ("cns act",
           "ppc 302") or a well-known short form ("crpc"), in any case/spelling.
        3. Token overlap: the query contains the distinctive words of a document's
           title/aliases. Catches short hyphenated user forms like "Qanun-e-Shahadat"
           or "Code of Criminal Procedure" that never appear as an exact substring.
        4. Acronym-of-query match: a run of query words whose initials equal a
           document's short identifier token (e.g. "anti narcotics force" -> "anf").
        """
        q_norm = re.sub(r"[-_]+", " ", query.lower())
        # 1. verbatim alias substring (normalized); short aliases ("anf", "crpc")
        #    only as whole words so they can't fire inside an unrelated word.
        for alias, doc_id in self._alias_entries:
            if len(alias) < 3:
                continue
            a_norm = re.sub(r"[-_]+", " ", alias)
            if len(a_norm) <= 8:
                if re.search(rf"\b{re.escape(a_norm)}\b", q_norm):
                    return self.records[doc_id].title
            elif a_norm in q_norm:
                return self.records[doc_id].title

        # 2. abbreviation / acronym short forms ("cns act", "crpc", "c.n.s.a")
        abbrev_match = self._match_abbreviation(query)
        if abbrev_match:
            return abbrev_match

        # 3. distinctive-token overlap (typo-tolerant)
        q_tokens = self._distinctive_tokens(query)
        if not q_tokens:
            return None
        token_doc_counts = self._token_document_counts()
        best_doc: str | None = None
        best_rank: tuple[int, float, int] = (0, 0.0, 0)
        for doc_id, record in self.records.items():
            # Denominator is the TITLE's distinctive words only; aliases just add
            # extra ways to match (they must not inflate the denominator, or a doc
            # with many aliases would never clear the coverage threshold).
            title_tokens = self._distinctive_tokens(record.title)
            if not title_tokens:
                continue
            match_tokens = set(title_tokens)
            for alias in record.aliases:
                match_tokens |= self._distinctive_tokens(alias)
            # Fuzzy overlap so spelling variants/typos ("qanuan"→"qanun") still count.
            if not self._fuzzy_overlap(q_tokens, match_tokens):
                continue
            covered = self._fuzzy_overlap(q_tokens, title_tokens)
            score = len(covered) / len(title_tokens)
            # Accept when the query covers the document's ENTIRE title set (handles
            # short identifiers like "anf"), has strong partial coverage (>=2 title
            # words and >=60% of them), OR matches a token UNIQUE to this document in
            # the whole corpus (e.g. "shahadat", "laundering" — a decisive identifier
            # on its own, so any misspelling of the other words still resolves).
            full_set = covered == title_tokens and any(len(t) >= 3 for t in covered)
            strong_partial = len(covered) >= 2 and score >= 0.6
            unique_hit = any(
                len(t) >= 5 and token_doc_counts.get(t) == 1 and t not in _GENERIC_TITLE_WORDS
                for t in covered
            )
            if not (full_set or strong_partial or unique_hit):
                continue
            # Prefer the doc with the most covered title words, then best coverage
            # ratio; a unique-token hit breaks ties upward so it isn't ignored.
            # Remaining ties ("…Act" with no year matches a base Act AND its
            # amendments equally) go to the BASE act — the named-instrument filter
            # keys off this title, and the base title keeps the whole group
            # (amendments included) inside the filter, while an amendment title
            # would exclude the base act's chunks.
            rank = (
                len(covered) + (1 if unique_hit else 0),
                score,
                1 if self.records[doc_id].document_type == "base_act" else 0,
            )
            if rank > best_rank:
                best_rank, best_doc = rank, doc_id
        if best_doc is not None:
            return self.records[best_doc].title

        # 4. acronym-of-query match (e.g. "anti narcotics force" -> doc token "anf")
        q_acronyms = self._query_acronyms(query)
        if q_acronyms:
            for doc_id, record in self.records.items():
                doc_tokens = self._distinctive_tokens(record.title)
                for alias in record.aliases:
                    doc_tokens |= self._distinctive_tokens(alias)
                # only short identifier-like tokens are meaningful acronyms
                for tok in doc_tokens:
                    if 2 <= len(tok) <= 6 and tok in q_acronyms:
                        return record.title
        return None

    def related_document_ids(self, title: str) -> list[str]:
        """Siblings in the same group (volumes) plus amendment/base links."""
        record = next((r for r in self.records.values() if r.title == title), None)
        if not record:
            return []
        related: set[str] = set()
        for doc_id, rec in self.records.items():
            if rec.document_group_id == record.document_group_id:
                related.add(doc_id)
            if record.amends_group_id and rec.document_group_id == record.amends_group_id:
                related.add(doc_id)
            if rec.amends_group_id == record.document_group_id:
                related.add(doc_id)
        related.discard(record.document_id)
        return list(related)
