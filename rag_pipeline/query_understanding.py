"""Generic LLM-based query understanding for legal search.

The retriever is keyword + dense hybrid over a general-English embedding model that
does not know Pakistani-legal shorthand, so a question phrased in the user's words
("FIR", "challan", "u/s 9") often misses the provision that is worded differently in
the statute ("first information report", "police report", "under section 9"). This
module rewrites ANY question into the vocabulary the legislation itself uses — concept
terms, expanded abbreviations, colloquial→statutory mappings, and likely statute/section
anchors — so retrieval is driven by meaning rather than surface wording.

It is generic: there is no hardcoded term list. The model interprets each query in
context. The result is appended to the retrieval query only; it never replaces the
user's original wording (which the section/document heuristics still need verbatim).

Failure is always non-fatal: any error, missing API key, or unparsable output yields an
empty expansion and retrieval proceeds on the original query unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-5-mini"

_SYSTEM_PROMPT = """You are a query-understanding component for a Pakistani-law legal search engine. You DO NOT answer the user's legal question. Your only job is to translate the question into the vocabulary the statutes themselves use, so a keyword + semantic search can locate the right provisions.

Given a user question, return ONLY a JSON object with these keys:
- "retrieval_query": a single ENGLISH-language restatement of the question, suitable for keyword + semantic search over English legal text. If the question is in Urdu (or any non-English language), TRANSLATE it to English. Write any section/article/rule numbers as DIGITS, never words — e.g. Urdu "چوبیس" or English "twenty-four" -> "24"; "پچیس اعشاریہ دو" -> "25.2"; keep numbers like "25.2", "9-C", "14(1)(a)" exactly. Keep the statute name in its common English form (e.g. "پولیس رولز" -> "Police Rules"; "پاکستان پینل کوڈ" -> "Pakistan Penal Code"). If the question is already plain English, return it essentially unchanged (only normalising number words to digits). Keep it concise — this is a search query, not an answer.
- "concepts": array of the core legal concepts the question is about, in formal statutory wording.
- "synonyms": array of alternative terms — expand every abbreviation to its full form, and map colloquial wording to the statutory term AND vice-versa. Examples of the KIND of mapping expected (do not limit yourself to these): "FIR" -> "first information report"; "challan" -> "police report", "final report"; "u/s" -> "under section"; "bail" -> "release on bond"; "absolved / let off" -> "not required to / exemption / time-limit / report". Include BOTH the user's term and the statutory term.
- "statute_anchors": array of the specific Acts/Codes plus section or article numbers most likely to govern the question, ONLY when you are confident (e.g. "Code of Criminal Procedure section 173", "section 154"). Use an empty array if unsure.

Rules:
- Base this on general knowledge of Pakistani law. Be precise; never invent a section number you are unsure of.
- Keep each array short (at most ~8 items) and high-signal — these become search terms, so noise hurts.
- Output the JSON object and nothing else."""


@dataclass(slots=True)
class QueryExpansion:
    """Result of interpreting a query into statutory search vocabulary."""

    expanded_terms: str  # space-joined terms appended to the retrieval query
    interpretation: str  # short human-readable summary for the "thinking" panel
    anchors: list[str]   # likely statute/section anchors (may be empty)
    retrieval_query: str = ""  # English-normalised restatement for retrieval (empty if none)

    @property
    def is_empty(self) -> bool:
        return not self.expanded_terms.strip()


_EMPTY = QueryExpansion("", "", [], "")


def _is_reasoning_model(model: str) -> bool:
    return model.lower().startswith(("gpt-5", "o1", "o3", "o4"))


def _expansion_params(model: str) -> dict:
    """Cheap, fast settings: minimal reasoning, JSON output, tight token budget."""
    if _is_reasoning_model(model):
        return {
            "max_completion_tokens": 2000,
            "reasoning_effort": "minimal" if model.lower().startswith("gpt-5") else "low",
            "response_format": {"type": "json_object"},
        }
    return {
        "temperature": 0.0,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
    }


def _coerce_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _parse_json(raw: str) -> dict:
    """Tolerant JSON extraction: strips code fences and grabs the outermost object."""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = re.sub(r"^\s*json", "", raw, flags=re.IGNORECASE).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=512)
def expand_query(question: str, model: str | None = None) -> QueryExpansion:
    """Interpret *question* into statutory search vocabulary via the LLM.

    Cached per (question, model). Returns an empty expansion (retrieval proceeds on the
    original query) when there is no API key or anything goes wrong — never raises.
    """
    question = (question or "").strip()
    if not question:
        return _EMPTY

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _EMPTY
    model = model or os.getenv("OPENAI_MODEL", _DEFAULT_MODEL)

    try:
        from openai import OpenAI

        # Expansion is best-effort and blocks the streaming answer until it returns,
        # so it must fail fast. Disable the SDK's default retries (which would stack
        # 3 attempts × the timeout, freezing the "Thinking…" UI for ~90s on a hung
        # connection) and keep a tight timeout — if expansion is this slow it isn't
        # worth delaying retrieval for.
        client = OpenAI(api_key=api_key, max_retries=0, timeout=12)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            **_expansion_params(model),
        )
        data = _parse_json(response.choices[0].message.content or "")
    except Exception as exc:  # noqa: BLE001 — expansion is best-effort, never block retrieval
        logger.warning("Query expansion failed (%s: %s); proceeding without it", type(exc).__name__, exc)
        return _EMPTY

    concepts = _coerce_list(data.get("concepts"))
    synonyms = _coerce_list(data.get("synonyms"))
    anchors = _coerce_list(data.get("statute_anchors"))
    retrieval_query = str(data.get("retrieval_query") or "").strip()

    # De-duplicate while preserving order; drop terms already in the question so we add
    # signal, not repetition.
    q_lower = question.lower()
    seen: set[str] = set()
    terms: list[str] = []
    for term in [*concepts, *synonyms, *anchors]:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)

    expanded_terms = " ".join(terms)
    interpretation = "; ".join([*concepts[:3], *synonyms[:4]]) or "; ".join(anchors[:3])
    logger.info("Query expansion for %r -> %s", question[:80], expanded_terms[:200])
    return QueryExpansion(
        expanded_terms=expanded_terms.strip(),
        interpretation=interpretation.strip(),
        anchors=anchors,
        retrieval_query=retrieval_query,
    )
