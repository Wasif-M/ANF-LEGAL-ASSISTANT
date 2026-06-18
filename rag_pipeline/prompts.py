"""
Advanced Prompt Templates for Legal Document Q&A
Engineered for accuracy, clarity, and proper source attribution.

Features:
- Query intent classification (6 intent types)
- Intent-specific prompt templates for dynamic responses
- Cross-document reference detection
- Source attribution and formatting
"""

import re

from .utils import chunk_metadata


# ─── Query Intent Classification ───

INTENT_SIMPLE_LOOKUP = "simple_lookup"
INTENT_EXPLANATION = "explanation"
INTENT_COMPARISON = "comparison"
INTENT_PROCEDURAL = "procedural"
INTENT_PENALTY = "penalty"
INTENT_CROSS_REFERENCE = "cross_reference"


def classify_query_intent(question: str) -> str:
    """Classify the query intent to select the appropriate prompt template.
    
    Returns one of 6 intent types:
    - simple_lookup: "What is Section 9?", "Section 161"
    - explanation: "Explain the procedure for...", "What does Section 9 mean?"
    - comparison: "Compare...", "difference between...", "how do X and Y differ?"
    - procedural: "What is the process for...", "steps to..."
    - penalty: "What is the punishment for...", "consequences of..."
    - cross_reference: "Which sections relate to...", "provisions across..."
    """
    q = question.lower().strip()

    # Lookup questions (often after pasted statute text): prefer simple lookup over
    # penalty/explanation triggers from words like "punishment" inside the paste.
    tail = question.strip()[-1200:]
    tail_lower = tail.lower()
    if re.search(
        r"\b(?:what|which|tell\s+me|define|explain|summarize|describe)\b[\s\S]{0,200}\b(?:section|article|rule)\s+[\w.\-()]+\b",
        tail_lower,
        re.DOTALL,
    ):
        return INTENT_SIMPLE_LOOKUP

    # Comparison intent — highest priority
    comparison_patterns = [
        r"\bcompar(?:e|ing|ison)\b",
        r"\bdiff(?:er|erence|erent)\b",
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bbetween\b.*\band\b",
        r"\bhow\s+(?:do|does|is|are)\b.*\bdiffer\b",
        r"\bdistinguish\b",
        r"\bcontrast\b",
        r"\bsimilar(?:ity|ities)?\b",
        r"\bboth\b.*\band\b",
    ]
    for pat in comparison_patterns:
        if re.search(pat, q):
            return INTENT_COMPARISON
    
    # Penalty intent
    penalty_patterns = [
        r"\bpunish(?:ment|able)?\b",
        r"\bpenalt(?:y|ies)\b",
        r"\bsentenc(?:e|ing)\b",
        r"\bimprison(?:ment)?\b",
        r"\bfine(?:s|d)?\b",
        r"\bconsequence(?:s)?\b",
        r"\boffence\b.*\bpunish\b",
        r"\bwhat\s+(?:is|are)\s+the\s+(?:punishment|penalty|consequence)",
    ]
    for pat in penalty_patterns:
        if re.search(pat, q):
            return INTENT_PENALTY
    
    # Procedural intent
    procedural_patterns = [
        r"\bprocedure\b",
        r"\bprocess\b",
        r"\bsteps?\s+(?:to|for|in|of)\b",
        r"\bhow\s+(?:to|do|does|is|can)\b",
        r"\bfil(?:e|ing)\s+(?:a|an|the)?\s*(?:case|complaint|appeal|petition)\b",
        r"\bwhat\s+is\s+the\s+(?:procedure|process)\b",
        r"\brequirements?\s+(?:for|to)\b",
        r"\bconditions?\s+(?:for|to|of)\b",
    ]
    for pat in procedural_patterns:
        if re.search(pat, q):
            return INTENT_PROCEDURAL
    
    # Cross-reference intent
    cross_ref_patterns = [
        r"\brelat(?:ed|ing)\s+(?:sections?|provisions?|articles?)\b",
        r"\bcross[\s-]?referenc\b",
        r"\bprovisions?\s+(?:across|in\s+(?:other|different))\b",
        r"\bwhich\s+(?:sections?|articles?|rules?)\s+(?:relate|apply|govern)\b",
        r"\bapplicable\s+(?:sections?|articles?|laws?|provisions?)\b",
        r"\ball\s+(?:sections?|provisions?)\s+(?:about|regarding|on|related)\b",
    ]
    for pat in cross_ref_patterns:
        if re.search(pat, q):
            return INTENT_CROSS_REFERENCE
    
    # Explanation intent
    explanation_patterns = [
        r"\bexplain\b",
        r"\bwhat\s+does\b.*\bmean\b",
        r"\bmeaning\s+of\b",
        r"\binterpret(?:ation)?\b",
        r"\bdefin(?:e|ition)\b",
        r"\bscope\s+(?:of|and)\b",
        r"\bapplicability\b",
        r"\bwhen\s+does\b.*\bapply\b",
        r"\bpurpose\s+(?:of|behind)\b",
        r"\bwhy\b.*\b(?:enacted|created|introduced)\b",
    ]
    for pat in explanation_patterns:
        if re.search(pat, q):
            return INTENT_EXPLANATION
    
    # Default: simple lookup
    return INTENT_SIMPLE_LOOKUP


# ─── System Prompt ───

SYSTEM_PROMPT = """You are an expert legal document analyst specializing in Pakistani law and legal terminology. Your expertise spans criminal law, evidence law, procedural law, drug control, anti-money laundering, civil service law, and police rules.

Your core responsibilities:
1. Provide accurate, legally sound answers based ONLY on provided document excerpts
2. Cite specific sections, articles, rules, and clauses with their full document references (use the DOCUMENT / SOURCE lines in the excerpts)
3. When excerpts include the same section or article number from different Acts, orders, or rules, choose the one that best matches the user's topic and named law (if any). Briefly note other sources only when they would change the answer or the user asked broadly
4. Distinguish between direct legal provisions and their interpretation
5. Flag ambiguities, alternative interpretations, and limitations
6. Maintain precise legal terminology and correct citations

Citation Format: Use exact references like "Article 76, Pakistan Penal Code" or "Section 161, Qanun-e-Shahadat Order, 1984" or "Section 9, Control of Narcotic Substances Act, 1997"

CRITICAL RULES:
- NEVER fabricate or assume legal provisions not present in the provided documents
- If the documents don't contain sufficient information, state this explicitly
- NEVER RELABEL OR RENUMBER A PROVISION: each excerpt carries its own section/article number
  (in its heading and "Section Number:" line). If the user asks for "Section/Article N" and no
  excerpt actually IS that number, say the provided documents do not contain it — do NOT take a
  different section's text (e.g. an excerpt that is Article 3) and present it under the requested
  number. Many Pakistani statutes use "Article" rather than "Section"; treat the user's "section"
  and the document's "article" as the same when (and only when) the NUMBER matches.
- ABBREVIATIONS & SPELLINGS: Users cite laws by common short forms, in any letter-case or with
  dots — "CNS Act"/"CNSA" = Control of Narcotic Substances Act, 1997; "CrPC"/"Cr.P.C." = Code of
  Criminal Procedure, 1898; "PPC" = Pakistan Penal Code, 1860; "QSO" = Qanun-e-Shahadat Order,
  1984; "AMLA"/"AML Act" = Anti-Money Laundering Act, 2010; "ANF Act" = Anti-Narcotics Force Act,
  1997. Treat such a short form exactly as the full statute name when matching excerpts; NEVER
  refuse merely because the user used an abbreviation, different capitalisation, or a minor
  misspelling of the Act's name
- SUBSECTIONS & CLAUSES: A request for "Section 9(c)", "9-C" or "14(1)(a)" is satisfied by an
  excerpt whose Section Number is the PARENT section (9 or 14) when that excerpt's text contains
  the requested clause — quote the parent provision's opening and the requested clause in full.
  Treat a lettered section as a distinct section only when the statute itself numbers it that
  way (e.g. "20B")
- Prefer the most relevant Act or instrument for the question; do not merge unrelated statutes into one answer
- Use the exact text from documents when quoting provisions
- COMPLETE QUOTES: When a provision contains enumerated sub-clauses (e.g. (a), (b), (c)… or (1), (2), (3)…), quote ALL of them in full. NEVER cut a provision short with "...", "etc.", or "and so on", and never quote only the first clause and imply the rest. If the excerpt contains the whole list, reproduce the whole list.
- AMENDMENTS: When a provision has been amended (the excerpts include both a base Act and an amending Act, or text states a clause was substituted/inserted/omitted), answer with the CURRENT amended text as the operative law, then note what changed and reference the earlier/original provision (Act name and year). Never present superseded text as if it were still in force."""


# ─── Intent-Specific Prompt Templates ───

SIMPLE_LOOKUP_PROMPT = """USER QUESTION: {question}

RELEVANT LEGAL DOCUMENTS:
{context}

{cross_references}

INSTRUCTIONS:
You are answering a direct lookup question. Be precise and complete, and format the
answer as clean GitHub-flavored Markdown using EXACTLY this structure:

## <The section heading / title of the provision>

<One to three sentences of plain-language answer that directly address the question.>

> <opening/definitional text of the provision, e.g. the (1) or (2) sub-section>
> (a) <first clause, on its own line>
> (b) <second clause, on its own line>
> (c) <…and so on — EVERY clause on its own separate line>

**Citation:** <Section/Article number, exact Act name, Year — matching the excerpt you used>

**Significance:** <One sentence on why the provision matters. Omit this whole line if not directly relevant.>

VERIFY BEFORE YOU ANSWER:
- If the user named a specific Section/Article number, check the excerpts' "Section Number:" lines
  and headings (a document "Article 6" satisfies a request for "Section 6").
- An excerpt's IDENTITY is its "(Section Number: N)" value and the number at the START of its
  heading — NOT any section number that appears INSIDE its text. A provision routinely refers to
  other sections ("specially empowered under Section 30", "for the purposes of Section 2", or it
  may even BEGIN with such a reference because the marginal-note title and the operative text were
  split by the PDF). Those internal references are cross-references; they do NOT change which
  section the excerpt is. Never reclassify an excerpt as a different section because its body
  mentions or begins with "Section X".
- If an excerpt's "(Section Number:)" equals the requested number, that excerpt IS the requested
  section — ANSWER from it. Trust it even when: its parent/heading looks garbled or noisy (e.g.
  "& Sched , ]", stray brackets, PDF artifacts); it sits under a generic parent like
  "COMMENTS"/"NOTES"/"PRELIMINARY"; the title and the operative sentence are split between the
  heading line and the body (reassemble them into one provision); or the body opens with a
  cross-reference. Reconstruct and quote the full provision from the heading + body together.
- SUBSECTION/CLAUSE REQUESTS: if the user asked for a sub-part such as "9(c)", "9-C", "9c" or
  "14(1)(a)", an excerpt whose "(Section Number:)" equals the PARENT number (9, or 14 / 14(1))
  CONTAINS the requested provision — answer from it: quote the parent section's opening text and
  the requested clause in full, and name the clause in your citation. Do NOT reply "not found"
  merely because no excerpt is labelled with the exact sub-part.
- ABBREVIATED ACT NAMES: the user may name the law by a short form in any case ("cns act" =
  Control of Narcotic Substances Act, 1997; "crpc" = Code of Criminal Procedure, 1898; "ppc" =
  Pakistan Penal Code; "qso" = Qanun-e-Shahadat Order, 1984; "anf act" = Anti-Narcotics Force
  Act, 1997). An excerpt from the corresponding full-named Act (or its amendment Acts) DOES
  match the user's named law.
- Only if NO excerpt's "(Section Number:)" matches the requested number (nor, for a clause-level
  request, its parent section), reply: "The provided documents do not contain Section/Article <N>
  of <Act>." Never substitute a different section's text under the asked number.

FORMATTING RULES:
- Start with the section heading as a "## " heading whenever the provision has a title.
- Put the definitional/opening text first, then each enumerated clause ((a), (b), (c)… or
  (i), (ii)… or (1), (2)…) on ITS OWN separate line inside the blockquote — one clause per line.
- Every quoted line must start with "> ". Reproduce the FULL provision — never stop after the
  first clause and never abbreviate with "..." or "etc.".
- Quoting "verbatim" means the same WORDS, not the same line-wrapping. The source excerpts are
  wrapped at arbitrary points by the PDF; you MUST rejoin those wrapped lines. Put a line break
  ONLY before an enumerated clause ((a), (1), (i)…) — never in the middle of a sentence. A clause
  with no sub-parts must be a single continuous line.
- Use the bold labels (**Citation:**, **Significance:**) exactly as shown above.
- If the same number appears in different laws, answer from the source matching the user's
  question (named Act, subject, or heading); mention other instruments in one sentence only
  when it would change the answer.

ANSWER:"""

# Shared formatting standard appended to every intent template so EVERY response
# type renders with the same structure: bold "## " headings, blockquoted verbatim
# provisions, and bold inline labels. (SIMPLE_LOOKUP carries its own detailed copy.)
_FORMAT_STANDARD = """
FORMATTING (GitHub-flavored Markdown — apply to the WHOLE answer):
- Use a "## " heading for EVERY section title listed above. Do NOT write a section as
  a plain paragraph or as a numbered "1. **Label**" line — each section title MUST be a
  "## " heading. Headings render in bold automatically; never wrap a "## " heading in
  asterisks.
- Quote any verbatim statute text inside a "> " blockquote, with each enumerated clause
  ((a),(b),(c)… / (1),(2)… / (i),(ii)…) on ITS OWN "> " line. Rejoin words the PDF wrapped
  across lines so sentences are not broken; break a line only before an enumerated clause.
- Use **bold** for inline labels (e.g. **Citation:**, **Step 1:**, **Imprisonment:**) and
  for key terms.
- Keep exact section/article numbers and full Act names in every citation.
- Omit any heading whose section genuinely does not apply, rather than writing "N/A".
- NEVER use HTML tags (no <br>, <b>, <ul>, etc.) — they are shown as literal text, not
  rendered. Use real Markdown only: actual line breaks, "-" bullets, "> " blockquotes.
- TABLE CELLS hold only SHORT plain summaries (a phrase, ideally under ~15 words). NEVER
  put verbatim statute text, "> " blockquote markers, enumerated clause lists ((a),(i),(1)…),
  or "<br>" inside a table cell. Any full provision text goes in a "> " blockquote OUTSIDE
  the table, under its own heading.
"""


EXPLANATION_PROMPT = """USER QUESTION: {question}

RELEVANT LEGAL DOCUMENTS:
{context}

{cross_references}

INSTRUCTIONS:
You are providing a detailed legal explanation. Format the answer with these "## "
headings, in this order (omit a heading only if its content genuinely does not apply):

## Direct Answer
A clear 2-3 sentence answer to the question.

## Legal Provision
The exact text of the relevant section(s), quoted in a "> " blockquote.

## Interpretation
What the provision means in practical legal context.

## Scope & Applicability
What situations it covers and any exceptions.

## Related Provisions
Connected sections from the same or other documents.

## Key Takeaway
The single most important point.

Then add a final bold label: **Citation:** <Section/Article number, exact Act name, Year>.
""" + _FORMAT_STANDARD + """
ANSWER:"""

COMPARISON_PROMPT = """USER QUESTION: {question}

RELEVANT LEGAL DOCUMENTS:
{context}

{cross_references}

INSTRUCTIONS:
You are answering a comparison question.

FIRST decide whether there are genuinely TWO OR MORE distinct provisions/laws to compare:
- If only ONE provision is actually in scope (the question or the excerpts point to a single
  section/Act and there is nothing to compare it against), DO NOT build a comparison table.
  Instead answer as a normal lookup using these headings:
    ## <Section title>
    ## Provision   (quote the full text in a "> " blockquote, each clause on its own "> " line)
    ## Explanation
    **Citation:** <Section/Article number, exact Act name, Year>
- Only when there really are 2+ provisions to compare, use the comparison layout below.

COMPARISON LAYOUT (two or more provisions):

## Overview
What is being compared (1-2 sentences).

## Comparison
A Markdown table whose cells contain ONLY short plain-language summaries — a brief phrase per
cell, never verbatim text, never "<br>", never "> ", never enumerated clause lists:
   | Aspect | <Section/Act 1> | <Section/Act 2> |
   |--------|-----------------|-----------------|
   | Scope | short phrase | short phrase |
   | Key point | short phrase | short phrase |
   | Penalty (if any) | short phrase | short phrase |
   | Exceptions | short phrase | short phrase |

## Provisions Quoted
For each provision being compared, quote its operative text in a "> " blockquote (each
enumerated clause on its own "> " line). This is where full statute text goes — NOT the table.

## Key Differences
The most important differences, each with a specific citation.

## Similarities
Where the provisions align.

## Practical Implications
What these differences mean in practice.

Use exact section/article numbers and document names throughout.
""" + _FORMAT_STANDARD + """
ANSWER:"""

PROCEDURAL_PROMPT = """USER QUESTION: {question}

RELEVANT LEGAL DOCUMENTS:
{context}

{cross_references}

INSTRUCTIONS:
You are explaining a legal procedure. Use these "## " headings:

## Overview
What procedure is described and under which law (1-2 sentences).

## Step-by-Step Process
A numbered or bulleted list, each step with its citation:
   - **Step 1:** [Action] — (Section X, Act Name)
   - **Step 2:** [Action] — (Section Y, Act Name)
   - …continue for all steps.

## Requirements
Prerequisites, documents, or conditions needed.

## Timeline
Any time limits or deadlines specified in the law.

## Authorities Involved
Which courts, officers, or bodies are responsible.

## Important Notes
Exceptions, special circumstances, or common pitfalls.

Cite the specific section for EACH step. Use the exact legal terminology from the documents.
""" + _FORMAT_STANDARD + """
ANSWER:"""

PENALTY_PROMPT = """USER QUESTION: {question}

RELEVANT LEGAL DOCUMENTS:
{context}

{cross_references}

INSTRUCTIONS:
You are detailing penalties and punishments under Pakistani law. Use these "## " headings:

## Offence
What constitutes the offence (cite the section).

## Classification
Type of offence (cognizable/non-cognizable, bailable/non-bailable, if stated).

## Punishment
   - **Imprisonment:** duration (minimum and maximum if specified)
   - **Fine:** amount (minimum and maximum if specified)
   - **Both:** if imprisonment AND fine apply

## Aggravating Factors
Circumstances that increase the penalty.

## Mitigating Factors
Circumstances that may reduce the penalty.

## Additional Consequences
Forfeiture, disqualification, confiscation, etc.

## Related Offences
Similar offences under the same or other acts.

Quote the EXACT penalty provisions from the documents. Do not paraphrase penalty amounts or durations.
""" + _FORMAT_STANDARD + """
ANSWER:"""

CROSS_REFERENCE_PROMPT = """USER QUESTION: {question}

RELEVANT LEGAL DOCUMENTS:
{context}

{cross_references}

INSTRUCTIONS:
You are mapping related provisions across legal documents. Use these "## " headings:

## Topic Overview
What legal topic is being examined (1-2 sentences).

## Provisions by Document
Group by document using a "### " sub-heading per document, then bullet its sections:
   ### [Document 1 Name]
   - Section X: [brief description]
   - Section Y: [brief description]
   ### [Document 2 Name]
   - Section A: [brief description]

## How They Interconnect
How provisions reference or complement each other.

## Gaps & Overlaps
Where documents overlap or where gaps exist.

## Practical Application
When you would consult which document.

List each relevant section from each document in the excerpts; where the same number appears in different laws, separate them clearly by instrument.
""" + _FORMAT_STANDARD + """
ANSWER:"""

# Legacy comprehensive prompt (fallback)
COMPREHENSIVE_LEGAL_PROMPT = """You are an expert legal analyst specializing in Pakistani law. Analyze the provided legal documents to answer the user's question accurately and comprehensively.

USER QUESTION: {question}

LEGAL DOCUMENTS AND EXCERPTS:
{context}

{cross_references}

INSTRUCTIONS:
Answer the question accurately and comprehensively, formatted with these "## " headings
(omit one only if its content genuinely does not apply):

## Answer
A direct answer to the question in 1-3 sentences.

## Legal Basis
The relevant section/article text quoted in a "> " blockquote, with exact numbers.

## Explanation
What the provisions mean and their legal purpose.

## Scope & Limits
What the provision covers and does not, including exceptions.

## Related Provisions
Connected sections or laws across the documents.

Then add a final bold label: **Citation:** <exact references, e.g. "Section 161, Qanun-e-Shahadat Order, 1984">.
If the documents do not fully address the question, say so explicitly. If the same section
number appears in more than one Act, prioritize the instrument that fits the question and
note others briefly when relevant.
""" + _FORMAT_STANDARD + """
ANSWER:"""

RECOMMENDATIONS_PROMPT = """Based on the user's question and situation, identify which specific legal rules, sections, and articles are applicable.

QUESTION: {question}

RELEVANT EXCERPTS:
{context}

INSTRUCTIONS:
Use these "## " headings (omit one only if its content genuinely does not apply):

## Directly Applicable Sections
Each section/article with its exact reference and a one-line summary.

## Why Each Applies
1-2 sentences on how each provision relates to the question.

## Key Provisions
Specific provisions and conditions that apply.

## Procedural Rules
Procedural sections that govern this situation.

## Legal Classification
Type of offence or violation, if applicable.

## Possible Consequences
Potential legal consequences based on the provisions.

## Cross-Referenced Laws
Related provisions in other documents.

## Limitations
Note if the documents do not fully address the situation.
""" + _FORMAT_STANDARD


# ─── Helper Functions ───

def extract_document_sources(retrieved_chunks: list) -> dict:
    """Extract document metadata from retrieved chunks.
    
    Returns a dictionary mapping document names to their sections/articles quoted.
    """
    sources = {}
    for chunk in retrieved_chunks:
        if hasattr(chunk, "chunk"):
            metadata = chunk_metadata(chunk.chunk)
            doc_title = metadata.get("title", "Unknown Document")
            section_path = metadata.get("section_path", [])
            section_number = metadata.get("section_number")
            
            if doc_title not in sources:
                sources[doc_title] = {
                    "sections": set(),
                    "count": 0
                }
            
            if section_number:
                sources[doc_title]["sections"].add(section_number)
            if section_path:
                sources[doc_title]["sections"].update(section_path)
            sources[doc_title]["count"] += 1
    
    return sources


def detect_amendment_relationship(retrieved_chunks: list) -> str:
    """Flag when the excerpts include both a base Act and an amendment of the SAME
    instrument (i.e. the queried provision has been amended).

    Returns an instruction telling the model to answer from the amended/current text
    and reference the earlier provision, or "" when no such relationship is present.
    A single self-consolidated document (e.g. "...amended upto 2020") does not count.
    """
    groups: dict[str, dict[str, set]] = {}
    for chunk in retrieved_chunks:
        c = chunk.chunk if hasattr(chunk, "chunk") else chunk
        meta = chunk_metadata(c)
        group = meta.get("document_group_id")
        if not group:
            continue
        entry = groups.setdefault(group, {"types": set(), "titles": set()})
        entry["types"].add(meta.get("document_type", "unknown"))
        entry["titles"].add(meta.get("display_title") or meta.get("title", ""))

    notes: list[str] = []
    for info in groups.values():
        # need at least two distinct documents in the group, and an amendment among them
        if len(info["titles"]) >= 2 and "amendment" in info["types"]:
            titles = ", ".join(sorted(t for t in info["titles"] if t))
            notes.append(titles)

    if not notes:
        return ""

    return (
        "AMENDMENT NOTICE: The excerpts include a base law together with an amending "
        "law for the same instrument (" + "; ".join(notes) + "). Answer with the CURRENT, "
        "amended position as the operative law, quoting the amended text. Then briefly state "
        "what changed and reference the earlier/original provision (with its Act name and year) "
        "so the user can see the prior position. Do not present the superseded text as current."
    )


def detect_cross_references(question: str, retrieved_chunks: list) -> str:
    """Detect if similar topics are mentioned in multiple documents.
    
    Returns a summary of cross-references found across documents.
    """
    sources = extract_document_sources(retrieved_chunks)
    
    if len(sources) <= 1:
        return "No cross-references detected - information from single source."
    
    cross_ref_summary = f"CROSS-DOCUMENT INFORMATION: Results retrieved from {len(sources)} different legal document(s):\n"
    
    for doc_name, info in sources.items():
        sections_str = ", ".join(sorted(info["sections"])) if info["sections"] else "Various sections"
        cross_ref_summary += f"- {doc_name}: {sections_str} ({info['count']} excerpt(s))\n"
    
    cross_ref_summary += (
        "\nIMPORTANT: The same section or article number may appear in more than one instrument. "
        "Use the excerpt whose Act name and substance match the user's question. "
        "If the user did not name a specific law, state which source you are using and mention other "
        "instruments in one sentence only when they would materially change the answer.\n"
    )
    
    return cross_ref_summary.strip()


def format_context_with_sources(retrieved_chunks: list) -> str:
    """Format retrieved chunks with their source documents and metadata.
    
    Returns formatted text with clear source attribution per chunk.
    """
    formatted = ""
    current_doc = None
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        if hasattr(chunk, "chunk"):
            metadata = chunk_metadata(chunk.chunk)
            doc_title = metadata.get("title", "Unknown Document")
            section_label = metadata.get("section_label", "Unknown Section")
            section_number = metadata.get("section_number", "")
            text = chunk.chunk.text
            source_path = chunk.chunk.source_path
            
            # Add document header if switching documents
            if doc_title != current_doc:
                formatted += f"\n{'='*80}\nDOCUMENT: {doc_title}\nSOURCE: {source_path}\n{'='*80}\n"
                current_doc = doc_title
            
            # Add section reference with metadata
            section_info = f"[EXCERPT {i}] [{section_label}]"
            if section_number:
                section_info += f" (Section Number: {section_number})"
            formatted += f"\n{section_info}\n"
            formatted += f"{text}\n"
    
    return formatted


def build_legal_prompt(
    question: str,
    context: str,
    retrieved_chunks: list = None,
    prompt_type: str = "comprehensive",
    query_intent: str = None,
) -> str:
    """
    Build a specialized legal prompt with dynamic source attribution and cross-references.
    
    Auto-detects query intent and selects the appropriate template unless overridden.
    
    Args:
        question: User's legal question
        context: Retrieved document excerpts
        retrieved_chunks: List of RetrievedChunk objects with metadata
        prompt_type: Type of prompt to use (legacy, overridden by query_intent)
        query_intent: Explicit intent override. If None, auto-detected.
    
    Returns:
        Formatted prompt string with cross-references
    """
    # Auto-detect intent if not provided
    if query_intent is None:
        query_intent = classify_query_intent(question)
    
    # Map intent to template
    intent_templates = {
        INTENT_SIMPLE_LOOKUP: SIMPLE_LOOKUP_PROMPT,
        INTENT_EXPLANATION: EXPLANATION_PROMPT,
        INTENT_COMPARISON: COMPARISON_PROMPT,
        INTENT_PROCEDURAL: PROCEDURAL_PROMPT,
        INTENT_PENALTY: PENALTY_PROMPT,
        INTENT_CROSS_REFERENCE: CROSS_REFERENCE_PROMPT,
    }
    
    # Legacy prompt_type mapping (backwards compatibility)
    legacy_templates = {
        "comprehensive": COMPREHENSIVE_LEGAL_PROMPT,
        "recommendations": RECOMMENDATIONS_PROMPT,
    }
    
    # Select template: intent-based takes priority over legacy prompt_type
    if prompt_type == "recommendations":
        template = RECOMMENDATIONS_PROMPT
    elif query_intent in intent_templates:
        template = intent_templates[query_intent]
    else:
        template = legacy_templates.get(prompt_type, COMPREHENSIVE_LEGAL_PROMPT)
    
    # Extract source information if chunks are provided
    cross_ref_data = ""
    context_with_sources = context
    
    if retrieved_chunks:
        cross_ref_data = detect_cross_references(question, retrieved_chunks)
        amendment_note = detect_amendment_relationship(retrieved_chunks)
        if amendment_note:
            cross_ref_data = f"{amendment_note}\n\n{cross_ref_data}" if cross_ref_data else amendment_note
        context_with_sources = format_context_with_sources(retrieved_chunks)
    
    # Format the template
    return template.format(
        question=question,
        context=context_with_sources if retrieved_chunks else context,
        cross_references=cross_ref_data,
    )


def build_legal_prompt_legacy(question: str, context: str, prompt_type: str = "general") -> str:
    """Legacy function for backwards compatibility."""
    return build_legal_prompt(question, context, prompt_type="comprehensive")


def get_system_prompt() -> str:
    """Get the system prompt for legal QA."""
    return SYSTEM_PROMPT


# ─── Language handling (English / Urdu) ───

# Urdu / Arabic-script Unicode ranges. We only treat a question as Urdu when a
# meaningful share of its letters are in these ranges (so an English question
# that happens to contain one Urdu word stays English).
_URDU_RANGE = (
    "؀-ۿ"   # Arabic
    "ݐ-ݿ"   # Arabic Supplement
    "ࢠ-ࣿ"   # Arabic Extended-A
    "ﭐ-﷿"   # Arabic Presentation Forms-A
    "ﹰ-﻿"   # Arabic Presentation Forms-B
)
_URDU_CHAR_RE = re.compile(f"[{_URDU_RANGE}]")
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def detect_language(text: str) -> str:
    """Best-effort language detection. Returns "ur" if the text is predominantly
    Urdu/Arabic script, otherwise "en". Used when the caller asks for "auto"."""
    if not text:
        return "en"
    letters = _LETTER_RE.findall(text)
    if not letters:
        return "en"
    urdu = len(_URDU_CHAR_RE.findall(text))
    return "ur" if (urdu / len(letters)) >= 0.30 else "en"


def resolve_language(requested: str | None, question: str) -> str:
    """Map a requested language ("auto"/"en"/"ur"/None) to a concrete "en"/"ur"."""
    req = (requested or "auto").lower()
    if req in ("ur", "urdu"):
        return "ur"
    if req in ("en", "english"):
        return "en"
    return detect_language(question)


# Appended to the user prompt when the answer must be in Urdu. Keeps the SAME
# markdown layout the English templates produce (headings, blockquotes with each
# clause on its own line, bold labels) so the frontend renders Urdu answers with
# the same structure — only the prose language changes.
_URDU_DIRECTIVE = """

──────────────────────────────────────────
LANGUAGE: ANSWER IN URDU (اردو میں جواب دیں)
──────────────────────────────────────────
Write the ENTIRE explanatory answer in clear, natural, fluent Urdu — the kind a
Pakistani lawyer would use when explaining the law to a client. Do NOT answer in
English. Important rules:

1. STRUCTURE — keep EXACTLY the same markdown layout described above (the same
   `##` headings, the `>` blockquote for the verbatim provision with each
   enumerated clause on its own `> ` line, and the bold labels). Only translate
   the labels: use **حوالہ:** for "Citation", **اہمیت:** for "Significance",
   **خلاصہ:** for a summary, etc. The headings should be in Urdu.
2. CITATIONS STAY PRECISE — section/article numbers and the official Act names
   must remain accurate. Write references like: «دفعہ 9, Control of Narcotic
   Substances Act, 1997» — keep the English statute name and the number intact,
   put only the connecting words in Urdu (دفعہ / آرٹیکل / قانون). Do NOT
   translate or invent Urdu names for the Acts.
3. VERBATIM TEXT — the law itself is in English in the source documents. Quote
   the operative provision in its original English inside the `>` blockquote
   (do not translate the statute text), then explain it in Urdu around the quote.
4. Natural, fluent, well-organised Urdu — short clear sentences, correct legal
   terminology (e.g. سزا، جرم، ضمانت، تفتیش، استغاثہ). Avoid stiff or robotic
   word-for-word translation.
5. Never fabricate provisions; the same grounding rules above apply.
"""


def language_directive(lang: str) -> str:
    """Return the instruction block to append to a legal prompt for the target
    language. Empty for English (the templates are already English)."""
    return _URDU_DIRECTIVE if lang == "ur" else ""
