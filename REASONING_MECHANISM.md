# Adaptive Reasoning Mechanism

How the legal RAG decides, **per query**, whether the LLM should "think" (spend reasoning
tokens) or answer directly — and how scenario-based questions are handled.

> **Short answer:** Reasoning is NOT applied at full strength to every query. Every query is
> first analyzed with a zero-cost regex classifier; only queries that actually need
> multi-provision reasoning (scenarios, comparisons, long questions) get real thinking.
> Verbatim section lookups run at `minimal` effort — effectively no thinking — so they stay
> fast and cheap.

---

## 1. Pipeline overview

```
question
   │
   ▼
classify_query_intent(question)          rag_pipeline/prompts.py:27
   │   pure regex, no LLM call, ~0 ms
   │   → one of 6 intents
   ▼
_reasoning_effort_for(model, q, intent)  api.py:51
   │   intent + question length → minimal / low / medium
   ▼
_completion_params(...)                  api.py:67
   │   reasoning model → {reasoning_effort, max_completion_tokens}
   │   non-reasoning model → {temperature: 0.3, max_tokens}
   ▼
stream_answer(...)                       api.py:313
       effort != "minimal" → SSE "Thinking through the provisions (medium reasoning)…"
       effort == "minimal" → SSE "Composing the answer from the retrieved provisions…"
```

The same intent label is reused for two other things, so classification is not wasted work
even when no thinking is needed:

- **Prompt template selection** — `build_legal_prompt()` picks one of 6 intent-specific
  templates (`prompts.py:572`).
- **Retrieval strategy** — comparison intent unlocks multi-document retrieval instead of
  locking onto a single section/document (`retrieval.py:294,318,354`).

## 2. Why analyze the query at all? (cost rationale)

The analysis step is a handful of compiled regexes — microseconds, zero API cost. The thing
it gates — reasoning tokens on `gpt-5-mini` — is the expensive part (billed as output tokens,
and adds seconds of latency). So the trade is:

|                             | Cost                    | Latency          |
| --------------------------- | ----------------------- | ---------------- |
| Regex intent classification | free                    | ~0 ms            |
| `minimal` effort answer   | cheapest                | fastest          |
| `medium` effort answer    | reasoning tokens billed | +several seconds |

Running `medium` on everything would make "What is Section 9?" as slow and expensive as a
full scenario analysis, for no quality gain — the answer is a verbatim quote from the
retrieved chunk. Running `minimal` on everything would make scenario questions shallow
(single-provision answers where 3–4 statutes interact). The classifier is what lets each
query pay only for the reasoning it needs.

## 3. Intent classification (`classify_query_intent`, prompts.py:27)

Checked in priority order; first match wins:

| Priority | Intent                        | Example triggers                                                                                                                                              |
| -------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0        | `simple_lookup` (fast-path) | "what/which/define/explain … Section/Article/Rule N" near the end of the question — wins even if pasted statute text above contains words like "punishment" |
| 1        | `comparison`                | compare, difference, vs, versus, between…and, distinguish, contrast, both…and                                                                               |
| 2        | `penalty`                   | punishment, penalty, sentence, imprisonment, fine, consequences                                                                                               |
| 3        | `procedural`                | procedure, process, steps to, how to/do/can, filing a case/appeal, requirements for                                                                           |
| 4        | `cross_reference`           | related sections, cross-reference, which sections apply/govern, applicable sections/laws                                                                      |
| 5        | `explanation`               | explain, what does…mean, meaning of, interpret, define, scope of, when does…apply                                                                           |
| —       | `simple_lookup` (default)   | anything else                                                                                                                                                 |

## 4. Effort mapping (`_reasoning_effort_for`, api.py:51)

```python
if intent == simple_lookup and len(question) <= 200:   effort = "minimal"
elif intent in (explanation, penalty) and len(q) <= 300: effort = "low"
else:                                                    effort = "medium"
# o1/o3/o4 don't accept "minimal" → bumped to "low" (gpt-5* keeps "minimal")
```

| Query shape                                        | Intent                 | Length | Effort            | Thinks? |
| -------------------------------------------------- | ---------------------- | ------ | ----------------- | ------- |
| "What is Section 9 CNSA?"                          | simple_lookup          | short  | **minimal** | No      |
| "Explain section 24 of qanun-e-shahadat"           | explanation            | short  | **low**     | Briefly |
| "What is the punishment for possession of heroin?" | penalty                | short  | **low**     | Briefly |
| "Compare bail provisions in CNSA vs CrPC"          | comparison             | any    | **medium**  | Yes     |
| Any procedural / cross-reference question          | procedural / cross_ref | any    | **medium**  | Yes     |
| Any question > 200–300 chars (incl. scenarios)    | any                    | long   | **medium**  | Yes     |

The frontend only shows the thinking indicator when effort ≠ minimal (`api.py:337–340`),

so the UI honestly reflects whether the model is reasoning.

## 5. Scenario-based questions

There is **no dedicated `scenario` intent** — scenarios are caught by two overlapping nets,
which in practice always land them on `medium` (or at least `low`) effort:

**Net 1 — length.** A scenario is a narrative: *"A person was arrested at Karachi airport
with 2 kg of heroin concealed in luggage; he claims he didn't know the contents. What
charges and defenses apply?"* That's well over 200 characters, so even if intent
classification falls through to the `simple_lookup` default, the `q_len <= 200` guard fails
→ `medium`.

**Net 2 — phrasing.** The "what should happen" tail of a scenario matches one of the
heavier intents anyway:

- "…what punishment does he face?" → `penalty`
- "…what is the procedure for his remand?" → `procedural`
- "…which sections apply to this case?" → `cross_reference`
- "…how is this different from simple possession?" → `comparison`

All of `procedural`, `cross_reference`, `comparison` → `medium` unconditionally.

**Downstream, a scenario query then gets:**

1. **Multi-document retrieval** — scenarios rarely name a single section, so the
   single-section/single-document lock in `retrieval.py` doesn't engage and chunks are
   gathered across statutes (CNSA + CrPC + Evidence, etc.).
2. **An intent-matched prompt template** (penalty / procedural / cross-reference) that
   instructs the model to map facts → elements of the offence → applicable provisions,
   under the no-relabel grounding rules (it may only cite sections actually present in the
   retrieved excerpts).
3. **`medium` reasoning effort** with `max_completion_tokens = answer + 4000` headroom
   (`api.py:72`) so thinking tokens don't starve the final answer.

**Known edge case:** a *short* scenario whose tail matches `penalty`
("Caught with 50 g heroin — punishment?") gets `low`, not `medium`. That's acceptable:
short single-offence questions are effectively penalty lookups. If deeper analysis is ever
needed there, the fix is a scenario cue list (narrative markers like "a person", "my
client", "was arrested", "caught with") that forces `medium` regardless of length.

## 6. Model handling notes

- Default model: `gpt-5-mini` (`api.py:35`), overridable via `OPENAI_MODEL` in `.env`.
- `_is_reasoning_model()` (`api.py:47`): `gpt-5*`, `o1`, `o3`, `o4` → reasoning params;
  anything else (e.g. `gpt-4o`) falls back to `temperature=0.3` / `max_tokens` and the
  whole effort mechanism is bypassed.
- Reasoning models reject `temperature`/`max_tokens`, hence the separate param branch in
  `_completion_params()`.
