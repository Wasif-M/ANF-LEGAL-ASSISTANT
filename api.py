from __future__ import annotations

from typing import Optional
import io
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import sqlite3

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

import db
from rag_pipeline import PipelineConfig, RAGPipeline
from rag_pipeline.prompts import (
    build_legal_prompt,
    get_system_prompt,
    classify_query_intent,
    detect_language,
    language_directive,
    resolve_language,
)
from rag_pipeline.query_understanding import expand_query
from rag_pipeline.utils import (
    build_section_references_from_chunks,
    chunk_metadata,
    clean_markdown_formatting,
    reflow_provision_text,
)


class QueryRequest(BaseModel):
    question: str
    max_chars: Optional[int] = 15000
    prompt_type: Optional[str] = "general"
    conversation_id: Optional[int] = None
    # "auto" (detect from the question), "en", or "ur". Controls the answer language.
    language: Optional[str] = "auto"


class TTSRequest(BaseModel):
    text: str
    # "en", "ur", or "auto" (detect from the text). Picks the accent/voice tuning.
    language: Optional[str] = "auto"
    voice: Optional[str] = None


class SignupRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username: str  # username or email
    password: str


class ConversationCreate(BaseModel):
    title: Optional[str] = "New Chat"


class ConversationRename(BaseModel):
    title: str


class RatingRequest(BaseModel):
    rating: int  # 1-5 stars


# ─── LLM model handling ───
# Default is a cost-effective reasoning model: simple lookups run at minimal
# reasoning effort (fast/cheap), scenario and comparison questions get real
# thinking. Override with OPENAI_MODEL in .env.
DEFAULT_MODEL = "gpt-5-mini"

_INTENT_LABELS = {
    "simple_lookup": "direct section lookup",
    "explanation": "legal explanation",
    "comparison": "comparison of provisions",
    "procedural": "procedural question",
    "penalty": "penalty/punishment question",
    "cross_reference": "cross-reference mapping",
}


def _is_reasoning_model(model: str) -> bool:
    return model.lower().startswith(("gpt-5", "o1", "o3", "o4"))


def _reasoning_effort_for(model: str, question: str, query_intent: str) -> str:
    """Scale thinking to the question: cheap for verbatim lookups, deeper for
    scenario/comparison questions that need multi-provision reasoning."""
    q_len = len(question.strip())
    if query_intent == "simple_lookup" and q_len <= 200:
        effort = "minimal"
    elif query_intent in ("explanation", "penalty") and q_len <= 300:
        effort = "low"
    else:
        effort = "medium"
    # o-series models do not accept "minimal"
    if effort == "minimal" and not model.lower().startswith("gpt-5"):
        effort = "low"
    return effort


def _completion_params(model: str, question: str, query_intent: str, answer_tokens: int = 4000) -> dict:
    """Per-model request kwargs: reasoning models reject `temperature`/`max_tokens`
    and need headroom in `max_completion_tokens` for their reasoning tokens."""
    if _is_reasoning_model(model):
        return {
            "max_completion_tokens": answer_tokens + 4000,
            "reasoning_effort": _reasoning_effort_for(model, question, query_intent),
        }
    return {"temperature": 0.3, "max_tokens": answer_tokens}


def _sse(payload: dict) -> str:
    """Encode one server-sent event."""
    return f"data: {json.dumps(payload)}\n\n"


app = FastAPI(
    title="Legal QA RAG API",
    description="Advanced legal document QA with hybrid retrieval",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    db.init_db()
    config = PipelineConfig()
    app.state.pipeline = RAGPipeline(config)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "legal-qa-api",
        "version": "1.0.0"
    }


@app.get("/documents")
def list_documents() -> dict:
    """List the legal documents currently indexed in the pipeline."""
    from pathlib import Path as _Path

    pipeline: RAGPipeline = app.state.pipeline
    docs: dict[str, dict] = {}
    for chunk in pipeline.retriever._chunks:
        entry = docs.get(chunk.source_path)
        if entry is None:
            meta = chunk_metadata(chunk)
            entry = {
                "file": _Path(chunk.source_path).name,
                "title": meta.get("title") or _Path(chunk.source_path).stem,
                "chunks": 0,
            }
            docs[chunk.source_path] = entry
        entry["chunks"] += 1
    documents = sorted(docs.values(), key=lambda d: d["title"].lower())
    return {
        "documents": documents,
        "total_documents": len(documents),
        "total_chunks": sum(d["chunks"] for d in documents),
    }


# ─── Auth ───

def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """Resolve the logged-in user from a `Bearer <token>` header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = db.get_user_by_token(authorization.removeprefix("Bearer ").strip())
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


def get_optional_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return db.get_user_by_token(authorization.removeprefix("Bearer ").strip())


@app.post("/auth/signup")
def signup(req: SignupRequest) -> dict:
    username = req.username.strip()
    email = req.email.strip()
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Please enter a valid email address")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    try:
        user = db.create_user(username, email, req.password)
    except sqlite3.IntegrityError as e:
        field = "Email" if "email" in str(e).lower() else "Username"
        raise HTTPException(status_code=409, detail=f"{field} is already registered")
    token = db.create_session(user["id"])
    return {"token": token, "user": user}


@app.post("/auth/login")
def login(req: LoginRequest) -> dict:
    user = db.authenticate_user(req.username, req.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username/email or password")
    token = db.create_session(user["id"])
    return {"token": token, "user": user}


@app.post("/auth/logout")
def logout(authorization: Optional[str] = Header(None)) -> dict:
    if authorization and authorization.startswith("Bearer "):
        db.delete_session(authorization.removeprefix("Bearer ").strip())
    return {"status": "ok"}


@app.get("/auth/me")
def me(user: dict = Depends(get_current_user)) -> dict:
    return {"user": user}


# ─── Conversations ───

@app.get("/conversations")
def get_conversations(user: dict = Depends(get_current_user)) -> dict:
    return {"conversations": db.list_conversations(user["id"])}


@app.post("/conversations")
def post_conversation(req: ConversationCreate, user: dict = Depends(get_current_user)) -> dict:
    return db.create_conversation(user["id"], req.title or "New Chat")


@app.patch("/conversations/{conv_id}")
def patch_conversation(conv_id: int, req: ConversationRename, user: dict = Depends(get_current_user)) -> dict:
    if not db.rename_conversation(conv_id, user["id"], req.title):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "ok"}


@app.delete("/conversations/{conv_id}")
def remove_conversation(conv_id: int, user: dict = Depends(get_current_user)) -> dict:
    if not db.delete_conversation(conv_id, user["id"]):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "ok"}


@app.get("/conversations/{conv_id}/messages")
def get_messages(conv_id: int, user: dict = Depends(get_current_user)) -> dict:
    if db.get_conversation(conv_id, user["id"]) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"messages": db.list_messages(conv_id)}


# ─── Ratings ───

@app.post("/messages/{message_id}/rating")
def rate_message(message_id: int, req: RatingRequest, user: dict = Depends(get_current_user)) -> dict:
    if not 1 <= req.rating <= 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5 stars")
    if not db.set_message_rating(message_id, user["id"], req.rating):
        raise HTTPException(status_code=404, detail="Message not found")
    return {"status": "ok", "rating": req.rating}


@app.get("/ratings/stats")
def rating_stats(user: dict = Depends(get_current_user)) -> dict:
    """Aggregate response-quality stats for accuracy review and testing."""
    return db.get_rating_stats()


@app.post("/query")
def query(req: QueryRequest) -> dict:
    pipeline: RAGPipeline = app.state.pipeline
    terms = expand_query(req.question).expanded_terms or None
    prompt, _ = pipeline.build_prompt(req.question, max_chars=req.max_chars, expansion_terms=terms)
    # also return chunk-level details from the retriever for inspection
    candidates = pipeline.retriever.search(req.question, expansion_terms=terms)
    chunks = []
    for rank, cand in enumerate(candidates, start=1):
        c = cand.chunk
        chunks.append(
            {
                "rank": rank,
                "chunk_id": c.chunk_id,
                "source_path": c.source_path,
                "section_path": list(c.section_path),
                "text": c.text,
                "dense_score": cand.dense_score,
                "lexical_score": cand.lexical_score,
                "fused_score": cand.fused_score,
                "rerank_score": cand.rerank_score,
            }
        )

    return {"prompt": prompt, "chunks": chunks}


@app.post("/chat")
def chat(req: QueryRequest, user: Optional[dict] = Depends(get_optional_user)) -> StreamingResponse:
    """
    Main chat endpoint for legal document Q&A with streaming
    Performs advanced retrieval and streams answer generation.
    When the caller is authenticated and sends a conversation_id, the user
    question and the final assistant answer are persisted to SQLite.
    """
    # Validate conversation ownership up front (before streaming starts)
    conversation_id = None
    if user and req.conversation_id is not None:
        if db.get_conversation(req.conversation_id, user["id"]) is not None:
            conversation_id = req.conversation_id

    def persist_events(events):
        """Pass SSE events through while accumulating what to save."""
        answer_parts: list[str] = []
        thinking_steps: list[str] = []
        sections: list = []
        for evt in events:
            try:
                payload = json.loads(evt[len("data: "):].strip())
                if payload.get("type") == "answer":
                    answer_parts.append(payload.get("content", ""))
                elif payload.get("type") == "thinking":
                    thinking_steps.append(payload.get("content", ""))
                elif payload.get("type") == "sections":
                    sections = payload.get("content", [])
            except (json.JSONDecodeError, ValueError):
                pass
            yield evt
        if conversation_id is not None:
            answer = "".join(answer_parts).strip()
            if answer:
                saved = db.add_message(
                    conversation_id, "assistant", answer,
                    thinking=thinking_steps, sections=sections,
                )
                # Tell the UI the DB id so the response can be rated
                yield _sse({"type": "saved", "message_id": saved["id"]})

    def generate():
        try:
            pipeline: RAGPipeline = app.state.pipeline

            # Persist the user's question first
            if conversation_id is not None:
                db.add_message(conversation_id, "user", req.question)

            # Resolve the answer language ("auto" detects Urdu vs English from the
            # question script; "en"/"ur" force it). Tell the UI so it can render RTL.
            answer_language = resolve_language(req.language, req.question)
            yield _sse({"type": "language", "content": answer_language})

            understanding = "سوال کو سمجھا جا رہا ہے…" if answer_language == "ur" else "Understanding the question…"
            yield _sse({"type": "thinking", "content": understanding})

            # Generic query understanding: rewrite the question into the statutory
            # vocabulary the legislation uses (e.g. "FIR" -> "first information report",
            # "challan" -> "police report under section 173") so retrieval is driven by
            # meaning, not the user's surface wording. It ALSO returns an English-
            # normalised restatement (retrieval_query) — critical for non-English
            # questions, because the embedding model + BM25 index + the section/document
            # heuristics are all English-only and cannot match Urdu text or Urdu number
            # words (e.g. "چوبیس" = 24). Best-effort: empty on any failure.
            expansion = expand_query(req.question)

            # When the question isn't English, retrieve over the English translation;
            # for English questions keep the original wording verbatim (the heuristics
            # rely on it). Falls back to the original if translation is unavailable.
            question_is_english = detect_language(req.question) == "en"
            retrieval_query = (
                req.question if question_is_english
                else (expansion.retrieval_query or req.question)
            )
            if retrieval_query != req.question:
                yield _sse({"type": "thinking", "content": f"Interpreting as: {retrieval_query}"})
            elif not expansion.is_empty and expansion.interpretation:
                yield _sse({"type": "thinking", "content": f"Interpreting: {expansion.interpretation}"})

            # Classify intent off the English query so the right template is chosen
            # even for Urdu questions (the classifier is English-regex based).
            query_intent = classify_query_intent(retrieval_query)

            # Surface what the retriever resolved (named act / section) so the
            # "thinking" panel shows real progress, not a generic spinner.
            target_section = pipeline.retriever._extract_section_from_query(retrieval_query)
            target_document = pipeline.retriever._extract_document_context(retrieval_query)
            if target_document or target_section:
                parts = []
                if target_document:
                    parts.append(f"law: {target_document}")
                if target_section:
                    parts.append(f"section: {target_section}")
                yield _sse({"type": "thinking", "content": "Identified " + " • ".join(parts)})

            yield _sse({"type": "thinking", "content": "Searching the indexed legal documents…"})

            # Perform hybrid retrieval over the English retrieval query (widened by the
            # statutory-vocabulary expansion).
            candidates = pipeline.retriever.search(
                retrieval_query, expansion_terms=expansion.expanded_terms or None
            )

            if not candidates:
                response_data = {"type": "answer", "content": "I couldn't find relevant information in the documents to answer your question. Please try rephrasing your question or check if the relevant documents are indexed."}
                yield f"data: {json.dumps(response_data)}\n\n"
                return
            
            # Build context from retrieved chunks
            context_parts = []
            chunks_data = []
            sources_set = set()
            
            # Use top 10 candidates (increased from 5 for multi-document coverage)
            top_candidates = candidates[:10]
            
            for rank, cand in enumerate(top_candidates, start=1):
                c = cand.chunk
                metadata = chunk_metadata(c)
                doc_title = metadata.get("title", c.source_path)
                section_num = metadata.get("section_number", "")
                
                context_parts.append(
                    f"[{rank}] Source: {doc_title}\n"
                    f"File: {c.source_path}\n"
                    f"Section: {' > '.join(c.section_path) if c.section_path else 'Document'}\n"
                    + (f"Section Number: {section_num}\n" if section_num else "")
                    + f"Text: {reflow_provision_text(c.text)}"
                )
                sources_set.add(c.source_path)
                
                chunks_data.append({
                    "rank": rank,
                    "chunk_id": c.chunk_id,
                    "source_path": c.source_path,
                    "section_path": list(c.section_path),
                    "text": c.text[:200],
                    "dense_score": float(cand.dense_score or 0),
                    "lexical_score": float(cand.lexical_score or 0),
                    "fused_score": float(cand.fused_score or 0),
                    "rerank_score": float(cand.rerank_score or 0),
                })
            
            context = "\n\n---\n\n".join(context_parts)

            doc_count = len({chunk_metadata(c.chunk).get("title", c.chunk.source_path) for c in top_candidates})
            yield _sse({
                "type": "thinking",
                "content": f"Retrieved {len(top_candidates)} relevant excerpt(s) from {doc_count} document(s)",
            })

            # Build specialized prompt WITH retrieved chunks for cross-reference detection
            legal_prompt = build_legal_prompt(
                req.question,
                context,
                retrieved_chunks=top_candidates,  # CRITICAL: pass chunks for cross-ref detection
                query_intent=query_intent,
            )
            # Append the Urdu directive (no-op for English) so the answer comes
            # back in the requested language with the same markdown structure.
            legal_prompt += language_directive(answer_language)

            # Stream the answer
            yield from stream_answer(
                legal_prompt, req.question, context, sources_set, top_candidates,
                query_intent=query_intent,
            )
            
        except Exception as e:
            error_msg = str(e)
            response_data = {"type": "error", "content": f"Error processing query: {error_msg}"}
            yield f"data: {json.dumps(response_data)}\n\n"

    return StreamingResponse(persist_events(generate()), media_type="text/event-stream")


# ─── Voice: speech-to-text (ask by audio) ───

# Default audio models. Override via .env if needed.
STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe")
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")

# Voices tuned for a natural (not "hard") accent in each language.
_VOICE_BY_LANG = {"en": "alloy", "ur": "alloy"}

# Steering instructions so gpt-4o-mini-tts speaks naturally, not robotically.
_TTS_INSTRUCTIONS = {
    "en": (
        "Speak in clear, natural, conversational English with a calm, professional "
        "tone, like a helpful legal assistant. Use a soft, friendly accent — not "
        "harsh or robotic. Pace it steadily and pronounce legal terms clearly."
    ),
    "ur": (
        "خالص، فطری اور روانی والی اردو میں بولیں، جیسے کوئی مہذب پاکستانی وکیل "
        "نرم اور دوستانہ لہجے میں بات کرتا ہے۔ لہجہ سخت یا روبوٹ جیسا نہ ہو۔ "
        "Speak Urdu with a soft, natural Pakistani accent; pronounce any embedded "
        "English law names and section numbers clearly and calmly."
    ),
}


@app.post("/transcribe")
def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("auto"),
) -> dict:
    """Speech-to-text: accept a recorded audio clip and return the transcribed
    question. Supports English and Urdu (auto-detected when language='auto')."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Transcription is not configured (no API key).")

    data = audio.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    # Preserve the uploaded filename/extension so the API can sniff the format
    # (webm/ogg/mp3/wav/m4a all supported); fall back to a webm name.
    filename = audio.filename or "recording.webm"
    kwargs = {"model": STT_MODEL, "file": (filename, data, audio.content_type or "audio/webm")}
    lang = (language or "auto").lower()
    if lang in ("en", "ur"):
        kwargs["language"] = lang  # ISO-639-1 hint improves accuracy when known
    try:
        result = client.audio.transcriptions.create(**kwargs)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Transcription failed: {e}")

    text = (getattr(result, "text", "") or "").strip()
    detected = resolve_language(language, text)
    return {"text": text, "language": detected}


# ─── Voice: text-to-speech (play the response) ───

@app.post("/tts")
def tts(req: TTSRequest) -> Response:
    """Text-to-speech: turn an answer into natural-sounding speech (English/Urdu)
    and return MP3 audio bytes."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Speech synthesis is not configured (no API key).")

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text to speak.")

    # Strip markdown so the voice reads prose, not '##' or '>' characters, and cap
    # length to keep latency/cost reasonable for very long answers.
    spoken = clean_markdown_formatting(text)
    if len(spoken) > 4000:
        spoken = spoken[:4000].rsplit(" ", 1)[0] + "…"

    lang = resolve_language(req.language, spoken)
    voice = req.voice or _VOICE_BY_LANG.get(lang, "alloy")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    try:
        speech = client.audio.speech.create(
            model=TTS_MODEL,
            voice=voice,
            input=spoken,
            instructions=_TTS_INSTRUCTIONS.get(lang, _TTS_INSTRUCTIONS["en"]),
            response_format="mp3",
        )
        audio_bytes = speech.read() if hasattr(speech, "read") else speech.content
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Speech synthesis failed: {e}")

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"X-Audio-Language": lang, "Cache-Control": "no-store"},
    )


def generate_recommendations(question: str, context: str, sections: list[dict] = None) -> str:
    """
    Generate section/article recommendations using LLM
    Falls back to structured section extraction if LLM fails
    """
    try:
        from openai import OpenAI
        import os
        
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

        if not api_key:
            print("WARNING: No API key, generating fallback recommendations")
            return generate_fallback_recommendations(question, sections)

        client = OpenAI(api_key=api_key)

        # Build recommendations prompt
        recommendations_prompt = build_legal_prompt(
            question, context, prompt_type="recommendations"
        )

        print(f"Calling LLM for recommendations...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": recommendations_prompt}
            ],
            timeout=60,
            **_completion_params(model, question, "simple_lookup", answer_tokens=800),
        )
        
        result = response.choices[0].message.content
        # Clean up any remaining markdown formatting
        result = clean_markdown_formatting(result)
        print(f"Recommendations generated successfully: {len(result)} chars")
        return result
        
    except Exception as e:
        print(f"LLM recommendations generation failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to section-based recommendations
        return generate_fallback_recommendations(question, sections)


def generate_fallback_recommendations(question: str, sections: list[dict] = None) -> str:
    """
    Generate recommendations based on extracted sections when LLM is unavailable
    """
    if not sections:
        sections = []
    
    if not sections:
        return "Based on your question about the criminal case, the relevant sections and articles from the legal documents have been identified and are shown in the 'Applicable Sections' tab. Please review them for the specific rules and provisions that apply to your scenario."
    
    # Build a structured recommendation from extracted sections
    recommendation = f"Based on your question: '{question}'\n\n"
    recommendation += "The following legal provisions are applicable to this scenario:\n\n"
    
    for i, section in enumerate(sections[:8], 1):
        relevance_pct = int((section.get('relevance', 0.8) or 0.8) * 100)
        recommendation += f"{i}. {section['full_reference']} ({relevance_pct}% relevant)\n"
    
    recommendation += "\nThese sections contain rules, penalties, and procedures that apply to the arrested person's case. Review each section in the 'Applicable Sections' tab for detailed information."
    
    return recommendation


def stream_answer(
    legal_prompt: str,
    question: str,
    context: str,
    sources_set: set,
    candidates: list | None = None,
    query_intent: str = "simple_lookup",
):
    """Stream answer from LLM with source attribution and recommendations."""
    try:
        from openai import OpenAI
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

        extracted_sections = (
            build_section_references_from_chunks(candidates) if candidates else []
        )

        if api_key:
            params = _completion_params(model, question, query_intent)
            effort = params.get("reasoning_effort")
            print(f"Using OpenAI model: {model}" + (f" (reasoning effort: {effort})" if effort else ""))
            if effort and effort != "minimal":
                yield _sse({"type": "thinking", "content": f"Thinking through the provisions ({effort} reasoning)…"})
            else:
                yield _sse({"type": "thinking", "content": "Composing the answer from the retrieved provisions…"})
            client = OpenAI(api_key=api_key)

            # Use streaming for real-time response
            with client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": legal_prompt}
                ],
                stream=True,
                **params,
            ) as response:
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        response_data = {"type": "answer", "content": content}
                        yield f"data: {json.dumps(response_data)}\n\n"
            
            # Send extracted section references
            if extracted_sections:
                response_data = {"type": "sections", "content": extracted_sections}
                yield f"data: {json.dumps(response_data)}\n\n"
                print(f"Sent {len(extracted_sections)} extracted sections")
            
            # Send completion signal. Sources are printed to server terminal only.
            print(f"LLM response generated successfully")
            print("Sources:", list(sources_set))
            done_data = {"type": "done"}
            yield f"data: {json.dumps(done_data)}\n\n"
        else:
            print("WARNING: OPENAI_API_KEY not found in environment")
            answer = generate_fallback_answer(question, context)
            response_data = {"type": "answer", "content": answer}
            yield f"data: {json.dumps(response_data)}\n\n"
            
            # Still send recommendations from sections
            fallback_rec = generate_fallback_recommendations(question, extracted_sections)
            if fallback_rec:
                response_data = {"type": "recommendations", "content": fallback_rec}
                yield f"data: {json.dumps(response_data)}\n\n"
            
            if extracted_sections:
                response_data = {"type": "sections", "content": extracted_sections}
                yield f"data: {json.dumps(response_data)}\n\n"
            
            # Log sources to server terminal instead of sending to UI
            print("Sources:", list(sources_set))
            done_data = {"type": "done"}
            yield f"data: {json.dumps(done_data)}\n\n"
            
    except Exception as e:
        print(f"LLM generation failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        extracted_sections = (
            build_section_references_from_chunks(candidates) if candidates else []
        )
        
        answer = generate_fallback_answer(question, context)
        response_data = {"type": "answer", "content": answer}
        yield f"data: {json.dumps(response_data)}\n\n"
        
        fallback_rec = generate_fallback_recommendations(question, extracted_sections)
        if fallback_rec:
            response_data = {"type": "recommendations", "content": fallback_rec}
            yield f"data: {json.dumps(response_data)}\n\n"
        
        if extracted_sections:
            response_data = {"type": "sections", "content": extracted_sections}
            yield f"data: {json.dumps(response_data)}\n\n"
        
        # On error, log sources to server terminal instead of sending to UI
        print("Sources:", list(sources_set))
        done_data = {"type": "done"}
        yield f"data: {json.dumps(done_data)}\n\n"


@app.get("/debug/sections")
def debug_sections(doc: str, section: Optional[str] = None, contains: Optional[str] = None) -> dict:
    """TEMP: list stored section numbers for chunks of a document (substring match on path)."""
    pipeline: RAGPipeline = app.state.pipeline
    out = []
    for chunk in pipeline.retriever._chunks:
        if doc.lower() not in chunk.source_path.lower():
            continue
        meta = chunk_metadata(chunk)
        sn = meta.get("section_number")
        if section is not None and str(sn) != section:
            continue
        if contains is not None and contains.lower() not in chunk.text.lower():
            continue
        out.append({
            "chunk_id": chunk.chunk_id,
            "section_number": sn,
            "section_path": list(chunk.section_path)[:2],
            "text": chunk.text[:90],
        })
    return {"count": len(out), "chunks": out[:40]}


@app.get("/query")
def query_get(q: str, max_chars: Optional[int] = 15000) -> dict:
    """Quick GET wrapper so you can test from a browser: /query?q=your+question"""
    pipeline: RAGPipeline = app.state.pipeline
    terms = expand_query(q).expanded_terms or None
    prompt, _ = pipeline.build_prompt(q, max_chars=max_chars, expansion_terms=terms)
    candidates = pipeline.retriever.search(q, expansion_terms=terms)
    chunks = []
    for rank, cand in enumerate(candidates, start=1):
        c = cand.chunk
        chunks.append(
            {
                "rank": rank,
                "chunk_id": c.chunk_id,
                "source_path": c.source_path,
                "section_path": list(c.section_path),
                "text": c.text,
                "dense_score": cand.dense_score,
                "lexical_score": cand.lexical_score,
                "fused_score": cand.fused_score,
                "rerank_score": cand.rerank_score,
            }
        )

    return {"prompt": prompt, "chunks": chunks}


def generate_fallback_answer(question: str, context: str) -> str:
    """
    Fallback answer generation when LLM is not available
    Extracts and summarizes key information from context
    """
    lines = context.split('\n')
    # Simple heuristic: return first few relevant lines
    relevant_lines = [line for line in lines if line.strip() and len(line) > 20][:3]
    
    if relevant_lines:
        answer = f"Based on the legal documents:\n\n" + "\n\n".join(relevant_lines)
        return answer
    
    return "I found relevant documents but couldn't extract text. Please check document formatting."
