import os
import time
import uuid
import json
from typing import Dict, Optional

# Disable telemetry that triggers PostHog calls before any LlamaIndex imports.
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("LLAMA_INDEX_DISABLE_TELEMETRY", "1")
os.environ.setdefault("LLAMA_INDEX_TELEMETRY_ENABLED", "false")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from models import model_manager
from llama_law import answer_query_with_trace_withoutUploadFile
from logger import logger
from llama_index.core import Document, VectorStoreIndex,Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_law import answer_case_question_withuploadFile, compress_case_facts
from sqlalchemy.orm import Session
from db import get_db, init_db
from db_models import User, Case, QuestionAnswer
from config import Config
from auth import get_current_user_id
from clarify import (
    apply_clarification_answers,
    detect_topic,
    get_clarification,
    get_clarification_for_topic,
    get_missing_fields_for_topic,
)
from summary_pipeline import (
    generate_summary_dict,
    SUMMARY_LIST_LIMITS_PRIMARY,
    SUMMARY_LIST_LIMITS_FALLBACK,
    summary_json_to_sections,
    summary_json_to_text,
)


app = FastAPI(title="Law AI Assistant")

MAX_UPLOAD_MB = 5
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

session_case_ids: Dict[str, int] = {}   # {user/session id: case DB id}
case_summary_sections: Dict[int, Dict[str, str]] = {}  # {case id: {topic_name: topic_text}}
pending_clarifications: Dict[str, Dict[str, object]] = {}  # {user/session id: pending data}
session_history: Dict[str, list[dict]] = {} 
# Example Entry
# session_history["user_123"] = [
#    {"role": "user", "content": "How is the house split?", "topic": "property"},
#    {"role": "assistant", "content": "Based on the Act...", "topic": "property"}
# ]

def _get_cors_origins() -> list[str]:
    if Config.ENV == "prd":
        return Config.CORS_ORIGINS_LIST
    return ["http://localhost:5173"]


# Allow requests from React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class Question(BaseModel):
    question: str
    session_id: Optional[str] = None
    topic: Optional[str] = None


class Clarification(BaseModel):
    answers: list[str]
    session_id: Optional[str] = None


class ResetSession(BaseModel):
    session_id: Optional[str] = None


def split_into_chunks(text: str):
    # Create Document object
    document = Document(text=text)

    # Create SentenceSplitter
    splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

    # Use the public method to get nodes
    nodes = splitter.get_nodes_from_documents([document])

    return nodes


def _format_history(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return ""
    lines: list[str] = []
    for q, a in pairs:
        lines.append(f"User: {q}\nAssistant: {a}")
    return "\n\n".join(lines)


def _clear_session(resolved_user_id: str) -> None:
    session_case_ids.pop(resolved_user_id, None) 
    pending_clarifications.pop(resolved_user_id, None)
    session_history.pop(resolved_user_id, None)

def _build_case_summary_for_query(case_id: int, topic: Optional[str], full_summary: Optional[str]) -> str:
    if not topic:
        return full_summary or ""
    if not full_summary:
        return ""

    cached_section = case_summary_sections.get(case_id, {}).get(topic)
    if cached_section:
        return cached_section

    try:
        summary_obj = json.loads(full_summary)
    except Exception:
        return full_summary

    if not isinstance(summary_obj, dict):
        return full_summary

    section_rows = summary_json_to_sections(summary_obj)
    for row in section_rows:
        if isinstance(row, dict) and row.get("section") == topic and row.get("text"):
            return row.get("text")
    return full_summary

def _refresh_case_summary_cache(case_id: int, summary_sections: List[dict]) -> None:
    """
    Directly maps the pre-processed list to the in-memory cache.
    """
    # Create a simple { "topic_name": "topic_text" } map
    sections_map = {
        item["section"]: item["text"] 
        for item in summary_sections 
    }
    
    # Store it in your global/manager dictionary
    case_summary_sections[case_id] = sections_map

def _ensure_uploaded_case_embeddings(
    case_id: int,
    filename: str,
    summary_payload: object,
    *,
    summary_sections: Optional[list[dict]] = None,
    summary_text: Optional[str] = None,
) -> None:
    if not summary_payload:
        return
    if model_manager.has_uploaded_case(case_id):
        return

    summary_obj = None
    if isinstance(summary_payload, str):
        try:
            summary_obj = json.loads(summary_payload)
        except Exception:
            summary_obj = {"raw_summary": summary_payload}
    elif isinstance(summary_payload, dict):
        summary_obj = summary_payload

    if not isinstance(summary_obj, dict):
        return

    sections = summary_sections or summary_json_to_sections(summary_obj)
    documents = []
    if sections:
        for section in sections:
            text = section.get("text") if isinstance(section, dict) else None
            if not text:
                continue
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source_type": "uploaded_case",
                        "case_id": str(case_id),
                        "source": filename,
                        "summary_section": section.get("section", "unknown"),
                    },
                )
            )
    else:
        fallback_text = summary_text or summary_json_to_text(summary_obj)
        if fallback_text:
            documents.append(
                Document(
                    text=fallback_text,
                    metadata={
                        "source_type": "uploaded_case",
                        "case_id": str(case_id),
                        "source": filename,
                        "summary_section": "full",
                    },
                )
            )

    model_manager.add_uploaded_case_documents(case_id, documents)

async def upload_case(file: UploadFile, session_user_id: str):
    # Read file
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_UPLOAD_MB}MB. Please reupload a smaller file.",
        )
    text = content.decode("utf-8", errors="ignore")
    return {"message": f"Uploaded {file.filename} for session {session_user_id}", "text": text}

@app.on_event("startup")
def startup_event():
    logger.info("Starting up: initializing models and vector index...")
    model_manager.init_models()
    model_manager.create_or_load_index()
    init_db()
    logger.info("Startup complete!")

@app.post("/upload_case")
async def upload_case_endpoint(
    file: UploadFile,
    session_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_current_user_id),
):
    Settings.llm = OpenAI(model=Config.OPENAI_MODEL, temperature=0.1)
    llm = Settings.llm
    resolved_user_id = user_id or session_id or Config.DEV_DEFAULT_USER_ID
    
    # 1. Get raw text from upload
    result = await upload_case(file, resolved_user_id)
    case_text = result.pop("text")

    # 2. Identify/Create User (Standard logic)
    user = db.query(User).filter(User.external_id == resolved_user_id).first()
    if not user:
        user = User(external_id=resolved_user_id)
        db.add(user)
        db.commit()
        db.refresh(user)

    # 3. Check for existing case to avoid re-summarizing (Save Credits!)
    existing_case = db.query(Case).filter(
        Case.user_id == user.id,
        Case.filename == file.filename,
        Case.text == case_text
    ).first()

    if existing_case:
        session_case_ids[resolved_user_id] = existing_case.id
        # Refresh uploaded-case embeddings and summary cache from stored summary JSON
        try:
            summary_obj = json.loads(existing_case.case_summary or "")
        except Exception:
            summary_obj = None
        if isinstance(summary_obj, dict):
            summary_text = summary_json_to_text(summary_obj, include_outcome_reasons=False)
            summary_sections = summary_json_to_sections(summary_obj, include_outcome_reasons=False)
            _ensure_uploaded_case_embeddings(
                existing_case.id,
                existing_case.filename,
                summary_obj,
                summary_sections=summary_sections,
                summary_text=summary_text,
            )
            _refresh_case_summary_cache(existing_case.id, summary_sections)
        else:
            _ensure_uploaded_case_embeddings(existing_case.id, existing_case.filename, existing_case.case_summary)
        result["case_id"] = existing_case.id
        return result

    # 4. Process NEW case using the new pipeline
    summary = generate_summary_dict(
        case_text,
        target_words=1200,
        max_words=1800,
        list_limits_primary=SUMMARY_LIST_LIMITS_PRIMARY,
        list_limits_fallback=SUMMARY_LIST_LIMITS_FALLBACK,
        llm=llm,
    )
    summary_text = summary_json_to_text(summary, include_outcome_reasons=False)
    summary_sections = summary_json_to_sections(summary, include_outcome_reasons=False)
    summary_json_str = json.dumps(summary)

    # 5. Save to SQL
    case_row = Case(user_id=user.id, filename=file.filename, text=case_text, case_summary=summary_json_str)
    db.add(case_row)
    db.commit()
    db.refresh(case_row)

    # 6. Embed into ChromaDB (The "Search Index")
    _ensure_uploaded_case_embeddings(
        case_row.id,
        file.filename,
        summary,
        summary_sections=summary_sections,
        summary_text=summary_text,
    )

    # 7. Finalize
    _refresh_case_summary_cache(case_row.id, summary_sections)
    session_case_ids[resolved_user_id] = case_row.id
    result["case_id"] = case_row.id
    return result

@app.post("/reset")
def reset_session(
    payload: ResetSession,
    user_id: Optional[str] = Depends(get_current_user_id),
):
    resolved_user_id = user_id or payload.session_id or Config.DEV_DEFAULT_USER_ID
    if not resolved_user_id:
        raise HTTPException(status_code=400, detail="session_id is required to reset.")
    _clear_session(resolved_user_id)
    return {"ok": True}

@app.post("/ask")
async def ask_ai(
    q: Question,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_current_user_id),
):
    resolved_user_id = user_id or q.session_id or Config.DEV_DEFAULT_USER_ID
    
    # --- 1. GET HISTORY ---
    chat_history = session_history.get(resolved_user_id, [])
    history_text = "\n".join([f"{'Client' if t['role']=='user' else 'Lawyer'}: {t['content']}" for t in chat_history[-8:]])

    # --- 2. PREPARE CONTEXT & TOPIC ---
    case_id = session_case_ids.get(resolved_user_id)
    if not case_id:
        return {"error": "No active case found. Please upload a case summary first."}
    case_sections = case_summary_sections.get(case_id, {})
    detected_topic = q.topic if (q.topic and q.topic != "other") else await detect_topic(q.question)
    
    # --- 3. CLARIFICATION GATE (NEW) ---
    # We pass the user's specific question to ensure relevance
    final_topic, missing_info_questions = await get_clarification_for_topic(
        question=q.question, 
        topic=detected_topic, 
        case_sections=case_sections
    )
    if missing_info_questions:
        # Store the pending state so /clarify knows what to update
        pending_clarifications[resolved_user_id] = {
            "original_question": q.question,
            "topic": detected_topic,
            "clarifying_questions": missing_info_questions
        }
        return {
            "needs_clarification": True,
            "questions": missing_info_questions,
            "topic": detected_topic
        }

    # --- 4. GENERATE ANSWER (Proceed if no missing info) ---
    case_section_text = _build_case_summary_for_query(case_id, final_topic, case_sections)
    
    answer, citations = await answer_case_question_withuploadFile(
        question=q.question,
        case_section_text=case_section_text,
        history_text=history_text,
        topic=final_topic,
    )
    # --- 5. UPDATE CACHE & PERSIST ---
    # Update the In-Memory Cache (Fast)
    new_history = [
        {"role": "user", "content": q.question, "topic": detected_topic},
        {"role": "assistant", "content": answer, "topic": detected_topic}
    ]
    session_history.setdefault(resolved_user_id, []).extend(new_history)
    
    # Trim history to keep memory usage low (e.g., last 20 messages)
    if len(session_history[resolved_user_id]) > 20:
        session_history[resolved_user_id] = session_history[resolved_user_id][-20:]

    # Save to SQL (Persistent backup)
    qa = QuestionAnswer(
        case_id=case_id, 
        question=q.question, 
        answer=answer, 
        topic=detected_topic,
        user_id=resolved_user_id
    )
    db.add(qa)
    db.commit()

    return {
        "answer": answer,
        "topic": detected_topic,
        "citations": citations
    }

@app.post("/clarify")
async def clarify_answer(
    c: Clarification,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_current_user_id),
):
    resolved_user_id = user_id or c.session_id or Config.DEV_DEFAULT_USER_ID
    pending = pending_clarifications.get(resolved_user_id)
    
    if not pending:
        raise HTTPException(status_code=400, detail="No pending clarification.")

    case_id = session_case_ids.get(resolved_user_id)
    topic = pending.get("topic")
    original_question = pending.get("original_question")
    # 1. Update SQL & JSON
    case_row = db.query(Case).filter(Case.id == case_id).first()
    # Assuming apply_clarification_answers returns a fresh JSON string
    updated_json_str = await apply_clarification_answers(
        context_summary=case_row.case_summary,
        topic=topic,
        answers=c.answers,
        questions=pending.get("clarifying_questions")
    )
    case_row.case_summary = updated_json_str
    db.commit()

    # 2. Parse for Cache and Embeddings
    summary_obj = json.loads(updated_json_str)
    # include_outcome_reasons=False to keep the search index focused on facts
    summary_sections = summary_json_to_sections(summary_obj, include_outcome_reasons=False)
    
    # 3. REFRESH EVERYTHING
    # Update the in-memory topic map
    _refresh_case_summary_cache(case_id, summary_sections)
    
    # Update the Vector Store (ChromaDB)
    _ensure_uploaded_case_embeddings(
        case_id=case_id,
        filename=case_row.filename,
        summary_payload=summary_obj,
        summary_sections=summary_sections,
        force_refresh=True # THIS IS THE KEY
    )

    return await ask_ai(
        q=Question(question=original_question, topic=topic, session_id=resolved_user_id),
        db=db,
        user_id=resolved_user_id
    )


@app.get("/history/{session_id}")
def get_history(
    session_id: str,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_current_user_id),
):
    resolved_user_id = user_id or session_id or Config.DEV_DEFAULT_USER_ID
    user = db.query(User).filter(User.external_id == resolved_user_id).first()
    if user is None:
        return {"cases": []}

    cases_payload = []
    cases = db.query(Case).filter(Case.user_id == user.id).order_by(Case.created_at.desc()).all()
    for c in cases:
        qa_items = (
            db.query(QuestionAnswer)
            .filter(QuestionAnswer.case_id == c.id)
            .order_by(QuestionAnswer.created_at.asc())
            .all()
        )
        cases_payload.append(
            {
                "case_id": c.id,
                "filename": c.filename,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "qa": [
                    {
                        "question": qa.question,
                        "answer": qa.answer,
                        "created_at": qa.created_at.isoformat() if qa.created_at else None,
                    }
                    for qa in qa_items
                ],
            }
        )

    return {"cases": cases_payload}
