"""
API Routes for Law AI Backend
All endpoint handlers organized in one place.
"""
import json
from typing import Dict, List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI

from app.services.summary_service import  summary_json_to_sections
from app.core.config import Config
from app.core.dev_logger import format_and_log
from app.core.logger import logger
from app.core.models import model_manager
from app.core.auth import get_current_user_id
from app.db import get_db
from app.db.models import User, Case, QuestionAnswer
from app.services.rag_service import (
    answer_query_with_trace_withoutUploadFile,
    answer_case_question_withuploadFile,
    compress_case_facts
)
from app.services.clarify_service import (
    apply_clarification_answers,
    detect_topic,
    get_clarification_for_topic,
    summarize_answer_if_needed,
)
from app.services.summary_service import (
    generate_summary_dict,
    SUMMARY_LIST_LIMITS_PRIMARY,
    SUMMARY_LIST_LIMITS_FALLBACK,
)

router = APIRouter()

# Constants
MAX_UPLOAD_MB = 5
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# In-memory caches (case_summary_sections, pending_clarifications, session_history keyed by user_id -> case_id)
case_summary_sections: Dict[int, Dict[int, Dict[str, str]]] = {}
pending_clarifications: Dict[int, Dict[int, Dict[str, object]]] = {}
session_history: Dict[int, Dict[int, list[dict]]] = {}


# Request/Response Models
class Question(BaseModel):
    question: str
    case_id: Optional[int] = None
    session_id: Optional[str] = None  # For backwards compatibility and user identification
    topic: Optional[str] = None


class Clarification(BaseModel):
    answers: Dict[str, str]
    missing_fields: Optional[List[str]] = None
    filename: Optional[str] = None
    case_id: Optional[int] = None
    session_id: Optional[str] = None  # For backwards compatibility


class ResetCase(BaseModel):
    case_id: int


# Helper Functions
def _clear_case(case_id: int) -> None:
    """Clear all cached data for a case."""
    for user_id, cases_map in list(case_summary_sections.items()):
        if case_id in cases_map:
            cases_map.pop(case_id, None)
        if not cases_map:
            case_summary_sections.pop(user_id, None)
    for user_id, cases_map in list(pending_clarifications.items()):
        if case_id in cases_map:
            cases_map.pop(case_id, None)
        if not cases_map:
            pending_clarifications.pop(user_id, None)
    for user_id, cases_map in list(session_history.items()):
        if case_id in cases_map:
            cases_map.pop(case_id, None)
        if not cases_map:
            session_history.pop(user_id, None)


def _build_case_summary_for_query(
    case_id: int,
    db: Session,
) -> Optional[dict]:
    """
    Fetch the full case summary from the database, parse it into a dictionary,
    and refresh the cache.
    """
    # Fetch the full summary from the database
    case_row = db.query(Case).filter(Case.id == case_id).first()
    if not case_row or not case_row.case_summary:
        logger.warning(f"No case summary found in DB for case_id: {case_id}")
        return None

    full_summary_str = case_row.case_summary

    # Parse the full summary
    try:
        summary_obj = json.loads(full_summary_str)
        if not isinstance(summary_obj, dict):
            logger.error(f"Parsed summary for case {case_id} is not a dict.")
            return None # Or handle as just text
    except (json.JSONDecodeError, TypeError):
        logger.error(f"Failed to parse summary JSON for case {case_id}.")
        return None # Or return the raw string in a dict

    # Refresh the cache with all sections from the fetched summary
    summary_sections = summary_json_to_sections(summary_obj, include_outcome_reasons=False)
    """
    summary_sections = [dict(section="facts", text="..."), dict(section="issues", text="..."), ...]
    """
    _refresh_case_summary_cache(case_id, summary_sections, case_row.user_id)
    
    return summary_sections


def _refresh_case_summary_cache(case_id: int, summary_sections: List[dict], user_id: int) -> None:
    """Update the in-memory cache with parsed summary sections."""
    sections_map = {
        item["section"]: item["text"]
        for item in summary_sections
        if isinstance(item, dict)
    }
    """
      case_summary_sections = {
        123: {
            "facts": "- Fact: The parties were married for 12 years and separated in 2022.\n- Fact: There are two children, aged 8 and 10, who live primarily with the mother.\n- Fact: The main asset is the former matrimonial home, valued at $1.2 million.",
            "issues": "- Issue: What is the appropriate division of the matrimonial home?\n- Issue: Should the father pay ongoing spousal maintenance to the mother?",
            "property_division": "- Asset Pool: Matrimonial home: $1,200,000\n- Asset Pool: Father's superannuation: $450,000\n- Asset Pool: Mother's superannuation: $150,000\n- Contributions: The father was the primary income earner during the relationship.\n- Contributions: The mother was the primary caregiver for the children.",
            "impact_analysis": "- Pivotal Finding: The court will likely consider the long duration of the marriage and the mother's role as primary caregiver as significant non-financial contributions.\n- Statutory Pivot: Section 79(4) of the Family Law Act regarding contributions and Section 75(2) regarding future needs will be central to the court's decision."
        }
        }
    """
    case_summary_sections.setdefault(user_id, {})[case_id] = sections_map


def _ensure_uploaded_case_embeddings(
    case_id: int,
    filename: str,
    summary_payload: object,
    *,
    summary_sections: Optional[list[dict]] = None,
    summary_text: Optional[str] = None,
) -> None:
    """Create vector embeddings for an uploaded case summary."""
    if not summary_payload:
        return
    if model_manager.has_uploaded_case(case_id):
        return

    # Parse summary
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

    # Create documents for embedding
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
        # Fallback to full text
        fallback_text = "\n".join(summary_text) if summary_text else None
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


def _update_uploaded_case_section_embeddings(
    case_id: int,
    filename: str,
    section_name: str,
    section_text: Optional[str],
) -> None:
    if not section_text:
        return
    document = Document(
        text=section_text,
        metadata={
            "source_type": "uploaded_case",
            "case_id": str(case_id),
            "source": filename,
            "summary_section": section_name,
        },
    )
    model_manager.add_uploaded_case_documents(case_id, [document], allow_existing=True)


async def upload_case(file: UploadFile, session_user_id: str):
    """Helper to read and validate uploaded file."""
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_UPLOAD_MB}MB. Please reupload a smaller file.",
        )
    text = content.decode("utf-8", errors="ignore")
    return {"message": f"Uploaded {file.filename} for session {session_user_id}", "text": text}


# API Endpoints
@router.post("/upload_case")
async def upload_case_endpoint(
    file: UploadFile,
    session_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_current_user_id),
):
    """Upload and process a new case document."""
    if Config.ENV == "dev":
        format_and_log("/upload_case", "Endpoint Called", "Initial Request", {"filename": file.filename, "session_id": session_id})

    resolved_user_id = user_id or session_id or Config.DEV_DEFAULT_USER_ID

    # 1. Read file
    result = await upload_case(file, resolved_user_id)
    case_text = result.pop("text")

    # 2. Get or create user
    user = db.query(User).filter(User.external_id == resolved_user_id).first()
    if not user:
        user = User(external_id=resolved_user_id)
        db.add(user)
        db.commit()
        db.refresh(user)
        if Config.ENV == "dev":
            format_and_log(f"/upload_case{file.filename}", "DB Update", "User Table", {"action": "create", "user_id": user.id, "external_id": user.external_id})

    # 3. Check for existing case (save API credits!)
    existing_case = db.query(Case).filter(
        Case.user_id == user.id,
        Case.filename == file.filename,
        Case.text == case_text
    ).first()
    summary_text = []
    if existing_case:
        if Config.ENV == "dev":
            format_and_log("/upload_case", "Case Found", "existing_case", {"case_id": existing_case.id})

        # Restore embeddings and cache from stored summary
        try:
            summary_obj = json.loads(existing_case.case_summary or "")
        except Exception:
            summary_obj = None

        if Config.ENV == "dev":
            format_and_log(
                "/upload_case",
                "Summary",
                "case_summary",
                {
                    "case_id": existing_case.id,
                    "source": "db_existing_case",
                    "raw_summary_json": (existing_case.case_summary or "")[:8000],
                    "parsed_summary": summary_obj,
                },
            )

        if isinstance(summary_obj, dict):
            summary_sections = summary_json_to_sections(summary_obj, include_outcome_reasons=False)
            for section in summary_sections:
                summary_text.append(f"--- {section.get('section', 'Unknown').replace('_', ' ').title()} ---")
                summary_text.append(section.get('text', ''))
                summary_text.append('')
            _ensure_uploaded_case_embeddings(
                existing_case.id,
                existing_case.filename,
                summary_obj,
                summary_sections=summary_sections,
                summary_text=summary_text,
            )
            _refresh_case_summary_cache(existing_case.id, summary_sections, existing_case.user_id)
            if Config.ENV == "dev":
                format_and_log("/upload_case", "Cache Update", "case_summary_sections", case_summary_sections)
        else:
            _ensure_uploaded_case_embeddings(
                existing_case.id,
                existing_case.filename,
                existing_case.case_summary
            )

        result["case_id"] = existing_case.id
        return result

    # 4. Generate summary for new case
    summary_json_str = compress_case_facts(case_text)
    """
    {
  "facts": [
    "The parties were married for 12 years and separated in 2022.",
    "There are two children, aged 8 and 10, who live primarily with the mother.",
    "The main asset is the former matrimonial home, valued at $1.2 million."
  ],
  "issues": [
    "What is the appropriate division of the matrimonial home?",
    "Should the father pay ongoing spousal maintenance to the mother?"
  ],
  "property_division": {
    "asset_pool": [
      "Matrimonial home: $1,200,000",
      "Father's superannuation: $450,000",
      "Mother's superannuation: $150,000"
    ],
    "contributions": [
      "The father was the primary income earner during the relationship.",
      "The mother was the primary caregiver for the children."
    ]
  },
  "impact_analysis": {
    "pivotal_findings": [
      "The court will likely consider the long duration of the marriage and the mother's role as primary caregiver as significant non-financial contributions."
    ],
    "statutory_pivots": [
      "Section 79(4) of the Family Law Act regarding contributions and Section 75(2) regarding future needs will be central to the court's decision."
    ]
  }
}
    """
    try:
        summary = json.loads(summary_json_str)
    except (json.JSONDecodeError, TypeError):
        summary = {} # or handle error appropriately

    if Config.ENV == "dev":
        format_and_log(
            "/upload_case",
            "Summary",
            "case_summary",
            {
                "case_id": None,
                "source": "generated_pre_db",
                "raw_summary_json": (summary_json_str or "")[:8000],
                "parsed_summary": summary,
            },
        )

    
    summary_sections = summary_json_to_sections(summary, include_outcome_reasons=False)
    for section in summary_sections:
        summary_text.append(f"--- {section.get('section', 'Unknown').replace('_', ' ').title()} ---")
        summary_text.append(section.get('text', ''))
        summary_text.append('')

    # 5. Save to database
    case_row = Case(
        user_id=user.id,
        filename=file.filename,
        text=case_text,
        case_summary=summary_json_str
    )
    db.add(case_row)
    db.commit()
    db.refresh(case_row)
    if Config.ENV == "dev":
        format_and_log("/upload_case", "DB Update", "Case Table", {"action": "create", "case_id": case_row.id, "filename": case_row.filename})
        format_and_log(
            "/upload_case",
            "Summary",
            "case_summary",
            {
                "case_id": case_row.id,
                "source": "generated_saved_to_db",
                "raw_summary_json": (case_row.case_summary or "")[:8000],
                "parsed_summary": summary,
            },
        )

    # 6. Create embeddings
    _ensure_uploaded_case_embeddings(
        case_row.id,
        file.filename,
        summary,
        summary_sections=summary_sections,
        summary_text=summary_text,
    )

    # 7. Update cache
    _refresh_case_summary_cache(case_row.id, summary_sections, case_row.user_id)
    if Config.ENV == "dev":
        format_and_log("/upload_case", "Cache Update", "case_summary_sections", case_summary_sections)
    result["case_id"] = case_row.id

    return result


@router.post("/reset")
def reset_case(
    payload: ResetCase,
    user_id: Optional[str] = Depends(get_current_user_id),
):
    """Clear cached data for a specific case."""
    if Config.ENV == "dev":
        format_and_log("/reset", "Endpoint Called", "Initial Request", {"case_id": payload.case_id})
    _clear_case(payload.case_id)
    if Config.ENV == "dev":
        format_and_log("/reset", "Cache Update", "pending_clarifications", pending_clarifications)
        format_and_log("/reset", "Cache Update", "session_history", session_history)
    return {"ok": True}


@router.post("/ask")
#to be updated
async def ask_ai(
    q: Question,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_current_user_id),
):
    """Ask a question about the uploaded case."""
    if Config.ENV == "dev":
        format_and_log("/ask", "Endpoint Called", "Initial Request", q.dict())
    
    # Get case_id from request or error
    case_id = q.case_id
    if not case_id:
        raise HTTPException(status_code=400, detail="case_id is required")

    # 1. Verify case exists
    resolved_user_id = user_id or q.session_id or Config.DEV_DEFAULT_USER_ID
    user_row = db.query(User).filter(User.external_id == resolved_user_id).first()
    if not user_row:
        raise HTTPException(status_code=404, detail="User not found")
    case_row = (
        db.query(Case)
        .filter(Case.id == case_id, Case.user_id == user_row.id)
        .first()
    )
    if not case_row:
        raise HTTPException(status_code=404, detail="Case not found")

    # 2. Get conversation history (keyed by user_id -> case_id)
    chat_history = session_history.get(case_row.user_id, {}).get(case_id, [])
    if Config.ENV == "dev":
        format_and_log(
            "/ask",
            "Cache Lookup",
            "session_history",
            {
                "user_id": case_row.user_id,
                "case_id": case_id,
                "items": len(chat_history),
            },
        )
    history_text = "\n".join([
        f"{'Client' if t['role']=='user' else 'Lawyer'}: {t['content']}"
        for t in chat_history[-8:]
    ])

    # 3. Detect topic
    detected_topic = q.topic if (q.topic and q.topic != "other") else None
    if not detected_topic:
        detected_topic = await detect_topic(q.question)
    logger.info(f"Detected topic for question: {detected_topic}")
    # 5. Extract the specific text section for the RAG query
    # We can now reliably get this from the cache, which was just updated.
    case_section_text = case_summary_sections.get(case_row.user_id, {}).get(case_id, {}).get(detected_topic, "")
    impact_text = case_summary_sections.get(case_row.user_id, {}).get(case_id, {}).get("impact_analysis", "")
    logger.info(f"Using summary section for topic '{detected_topic}'")

    if not case_section_text:
        logger.warning(f"No summary section found for topic '{detected_topic}' in case {case_id}. Using full summary text as fallback.")
        summary_sections = _build_case_summary_for_query(case_id, db)
        case_section_text = "\n".join([section.get("text", "") for section in summary_sections if section.get("section") == detected_topic])
        
    # 6. Check for missing factors using the full summary object
    missing_fields, clarifying_questions = get_clarification_for_topic(
        topic=detected_topic,
        case_summary=case_section_text,
    )

    if clarifying_questions:
        logger.info(f"Clarification needed for topic '{detected_topic}'. Asking questions.")
        pending_clarifications.setdefault(case_row.user_id, {})[case_id] = {
            "question": q.question,
            "topic": detected_topic,
            "questions": clarifying_questions,
            "missing_fields": missing_fields or [],
        }
        if Config.ENV == "dev":
            format_and_log("/ask", "Cache Update", "pending_clarifications", pending_clarifications)
        return {
            "clarification_needed": True,
            "questions": clarifying_questions,
            "missing_fields": missing_fields or [],
            "message": "I need a bit more information to give you a complete answer. Please answer the following questions:",
        }

    # 7. Generate answer
    response_text, citations = await answer_case_question_withuploadFile(
        question=q.question,
        case_section_text=case_section_text,
        history_text=history_text,
        topic=detected_topic,
        impact_analysis=impact_text
    )

    # Split the response to get the main answer and the cache summary
    parts = response_text.split("---CACHE_SUMMARY---")
    main_answer = parts[0].strip()
    cache_summary = parts[1].strip() if len(parts) > 1 else "Summary not available."

    # Store the concise summary in history, not the full answer (keyed by case_id)
    session_history.setdefault(case_row.user_id, {}).setdefault(case_id, [])
    session_history[case_row.user_id][case_id].append({"role": "user", "content": q.question})
    session_history[case_row.user_id][case_id].append({"role": "assistant", "content": cache_summary})
    if Config.ENV == "dev":
        format_and_log(
            "/ask",
            "Cache Update",
            "session_history",
            {
                "user_id": case_row.user_id,
                "case_id": case_id,
                "items": len(session_history[case_row.user_id][case_id]),
                "last_user": q.question,
                "last_assistant": cache_summary,
            },
        )

    # Save the full Q&A to the database
    db_entry = QuestionAnswer(
        case_id=case_id,
        user_id=case_row.user_id,
        question=q.question,
        answer=main_answer,  # Store the full answer in the DB
        topic=detected_topic
    )
    db.add(db_entry)
    db.commit()
    if Config.ENV == "dev":
        format_and_log("/ask", "DB Update", "QuestionAnswer Table", {"action": "create", "qa_id": db_entry.id})

    return {"answer": main_answer, "citations": citations}


@router.post("/clarify")
async def clarify_answer(
    c: Clarification,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_current_user_id),
):
    """
    c(Clarification):{"answers":{"agreement_date":"Signed 12 May 2019, before the wedding (s 90B).",
    "legal_advice":"Yes, both had separate lawyers; certificates signed.",
    "financial_disclosure":"No, a Queensland property and super were not disclosed.",
    "pressure_duress":"Yes, three days before wedding with ceremony threatened.",
    "changed_circumstances":"Yes, two children were born after signing."},
    "missing_fields":["agreement_date","legal_advice","financial_disclosure","pressure_duress","changed_circumstances"],"case_id":6,"session_id":"005"}

    Process clarification answers and update case summary."""
    if Config.ENV == "dev":
        format_and_log("/clarify", "Endpoint Called", "Initial Request", c.dict())
    
    # Get case_id from request
    case_id = c.case_id
    if not case_id:
        raise HTTPException(status_code=400, detail="case_id is required")

    resolved_user_id = user_id or c.session_id or Config.DEV_DEFAULT_USER_ID
    user_row = db.query(User).filter(User.external_id == resolved_user_id).first()
    if not user_row:
        raise HTTPException(status_code=404, detail="User not found")

    # Get case from database (prefer user_id + filename when provided)
    if c.filename:
        case_row = (
            db.query(Case)
            .filter(Case.user_id == user_row.id, Case.filename == c.filename)
            .first()
        )
    else:
        case_row = (
            db.query(Case)
            .filter(Case.id == case_id, Case.user_id == user_row.id)
            .first()
        )
    if not case_row:
        raise HTTPException(status_code=404, detail="Case not found")

    pending = pending_clarifications.get(case_row.user_id, {}).get(case_id)
    if Config.ENV == "dev":
        format_and_log(
            "/clarify",
            "Cache Lookup",
            "pending_clarifications",
            {"user_id": case_row.user_id, "case_id": case_id, "found": bool(pending)},
        )
    """
    pending_clarifications = {
        "5": {
            "6": {
            "question": "the 2 parties signed a Binding Financial Agreement (BFA) three days before our wedding. Husband told wife that if she didn't sign it, the ceremony wouldn't go ahead. Wife also recently discovered they didn't disclose a property they own in Queensland. Can wife get this agreement set aside?",
            "topic": "prenup_postnup",
            "questions": [
                "When was the agreement signed? Was it before (Section 90B) or after (Section 90C) the marriage?",
                "Did both parties receive independent legal advice from separate lawyers before signing?",
                "Was there full and frank financial disclosure of all assets and liabilities before signing?",
                "Was there any pressure, urgency, or 'unfair' circumstances surrounding the signing of the document?",
                "Have there been major changes since signing, such as the birth of a child, that the agreement didn't account for?"
            ],
            "missing_fields": [
                "agreement_date",
                "legal_advice",
                "financial_disclosure",
                "pressure_duress",
                "changed_circumstances"
            ]
            }
        }
        }
    """
    if not pending:
        if Config.ENV == "dev":
            format_and_log(
                "/clarify",
                "Cache Miss",
                "pending_clarifications",
                {"user_id": case_row.user_id, "case_id": case_id},
            )
        raise HTTPException(status_code=400, detail="No pending clarification")

    # Build combined question
    missing_fields = c.missing_fields or (pending.get("missing_fields", []) or [])
    answer_map = {str(k): (v or "") for k, v in c.answers.items()}
    """
    answer_map
    {"agreement_date":"Signed 12 May 2019, before the wedding (s 90B).",
    "legal_advice":"Yes, both had separate lawyers; certificates signed.",
    "financial_disclosure":"No, a Queensland property and super were not disclosed.",
    "pressure_duress":"Yes, three days before wedding with ceremony threatened.",
    "changed_circumstances":"Yes, two children were born after signing."}
    """
    pending_clarifications.get(case_row.user_id, {}).pop(case_id, None)
    if not pending_clarifications.get(case_row.user_id):
        pending_clarifications.pop(case_row.user_id, None)
    if Config.ENV == "dev":
        format_and_log("/clarify", "Cache Update", "pending_clarifications", pending_clarifications)

    summarized_dict = await summarize_answer_if_needed(
            {field: answer_map.get(field, "") for field in missing_fields},
            max_words=50,
        )
    """
    answer_map
    {"agreement_date":"Signed 12 May 2019, before the wedding (s 90B).",
    "legal_advice":"Yes, both had separate lawyers; certificates signed.",
    "financial_disclosure":"No, a Queensland property and super were not disclosed.",
    "pressure_duress":"Yes, three days before wedding with ceremony threatened.",
    "changed_circumstances":"Yes, two children were born after signing."}
    """
    topic = pending.get("topic")
    if topic:
        summary_lines = [
            f"- {key.replace('_', ' ').title()}: {value}"
            for key, value in summarized_dict.items()
            if value
        ]
        summary_text = "\n".join(summary_lines)
        if summary_text:
            user_cache = case_summary_sections.setdefault(case_row.user_id, {})
            case_cache = user_cache.setdefault(case_id, {})
            existing_text = case_cache.get(topic, "")
            if existing_text:
                case_cache[topic] = f"{existing_text}\n{summary_text}"
            else:
                case_cache[topic] = summary_text

            # Update DB case_summary JSON for the topic + missing_fields
            try:
                summary_obj = json.loads(case_row.case_summary or "{}")
                if not isinstance(summary_obj, dict):
                    summary_obj = {}
            except (json.JSONDecodeError, TypeError):
                summary_obj = {}

            topic_obj = summary_obj.get(topic, {})
            for field in missing_fields:
                value = summarized_dict.get(field, "")
                if not value:
                    continue
                field_list = topic_obj.get(field)
                if not isinstance(field_list, list):
                    field_list = []
                field_list.append(value)
                topic_obj[field] = field_list

            summary_obj[topic] = topic_obj
            case_row.case_summary = json.dumps(summary_obj)
            db.add(case_row)
            db.commit()
            if Config.ENV == "dev":
                format_and_log(
                    "/clarify",
                    "DB Update",
                    "Case Table",
                    {"action": "update", "case_id": case_row.id, "reason": "Persisted clarification fields"},
                )
                format_and_log(
                    "/clarify",
                    "Summary",
                    "case_summary",
                    {
                        "case_id": case_row.id,
                        "source": "clarify_saved_to_db",
                        "raw_summary_json": (case_row.case_summary or "")[:8000],
                        "parsed_summary": summary_obj,
                    },
                )

            # Update embeddings for the changed topic section
            document = [Document(
                    text=summary_text,
                    metadata={
                        "source_type": "uploaded_case",
                        "case_id": str(case_id),
                        "source": case_row.filename,
                        "summary_section": topic,
                    },
                )]
            model_manager.add_uploaded_case_documents(case_id, document, allow_existing=True)

    # Generate answer with updated context
    history_text = "\n".join([
        f"{'Client' if t['role']=='user' else 'Lawyer'}: {t['content']}"
        for t in session_history.get(case_row.user_id, {}).get(case_id, [])[-8:]
    ])
    if Config.ENV == "dev":
        format_and_log(
            "/clarify",
            "Cache Lookup",
            "session_history",
            {
                "user_id": case_row.user_id,
                "case_id": case_id,
                "items": len(session_history.get(case_row.user_id, {}).get(case_id, [])),
            },
        )

    case_section_text = case_summary_sections.get(case_row.user_id, {}).get(case_id, {}).get(topic, "")
    impact_text = case_summary_sections.get(case_row.user_id, {}).get(case_id, {}).get("impact_analysis", "")
    logger.info(f"Using summary section for topic '{topic}'")
    if not case_section_text:
        logger.warning(f"No summary section found for topic '{topic}' in case {case_id}. Using full summary text as fallback.")
        summary_sections = _build_case_summary_for_query(case_id, db)
        case_section_text = "\n".join([section.get("text", "") for section in summary_sections if section.get("section") == topic])

    pending_question = pending.get("question", "") if isinstance(pending, dict) else ""
    answer, retrieved_nodes = await answer_case_question_withuploadFile(
        question=pending_question,
        case_section_text=case_section_text,
        history_text=history_text,
        topic=topic,
        impact_analysis=impact_text
    )

    # Strip CACHE_SUMMARY from the answer
    parts = answer.split("---CACHE_SUMMARY---")
    main_answer = parts[0].strip()

    qa = QuestionAnswer(
        case_id=case_id,
        user_id=case_row.user_id,
        question=pending_question,
        answer=main_answer,
    )
    db.add(qa)
    db.commit()
    if Config.ENV == "dev":
        format_and_log("/clarify", "DB Update", "QuestionAnswer Table", {"action": "create", "qa_id": qa.id})

    return {
        "answer": main_answer,
        "citations": retrieved_nodes,  # These are actually citations, not nodes
    }


@router.get("/history/{session_id}")
def get_history(
    session_id: str,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_current_user_id),
):
    """Get conversation history for a session."""
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


@router.get("/debug/uploaded_case_embeddings")
def debug_uploaded_case_embeddings(
    case_id: int,
    summary_section: Optional[str] = None,
    limit: int = 5,
):
    if Config.ENV != "dev":
        raise HTTPException(status_code=404, detail="Not found")

    if model_manager.uploaded_cases_index is None:
        return {"items": []}

    vector_store = model_manager.create_or_load_uploaded_cases_index().storage_context.vector_store
    items: list[dict] = []

    if hasattr(vector_store, "data") and hasattr(vector_store.data, "embedding_dict"):
        embedding_dict = vector_store.data.embedding_dict
        metadata_dict = vector_store.data.metadata_dict
        for node_id, embedding in embedding_dict.items():
            metadata = metadata_dict.get(node_id, {})
            if metadata.get("case_id") != str(case_id):
                continue
            if summary_section and metadata.get("summary_section") != summary_section:
                continue
            items.append(
                {
                    "id": node_id,
                    "embedding": embedding,
                    "metadata": metadata,
                }
            )
            if len(items) >= limit:
                break
    elif hasattr(vector_store, "to_dict"):
        data = vector_store.to_dict()
        embedding_dict = data.get("embedding_dict", {})
        metadata_dict = data.get("metadata_dict", {})
        for node_id, embedding in embedding_dict.items():
            metadata = metadata_dict.get(node_id, {})
            if metadata.get("case_id") != str(case_id):
                continue
            if summary_section and metadata.get("summary_section") != summary_section:
                continue
            items.append(
                {
                    "id": node_id,
                    "embedding": embedding,
                    "metadata": metadata,
                }
            )
            if len(items) >= limit:
                break

    return {"items": items}
