import json
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from llama_index.core import Settings

#topc and factor keywords for rule-based clarification when LLM is not available, and for detecting which factors are missing from uploaded case summary for structured prompting and memory patching
TOPIC_FACTORS: Dict[str, Dict[str, List[str]]] = {
    "property_division": {
        "asset_pool": ["asset", "assets", "liability", "debt", "mortgage", "value"],
        "contributions": [
            "financial",
            "contribution",
            "income",
            "salary",
            "payment",
            "non-financial",
            "renovation",
            "improvement",
            "un-remunerated",
            "to the welfare of the family",
            "homemaker",
            "domestic labour",
            "cooking",
            "cleaning",
            "laundry",
            "gardening",
            "caregiver",
            "childcare",
            "school pickup",
            "school dropoff",
            "feeding",
            "bathing",
        ],
        "future_needs": [
            "income",
            "income-earning disparity",
            "effect of orders on income-earning capacity",
            "health",
            "age",
            "care and control of children",
            "caring responsibility for other persons",
            "caregiver",
            "necessary living expenses",
            "reasonable standard of living",
        ],
        "existing_agreements": ["agreement", "bfa", "binding", "order"],
    },
    "children_parenting": {
        "child_ages": ["age", "school", "toddler", "teen"],
        "current_arrangements": ["currently", "live", "reside", "weekend", "schedule"],
        "caregiver_history": ["primary", "carer", "caregiver", "routine"],
        "availability": ["work", "hours", "shift", "availability", "travel"],
        "safety_concerns": ["violence", "abuse", "safety", "order"],
        "child_views": ["child", "preference", "wish", "view"],
    },
    "spousal_maintenance": {
        "income_expenses": ["income", "expense", "budget", "cost", "pay"],
        "earning_capacity": ["work", "job", "employ", "capacity", "qualification"],
        "health_care": ["health", "illness", "disability", "care"],
        "relationship_length": ["years", "duration", "relationship", "marriage"],
        "standard_of_living": ["lifestyle", "standard", "living"],
    },
    "family_violence_safety": {
        "incidents": ["incident", "violence", "abuse", "threat", "assault"],
        "protection_orders": ["order", "avro", "intervention", "restraining"],
        "police_court": ["police", "court", "report", "charge"],
        "child_exposure": ["child", "witness", "exposed"],
        "safety_plan": ["safety", "plan", "support", "shelter"],
    },
    "prenup_postnup": {
        "agreement_date": ["date", "signed", "before", "after"],
        "legal_advice": ["lawyer", "legal", "advice", "independent"],
        "financial_disclosure": ["disclosure", "assets", "liabilities", "full"],
        "pressure_duress": ["pressure", "duress", "coerce", "forced"],
        "changed_circumstances": ["children", "assets", "change", "major"],
    },
}

# Client-facing question prompts for each legal factor(used in clarification questions when uploaded case summary is missing key info for RAG)
QUESTION_MAP = {
    # --- Property Division ---
    "asset_pool": "Could you provide details about the asset pool, including values for property, superannuation, and any debts?",
    "contributions": "Please describe both the financial contributions (like salary) and non-financial contributions (like homemaking/parenting) made by each party.",
    "future_needs": "Are there any factors affecting future needs, such as a significant difference in income-earning capacity or health issues?",
    "existing_agreements": "Are there any existing BFAs, child support agreements, or court orders already in place?",
    "just_equitable": "Is there anything else that makes your proposed split 'just and equitable' in your view?",
    "living_arrangements_prop": "What are the current living arrangements for both parties post-separation?",

    # --- Children & Parenting ---
    "child_ages": "What are the ages of the children? This helps determine their developmental needs.",
    "current_arrangements": "What is the current schedule? Please describe where the children live and how much time they spend with each parent.",
    "caregiver_history": "Who has historically been the primary caregiver for the children's daily routines?",
    "availability": "What are the parents' work schedules or availability to care for the children during the week?",
    "safety_concerns": "Are there any family violence or safety risks we should be aware of regarding the children's environment?",
    "child_views": "Have the children expressed any particular wishes or views regarding their living arrangements?",
    "allegations": "Are there any specific allegations of neglect or harm that need to be addressed?",
    "expert_evidence": "Has there been any previous involvement from family consultants or expert reports?",

    # --- Spousal Maintenance ---
    "income_expenses": "What are your current weekly/monthly income and necessary living expenses?",
    "earning_capacity": "What are your professional qualifications, and is there anything currently preventing you from working full-time?",
    "health_care": "Are there any ongoing health issues or disabilities that require significant care or expense?",
    "relationship_length": "How many years were you in the relationship or marriage?",
    "standard_of_living": "How would you describe the standard of living enjoyed during the relationship?",
    "need": "Can you elaborate on your current financial need for maintenance?",
    "capacity_to_pay": "Does the other party have the financial capacity to pay maintenance after meeting their own expenses?",

    # --- Family Violence & Safety ---
    "incidents": "Could you describe any specific incidents of violence, threats, or coercive control?",
    "protection_orders": "Are there currently any AVOs, IVOs, or other protection orders in place?",
    "police_court": "Have there been any police reports filed or criminal charges laid related to family violence?",
    "child_exposure": "Were the children present during or exposed to the effects of any violent incidents?",
    "safety_plan": "Do you currently have a safety plan or support services in place?",

    # --- Prenup & Postnup (BFA) ---
    "agreement_date": "When was the agreement signed? Was it before (Section 90B) or after (Section 90C) the marriage?",
    "legal_advice": "Did both parties receive independent legal advice from separate lawyers before signing?",
    "financial_disclosure": "Was there full and frank financial disclosure of all assets and liabilities before signing?",
    "pressure_duress": "Was there any pressure, urgency, or 'unfair' circumstances surrounding the signing of the document?",
    "changed_circumstances": "Have there been major changes since signing, such as the birth of a child, that the agreement didn't account for?"
    }

TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "property_division": [
        "asset pool", "liabilities", "superannuation", "valuations",  # The Assets
        "financial contributions", "non-financial contributions", "homemaker", "parenting contributions", # The Past
        "future needs", "earning capacity", "health", "age", "financial resources", # The Future
        "just and equitable", "percentage split", "initial contributions", "inheritance" # Legal Thresholds
    ],
    "children_parenting": [
        "living arrangements", "spend time", "communication", "changeover", # The Routine
        "best interests", "primary carer", "parental responsibility", "decision making", # Legal Status
        "safety", "risk of harm", "family violence", "abuse", "neglect", # Safety/Risk
        "child's views", "wishes", "maturity", "expert reports", "family consultant" # Evidence
    ],
    "spousal_maintenance": [
        "financial need", "adequately support", "capacity to pay", # The Test
        "income", "expenses", "budget", "shortfall", # The Math
        "earning capacity", "vocational skills", "health", "illness", # Ability to Work
        "duration of marriage", "standard of living" # Context
    ],
    "family_violence_safety": [
        "incidents", "physical abuse", "emotional abuse", "coercive control", # Behaviors
        "protection orders", "intervention orders", "IVOs", "AVOs", "undertakings", # Legal Orders
        "police reports", "charges", "criminal history", "witnesses", # Evidence
        "impact on children", "exposure to violence", "safety planning" # Future Risk
    ],
    "prenup_postnup": [
        "binding financial agreement", "BFA", "pre-nuptial", "post-nuptial", # The Document
        "independent legal advice", "certificates of advice", "full disclosure", # Validity
        "duress", "undue influence", "unconscionable conduct", "pressure", # Fairness
        "material change in circumstances", "hardship", "children's impact" # Set-aside factors
    ],
}

_TOPIC_LABELS = {
    "property_division": "Property division",
    "children_parenting": "Children custody & parenting",
    "spousal_maintenance": "Spousal maintenance",
    "family_violence_safety": "Family violence & safety",
    "prenup_postnup": "Pre/post-nuptial agreement",
}

# Dotted paths used for structured completeness checks and memory patching.(for clarifying questions based on uploaded case summary)
TOPIC_REQUIRED_FIELDS: Dict[str, List[Tuple[str, str]]] = {
    "property_division": [
        ("property.asset_pool", "asset pool"),
        ("property.contributions", "contributions"),
        ("property.future_needs", "future needs"),
        ("property.just_equitable", "just and equitable assessment"),
        ("property.existing_agreements", "existing agreements"),
    ],
    "children_parenting": [
        ("parenting.child_ages", "child ages"),
        ("parenting.current_arrangements", "current parenting arrangements"),
        ("parenting.caregiver_history", "caregiver history"),
        ("parenting.availability", "parental availability"),
        ("parenting.safety_concerns", "safety concerns"),
        ("parenting.child_views", "child views"),
    ],
    "spousal_maintenance": [
        ("spousal_maintenance.need", "claimant need"),
        ("spousal_maintenance.capacity_to_pay", "capacity to pay"),
        ("spousal_maintenance.income_expenses", "income and expenses"),
        ("spousal_maintenance.earning_capacity", "earning capacity"),
        ("spousal_maintenance.health_care", "health care"),
        ("spousal_maintenance.relationship_length", "relationship length"),
        ("spousal_maintenance.standard_of_living", "standard of living"),
    ],
    "family_violence_safety": [
        ("family_violence_safety.incidents", "violence or abuse incidents"),
        ("family_violence_safety.protection_orders", "protection orders"),
        ("family_violence_safety.police_court", "police or court involvement"),
        ("family_violence_safety.child_exposure", "child exposure"),
        ("family_violence_safety.safety_plan", "safety plan"),
    ],
    "prenup_postnup": [
        ("prenup_postnup.agreement_date", "agreement date"),
        ("prenup_postnup.legal_advice", "independent legal advice"),
        ("prenup_postnup.financial_disclosure", "financial disclosure"),
        ("prenup_postnup.pressure_duress", "pressure or duress"),
        ("prenup_postnup.changed_circumstances", "changed circumstances"),
    ],
}

MAX_CLARIFICATION_QUESTIONS = 5


def parse_summary_json(summary_text: Optional[str]) -> Optional[dict]:
    if not summary_text:
        return None
    try:
        parsed = json.loads(summary_text)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def serialize_summary_json(summary_obj: dict) -> str:
    return json.dumps(summary_obj, ensure_ascii=False)


async def detect_topic(question: str) -> str:
    # Use the TOPIC_KEYWORDS we built as the reference
    prompt = f"""
    Based on the following user question, which legal topic does it belong to?
    Question: "{question}"
    
    Topics: {list(TOPIC_KEYWORDS.keys())}
    
    Return only the topic name or 'other' if it doesn't fit.
    """
    response = await Settings.llm.acomplete(prompt)
    return response.text.strip().lower()


def missing_factors(text: str, topic: str) -> List[str]:
    text_lower = text.lower()
    missing = []
    for factor, keywords in TOPIC_FACTORS.get(topic, {}).items():
        if not any(k in text_lower for k in keywords):
            missing.append(factor)
    return missing


def _llm_prompt(text: str, context_summary: Optional[str]) -> str:
    topics = "\n".join(
        f"- {key}: {_TOPIC_LABELS.get(key, key)} (factors: {', '.join(factors.keys())})"
        for key, factors in TOPIC_FACTORS.items()
    )
    summary_block = ""
    if context_summary:
        summary_block = f"\nUploaded case summary:\n{context_summary}\n"
    return (
        "You are a family law intake assistant. Decide the best topic and ask for missing key factors.\n"
        "Return STRICT JSON with keys: topic, needs_clarification, questions.\n"
        "Use topic as one of the provided keys, or null if none match.\n"
        "If the question is clear, set needs_clarification to false and questions to [].\n"
        "If unclear, ask up to 3 concise follow-up questions about missing factors.\n\n"
        f"Topics:\n{topics}\n\n"
        f"User question:\n{text}\n"
        f"{summary_block}"
    )


def _path_get_list(data: dict, dotted_path: str) -> List[str]:
    current = data
    parts = dotted_path.split(".")
    for part in parts:
        if not isinstance(current, dict):
            return []
        current = current.get(part)
    if isinstance(current, list):
        return [str(item).strip() for item in current if str(item).strip()]
    if isinstance(current, str) and current.strip():
        return [current.strip()]
    return []


def _path_append_item(data: dict, dotted_path: str, value: str) -> None:
    parts = dotted_path.split(".")
    current = data
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    leaf = parts[-1]
    target = current.get(leaf)
    if isinstance(target, list):
        target.append(value)
    elif isinstance(target, str) and target.strip():
        current[leaf] = [target, value]
    else:
        current[leaf] = [value]


def _summarize_if_needed(text: str, max_words: int = 50) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + "…"


async def summarize_answer_if_needed(answer_dict: Dict[str, str], max_words: int = 50) -> Dict[str, str]:
    if not answer_dict:
        return {}

    llm = Settings.llm
    summarized: Dict[str, str] = {}

    for key, value in answer_dict.items():
        cleaned = (value or "").strip()
        if not cleaned:
            summarized[key] = ""
            continue
        if len(cleaned.split()) <= max_words:
            summarized[key] = cleaned
            continue

        prompt = (
            "Summarize the following text to 50 words or fewer. "
            "Keep concrete facts, dates, amounts, and parties. "
            "Return only the summary.\n\n"
            f"TEXT:\n{cleaned}"
        )
        try:
            response = await llm.acomplete(prompt)
            summary = (response.text if hasattr(response, "text") else str(response)).strip()
            summarized[key] = summary or _summarize_if_needed(cleaned, max_words)
        except Exception:
            summarized[key] = _summarize_if_needed(cleaned, max_words)

    return summarized


def validate_legal_config(factors_dict: dict, map_dict: dict):
    """
    Checks if every sub-category in TOPIC_FACTORS has a 
    matching user-friendly question in QUESTION_MAP.
    """
    missing_keys = []
    
    for topic, sub_categories in factors_dict.items():
        for factor_key in sub_categories.keys():
            if factor_key not in map_dict:
                missing_keys.append(f"{topic} -> {factor_key}")
    
    if missing_keys:
        print("⚠️ CONFIGURATION WARNING: The following factors are missing questions:")
        for missing in missing_keys:
            print(f" - {missing}")
    else:
        print("✅ SUCCESS: All legal factors are mapped to questions.")


def get_clarification_for_topic(
    topic: str, 
    case_summary: str
) -> list[str]:
    """
    Checks the case_summary string for the detected topic against TOPIC_FACTORS.
    Returns a list of clarifying questions if key info is missing.
    """
    questions = []
    missing_fields = []
    # Get the required factors for THIS detected topic
    factors = TOPIC_FACTORS.get(topic, {})

    for factor_name, keywords in factors.items():
        has_info = any(kw.lower() in case_summary.lower() for kw in keywords)

        if not has_info:
            missing_fields.append(factor_name) # Capture the key!
            q_text = QUESTION_MAP.get(factor_name, f"Details about {factor_name}?")
            questions.append(q_text)

    # Smart Limit: Don't overwhelm the user.
    limit = MAX_CLARIFICATION_QUESTIONS
    return  missing_fields[:limit], questions[:limit]


def _validate_info_quality(text: str, factor_name: str) -> bool:
    # We clean up the factor name for the LLM (e.g., 'asset_pool' -> 'Asset Pool')
    readable_factor = factor_name.replace('_', ' ').title()
    
    prompt = f"""
    ROLE: Legal Data Auditor
    TASK: Determine if the provided text contains SUBSTANTIVE INFORMATION regarding '{readable_factor}'.
    
    TEXT TO EVALUATE: 
    "{text}"
    
    SCORING CRITERIA:
    - Return 'YES' if:
        1. There are specific quantities, dates, names, or locations (e.g., "$500k mortgage", "married in 2012").
        2. There is a specific narrative of an event (e.g., "The parties separated after an argument in June").
        3. The user explicitly states the factor is NIL or N/A (e.g., "No history of violence", "No assets").
    - Return 'NO' if:
        1. The text only mentions the word without context (e.g., "We have assets.")
        2. The text is a placeholder (e.g., "TBA", "Unknown", "Will provide later").
        3. The text is purely emotional without factual data (e.g., "It's all very unfair").
    
    OUTPUT: Return exactly one word: 'YES' or 'NO'.
    """
    
    response = Settings.llm.acomplete(prompt)
    return "YES" in response.text.upper()


def clarification_questions(topic: str, missing: List[str]) -> List[str]:
    label_map = {path: label for path, label in TOPIC_REQUIRED_FIELDS.get(topic, [])}
    prompts = []
    for dotted_path in missing[:MAX_CLARIFICATION_QUESTIONS]:
        label = label_map.get(dotted_path, dotted_path.split(".")[-1].replace("_", " "))
        prompts.append(f"Please provide details about {label}.")
    return prompts


def apply_clarification_answers(
    context_summary: Optional[str],
    topic: Optional[str],
    missing_fields: List[str],
    answers: List[str],
) -> Optional[str]:
    """Patch the uploaded summary with user-provided answers to clarification questions for more complete context."""
    if not topic:
        return context_summary
    summary_obj = parse_summary_json(context_summary)
    if summary_obj is None:
        return context_summary

    field_map = {path.split(".")[-1]: path for path, _ in TOPIC_REQUIRED_FIELDS.get(topic, [])}
    topic_prefix_map = {
        "property_division": "property",
        "children_parenting": "parenting",
        "spousal_maintenance": "spousal_maintenance",
        "family_violence_safety": "family_violence_safety",
        "prenup_postnup": "prenup_postnup",
    }
    topic_prefix = topic_prefix_map.get(topic)

    for idx, dotted_path in enumerate(missing_fields):
        if idx >= len(answers):
            break
        answer = (answers[idx] or "").strip()
        if not answer:
            continue
        normalized_path = dotted_path
        if "." not in dotted_path:
            normalized_path = field_map.get(dotted_path)
            if not normalized_path and topic_prefix:
                normalized_path = f"{topic_prefix}.{dotted_path}"
        if not normalized_path:
            continue
        _path_append_item(summary_obj, normalized_path, answer)

    return serialize_summary_json(summary_obj)


def get_topic_section_text(context_summary: Optional[str], topic: Optional[str]) -> str:
    """extract section-specific context from uploaded summary for in-context prompting"""
    if not context_summary or not topic:
        return ""
    summary_obj = parse_summary_json(context_summary)
    if summary_obj is None:
        return ""

    lines: List[str] = []
    for dotted_path, label in TOPIC_REQUIRED_FIELDS.get(topic, []):
        values = _path_get_list(summary_obj, dotted_path)
        for value in values:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def get_clarification_llm(text: str, context_summary: Optional[str]) -> Tuple[Optional[str], List[str]]:
    llm = Settings.llm
    if llm is None:
        return get_clarification_rules(text)

    try:
        response = llm.complete(_llm_prompt(text, context_summary))
        raw = response.text if hasattr(response, "text") else str(response)
        data = json.loads(raw)
        topic = data.get("topic")
        questions = data.get("questions") or []
        if data.get("needs_clarification") is True and questions:
            return topic, questions
        return topic, []
    except Exception:
        return get_clarification_rules(text)


def get_clarification_rules(text: str) -> Tuple[Optional[str], List[str]]:
    topic = detect_topic(text)
    if not topic:
        return None, []
    missing = missing_factors(text, topic)
    if not missing:
        return topic, []
    prompts = [f"Please provide details about {m.replace('_', ' ')}." for m in missing[:MAX_CLARIFICATION_QUESTIONS]]
    return topic, prompts


class AuditResponse(BaseModel):
    is_sufficient: bool = Field(description="True if facts are enough to answer the question")
    missing_pillars: List[str] = Field(description="List of legal pillars that are absent")
    clarifying_questions: List[str] = Field(description="1-3 empathetic questions to ask the user")


def get_clarification(text: str, context_summary: Optional[str] = None) -> Tuple[Optional[str], List[str]]:
    return get_clarification_llm(text, context_summary)
