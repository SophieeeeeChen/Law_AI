import json
import os
from pyexpat import model
import re
from typing import Iterable, List, Optional, Tuple
import traceback
from llama_index.core.llms import ChatMessage, TextBlock
from llama_index.core import Settings
from typing import Optional

from app.core.config import Config
from app.core.logger import logger
from app.services.summary_prompt import build_case_summary_prompt
from typing import List, Dict, Any, Iterable

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

DEBUG_SUMMARY = str(getattr(Config, "ENV", "")).lower() in {"dev", "development"}
gemini_ai = genai.GenerativeModel('gemini-3-flash-preview')

# Shared list caps for summaries (kept in one place so uploads + batch stay consistent).
SUMMARY_LIST_LIMITS_PRIMARY = {
    "facts": 18,
    "outcome_orders": 15,
    "reasons_rationale": 15,
    "uncertainties": 6,
    "asset_pool": 15,
    "contributions": 12,
    "future_needs": 10,
    "just_equitable": 10,
    "living_arrangements": 8,
    "existing_agreements": 8,
    "need": 10,
    "capacity_to_pay": 10,
    "statutory_factors": 10,
    "income_expenses": 10,
    "earning_capacity": 10,
    "health_care": 8,
    "relationship_length": 5,
    "standard_of_living": 8,
    "child_ages": 10,
    "current_arrangements": 10,
    "caregiver_history": 10,
    "availability": 8,
    "safety_concerns": 10,
    "child_views": 10,
    "allegations": 10,
    "expert_evidence": 10,
    "best_interests": 10,
    "notable_conduct_and_judicial_commentary": 10,
    "orders": 10,
    "incidents": 10,
    "protection_orders": 8,
    "police_court": 8,
    "child_exposure": 8,
    "safety_plan": 8,
    "agreement_date": 4,
    "legal_advice": 8,
    "financial_disclosure": 8,
    "pressure_duress": 8,
    "changed_circumstances": 8,
    "parties": 10,
    "pivotal_findings": 10,
    "statutory_pivots": 10,
    "reasoning": 12,
    "evidentiary_gaps": 10,
    "general_credibility_risk": 10,
}

SUMMARY_LIST_LIMITS_FALLBACK = {
    **SUMMARY_LIST_LIMITS_PRIMARY,
    "facts": 10,
    "issues": 6,
    "outcome_orders": 6,
    "reasons_rationale": 8,
    "uncertainties": 3,
    "pivotal_findings": 8,
    "statutory_pivots": 8,
}


def safe_parse_summary_json(text: str) -> dict:
    """Parse JSON tolerantly, handling common LLM output issues."""
    text = (text or "").strip()
    if not text:
        return {}

    # Remove code fences
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text, flags=re.IGNORECASE).strip()

    # Fix common LLM errors
    # 1. Remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    # 2. Fix duplicate brackets like ]] or }} 
    cleaned = re.sub(r"\]\s*\](\s*,)", r"]\1", cleaned)
    cleaned = re.sub(r"\}\s*\}(\s*,)", r"}\1", cleaned)
    # 3. Fix missing comma between ] and "key"
    cleaned = re.sub(r'\](\s*)"(\w+)":', r'],\1"\2":', cleaned)

    # Try standard parse
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    # Try raw_decode for first valid object
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback: extract first {...} block
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    logger.warning(f"Failed to parse JSON. First 500 chars: {cleaned[:500]}")
    return {}
    


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+(?:'\w+)?\b", text or ""))


def apply_list_limits(summary: dict, limits: dict) -> dict:
    """
    Apply length limits to list fields in the summary dictionary.
    Handles both top-level lists and lists nested within subdictionaries.
    
    Args:
        summary: The summary dictionary to process
        limits: Dictionary mapping field names to maximum list lengths
        
    Returns:
        The modified summary dictionary
    """
    def normalize_list(value):
        """Convert various input types to a normalized list of strings."""
        if value is None:
            return []
        if isinstance(value, list):
            return [v for v in value if isinstance(v, str) and v.strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def walk(obj, parent_key=""):
        """Recursively walk the dictionary and apply limits to lists."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    walk(v, parent_key=k)
                elif v is None and k in {"outcome_orders"}:
                    obj[k] = None
                elif k == "description" and isinstance(v, str):
                    pass  # FIX: preserve description as a string, don't convert to list
                elif isinstance(v, (list, str)) or v is None:
                    items = normalize_list(v)
                    limit = limits.get(k, 5)
                    obj[k] = items[:limit] if items else []

    walk(summary)
    return summary


def summary_json_to_sections(summary: dict, *, include_outcome_reasons: bool = True) -> dict:
    """
    COMPLETE PRODUCTION VERSION: 
    Converts nested legal JSON into formatted Markdown sections for RAG.
    Ensures every section includes its specific reasoning, orders, and legal impacts.
    Omits any section that contains no meaningful content.
    """
    sections = {}

    _EMPTY_VALUES = {
        "", "n/a", "na", "none", "not applicable", "not stated",
        "no information", "no data", "unknown", "nil", "string",
        "not applicable.", "not stated.", "n/a.", "none.",
    }

    def is_meaningful(value: str) -> bool:
        """Return True only if the value carries real information."""
        return bool(value) and value.strip().lower() not in _EMPTY_VALUES

    def add_section(key: str, lines: List[str]):
        if lines:
            sections[key] = "\n\n".join(lines).strip()

    def add_grouped_items(line_list: List[str], header: str, items: Any):
        if isinstance(items, str):
            items = [items]
        meaningful = [str(item).strip() for item in (items or []) if is_meaningful(str(item))]
        if meaningful:
            line_list.append(f"### {header}")
            for item in meaningful:
                line_list.append(f"  • {item}")

    def add_legal_metadata(line_list: List[str], section_data: dict):
        if not isinstance(section_data, dict):
            return
        add_grouped_items(line_list, "Orders", section_data.get("orders", []))
        add_grouped_items(line_list, "Reasoning", section_data.get("reasoning", []))
        impact = section_data.get("impact_analysis", {})
        if isinstance(impact, dict):
            add_grouped_items(line_list, "Pivotal Findings", impact.get("pivotal_findings", []))
            add_grouped_items(line_list, "Statutory Pivots", impact.get("statutory_pivots", []))
        add_grouped_items(line_list, "Evidentiary Gaps", section_data.get("evidentiary_gaps", []))

    # --- 1. CORE CASE INFO ---
    facts = [f"  • {f}" for f in summary.get("facts", []) if is_meaningful(f)]
    if facts:
        add_section("facts", ["### Facts"] + facts)

    # --- 2. PROPERTY DIVISION ---
    prop = summary.get("property_division", {}) or {}
    prop_lines = []
    add_grouped_items(prop_lines, "Asset Pool", prop.get("asset_pool", []))
    add_grouped_items(prop_lines, "Contributions", prop.get("contributions", []))
    add_grouped_items(prop_lines, "Future Needs", prop.get("future_needs", []))
    add_grouped_items(prop_lines, "Just & Equitable", prop.get("just_equitable", []))
    add_grouped_items(prop_lines, "Living Arrangements", prop.get("living_arrangements", []))       # NEW
    add_grouped_items(prop_lines, "Existing Agreements", prop.get("existing_agreements", []))       # NEW
    add_legal_metadata(prop_lines, prop)
    add_section("property_division", prop_lines)

    # --- 3. CHILDREN & PARENTING ---
    parenting = summary.get("children_parenting", {}) or {}
    parenting_lines = []
    add_grouped_items(parenting_lines, "Child Ages", parenting.get("child_ages", []))
    add_grouped_items(parenting_lines, "Current Arrangements", parenting.get("current_arrangements", []))   # NEW
    add_grouped_items(parenting_lines, "Caregiver History", parenting.get("caregiver_history", []))         # NEW
    add_grouped_items(parenting_lines, "Availability", parenting.get("availability", []))                   # NEW
    add_grouped_items(parenting_lines, "Safety Concerns", parenting.get("safety_concerns", []))
    add_grouped_items(parenting_lines, "Child Views", parenting.get("child_views", []))                     # NEW
    add_grouped_items(parenting_lines, "Allegations", parenting.get("allegations", []))                     # NEW
    add_grouped_items(parenting_lines, "Expert Evidence", parenting.get("expert_evidence", []))
    add_grouped_items(parenting_lines, "Best Interests", parenting.get("best_interests", []))
    add_legal_metadata(parenting_lines, parenting)
    add_section("children_parenting", parenting_lines)

    # --- 3b. NOTABLE CONDUCT & JUDICIAL COMMENTARY (standalone for focused RAG retrieval) ---
    conduct_lines = []
    add_grouped_items(conduct_lines, "Notable Conduct & Judicial Commentary", parenting.get("notable_conduct_and_judicial_commentary", []))
    add_section("notable_conduct_and_judicial_commentary", conduct_lines)

    # --- 4. SPOUSAL MAINTENANCE ---
    sm = summary.get("spousal_maintenance", {}) or {}
    sm_lines = []
    add_grouped_items(sm_lines, "Need", sm.get("need", []))
    add_grouped_items(sm_lines, "Capacity to Pay", sm.get("capacity_to_pay", []))
    add_grouped_items(sm_lines, "Statutory Factors", sm.get("statutory_factors", []))                       # NEW
    add_grouped_items(sm_lines, "Income & Expenses", sm.get("income_expenses", []))
    add_grouped_items(sm_lines, "Earning Capacity", sm.get("earning_capacity", []))                         # NEW
    add_grouped_items(sm_lines, "Health & Care", sm.get("health_care", []))                                 # NEW
    add_grouped_items(sm_lines, "Relationship Length", sm.get("relationship_length", []))                   # NEW
    add_grouped_items(sm_lines, "Standard of Living", sm.get("standard_of_living", []))                     # NEW
    add_legal_metadata(sm_lines, sm)
    add_section("spousal_maintenance", sm_lines)

    # --- 5. FAMILY VIOLENCE & SAFETY ---
    violence = summary.get("family_violence_safety", {}) or {}
    fv_lines = []
    add_grouped_items(fv_lines, "Incidents", violence.get("incidents", []))
    add_grouped_items(fv_lines, "Protection Orders", violence.get("protection_orders", []))
    add_grouped_items(fv_lines, "Police & Court", violence.get("police_court", []))                         # NEW
    add_grouped_items(fv_lines, "Child Exposure", violence.get("child_exposure", []))
    add_grouped_items(fv_lines, "Safety Plan", violence.get("safety_plan", []))                             # NEW
    add_legal_metadata(fv_lines, violence)
    add_section("family_violence_safety", fv_lines)

    # --- 6. PRENUP / POSTNUP ---
    prenup = summary.get("prenup_postnup", {}) or {}
    pn_lines = []
    add_grouped_items(pn_lines, "Agreement Details", prenup.get("agreement_date", []))
    add_grouped_items(pn_lines, "Legal Advice", prenup.get("legal_advice", []))                             # NEW
    add_grouped_items(pn_lines, "Disclosure & Advice", prenup.get("financial_disclosure", []))
    add_grouped_items(pn_lines, "Pressure & Duress", prenup.get("pressure_duress", []))                     # NEW
    add_grouped_items(pn_lines, "Changed Circumstances", prenup.get("changed_circumstances", []))           # NEW
    add_legal_metadata(pn_lines, prenup)
    add_section("prenup_postnup", pn_lines)

    overall_impact = summary.get("overall_impact_analysis", {}) or {}
    if isinstance(overall_impact, dict):
        impact_lines = []
        desc = overall_impact.get("description", "")
        if isinstance(desc, list):                                                                          # FIX: handle corrupted list
            desc = desc[0] if desc else ""
        if is_meaningful(str(desc)):
            impact_lines.append("### Overall Impact Analysis")
            impact_lines.append(f"  • {str(desc).strip()}")
        add_grouped_items(impact_lines, "Pivotal Findings", overall_impact.get("pivotal_findings", []))
        add_grouped_items(impact_lines, "Statutory Pivots", overall_impact.get("statutory_pivots", []))
        add_section("overall_impact_analysis", impact_lines)

    credibility_lines = []
    add_grouped_items(credibility_lines, "General Credibility & Risk", summary.get("general_credibility_risk", []))
    add_section("general_credibility_risk", credibility_lines)
    
    # --- 7. OUTCOME, IMPACT, RATIONALE, CREDIBILITY (each as its own section) ---
    if include_outcome_reasons:
        # 7a. Final Orders
        outcome_lines = []
        add_grouped_items(outcome_lines, "Final Orders", summary.get("outcome_orders", []))
        add_section("outcome_orders", outcome_lines)

        # 7c. Reasons & Rationale
        rationale_lines = []
        add_grouped_items(rationale_lines, "Reasons & Rationale", summary.get("reasons_rationale", []))
        add_section("reasons_rationale", rationale_lines)

    return sections

def _iter_list_nodes(summary: dict):
    if isinstance(summary, dict):
        for k, v in summary.items():
            if isinstance(v, list):
                yield summary, k, v
            elif isinstance(v, dict):
                yield from _iter_list_nodes(v)

def _summary_word_limits(case_text: str) -> Tuple[int, int]:
    """Return (target_words, max_words) scaled to input complexity."""
    input_words = len(case_text.split())
    if input_words < 3_000:
        return 800, 1_200
    elif input_words < 10_000:
        return 1_200, 1_800
    elif input_words < 25_000:
        return 1_600, 2_500
    elif input_words < 50_000:
        return 2_200, 2_800
    else:                        
        return 2_800, 3_500

def generate_summary_dict(
    case_text: str,
    path_stem: str,
    *,
    list_limits_primary: dict = SUMMARY_LIST_LIMITS_PRIMARY,
    llm=None,
    case_name: Optional[str] = None,
) -> dict:
    case_text = (case_text or "").strip()
    print(f"Generating summary for {path_stem} (input ~{word_count(case_text)} words)"  )
    target_words, max_words = _summary_word_limits(case_text)
    llm = Settings.llm if llm is None else llm
    model_name = getattr(llm, "model", "default").lower()
    #to check how many tokens each case consumes(for test)
    RATES = {
        "gemini-2.5-flash": {"in": 0.30, "out": 2.50},
        "gpt-5.2": {"in": 1.75, "out": 14.00},
        "claude-4.5-sonnet": {"in": 3.00, "out": 15.00},
        # Gemini 3 Family (2026 Preview)
        "gemini-3-flash": {"in": 0.50, "out": 3.00},
        "gemini-3.1-flash-lite": {"in": 0.25, "out": 1.50},
        "gemini-3.1-pro": {"in": 2.00, "out": 12.00}, # Under 200k context
        "default": {"in": 2.00, "out": 10.00}
    }
    prompt = build_case_summary_prompt(
        case_text=case_text,
        target_words=target_words,
        max_words=max_words,
    )

    try:
        
        # Then pass it in the generate_content call (around line 82):
        # response = gemini_ai.generate_content(prompt)

        # Apply it to your call
        response = llm.complete(prompt)

        raw = response.text if hasattr(response, "text") else str(response)
        # print(f"DEBUG - Finish Reason: {response.raw.get('candidates', [{}])[0].get('finish_reason')}")
        # print(f"Raw LLM response: {raw}")
        # Extract token usage - handle both OpenAI and Anthropic formats
        prompt_tokens = 0
        completion_tokens = 0
        thinking_tokens = 0
        if hasattr(response, "raw") and response.raw and "usage_metadata" in response.raw:
            usage = response.raw.get("usage_metadata", {})
            prompt_tokens = usage.get("prompt_token_count", 0) or 0
            completion_tokens = usage.get("candidates_token_count", 0) or 0
            thinking_tokens = usage.get("thoughts_token_count", 0) or 0
        
        # Fallback for other models (OpenAI/Anthropic)
        elif hasattr(response, "raw") and hasattr(response.raw, "usage"):
            usage_obj = getattr(response.raw, "usage", None)
            if usage_obj:
                prompt_tokens = getattr(usage_obj, "prompt_tokens", getattr(usage_obj, "input_tokens", 0)) or 0
                completion_tokens = getattr(usage_obj, "completion_tokens", getattr(usage_obj, "output_tokens", 0)) or 0
        # --- UPDATED COST CALCULATION ---
        
        # Fallback to 'gpt-5' or 'claude-3-7-sonnet' based on class if name doesn't match
        # Mapping logic for 2026 Model Suite
        if "gemini-3" in model_name:
            rate_key = "gemini-3-flash"
        elif "gemini-2.5" in model_name:
            rate_key = "gemini-2.5-flash"
        elif "gpt-5" in model_name:
            rate_key = "gpt-5.2"
        elif "claude-4.5" in model_name:
            rate_key = "claude-4.5-sonnet"
        elif "claude-4" in model_name:
            # Catch-all for other Claude 4 variants
            rate_key = "claude-4.5-sonnet"
        else:
            rate_key = "default"

        rate = RATES.get(rate_key, RATES["default"])
        
        total_out = completion_tokens + thinking_tokens
        cost = ((prompt_tokens * rate["in"]) + (total_out * rate["out"])) / 1_000_000

        record_entry = {
            "case_name": case_name,
            "model_name": model_name,
            "cost": round(cost, 6),
            "tokens_in": prompt_tokens,
            "tokens_out": completion_tokens,
            "tokens_thought": thinking_tokens # New for 2026 tracking
        }

        # Saving log... (as per your code)
        _cost_log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs", "record_cost_gemini.json")
        os.makedirs(os.path.dirname(_cost_log_path), exist_ok=True)
        with open(_cost_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record_entry) + "\n")

        clean_json = raw.replace("```json", "").replace("```", "").strip()
        summary = safe_parse_summary_json(clean_json)
        apply_list_limits(summary, list_limits_primary)
        
        return summary
    except Exception as e:
        raise e


