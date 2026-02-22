import json
import os
import re
from typing import Iterable, List, Optional, Tuple
import traceback
from llama_index.core import Settings

from app.core.config import Config
from app.core.logger import logger
from app.services.summary_prompt import build_case_summary_prompt, empty_case_summary


DEBUG_SUMMARY = str(getattr(Config, "ENV", "")).lower() in {"dev", "development"}


# Shared list caps for summaries (kept in one place so uploads + batch stay consistent).
SUMMARY_LIST_LIMITS_PRIMARY = {
    "facts": 16,
    "issues": 10,
    "outcome_orders": 10,
    "reasons_rationale": 12,
    "uncertainties": 4,
    "asset_pool": 8,
    "contributions": 10,
    "future_needs": 8,
    "just_equitable": 8,
    "living_arrangements": 6,
    "existing_agreements": 6,
    "need": 8,
    "capacity_to_pay": 8,
    "statutory_factors": 8,
    "income_expenses": 8,
    "earning_capacity": 8,
    "health_care": 6,
    "relationship_length": 3,
    "standard_of_living": 6,
    "child_ages": 6,
    "current_arrangements": 8,
    "caregiver_history": 8,
    "availability": 6,
    "safety_concerns": 8,
    "child_views": 8,
    "allegations": 8,
    "expert_evidence": 6,
    "best_interests": 8,
    "orders": 10,
    "incidents": 8,
    "protection_orders": 6,
    "police_court": 6,
    "child_exposure": 6,
    "safety_plan": 6,
    "agreement_date": 2,
    "legal_advice": 6,
    "financial_disclosure": 6,
    "pressure_duress": 6,
    "changed_circumstances": 6,
    "parties": 8,
    "pivotal_findings": 8,
    "statutory_pivots": 8,
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
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {"raw_summary": (text or "").strip()}


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+(?:'\w+)?\b", text or ""))


def apply_list_limits(summary: dict, limits: dict) -> dict:
    def normalize_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return [v for v in value if isinstance(v, str) and v.strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    walk(v)
                elif v is None and k in {"outcome_orders"}:
                    # Preserve explicit nulls for fields where we want to distinguish
                    # "not provided" from an empty list.
                    obj[k] = None
                elif isinstance(v, list) or isinstance(v, str) or v is None:
                    items = normalize_list(v)
                    limit = limits.get(k, 5)
                    obj[k] = items[:limit]

    walk(summary)
    return summary


def summary_json_to_sections(summary: dict, *, include_outcome_reasons: bool = True) -> List[dict]:
    sections = []

    def add_section(key: str, lines: List[str]):
        if lines:
            sections.append({"section": key, "text": "\n".join(lines)})

    def add_items(line_list: List[str], label: str, items: Iterable[str]):
        for item in items:
            if item:
                line_list.append(f"- {label}: {item}")

    fact_lines: List[str] = []
    add_items(fact_lines, "Fact", summary.get("facts", []))
    add_section("facts", fact_lines)

    issue_lines: List[str] = []
    add_items(issue_lines, "Issue", summary.get("issues", []))
    add_section("issues", issue_lines)

    prop = summary.get("property", {}) or {}
    prop_lines: List[str] = []
    add_items(prop_lines, "Asset Pool", prop.get("asset_pool", []))
    add_items(prop_lines, "Contributions", prop.get("contributions", []))
    add_items(prop_lines, "Future Needs", prop.get("future_needs", []))
    add_items(prop_lines, "Just & Equitable", prop.get("just_equitable", []))
    add_section("property_division", prop_lines)

    sm = summary.get("spousal_maintenance", {}) or {}
    sm_lines: List[str] = []
    add_items(sm_lines, "Need", sm.get("need", []))
    add_items(sm_lines, "Capacity to Pay", sm.get("capacity_to_pay", []))
    add_items(sm_lines, "Statutory Factors", sm.get("statutory_factors", []))
    add_section("spousal_maintenance", sm_lines)

    parenting = summary.get("parenting", {}) or {}
    parenting_lines: List[str] = []
    add_items(parenting_lines, "Child Ages", parenting.get("child_ages", []))
    add_items(parenting_lines, "Current Arrangements", parenting.get("current_arrangements", []))
    add_items(parenting_lines, "Safety Concerns", parenting.get("safety_concerns", []))
    add_items(parenting_lines, "Best Interests", parenting.get("best_interests", []))
    add_section("children_parenting", parenting_lines)

    violence = summary.get("family_violence_safety", {}) or {}
    violence_lines: List[str] = []
    add_items(violence_lines, "Incidents", violence.get("incidents", []))
    add_items(violence_lines, "Protection Orders", violence.get("protection_orders", []))
    add_section("family_violence_safety", violence_lines)

    prenup = summary.get("prenup_postnup", {}) or {}
    prenup_lines: List[str] = []
    add_items(prenup_lines, "Agreement Date", prenup.get("agreement_date", []))
    add_items(prenup_lines, "Legal Advice", prenup.get("legal_advice", []))
    add_items(prenup_lines, "Financial Disclosure", prenup.get("financial_disclosure", []))
    add_items(prenup_lines, "Pressure/Duress", prenup.get("pressure_duress", []))
    add_items(prenup_lines, "Changed Circumstances", prenup.get("changed_circumstances", []))
    add_section("prenup_postnup", prenup_lines)

    if include_outcome_reasons:
        outcome_lines: List[str] = []
        add_items(outcome_lines, "Outcome", summary.get("outcome_orders") or [])
        add_section("outcome_orders", outcome_lines)

        reasons_lines: List[str] = []
        add_items(reasons_lines, "Reasons", summary.get("reasons_rationale") or [])
        add_section("reasons_rationale", reasons_lines)

    impact = summary.get("impact_analysis", {}) or {}
    impact_lines: List[str] = []
    add_items(impact_lines, "Pivotal Finding", impact.get("pivotal_findings", []))
    add_items(impact_lines, "Statutory Pivot", impact.get("statutory_pivots", []))
    add_section("impact_analysis", impact_lines)

    uncertainty_lines: List[str] = []
    add_items(uncertainty_lines, "Uncertainties", summary.get("uncertainties", []))
    add_section("uncertainties", uncertainty_lines)

    raw_summary = summary.get("raw_summary")
    if raw_summary:
        add_section("raw_summary", [f"- RawSummary: {raw_summary}"])
    if DEBUG_SUMMARY:
        logger.debug(f"Generated sections from summary JSON: {sections}")
    return sections



def _iter_list_nodes(summary: dict):
    if isinstance(summary, dict):
        for k, v in summary.items():
            if isinstance(v, list):
                yield summary, k, v
            elif isinstance(v, dict):
                yield from _iter_list_nodes(v)


def shrink_to_max_words(summary: dict, max_words: int) -> dict:
    def get_text(s):
        sections = summary_json_to_sections(s)
        return "\n".join(section.get('text', '') for section in sections)

    words = word_count(get_text(summary))
    while words > max_words:
        nodes: List[Tuple[int, dict, str, list]] = [
            (len(lst), parent, key, lst)
            for parent, key, lst in _iter_list_nodes(summary)
            if len(lst) > 1
        ]
        if not nodes:
            break
        nodes.sort(key=lambda x: x[0], reverse=True)
        _, parent, key, lst = nodes[0]
        parent[key] = lst[:-1]
        words = word_count(get_text(summary))
    return summary


def generate_summary_dict(
    case_text: str,
    *,
    target_words: int,
    max_words: int,
    list_limits_primary: dict = SUMMARY_LIST_LIMITS_PRIMARY,
    list_limits_fallback: dict = SUMMARY_LIST_LIMITS_FALLBACK,
    raw_excerpt_chars: int = 2000,
    llm=None,
    case_name: Optional[str] = None,
) -> dict:
    case_text = (case_text or "").strip()
    llm = Settings.llm if llm is None else llm
    prompt = build_case_summary_prompt(
        case_text=case_text,
        target_words=target_words,
        max_words=max_words,
    )
    try:
        response = llm.complete(prompt)
        raw = response.text if hasattr(response, "text") else str(response)
        raw = (raw or "").strip()

        summary = safe_parse_summary_json(raw)

        if "raw_summary" in summary:
            if DEBUG_SUMMARY:
                return empty_case_summary(
                    raw_excerpt=raw[:raw_excerpt_chars],
                    uncertainty="Summary JSON parse failed; raw model output returned.",
                )
            return empty_case_summary(
                raw_excerpt=case_text[:raw_excerpt_chars] if case_text else None,
                uncertainty="Summary JSON parse failed; using raw excerpt.",
            )

        apply_list_limits(summary, list_limits_primary)

        # Check word count and shrink if necessary
        sections = summary_json_to_sections(summary)
        rendered_text = "\n".join(s.get('text', '') for s in sections)
        if word_count(rendered_text) > max_words:
            apply_list_limits(summary, list_limits_fallback)
            shrink_to_max_words(summary, max_words)

        return summary

    except Exception as e:
        # Log failed case to file
        if case_name:
            try:
                with open("failed_summaries.log", "a", encoding="utf-8") as f:
                    f.write(f"{case_name}\n")
            except Exception:
                pass  # Don't fail if logging fails
        
        if DEBUG_SUMMARY:
            err = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            return empty_case_summary(
                raw_excerpt=(f"{err}\n\n{tb}\n\nRAW MODEL OUTPUT:\n{raw[:raw_excerpt_chars]}"
                             if 'raw' in locals() else f"{err}\n\n{tb}"),
                uncertainty="Summary generation failed; debug captured exception.",
            )
        return empty_case_summary(
            raw_excerpt=case_text[:raw_excerpt_chars] if case_text else None,
            uncertainty="Summary generation failed; using raw excerpt.",
        )
