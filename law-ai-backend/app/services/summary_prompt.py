from typing import Any, Dict, Optional


def build_case_summary_prompt(case_text: str, target_words: int, max_words: int) -> str:
    return f"""You are a legal analyst specializing in Australian family law (AustLII decisions).
Read the provided case text and produce a STRICT JSON summary for retrieval.

OUTPUT RULES:
- Output ONLY valid JSON. No markdown, no commentary.
- Use double quotes for all keys and string values.
- If a field is not stated, use an empty list or empty string as specified.
- Do not invent details. Do not quote long passages.
- Do not invent details. Do not quote long passages.
- IMPORTANT (Undecided / Hypothetical uploads): If the text does NOT contain actual court orders or a decided outcome,
  set "outcome_orders" to null. Do NOT provide a "likely outcome" or predicted split as if it were ordered.
- Target length is around {target_words} words when rendered to text; allow up to {max_words} words for complex cases.

TOPICS TO COVER (derive from the case text):
- property_division: Property division (factors: asset_pool, contributions(include domestic/caregiver details), future_needs, living_arrangements, existing_agreements)
- children_parenting: Children custody & parenting (factors: child_ages, current_arrangements, caregiver_history, availability, safety_concerns, child_views)
- spousal_maintenance: Spousal maintenance (factors: income_expenses, earning_capacity, health_care, relationship_length, standard_of_living)
- family_violence_safety: Family violence & safety (factors: incidents, protection_orders, police_court, child_exposure, safety_plan)
- prenup_postnup: Pre/post-nuptial agreement (factors: agreement_date, legal_advice, financial_disclosure, pressure_duress, changed_circumstances)
- impact_analysis: Critical turning points (factors: pivotal_findings, statutory_pivots).

CRITICAL SEARCH INDICATORS (Ensure these factors are addressed in your summary if present):
- Property: Look for assets/liabilities, contributions (financial, non-financial, homemaker, domestic labour, gardening, childcare), future needs (disparity, health, standard of living).
- Parenting: Look for ages, school routines, primary carer history, availability (travel/shifts), safety (abuse/violence), child's wishes.
- Maintenance: Look for budget/expenses, capacity to work, qualifications, relationship duration.
- Agreements: Look for legal advice, financial disclosure, pressure/duress, changed circumstances.

ADDITIONAL REQUIRED DETAILS (when applicable):
- Property cases: what was included/excluded in the asset pool; each party's contributions; each party's future needs; what was just and equitable and the final percentage split.
- Spousal maintenance: the claimant's need, the other party's capacity to pay, and the factors relied on.
- Parenting: allegations of abuse/neglect/family violence and how they were determined; what family consultant/experts recommended; what arrangements were found to be in the child's best interests.
- Impact Analysis: Identify specific behaviors or evidence (pivotal findings) and specific sections of the Family Law Act or Court Rules (statutory pivots) that fundamentally shifted the judge's decision.
- Outcome: what the court ordered to determine the controversy or legal dispute.
- Reasons: the reasons for the decision and orders.

DECIDED vs UNDECIDED RULE:
- If the text contains explicit indicators of a decided judgment (e.g., "Final Orders", "The Court orders", "It is ordered",
  "Judgment", "Reasons for judgment", or a neutral citation like "[YYYY] ... N"), you may populate "outcome_orders".
- Otherwise, set "outcome_orders" to null.

JSON SCHEMA (keys required):
{{{{
  "case_name": "string",
  "court": "string",
  "date": "string",
  "parties": ["string"],
  "issues": ["string"],
  "facts": ["string"],
  "property": {{{{
    "asset_pool": ["string"],
    "contributions": ["string"],
    "future_needs": ["string"],
    "just_equitable": ["string"],
    "living_arrangements": ["string"],
    "existing_agreements": ["string"]
  }}}},
  "spousal_maintenance": {{{{
    "need": ["string"],
    "capacity_to_pay": ["string"],
    "statutory_factors": ["string"],
    "income_expenses": ["string"],
    "earning_capacity": ["string"],
    "health_care": ["string"],
    "relationship_length": ["string"],
    "standard_of_living": ["string"]
  }}}},
  "parenting": {{{{
    "child_ages": ["string"],
    "current_arrangements": ["string"],
    "caregiver_history": ["string"],
    "availability": ["string"],
    "safety_concerns": ["string"],
    "child_views": ["string"],
    "allegations": ["string"],
    "expert_evidence": ["string"],
    "best_interests": ["string"],
    "orders": ["string"]
  }}}},
  "family_violence_safety": {{{{
    "incidents": ["string"],
    "protection_orders": ["string"],
    "police_court": ["string"],
    "child_exposure": ["string"],
    "safety_plan": ["string"]
  }}}},
  "prenup_postnup": {{{{
    "agreement_date": ["string"],
    "legal_advice": ["string"],
    "financial_disclosure": ["string"],
    "pressure_duress": ["string"],
    "changed_circumstances": ["string"]
  }}}},
  "outcome_orders": null,
  "impact_analysis": {{{{
    "description": "Analyze the legal significance of the case. If the case is UNDECIDED (an ongoing matter), pivot from 'findings' to 'thresholds'.",
    "pivotal_findings": [
        "If decided: The key factual determinations made by the judge.",
        "If undecided: The primary factual disputes or 'battlegrounds' that will determine the outcome (e.g., 'Dispute over the valuation of the husband's business interests')."
    ],
    "statutory_pivots": [
        "Identify the specific sections of the Family Law Act (e.g., s 79, s 60CC) that are most critical to this specific case's outcome."
    ]
}}}},
  "reasons_rationale": [
    "Summarize the judge's logic or the primary legal arguments presented by the parties.",
    "Detail how the court balanced competing factors (e.g., weighing s 79(4) contributions against s 75(2) future needs).",
    "Explain the 'why' behind any significant adjustments or deviations from a 50/50 split."
  ],
  "uncertainties": [
    "Identify missing evidence, disputed valuations without expert reports, or conflicting testimony (e.g., 'He-said-she-said' regarding family violence).",
    "Note any procedural complexities, such as pending valuations for trust interests or superannuation splitting issues.",
    "If the case is undecided, list the key evidentiary gaps that prevent a clear prediction of the outcome."
  ]
}}}}

GUIDANCE:
- facts: concise, neutral, abstracted facts (one fact per item).
- issues: legal issues in dispute (one per item).
- property: group assets, use totals or ranges if stated.
- outcome_orders: final orders and outcomes; Must be null if no final determination is present.
- impact_analysis: For undecided cases, focus on thresholds (what must be proven). For decided cases, focus on ratios (why it was decided).
- reasons_rationale: key reasons and credibility findings if mentioned.

INPUT:
{case_text}
"""


def empty_case_summary(raw_excerpt: Optional[str] = None, uncertainty: Optional[str] = None) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "case_name": "",
        "court": "",
        "date": "",
        "parties": [],
        "issues": [],
        "facts": [],
        "property": {
            "asset_pool": [],
            "contributions": [],
            "future_needs": [],
            "just_equitable": [],
            "living_arrangements": [],
            "existing_agreements": [],
        },
        "spousal_maintenance": {
            "need": [],
            "capacity_to_pay": [],
            "statutory_factors": [],
            "income_expenses": [],
            "earning_capacity": [],
            "health_care": [],
            "relationship_length": [],
            "standard_of_living": [],
        },
        "parenting": {
            "child_ages": [],
            "current_arrangements": [],
            "caregiver_history": [],
            "availability": [],
            "safety_concerns": [],
            "child_views": [],
            "allegations": [],
            "expert_evidence": [],
            "best_interests": [],
            "orders": [],
        },
        "family_violence_safety": {
            "incidents": [],
            "protection_orders": [],
            "police_court": [],
            "child_exposure": [],
            "safety_plan": [],
        },
        "prenup_postnup": {
            "agreement_date": [],
            "legal_advice": [],
            "financial_disclosure": [],
            "pressure_duress": [],
            "changed_circumstances": [],
        },
        "outcome_orders": [],
        "impact_analysis": {
            "pivotal_findings": ["string"],
            "statutory_pivots": ["string"]
        },
        "reasons_rationale": [],
        "uncertainties": [],
    }
    if raw_excerpt:
        summary["facts"] = [raw_excerpt]
    if uncertainty:
        summary["uncertainties"] = [uncertainty]
    return summary
