from typing import Any, Dict, Optional


def build_case_summary_prompt(case_text: str, target_words: int, max_words: int) -> str:
    return f"""You are a legal analyst specializing in Australian family law (AustLII decisions).
Read the provided case text and produce a STRICT JSON summary for retrieval.

OUTPUT RULES:
- Output ONLY valid JSON. No markdown, no commentary.
- Use double quotes for all keys and string values.
- If a field is not stated, use an empty list or empty string as specified.
- Do not invent details. Do not quote long passages.
- IMPORTANT (Undecided / Hypothetical uploads): If the text does NOT contain actual court orders or a decided outcome, set "outcome_orders" to null and leave every section-level "orders" as an empty list.
- Target length is around {target_words} words; allow up to {max_words} words for complex cases.
- RETAIN ENTITIES (CRITICAL): Ensure every section includes specific dollar amounts, ages, dates, and names of experts/consultants mentioned in the text. Never generalize "the house" if the text specifies "the $1.2M property in Richmond."
- SECTION-LEVEL FIELDS: Each topical section contains its own "orders", "reasoning", "impact_analysis", and "evidentiary_gaps".
--Your final output must be ONLY the JSON block. Do not include your thinking process in the final code block.
--CRITICAL: If a section (e.g., spousal_maintenance) or a specific field is not mentioned or relevant to the judgment, you MUST return an empty list [] or an empty string "" for that field. Strictly DO NOT include filler text like 'Not addressed', 'None', or 'N/A'.
--CRITICAL: HOWEVER, the following 'Analysis Sections' are MANDATORY and must never be empty:
'overall_impact_analysis': You must synthesize the cross-cutting legal significance.
'reasons_rationale': You must provide the overarching judicial logic.
'general_credibility_risk': You must assess the reliability of the evidence/witnesses.
Even if the judgment is short, you must provide your expert synthesis in these fields based on the available text."

TOPICS TO COVER (derive from the case text):
- property_division: Property division (factors: asset_pool, contributions(include domestic/caregiver details), future_needs, living_arrangements, existing_agreements)
- children_parenting: Children custody & parenting (factors: child_ages, current_arrangements, caregiver_history, availability, safety_concerns, child_views)
- spousal_maintenance: Spousal maintenance (factors: income_expenses, earning_capacity, health_care, relationship_length, standard_of_living)
- family_violence_safety: Family violence & safety (factors: incidents, protection_orders, police_court, child_exposure, safety_plan)
- prenup_postnup: Pre/post-nuptial agreement (factors: agreement_date, legal_advice, financial_disclosure, pressure_duress, changed_circumstances)

CRITICAL SEARCH INDICATORS (Ensure these factors are addressed in your summary if present):
- Property: Look for assets/liabilities, contributions (financial, non-financial, homemaker, domestic labour, gardening, childcare), future needs (disparity, health, standard of living). Strictly distinguish between 'Agreed' values and 'Disputed' values. If a total net pool figure is mentioned (e.g., $1.2M), prioritize including it.
- Parenting: Look for ages, school routines, primary carer history, availability (travel/shifts), safety (abuse/violence), child's wishes.
- Parenting (TACTICAL & TRAVEL): Look for instances of strategic withholding of a child (fabricated illness, refusal to return after contact), unilateral travel bookings, passport disputes, contravention applications (Div 13A), risk-of-non-return allegations, Hague Convention references, Airport Watch List orders (s 67ZD), and any judicial commentary on a party's credibility being undermined by tactical behavior. Note: (1) HOW the court characterised the behavior (e.g., 'reasonable caution' vs 'deliberate frustration of the other parent's time' vs 'tactical withholding designed to create a status quo'), (2) WHAT consequences followed (e.g., costs, variation of orders, supervised handovers, makeup time, change of primary residence), and (3) any judicial GUIDANCE or WARNING about this type of conduct that a lawyer could cite in future cases.
- Maintenance: Look for budget/expenses, capacity to work, qualifications, relationship duration.
- Agreements: Look for legal advice, financial disclosure, pressure/duress, changed circumstances.

ADDITIONAL REQUIRED DETAILS (when applicable):
- Property cases: what was included/excluded in the asset pool (clearly labeling 'Agreed' vs 'Disputed' figures and the total net pool if stated); each party's contributions; each party's future needs; what was just and equitable and the final percentage split.
- Spousal maintenance: the claimant's need, the other party's capacity to pay, and the factors relied on.
- Parenting: allegations of abuse/neglect/family violence and how they were determined; what family consultant/experts recommended; what arrangements were found to be in the child's best interests.
- Parenting (tactical patterns): If a party engaged in grey-area conduct (e.g., withholding child citing illness without medical evidence, retaining child after contact to block a planned trip, unilateral travel decisions), capture: (a) what the party did, (b) their stated justification, (c) the court's finding on credibility, (d) the JUDGE'S CHARACTERISATION — how the judge described or labelled the behavior (e.g., 'deliberate frustration', 'tactical withholding to create a status quo', 'reasonable caution'), (e) the legal consequence (contravention finding, costs, order variation, makeup time, change of primary residence), (f) the statutory basis (e.g., Div 13A contravention, s 65Y passports, s 67ZD Airport Watch List, s 60CC(3)(c) willingness to facilitate relationship), and (g) any judicial GUIDANCE or WARNING about this type of conduct that could be cited as precedent.
- Impact Analysis (per section): Identify specific behaviors or evidence (pivotal findings) and specific sections of the Family Law Act or Court Rules (statutory pivots) that fundamentally shifted the judge's decision on THAT topic.
- Evidentiary Gaps (per section): Identify missing evidence, disputed valuations without expert reports, or conflicting testimony specific to THAT topic.
- Outcome: what the court ordered to determine the controversy or legal dispute.
- Reasons: the reasons for the decision and orders.

DECIDED vs UNDECIDED RULE:
- If the text contains explicit indicators of a decided judgment (e.g., "Final Orders", "The Court orders", "It is ordered",
  "Judgment", "Reasons for judgment", or a neutral citation like "[YYYY] ... N"), you may populate "outcome_orders" and section-level "orders".
- Otherwise, set "outcome_orders" to null and leave all section-level "orders" as empty lists.

JSON SCHEMA (keys required):
{{{{
  "case_name": "string",
  "parties": ["string"],
  "facts": ["string"],
  "property_division": {{{{
    "asset_pool": [
      "AGREED: [Description] - [Value]",
      "DISPUTED: [Description] - [Claimed Value vs Counter-Value]",
      "TOTAL NET POOL: [Total Amount]"
    ],
    "contributions": ["string"],
    "future_needs": ["string"],
    "just_equitable": ["string"],
    "living_arrangements": ["string"],   
    "existing_agreements": ["string"],  
    "orders": ["The specific property orders made by the court (e.g., 'Wife to retain the former matrimonial home', 'Superannuation splitting order of $Y')."],
    "reasoning": ["The judge's reasoning for the property division (e.g., 'Equal contributions found but 5% adjustment under s 75(2) for wife's reduced earning capacity')."],
    "impact_analysis": {{{{
      "pivotal_findings": ["Key factual determinations that drove the property outcome (e.g., 'Husband's pre-relationship inheritance of $300k was quarantined from the pool')."],
      "statutory_pivots": ["The specific statutory provisions most critical to the property outcome (e.g., 's 79(4) contributions', 's 75(2) future needs adjustment')."]
    }}}},
    "evidentiary_gaps": ["Missing or contested evidence specific to property (e.g., 'No independent valuation of the husband's business', 'Disputed value of superannuation interests')."]
  }}}},
  "children_parenting": {{{{
    "child_ages": ["string"],
    "current_arrangements": ["string"],   
    "caregiver_history": ["string"],    
    "availability": ["string"],         
    "safety_concerns": ["string"],      
    "child_views": ["string"],          
    "allegations": ["string"],          
    "expert_evidence": ["string"],      
    "best_interests": ["string"],       
    "tactical_behavior_patterns": ["Instances where a party engaged in strategic, deceptive, or 'grey area' conduct regarding parenting arrangements (e.g., 'Mother withheld child from weekend handover citing illness with no medical evidence', 'Father retained child after holiday contact and refused to return', 'Mother unilaterally booked international travel during father's scheduled time', 'Father's refusal to consent to travel found to be unreasonable gatekeeping'). For EACH instance, capture: (a) the tactic used, (b) the stated justification, (c) the court's finding on credibility, (d) the JUDGE'S CHARACTERISATION — how the judge described or labelled the behavior (e.g., 'deliberate frustration of the father's time', 'reasonable caution given history of violence', 'tactical withholding designed to create a status quo', 'not a genuine safety concern but an attempt to control'), (e) the legal consequence (contravention finding, costs, order variation, makeup time, change of primary residence), and (f) any judicial guidance or warning about this type of conduct. Also capture travel/relocation disputes, passport retention, Hague Convention considerations, and Airport Watch List (s 67ZD) orders where relevant."],
    "orders": ["The specific parenting orders (e.g., 'Children to live with mother', 'Supervised time for 6 months')."],
    "reasoning": ["The judge's reasoning for the parenting orders (e.g., 'Court preferred family consultant's recommendation that stability with primary carer was in children's best interests')."],
    "impact_analysis": {{{{
      "pivotal_findings": ["Key factual determinations that drove the parenting outcome (e.g., 'Father's work roster made weekday care impractical', 'Family consultant found children anxious after overnight stays')."],
      "statutory_pivots": ["The specific statutory provisions most critical to the parenting outcome (e.g., 's 60CC(2) primary considerations', 's 60CC(3) additional considerations', 's 61DA presumption of equal shared parental responsibility')."]
    }}}},
    "evidentiary_gaps": ["Missing or contested evidence specific to parenting (e.g., 'No updated family report', 'Conflicting accounts of the alleged incident on [date]')."]
  }}}},
  "spousal_maintenance": {{{{
    "need": ["string"],
    "capacity_to_pay": ["string"],   
    "statutory_factors": ["string"],   
    "income_expenses": ["string"],    
    "earning_capacity": ["string"],   
    "health_care": ["string"],        
    "relationship_length": ["string"],
    "standard_of_living": ["string"], 
    "orders": ["The specific maintenance orders (e.g., 'Husband to pay $X/month for 3 years', 'Maintenance dismissed')."],
    "reasoning": ["The judge's reasoning for the maintenance decision (e.g., 'Wife demonstrated need due to limited earning capacity after 20-year marriage as homemaker')."],
    "impact_analysis": {{{{
      "pivotal_findings": ["Key factual determinations that drove the maintenance outcome (e.g., 'Wife's medical evidence established inability to work full-time')."],
      "statutory_pivots": ["The specific statutory provisions most critical to maintenance (e.g., 's 72 duty to maintain', 's 75(2) factors')."]
    }}}},
    "evidentiary_gaps": ["Missing or contested evidence specific to maintenance (e.g., 'No medical report supporting claimed inability to work', 'Husband's true income unclear due to cash business')."]
  }}}},
  "family_violence_safety": {{{{
    "incidents": ["string"],
    "protection_orders": ["string"],
    "police_court": ["string"],   
    "child_exposure": ["string"],
    "safety_plan": ["string"],   
    "orders": ["Any specific orders related to family violence or safety (e.g., 'Injunction under s 68B', 'Supervised handovers at contact centre')."],
    "reasoning": ["The judge's reasoning regarding family violence findings (e.g., 'Court accepted mother's evidence of coercive control corroborated by police records')."],
    "impact_analysis": {{{{
      "pivotal_findings": ["Key factual determinations regarding violence (e.g., 'Text messages corroborated pattern of threats', 'Children disclosed witnessing physical violence to family consultant')."],
      "statutory_pivots": ["The specific statutory provisions most critical to the violence findings (e.g., 's 4AB definition of family violence', 's 60CC(2)(b) protection from harm', 's 60CG obligations regarding family violence')."]
    }}}},
    "evidentiary_gaps": ["Missing or contested evidence specific to violence (e.g., 'He-said-she-said with no independent witnesses', 'No police reports despite alleged incidents')."]
  }}}},
  "prenup_postnup": {{{{
    "agreement_date": ["string"],
    "legal_advice": ["string"],   
    "financial_disclosure": ["string"],  
    "pressure_duress": ["string"],     
    "changed_circumstances": ["string"],   
    "orders": ["Any specific orders regarding the agreement (e.g., 'BFA set aside under s 90K(1)(b)', 'Agreement upheld and enforced')."],
    "reasoning": ["The judge's reasoning regarding the agreement (e.g., 'Agreement set aside because wife did not receive independent legal advice and material non-disclosure of trust assets')."],
    "impact_analysis": {{{{
      "pivotal_findings": ["Key factual determinations regarding the agreement (e.g., 'Solicitor's certificate was deficient', 'Husband failed to disclose $500k in trust assets')."],
      "statutory_pivots": ["The specific statutory provisions most critical to the agreement (e.g., 's 90G requirements for BFAs', 's 90K grounds for setting aside')."]
    }}}},
    "evidentiary_gaps": ["Missing or contested evidence specific to the agreement (e.g., 'Disputed whether wife was given adequate time to seek legal advice', 'No independent evidence of alleged duress')."]
  }}}},
  "outcome_orders": "list[string] | null  —  Overall / cross-cutting orders not specific to a single section (e.g., costs, procedural directions, liberty to apply).",
  "overall_impact_analysis": {{{{
    "description": "Overarching legal significance of the case spanning multiple topics. If UNDECIDED, pivot from 'findings' to 'thresholds'.",
    "pivotal_findings": [
        "Cross-cutting factual determinations that shaped the overall outcome across multiple topics.",
        "If undecided: The primary factual disputes or 'battlegrounds' that will determine the outcome."
    ],
    "statutory_pivots": [
        "Cross-cutting statutory provisions critical to the overall case outcome (e.g., interplay between s 79 property and s 75(2) maintenance factors)."
    ]
  }}}},
  "reasons_rationale": [
    "Overall / cross-cutting reasoning that spans multiple topics.",
    "Summarize the judge's overarching logic or the primary legal arguments presented.",
    "Detail how the court balanced competing factors across topics (e.g., interplay between property and maintenance).",
    "Explain the 'why' behind any significant adjustments or deviations from a 50/50 split."
  ],
  "general_credibility_risk": [
    "Overall credibility assessments: which party's evidence was preferred and why (e.g., 'The court found the wife to be a generally reliable witness; the husband's evidence was evasive on financial matters').",
    "Overarching risk factors spanning multiple issues (e.g., 'Pattern of non-disclosure undermined the husband's credibility on both property and maintenance issues').",
    "General procedural concerns (e.g., 'Delay in bringing the application may affect interim arrangements').",
    "If undecided: overall assessment of which party bears the greater evidentiary burden or litigation risk."
  ]
}}}}

GUIDANCE:
- facts: concise, neutral, abstracted facts (one fact per item).
- property: group assets, use totals or ranges if stated.
- Section-level "orders": the specific orders the judge made for THAT topic. Must be empty list if undecided.
- Section-level "reasoning": the judge's reasoning for THAT topic. For undecided cases, capture each party's key arguments.
- Section-level "impact_analysis": pivotal findings and statutory pivots specific to THAT topic only.
- Section-level "evidentiary_gaps": missing/disputed evidence specific to THAT topic only.
- outcome_orders: overall / residual orders that span multiple sections; Must be null if no final determination is present.
- overall_impact_analysis: cross-cutting turning points spanning multiple sections.
- reasons_rationale: overall / cross-cutting reasoning spanning multiple sections.
- general_credibility_risk: holistic credibility assessments, overarching risk factors, and procedural concerns that do not belong to a single section.
INPUT:
{case_text}
"""
