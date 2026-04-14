"""
prepare_test_cases.py
=====================
Reads decided AustLII case files, summarises each using the production
summary prompt (build_case_summary_prompt), and saves them as structured
JSON test cases for the evaluation framework.

The output format mirrors the production summary schema so that embeddings,
section lookups, and scoring are fully consistent between eval and prod.

Usage:
    python -m eval.prepare_test_cases \
        --cases-dir ./eval_cases_txt \
        --output ./eval/test_cases.json \
        --limit 50

The input directory should contain .txt files of decided cases that are
**NOT** in the RAG database (to avoid data leakage).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Add project root to path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llama_index.core import Settings
from app.core.config import Config
from app.core.models import model_manager
from app.services.summary_service import (
    generate_summary_dict,
    SUMMARY_LIST_LIMITS_PRIMARY,
)


# ---------------------------------------------------------------------------
# Testable topics — must match the section keys used in rag_service.py
# ---------------------------------------------------------------------------

TESTABLE_TOPICS = {"property_division", "children_parenting", "spousal_maintenance"}


def _section_has_content(section_data) -> bool:
    """Return True if a summary section dict has meaningful content."""
    if not section_data or not isinstance(section_data, dict):
        return False
    for key, val in section_data.items():
        if key in ("orders", "reasoning", "impact_analysis", "evidentiary_gaps"):
            continue  # skip meta-fields for content check
        if isinstance(val, list) and len(val) > 0:
            return True
        if isinstance(val, str) and len(val.strip()) > 0:
            return True
    return False


def _extract_property_pct(summary: dict) -> float | None:
    """Try to extract property split % from the summary JSON."""
    prop = summary.get("property_division", {})
    # Check just_equitable and orders for percentage mentions
    search_texts = []
    for key in ("just_equitable", "orders", "reasoning"):
        val = prop.get(key, [])
        if isinstance(val, list):
            search_texts.extend(val)
        elif isinstance(val, str):
            search_texts.append(val)

    # Also check outcome_orders
    outcome = summary.get("outcome_orders")
    if isinstance(outcome, list):
        search_texts.extend(outcome)

    combined = " ".join(search_texts).lower()
    # Match patterns like "60/40", "60%", "60 per cent"
    patterns = [
        r"(\d{1,2})\s*/\s*(\d{1,2})",             # 60/40
        r"(\d{1,2})\s*%\s*(?:to|for|in favour)",   # 60% to wife
        r"(\d{1,2})\s*per\s*cent",                  # 60 per cent
    ]
    for p in patterns:
        m = re.search(p, combined)
        if m:
            val = int(m.group(1))
            if 30 <= val <= 90:
                return float(val)
    return None


def _extract_parenting_arrangement(summary: dict) -> str | None:
    """Try to classify parenting arrangement from summary."""
    cp = summary.get("children_parenting", {})
    orders = cp.get("orders", [])
    if not orders:
        return None

    combined = " ".join(orders).lower()
    if "equal" in combined and "shared" in combined:
        return "equal_shared"
    if any(kw in combined for kw in ("live with mother", "live with the mother", "primary care of mother", "primary carer mother")):
        return "primary_mother"
    if any(kw in combined for kw in ("live with father", "live with the father", "primary care of father", "primary carer father")):
        return "primary_father"
    if "supervis" in combined:
        return "supervised"
    return "other"


def _extract_maintenance_awarded(summary: dict) -> bool | None:
    """Determine if maintenance was awarded from summary."""
    sm = summary.get("spousal_maintenance", {})
    orders = sm.get("orders", [])
    if not orders:
        return None

    combined = " ".join(orders).lower()
    if any(kw in combined for kw in ("dismissed", "refused", "no order", "not awarded")):
        return False
    if any(kw in combined for kw in ("pay", "per month", "per week", "per annum", "lump sum")):
        return True
    return None


def _build_facts_only(summary: dict) -> str:
    """
    Reconstruct a facts-only text from the summary JSON.
    This strips out orders, reasoning, and outcome so the test case
    looks like an undecided client scenario.
    """
    parts = []

    # General facts
    facts = summary.get("facts", [])
    if facts:
        parts.append("FACTS:\n" + "\n".join(f"- {f}" for f in facts))

    # Per-section factual content (exclude orders/reasoning/impact/gaps)
    fact_fields = {
        "property_division": ["asset_pool", "contributions", "future_needs", "living_arrangements", "existing_agreements"],
        "children_parenting": ["child_ages", "current_arrangements", "caregiver_history", "availability", "safety_concerns", "child_views", "allegations", "expert_evidence", "tactical_behavior_patterns"],
        "spousal_maintenance": ["need", "capacity_to_pay", "income_expenses", "earning_capacity", "health_care", "relationship_length", "standard_of_living"],
        "family_violence_safety": ["incidents", "protection_orders", "police_court", "child_exposure", "safety_plan"],
    }

    for section, fields in fact_fields.items():
        section_data = summary.get(section, {})
        if not isinstance(section_data, dict):
            continue
        section_lines = []
        for field in fields:
            val = section_data.get(field, [])
            if isinstance(val, list) and val:
                section_lines.extend(f"- {v}" for v in val if v)
            elif isinstance(val, str) and val.strip():
                section_lines.append(f"- {val}")
        if section_lines:
            label = section.replace("_", " ").title()
            parts.append(f"\n{label}:\n" + "\n".join(section_lines))

    return "\n".join(parts)


def _build_actual_outcome(summary: dict) -> str:
    """
    Reconstruct the actual outcome from orders, reasoning, and impact analysis.
    """
    parts = []

    # Overall outcome orders
    outcome = summary.get("outcome_orders")
    if isinstance(outcome, list) and outcome:
        parts.append("ORDERS:\n" + "\n".join(f"- {o}" for o in outcome))

    # Per-section orders and reasoning
    for section in ("property_division", "children_parenting", "spousal_maintenance", "family_violence_safety", "prenup_postnup"):
        section_data = summary.get(section, {})
        if not isinstance(section_data, dict):
            continue
        orders = section_data.get("orders", [])
        reasoning = section_data.get("reasoning", [])
        if orders or reasoning:
            label = section.replace("_", " ").title()
            if orders:
                parts.append(f"\n{label} Orders:\n" + "\n".join(f"- {o}" for o in orders))
            if reasoning:
                parts.append(f"\n{label} Reasoning:\n" + "\n".join(f"- {r}" for r in reasoning))

    # Overall reasoning
    reasons = summary.get("reasons_rationale", [])
    if reasons:
        parts.append("\nOverall Reasoning:\n" + "\n".join(f"- {r}" for r in reasons))

    return "\n".join(parts)


def summarise_case_with_llm(case_text: str, case_name: str, llm) -> dict | None:
    """
    Use the production generate_summary_dict to summarise a decided case,
    then split it into facts_only + actual_outcome for evaluation.

    This reuses the exact same prompt, JSON parsing, list-limit logic,
    and cost tracking as the production pipeline.
    """
    try:
        summary = generate_summary_dict(
            case_text[:15000],
            path_stem=case_name,
            list_limits_primary=SUMMARY_LIST_LIMITS_PRIMARY,
            llm=llm,
            case_name=case_name,
        )

        if not summary:
            return None

        summary["case_name"] = case_name

        # Derive which topics have substantive content
        topics = []
        for t in TESTABLE_TOPICS:
            if _section_has_content(summary.get(t)):
                topics.append(t)

        # Build the evaluation test case
        result = {
            "case_name": case_name,
            "facts_only": _build_facts_only(summary),
            "actual_outcome": _build_actual_outcome(summary),
            "topics": topics,
            "outcome_metrics": {
                "property_split_pct": _extract_property_pct(summary),
                "parenting_arrangement": _extract_parenting_arrangement(summary) if "children_parenting" in topics else None,
                "maintenance_awarded": _extract_maintenance_awarded(summary) if "spousal_maintenance" in topics else None,
            },
            # Keep the full summary for deeper analysis / embedding consistency
            "full_summary": summary,
        }
        return result

    except Exception as e:
        error_msg = str(e).lower()
        if any(kw in error_msg for kw in ("safety", "blocked", "no candidates", "google", "gemini")):
            print(f"  ⚠️  LLM blocked for {case_name}: {e}")
            return None
        raise


def main():
    parser = argparse.ArgumentParser(description="Prepare test cases from decided AustLII cases")
    parser.add_argument(
        "--cases-dir",
        type=str,
        default="./eval_cases_txt",
        help="Directory containing decided case .txt files (NOT in RAG DB)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./eval/test_cases.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max number of cases to process",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="openai",
        choices=["openai", "gemini"],
        help="LLM to use for case splitting",
    )
    args = parser.parse_args()

    cases_dir = Path(args.cases_dir)
    if not cases_dir.exists():
        print(f"❌ Cases directory not found: {cases_dir}")
        print(f"   Create it and add .txt files of decided cases NOT in your RAG database.")
        sys.exit(1)

    case_files = sorted(cases_dir.glob("*.txt"))
    if not case_files:
        print(f"❌ No .txt files found in {cases_dir}")
        sys.exit(1)

    case_files = case_files[: args.limit]
    print(f"📂 Found {len(case_files)} case files in {cases_dir}")

    # Initialise LLM
    model_manager.initialize()

    if args.llm == "openai":
        from llama_index.llms.openai import OpenAI as OpenAILLM

        llm = OpenAILLM(
            model=Config.SYNTHESIS_OPENAI_LLM_MODEL,
            temperature=0.0,
            timeout=300.0,
            max_retries=2,
        )
    else:
        llm = Settings.llm

    print(f"🤖 Using LLM: {args.llm}")

    # Process each case
    test_cases = []
    for i, case_file in enumerate(case_files, 1):
        case_name = case_file.stem
        print(f"[{i}/{len(case_files)}] Processing {case_name}...", end=" ")

        case_text = case_file.read_text(encoding="utf-8", errors="replace")
        if len(case_text) < 500:
            print("⏭️  Too short, skipping")
            continue

        result = summarise_case_with_llm(case_text, case_name, llm)
        if result is None:
            print("❌ Failed")
            continue

        # Validate required fields
        if not result.get("facts_only") or not result.get("actual_outcome"):
            print("⏭️  Missing facts or outcome")
            continue

        if not result.get("topics"):
            print("⏭️  No testable topics")
            continue

        test_cases.append(result)
        print(f"✅ topics={result['topics']}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(test_cases, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"✅ Saved {len(test_cases)} test cases to {output_path}")

    # Summary stats
    topic_counts = {}
    for tc in test_cases:
        for t in tc.get("topics", []):
            topic_counts[t] = topic_counts.get(t, 0) + 1
    print(f"📊 Topic distribution:")
    for topic, count in sorted(topic_counts.items()):
        print(f"   {topic}: {count}")


if __name__ == "__main__":
    main()
