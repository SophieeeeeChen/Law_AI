"""
run_evaluation.py
=================
Runs the SophieAI evaluation pipeline:

1. Loads prepared test cases (from prepare_test_cases.py)
2. For each test case:
   a. Uploads facts as the "user case" (via summary + embedding)
   b. Asks a fixed question per topic
   c. Retrieves top-N precedents and records similarity scores
   d. Optionally calls synthesis LLM and compares prediction vs actual
3. Outputs metrics to eval/results/

Usage:
    # Retrieval-only evaluation (fast, no LLM synthesis cost)
    python -m eval.run_evaluation --mode retrieval

    # Full end-to-end evaluation (slow, costs LLM calls)
    python -m eval.run_evaluation --mode full --limit 10
"""

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.spatial.distance import cosine

from app.core.config import Config
from app.core.models import model_manager
from app.core.logger import logger
from app.services.summary_service import summary_json_to_sections, SUMMARY_LIST_LIMITS_PRIMARY
from app.services.rag_service import (
    _build_structured_query,
    _hybrid_retrieve,
    get_precedent_data,
    calculate_similarity,
)
from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter


# ---------------------------------------------------------------------------
# Fixed evaluation questions per topic
# ---------------------------------------------------------------------------

EVAL_QUESTIONS = {
    "property_division": "What is the likely property split percentage for each party and what orders should be made?",
    "children_parenting": "What parenting orders are likely to be made regarding custody and living arrangements?",
    "spousal_maintenance": "Is spousal maintenance likely to be ordered, and if so, what amount and duration?",
}


# ---------------------------------------------------------------------------
# Outcome extraction helpers
# ---------------------------------------------------------------------------

def extract_property_pct_from_text(text: str) -> float | None:
    """Try to extract a property split percentage from outcome text."""
    if not text:
        return None
    patterns = [
        r"(\d{1,2})[/%]\s*(?:to|for)\s*(?:the\s*)?(?:wife|applicant|mother)",
        r"(?:wife|applicant|mother).*?(\d{1,2})\s*[/%]",
        r"(\d{1,2})\s*[/%]\s*[-/]\s*(\d{1,2})\s*[/%]",
        r"(\d{1,2})\s*per\s*cent",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 30 <= val <= 90:  # sanity check
                return float(val)
    return None


def extract_property_pct_from_prediction(text: str) -> tuple[float | None, float | None]:
    """Extract predicted low/high percentage range from LLM response."""
    if not text:
        return None, None
    # Match patterns like "55% to 62%", "55%–62%", "55-62%"
    m = re.search(r"(\d{1,2})\s*[%]?\s*(?:to|–|-|—)\s*(\d{1,2})\s*%", text, re.IGNORECASE)
    if m:
        low, high = int(m.group(1)), int(m.group(2))
        if 30 <= low <= 90 and 30 <= high <= 90:
            return float(min(low, high)), float(max(low, high))
    return None, None


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

async def evaluate_single_case(
    test_case: dict,
    topic: str,
    summary_index,
    weight_configs: dict,
) -> dict:
    """
    Evaluate retrieval for a single (test_case, topic) pair.
    Returns raw scores and per-config top-3 selections.
    """
    case_name = test_case["case_name"]
    facts_text = test_case["facts_only"]

    # --- A. Get embeddings for the test case facts ---
    # We embed the facts text directly using the same embedding model
    embed_model = Settings.embed_model
    test_topic_vec = embed_model.get_text_embedding(
        test_case.get("facts_only", "")[:3000]
    )
    test_impact_vec = embed_model.get_text_embedding(
        test_case.get("actual_outcome", "")[:1000]  # proxy: use outcome as impact
    )
    # For uncertainty, use a generic empty signal (test cases rarely have explicit uncertainty)
    test_uncert_vec = embed_model.get_text_embedding("No specific uncertainties identified.")

    # --- B. Retrieve top 10 summary nodes ---
    summary_filters = MetadataFilters(
        filters=[ExactMatchFilter(key="summary_section", value=topic)]
    )
    summary_query = _build_structured_query(
        EVAL_QUESTIONS[topic], facts_text[:1000], None, topic=topic
    )
    summary_nodes = _hybrid_retrieve(
        summary_index,
        summary_query,
        use_rerank=False,  # Skip LLM rerank for evaluation speed
        metadata_filters=summary_filters,
        limit=10,
    )

    if not summary_nodes:
        return {
            "case_name": case_name,
            "topic": topic,
            "error": "No summary nodes retrieved",
            "precedent_scores": [],
            "config_results": {},
        }

    # --- C. Compute raw cosine scores for each precedent (done ONCE) ---
    precedent_scores = []
    for s_node in summary_nodes:
        prec_name = s_node.metadata.get("case_name", "unknown")

        # Skip if the retrieved case is the test case itself
        if prec_name == case_name:
            continue

        prec_topic_vec, _ = await get_precedent_data(summary_index, prec_name, topic)
        prec_impact_vec, _ = await get_precedent_data(summary_index, prec_name, "overall_impact_analysis")
        prec_uncert_vec, _ = await get_precedent_data(summary_index, prec_name, "general_credibility_risk")
        _, prec_outcome_text = await get_precedent_data(summary_index, prec_name, "outcome_orders")

        topic_score = calculate_similarity(test_topic_vec, prec_topic_vec) if prec_topic_vec is not None else 0.0
        impact_score = calculate_similarity(test_impact_vec, prec_impact_vec) if prec_impact_vec is not None else 0.0
        uncert_score = calculate_similarity(test_uncert_vec, prec_uncert_vec) if prec_uncert_vec is not None else 0.0

        # Extract outcome percentage from the precedent for relevance labelling
        prec_outcome_pct = extract_property_pct_from_text(prec_outcome_text) if topic == "property_division" else None

        precedent_scores.append({
            "precedent_name": prec_name,
            "topic_score": round(topic_score, 4),
            "impact_score": round(impact_score, 4),
            "uncert_score": round(uncert_score, 4),
            "vector_retrieval_score": round(s_node.score, 4) if s_node.score else None,
            "precedent_outcome_pct": prec_outcome_pct,
        })

    # --- D. Apply each weight config and get top-3 ranking ---
    config_results = {}
    for config_name, weights in weight_configs.items():
        tw, iw, uw = weights["topic_w"], weights["impact_w"], weights["uncert_w"]

        scored = []
        for ps in precedent_scores:
            final = (ps["topic_score"] * tw) + (ps["impact_score"] * iw) + (ps["uncert_score"] * uw)
            scored.append({**ps, "final_score": round(final, 4)})

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        top3 = scored[:3]

        # Metrics for this config
        top3_names = [p["precedent_name"] for p in top3]
        all_finals = [p["final_score"] for p in scored]
        score_spread = (max(all_finals) - min(all_finals)) if all_finals else 0.0

        # Outcome distance (property_division only)
        test_pct = test_case.get("outcome_metrics", {}).get("property_split_pct")
        outcome_distances = []
        if test_pct is not None and topic == "property_division":
            for p in top3:
                if p.get("precedent_outcome_pct") is not None:
                    outcome_distances.append(abs(test_pct - p["precedent_outcome_pct"]))

        config_results[config_name] = {
            "top3_cases": top3_names,
            "top3_scores": [p["final_score"] for p in top3],
            "score_spread": round(score_spread, 4),
            "avg_outcome_distance": round(np.mean(outcome_distances), 2) if outcome_distances else None,
        }

    return {
        "case_name": case_name,
        "topic": topic,
        "actual_outcome_pct": test_case.get("outcome_metrics", {}).get("property_split_pct"),
        "precedent_count": len(precedent_scores),
        "precedent_scores": precedent_scores,
        "config_results": config_results,
    }


async def run_full_synthesis(
    test_case: dict,
    topic: str,
) -> dict:
    """
    Run the full synthesis LLM and compare prediction to actual outcome.
    This is expensive — use sparingly.
    """
    from app.services.rag_service import answer_case_question_withuploadFile, compress_case_facts
    from app.services.summary_service import summary_json_to_sections

    facts_text = test_case["facts_only"]

    # Summarise the facts (same as upload flow)
    summary_json_str = compress_case_facts(facts_text)
    summary_obj = json.loads(summary_json_str)
    sections = summary_json_to_sections(summary_obj)

    case_section_text = sections.get(topic, sections.get("facts", ""))

    # Create in-memory embeddings for this test case
    docs = []
    for section_name, section_text in sections.items():
        if section_text:
            docs.append(Document(
                text=section_text,
                metadata={
                    "case_id": "eval_temp",
                    "case_name": test_case["case_name"],
                    "summary_section": section_name,
                },
            ))

    if docs:
        temp_index = VectorStoreIndex.from_documents(docs)
        model_manager.uploaded_cases_index = temp_index

    # Call the full RAG pipeline
    try:
        answer, statutes = await answer_case_question_withuploadFile(
            question=EVAL_QUESTIONS[topic],
            case_section_text=case_section_text,
            history_text="",
            topic=topic,
            case_id="eval_temp",
        )
    except Exception as e:
        return {
            "case_name": test_case["case_name"],
            "topic": topic,
            "error": str(e),
        }

    # Extract prediction from answer
    pred_low, pred_high = extract_property_pct_from_prediction(answer)
    actual_pct = test_case.get("outcome_metrics", {}).get("property_split_pct")

    # Check if actual falls within predicted range
    in_range = None
    if pred_low is not None and pred_high is not None and actual_pct is not None:
        in_range = pred_low <= actual_pct <= pred_high

    return {
        "case_name": test_case["case_name"],
        "topic": topic,
        "actual_pct": actual_pct,
        "predicted_range": [pred_low, pred_high],
        "in_range": in_range,
        "answer_length": len(answer),
        "answer_preview": answer[:500],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(args):
    # Load test cases
    test_cases_path = Path(args.test_cases)
    if not test_cases_path.exists():
        print(f"❌ Test cases not found: {test_cases_path}")
        print(f"   Run: python -m eval.prepare_test_cases first")
        sys.exit(1)

    test_cases = json.loads(test_cases_path.read_text(encoding="utf-8"))
    print(f"📂 Loaded {len(test_cases)} test cases")

    if args.limit:
        test_cases = test_cases[: args.limit]
        print(f"   (limited to {args.limit})")

    # Initialise models
    print("🔧 Initialising models...")
    model_manager.initialize()
    summary_index = model_manager.create_or_load_case_summaries_index()

    # Weight configs for ablation
    weight_configs = {
        "current":       {"topic_w": 0.70, "impact_w": 0.20, "uncert_w": 0.10},
        "topic_only":    {"topic_w": 1.00, "impact_w": 0.00, "uncert_w": 0.00},
        "impact_only":   {"topic_w": 0.00, "impact_w": 1.00, "uncert_w": 0.00},
        "uncert_only":   {"topic_w": 0.00, "impact_w": 0.00, "uncert_w": 1.00},
        "no_uncert":     {"topic_w": 0.78, "impact_w": 0.22, "uncert_w": 0.00},
        "equal":         {"topic_w": 0.33, "impact_w": 0.33, "uncert_w": 0.34},
        "topic_heavy":   {"topic_w": 0.85, "impact_w": 0.10, "uncert_w": 0.05},
        "impact_heavy":  {"topic_w": 0.40, "impact_w": 0.50, "uncert_w": 0.10},
    }

    # --- Run retrieval evaluation ---
    retrieval_results = []
    synthesis_results = []
    total_evals = 0

    for i, tc in enumerate(test_cases, 1):
        case_name = tc["case_name"]
        topics = tc.get("topics", [])

        for topic in topics:
            if topic not in EVAL_QUESTIONS:
                continue

            total_evals += 1
            print(f"[{total_evals}] {case_name} | {topic}...", end=" ", flush=True)
            t0 = time.time()

            # Retrieval evaluation (always run)
            try:
                result = await evaluate_single_case(tc, topic, summary_index, weight_configs)
                retrieval_results.append(result)
                elapsed = time.time() - t0
                print(f"✅ {result['precedent_count']} precedents ({elapsed:.1f}s)")
            except Exception as e:
                print(f"❌ {e}")
                retrieval_results.append({
                    "case_name": case_name,
                    "topic": topic,
                    "error": str(e),
                })
                continue

            # Full synthesis evaluation (optional, expensive)
            if args.mode == "full" and topic == "property_division":
                print(f"      → Running synthesis...", end=" ", flush=True)
                t1 = time.time()
                try:
                    synth_result = await run_full_synthesis(tc, topic)
                    synthesis_results.append(synth_result)
                    elapsed = time.time() - t1
                    in_range = synth_result.get("in_range")
                    mark = "✅" if in_range else ("❌" if in_range is False else "⚠️")
                    print(f"{mark} range={synth_result.get('predicted_range')} actual={synth_result.get('actual_pct')} ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"❌ Synthesis failed: {e}")

    # --- Save results ---
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    retrieval_path = results_dir / f"retrieval_{timestamp}.json"
    retrieval_path.write_text(
        json.dumps(retrieval_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n📁 Retrieval results saved to {retrieval_path}")

    if synthesis_results:
        synthesis_path = results_dir / f"synthesis_{timestamp}.json"
        synthesis_path.write_text(
            json.dumps(synthesis_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"📁 Synthesis results saved to {synthesis_path}")

    # --- Print summary ---
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total evaluations: {len(retrieval_results)}")

    # Ablation comparison
    print(f"\n--- ABLATION RESULTS (Retrieval) ---")
    print(f"{'Config':<18} | {'Avg Outcome Dist':>16} | {'Avg Score Spread':>16} | {'Top-3 Stability':>15}")
    print(f"{'-'*18}-+-{'-'*16}-+-{'-'*16}-+-{'-'*15}")

    baseline_top3s = []
    for config_name in weight_configs:
        outcome_dists = []
        score_spreads = []
        overlap_with_current = []

        for r in retrieval_results:
            if "error" in r:
                continue
            cr = r.get("config_results", {}).get(config_name, {})
            if cr.get("avg_outcome_distance") is not None:
                outcome_dists.append(cr["avg_outcome_distance"])
            if cr.get("score_spread") is not None:
                score_spreads.append(cr["score_spread"])

            # Compute overlap with "current" config
            if config_name != "current":
                current_top3 = set(
                    r.get("config_results", {}).get("current", {}).get("top3_cases", [])
                )
                this_top3 = set(cr.get("top3_cases", []))
                if current_top3:
                    overlap_with_current.append(len(current_top3 & this_top3))

        avg_dist = f"{np.mean(outcome_dists):.1f}%" if outcome_dists else "N/A"
        avg_spread = f"{np.mean(score_spreads):.4f}" if score_spreads else "N/A"
        avg_overlap = f"{np.mean(overlap_with_current):.1f}/3" if overlap_with_current else "—"

        print(f"{config_name:<18} | {avg_dist:>16} | {avg_spread:>16} | {avg_overlap:>15}")

    # Synthesis summary
    if synthesis_results:
        print(f"\n--- SYNTHESIS RESULTS ---")
        total_synth = len(synthesis_results)
        in_range_count = sum(1 for r in synthesis_results if r.get("in_range") is True)
        out_range_count = sum(1 for r in synthesis_results if r.get("in_range") is False)
        unknown_count = sum(1 for r in synthesis_results if r.get("in_range") is None)
        print(f"Total: {total_synth}")
        print(f"  In range:  {in_range_count} ({in_range_count/total_synth*100:.0f}%)")
        print(f"  Out range: {out_range_count} ({out_range_count/total_synth*100:.0f}%)")
        print(f"  Unknown:   {unknown_count} ({unknown_count/total_synth*100:.0f}%)")

        # Average range width
        widths = []
        for r in synthesis_results:
            rng = r.get("predicted_range", [None, None])
            if rng[0] is not None and rng[1] is not None:
                widths.append(rng[1] - rng[0])
        if widths:
            print(f"  Avg range width: {np.mean(widths):.1f} percentage points")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Run SophieAI evaluation")
    parser.add_argument(
        "--test-cases",
        type=str,
        default="./eval/test_cases.json",
        help="Path to test cases JSON (from prepare_test_cases.py)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="retrieval",
        choices=["retrieval", "full"],
        help="'retrieval' = fast ablation only, 'full' = also run synthesis LLM",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test cases to evaluate",
    )
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
