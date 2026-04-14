"""
analyze_results.py
==================
Reads evaluation results JSON and produces detailed analysis reports.

Usage:
    python -m eval.analyze_results --results eval/results/retrieval_20250101_120000.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def analyze_retrieval(results: list[dict]):
    """Analyze retrieval evaluation results."""

    # Filter out errors
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    print(f"\n{'='*70}")
    print(f"RETRIEVAL ANALYSIS")
    print(f"{'='*70}")
    print(f"Total evaluations: {len(results)}")
    print(f"  Valid: {len(valid)}")
    print(f"  Errors: {len(errors)}")

    if not valid:
        print("No valid results to analyze.")
        return

    # --- Per-topic breakdown ---
    by_topic = defaultdict(list)
    for r in valid:
        by_topic[r["topic"]].append(r)

    print(f"\n--- PER-TOPIC BREAKDOWN ---")
    for topic, topic_results in sorted(by_topic.items()):
        print(f"\n  📌 {topic} ({len(topic_results)} cases)")

        # Analyze each config for this topic
        configs = list(topic_results[0].get("config_results", {}).keys())
        for config_name in configs:
            outcome_dists = []
            score_spreads = []
            for r in topic_results:
                cr = r.get("config_results", {}).get(config_name, {})
                if cr.get("avg_outcome_distance") is not None:
                    outcome_dists.append(cr["avg_outcome_distance"])
                if cr.get("score_spread") is not None:
                    score_spreads.append(cr["score_spread"])

            avg_dist = f"{np.mean(outcome_dists):.1f}%" if outcome_dists else "N/A"
            avg_spread = f"{np.mean(score_spreads):.4f}" if score_spreads else "N/A"
            print(f"     {config_name:<18}: outcome_dist={avg_dist:<8} spread={avg_spread}")

    # --- Score correlation analysis ---
    print(f"\n--- SCORE CORRELATION ANALYSIS ---")
    print(f"  Do higher similarity scores correlate with closer outcomes?")

    topic_scores = []
    impact_scores = []
    uncert_scores = []
    outcome_diffs = []

    for r in valid:
        actual_pct = r.get("actual_outcome_pct")
        if actual_pct is None:
            continue
        for ps in r.get("precedent_scores", []):
            prec_pct = ps.get("precedent_outcome_pct")
            if prec_pct is None:
                continue
            topic_scores.append(ps["topic_score"])
            impact_scores.append(ps["impact_score"])
            uncert_scores.append(ps["uncert_score"])
            outcome_diffs.append(abs(actual_pct - prec_pct))

    if len(topic_scores) >= 10:
        # Simple Pearson correlation
        from scipy.stats import pearsonr

        for name, scores in [
            ("topic_score", topic_scores),
            ("impact_score", impact_scores),
            ("uncert_score", uncert_scores),
        ]:
            corr, pval = pearsonr(scores, outcome_diffs)
            # Negative correlation = higher score → lower outcome diff = GOOD
            direction = "✅ GOOD (higher score → closer outcome)" if corr < 0 else "⚠️ WEAK (higher score → farther outcome)"
            print(f"  {name:<15}: r={corr:+.3f}  p={pval:.4f}  {direction}")
    else:
        print(f"  Not enough data points with outcome percentages ({len(topic_scores)}). "
              f"Need ≥10 for correlation analysis.")

    # --- Top-3 stability across configs ---
    print(f"\n--- CONFIG STABILITY MATRIX ---")
    print(f"  How much do the top-3 precedents change across weight configs?")

    configs = list(valid[0].get("config_results", {}).keys()) if valid else []
    if len(configs) > 1:
        print(f"  {'':>18}", end="")
        for c in configs[:6]:
            print(f"  {c[:8]:>8}", end="")
        print()

        for c1 in configs[:6]:
            print(f"  {c1:<18}", end="")
            for c2 in configs[:6]:
                overlaps = []
                for r in valid:
                    cr1 = r.get("config_results", {}).get(c1, {})
                    cr2 = r.get("config_results", {}).get(c2, {})
                    t1 = set(cr1.get("top3_cases", []))
                    t2 = set(cr2.get("top3_cases", []))
                    if t1 and t2:
                        overlaps.append(len(t1 & t2))
                avg = np.mean(overlaps) if overlaps else 0
                print(f"  {avg:>8.1f}", end="")
            print()

    # --- Recommendations ---
    print(f"\n--- RECOMMENDATIONS ---")

    # Find best config by outcome distance
    best_config = None
    best_dist = float("inf")
    for config_name in configs:
        dists = []
        for r in valid:
            cr = r.get("config_results", {}).get(config_name, {})
            if cr.get("avg_outcome_distance") is not None:
                dists.append(cr["avg_outcome_distance"])
        if dists:
            avg = np.mean(dists)
            if avg < best_dist:
                best_dist = avg
                best_config = config_name

    if best_config:
        print(f"  🏆 Best config by outcome distance: '{best_config}' (avg {best_dist:.1f}%)")
        if best_config != "current":
            print(f"     → Consider switching from 'current' to '{best_config}'")
        else:
            print(f"     → Current weights are already optimal among tested configs")


def analyze_synthesis(results: list[dict]):
    """Analyze synthesis evaluation results."""
    print(f"\n{'='*70}")
    print(f"SYNTHESIS ANALYSIS")
    print(f"{'='*70}")

    valid = [r for r in results if "error" not in r]
    print(f"Total: {len(results)}, Valid: {len(valid)}")

    if not valid:
        return

    in_range = [r for r in valid if r.get("in_range") is True]
    out_range = [r for r in valid if r.get("in_range") is False]
    unknown = [r for r in valid if r.get("in_range") is None]

    print(f"\n  In range:  {len(in_range)}/{len(valid)} ({len(in_range)/len(valid)*100:.0f}%)")
    print(f"  Out range: {len(out_range)}/{len(valid)} ({len(out_range)/len(valid)*100:.0f}%)")
    print(f"  Unknown:   {len(unknown)}/{len(valid)} ({len(unknown)/len(valid)*100:.0f}%)")

    # Detail on misses
    if out_range:
        print(f"\n  --- MISSES ---")
        for r in out_range:
            rng = r.get("predicted_range", [None, None])
            actual = r.get("actual_pct")
            print(f"  {r['case_name']}: predicted {rng[0]}-{rng[1]}%, actual {actual}%")


def main():
    parser = argparse.ArgumentParser(description="Analyze SophieAI evaluation results")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results JSON file",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        sys.exit(1)

    results = json.loads(results_path.read_text(encoding="utf-8"))

    # Detect type from filename
    if "synthesis" in results_path.name:
        analyze_synthesis(results)
    else:
        analyze_retrieval(results)


if __name__ == "__main__":
    main()
