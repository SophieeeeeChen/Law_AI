# SophieAI Evaluation Framework

## Overview

This framework evaluates SophieAI's retrieval quality and prediction accuracy
by testing against decided AustLII cases with known outcomes.

## Quick Start

### Step 1: Prepare test cases

Place decided case `.txt` files in `eval_cases_txt/` directory.
**Important:** These cases must NOT already be in your RAG database (to avoid data leakage).

```bash
# Create directory and add case files
mkdir eval_cases_txt
# Copy .txt files of decided cases into this directory

# Run preparation (uses LLM to split facts from outcomes)
python -m eval.prepare_test_cases --cases-dir ./eval_cases_txt --output ./eval/test_cases.json --limit 50
```

### Step 2: Run retrieval evaluation (fast, free)

This tests different weight configurations (ablation study) without calling the synthesis LLM.

```bash
python -m eval.run_evaluation --mode retrieval --test-cases ./eval/test_cases.json
```

### Step 3: Run full evaluation (slow, costs LLM calls)

This also runs the synthesis LLM and compares predicted ranges against actual outcomes.

```bash
python -m eval.run_evaluation --mode full --test-cases ./eval/test_cases.json --limit 10
```

### Step 4: Analyze results

```bash
python -m eval.analyze_results --results eval/results/retrieval_YYYYMMDD_HHMMSS.json
```

## File Structure

```
eval/
├── __init__.py
├── README.md
├── prepare_test_cases.py    # Step 1: Split decided cases → facts + outcome
├── run_evaluation.py        # Step 2/3: Run retrieval ablation + optional synthesis
├── analyze_results.py       # Step 4: Detailed analysis and recommendations
├── test_cases.json          # Generated: prepared test cases
└── results/                 # Generated: evaluation outputs
    ├── retrieval_YYYYMMDD_HHMMSS.json
    └── synthesis_YYYYMMDD_HHMMSS.json
```

## What it measures

### Retrieval evaluation (mode=retrieval)
- **Outcome distance**: Are retrieved precedents' outcomes close to the test case's actual outcome?
- **Score spread**: Do the weights produce meaningfully different scores across precedents?
- **Top-3 stability**: How much do the selected precedents change across weight configs?
- **Score correlation**: Does higher similarity actually predict closer outcomes?

### Synthesis evaluation (mode=full)
- **Range accuracy**: Does the actual outcome fall within the predicted percentage range?
- **Range width**: How narrow are the predictions? (narrower = more confident)
- **Direction accuracy**: Did it predict the right party gets the larger share?

## Weight configs tested (ablation)

| Config | topic_w | impact_w | uncert_w | Hypothesis |
|--------|---------|----------|----------|------------|
| current | 0.70 | 0.20 | 0.10 | Baseline |
| topic_only | 1.00 | 0.00 | 0.00 | Is topic similarity alone sufficient? |
| impact_only | 0.00 | 1.00 | 0.00 | Is impact similarity alone sufficient? |
| uncert_only | 0.00 | 0.00 | 1.00 | Is uncertainty similarity useful at all? |
| no_uncert | 0.78 | 0.22 | 0.00 | Does removing uncertainty improve results? |
| equal | 0.33 | 0.33 | 0.34 | No assumptions about signal importance |
| topic_heavy | 0.85 | 0.10 | 0.05 | Topic matters even more? |
| impact_heavy | 0.40 | 0.50 | 0.10 | Impact matters more than topic? |
