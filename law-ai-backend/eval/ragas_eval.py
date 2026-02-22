from typing import Dict, List

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
) 

from llama_law import answer_query_with_trace_withoutUploadFile


def run_ragas_eval(items: List[Dict[str, str]]):
    """
    items: list of dicts like
      {"question": "...", "ground_truth": ".the expected correct answer}
    """
    rows = []
    for item in items:
        question = item["question"]
        ground_truth = item["ground_truth"]

        answer, retrieved_nodes = answer_query_with_trace_withoutUploadFile(question)
        contexts = [n["content"] for n in retrieved_nodes]

        rows.append(
            {
                "question": question,
                "answer": answer,  #answer from the RAG system
                "contexts": contexts,  #list of retrieved source passages
                "ground_truth": ground_truth, # expected correct answer
            }
        )

    dataset = Dataset.from_list(rows)
    return evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )
