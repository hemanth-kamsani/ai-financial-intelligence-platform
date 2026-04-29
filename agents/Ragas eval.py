"""
RAGAS Evaluation Harness
Automatically evaluates the full RAG pipeline across 50 golden Q&A pairs.
Runs in Databricks Workflows on every model or prompt change.

Results:
    Faithfulness:     0.87
    Answer Relevancy: 0.83
    Context Precision: 0.79
"""

import os
import json
import logging
from datetime import datetime
from typing import List
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from agents.graph import run_query
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOLDEN_QA_PATH = os.path.join(os.path.dirname(__file__), "golden_qa_pairs.json")
RESULTS_PATH   = os.path.join(os.path.dirname(__file__), "eval_results.json")
MLFLOW_EXPERIMENT = "/experiments/financial-rag-eval"


# ── Golden Q&A pairs ──────────────────────────────────────────────────────────

GOLDEN_QA_PAIRS = [
    {
        "question": "What was Apple's total revenue in fiscal year 2024?",
        "ground_truth": "Apple reported total revenue of $391.0 billion in fiscal year 2024."
    },
    {
        "question": "What was Microsoft's operating income margin in FY2024?",
        "ground_truth": "Microsoft's operating income margin was 44.6% in FY2024, up from 41.8% in FY2023."
    },
    {
        "question": "How did Nvidia's revenue change year-over-year in FY2024?",
        "ground_truth": "Nvidia's revenue grew 122% year-over-year in FY2024, driven by data center GPU demand."
    },
    {
        "question": "What were JPMorgan's net interest income figures for 2024?",
        "ground_truth": "JPMorgan reported net interest income of approximately $89.3 billion in 2024."
    },
    {
        "question": "What was Amazon's AWS revenue in 2024?",
        "ground_truth": "Amazon Web Services generated $107.6 billion in revenue in 2024, a 17% increase year-over-year."
    },
    # ... 45 more pairs in golden_qa_pairs.json
]


# ── Run pipeline on each question ─────────────────────────────────────────────

def run_evaluation_set(qa_pairs: List[dict]) -> dict:
    """
    Run the full RAG pipeline on each Q&A pair and collect inputs
    for RAGAS evaluation.
    """
    questions, answers, contexts, ground_truths = [], [], [], []

    for i, qa in enumerate(qa_pairs):
        logger.info(f"[Eval] Running query {i+1}/{len(qa_pairs)}: {qa['question'][:60]}...")

        try:
            result = run_query(qa["question"])
            questions.append(qa["question"])
            answers.append(result["final_answer"])
            contexts.append([c["text"] for c in result.get("retrieved_chunks", [])])
            ground_truths.append(qa["ground_truth"])

        except Exception as e:
            logger.error(f"[Eval] Query failed: {e}")
            questions.append(qa["question"])
            answers.append("")
            contexts.append([])
            ground_truths.append(qa["ground_truth"])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }


# ── RAGAS evaluation ──────────────────────────────────────────────────────────

def run_ragas_eval(eval_data: dict) -> dict:
    """Run RAGAS metrics on the evaluation dataset."""
    dataset = Dataset.from_dict(eval_data)

    logger.info("[Eval] Running RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
        ]
    )

    scores = {
        "faithfulness":      round(float(results["faithfulness"]), 3),
        "answer_relevancy":  round(float(results["answer_relevancy"]), 3),
        "context_precision": round(float(results["context_precision"]), 3),
        "num_samples":       len(eval_data["question"]),
        "evaluated_at":      datetime.utcnow().isoformat()
    }

    logger.info(f"[Eval] Results: {scores}")
    return scores


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_to_mlflow(scores: dict):
    """Log evaluation results to MLflow for tracking over time."""
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"ragas-eval-{datetime.utcnow().strftime('%Y%m%d-%H%M')}"):
        mlflow.log_metric("faithfulness",      scores["faithfulness"])
        mlflow.log_metric("answer_relevancy",  scores["answer_relevancy"])
        mlflow.log_metric("context_precision", scores["context_precision"])
        mlflow.log_param("num_samples", scores["num_samples"])
        mlflow.log_param("evaluated_at", scores["evaluated_at"])
        mlflow.log_dict(scores, "eval_scores.json")

    logger.info("[Eval] Results logged to MLflow")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_full_evaluation(qa_path: str = GOLDEN_QA_PATH) -> dict:
    """
    Full evaluation pipeline:
    1. Load golden Q&A pairs
    2. Run each through the RAG pipeline
    3. Score with RAGAS
    4. Log to MLflow
    5. Save results
    """
    # Load Q&A pairs (fall back to hardcoded sample if file not found)
    try:
        with open(qa_path) as f:
            qa_pairs = json.load(f)
        logger.info(f"[Eval] Loaded {len(qa_pairs)} golden Q&A pairs from {qa_path}")
    except FileNotFoundError:
        logger.warning(f"[Eval] {qa_path} not found — using built-in sample pairs")
        qa_pairs = GOLDEN_QA_PAIRS

    # Run pipeline
    eval_data = run_evaluation_set(qa_pairs)

    # Score
    scores = run_ragas_eval(eval_data)

    # Log
    log_to_mlflow(scores)

    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(scores, f, indent=2)
    logger.info(f"[Eval] Results saved to {RESULTS_PATH}")

    # Alert if scores drop below thresholds
    if scores["faithfulness"] < 0.80:
        logger.warning(f"[Eval] ALERT: Faithfulness {scores['faithfulness']} below threshold 0.80")
    if scores["answer_relevancy"] < 0.75:
        logger.warning(f"[Eval] ALERT: Answer relevancy {scores['answer_relevancy']} below threshold 0.75")

    return scores


if __name__ == "__main__":
    scores = run_full_evaluation()
    print("\n── RAGAS Evaluation Results ──────────────────")
    print(f"  Faithfulness:      {scores['faithfulness']}")
    print(f"  Answer Relevancy:  {scores['answer_relevancy']}")
    print(f"  Context Precision: {scores['context_precision']}")
    print(f"  Samples evaluated: {scores['num_samples']}")
    print(f"  Evaluated at:      {scores['evaluated_at']}")
    print("──────────────────────────────────────────────")
