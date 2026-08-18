from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value) -> float:
    """RAGAS trả NaN/None khi 1 metric fail trên 1 câu. NaN không hợp JSON → về 0.0."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return f if f == f else 0.0                        # f != f chỉ đúng với NaN


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation.

    4 metrics: faithfulness, answer_relevancy, context_precision, context_recall.
    RAGAS cần OPENAI_API_KEY (metrics tự gọi LLM để chấm) — thiếu key hoặc lỗi
    network thì trả toàn 0.0 để pipeline vẫn chạy hết và vẫn ghi được report.
    """
    zeros = {"faithfulness": 0.0, "answer_relevancy": 0.0,
             "context_precision": 0.0, "context_recall": 0.0, "per_question": []}
    if not questions:
        return zeros

    try:
        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                   context_precision, context_recall)
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall])
        df = result.to_pandas()

        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                faithfulness=_safe_float(row.get("faithfulness")),
                answer_relevancy=_safe_float(row.get("answer_relevancy")),
                context_precision=_safe_float(row.get("context_precision")),
                context_recall=_safe_float(row.get("context_recall")),
            )
            for _, row in df.iterrows()
        ]

        # Tự tính mean trên giá trị đã lọc NaN thay vì đọc aggregate của RAGAS,
        # để số trong report khớp đúng với per_question ghi ra JSON.
        n = len(per_question) or 1
        return {
            "faithfulness": sum(r.faithfulness for r in per_question) / n,
            "answer_relevancy": sum(r.answer_relevancy for r in per_question) / n,
            "context_precision": sum(r.context_precision for r in per_question) / n,
            "context_recall": sum(r.context_recall for r in per_question) / n,
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {type(e).__name__}: {e}")
        return zeros


# Diagnostic Tree: metric thấp nhất → nguyên nhân gốc → cách sửa
DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM hallucinating — câu trả lời không có trong context",
                     "Tighten prompt, lower temperature, thêm câu 'chỉ dùng context'"),
    "context_recall": ("Missing relevant chunks — retrieval không lấy đủ thông tin",
                       "Improve chunking hoặc tăng BM25/dense top_k"),
    "context_precision": ("Too many irrelevant chunks — nhiễu trong context",
                          "Add reranking hoặc metadata filter"),
    "answer_relevancy": ("Answer không trả lời đúng câu hỏi",
                         "Improve prompt template, yêu cầu trả lời trực tiếp câu hỏi"),
}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree.

    Xếp hạng theo trung bình 4 metrics, lấy bottom_n; mỗi câu lấy metric thấp
    nhất làm điểm vào Diagnostic Tree để suy ra nguyên nhân + cách sửa.
    """
    scored = []
    for r in eval_results:
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=lambda m: metrics[m])
        diagnosis, fix = DIAGNOSTIC_TREE[worst_metric]
        scored.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "avg_score": round(avg, 4),
            "worst_metric": worst_metric,
            "score": round(metrics[worst_metric], 4),
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    scored.sort(key=lambda d: d["avg_score"])
    return scored[:bottom_n]


REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)

    Path không có thư mục → ghi vào reports/ (check_lab.py đọc
    reports/ragas_report.json, còn caller chỉ truyền tên file).
    """
    if not os.path.dirname(path):
        path = os.path.join(REPORTS_DIR, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
