from __future__ import annotations

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from .gpt_scorer import GPTScorer


def calculate_bleu(reference: str, candidate: str) -> float:
    reference_tokens = list(reference)
    candidate_tokens = list(candidate)
    smoothie = SmoothingFunction().method1
    return float(
        sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            weights=(0.7, 0.3),
            smoothing_function=smoothie,
        )
    )


def score_understanding_prediction(
    *,
    task_name: str,
    query: str,
    ground_truth: str,
    prediction: str,
    gpt_scorer: GPTScorer | None,
) -> dict[str, float]:
    if task_name == "rec":
        if ground_truth == "No.":
            if "no" in prediction.lower():
                return {"no_recall": 1.0, "accuracy": 1.0}
            return {"no_recall": 0.0, "accuracy": 0.0}
        if "yes" in prediction.lower():
            return {"recall": 1.0, "accuracy": 1.0}
        return {"recall": 0.0, "accuracy": 0.0}

    if "choice" in task_name:
        return {"accuracy": 1.0 if ground_truth in prediction else 0.0}

    if task_name in {"vqa", "text_only", "rea", "dense_rea"}:
        if gpt_scorer is None:
            raise ValueError("GPT scorer is required for text-judged understanding tasks")
        gpt_score = gpt_scorer.score_text_answer(query, ground_truth, prediction)
        return {"bleu": calculate_bleu(ground_truth, prediction), "gpt": gpt_score}

    raise ValueError(f"Unsupported understanding task: {task_name}")
