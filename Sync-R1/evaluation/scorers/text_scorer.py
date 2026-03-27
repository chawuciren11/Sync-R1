from __future__ import annotations

import re

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from .gpt_scorer import GPTScorer


YES_PATTERN = re.compile(r"\byes\b", flags=re.IGNORECASE)
NO_PATTERN = re.compile(r"\bno\b", flags=re.IGNORECASE)


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
        normalized_ground_truth = ground_truth.strip().lower()
        normalized_prediction = prediction.strip()
        predicts_yes = bool(YES_PATTERN.search(normalized_prediction))
        predicts_no = bool(NO_PATTERN.search(normalized_prediction))

        if normalized_ground_truth.startswith("no"):
            hit = 1.0 if predicts_no else 0.0
            return {
                "negative_recall": hit,
                "no_recall": hit,
                "accuracy": hit,
            }

        hit = 1.0 if predicts_yes else 0.0
        return {
            "positive_recall": hit,
            "recall": hit,
            "accuracy": hit,
        }

    if "choice" in task_name:
        return {"accuracy": 1.0 if ground_truth in prediction else 0.0}

    if task_name == "rea":
        return {"bleu": calculate_bleu(ground_truth, prediction)}

    if task_name == "dense_rea":
        if gpt_scorer is None:
            raise ValueError("GPT scorer is required for dense_rea scoring")
        gpt_score = gpt_scorer.score_prompt_similarity(query, ground_truth, prediction)
        return {"gpt": gpt_score}

    if task_name in {"vqa", "qa", "text_only"}:
        if gpt_scorer is None:
            raise ValueError("GPT scorer is required for text-judged understanding tasks")
        gpt_score = gpt_scorer.score_text_answer(query, ground_truth, prediction)
        return {"bleu": calculate_bleu(ground_truth, prediction), "gpt": gpt_score}

    raise ValueError(f"Unsupported understanding task: {task_name}")
