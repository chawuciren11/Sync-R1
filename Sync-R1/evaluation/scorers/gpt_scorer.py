from __future__ import annotations

import base64
import os
import re
from pathlib import Path

from openai import OpenAI


def _extract_score(text: str) -> float:
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        raise ValueError(f"Could not parse score from response: {text}")
    score = float(match.group(1))
    return max(0.0, min(1.0, score))


def _sanitize_visual_text(text: str) -> str:
    sanitized = re.sub(r"<[^>]+>", "the subject", text)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    if sanitized.lower().startswith("a photo of the subject "):
        return sanitized
    return sanitized


def _split_prompt(prompt: str) -> list[str]:
    clauses = re.split(r"[,.!?;:()\[\]{}<>/\\-]+", prompt)
    return [clause.strip() for clause in clauses if clause.strip()]


def _split_prompt_by_comma(prompt: str) -> list[str]:
    return [clause.strip() for clause in prompt.split(",") if clause.strip()]


class GPTScorer:
    """OpenAI-compatible multimodal/text judge used by both evaluation families."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 1e-5,
        max_tokens: int = 64,
        timeout: float = 120.0,
    ) -> None:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("GPT scoring requires --gpt-api-key or OPENAI_API_KEY")

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _parse_score_or_default(
        self,
        content: str,
        *,
        default: float,
        context: str,
    ) -> float:
        try:
            return _extract_score(content)
        except ValueError:
            normalized = " ".join(content.split()) if content else "<empty>"
            print(
                f"[warn] GPT scorer returned a non-numeric response for {context}; "
                f"defaulting to {default}. response={normalized}",
                flush=True,
            )
            return default

    def score_image_prompt_alignment(self, image_path: str | Path, prompt: str) -> float:
        clauses = _split_prompt(prompt)
        if not clauses:
            return 0.0

        scores = [self._score_single_image_clause(image_path, clause) for clause in clauses]
        return float(sum(scores) / len(scores))

    def score_dense_prompt_clause_coverage(self, image_path: str | Path, prompt: str) -> float:
        clauses = _split_prompt_by_comma(prompt)
        if not clauses:
            return 0.0

        clause_hits = [self._score_single_image_clause_binary(image_path, clause) for clause in clauses]
        return float(sum(clause_hits) / len(clause_hits))

    def score_text_answer(self, query: str, reference: str, prediction: str) -> float:
        system_prompt = (
            "You are a score evaluator. Given a question, a reference answer, and a predicted answer, "
            "you need to give a score from {0, 0.5, 1}. "
            "0 means completely irrelevant, 1 means completely relevant, and 0.5 means partially relevant. "
            "Ignore grammar and focus only on whether the correct content is answered. "
            "Return only the numeric score."
        )
        user_prompt = (
            f"Question: {query}\n"
            f"Reference answer: {reference}\n"
            f"Predicted answer: {prediction}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        return self._parse_score_or_default(content, default=0.0, context="text_answer")

    def score_prompt_similarity(
        self,
        source_prompt: str,
        reference_prompt: str,
        prediction: str,
    ) -> float:
        system_prompt = (
            "You are evaluating a rewritten image-generation prompt. "
            "Compare the predicted prompt against the reference rewritten prompt and score it with one of "
            "{0, 0.5, 1}. "
            "1 means the predicted prompt preserves the original intent and captures the key personalized "
            "details in the reference prompt. "
            "0.5 means it is partially correct but misses or weakens some important personalized details. "
            "0 means it fails to capture the required personalized details or changes the intended meaning. "
            "Ignore grammar and small wording differences. Return only the numeric score."
        )
        user_prompt = (
            f"Original prompt: {source_prompt}\n"
            f"Reference rewritten prompt: {reference_prompt}\n"
            f"Predicted rewritten prompt: {prediction}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        return self._parse_score_or_default(content, default=0.0, context="prompt_similarity")

    def _score_single_image_clause(self, image_path: str | Path, clause: str) -> float:
        image_path = Path(image_path)
        mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        image_url = f"data:{mime_type};base64,{encoded}"
        sanitized_clause = _sanitize_visual_text(clause)

        system_prompt = (
            "You are a professional image evaluator. "
            "Determine whether the text fragment is reflected in the image. "
            "Do not identify who is in the image or infer identity. "
            "If a fragment mentions a specific name or token, treat it only as referring to the same subject in the image. "
            "Return only one score from {0, 0.5, 1}. "
            "1 means fully reflected, 0.5 means partially reflected, and 0 means not reflected."
        )
        user_prompt = f"Text fragment: {sanitized_clause}"

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        )
        content = response.choices[0].message.content or ""
        return self._parse_score_or_default(
            content,
            default=0.0,
            context=f"image_clause:{sanitized_clause}",
        )

    def _score_single_image_clause_binary(self, image_path: str | Path, clause: str) -> float:
        image_path = Path(image_path)
        mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        image_url = f"data:{mime_type};base64,{encoded}"
        sanitized_clause = _sanitize_visual_text(clause)

        system_prompt = (
            "You are a professional image evaluator. "
            "Determine whether the image clearly presents the described text clause. "
            "Do not identify who is in the image or infer identity. "
            "If a clause mentions a specific name or token, treat it only as referring to the same subject in the image. "
            "Return only one score from {0, 1}. "
            "1 means the clause is clearly present in the image, and 0 means it is not present or too uncertain."
        )
        user_prompt = f"Text clause: {sanitized_clause}"

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        )
        content = response.choices[0].message.content or ""
        score = self._parse_score_or_default(
            content,
            default=0.0,
            context=f"image_clause_binary:{sanitized_clause}",
        )
        return 1.0 if score >= 0.5 else 0.0
