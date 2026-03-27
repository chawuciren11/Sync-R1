from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from ..common import PromptSpec, RunManifest, UnderstandingExampleSpec, UnderstandingRunManifest


class BaseEvalAdapter(ABC):
    name = "base"

    @abstractmethod
    def has_model_epoch(self, concept: str, model_epoch: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate_images(
        self,
        *,
        task_name: str,
        concept: str,
        prompt_specs: Sequence[PromptSpec],
        output_dir: str | Path,
        model_epoch: int,
        num_images: int,
        batch_size: int,
        overwrite: bool = False,
    ) -> RunManifest:
        raise NotImplementedError

    @abstractmethod
    def predict_understanding(
        self,
        *,
        task_name: str,
        concept: str,
        example_specs: Sequence[UnderstandingExampleSpec],
        output_dir: str | Path,
        model_epoch: int,
        overwrite: bool = False,
        top_k: int = 1,
        max_new_tokens: int = 100,
    ) -> UnderstandingRunManifest:
        raise NotImplementedError
