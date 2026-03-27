from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    family: str
    metrics: tuple[str, ...]
    default_batch_size: int = 1
    default_num_outputs: int = 1


@dataclass
class PromptSpec:
    prompt_id: str
    source_prompt: str
    generation_prompt: str
    scoring_prompt: str
    baseline_prompt: str | None = None
    conditioning_text: str | None = None
    reference_image_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptArtifact:
    prompt_id: str
    source_prompt: str
    generation_prompt: str
    scoring_prompt: str
    baseline_prompt: str | None
    conditioning_text: str | None
    reference_image_paths: list[str]
    image_files: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunManifest:
    task: str
    family: str
    concept: str
    token_epoch: int
    model_epoch: int
    adapter: str
    prompt_artifacts: list[PromptArtifact]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "family": self.family,
            "concept": self.concept,
            "token_epoch": self.token_epoch,
            "model_epoch": self.model_epoch,
            "adapter": self.adapter,
            "prompt_artifacts": [asdict(artifact) for artifact in self.prompt_artifacts],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunManifest":
        return cls(
            task=payload["task"],
            family=payload.get("family", "generation"),
            concept=payload["concept"],
            token_epoch=payload["token_epoch"],
            model_epoch=payload["model_epoch"],
            adapter=payload.get("adapter", payload.get("generator", "unknown")),
            prompt_artifacts=[
                PromptArtifact(**artifact) for artifact in payload["prompt_artifacts"]
            ],
        )


@dataclass
class UnderstandingExampleSpec:
    item_id: str
    source_prompt: str
    model_prompt: str
    scoring_query: str
    ground_truth: str
    image_path: str
    baseline_prompt: str | None = None
    conditioning_text: str | None = None
    reference_image_paths: list[str] = field(default_factory=list)
    prepend_system_prompt: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnderstandingArtifact:
    item_id: str
    source_prompt: str
    model_prompt: str
    scoring_query: str
    ground_truth: str
    image_path: str
    prediction: str
    baseline_prompt: str | None = None
    conditioning_text: str | None = None
    reference_image_paths: list[str] = field(default_factory=list)
    prepend_system_prompt: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnderstandingRunManifest:
    task: str
    family: str
    concept: str
    token_epoch: int
    model_epoch: int
    adapter: str
    items: list[UnderstandingArtifact]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "family": self.family,
            "concept": self.concept,
            "token_epoch": self.token_epoch,
            "model_epoch": self.model_epoch,
            "adapter": self.adapter,
            "items": [asdict(item) for item in self.items],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UnderstandingRunManifest":
        return cls(
            task=payload["task"],
            family=payload.get("family", "understanding"),
            concept=payload["concept"],
            token_epoch=payload["token_epoch"],
            model_epoch=payload["model_epoch"],
            adapter=payload.get("adapter", "unknown"),
            items=[UnderstandingArtifact(**item) for item in payload["items"]],
        )


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_adj_token_string(concept: str, stage_1_tokens: int, stage_2_tokens: int) -> str:
    total_tokens = stage_1_tokens + stage_2_tokens
    token_str = "".join(f"<token_{idx}>" for idx in range(total_tokens))
    return f"{token_str} <{concept}>"


def build_showo_system_prompt(concept: str, stage_1_tokens: int) -> str:
    token_str = "".join(f"<token_{idx}>" for idx in range(stage_1_tokens))
    return f"<{concept}> is {token_str}.\n"


def build_simple_conditioning_text(
    concept: str,
    train_info: dict[str, Any],
    max_facts: int = 1,
) -> str:
    class_name = train_info["class"]
    extra_info = train_info.get("extra_info", [])
    selected = [
        item.replace(f"<{concept}>", f"the reference {class_name}")
        for item in extra_info[:max_facts]
    ]
    if selected:
        return f"Reference {class_name}: " + " ".join(selected)
    return f"Reference {class_name}."


def load_reference_image_paths(
    data_root: str | Path,
    concept: str,
    count: int = 1,
) -> list[str]:
    data_root = Path(data_root)
    train_images = load_json(data_root / "concept" / "train" / "train_images.json")
    image_names = train_images[concept][:count]
    return [
        str(data_root / "concept" / "train" / concept / image_name)
        for image_name in image_names
    ]


def resolve_dataset_path(data_root: str | Path, raw_path: str | Path) -> Path:
    data_root = Path(data_root)
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    raw_text = str(raw_path).replace("\\", "/")
    anchor = "unictokens_data/"
    if anchor in raw_text:
        suffix = PurePosixPath(raw_text.split(anchor, 1)[1])
        return data_root / Path(*suffix.parts)

    return data_root / Path(candidate.name)


def list_concepts(data_root: str | Path, requested: Iterable[str] | None) -> list[str]:
    if requested is None:
        requested = ["all"]

    requested = list(requested)
    if requested == ["all"]:
        train_root = Path(data_root) / "concept" / "train"
        concepts = []
        for candidate in sorted(train_root.iterdir()):
            if candidate.is_dir() and (candidate / "info.json").exists():
                concepts.append(candidate.name)
        return concepts
    return requested


def mean_metric_dict(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}

    keys = sorted({key for metric_dict in metric_dicts for key in metric_dict})
    summary: dict[str, float] = {}
    for key in keys:
        values = [metric_dict[key] for metric_dict in metric_dicts if key in metric_dict]
        summary[key] = float(sum(values) / len(values)) if values else 0.0
    return summary


def stable_seed(base_seed: int, *parts: object) -> int:
    material = "::".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
