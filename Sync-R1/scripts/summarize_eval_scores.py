from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import mean_metric_dict, write_json
from evaluation.tasks import resolve_task_names
from evaluation.user_settings import DEFAULT_TEST_CONCEPTS, PIPELINE_DEFAULTS


TASK_METRICS: dict[str, tuple[str, ...]] = {
    "rec": ("weight",),
    "rea": ("bleu",),
    "dense_rea": ("gpt",),
    "vqa": ("bleu", "gpt"),
    "qa": ("bleu", "gpt"),
    "pure_gen": ("clip_t", "clip_i"),
    "dense_gen": ("gpt", "clip_i"),
    "rea_gen": ("clip_t", "clip_i"),
    "dense_rea_gen": ("gpt", "clip_i"),
}


FLAT_TABLE_KEYS: dict[tuple[str, str], str] = {
    ("rec", "weight"): "rec.weight",
    ("rea", "bleu"): "rea.bleu",
    ("dense_rea", "gpt"): "dense_rea.gpt",
    ("vqa", "bleu"): "vqa.bleu",
    ("vqa", "gpt"): "vqa.gpt",
    ("qa", "bleu"): "qa.bleu",
    ("qa", "gpt"): "qa.gpt",
    ("pure_gen", "clip_t"): "pure_gen.clip_t",
    ("pure_gen", "clip_i"): "pure_gen.clip_i",
    ("dense_gen", "gpt"): "dense_gen.gpt",
    ("dense_gen", "clip_i"): "dense_gen.clip_i",
    ("rea_gen", "clip_t"): "rea_gen.clip_t",
    ("rea_gen", "clip_i"): "rea_gen.clip_i",
    ("dense_rea_gen", "gpt"): "dense_rea_gen.gpt",
    ("dense_rea_gen", "clip_i"): "dense_rea_gen.clip_i",
}


def _default_output_roots() -> list[str]:
    return [str(PIPELINE_DEFAULTS["output_root"])]


def _canonical_metric(summary: dict[str, Any], metric_name: str) -> float | None:
    aliases = {
        "gpt": ("gpt", "ds-score"),
        "clip_i": ("clip_i", "clip-i"),
        "clip_t": ("clip_t", "clip-t"),
        "weight": ("weight", "accuracy"),
        "bleu": ("bleu",),
    }
    for candidate in aliases.get(metric_name, (metric_name,)):
        value = summary.get(candidate)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _score_file_path(output_root: Path, task_name: str, model_epoch: int, concept: str) -> Path:
    family = "generation" if "_gen" in task_name or task_name == "pure_gen" else "understanding"
    return output_root / "scores" / family / task_name / f"model_epoch_{model_epoch}" / f"{concept}.json"


def _load_concept_result(output_roots: list[Path], task_name: str, model_epoch: int, concept: str) -> dict[str, Any] | None:
    for output_root in output_roots:
        score_path = _score_file_path(output_root, task_name, model_epoch, concept)
        if score_path.exists():
            with score_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    return None


def _summarize_task(
    *,
    output_roots: list[Path],
    task_name: str,
    model_epoch: int,
    concepts: list[str],
    adapter_filter: str | None,
) -> dict[str, Any]:
    concept_summaries: dict[str, dict[str, float]] = {}
    adapters_seen: dict[str, int] = {}
    missing_concepts: list[str] = []

    for concept in concepts:
        payload = _load_concept_result(output_roots, task_name, model_epoch, concept)
        if payload is None:
            missing_concepts.append(concept)
            continue

        adapter_name = str(payload.get("adapter", "unknown"))
        if adapter_filter and adapter_name != adapter_filter:
            missing_concepts.append(concept)
            continue

        adapters_seen[adapter_name] = adapters_seen.get(adapter_name, 0) + 1
        summary_block = payload.get("summary", {})
        metrics: dict[str, float] = {}
        for metric_name in TASK_METRICS[task_name]:
            metric_value = _canonical_metric(summary_block, metric_name)
            if metric_value is not None:
                metrics[metric_name] = metric_value
        if metrics:
            concept_summaries[concept] = metrics
        else:
            missing_concepts.append(concept)

    task_summary = mean_metric_dict(list(concept_summaries.values()))
    return {
        "task": task_name,
        "model_epoch": model_epoch,
        "num_concepts_found": len(concept_summaries),
        "num_concepts_expected": len(concepts),
        "missing_concepts": missing_concepts,
        "adapters_seen": adapters_seen,
        "summary": task_summary,
        "concepts": concept_summaries,
    }


def _flatten_epoch_summary(task_summaries: dict[str, dict[str, Any]]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for task_name, task_payload in task_summaries.items():
        summary = task_payload.get("summary", {})
        for metric_name in TASK_METRICS[task_name]:
            flat_key = FLAT_TABLE_KEYS[(task_name, metric_name)]
            metric_value = summary.get(metric_name)
            if isinstance(metric_value, (int, float)):
                flattened[flat_key] = float(metric_value)
    return flattened


def _print_epoch_summary(model_epoch: int, epoch_payload: dict[str, Any]) -> None:
    print(f"\n[model_epoch_{model_epoch}]")
    flattened = epoch_payload["flat_summary"]
    if not flattened:
        print("No scores found.")
        return
    for flat_key in sorted(flattened):
        print(f"{flat_key}: {flattened[flat_key]:.4f}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate per-concept evaluation scores into overall multi-concept summaries.")
    parser.add_argument("--output-roots", nargs="+", default=_default_output_roots())
    parser.add_argument("--concepts", nargs="+", default=DEFAULT_TEST_CONCEPTS.copy())
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--model-epochs", nargs="+", type=int, default=PIPELINE_DEFAULTS["model_epochs"])
    parser.add_argument("--adapter", type=str, default=None, help="Optional adapter filter such as bagel_tp or bagel_ip.")
    parser.add_argument("--write-root", type=str, default=None, help="Where to write aggregated JSON outputs. Defaults to the first output root.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    task_names = resolve_task_names(args.tasks)
    output_roots = [Path(root).expanduser().resolve() for root in args.output_roots]
    write_root = Path(args.write_root).expanduser().resolve() if args.write_root else output_roots[0]

    aggregate_root = write_root / "scores" / "_aggregates"
    aggregate_root.mkdir(parents=True, exist_ok=True)

    all_epochs_payload: dict[str, Any] = {
        "output_roots": [str(path) for path in output_roots],
        "concepts": list(args.concepts),
        "tasks": task_names,
        "adapter_filter": args.adapter,
        "model_epochs": {},
    }

    for model_epoch in args.model_epochs:
        task_summaries: dict[str, dict[str, Any]] = {}
        for task_name in task_names:
            task_summaries[task_name] = _summarize_task(
                output_roots=output_roots,
                task_name=task_name,
                model_epoch=model_epoch,
                concepts=list(args.concepts),
                adapter_filter=args.adapter,
            )

        epoch_payload = {
            "model_epoch": model_epoch,
            "tasks": task_summaries,
            "flat_summary": _flatten_epoch_summary(task_summaries),
        }
        all_epochs_payload["model_epochs"][str(model_epoch)] = epoch_payload

        epoch_dir = aggregate_root / f"model_epoch_{model_epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        write_json(epoch_dir / "summary.json", epoch_payload)
        _print_epoch_summary(model_epoch, epoch_payload)

    write_json(aggregate_root / "all_epochs_summary.json", all_epochs_payload)
    print(f"\n[written] {aggregate_root / 'all_epochs_summary.json'}")


if __name__ == "__main__":
    main()
