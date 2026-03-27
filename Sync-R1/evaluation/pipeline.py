from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .common import (
    RunManifest,
    UnderstandingRunManifest,
    list_concepts,
    load_json,
    mean_metric_dict,
    set_global_seed,
    write_json,
)
from .tasks import (
    TASKS,
    build_generation_prompt_specs,
    build_understanding_example_specs,
    resolve_task_names,
)
from .user_settings import PIPELINE_DEFAULTS

if TYPE_CHECKING:
    from .adapters import BaseEvalAdapter
    from .scorers.clip_scorer import SHOWOConceptClipEvaluator
    from .scorers.gpt_scorer import GPTScorer


def _attach_legacy_metric_aliases(metrics: dict[str, float]) -> dict[str, float]:
    aliases = {
        "gpt": "ds-score",
        "clip_i": "clip-i",
        "clip_t": "clip-t",
    }
    expanded = dict(metrics)
    for metric_name, alias_name in aliases.items():
        if metric_name in metrics:
            expanded[alias_name] = metrics[metric_name]
    return expanded


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    project_root = _default_project_root()
    parser = argparse.ArgumentParser(description="Unified two-stage evaluation pipeline for Show-o style models.")
    parser.add_argument("--mode", choices=("generate", "score", "run"), default=PIPELINE_DEFAULTS["mode"])
    parser.add_argument("--adapter", type=str, default=PIPELINE_DEFAULTS["adapter"])
    parser.add_argument("--tasks", nargs="+", default=PIPELINE_DEFAULTS["tasks"])
    parser.add_argument("--concepts", nargs="+", default=PIPELINE_DEFAULTS["concepts"])
    parser.add_argument("--model-epochs", nargs="+", type=int, default=PIPELINE_DEFAULTS["model_epochs"])
    parser.add_argument("--output-root", type=str, default=PIPELINE_DEFAULTS["output_root"])
    parser.add_argument("--overwrite", action="store_true", default=PIPELINE_DEFAULTS["overwrite"])
    parser.add_argument("--seed", type=int, default=PIPELINE_DEFAULTS["seed"])

    parser.add_argument("--config-file", type=str, default=PIPELINE_DEFAULTS["config_file"])
    parser.add_argument("--data-root", type=str, default=PIPELINE_DEFAULTS["data_root"])
    parser.add_argument("--prompt-file", type=str, default=PIPELINE_DEFAULTS["prompt_file"])
    parser.add_argument("--token-weight-root", type=str, default=PIPELINE_DEFAULTS["token_weight_root"])
    parser.add_argument("--rl-weight-root", type=str, default=PIPELINE_DEFAULTS["rl_weight_root"])
    parser.add_argument("--epoch-to-load", type=int, default=PIPELINE_DEFAULTS["epoch_to_load"])
    parser.add_argument("--nums-new-token-i-stage-1", type=int, default=PIPELINE_DEFAULTS["nums_new_token_i_stage_1"])
    parser.add_argument("--nums-new-token-i-stage-2", type=int, default=PIPELINE_DEFAULTS["nums_new_token_i_stage_2"])
    parser.add_argument("--device", type=str, default=PIPELINE_DEFAULTS["device"])
    parser.add_argument("--inverse-prompt", dest="inverse_prompt", action="store_true", default=PIPELINE_DEFAULTS["inverse_prompt"])
    parser.add_argument("--no-inverse-prompt", dest="inverse_prompt", action="store_false")
    parser.add_argument(
        "--num-images",
        type=int,
        default=PIPELINE_DEFAULTS["num_images"],
        help="Number of images to generate per prompt for all generation tasks.",
    )
    parser.add_argument("--batch-size", type=int, default=PIPELINE_DEFAULTS["batch_size"], help="Override generation task batch size.")
    parser.add_argument("--reference-image-count", type=int, default=PIPELINE_DEFAULTS["reference_image_count"])
    parser.add_argument("--clip-i-reference-image-count", type=int, default=PIPELINE_DEFAULTS["clip_i_reference_image_count"])
    parser.add_argument("--mmu-top-k", type=int, default=PIPELINE_DEFAULTS["mmu_top_k"])
    parser.add_argument("--mmu-max-new-tokens", type=int, default=PIPELINE_DEFAULTS["mmu_max_new_tokens"])
    parser.add_argument("--allow-remote-models", action="store_true", default=PIPELINE_DEFAULTS["allow_remote_models"], help="Disable local_files_only for model loading.")
    parser.add_argument("--showo-model-path", type=str, default=PIPELINE_DEFAULTS["showo_model_path"])
    parser.add_argument("--vq-model-path", type=str, default=PIPELINE_DEFAULTS["vq_model_path"])
    parser.add_argument("--llm-model-path", type=str, default=PIPELINE_DEFAULTS["llm_model_path"])
    parser.add_argument("--generation-guidance-scale", type=float, default=PIPELINE_DEFAULTS["generation_guidance_scale"])
    parser.add_argument("--generation-timesteps", type=int, default=PIPELINE_DEFAULTS["generation_timesteps"])
    parser.add_argument("--generation-temperature", type=float, default=PIPELINE_DEFAULTS["generation_temperature"])
    parser.add_argument("--generation-noise-type", type=str, default=PIPELINE_DEFAULTS["generation_noise_type"])
    parser.add_argument("--adapter-model-id", type=str, default=PIPELINE_DEFAULTS["adapter_model_id"])
    parser.add_argument("--adapter-code-root", type=str, default=PIPELINE_DEFAULTS["adapter_code_root"])
    parser.add_argument("--adapter-api-key", type=str, default=PIPELINE_DEFAULTS["adapter_api_key"])
    parser.add_argument("--adapter-base-url", type=str, default=PIPELINE_DEFAULTS["adapter_base_url"])
    parser.add_argument("--adapter-max-memory-per-gpu", type=str, default=PIPELINE_DEFAULTS["adapter_max_memory_per_gpu"])
    parser.add_argument("--adapter-offload-dir", type=str, default=PIPELINE_DEFAULTS["adapter_offload_dir"])
    parser.add_argument("--adapter-image-height", type=int, default=PIPELINE_DEFAULTS["adapter_image_height"])
    parser.add_argument("--adapter-image-width", type=int, default=PIPELINE_DEFAULTS["adapter_image_width"])
    parser.add_argument("--adapter-cfg-text-scale", type=float, default=PIPELINE_DEFAULTS["adapter_cfg_text_scale"])
    parser.add_argument("--adapter-cfg-img-scale", type=float, default=PIPELINE_DEFAULTS["adapter_cfg_img_scale"])
    parser.add_argument("--adapter-cfg-interval-start", type=float, default=PIPELINE_DEFAULTS["adapter_cfg_interval_start"])
    parser.add_argument("--adapter-timestep-shift", type=float, default=PIPELINE_DEFAULTS["adapter_timestep_shift"])
    parser.add_argument("--adapter-num-timesteps", type=int, default=PIPELINE_DEFAULTS["adapter_num_timesteps"])
    parser.add_argument("--adapter-cfg-renorm-min", type=float, default=PIPELINE_DEFAULTS["adapter_cfg_renorm_min"])
    parser.add_argument("--adapter-cfg-renorm-type", type=str, default=PIPELINE_DEFAULTS["adapter_cfg_renorm_type"])
    parser.add_argument("--adapter-use-thinking", action="store_true", default=PIPELINE_DEFAULTS["adapter_use_thinking"])
    parser.add_argument("--adapter-temperature", type=float, default=PIPELINE_DEFAULTS["adapter_temperature"])
    parser.add_argument("--adapter-top-p", type=float, default=PIPELINE_DEFAULTS["adapter_top_p"])
    parser.add_argument("--adapter-top-k", type=int, default=PIPELINE_DEFAULTS["adapter_top_k"])
    parser.add_argument("--adapter-max-new-tokens", type=int, default=PIPELINE_DEFAULTS["adapter_max_new_tokens"])

    parser.add_argument("--clip-model-path", type=str, default=PIPELINE_DEFAULTS["clip_model_path"])
    parser.add_argument("--gpt-model", type=str, default=PIPELINE_DEFAULTS["gpt_model"])
    parser.add_argument("--gpt-api-key", type=str, default=PIPELINE_DEFAULTS["gpt_api_key"])
    parser.add_argument("--gpt-base-url", type=str, default=PIPELINE_DEFAULTS["gpt_base_url"])
    parser.add_argument("--gpt-temperature", type=float, default=PIPELINE_DEFAULTS["gpt_temperature"])
    parser.add_argument("--gpt-max-tokens", type=int, default=PIPELINE_DEFAULTS["gpt_max_tokens"])
    parser.add_argument("--gpt-timeout", type=float, default=PIPELINE_DEFAULTS["gpt_timeout"])
    return parser


def _artifact_dir(
    output_root: Path,
    family: str,
    task_name: str,
    concept: str,
    token_epoch: int,
    model_epoch: int,
) -> Path:
    return (
        output_root
        / "artifacts"
        / family
        / task_name
        / concept
        / f"token_epoch_{token_epoch}"
        / f"model_epoch_{model_epoch}"
    )


def _scores_dir(output_root: Path, family: str, task_name: str, model_epoch: int) -> Path:
    return output_root / "scores" / family / task_name / f"model_epoch_{model_epoch}"


def _manifest_path(
    output_root: Path,
    family: str,
    task_name: str,
    concept: str,
    token_epoch: int,
    model_epoch: int,
) -> Path:
    return _artifact_dir(output_root, family, task_name, concept, token_epoch, model_epoch) / "manifest.json"


def _build_adapter(args: argparse.Namespace) -> "BaseEvalAdapter":
    from .adapters import build_adapter

    return build_adapter(
        args.adapter,
        config_file=args.config_file,
        token_weight_root=args.token_weight_root,
        rl_weight_root=args.rl_weight_root,
        epoch_to_load=args.epoch_to_load,
        nums_new_token_i_stage_1=args.nums_new_token_i_stage_1,
        nums_new_token_i_stage_2=args.nums_new_token_i_stage_2,
        device=args.device,
        seed=args.seed,
        local_files_only=not args.allow_remote_models,
        showo_model_path=args.showo_model_path,
        vq_model_path=args.vq_model_path,
        llm_model_path=args.llm_model_path,
        guidance_scale=args.generation_guidance_scale,
        generation_timesteps=args.generation_timesteps,
        generation_temperature=args.generation_temperature,
        noise_type=args.generation_noise_type,
        model_id=args.adapter_model_id,
        code_root=args.adapter_code_root,
        api_key=args.adapter_api_key,
        base_url=args.adapter_base_url,
        max_memory_per_gpu=args.adapter_max_memory_per_gpu,
        offload_dir=args.adapter_offload_dir,
        image_height=args.adapter_image_height,
        image_width=args.adapter_image_width,
        cfg_text_scale=args.adapter_cfg_text_scale,
        cfg_img_scale=args.adapter_cfg_img_scale,
        cfg_interval_start=args.adapter_cfg_interval_start,
        timestep_shift=args.adapter_timestep_shift,
        num_timesteps=args.adapter_num_timesteps,
        cfg_renorm_min=args.adapter_cfg_renorm_min,
        cfg_renorm_type=args.adapter_cfg_renorm_type,
        use_thinking=args.adapter_use_thinking,
        adapter_temperature=args.adapter_temperature,
        adapter_top_p=args.adapter_top_p,
        adapter_top_k=args.adapter_top_k,
        adapter_max_new_tokens=args.adapter_max_new_tokens,
        reference_image_count=args.reference_image_count,
    )


def _generation_stage_task_order(task_names: list[str]) -> list[str]:
    original_order = {task_name: idx for idx, task_name in enumerate(task_names)}
    priority = {
        "rec": 10,
        "vqa": 10,
        "qa": 10,
        "rea": 10,
        "dense_rea": 10,
        "pure_gen": 20,
        "dense_gen": 20,
        "rea_gen": 30,
        "dense_rea_gen": 30,
    }
    return sorted(task_names, key=lambda name: (priority.get(name, 20), original_order[name]))


def run_generation_stage(args: argparse.Namespace, task_names: list[str], concepts: list[str]) -> list[Path]:
    adapter = _build_adapter(args)
    output_root = Path(args.output_root)
    generated_manifests: list[Path] = []
    ordered_task_names = _generation_stage_task_order(task_names)

    for task_name in ordered_task_names:
        task_definition = TASKS[task_name]
        for concept in concepts:
            for model_epoch in args.model_epochs:
                if not adapter.has_model_epoch(concept, model_epoch):
                    print(f"[skip] missing checkpoint for {concept} epoch {model_epoch}")
                    continue

                output_dir = _artifact_dir(
                    output_root=output_root,
                    family=task_definition.family,
                    task_name=task_name,
                    concept=concept,
                    token_epoch=args.epoch_to_load,
                    model_epoch=model_epoch,
                )

                if task_definition.family == "generation":
                    prompt_specs = build_generation_prompt_specs(
                        task_name=task_name,
                        concept=concept,
                        data_root=args.data_root,
                        prompts_to_eval_path=args.prompt_file,
                        inverse_prompt=args.inverse_prompt,
                        nums_new_token_i_stage_1=args.nums_new_token_i_stage_1,
                        nums_new_token_i_stage_2=args.nums_new_token_i_stage_2,
                        reference_image_count=args.reference_image_count,
                        clip_i_reference_image_count=args.clip_i_reference_image_count,
                        output_root=args.output_root,
                        token_epoch=args.epoch_to_load,
                        model_epoch=model_epoch,
                    )
                    num_outputs = args.num_images
                    batch_size = args.batch_size or task_definition.default_batch_size
                    adapter.generate_images(
                        task_name=task_name,
                        concept=concept,
                        prompt_specs=prompt_specs,
                        output_dir=output_dir,
                        model_epoch=model_epoch,
                        num_images=num_outputs,
                        batch_size=batch_size,
                        overwrite=args.overwrite,
                    )
                else:
                    example_specs = build_understanding_example_specs(
                        task_name=task_name,
                        concept=concept,
                        data_root=args.data_root,
                        reference_image_count=args.reference_image_count,
                    )
                    adapter.predict_understanding(
                        task_name=task_name,
                        concept=concept,
                        example_specs=example_specs,
                        output_dir=output_dir,
                        model_epoch=model_epoch,
                        overwrite=args.overwrite,
                        top_k=args.mmu_top_k,
                        max_new_tokens=args.mmu_max_new_tokens,
                    )

                manifest_path = output_dir / "manifest.json"
                generated_manifests.append(manifest_path)
                print(f"[generated] {task_name} | {concept} | epoch {model_epoch} -> {manifest_path}")

    return generated_manifests


def _load_generation_manifest(path: Path) -> RunManifest:
    return RunManifest.from_dict(load_json(path))


def _load_understanding_manifest(path: Path) -> UnderstandingRunManifest:
    return UnderstandingRunManifest.from_dict(load_json(path))


def _build_gpt_scorer_if_needed(args: argparse.Namespace, task_names: list[str]) -> "GPTScorer | None":
    from .scorers.gpt_scorer import GPTScorer

    requires_gpt = any("gpt" in TASKS[task_name].metrics for task_name in task_names)
    if not requires_gpt:
        return None
    return GPTScorer(
        model=args.gpt_model,
        api_key=args.gpt_api_key,
        base_url=args.gpt_base_url,
        temperature=args.gpt_temperature,
        max_tokens=args.gpt_max_tokens,
        timeout=args.gpt_timeout,
    )


def _build_generation_score_payload(
    manifest: RunManifest,
    prompt_results: list[dict[str, Any]],
    *,
    complete: bool,
) -> dict[str, Any]:
    summary = mean_metric_dict([item["metrics"] for item in prompt_results])
    summary = _attach_legacy_metric_aliases(summary)
    return {
        "task": manifest.task,
        "family": manifest.family,
        "concept": manifest.concept,
        "token_epoch": manifest.token_epoch,
        "model_epoch": manifest.model_epoch,
        "adapter": manifest.adapter,
        "complete": complete,
        "progress": {
            "completed": len(prompt_results),
            "total": len(manifest.prompt_artifacts),
        },
        "prompt_results": prompt_results,
        "summary": summary,
    }


def _build_understanding_score_payload(
    manifest: UnderstandingRunManifest,
    item_results: list[dict[str, Any]],
    *,
    complete: bool,
) -> dict[str, Any]:
    summary = mean_metric_dict([item["metrics"] for item in item_results])
    if manifest.task == "rec":
        positive_values = [
            item["metrics"]["positive_recall"]
            for item in item_results
            if "positive_recall" in item["metrics"]
        ]
        negative_values = [
            item["metrics"]["negative_recall"]
            for item in item_results
            if "negative_recall" in item["metrics"]
        ]
        positive_recall = float(sum(positive_values) / len(positive_values)) if positive_values else 0.0
        negative_recall = float(sum(negative_values) / len(negative_values)) if negative_values else 0.0
        summary["positive_recall"] = positive_recall
        summary["negative_recall"] = negative_recall
        summary["recall"] = positive_recall
        summary["no_recall"] = negative_recall
        summary["weight"] = float((positive_recall + negative_recall) / 2.0)
    summary = _attach_legacy_metric_aliases(summary)
    return {
        "task": manifest.task,
        "family": manifest.family,
        "concept": manifest.concept,
        "token_epoch": manifest.token_epoch,
        "model_epoch": manifest.model_epoch,
        "adapter": manifest.adapter,
        "complete": complete,
        "progress": {
            "completed": len(item_results),
            "total": len(manifest.items),
        },
        "items": item_results,
        "summary": summary,
    }


def _score_generation_manifest(
    manifest: RunManifest,
    clip_scorer: "SHOWOConceptClipEvaluator",
    gpt_scorer: "GPTScorer | None",
    artifact_root: Path,
    existing_result: dict[str, Any] | None = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    task_definition = TASKS[manifest.task]
    existing_prompt_results = {
        item["prompt_id"]: item
        for item in (existing_result or {}).get("prompt_results", [])
    }
    prompt_results: list[dict[str, Any]] = []

    for prompt_artifact in manifest.prompt_artifacts:
        if prompt_artifact.prompt_id in existing_prompt_results:
            prompt_results.append(existing_prompt_results[prompt_artifact.prompt_id])
            if progress_callback is not None:
                progress_callback(
                    _build_generation_score_payload(
                        manifest,
                        prompt_results,
                        complete=(len(prompt_results) == len(manifest.prompt_artifacts)),
                    )
                )
            continue

        prompt_dir = artifact_root / prompt_artifact.prompt_id
        image_paths = [prompt_dir / image_file for image_file in prompt_artifact.image_files]
        clip_i_reference_image_paths = prompt_artifact.metadata.get("clip_i_reference_image_paths")

        clip_score = clip_scorer.score_prompt_images(
            concept=manifest.concept,
            image_paths=image_paths,
            prompt=prompt_artifact.scoring_prompt if "clip_t" in task_definition.metrics else None,
            reference_image_paths=clip_i_reference_image_paths,
        )

        metrics: dict[str, float] = {}
        if "clip_i" in task_definition.metrics:
            metrics["clip_i"] = clip_score.clip_i
        if "clip_t" in task_definition.metrics and clip_score.clip_t is not None:
            metrics["clip_t"] = clip_score.clip_t
        if "gpt" in task_definition.metrics:
            if gpt_scorer is None:
                raise ValueError("GPT scorer was not initialized, but the task requires GPT scoring")
            if manifest.task in {"dense_gen", "dense_rea_gen"}:
                gpt_scores = [
                    gpt_scorer.score_dense_prompt_clause_coverage(
                        image_path,
                        prompt_artifact.scoring_prompt,
                    )
                    for image_path in image_paths
                ]
            else:
                gpt_scores = [
                    gpt_scorer.score_image_prompt_alignment(image_path, prompt_artifact.scoring_prompt)
                    for image_path in image_paths
                ]
            metrics["gpt"] = float(sum(gpt_scores) / len(gpt_scores))
        metrics = _attach_legacy_metric_aliases(metrics)

        prompt_results.append(
            {
                "prompt_id": prompt_artifact.prompt_id,
                "source_prompt": prompt_artifact.source_prompt,
                "generation_prompt": prompt_artifact.generation_prompt,
                "scoring_prompt": prompt_artifact.scoring_prompt,
                "baseline_prompt": prompt_artifact.baseline_prompt,
                "tp_prefix_text": prompt_artifact.tp_prefix_text,
                "conditioning_text": prompt_artifact.conditioning_text,
                "reference_image_paths": prompt_artifact.reference_image_paths,
                "image_files": prompt_artifact.image_files,
                "metrics": metrics,
                "metadata": prompt_artifact.metadata,
            }
        )

        if progress_callback is not None:
            progress_callback(
                _build_generation_score_payload(
                    manifest,
                    prompt_results,
                    complete=(len(prompt_results) == len(manifest.prompt_artifacts)),
                )
            )

    return _build_generation_score_payload(manifest, prompt_results, complete=True)


def _score_understanding_manifest(
    manifest: UnderstandingRunManifest,
    gpt_scorer: "GPTScorer | None",
    existing_result: dict[str, Any] | None = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    from .scorers.text_scorer import score_understanding_prediction

    existing_item_results = {
        item["item_id"]: item
        for item in (existing_result or {}).get("items", [])
    }
    item_results: list[dict[str, Any]] = []
    for item in manifest.items:
        if item.item_id in existing_item_results:
            item_results.append(existing_item_results[item.item_id])
            if progress_callback is not None:
                progress_callback(
                    _build_understanding_score_payload(
                        manifest,
                        item_results,
                        complete=(len(item_results) == len(manifest.items)),
                    )
                )
            continue

        metrics = score_understanding_prediction(
            task_name=manifest.task,
            query=item.scoring_query,
            ground_truth=item.ground_truth,
            prediction=item.prediction,
            gpt_scorer=gpt_scorer,
        )
        metrics = _attach_legacy_metric_aliases(metrics)
        item_results.append(
            {
                "item_id": item.item_id,
                "source_prompt": item.source_prompt,
                "model_prompt": item.model_prompt,
                "scoring_query": item.scoring_query,
                "ground_truth": item.ground_truth,
                "prediction": item.prediction,
                "image_path": item.image_path,
                "baseline_prompt": item.baseline_prompt,
                "tp_prefix_text": item.tp_prefix_text,
                "conditioning_text": item.conditioning_text,
                "reference_image_paths": item.reference_image_paths,
                "prepend_system_prompt": item.prepend_system_prompt,
                "metrics": metrics,
                "metadata": item.metadata,
            }
        )

        if progress_callback is not None:
            progress_callback(
                _build_understanding_score_payload(
                    manifest,
                    item_results,
                    complete=(len(item_results) == len(manifest.items)),
                )
            )

    return _build_understanding_score_payload(manifest, item_results, complete=True)


def _write_task_summary(
    *,
    score_dir: Path,
    task_definition: Any,
    task_name: str,
    model_epoch: int,
    concept_results: dict[str, Any],
) -> None:
    summary = mean_metric_dict([concept_results[name]["summary"] for name in concept_results])
    summary = _attach_legacy_metric_aliases(summary)
    write_json(
        score_dir / "summary.json",
        {
            "task": task_name,
            "family": task_definition.family,
            "model_epoch": model_epoch,
            "concepts": {
                name: concept_results[name]["summary"] for name in concept_results
            },
            "summary": summary,
        },
    )


def _manifest_entry_count(path: Path, family: str) -> int | None:
    if not path.exists():
        return None
    payload = load_json(path)
    if family == "generation":
        return len(payload.get("prompt_artifacts", []))
    return len(payload.get("items", []))


def run_scoring(args: argparse.Namespace, task_names: list[str], concepts: list[str]) -> None:
    output_root = Path(args.output_root)
    clip_scorer = None
    gpt_scorer = None

    for task_name in task_names:
        task_definition = TASKS[task_name]
        task_results_by_epoch: dict[int, dict[str, Any]] = {}

        for model_epoch in args.model_epochs:
            score_dir = _scores_dir(output_root, task_definition.family, task_name, model_epoch)
            score_dir.mkdir(parents=True, exist_ok=True)
            concept_results: dict[str, Any] = {}

            for concept in concepts:
                manifest_path = _manifest_path(
                    output_root,
                    task_definition.family,
                    task_name,
                    concept,
                    args.epoch_to_load,
                    model_epoch,
                )
                manifest_entry_total = _manifest_entry_count(manifest_path, task_definition.family)
                score_path = score_dir / f"{concept}.json"
                if score_path.exists() and not args.overwrite:
                    existing_score = load_json(score_path)
                    existing_completed = existing_score.get("progress", {}).get("completed")
                    if existing_completed is None:
                        existing_completed = len(
                            existing_score.get(
                                "prompt_results" if task_definition.family == "generation" else "items",
                                [],
                            )
                        )
                    manifest_matches_existing = (
                        manifest_entry_total is None or existing_completed == manifest_entry_total
                    )
                    if existing_score.get("complete", True) and manifest_matches_existing:
                        concept_results[concept] = existing_score
                        print(f"[resume-score] {task_name} | {concept} | epoch {model_epoch}")
                        _write_task_summary(
                            score_dir=score_dir,
                            task_definition=task_definition,
                            task_name=task_name,
                            model_epoch=model_epoch,
                            concept_results=concept_results,
                        )
                        continue
                    if existing_score.get("complete", True) and not manifest_matches_existing:
                        existing_score["complete"] = False
                        existing_score["progress"] = {
                            "completed": existing_completed,
                            "total": manifest_entry_total,
                        }
                        print(
                            f"[resume-score-stale] {task_name} | {concept} | epoch {model_epoch} | "
                            f"{existing_completed}/{manifest_entry_total}",
                            flush=True,
                        )
                    progress = existing_score.get("progress", {})
                    print(
                        f"[resume-score-progress] {task_name} | {concept} | epoch {model_epoch} | "
                        f"{progress.get('completed', 0)}/{progress.get('total', '?')}",
                        flush=True,
                    )
                else:
                    existing_score = None
                if not manifest_path.exists():
                    print(f"[skip] manifest not found: {manifest_path}")
                    continue

                if task_definition.family == "generation":
                    if clip_scorer is None:
                        from .scorers.clip_scorer import SHOWOConceptClipEvaluator

                        clip_scorer = SHOWOConceptClipEvaluator(
                            device=args.device,
                            clip_model_path=args.clip_model_path,
                            data_root=args.data_root,
                        )
                    if "gpt" in task_definition.metrics and gpt_scorer is None:
                        gpt_scorer = _build_gpt_scorer_if_needed(args, task_names)
                    manifest = _load_generation_manifest(manifest_path)
                    result = _score_generation_manifest(
                        manifest=manifest,
                        clip_scorer=clip_scorer,
                        gpt_scorer=gpt_scorer,
                        artifact_root=manifest_path.parent,
                        existing_result=existing_score,
                        progress_callback=lambda partial_result, concept=concept: (
                            concept_results.__setitem__(concept, partial_result),
                            write_json(score_path, partial_result),
                            _write_task_summary(
                                score_dir=score_dir,
                                task_definition=task_definition,
                                task_name=task_name,
                                model_epoch=model_epoch,
                                concept_results=concept_results,
                            ),
                        ),
                    )
                else:
                    if "gpt" in task_definition.metrics and gpt_scorer is None:
                        gpt_scorer = _build_gpt_scorer_if_needed(args, task_names)
                    manifest = _load_understanding_manifest(manifest_path)
                    result = _score_understanding_manifest(
                        manifest=manifest,
                        gpt_scorer=gpt_scorer,
                        existing_result=existing_score,
                        progress_callback=lambda partial_result, concept=concept: (
                            concept_results.__setitem__(concept, partial_result),
                            write_json(score_path, partial_result),
                            _write_task_summary(
                                score_dir=score_dir,
                                task_definition=task_definition,
                                task_name=task_name,
                                model_epoch=model_epoch,
                                concept_results=concept_results,
                            ),
                        ),
                    )

                concept_results[concept] = result
                write_json(score_path, result)
                _write_task_summary(
                    score_dir=score_dir,
                    task_definition=task_definition,
                    task_name=task_name,
                    model_epoch=model_epoch,
                    concept_results=concept_results,
                )
                print(f"[scored] {task_name} | {concept} | epoch {model_epoch}")

            if not concept_results:
                continue

            summary = mean_metric_dict([concept_results[concept]["summary"] for concept in concept_results])
            summary = _attach_legacy_metric_aliases(summary)
            task_results_by_epoch[model_epoch] = {
                "task": task_name,
                "family": task_definition.family,
                "model_epoch": model_epoch,
                "concepts": {
                    concept: concept_results[concept]["summary"] for concept in concept_results
                },
                "summary": summary,
            }
            write_json(score_dir / "summary.json", task_results_by_epoch[model_epoch])

        if task_results_by_epoch:
            write_json(
                output_root / "scores" / task_definition.family / task_name / "all_epochs_summary.json",
                task_results_by_epoch,
            )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    set_global_seed(args.seed)

    task_names = resolve_task_names(args.tasks)
    concepts = list_concepts(args.data_root, args.concepts)

    if args.mode in ("generate", "run"):
        run_generation_stage(args, task_names, concepts)
    if args.mode in ("score", "run"):
        run_scoring(args, task_names, concepts)
