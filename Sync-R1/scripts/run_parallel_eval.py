from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.user_settings import PARALLEL_DEFAULTS


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _split_round_robin(items: list[str], num_groups: int) -> list[list[str]]:
    groups = [[] for _ in range(num_groups)]
    for idx, item in enumerate(items):
        groups[idx % num_groups].append(item)
    return groups


def _normalize_extra_args(extra_args: list[str]) -> list[str]:
    if extra_args[:1] == ["--"]:
        return extra_args[1:]
    return extra_args


def _normalize_score_visible_devices(group: str) -> str:
    return group.split(",")[0]


def _build_pipeline_command(
    *,
    project_root: Path,
    mode: str,
    adapter: str,
    tasks: list[str],
    concepts: list[str],
    model_epochs: list[int],
    data_root: str,
    output_root: str | None,
    device: str,
    extra_args: list[str],
) -> list[str]:
    command = [
        sys.executable,
        str(project_root / "eval_pipeline.py"),
        "--mode",
        mode,
        "--adapter",
        adapter,
        "--tasks",
        *tasks,
        "--concepts",
        *concepts,
        "--model-epochs",
        *[str(epoch) for epoch in model_epochs],
        "--data-root",
        data_root,
        "--device",
        device,
    ]
    if output_root:
        command.extend(["--output-root", output_root])
    command.extend(extra_args)
    return command


def _load_all_concepts(project_root: Path, data_root: str, concepts: list[str]) -> list[str]:
    sys.path.insert(0, str(project_root))
    from evaluation.common import list_concepts

    return list_concepts(data_root, concepts)


def main() -> None:
    project_root = _default_project_root()
    parser = argparse.ArgumentParser(description="Parallel generation launcher for eval_pipeline.py")
    parser.add_argument("--gpu-groups", nargs="+", default=PARALLEL_DEFAULTS["gpu_groups"], help="Examples: `0 1` or `0,1 2,3`")
    parser.add_argument("--mode", choices=("generate", "run"), default=PARALLEL_DEFAULTS["mode"])
    parser.add_argument("--adapter", default=PARALLEL_DEFAULTS["adapter"])
    parser.add_argument("--tasks", nargs="+", default=PARALLEL_DEFAULTS["tasks"])
    parser.add_argument("--concepts", nargs="+", default=PARALLEL_DEFAULTS["concepts"])
    parser.add_argument("--model-epochs", nargs="+", type=int, default=PARALLEL_DEFAULTS["model_epochs"])
    parser.add_argument("--data-root", type=str, default=PARALLEL_DEFAULTS["data_root"])
    parser.add_argument("--output-root", type=str, default=PARALLEL_DEFAULTS["output_root"])
    parser.add_argument("--generate-device", type=str, default=PARALLEL_DEFAULTS["generate_device"])
    parser.add_argument("--score-device", type=str, default=PARALLEL_DEFAULTS["score_device"])
    parser.add_argument("--stagger-seconds", type=float, default=PARALLEL_DEFAULTS["stagger_seconds"])
    parser.add_argument("--skip-score", action="store_true", default=PARALLEL_DEFAULTS["skip_score"])
    parser.add_argument("pipeline_extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    extra_args = _normalize_extra_args(args.pipeline_extra_args)
    all_concepts = _load_all_concepts(project_root, args.data_root, args.concepts)
    concept_groups = _split_round_robin(all_concepts, len(args.gpu_groups))

    workers: list[tuple[subprocess.Popen[str], str, list[str]]] = []
    for gpu_group, concept_subset in zip(args.gpu_groups, concept_groups):
        if not concept_subset:
            continue
        command = _build_pipeline_command(
            project_root=project_root,
            mode="generate",
            adapter=args.adapter,
            tasks=args.tasks,
            concepts=concept_subset,
            model_epochs=args.model_epochs,
            data_root=args.data_root,
            output_root=args.output_root,
            device=args.generate_device,
            extra_args=extra_args,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_group
        print(f"[launch] GPUs={gpu_group} concepts={concept_subset}")
        process = subprocess.Popen(command, cwd=str(project_root), env=env)
        workers.append((process, gpu_group, concept_subset))
        if args.stagger_seconds > 0:
            time.sleep(args.stagger_seconds)

    failures: list[tuple[str, list[str], int]] = []
    for process, gpu_group, concept_subset in workers:
        return_code = process.wait()
        if return_code != 0:
            failures.append((gpu_group, concept_subset, return_code))

    if failures:
        for gpu_group, concept_subset, return_code in failures:
            print(f"[failed] GPUs={gpu_group} concepts={concept_subset} code={return_code}")
        raise SystemExit(1)

    if args.mode == "generate" or args.skip_score:
        return

    score_command = _build_pipeline_command(
        project_root=project_root,
        mode="score",
        adapter=args.adapter,
        tasks=args.tasks,
        concepts=all_concepts,
        model_epochs=args.model_epochs,
        data_root=args.data_root,
        output_root=args.output_root,
        device=args.score_device,
        extra_args=extra_args,
    )
    score_env = os.environ.copy()
    if args.score_device != "cpu":
        score_env["CUDA_VISIBLE_DEVICES"] = _normalize_score_visible_devices(args.gpu_groups[0])
    print("[score] launching aggregate scoring")
    raise SystemExit(subprocess.call(score_command, cwd=str(project_root), env=score_env))


if __name__ == "__main__":
    main()
