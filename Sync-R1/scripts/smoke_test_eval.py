from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.user_settings import PIPELINE_DEFAULTS, SMOKE_TEST_DEFAULTS


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_command() -> list[str]:
    project_root = _default_project_root()
    command = [
        sys.executable,
        str(project_root / "eval_pipeline.py"),
        "--mode",
        str(SMOKE_TEST_DEFAULTS["mode"]),
        "--adapter",
        str(SMOKE_TEST_DEFAULTS["adapter"]),
        "--tasks",
        *[str(item) for item in SMOKE_TEST_DEFAULTS["tasks"]],
        "--concepts",
        *[str(item) for item in SMOKE_TEST_DEFAULTS["concepts"]],
        "--model-epochs",
        *[str(item) for item in SMOKE_TEST_DEFAULTS["model_epochs"]],
        "--output-root",
        str(SMOKE_TEST_DEFAULTS["output_root"]),
        "--epoch-to-load",
        str(SMOKE_TEST_DEFAULTS["epoch_to_load"]),
        "--reference-image-count",
        str(SMOKE_TEST_DEFAULTS["reference_image_count"]),
        "--data-root",
        str(PIPELINE_DEFAULTS["data_root"]),
    ]

    optional_pairs = [
        ("--num-images", SMOKE_TEST_DEFAULTS["num_images"]),
        ("--batch-size", SMOKE_TEST_DEFAULTS["batch_size"]),
        ("--adapter-model-id", PIPELINE_DEFAULTS["adapter_model_id"]),
        ("--adapter-code-root", PIPELINE_DEFAULTS["adapter_code_root"]),
        ("--adapter-max-memory-per-gpu", PIPELINE_DEFAULTS["adapter_max_memory_per_gpu"]),
        ("--adapter-offload-dir", PIPELINE_DEFAULTS["adapter_offload_dir"]),
        ("--gpt-model", PIPELINE_DEFAULTS["gpt_model"]),
        ("--gpt-api-key", PIPELINE_DEFAULTS["gpt_api_key"]),
        ("--gpt-base-url", PIPELINE_DEFAULTS["gpt_base_url"]),
    ]
    for flag, value in optional_pairs:
        if value is not None:
            command.extend([flag, str(value)])

    return command


def main() -> None:
    command = _build_command()
    print("[smoke-test] command:")
    print(" ".join(command))
    raise SystemExit(subprocess.call(command, cwd=str(_default_project_root())))


if __name__ == "__main__":
    main()
