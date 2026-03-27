from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.tasks import build_generation_prompt_specs, build_understanding_example_specs
from evaluation.user_settings import PIPELINE_DEFAULTS, SMOKE_TEST_DEFAULTS


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _print_check(label: str, ok: bool, detail: str) -> None:
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {label}: {detail}")


def main() -> None:
    project_root = _project_root()
    adapter_name = str(SMOKE_TEST_DEFAULTS["adapter"])
    concept = str(SMOKE_TEST_DEFAULTS["concepts"][0])
    generation_task = str(SMOKE_TEST_DEFAULTS["tasks"][0])
    data_root = Path(str(PIPELINE_DEFAULTS["data_root"]))
    prompt_file = Path(str(PIPELINE_DEFAULTS["prompt_file"]))

    _print_check("project_root", project_root.exists(), str(project_root))
    _print_check("data_root", data_root.exists(), str(data_root))
    _print_check("prompt_file", prompt_file.exists(), str(prompt_file))

    try:
        prompt_specs = build_generation_prompt_specs(
            task_name=generation_task,
            concept=concept,
            data_root=data_root,
            prompts_to_eval_path=prompt_file,
            inverse_prompt=bool(PIPELINE_DEFAULTS["inverse_prompt"]),
            nums_new_token_i_stage_1=int(PIPELINE_DEFAULTS["nums_new_token_i_stage_1"]),
            nums_new_token_i_stage_2=int(PIPELINE_DEFAULTS["nums_new_token_i_stage_2"]),
            reference_image_count=int(SMOKE_TEST_DEFAULTS["reference_image_count"]),
        )
        _print_check("generation_task", True, f"{generation_task} -> {len(prompt_specs)} prompt(s)")
    except Exception as exc:
        _print_check("generation_task", False, repr(exc))

    try:
        understanding_specs = build_understanding_example_specs(
            task_name="vqa",
            concept=concept,
            data_root=data_root,
            reference_image_count=int(SMOKE_TEST_DEFAULTS["reference_image_count"]),
        )
        _print_check("understanding_task", True, f"vqa -> {len(understanding_specs)} item(s)")
    except Exception as exc:
        _print_check("understanding_task", False, repr(exc))

    try:
        from evaluation.adapters import build_adapter

        kwargs = dict(
            config_file=str(PIPELINE_DEFAULTS["config_file"]),
            token_weight_root=str(PIPELINE_DEFAULTS["token_weight_root"]),
            rl_weight_root=str(PIPELINE_DEFAULTS["rl_weight_root"]),
            epoch_to_load=int(SMOKE_TEST_DEFAULTS["epoch_to_load"]),
            nums_new_token_i_stage_1=int(PIPELINE_DEFAULTS["nums_new_token_i_stage_1"]),
            nums_new_token_i_stage_2=int(PIPELINE_DEFAULTS["nums_new_token_i_stage_2"]),
            device=str(PIPELINE_DEFAULTS["device"]),
            seed=int(PIPELINE_DEFAULTS["seed"]),
            model_id=PIPELINE_DEFAULTS["adapter_model_id"],
            code_root=PIPELINE_DEFAULTS["adapter_code_root"],
            max_memory_per_gpu=PIPELINE_DEFAULTS["adapter_max_memory_per_gpu"],
            offload_dir=PIPELINE_DEFAULTS["adapter_offload_dir"],
            reference_image_count=int(SMOKE_TEST_DEFAULTS["reference_image_count"]),
        )
        adapter = build_adapter(adapter_name, **kwargs)
        _print_check("adapter_import", True, f"{type(adapter).__name__} ({adapter.name})")
    except Exception as exc:
        _print_check("adapter_import", False, repr(exc))

    if adapter_name.startswith("bagel"):
        code_root = PIPELINE_DEFAULTS["adapter_code_root"]
        model_root = PIPELINE_DEFAULTS["adapter_model_id"]
        code_ok = code_root is not None and Path(str(code_root)).exists()
        model_ok = model_root is not None and Path(str(model_root)).exists()
        _print_check("bagel_code_root", code_ok, str(code_root))
        _print_check("bagel_model_root", model_ok, str(model_root))

    print("Preflight finished.")


if __name__ == "__main__":
    main()
