from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_NUM_IMAGES_PER_PROMPT = 1
DEFAULT_ADAPTER = "showo"

# Janus presets:
# 1. Switch DEFAULT_ADAPTER to "janus_tp" or "janus_ip" to enable Janus.
# 2. Switch JANUS_DEFAULT_VARIANT between "januspro1b" and "januspro7b".
# 3. If Janus is not pip-installed, point JANUS_CODE_ROOT to the Janus repo root.
JANUS_MODEL_PRESETS: dict[str, dict[str, object]] = {
    "januspro1b": {
        "model_id": "deepseek-ai/Janus-Pro-1B",
        "image_size": 384,
        "max_memory_per_gpu": None,
    },
    "januspro7b": {
        "model_id": "deepseek-ai/Janus-Pro-7B",
        "image_size": 384,
        "max_memory_per_gpu": None,
    },
}
JANUS_DEFAULT_VARIANT = "januspro7b"
JANUS_CODE_ROOT: str | None = None


def _is_external_single_epoch_adapter(adapter_name: str) -> bool:
    return adapter_name.startswith(("bagel", "janus"))


def _is_janus_adapter(adapter_name: str) -> bool:
    return adapter_name.startswith("janus")


def _active_janus_preset() -> dict[str, object]:
    return JANUS_MODEL_PRESETS[JANUS_DEFAULT_VARIANT]


ACTIVE_JANUS_PRESET = _active_janus_preset()
DEFAULT_MODEL_EPOCHS = [0] if _is_external_single_epoch_adapter(DEFAULT_ADAPTER) else [0, 2, 4, 6]
DEFAULT_EPOCH_TO_LOAD = 0 if _is_external_single_epoch_adapter(DEFAULT_ADAPTER) else 15
DEFAULT_ADAPTER_MODEL_ID = ACTIVE_JANUS_PRESET["model_id"] if _is_janus_adapter(DEFAULT_ADAPTER) else None
DEFAULT_ADAPTER_CODE_ROOT = JANUS_CODE_ROOT if _is_janus_adapter(DEFAULT_ADAPTER) else None
DEFAULT_ADAPTER_MAX_MEMORY_PER_GPU = (
    ACTIVE_JANUS_PRESET["max_memory_per_gpu"] if _is_janus_adapter(DEFAULT_ADAPTER) else None
)
DEFAULT_ADAPTER_IMAGE_SIZE = ACTIVE_JANUS_PRESET["image_size"] if _is_janus_adapter(DEFAULT_ADAPTER) else 1024
DEFAULT_SMOKE_TEST_ADAPTER = DEFAULT_ADAPTER if DEFAULT_ADAPTER.startswith(("bagel", "janus")) else "janus_tp"

DEFAULT_TEST_CONCEPTS = [
    "fine_woolfhard",
    "butin",
    "will",
    "gold_pineapple",
    "coco",
    "nha_tho_hanoi",
    "wangkai",
    "adrien_brody",
    "bo",
    "ningning",
    "emma",
    "dunpai",
    "mam",
    "maeve_dog",
    "leonardo",
    "pig_cup",
    "skulls_mug",
    "mydieu",
    "b_jordan",
    "willinvietnam",
]


def _project_path(*parts: str) -> Path:
    return (PROJECT_ROOT / Path(*parts)).resolve()


PIPELINE_DEFAULTS: dict[str, object] = {
    "mode": "run",
    "adapter": DEFAULT_ADAPTER,
    "tasks": ["all"],
    "concepts": DEFAULT_TEST_CONCEPTS.copy(),
    "model_epochs": DEFAULT_MODEL_EPOCHS,
    "output_root": str(_project_path("evaluation_outputs")),
    "overwrite": False,
    "seed": 3407,
    "config_file": str(_project_path("configs", "showo_demo_512x512.yaml")),
    "data_root": str(_project_path("..", "dataset", "unictokens_data")),
    "prompt_file": str(_project_path("prompts_to_eval.json")),
    "token_weight_root": str(_project_path("..", "weight")),
    "rl_weight_root": str(_project_path("tmp_result", "model_weights")),
    "epoch_to_load": DEFAULT_EPOCH_TO_LOAD,
    "nums_new_token_i_stage_1": 16,
    "nums_new_token_i_stage_2": 8,
    "device": "cuda",
    "inverse_prompt": True,
    "generation_num_images_per_prompt": GENERATION_NUM_IMAGES_PER_PROMPT,
    "num_images": GENERATION_NUM_IMAGES_PER_PROMPT,
    "batch_size": None,
    "reference_image_count": 1,
    "clip_i_reference_image_count": 1,
    "mmu_top_k": 1,
    "mmu_max_new_tokens": 100,
    "allow_remote_models": False,
    "showo_model_path": None,
    "vq_model_path": None,
    "llm_model_path": None,
    "generation_guidance_scale": 5.0,
    "generation_timesteps": 50,
    "generation_temperature": 1.0,
    "generation_noise_type": "mask",
    "adapter_model_id": DEFAULT_ADAPTER_MODEL_ID,
    "adapter_code_root": DEFAULT_ADAPTER_CODE_ROOT,
    "adapter_api_key": None,
    "adapter_base_url": None,
    "adapter_max_memory_per_gpu": DEFAULT_ADAPTER_MAX_MEMORY_PER_GPU,
    "adapter_offload_dir": str(_project_path("tmp_bagel_offload")),
    "adapter_image_height": DEFAULT_ADAPTER_IMAGE_SIZE,
    "adapter_image_width": DEFAULT_ADAPTER_IMAGE_SIZE,
    "adapter_cfg_text_scale": 4.0,
    "adapter_cfg_img_scale": None,
    "adapter_cfg_interval_start": None,
    "adapter_timestep_shift": 3.0,
    "adapter_num_timesteps": 50,
    "adapter_cfg_renorm_min": 0.0,
    "adapter_cfg_renorm_type": None,
    "adapter_use_thinking": False,
    "adapter_temperature": 1e-5,
    "adapter_top_p": 1.0,
    "adapter_top_k": 50,
    "adapter_max_new_tokens": 256,
    "clip_model_path": str(_project_path("ViT-B-32.pt")),
    "gpt_model": "gpt-4.1-mini",
    "gpt_api_key": None,
    "gpt_base_url": None,
    "gpt_temperature": 1e-5,
    "gpt_max_tokens": 64,
    "gpt_timeout": 120.0,
}


PARALLEL_DEFAULTS: dict[str, object] = {
    "gpu_groups": ["0"],
    "mode": "run",
    "adapter": DEFAULT_ADAPTER,
    "tasks": ["all"],
    "concepts": DEFAULT_TEST_CONCEPTS.copy(),
    "model_epochs": DEFAULT_MODEL_EPOCHS.copy(),
    "data_root": PIPELINE_DEFAULTS["data_root"],
    "output_root": None,
    "generate_device": "cuda",
    "score_device": "cuda",
    "stagger_seconds": 3.0,
    "skip_score": False,
}


SMOKE_TEST_DEFAULTS: dict[str, object] = {
    "mode": "generate",
    "adapter": DEFAULT_SMOKE_TEST_ADAPTER,
    "tasks": ["pure_gen"],
    "concepts": ["adrien_brody"],
    "model_epochs": DEFAULT_MODEL_EPOCHS.copy(),
    "num_images": GENERATION_NUM_IMAGES_PER_PROMPT,
    "batch_size": 1,
    "reference_image_count": 1,
    "epoch_to_load": DEFAULT_EPOCH_TO_LOAD,
    "output_root": str(_project_path("smoke_test_outputs")),
}
