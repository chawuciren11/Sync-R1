from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

from ..common import (
    PromptArtifact,
    PromptSpec,
    RunManifest,
    UnderstandingArtifact,
    UnderstandingExampleSpec,
    UnderstandingRunManifest,
    set_global_seed,
    stable_seed,
    write_json,
)
from .base import BaseEvalAdapter


class ExternalBaselineAdapter(BaseEvalAdapter):
    model_family = "external"
    prompt_mode = "tp"

    def __init__(
        self,
        *,
        config_file: str | Path,
        token_weight_root: str | Path,
        rl_weight_root: str | Path,
        epoch_to_load: int,
        nums_new_token_i_stage_1: int,
        nums_new_token_i_stage_2: int,
        device: str,
        seed: int = 3407,
        local_files_only: bool = True,
        showo_model_path: str | None = None,
        vq_model_path: str | None = None,
        llm_model_path: str | None = None,
        guidance_scale: float = 5.0,
        generation_timesteps: int = 50,
        generation_temperature: float = 1.0,
        noise_type: str = "mask",
        model_id: str | None = None,
        code_root: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_memory_per_gpu: str | None = None,
        offload_dir: str | None = None,
        image_height: int = 1024,
        image_width: int = 1024,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float | None = None,
        cfg_interval_start: float | None = None,
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str | None = None,
        use_thinking: bool = False,
        adapter_temperature: float = 1e-5,
        adapter_top_p: float = 1.0,
        adapter_top_k: int = 50,
        adapter_max_new_tokens: int = 256,
        reference_image_count: int = 1,
    ) -> None:
        self.config_file = Path(config_file)
        self.token_weight_root = Path(token_weight_root)
        self.rl_weight_root = Path(rl_weight_root)
        self.epoch_to_load = epoch_to_load
        self.nums_new_token_i_stage_1 = nums_new_token_i_stage_1
        self.nums_new_token_i_stage_2 = nums_new_token_i_stage_2
        self.device = device
        self.seed = seed
        self.local_files_only = local_files_only
        self.showo_model_path = showo_model_path
        self.vq_model_path = vq_model_path
        self.llm_model_path = llm_model_path
        self.guidance_scale = guidance_scale
        self.generation_timesteps = generation_timesteps
        self.generation_temperature = generation_temperature
        self.noise_type = noise_type
        self.model_id = model_id
        self.code_root = Path(code_root).expanduser() if code_root else None
        self.api_key = api_key
        self.base_url = base_url
        self.max_memory_per_gpu = max_memory_per_gpu
        self.offload_dir = Path(offload_dir).expanduser() if offload_dir else None
        self.image_height = image_height
        self.image_width = image_width
        self.cfg_text_scale = cfg_text_scale
        self.cfg_img_scale = cfg_img_scale
        self.cfg_interval_start = cfg_interval_start
        self.timestep_shift = timestep_shift
        self.num_timesteps = num_timesteps
        self.cfg_renorm_min = cfg_renorm_min
        self.cfg_renorm_type = cfg_renorm_type
        self.use_thinking = use_thinking
        self.adapter_temperature = adapter_temperature
        self.adapter_top_p = adapter_top_p
        self.adapter_top_k = adapter_top_k
        self.adapter_max_new_tokens = adapter_max_new_tokens
        self.reference_image_count = reference_image_count
        self.name = f"{self.model_family}_{self.prompt_mode}"

    def has_model_epoch(self, concept: str, model_epoch: int) -> bool:
        return True

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
        output_dir = Path(output_dir)
        if overwrite and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        prompt_artifacts: list[PromptArtifact] = []
        for prompt_spec in prompt_specs:
            request_seed = stable_seed(
                self.seed,
                self.name,
                "generation",
                task_name,
                concept,
                model_epoch,
                prompt_spec.prompt_id,
            )
            set_global_seed(request_seed)

            prompt_dir = output_dir / prompt_spec.prompt_id
            prompt_dir.mkdir(parents=True, exist_ok=True)
            input_prompt = self._select_generation_prompt(prompt_spec)
            reference_images = self._select_reference_images(prompt_spec.reference_image_paths)

            write_json(
                prompt_dir / "prompt.json",
                {
                    "prompt_id": prompt_spec.prompt_id,
                    "source_prompt": prompt_spec.source_prompt,
                    "generation_prompt": prompt_spec.generation_prompt,
                    "baseline_prompt": prompt_spec.baseline_prompt,
                    "input_prompt": input_prompt,
                    "conditioning_text": prompt_spec.conditioning_text,
                    "reference_image_paths": reference_images,
                    "scoring_prompt": prompt_spec.scoring_prompt,
                    "prompt_mode": self.prompt_mode,
                    "metadata": prompt_spec.metadata,
                    "seed": request_seed,
                },
            )

            image_files = self._generate_images_for_prompt(
                task_name=task_name,
                concept=concept,
                prompt_spec=prompt_spec,
                input_prompt=input_prompt,
                reference_image_paths=reference_images,
                prompt_dir=prompt_dir,
                model_epoch=model_epoch,
                num_images=num_images,
                batch_size=batch_size,
                seed=request_seed,
            )
            prompt_artifacts.append(
                PromptArtifact(
                    prompt_id=prompt_spec.prompt_id,
                    source_prompt=prompt_spec.source_prompt,
                    generation_prompt=input_prompt,
                    scoring_prompt=prompt_spec.scoring_prompt,
                    baseline_prompt=prompt_spec.baseline_prompt,
                    conditioning_text=prompt_spec.conditioning_text,
                    reference_image_paths=reference_images,
                    image_files=image_files,
                    metadata={
                        **prompt_spec.metadata,
                        "prompt_mode": self.prompt_mode,
                        "model_family": self.model_family,
                        "seed": request_seed,
                    },
                )
            )

        manifest = RunManifest(
            task=task_name,
            family="generation",
            concept=concept,
            token_epoch=self.epoch_to_load,
            model_epoch=model_epoch,
            adapter=self.name,
            prompt_artifacts=prompt_artifacts,
        )
        write_json(output_dir / "manifest.json", manifest.to_dict())
        return manifest

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
        output_dir = Path(output_dir)
        if overwrite and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        items: list[UnderstandingArtifact] = []
        for example in example_specs:
            request_seed = stable_seed(
                self.seed,
                self.name,
                "understanding",
                task_name,
                concept,
                model_epoch,
                example.item_id,
            )
            set_global_seed(request_seed)
            input_prompt = self._select_understanding_prompt(example)
            reference_images = self._select_reference_images(example.reference_image_paths)
            prediction = self._predict_text_for_example(
                task_name=task_name,
                concept=concept,
                example=example,
                input_prompt=input_prompt,
                reference_image_paths=reference_images,
                model_epoch=model_epoch,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                seed=request_seed,
            )
            artifact = UnderstandingArtifact(
                item_id=example.item_id,
                source_prompt=example.source_prompt,
                model_prompt=input_prompt,
                scoring_query=example.scoring_query,
                ground_truth=example.ground_truth,
                image_path=example.image_path,
                prediction=prediction,
                baseline_prompt=example.baseline_prompt,
                conditioning_text=example.conditioning_text,
                reference_image_paths=reference_images,
                prepend_system_prompt=example.prepend_system_prompt,
                metadata={
                    **example.metadata,
                    "prompt_mode": self.prompt_mode,
                    "model_family": self.model_family,
                    "seed": request_seed,
                },
            )
            items.append(artifact)
            write_json(output_dir / f"{example.item_id}.json", artifact.__dict__)

        manifest = UnderstandingRunManifest(
            task=task_name,
            family="understanding",
            concept=concept,
            token_epoch=self.epoch_to_load,
            model_epoch=model_epoch,
            adapter=self.name,
            items=items,
        )
        write_json(output_dir / "manifest.json", manifest.to_dict())
        return manifest

    def _select_generation_prompt(self, prompt_spec: PromptSpec) -> str:
        if self.prompt_mode == "ip":
            return prompt_spec.baseline_prompt or prompt_spec.scoring_prompt or prompt_spec.source_prompt
        return prompt_spec.baseline_prompt or prompt_spec.scoring_prompt or prompt_spec.generation_prompt

    def _select_understanding_prompt(self, example: UnderstandingExampleSpec) -> str:
        if self.prompt_mode == "ip":
            return example.baseline_prompt or example.source_prompt
        return example.baseline_prompt or example.model_prompt

    def _select_reference_images(self, reference_image_paths: Sequence[str]) -> list[str]:
        if self.prompt_mode != "ip":
            return []
        selected = list(reference_image_paths[: self.reference_image_count])
        if not selected:
            raise ValueError(f"{self.name} requires reference_image_paths for IP evaluation")
        return selected

    def _generate_images_for_prompt(
        self,
        *,
        task_name: str,
        concept: str,
        prompt_spec: PromptSpec,
        input_prompt: str,
        reference_image_paths: Sequence[str],
        prompt_dir: Path,
        model_epoch: int,
        num_images: int,
        batch_size: int,
        seed: int,
    ) -> list[str]:
        raise NotImplementedError(
            f"{self.name} is registered, but image generation is not implemented yet. "
            f"Add the model call in {Path(__file__).name} or a dedicated adapter file."
        )

    def _predict_text_for_example(
        self,
        *,
        task_name: str,
        concept: str,
        example: UnderstandingExampleSpec,
        input_prompt: str,
        reference_image_paths: Sequence[str],
        model_epoch: int,
        top_k: int,
        max_new_tokens: int,
        seed: int,
    ) -> str:
        raise NotImplementedError(
            f"{self.name} is registered, but understanding inference is not implemented yet. "
            f"Add the model call in {Path(__file__).name} or a dedicated adapter file."
        )
