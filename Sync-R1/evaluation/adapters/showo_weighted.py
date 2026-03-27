from __future__ import annotations

import copy
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from models import MAGVITv2, Showo, get_mask_chedule
from pdata import image_transform as personalized_image_transform
from training.prompting_utils import (
    UniversalPrompting,
    create_attention_mask_for_mmu,
    create_attention_mask_predict_next,
)
from utils import load_single_model_weights_from_file
from ..common import (
    PromptArtifact,
    PromptSpec,
    RunManifest,
    UnderstandingArtifact,
    UnderstandingExampleSpec,
    UnderstandingRunManifest,
    build_showo_system_prompt,
    load_json,
    set_global_seed,
    stable_seed,
    write_json,
)
from .base import BaseEvalAdapter


@dataclass
class _Runtime:
    config: Any
    tokenizer: Any
    uni_prompting: Any
    vq_model: Any
    model: Any


class ShowoWeightedAdapter(BaseEvalAdapter):
    name = "showo_weighted"

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
        adapter_temperature: float = 1.0,
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
        self.device = torch.device(device)
        self.seed = seed
        self.local_files_only = local_files_only
        self.showo_model_path = showo_model_path
        self.vq_model_path = vq_model_path
        self.llm_model_path = llm_model_path
        self.guidance_scale = guidance_scale
        self.generation_timesteps = generation_timesteps
        self.generation_temperature = generation_temperature
        self.noise_type = noise_type
        self.reference_image_count = reference_image_count
        self._cache_key: tuple[str, int] | None = None
        self._cache_runtime: _Runtime | None = None

    def has_model_epoch(self, concept: str, model_epoch: int) -> bool:
        if model_epoch == 0:
            return True
        return (self.rl_weight_root / concept / f"model_epoch{model_epoch}.pt").exists()

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
        runtime = self._get_runtime(concept, model_epoch)
        output_dir = Path(output_dir)
        if overwrite and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        prompt_artifacts: list[PromptArtifact] = []
        total_prompts = len(prompt_specs)
        for prompt_idx, prompt in enumerate(prompt_specs, start=1):
            request_seed = stable_seed(self.seed, self.name, "generation", task_name, concept, model_epoch, prompt.prompt_id)
            set_global_seed(request_seed)
            prompt_dir = output_dir / prompt.prompt_id
            prompt_dir.mkdir(parents=True, exist_ok=True)
            existing_images = self._existing_image_files(prompt_dir, num_images)
            input_prompt = self._apply_tp_prefix(prompt.tp_prefix_text, prompt.generation_prompt)
            write_json(
                prompt_dir / "prompt.json",
                {
                    "prompt_id": prompt.prompt_id,
                    "source_prompt": prompt.source_prompt,
                    "generation_prompt": prompt.generation_prompt,
                    "input_prompt": input_prompt,
                    "scoring_prompt": prompt.scoring_prompt,
                    "baseline_prompt": prompt.baseline_prompt,
                    "tp_prefix_text": prompt.tp_prefix_text,
                    "conditioning_text": prompt.conditioning_text,
                    "reference_image_paths": prompt.reference_image_paths,
                    "metadata": prompt.metadata,
                    "seed": request_seed,
                },
            )
            if not overwrite and len(existing_images) >= num_images:
                image_files = existing_images[:num_images]
                print(
                    f"[resume] {self.name} | gen | {task_name} | {concept} | epoch {model_epoch} | "
                    f"{prompt_idx}/{total_prompts} | {prompt.prompt_id} | "
                    f"images {len(image_files)}/{num_images} | {self._short_log_text(input_prompt)}",
                    flush=True,
                )
            else:
                status = "resume" if existing_images and not overwrite else "start"
                print(
                    f"[{status}] {self.name} | gen | {task_name} | {concept} | epoch {model_epoch} | "
                    f"{prompt_idx}/{total_prompts} | {prompt.prompt_id} | "
                    f"images {len(existing_images)}/{num_images} | {self._short_log_text(input_prompt)}",
                    flush=True,
                )
                image_files = self._generate_prompt_images(
                    runtime=runtime,
                    prompt_text=input_prompt,
                    prompt_dir=prompt_dir,
                    num_images=num_images,
                    batch_size=batch_size,
                )
            prompt_artifacts.append(
                PromptArtifact(
                    prompt_id=prompt.prompt_id,
                    source_prompt=prompt.source_prompt,
                    generation_prompt=input_prompt,
                    scoring_prompt=prompt.scoring_prompt,
                    baseline_prompt=prompt.baseline_prompt,
                    tp_prefix_text=prompt.tp_prefix_text,
                    conditioning_text=prompt.conditioning_text,
                    reference_image_paths=prompt.reference_image_paths,
                    image_files=image_files,
                    metadata=prompt.metadata,
                )
            )
            self._write_generation_manifest(
                output_dir=output_dir,
                task_name=task_name,
                concept=concept,
                model_epoch=model_epoch,
                prompt_artifacts=prompt_artifacts,
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
        runtime = self._get_runtime(concept, model_epoch)
        output_dir = Path(output_dir)
        if overwrite and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        items: list[UnderstandingArtifact] = []
        total_examples = len(example_specs)
        for example_idx, example in enumerate(example_specs, start=1):
            request_seed = stable_seed(self.seed, self.name, "understanding", task_name, concept, model_epoch, example.item_id)
            set_global_seed(request_seed)
            item_path = output_dir / f"{example.item_id}.json"
            if not overwrite and item_path.exists():
                artifact = UnderstandingArtifact(**load_json(item_path))
                print(
                    f"[resume] {self.name} | und | {task_name} | {concept} | epoch {model_epoch} | "
                    f"{example_idx}/{total_examples} | {example.item_id} | {self._short_log_text(example.scoring_query)}",
                    flush=True,
                )
            else:
                print(
                    f"[run] {self.name} | und | {task_name} | {concept} | epoch {model_epoch} | "
                    f"{example_idx}/{total_examples} | {example.item_id} | {self._short_log_text(example.scoring_query)}",
                    flush=True,
                )
                input_prompt = self._apply_tp_prefix(example.tp_prefix_text, example.model_prompt)
                prediction = self._predict_text(
                    runtime=runtime,
                    concept=concept,
                    example=example,
                    prompt_text=input_prompt,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
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
                    tp_prefix_text=example.tp_prefix_text,
                    conditioning_text=example.conditioning_text,
                    reference_image_paths=example.reference_image_paths,
                    prepend_system_prompt=example.prepend_system_prompt,
                    metadata=example.metadata,
                )
                write_json(item_path, artifact.__dict__)
            items.append(artifact)
            self._write_understanding_manifest(
                output_dir=output_dir,
                task_name=task_name,
                concept=concept,
                model_epoch=model_epoch,
                items=items,
            )

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

    def _get_runtime(self, concept: str, model_epoch: int) -> _Runtime:
        cache_key = (concept, model_epoch)
        if self._cache_key == cache_key and self._cache_runtime is not None:
            return self._cache_runtime

        runtime = self._load_runtime(concept, model_epoch)
        self._cache_key = cache_key
        self._cache_runtime = runtime
        return runtime

    def _existing_image_files(self, prompt_dir: Path, num_images: int) -> list[str]:
        image_files: list[str] = []
        for image_idx in range(num_images):
            image_name = f"image_{image_idx:03d}.png"
            if (prompt_dir / image_name).exists():
                image_files.append(image_name)
        return image_files

    def _write_generation_manifest(
        self,
        *,
        output_dir: Path,
        task_name: str,
        concept: str,
        model_epoch: int,
        prompt_artifacts: Sequence[PromptArtifact],
    ) -> None:
        manifest = RunManifest(
            task=task_name,
            family="generation",
            concept=concept,
            token_epoch=self.epoch_to_load,
            model_epoch=model_epoch,
            adapter=self.name,
            prompt_artifacts=list(prompt_artifacts),
        )
        write_json(output_dir / "manifest.json", manifest.to_dict())

    def _write_understanding_manifest(
        self,
        *,
        output_dir: Path,
        task_name: str,
        concept: str,
        model_epoch: int,
        items: Sequence[UnderstandingArtifact],
    ) -> None:
        manifest = UnderstandingRunManifest(
            task=task_name,
            family="understanding",
            concept=concept,
            token_epoch=self.epoch_to_load,
            model_epoch=model_epoch,
            adapter=self.name,
            items=list(items),
        )
        write_json(output_dir / "manifest.json", manifest.to_dict())

    def _short_log_text(self, text: str | None, width: int = 140) -> str:
        if not text:
            return "-"
        normalized = " ".join(str(text).split())
        return shorten(normalized, width=width, placeholder="...")

    def _apply_tp_prefix(self, prefix_text: str | None, prompt_text: str) -> str:
        if not prefix_text:
            return prompt_text
        return f"{prefix_text}\n{prompt_text}"

    def _load_runtime(self, concept: str, model_epoch: int) -> _Runtime:
        config = OmegaConf.load(self.config_file)
        if self.showo_model_path:
            config.model.showo.pretrained_model_path = self.showo_model_path
        if self.vq_model_path:
            config.model.vq_model.vq_model_name = self.vq_model_path
        if self.llm_model_path:
            config.model.showo.llm_model_path = self.llm_model_path

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.showo.llm_model_path,
            padding_side="left",
            local_files_only=self.local_files_only,
        )
        uni_prompting = UniversalPrompting(
            tokenizer,
            max_text_len=config.dataset.preprocessing.max_seq_length,
            special_tokens=(
                "<|soi|>",
                "<|eoi|>",
                "<|sov|>",
                "<|eov|>",
                "<|t2i|>",
                "<|mmu|>",
                "<|t2v|>",
                "<|v2v|>",
                "<|lvg|>",
            ),
            ignore_id=-100,
            cond_dropout_prob=config.training.cond_dropout_prob,
        )

        vq_model = MAGVITv2.from_pretrained(
            config.model.vq_model.vq_model_name,
            local_files_only=self.local_files_only,
        ).to(self.device)
        vq_model.requires_grad_(False)
        vq_model.eval()

        model = Showo.from_pretrained(
            config.model.showo.pretrained_model_path,
            low_cpu_mem_usage=False,
            local_files_only=self.local_files_only,
        ).to(self.device)
        model.eval()

        self._inject_personalized_weights(concept, tokenizer, model)

        if model_epoch != 0:
            weight_path = self.rl_weight_root / concept / f"model_epoch{model_epoch}.pt"
            model, _ = load_single_model_weights_from_file(
                model,
                weight_file_path=str(weight_path),
                device=str(self.device),
            )

        config.mode = "t2i"
        config.batch_size = 1
        config.generation_timesteps = self.generation_timesteps
        config.guidance_scale = self.guidance_scale
        config.training.batch_size = config.batch_size
        config.training.guidance_scale = config.guidance_scale
        config.training.generation_timesteps = config.generation_timesteps
        config.training.generation_temperature = self.generation_temperature
        config.training.noise_type = self.noise_type
        config.model.showo.llm_vocab_size = len(tokenizer) - 10

        return _Runtime(
            config=config,
            tokenizer=tokenizer,
            uni_prompting=uni_prompting,
            vq_model=vq_model,
            model=model,
        )

    def _inject_personalized_weights(self, concept: str, tokenizer: Any, model: Any) -> None:
        ckpt_dir = self.token_weight_root / concept
        ckpt_embed_path = ckpt_dir / f"epoch_{self.epoch_to_load}_embed.pt"
        ckpt_lm_head_weight_path = ckpt_dir / f"epoch_{self.epoch_to_load}_lm_head_weight.pt"
        ckpt_lm_head_bias_path = ckpt_dir / f"epoch_{self.epoch_to_load}_lm_head_bias.pt"

        total_new_token_count = self.nums_new_token_i_stage_1 + self.nums_new_token_i_stage_2
        new_tokens_total = [f"<{concept}>"] + [f"<token_{idx}>" for idx in range(total_new_token_count)]
        num_new_tokens_total = len(new_tokens_total)

        original_text_vocab_size = len(tokenizer)
        original_image_vocab_size = model.showo.get_input_embeddings().num_embeddings - len(tokenizer)
        original_total_vocab = original_text_vocab_size + original_image_vocab_size
        new_text_vocab_size = original_text_vocab_size + num_new_tokens_total
        new_total_vocab = original_total_vocab + num_new_tokens_total

        tokenizer.add_tokens(new_tokens_total)

        with torch.no_grad():
            embeddings = model.showo.get_input_embeddings().weight.data
            model.showo.resize_token_embeddings(new_total_vocab)

            original_image_weights = embeddings[original_text_vocab_size:original_total_vocab].clone()
            model.showo.get_input_embeddings().weight.data[new_text_vocab_size:new_total_vocab] = original_image_weights

            if not ckpt_embed_path.exists():
                raise FileNotFoundError(f"Embedding weights do not exist: {ckpt_embed_path}")
            ckpt_embed_weight = torch.load(ckpt_embed_path)
            model.showo.get_input_embeddings().weight.data[original_text_vocab_size:new_text_vocab_size] = (
                ckpt_embed_weight.to(model.showo.get_input_embeddings().weight.device)
            )

            if model.showo.lm_head.weight.data.shape[0] != new_total_vocab:
                raise ValueError("lm_head weights do not match the resized vocabulary")

            lm_head = model.showo.lm_head
            new_lm_head = torch.nn.Linear(
                lm_head.in_features,
                new_total_vocab,
                bias=hasattr(lm_head, "bias"),
            )
            new_lm_head.weight.data = lm_head.weight.data.clone()
            new_lm_head.weight.data[new_text_vocab_size:new_total_vocab] = (
                lm_head.weight.data[original_text_vocab_size:original_total_vocab]
            )

            if not ckpt_lm_head_weight_path.exists():
                raise FileNotFoundError(f"lm_head weights do not exist: {ckpt_lm_head_weight_path}")
            ckpt_lm_head_weight = torch.load(ckpt_lm_head_weight_path)
            new_lm_head.weight.data[original_text_vocab_size:new_text_vocab_size] = (
                ckpt_lm_head_weight.to(new_lm_head.weight.device)
            )

            if hasattr(lm_head, "bias"):
                new_lm_head.bias.data = lm_head.bias.data.clone()
                new_lm_head.bias.data[new_text_vocab_size:new_total_vocab] = (
                    lm_head.bias.data[original_text_vocab_size:original_total_vocab]
                )

                if not ckpt_lm_head_bias_path.exists():
                    raise FileNotFoundError(f"lm_head bias does not exist: {ckpt_lm_head_bias_path}")
                ckpt_lm_head_bias = torch.load(ckpt_lm_head_bias_path)
                new_lm_head.bias.data[original_text_vocab_size:new_text_vocab_size] = (
                    ckpt_lm_head_bias.to(new_lm_head.weight.device)
                )

            model.showo.lm_head = new_lm_head

    def _generate_prompt_images(
        self,
        *,
        runtime: _Runtime,
        prompt_text: str,
        prompt_dir: Path,
        num_images: int,
        batch_size: int,
    ) -> list[str]:
        config = copy.deepcopy(runtime.config)
        model = runtime.model
        vq_model = runtime.vq_model
        uni_prompting = runtime.uni_prompting

        model.config.mask_token_id = model.showo.get_input_embeddings().num_embeddings - 1
        mask_token_id = model.showo.get_input_embeddings().num_embeddings - 1

        if config.get("mask_schedule", None) is not None:
            schedule = config.mask_schedule.schedule
            schedule_args = config.mask_schedule.get("params", {})
            mask_schedule = get_mask_chedule(schedule, **schedule_args)
        else:
            mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

        generated_count = 0
        while generated_count < num_images and (prompt_dir / f"image_{generated_count:03d}.png").exists():
            generated_count += 1

        image_files = [f"image_{image_idx:03d}.png" for image_idx in range(generated_count)]

        while generated_count < num_images:
            actual_batch_size = min(batch_size, num_images - generated_count)
            image_tokens = torch.ones(
                (actual_batch_size, config.model.showo.num_vq_tokens),
                dtype=torch.long,
                device=self.device,
            ) * mask_token_id
            conditions = [prompt_text] * actual_batch_size
            input_ids, _ = uni_prompting((conditions, image_tokens), "t2i_gen")

            if config.training.guidance_scale > 0:
                uncond_input_ids, _ = uni_prompting(([""] * actual_batch_size, image_tokens), "t2i_gen")
                attention_mask = create_attention_mask_predict_next(
                    torch.cat([input_ids, uncond_input_ids], dim=0),
                    pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
                    soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
                    eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
                    rm_pad_in_image=True,
                )
            else:
                uncond_input_ids = None
                attention_mask = create_attention_mask_predict_next(
                    input_ids,
                    pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
                    soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
                    eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
                    rm_pad_in_image=True,
                )

            with torch.no_grad():
                gen_token_ids = model.t2i_generate(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    guidance_scale=config.training.guidance_scale,
                    temperature=config.training.get("generation_temperature", self.generation_temperature),
                    timesteps=config.training.generation_timesteps,
                    noise_schedule=mask_schedule,
                    noise_type=config.training.get("noise_type", self.noise_type),
                    seq_len=config.model.showo.num_vq_tokens,
                    uni_prompting=uni_prompting,
                    config=config,
                )

            gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
            images = vq_model.decode_code(gen_token_ids)
            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            images = (images * 255.0).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

            for batch_idx, image_array in enumerate(images):
                image_name = f"image_{generated_count + batch_idx:03d}.png"
                Image.fromarray(image_array).save(prompt_dir / image_name)
                image_files.append(image_name)

            generated_count += actual_batch_size

        return image_files

    def _predict_text(
        self,
        *,
        runtime: _Runtime,
        concept: str,
        example: UnderstandingExampleSpec,
        prompt_text: str,
        top_k: int,
        max_new_tokens: int,
    ) -> str:
        config = runtime.config
        uni_prompting = runtime.uni_prompting
        vq_model = runtime.vq_model
        model = runtime.model
        if example.prepend_system_prompt:
            prompt_text = self._apply_tp_prefix(
                example.tp_prefix_text,
                build_showo_system_prompt(concept, self.nums_new_token_i_stage_1) + example.model_prompt,
            )

        image = Image.open(example.image_path).convert("RGB")
        image = personalized_image_transform(
            image,
            resolution=config.dataset.params.resolution,
        ).to(self.device)
        image = image.unsqueeze(0)

        with torch.no_grad():
            image_tokens_mmu = vq_model.get_code(image)
            image_tokens = image_tokens_mmu + len(uni_prompting.text_tokenizer)

            input_ids = uni_prompting.text_tokenizer(
                [f"USER: {prompt_text} ASSISTANT:"]
            )["input_ids"]
            input_ids = torch.tensor(input_ids).to(self.device)

            input_ids = torch.cat(
                [
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict["<|mmu|>"]).to(self.device),
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict["<|soi|>"]).to(self.device),
                    image_tokens,
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict["<|eoi|>"]).to(self.device),
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict["<|sot|>"]).to(self.device),
                    input_ids,
                ],
                dim=1,
            ).long()

            attention_mask = create_attention_mask_for_mmu(
                input_ids.to(self.device),
                eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
            )

            cont_toks_list = model.mmu_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                eot_token=uni_prompting.sptids_dict["<|eot|>"],
            )

        cont_toks = torch.stack(cont_toks_list).squeeze()[None]
        return uni_prompting.text_tokenizer.batch_decode(cont_toks, skip_special_tokens=True)[0].strip()
