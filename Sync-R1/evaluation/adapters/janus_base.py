from __future__ import annotations

import gc
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from ..common import (
    PromptArtifact,
    PromptSpec,
    RunManifest,
    UnderstandingExampleSpec,
    set_global_seed,
    stable_seed,
    write_json,
)
from .external_baseline import ExternalBaselineAdapter


@dataclass
class _JanusRuntime:
    model: Any
    processor: Any
    tokenizer: Any
    load_pil_images: Any
    device: torch.device


class JanusAdapterBase(ExternalBaselineAdapter):
    model_family = "janus"

    _MODEL_ALIASES = {
        "januspro1b": "deepseek-ai/Janus-Pro-1B",
        "janus-pro-1b": "deepseek-ai/Janus-Pro-1B",
        "deepseek-ai/janus-pro-1b": "deepseek-ai/Janus-Pro-1B",
        "januspro7b": "deepseek-ai/Janus-Pro-7B",
        "janus-pro-7b": "deepseek-ai/Janus-Pro-7B",
        "deepseek-ai/janus-pro-7b": "deepseek-ai/Janus-Pro-7B",
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._runtime: _JanusRuntime | None = None
        self._reference_description_cache: dict[tuple[str, ...], str] = {}
        if self.image_height == 1024 and self.image_width == 1024:
            # Janus-Pro generation defaults to 384x384; keep the global adapter
            # defaults from breaking Janus runs unless the user explicitly overrides.
            self.image_height = 384
            self.image_width = 384

    def has_model_epoch(self, concept: str, model_epoch: int) -> bool:
        del concept
        return model_epoch == 0

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
            import shutil

            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        prompt_artifacts: list[PromptArtifact] = []
        total_prompts = len(prompt_specs)
        for prompt_idx, prompt_spec in enumerate(prompt_specs, start=1):
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
            selected_reference_images = self._select_reference_images(prompt_spec.reference_image_paths)
            prompt_payload_path = prompt_dir / "prompt.json"
            existing_prompt_payload = None
            if not overwrite and prompt_payload_path.exists():
                existing_prompt_payload = self._load_prompt_payload(prompt_payload_path)
            if existing_prompt_payload is not None:
                input_prompt = str(existing_prompt_payload.get("input_prompt", "")).strip()
                reference_caption = existing_prompt_payload.get("reference_caption")
                reference_images = list(
                    existing_prompt_payload.get("reference_image_paths", selected_reference_images)
                )
                if not input_prompt:
                    input_prompt, reference_caption = self._prepare_generation_prompt(
                        prompt_spec=prompt_spec,
                        reference_image_paths=selected_reference_images,
                    )
                    reference_images = list(selected_reference_images)
            else:
                input_prompt, reference_caption = self._prepare_generation_prompt(
                    prompt_spec=prompt_spec,
                    reference_image_paths=selected_reference_images,
                )
                reference_images = list(selected_reference_images)

            existing_images = self._existing_image_files(prompt_dir, num_images)
            prompt_payload = {
                "prompt_id": prompt_spec.prompt_id,
                "source_prompt": prompt_spec.source_prompt,
                "generation_prompt": prompt_spec.generation_prompt,
                "baseline_prompt": prompt_spec.baseline_prompt,
                "tp_prefix_text": prompt_spec.tp_prefix_text,
                "input_prompt": input_prompt,
                "conditioning_text": prompt_spec.conditioning_text,
                "reference_image_paths": reference_images,
                "reference_caption": reference_caption,
                "scoring_prompt": prompt_spec.scoring_prompt,
                "prompt_mode": self.prompt_mode,
                "metadata": prompt_spec.metadata,
                "seed": request_seed,
            }
            write_json(prompt_payload_path, prompt_payload)

            if not overwrite and len(existing_images) >= num_images:
                image_files = existing_images[:num_images]
                print(
                    f"[resume] {self.name} | gen | {task_name} | {concept} | epoch {model_epoch} | "
                    f"{prompt_idx}/{total_prompts} | {prompt_spec.prompt_id} | "
                    f"images {len(image_files)}/{num_images} | {self._short_log_text(input_prompt)}",
                    flush=True,
                )
            else:
                status = "resume" if existing_images and not overwrite else "start"
                print(
                    f"[{status}] {self.name} | gen | {task_name} | {concept} | epoch {model_epoch} | "
                    f"{prompt_idx}/{total_prompts} | {prompt_spec.prompt_id} | "
                    f"images {len(existing_images)}/{num_images} | {self._short_log_text(input_prompt)}",
                    flush=True,
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

            artifact_metadata = {
                **prompt_spec.metadata,
                "prompt_mode": self.prompt_mode,
                "model_family": self.model_family,
                "seed": request_seed,
            }
            if reference_caption:
                artifact_metadata["reference_caption"] = reference_caption

            prompt_artifacts.append(
                PromptArtifact(
                    prompt_id=prompt_spec.prompt_id,
                    source_prompt=prompt_spec.source_prompt,
                    generation_prompt=input_prompt,
                    scoring_prompt=prompt_spec.scoring_prompt,
                    baseline_prompt=prompt_spec.baseline_prompt,
                    tp_prefix_text=prompt_spec.tp_prefix_text,
                    conditioning_text=prompt_spec.conditioning_text,
                    reference_image_paths=reference_images,
                    image_files=image_files,
                    metadata=artifact_metadata,
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

    def _select_understanding_prompt(self, example: UnderstandingExampleSpec) -> str:
        prompt_text = super()._select_understanding_prompt(example)
        return self._join_text_segments(example.conditioning_text, prompt_text)

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
        del task_name, concept, prompt_spec, reference_image_paths, model_epoch

        if self.image_height % 16 != 0 or self.image_width % 16 != 0:
            raise ValueError(
                f"{self.name} expects --adapter-image-height/width to be divisible by 16; "
                f"got {self.image_height}x{self.image_width}."
            )

        image_files: list[str] = []
        pending_indices: list[int] = []
        for image_idx in range(num_images):
            image_name = f"image_{image_idx:03d}.png"
            image_path = prompt_dir / image_name
            if image_path.exists():
                image_files.append(image_name)
            else:
                pending_indices.append(image_idx)

        batch_size = max(1, batch_size)
        for batch_start in range(0, len(pending_indices), batch_size):
            batch_indices = pending_indices[batch_start : batch_start + batch_size]
            batch_seed = stable_seed(seed, "janus_image_batch", batch_start)
            set_global_seed(batch_seed)
            images = self._run_generation_batch(
                prompt=input_prompt,
                parallel_size=len(batch_indices),
            )
            for image_idx, image in zip(batch_indices, images):
                image_name = f"image_{image_idx:03d}.png"
                image.save(prompt_dir / image_name)
            self._cleanup_cuda()

        return self._existing_image_files(prompt_dir, num_images)

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
        del task_name, concept, model_epoch, top_k

        set_global_seed(stable_seed(seed, "janus_text"))
        image_paths = list(reference_image_paths)
        if self._should_use_target_image(example.image_path):
            image_paths.append(example.image_path)
        prediction = self._run_understanding_chat(
            prompt_text=input_prompt,
            image_paths=image_paths,
            max_new_tokens=max_new_tokens,
        )
        self._cleanup_cuda()
        return prediction

    def _prepare_generation_prompt(
        self,
        *,
        prompt_spec: PromptSpec,
        reference_image_paths: Sequence[str],
    ) -> tuple[str, str | None]:
        prompt_text = super()._select_generation_prompt(prompt_spec)
        prompt_text = self._join_text_segments(prompt_spec.conditioning_text, prompt_text)
        reference_caption = None
        if self.prompt_mode == "ip" and reference_image_paths:
            reference_caption = self._describe_reference_images(reference_image_paths)
            prompt_text = self._join_text_segments(
                prompt_text,
                f"Reference appearance details: {reference_caption}",
            )
        return prompt_text, reference_caption

    def _describe_reference_images(self, reference_image_paths: Sequence[str]) -> str:
        key = tuple(str(Path(path)) for path in reference_image_paths)
        cached = self._reference_description_cache.get(key)
        if cached is not None:
            return cached

        description = self._run_understanding_chat(
            prompt_text=(
                "Describe the shared subject in these reference images in one concise sentence. "
                "Focus on appearance, clothing, accessories, colors, and distinctive attributes. "
                "Output only the description."
            ),
            image_paths=reference_image_paths,
            max_new_tokens=min(self.adapter_max_new_tokens, 128),
        )
        description = self._clean_text_output(description)
        self._reference_description_cache[key] = description
        return description

    def _run_understanding_chat(
        self,
        *,
        prompt_text: str,
        image_paths: Sequence[str],
        max_new_tokens: int,
    ) -> str:
        runtime = self._get_runtime()
        conversation = self._build_conversation(prompt_text=prompt_text, image_paths=image_paths)
        pil_images = runtime.load_pil_images(conversation) if image_paths else []
        prepare_inputs = runtime.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
        )
        if hasattr(prepare_inputs, "to"):
            prepare_inputs = prepare_inputs.to(runtime.device)
        inputs_embeds = runtime.model.prepare_inputs_embeds(**prepare_inputs)
        attention_mask = getattr(prepare_inputs, "attention_mask", None)
        if attention_mask is None:
            attention_mask = prepare_inputs["attention_mask"]
        outputs = runtime.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=runtime.tokenizer.eos_token_id,
            bos_token_id=runtime.tokenizer.bos_token_id,
            eos_token_id=runtime.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens or self.adapter_max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        generated = outputs[0]
        input_ids = None
        try:
            input_ids = prepare_inputs["input_ids"]
        except Exception:
            input_ids = getattr(prepare_inputs, "input_ids", None)
        if input_ids is not None and outputs.ndim == 2 and outputs.shape[1] > input_ids.shape[1]:
            generated = outputs[0, input_ids.shape[1] :]
        text = runtime.tokenizer.decode(generated.detach().cpu().tolist(), skip_special_tokens=True).strip()
        if not text:
            text = runtime.tokenizer.decode(outputs[0].detach().cpu().tolist(), skip_special_tokens=True).strip()
        try:
            sft_format = prepare_inputs["sft_format"][0]
        except Exception:
            sft_format = None
        if sft_format and text.startswith(sft_format):
            text = text[len(sft_format) :].strip()
        return self._clean_text_output(text)

    @torch.inference_mode()
    def _run_generation_batch(self, *, prompt: str, parallel_size: int) -> list[Image.Image]:
        runtime = self._get_runtime()
        prompt = prompt.strip()
        conversation = [
            {"role": "<|User|>", "content": prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = runtime.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=runtime.processor.sft_format,
            system_prompt="",
        )
        prompt_tokens = runtime.processor.tokenizer.encode(sft_format + runtime.processor.image_start_tag)
        device = runtime.device
        tokens = torch.zeros((parallel_size * 2, len(prompt_tokens)), dtype=torch.long, device=device)
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
        for row_idx in range(parallel_size * 2):
            tokens[row_idx, :] = prompt_tensor
            if row_idx % 2 == 1 and tokens.shape[1] > 2:
                tokens[row_idx, 1:-1] = runtime.processor.pad_id

        inputs_embeds = runtime.model.language_model.get_input_embeddings()(tokens)
        language_backbone = getattr(runtime.model.language_model, "model", runtime.model.language_model)
        h_tokens = self.image_height // 16
        w_tokens = self.image_width // 16
        image_token_count = h_tokens * w_tokens
        generated_tokens = torch.zeros((parallel_size, image_token_count), dtype=torch.long, device=device)

        past_key_values = None
        for token_idx in range(image_token_count):
            outputs = language_backbone(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            logits = runtime.model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            cfg_weight = float(self.cfg_text_scale)
            guided_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

            temperature = float(self.generation_temperature)
            if temperature <= 0:
                next_token = torch.argmax(guided_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(guided_logits / max(temperature, 1e-5), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, token_idx] = next_token.squeeze(-1)

            duplicated = torch.cat([next_token, next_token], dim=1).view(-1)
            img_embeds = runtime.model.prepare_gen_img_embeds(duplicated)
            inputs_embeds = img_embeds.unsqueeze(1)

        decoded = runtime.model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, h_tokens, w_tokens],
        )
        decoded = decoded.to(torch.float32).detach().cpu().numpy().transpose(0, 2, 3, 1)
        decoded = np.clip((decoded + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
        return [Image.fromarray(array) for array in decoded]

    def _build_conversation(self, *, prompt_text: str, image_paths: Sequence[str]) -> list[dict[str, Any]]:
        content_parts: list[str] = []
        if image_paths:
            content_parts.extend(["<image_placeholder>"] * len(image_paths))
        content_parts.append(prompt_text.strip())
        user_turn: dict[str, Any] = {
            "role": "<|User|>",
            "content": "\n".join(part for part in content_parts if part),
        }
        if image_paths:
            user_turn["images"] = [str(path) for path in image_paths]
        return [
            user_turn,
            {"role": "<|Assistant|>", "content": ""},
        ]

    def _load_prompt_payload(self, prompt_payload_path: Path) -> dict[str, Any] | None:
        try:
            from ..common import load_json

            return load_json(prompt_payload_path)
        except Exception:
            return None

    def _join_text_segments(self, *segments: str | None) -> str:
        parts = [str(segment).strip() for segment in segments if segment and str(segment).strip()]
        return "\n".join(parts)

    def _clean_text_output(self, text: str) -> str:
        cleaned = text.strip()
        markers = [
            "<|Assistant|>",
            "<｜Assistant｜>",
            "Assistant:",
        ]
        for marker in markers:
            if marker in cleaned:
                cleaned = cleaned.split(marker)[-1].strip()
        return cleaned

    def _get_runtime(self) -> _JanusRuntime:
        if self._runtime is None:
            self._runtime = self._load_runtime()
        return self._runtime

    def _load_runtime(self) -> _JanusRuntime:
        model_id = self._resolve_model_id()
        self._ensure_janus_import_path(model_id)

        try:
            janus_models = importlib.import_module("janus.models")
            janus_io = importlib.import_module("janus.utils.io")
        except ImportError as exc:
            raise ImportError(
                f"{self.name} could not import DeepSeek Janus modules. "
                "Install Janus or pass --adapter-code-root to the Janus repo root."
            ) from exc

        if not hasattr(janus_models, "VLChatProcessor") or not hasattr(janus_io, "load_pil_images"):
            raise ImportError(
                f"{self.name} found a module named 'janus', but it does not look like DeepSeek Janus. "
                "Check your environment or pass --adapter-code-root to the correct Janus source tree."
            )

        processor = janus_models.VLChatProcessor.from_pretrained(
            model_id,
            local_files_only=self.local_files_only,
        )

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
        }
        dtype = self._preferred_dtype()
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = dtype
            if self.max_memory_per_gpu:
                offload_dir = self._resolve_offload_dir()
                offload_dir.mkdir(parents=True, exist_ok=True)
                model_kwargs["device_map"] = "auto"
                model_kwargs["max_memory"] = self._build_max_memory_map()
                model_kwargs["offload_folder"] = str(offload_dir)

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs).eval()

        if "device_map" not in model_kwargs:
            if torch.cuda.is_available():
                model = model.to(dtype=dtype)
            model = model.to(self._requested_device()).eval()

        runtime_device = self._resolve_runtime_device(model)
        return _JanusRuntime(
            model=model,
            processor=processor,
            tokenizer=processor.tokenizer,
            load_pil_images=janus_io.load_pil_images,
            device=runtime_device,
        )

    def _resolve_model_id(self) -> str:
        if not self.model_id:
            raise ValueError(
                f"{self.name} requires --adapter-model-id set to a Janus checkpoint path or repo id, "
                "for example deepseek-ai/Janus-Pro-1B or deepseek-ai/Janus-Pro-7B."
            )
        raw_value = str(self.model_id).strip()
        alias_key = raw_value.lower().replace("_", "-")
        if alias_key in self._MODEL_ALIASES:
            return self._MODEL_ALIASES[alias_key]
        path_candidate = Path(raw_value).expanduser()
        if path_candidate.exists():
            return str(path_candidate)
        return raw_value

    def _ensure_janus_import_path(self, model_id: str) -> None:
        candidates: list[Path | None] = [self.code_root]
        model_path = Path(model_id)
        if model_path.exists():
            candidates.extend([model_path, model_path.parent, model_path.parent.parent])
        for candidate in candidates:
            if candidate is None:
                continue
            candidate = Path(candidate).expanduser()
            if (candidate / "janus" / "models").is_dir():
                candidate_str = str(candidate)
                if candidate_str not in sys.path:
                    sys.path.insert(0, candidate_str)
                return

    def _preferred_dtype(self) -> torch.dtype:
        if not torch.cuda.is_available():
            return torch.float32
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _requested_device(self) -> torch.device:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        device_text = str(self.device).strip().lower()
        if device_text == "cpu":
            return torch.device("cpu")
        if device_text.startswith("cuda"):
            return torch.device(str(self.device))
        return torch.device("cuda")

    def _resolve_runtime_device(self, model: Any) -> torch.device:
        hf_device_map = getattr(model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            for location in hf_device_map.values():
                if isinstance(location, int):
                    return torch.device(f"cuda:{location}")
                if isinstance(location, str) and location not in {"cpu", "disk"}:
                    return torch.device(location)
        try:
            return next(model.parameters()).device
        except StopIteration:
            return self._requested_device()

    def _build_max_memory_map(self) -> dict[int, str]:
        visible_gpus = torch.cuda.device_count()
        if visible_gpus == 0:
            raise RuntimeError("No CUDA devices are visible for Janus inference.")
        if self.max_memory_per_gpu:
            return {idx: self.max_memory_per_gpu for idx in range(visible_gpus)}

        max_memory: dict[int, str] = {}
        for idx in range(visible_gpus):
            free_bytes, _ = torch.cuda.mem_get_info(idx)
            free_gib = max(1, int((free_bytes / (1024**3)) * 0.9))
            max_memory[idx] = f"{free_gib}GiB"
        return max_memory

    def _resolve_offload_dir(self) -> Path:
        if self.offload_dir is not None:
            return self.offload_dir / self.name
        project_root = Path(self.config_file).resolve().parents[1]
        return project_root / "tmp_janus_offload" / self.name

    def _should_use_target_image(self, image_path: str | Path) -> bool:
        path = Path(image_path)
        return path.exists() and not path.name.startswith("black_")

    def _cleanup_cuda(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
