from __future__ import annotations

import gc
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

from ..common import PromptSpec, UnderstandingExampleSpec, set_global_seed, stable_seed
from .external_baseline import ExternalBaselineAdapter


@dataclass
class _BagelRuntime:
    inferencer: Any
    model: Any
    vae_model: Any
    tokenizer: Any


class BagelAdapterBase(ExternalBaselineAdapter):
    model_family = "bagel"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._runtime: _BagelRuntime | None = None

    def has_model_epoch(self, concept: str, model_epoch: int) -> bool:
        return model_epoch == 0

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
        del task_name, concept, model_epoch, batch_size

        image_files: list[str] = []
        for image_idx in range(num_images):
            sample_seed = stable_seed(seed, "bagel_image", image_idx)
            set_global_seed(sample_seed)
            inputs = self._build_generation_inputs(
                prompt_spec=prompt_spec,
                input_prompt=input_prompt,
                reference_image_paths=reference_image_paths,
            )
            image = self._run_generation(inputs)
            image_name = f"image_{image_idx:03d}.png"
            image.save(prompt_dir / image_name)
            image_files.append(image_name)
            self._cleanup_cuda()
        return image_files

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

        set_global_seed(stable_seed(seed, "bagel_text"))
        inputs = self._build_understanding_inputs(
            example=example,
            input_prompt=input_prompt,
            reference_image_paths=reference_image_paths,
        )
        prediction = self._run_understanding(inputs, max_new_tokens=max_new_tokens)
        self._cleanup_cuda()
        return prediction

    def _run_generation(self, inputs: list[Any]) -> Image.Image:
        runtime = self._get_runtime()
        output = None
        for chunk in runtime.inferencer.interleave_inference(
            input_lists=inputs,
            think=self.use_thinking,
            understanding_output=False,
            image_shapes=(self.image_height, self.image_width),
            **self._generation_hyperparams(),
        ):
            output = chunk
        if not isinstance(output, Image.Image):
            raise RuntimeError(f"{self.name} expected an image output, but received {type(output)!r}")
        return output

    def _run_understanding(self, inputs: list[Any], *, max_new_tokens: int) -> str:
        runtime = self._get_runtime()
        text_parts: list[str] = []
        for chunk in runtime.inferencer.interleave_inference(
            input_lists=inputs,
            think=self.use_thinking,
            understanding_output=True,
            max_think_token_n=max_new_tokens or self.adapter_max_new_tokens,
            do_sample=False,
            text_temperature=self.adapter_temperature,
        ):
            if isinstance(chunk, str):
                text_parts.append(chunk)
        return "".join(text_parts).strip()

    def _build_generation_inputs(
        self,
        *,
        prompt_spec: PromptSpec,
        input_prompt: str,
        reference_image_paths: Sequence[str],
    ) -> list[Any]:
        inputs: list[Any] = []
        if reference_image_paths:
            inputs.append(self._load_image(reference_image_paths[0]))
        inputs.append(input_prompt)
        return inputs

    def _build_understanding_inputs(
        self,
        *,
        example: UnderstandingExampleSpec,
        input_prompt: str,
        reference_image_paths: Sequence[str],
    ) -> list[Any]:
        inputs: list[Any] = []
        if reference_image_paths:
            inputs.append(self._load_image(reference_image_paths[0]))
            if example.conditioning_text:
                inputs.append(example.conditioning_text)

        if self._should_use_target_image(example.image_path):
            inputs.append(self._load_image(example.image_path))

        inputs.append(input_prompt)
        return inputs

    def _generation_hyperparams(self) -> dict[str, Any]:
        return {
            "cfg_text_scale": self.cfg_text_scale,
            "cfg_img_scale": self._cfg_img_scale(),
            "cfg_interval": [self._cfg_interval_start(), 1.0],
            "timestep_shift": self.timestep_shift,
            "num_timesteps": self.num_timesteps,
            "cfg_renorm_min": self.cfg_renorm_min,
            "cfg_renorm_type": self._cfg_renorm_type(),
        }

    def _cfg_img_scale(self) -> float:
        if self.cfg_img_scale is not None:
            return self.cfg_img_scale
        return 1.0 if self.prompt_mode == "tp" else 2.0

    def _cfg_interval_start(self) -> float:
        if self.cfg_interval_start is not None:
            return self.cfg_interval_start
        return 0.4 if self.prompt_mode == "tp" else 0.0

    def _cfg_renorm_type(self) -> str:
        if self.cfg_renorm_type is not None:
            return self.cfg_renorm_type
        return "global" if self.prompt_mode == "tp" else "text_channel"

    def _get_runtime(self) -> _BagelRuntime:
        if self._runtime is None:
            self._runtime = self._load_runtime()
        return self._runtime

    def _load_runtime(self) -> _BagelRuntime:
        if not torch.cuda.is_available():
            raise RuntimeError("Bagel inference requires CUDA.")

        model_path = self._resolve_model_path()
        code_root = self._resolve_code_root(model_path)
        self._ensure_import_path(code_root)

        inferencer_module = importlib.import_module("inferencer")
        data_utils = importlib.import_module("data.data_utils")
        transforms = importlib.import_module("data.transforms")
        bagel_module = importlib.import_module("modeling.bagel")
        qwen2_module = importlib.import_module("modeling.qwen2")
        autoencoder_module = importlib.import_module("modeling.autoencoder")
        accelerate = importlib.import_module("accelerate")

        llm_config = bagel_module.Qwen2Config.from_json_file(str(model_path / "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = bagel_module.SiglipVisionConfig.from_json_file(str(model_path / "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        vae_model, vae_config = autoencoder_module.load_ae(local_path=str(model_path / "ae.safetensors"))

        config = bagel_module.BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
        )

        with accelerate.init_empty_weights():
            language_model = bagel_module.Qwen2ForCausalLM(llm_config)
            vit_model = bagel_module.SiglipVisionModel(vit_config)
            model = bagel_module.Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        tokenizer = qwen2_module.Qwen2Tokenizer.from_pretrained(str(model_path))
        tokenizer, new_token_ids, _ = data_utils.add_special_tokens(tokenizer)
        vae_transform = transforms.ImageTransform(1024, 512, 16)
        vit_transform = transforms.ImageTransform(980, 224, 14)

        max_memory = self._build_max_memory_map()
        device_map = accelerate.infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        device_map = self._pin_shared_modules(device_map)

        offload_dir = self._resolve_offload_dir()
        offload_dir.mkdir(parents=True, exist_ok=True)
        model = accelerate.load_checkpoint_and_dispatch(
            model,
            checkpoint=str(model_path / "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder=str(offload_dir),
        )
        model = model.eval()

        inferencer = inferencer_module.InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )
        return _BagelRuntime(
            inferencer=inferencer,
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
        )

    def _resolve_model_path(self) -> Path:
        if not self.model_id:
            raise ValueError(
                f"{self.name} requires --adapter-model-id pointing to a local BAGEL-7B-MoT checkpoint directory."
            )
        model_path = Path(self.model_id).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Bagel model path not found: {model_path}")
        return model_path

    def _resolve_code_root(self, model_path: Path) -> Path:
        candidates = [
            self.code_root,
            model_path,
            model_path.parent,
            model_path.parent.parent,
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            if (candidate / "inferencer.py").exists() and (candidate / "modeling").is_dir() and (candidate / "data").is_dir():
                return candidate
        raise FileNotFoundError(
            f"{self.name} could not find Bagel source code. Pass --adapter-code-root to the local BAGEL repo root."
        )

    def _ensure_import_path(self, code_root: Path) -> None:
        code_root_str = str(code_root)
        if code_root_str not in sys.path:
            sys.path.insert(0, code_root_str)

    def _build_max_memory_map(self) -> dict[int, str]:
        visible_gpus = torch.cuda.device_count()
        if visible_gpus == 0:
            raise RuntimeError("No CUDA devices are visible for Bagel inference.")
        if self.max_memory_per_gpu:
            return {idx: self.max_memory_per_gpu for idx in range(visible_gpus)}

        max_memory: dict[int, str] = {}
        for idx in range(visible_gpus):
            free_bytes, _ = torch.cuda.mem_get_info(idx)
            free_gib = max(1, int((free_bytes / (1024**3)) * 0.9))
            max_memory[idx] = f"{free_gib}GiB"
        return max_memory

    def _pin_shared_modules(self, device_map: dict[str, Any]) -> dict[str, Any]:
        same_device_modules = [
            "language_model.model.embed_tokens",
            "time_embedder",
            "latent_pos_embed",
            "vae2llm",
            "llm2vae",
            "connector",
            "vit_pos_embed",
        ]
        pinned = dict(device_map)
        if torch.cuda.device_count() == 1:
            first_device = pinned.get(same_device_modules[0], "cuda:0")
            for module_name in same_device_modules:
                pinned[module_name] = first_device
            return pinned

        first_device = pinned.get(same_device_modules[0])
        if first_device is None:
            return pinned
        for module_name in same_device_modules:
            pinned[module_name] = first_device
        return pinned

    def _resolve_offload_dir(self) -> Path:
        if self.offload_dir is not None:
            return self.offload_dir / self.name / f"pid_{os.getpid()}"
        project_root = Path(self.config_file).resolve().parents[1]
        return project_root / "tmp_bagel_offload" / self.name / f"pid_{os.getpid()}"

    def _load_image(self, image_path: str | Path) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def _should_use_target_image(self, image_path: str | Path) -> bool:
        path = Path(image_path)
        return path.exists() and not path.name.startswith("black_")

    def _cleanup_cuda(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
