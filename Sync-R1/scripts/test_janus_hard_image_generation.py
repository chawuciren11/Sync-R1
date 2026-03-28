from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM


MODEL_ALIASES = {
    "januspro1b": "deepseek-ai/Janus-Pro-1B",
    "janus-pro-1b": "deepseek-ai/Janus-Pro-1B",
    "deepseek-ai/janus-pro-1b": "deepseek-ai/Janus-Pro-1B",
    "januspro7b": "deepseek-ai/Janus-Pro-7B",
    "janus-pro-7b": "deepseek-ai/Janus-Pro-7B",
    "deepseek-ai/janus-pro-7b": "deepseek-ai/Janus-Pro-7B",
}


@dataclass
class JanusRuntime:
    model: Any
    processor: Any
    tokenizer: Any
    load_pil_images: Any
    device: torch.device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Janus experiment script. It can run standard understanding, "
            "standard text-to-image generation, and an experimental mode that force-feeds "
            "reference images into the generation prefix."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("compare", "understand", "text_generate", "force_image_generate"),
        default="compare",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="januspro7b",
        help="Local path or repo id, for example deepseek-ai/Janus-Pro-1B or januspro7b.",
    )
    parser.add_argument(
        "--code-root",
        type=str,
        default=None,
        help="Optional Janus source repo root if janus is not installed as a package.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt used for text generation, and also as the question in compare/understand mode.",
    )
    parser.add_argument(
        "--image-paths",
        nargs="*",
        default=[],
        help="Optional reference image(s). Required for understand and force_image_generate modes.",
    )
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-weight", type=float, default=5.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-memory-per-gpu", type=str, default=None)
    parser.add_argument("--offload-dir", type=str, default=None)
    parser.add_argument(
        "--allow-remote-models",
        action="store_true",
        help="Disable local_files_only so Hugging Face downloads are allowed.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_model_id(raw_value: str) -> str:
    alias_key = raw_value.strip().lower().replace("_", "-")
    if alias_key in MODEL_ALIASES:
        return MODEL_ALIASES[alias_key]
    candidate = Path(raw_value).expanduser()
    if candidate.exists():
        return str(candidate)
    return raw_value


def ensure_janus_import_path(code_root: str | None, model_id: str) -> None:
    candidates: list[Path] = []
    if code_root:
        candidates.append(Path(code_root).expanduser())
    model_path = Path(model_id)
    if model_path.exists():
        candidates.extend([model_path, model_path.parent, model_path.parent.parent])
    for candidate in candidates:
        if (candidate / "janus" / "models").is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


def preferred_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def requested_device(device: str) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    device_text = device.strip().lower()
    if device_text == "cpu":
        return torch.device("cpu")
    if device_text.startswith("cuda"):
        return torch.device(device)
    return torch.device("cuda")


def build_max_memory_map(max_memory_per_gpu: str) -> dict[int, str]:
    visible_gpus = torch.cuda.device_count()
    if visible_gpus == 0:
        raise RuntimeError("No CUDA devices are visible.")
    if max_memory_per_gpu:
        return {idx: max_memory_per_gpu for idx in range(visible_gpus)}

    max_memory: dict[int, str] = {}
    for idx in range(visible_gpus):
        free_bytes, _ = torch.cuda.mem_get_info(idx)
        free_gib = max(1, int((free_bytes / (1024**3)) * 0.9))
        max_memory[idx] = f"{free_gib}GiB"
    return max_memory


def resolve_runtime_device(model: Any, fallback_device: torch.device) -> torch.device:
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
        return fallback_device


def load_runtime(args: argparse.Namespace) -> JanusRuntime:
    model_id = resolve_model_id(args.model_id)
    ensure_janus_import_path(args.code_root, model_id)

    try:
        janus_models = __import__("janus.models", fromlist=["VLChatProcessor"])
        janus_io = __import__("janus.utils.io", fromlist=["load_pil_images"])
    except ImportError as exc:
        raise ImportError(
            "Could not import DeepSeek Janus. Install the package or pass --code-root to the Janus repo root."
        ) from exc

    if not hasattr(janus_models, "VLChatProcessor") or not hasattr(janus_io, "load_pil_images"):
        raise ImportError(
            "A module named 'janus' was found, but it does not match DeepSeek Janus."
        )

    processor = janus_models.VLChatProcessor.from_pretrained(
        model_id,
        local_files_only=not args.allow_remote_models,
    )
    tokenizer = processor.tokenizer

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": not args.allow_remote_models,
    }
    dtype = preferred_dtype()
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = dtype
        if args.max_memory_per_gpu:
            offload_dir = Path(args.offload_dir or (Path(args.output_root) / "offload")).resolve()
            offload_dir.mkdir(parents=True, exist_ok=True)
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = build_max_memory_map(args.max_memory_per_gpu)
            model_kwargs["offload_folder"] = str(offload_dir)

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs).eval()
    fallback_device = requested_device(args.device)
    if "device_map" not in model_kwargs:
        if torch.cuda.is_available():
            model = model.to(dtype=dtype)
        model = model.to(fallback_device).eval()

    device = resolve_runtime_device(model, fallback_device)
    return JanusRuntime(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        load_pil_images=janus_io.load_pil_images,
        device=device,
    )


def build_conversation(prompt: str, image_paths: list[str]) -> list[dict[str, Any]]:
    content_parts: list[str] = []
    if image_paths:
        content_parts.extend(["<image_placeholder>"] * len(image_paths))
    content_parts.append(prompt.strip())
    user_turn: dict[str, Any] = {
        "role": "<|User|>",
        "content": "\n".join(part for part in content_parts if part),
    }
    if image_paths:
        user_turn["images"] = image_paths
    return [
        user_turn,
        {"role": "<|Assistant|>", "content": ""},
    ]


def clean_text_output(text: str) -> str:
    cleaned = text.strip()
    for marker in ("<|Assistant|>", "Assistant:"):
        if marker in cleaned:
            cleaned = cleaned.split(marker)[-1].strip()
    return cleaned


def run_understanding(
    runtime: JanusRuntime,
    *,
    prompt: str,
    image_paths: list[str],
    max_new_tokens: int,
) -> str:
    conversation = build_conversation(prompt, image_paths)
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
        max_new_tokens=max_new_tokens,
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
    return clean_text_output(text)


@torch.inference_mode()
def generate_text_only(
    runtime: JanusRuntime,
    *,
    prompt: str,
    num_images: int,
    temperature: float,
    cfg_weight: float,
    image_size: int,
) -> list[Image.Image]:
    conversation = [
        {"role": "<|User|>", "content": prompt.strip()},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = runtime.processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=runtime.processor.sft_format,
        system_prompt="",
    )
    prompt_with_image_start = sft_format + runtime.processor.image_start_tag
    prompt_tokens = runtime.processor.tokenizer.encode(prompt_with_image_start)
    device = runtime.device
    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)

    tokens = torch.zeros((num_images * 2, len(prompt_tokens)), dtype=torch.long, device=device)
    for row_idx in range(num_images * 2):
        tokens[row_idx, :] = prompt_tensor
        if row_idx % 2 == 1 and tokens.shape[1] > 2:
            tokens[row_idx, 1:-1] = runtime.processor.pad_id

    inputs_embeds = runtime.model.language_model.get_input_embeddings()(tokens)
    language_backbone = getattr(runtime.model.language_model, "model", runtime.model.language_model)
    image_token_count = (image_size // 16) * (image_size // 16)
    generated_tokens = torch.zeros((num_images, image_token_count), dtype=torch.long, device=device)

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
        guided_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        if temperature <= 0:
            next_token = torch.argmax(guided_logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(guided_logits / max(temperature, 1e-5), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, token_idx] = next_token.squeeze(-1)
        duplicated = torch.cat([next_token, next_token], dim=1).view(-1)
        img_embeds = runtime.model.prepare_gen_img_embeds(duplicated)
        inputs_embeds = img_embeds.unsqueeze(1)

    return decode_images(runtime, generated_tokens, num_images=num_images, image_size=image_size)


@torch.inference_mode()
def generate_force_image_multimodal_prefix(
    runtime: JanusRuntime,
    *,
    prompt: str,
    image_paths: list[str],
    num_images: int,
    temperature: float,
    image_size: int,
) -> list[Image.Image]:
    conversation = build_conversation(prompt, image_paths)
    pil_images = runtime.load_pil_images(conversation)
    prepare_inputs = runtime.processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
    )
    if hasattr(prepare_inputs, "to"):
        prepare_inputs = prepare_inputs.to(runtime.device)
    prefix_embeds = runtime.model.prepare_inputs_embeds(**prepare_inputs)
    prefix_embeds = prefix_embeds.expand(num_images, -1, -1).contiguous()

    image_start_ids = runtime.tokenizer.encode(
        runtime.processor.image_start_tag,
        add_special_tokens=False,
    )
    if not image_start_ids:
        raise RuntimeError("Janus image_start_tag encoded to an empty token sequence.")
    image_start_tensor = torch.tensor(image_start_ids, dtype=torch.long, device=runtime.device)
    image_start_embeds = runtime.model.language_model.get_input_embeddings()(image_start_tensor)
    image_start_embeds = image_start_embeds.unsqueeze(0).expand(num_images, -1, -1).contiguous()
    inputs_embeds = torch.cat([prefix_embeds, image_start_embeds], dim=1)

    language_backbone = getattr(runtime.model.language_model, "model", runtime.model.language_model)
    image_token_count = (image_size // 16) * (image_size // 16)
    generated_tokens = torch.zeros((num_images, image_token_count), dtype=torch.long, device=runtime.device)

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
        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, token_idx] = next_token.squeeze(-1)
        img_embeds = runtime.model.prepare_gen_img_embeds(next_token.view(-1))
        inputs_embeds = img_embeds.unsqueeze(1)

    return decode_images(runtime, generated_tokens, num_images=num_images, image_size=image_size)


def decode_images(
    runtime: JanusRuntime,
    generated_tokens: torch.Tensor,
    *,
    num_images: int,
    image_size: int,
) -> list[Image.Image]:
    spatial_tokens = image_size // 16
    decoded = runtime.model.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[num_images, 8, spatial_tokens, spatial_tokens],
    )
    decoded = decoded.to(torch.float32).detach().cpu().numpy().transpose(0, 2, 3, 1)
    decoded = np.clip((decoded + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    return [Image.fromarray(array) for array in decoded]


def save_images(images: list[Image.Image], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_names: list[str] = []
    for idx, image in enumerate(images):
        file_name = f"image_{idx:03d}.png"
        image.save(output_dir / file_name)
        file_names.append(file_name)
    return file_names


def write_report(output_root: Path, report: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = [str(Path(path).expanduser().resolve()) for path in args.image_paths]
    if args.mode in {"understand", "force_image_generate"} and not image_paths:
        raise ValueError(f"--image-paths is required for mode={args.mode}")

    if args.image_size % 16 != 0:
        raise ValueError("--image-size must be divisible by 16 for Janus generation.")

    set_seed(args.seed)
    runtime = load_runtime(args)

    report: dict[str, Any] = {
        "mode": args.mode,
        "model_id": resolve_model_id(args.model_id),
        "code_root": args.code_root,
        "prompt": args.prompt,
        "image_paths": image_paths,
        "num_images": args.num_images,
        "temperature": args.temperature,
        "cfg_weight": args.cfg_weight,
        "image_size": args.image_size,
        "device": str(runtime.device),
        "results": {},
        "notes": {
            "force_image_generate": (
                "This path is experimental: it feeds multimodal prefix embeddings into the Janus "
                "image-token sampler, even though the public Janus generation example is text-only."
            )
        },
    }

    try:
        if args.mode in {"compare", "understand"}:
            understanding_prompt = args.prompt
            if args.mode == "compare":
                understanding_prompt = (
                    f"Describe the content and identity details relevant to the following generation request: "
                    f"{args.prompt}"
                )
            answer = run_understanding(
                runtime,
                prompt=understanding_prompt,
                image_paths=image_paths,
                max_new_tokens=args.max_new_tokens,
            )
            report["results"]["understand"] = {
                "prompt": understanding_prompt,
                "answer": answer,
            }
            cleanup_cuda()

        if args.mode in {"compare", "text_generate"}:
            images = generate_text_only(
                runtime,
                prompt=args.prompt,
                num_images=args.num_images,
                temperature=args.temperature,
                cfg_weight=args.cfg_weight,
                image_size=args.image_size,
            )
            file_names = save_images(images, output_root / "text_generate")
            report["results"]["text_generate"] = {
                "output_dir": str((output_root / "text_generate").resolve()),
                "image_files": file_names,
            }
            cleanup_cuda()

        if args.mode in {"compare", "force_image_generate"}:
            images = generate_force_image_multimodal_prefix(
                runtime,
                prompt=args.prompt,
                image_paths=image_paths,
                num_images=args.num_images,
                temperature=args.temperature,
                image_size=args.image_size,
            )
            file_names = save_images(images, output_root / "force_image_generate")
            report["results"]["force_image_generate"] = {
                "output_dir": str((output_root / "force_image_generate").resolve()),
                "image_files": file_names,
            }
            cleanup_cuda()

    except Exception as exc:
        report.setdefault("errors", []).append(repr(exc))
        write_report(output_root, report)
        raise

    write_report(output_root, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
