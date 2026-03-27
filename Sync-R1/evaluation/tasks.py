from __future__ import annotations

from pathlib import Path

from .common import (
    PromptSpec,
    TaskDefinition,
    UnderstandingExampleSpec,
    build_adj_token_string,
    build_simple_conditioning_text,
    load_reference_image_paths,
    load_json,
    resolve_dataset_path,
)


TASKS: dict[str, TaskDefinition] = {
    "pure_gen": TaskDefinition(
        name="pure_gen",
        family="generation",
        metrics=("clip_t", "clip_i"),
        default_batch_size=2,
        default_num_outputs=12,
    ),
    "dense_gen": TaskDefinition(
        name="dense_gen",
        family="generation",
        metrics=("gpt", "clip_i"),
        default_batch_size=2,
        default_num_outputs=10,
    ),
    "rea_gen": TaskDefinition(
        name="rea_gen",
        family="generation",
        metrics=("clip_t", "clip_i"),
        default_batch_size=2,
        default_num_outputs=10,
    ),
    "dense_rea_gen": TaskDefinition(
        name="dense_rea_gen",
        family="generation",
        metrics=("gpt", "clip_i"),
        default_batch_size=2,
        default_num_outputs=20,
    ),
    "rec": TaskDefinition(
        name="rec",
        family="understanding",
        metrics=("accuracy",),
    ),
    "vqa": TaskDefinition(
        name="vqa",
        family="understanding",
        metrics=("bleu", "gpt"),
    ),
    "text_only": TaskDefinition(
        name="text_only",
        family="understanding",
        metrics=("bleu", "gpt"),
    ),
    "rea": TaskDefinition(
        name="rea",
        family="understanding",
        metrics=("bleu", "gpt"),
    ),
    "dense_rea": TaskDefinition(
        name="dense_rea",
        family="understanding",
        metrics=("bleu", "gpt"),
    ),
}


def resolve_task_names(requested: list[str] | None) -> list[str]:
    if not requested or requested == ["all"]:
        return list(TASKS.keys())

    unknown = [name for name in requested if name not in TASKS]
    if unknown:
        raise ValueError(f"Unknown task(s): {', '.join(unknown)}")
    return requested


def generation_task_names(task_names: list[str]) -> list[str]:
    return [name for name in task_names if TASKS[name].family == "generation"]


def understanding_task_names(task_names: list[str]) -> list[str]:
    return [name for name in task_names if TASKS[name].family == "understanding"]


def build_generation_prompt_specs(
    task_name: str,
    concept: str,
    data_root: str | Path,
    prompts_to_eval_path: str | Path,
    inverse_prompt: bool,
    nums_new_token_i_stage_1: int,
    nums_new_token_i_stage_2: int,
    reference_image_count: int = 1,
) -> list[PromptSpec]:
    data_root = Path(data_root)
    test_conditions = load_json(data_root / "concept" / "test" / concept / "t2i_conditions.json")
    train_info = load_json(data_root / "concept" / "train" / concept / "info.json")
    conditioning_text = build_simple_conditioning_text(concept, train_info)
    reference_image_paths = load_reference_image_paths(data_root, concept, count=reference_image_count)

    adj_tokens = build_adj_token_string(
        concept,
        nums_new_token_i_stage_1,
        nums_new_token_i_stage_2,
    )

    def apply_inverse_prompt(prompt_text: str, placeholder: str) -> str:
        if not inverse_prompt:
            return prompt_text
        return prompt_text.replace(placeholder, adj_tokens)

    if task_name == "pure_gen":
        prompts_to_eval = load_json(prompts_to_eval_path)
        generation_type = train_info["generation_type"]
        raw_prompts = prompts_to_eval[generation_type]
        return [
            PromptSpec(
                prompt_id=f"prompt_{idx:03d}",
                source_prompt=prompt,
                generation_prompt=apply_inverse_prompt(prompt, "<sks>"),
                scoring_prompt=prompt.replace("<sks>", train_info["class"]),
                baseline_prompt=prompt.replace("<sks>", train_info["class"]),
                conditioning_text=conditioning_text,
                reference_image_paths=reference_image_paths,
                metadata={"source": "prompts_to_eval", "generation_type": generation_type},
            )
            for idx, prompt in enumerate(raw_prompts)
        ]

    if task_name == "dense_gen":
        raw_prompts = test_conditions["dense_prompt_generation"]
        return [
            PromptSpec(
                prompt_id=f"prompt_{idx:03d}",
                source_prompt=prompt,
                generation_prompt=apply_inverse_prompt(prompt, f"<{concept}>"),
                scoring_prompt=prompt,
                baseline_prompt=prompt.replace(f"<{concept}>", train_info["class"]),
                conditioning_text=conditioning_text,
                reference_image_paths=reference_image_paths,
                metadata={"source": "dense_prompt_generation"},
            )
            for idx, prompt in enumerate(raw_prompts)
        ]

    if task_name == "rea_gen":
        raw_prompts = test_conditions["explicit_personalized_driven_generation"]
        num_testing = len(raw_prompts) // 2
        raw_prompts = raw_prompts[-num_testing:]
        return [
            PromptSpec(
                prompt_id=f"prompt_{idx:03d}",
                source_prompt=prompt,
                generation_prompt=apply_inverse_prompt(prompt, f"<{concept}>"),
                scoring_prompt=prompt,
                baseline_prompt=prompt.replace(f"<{concept}>", train_info["class"]),
                conditioning_text=conditioning_text,
                reference_image_paths=reference_image_paths,
                metadata={"source": "explicit_personalized_driven_generation"},
            )
            for idx, prompt in enumerate(raw_prompts)
        ]

    if task_name == "dense_rea_gen":
        raw_prompts = test_conditions["explicit_personalized_dense_prompt_generation"]
        return [
            PromptSpec(
                prompt_id=f"prompt_{idx:03d}",
                source_prompt=prompt,
                generation_prompt=apply_inverse_prompt(prompt, f"<{concept}>"),
                scoring_prompt=prompt,
                baseline_prompt=prompt.replace(f"<{concept}>", train_info["class"]),
                conditioning_text=conditioning_text,
                reference_image_paths=reference_image_paths,
                metadata={"source": "explicit_personalized_dense_prompt_generation"},
            )
            for idx, prompt in enumerate(raw_prompts)
        ]

    raise ValueError(f"Unsupported generation task: {task_name}")


def build_understanding_example_specs(
    task_name: str,
    concept: str,
    data_root: str | Path,
    reference_image_count: int = 1,
) -> list[UnderstandingExampleSpec]:
    data_root = Path(data_root)
    train_info = load_json(data_root / "concept" / "train" / concept / "info.json")
    conditioning_text = build_simple_conditioning_text(concept, train_info)
    reference_image_paths = load_reference_image_paths(data_root, concept, count=reference_image_count)
    class_name = train_info["class"]

    if task_name in {"rec", "vqa", "text_only"}:
        test_data = load_json(data_root / "test_data" / f"{concept}.json")
        filtered_items = [item for item in test_data if item["type"] == task_name]
        examples: list[UnderstandingExampleSpec] = []
        for idx, item in enumerate(filtered_items):
            query = item["conversations"][0]["value"].replace("<image>\n", "").strip()
            baseline_query = query.replace(f"<{concept}>", f"the reference {class_name}")
            examples.append(
                UnderstandingExampleSpec(
                    item_id=f"{task_name}_{idx:03d}",
                    source_prompt=query,
                    model_prompt=query,
                    scoring_query=query,
                    ground_truth=item["conversations"][1]["value"],
                    image_path=str(resolve_dataset_path(data_root, item["image"])),
                    baseline_prompt=baseline_query,
                    conditioning_text=conditioning_text,
                    reference_image_paths=reference_image_paths,
                    prepend_system_prompt=True,
                    metadata={"source": "test_data", "raw_type": item["type"]},
                )
            )
        return examples

    test_conditions = load_json(data_root / "concept" / "test" / concept / "t2i_conditions.json")
    black_image_path = str(data_root / "black_512x512.png")

    if task_name == "rea":
        prompts = test_conditions["personalized_driven_generation"]
        references = test_conditions["explicit_personalized_driven_generation"]
        num_testing = int(train_info["num_testing_info"])
        prompts = prompts[-num_testing:]
        references = references[-num_testing:]
        extra_info = "; ".join(train_info["extra_info"])
        return [
            UnderstandingExampleSpec(
                item_id=f"{task_name}_{idx:03d}",
                source_prompt=prompt,
                model_prompt=(
                    f"Facts about <{concept}>: {extra_info}\n"
                    f"Prompt: {prompt}\n"
                    "Return the matching fact only. If none match, return empty."
                ),
                scoring_query=prompt,
                ground_truth=reference,
                image_path=black_image_path,
                baseline_prompt=(
                    f"Prompt: {prompt.replace(f'<{concept}>', f'the reference {class_name}')}\n"
                    "Return the matching fact only."
                ),
                conditioning_text=conditioning_text,
                reference_image_paths=reference_image_paths,
                prepend_system_prompt=False,
                metadata={"source": "personalized_driven_generation"},
            )
            for idx, (prompt, reference) in enumerate(zip(prompts, references))
        ]

    if task_name == "dense_rea":
        prompt = test_conditions["personalized_dense_prompt_generation"][0]
        reference = test_conditions["explicit_personalized_dense_prompt_generation"][0]
        extra_info = "; ".join(train_info["extra_info"])
        return [
            UnderstandingExampleSpec(
                item_id=f"{task_name}_000",
                source_prompt=prompt,
                model_prompt=(
                    f"Facts about <{concept}>: {extra_info}\n"
                    f"Rewrite this prompt with relevant facts: {prompt}\n"
                    "If nothing is relevant, return the prompt unchanged."
                ),
                scoring_query=prompt,
                ground_truth=reference,
                image_path=black_image_path,
                baseline_prompt=(
                    "Rewrite this prompt with relevant facts: "
                    f"{prompt.replace(f'<{concept}>', f'the reference {class_name}')}"
                ),
                conditioning_text=conditioning_text,
                reference_image_paths=reference_image_paths,
                prepend_system_prompt=False,
                metadata={"source": "personalized_dense_prompt_generation"},
            )
        ]

    raise ValueError(f"Unsupported understanding task: {task_name}")
