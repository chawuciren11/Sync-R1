from __future__ import annotations

from pathlib import Path

from .common import (
    PromptSpec,
    TaskDefinition,
    UnderstandingExampleSpec,
    build_adj_token_string,
    build_ip_prefix_text,
    build_tp_prefix_text,
    load_reference_image_paths,
    load_test_ground_truth_image_paths,
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
        metrics=("weight",),
    ),
    "vqa": TaskDefinition(
        name="vqa",
        family="understanding",
        metrics=("bleu", "gpt"),
    ),
    "qa": TaskDefinition(
        name="qa",
        family="understanding",
        metrics=("bleu", "gpt"),
    ),
    "rea": TaskDefinition(
        name="rea",
        family="understanding",
        metrics=("bleu",),
    ),
    "dense_rea": TaskDefinition(
        name="dense_rea",
        family="understanding",
        metrics=("gpt",),
    ),
}


TASK_ALIASES: dict[str, str] = {
    "text_only": "qa",
}


PURE_GEN_OBJECT_TEMPLATES: tuple[str, ...] = (
    "a {unique_token} {class_token} in the jungle",
    "a {unique_token} {class_token} in the snow",
    "a {unique_token} {class_token} on the beach",
    "a {unique_token} {class_token} on a cobblestone street",
    "a {unique_token} {class_token} on top of pink fabric",
    "a {unique_token} {class_token} on top of a wooden floor",
    "a {unique_token} {class_token} with a city in the background",
    "a {unique_token} {class_token} with a mountain in the background",
    "a {unique_token} {class_token} with a blue house in the background",
    "a {unique_token} {class_token} on top of a purple rug in a forest",
    "a {unique_token} {class_token} with a wheat field in the background",
    "a {unique_token} {class_token} with a tree and autumn leaves in the background",
    "a {unique_token} {class_token} with the Eiffel Tower in the background",
    "a {unique_token} {class_token} floating on top of water",
    "a {unique_token} {class_token} floating in an ocean of milk",
    "a {unique_token} {class_token} on top of green grass with sunflowers around it",
    "a {unique_token} {class_token} on top of a mirror",
    "a {unique_token} {class_token} on top of the sidewalk in a crowded street",
    "a {unique_token} {class_token} on top of a dirt road",
    "a {unique_token} {class_token} on top of a white rug",
    "a red {unique_token} {class_token}",
    "a purple {unique_token} {class_token}",
    "a shiny {unique_token} {class_token}",
    "a wet {unique_token} {class_token}",
    "a cube shaped {unique_token} {class_token}",
)


PURE_GEN_LIVE_SUBJECT_TEMPLATES: tuple[str, ...] = (
    "a {unique_token} {class_token} in the jungle",
    "a {unique_token} {class_token} in the snow",
    "a {unique_token} {class_token} on the beach",
    "a {unique_token} {class_token} on a cobblestone street",
    "a {unique_token} {class_token} on top of pink fabric",
    "a {unique_token} {class_token} on top of a wooden floor",
    "a {unique_token} {class_token} with a city in the background",
    "a {unique_token} {class_token} with a mountain in the background",
    "a {unique_token} {class_token} with a blue house in the background",
    "a {unique_token} {class_token} on top of a purple rug in a forest",
    "a {unique_token} {class_token} wearing a red hat",
    "a {unique_token} {class_token} wearing a santa hat",
    "a {unique_token} {class_token} wearing a rainbow scarf",
    "a {unique_token} {class_token} wearing a black top hat and a monocle",
    "a {unique_token} {class_token} in a chef outfit",
    "a {unique_token} {class_token} in a firefighter outfit",
    "a {unique_token} {class_token} in a police outfit",
    "a {unique_token} {class_token} wearing pink glasses",
    "a {unique_token} {class_token} wearing a yellow shirt",
    "a {unique_token} {class_token} in a purple wizard outfit",
    "a red {unique_token} {class_token}",
    "a purple {unique_token} {class_token}",
    "a shiny {unique_token} {class_token}",
    "a wet {unique_token} {class_token}",
    "a cube shaped {unique_token} {class_token}",
)


def resolve_task_names(requested: list[str] | None) -> list[str]:
    if not requested or requested == ["all"]:
        return list(TASKS.keys())

    normalized = [TASK_ALIASES.get(name, name) for name in requested]
    deduped = list(dict.fromkeys(normalized))

    unknown = [name for name in deduped if name not in TASKS]
    if unknown:
        raise ValueError(f"Unknown task(s): {', '.join(unknown)}")
    return deduped


def generation_task_names(task_names: list[str]) -> list[str]:
    return [name for name in task_names if TASKS[name].family == "generation"]


def understanding_task_names(task_names: list[str]) -> list[str]:
    return [name for name in task_names if TASKS[name].family == "understanding"]


def _normalize_generation_type(generation_type: str) -> str:
    normalized = generation_type.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"live_subject", "live"}:
        return "live_subject"
    if normalized in {"object", "objects"}:
        return "object"
    raise ValueError(f"Unsupported generation_type for pure_gen: {generation_type}")


def _format_pure_gen_prompt(template: str, *, unique_token: str, class_token: str) -> str:
    prompt = template.format(unique_token=unique_token, class_token=class_token)
    return " ".join(prompt.split())


def _understanding_manifest_path(
    output_root: str | Path,
    task_name: str,
    concept: str,
    token_epoch: int,
    model_epoch: int,
) -> Path:
    return (
        Path(output_root)
        / "artifacts"
        / "understanding"
        / task_name
        / concept
        / f"token_epoch_{token_epoch}"
        / f"model_epoch_{model_epoch}"
        / "manifest.json"
    )


def _load_understanding_predictions(
    *,
    output_root: str | Path | None,
    task_name: str,
    concept: str,
    token_epoch: int | None,
    model_epoch: int | None,
) -> tuple[Path, dict[str, dict]]:
    if output_root is None or token_epoch is None or model_epoch is None:
        raise ValueError(
            f"{task_name}_gen requires output_root, token_epoch, and model_epoch to locate the "
            f"upstream {task_name} manifest."
        )

    manifest_path = _understanding_manifest_path(
        output_root=output_root,
        task_name=task_name,
        concept=concept,
        token_epoch=token_epoch,
        model_epoch=model_epoch,
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing upstream understanding manifest for {task_name}_gen: {manifest_path}. "
            f"Run task `{task_name}` first, or include both tasks in one run."
        )

    payload = load_json(manifest_path)
    predictions_by_source: dict[str, dict] = {}
    for item in payload.get("items", []):
        source_prompt = str(item.get("source_prompt", "")).strip()
        if source_prompt:
            predictions_by_source[source_prompt] = item
    return manifest_path, predictions_by_source


def build_generation_prompt_specs(
    task_name: str,
    concept: str,
    data_root: str | Path,
    prompts_to_eval_path: str | Path,
    inverse_prompt: bool,
    nums_new_token_i_stage_1: int,
    nums_new_token_i_stage_2: int,
    reference_image_count: int = 1,
    clip_i_reference_image_count: int = 1,
    output_root: str | Path | None = None,
    token_epoch: int | None = None,
    model_epoch: int | None = None,
) -> list[PromptSpec]:
    data_root = Path(data_root)
    test_conditions = load_json(data_root / "concept" / "test" / concept / "t2i_conditions.json")
    train_info = load_json(data_root / "concept" / "train" / concept / "info.json")
    tp_prefix_text = build_tp_prefix_text(concept, train_info)
    conditioning_text = build_ip_prefix_text(concept)
    reference_image_paths = load_reference_image_paths(data_root, concept, count=reference_image_count)
    clip_i_reference_image_paths = load_test_ground_truth_image_paths(
        data_root,
        concept,
        count=clip_i_reference_image_count,
    )

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
        del prompts_to_eval_path
        generation_type = _normalize_generation_type(train_info["generation_type"])
        raw_templates = (
            PURE_GEN_LIVE_SUBJECT_TEMPLATES
            if generation_type == "live_subject"
            else PURE_GEN_OBJECT_TEMPLATES
        )
        unique_token = concept
        class_token = train_info["class"]
        return [
            PromptSpec(
                prompt_id=f"prompt_{idx:03d}",
                source_prompt=_format_pure_gen_prompt(
                    template,
                    unique_token=unique_token,
                    class_token="",
                ),
                generation_prompt=apply_inverse_prompt(
                    _format_pure_gen_prompt(
                        template,
                        unique_token=unique_token,
                        class_token="",
                    ),
                    unique_token,
                ),
                scoring_prompt=_format_pure_gen_prompt(
                    template,
                    unique_token="",
                    class_token=class_token,
                ),
                baseline_prompt=_format_pure_gen_prompt(
                    template,
                    unique_token="",
                    class_token=class_token,
                ),
                tp_prefix_text=tp_prefix_text,
                conditioning_text=conditioning_text,
                reference_image_paths=reference_image_paths,
                metadata={
                    "source": "pure_gen_templates",
                    "generation_type": generation_type,
                    "clip_i_reference_image_paths": clip_i_reference_image_paths,
                    "clip_i_reference_source": "concept_test_ground_truth",
                },
            )
            for idx, template in enumerate(raw_templates)
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
                tp_prefix_text=tp_prefix_text,
                conditioning_text=conditioning_text,
                reference_image_paths=reference_image_paths,
                metadata={
                    "source": "dense_prompt_generation",
                    "gpt_scoring_clauses": [clause.strip() for clause in prompt.split(",") if clause.strip()],
                    "clip_i_reference_image_paths": clip_i_reference_image_paths,
                    "clip_i_reference_source": "concept_test_ground_truth",
                },
            )
            for idx, prompt in enumerate(raw_prompts)
        ]

    if task_name == "rea_gen":
        prompts = test_conditions["personalized_driven_generation"]
        explicit_prompts = test_conditions["explicit_personalized_driven_generation"]
        num_testing = int(train_info["num_testing_info"])
        prompts = prompts[-num_testing:]
        explicit_prompts = explicit_prompts[-num_testing:]
        rea_manifest_path, rea_predictions = _load_understanding_predictions(
            output_root=output_root,
            task_name="rea",
            concept=concept,
            token_epoch=token_epoch,
            model_epoch=model_epoch,
        )
        return [
            PromptSpec(
                prompt_id=f"prompt_{idx:03d}",
                source_prompt=prompt,
                generation_prompt=apply_inverse_prompt(
                    str(rea_predictions[prompt]["prediction"]).strip(),
                    f"<{concept}>",
                ),
                scoring_prompt=explicit_prompt,
                baseline_prompt=str(rea_predictions[prompt]["prediction"]).strip().replace(
                    f"<{concept}>",
                    train_info["class"],
                ),
                tp_prefix_text=tp_prefix_text,
                conditioning_text=conditioning_text,
                reference_image_paths=reference_image_paths,
                metadata={
                    "source": "rea_prediction",
                    "generation_input_mode": "text_only",
                    "rea_manifest_path": str(rea_manifest_path),
                    "rea_item_id": rea_predictions[prompt].get("item_id"),
                    "rea_prediction": str(rea_predictions[prompt]["prediction"]).strip(),
                    "ground_truth_source": "explicit_personalized_driven_generation",
                    "clip_i_reference_image_paths": clip_i_reference_image_paths,
                    "clip_i_reference_source": "concept_test_ground_truth",
                },
            )
            for idx, (prompt, explicit_prompt) in enumerate(zip(prompts, explicit_prompts))
        ]

    if task_name == "dense_rea_gen":
        prompts = test_conditions["personalized_dense_prompt_generation"]
        explicit_prompts = test_conditions["explicit_personalized_dense_prompt_generation"]
        dense_rea_manifest_path, dense_rea_predictions = _load_understanding_predictions(
            output_root=output_root,
            task_name="dense_rea",
            concept=concept,
            token_epoch=token_epoch,
            model_epoch=model_epoch,
        )
        return [
            PromptSpec(
                prompt_id=f"prompt_{idx:03d}",
                source_prompt=prompt,
                generation_prompt=apply_inverse_prompt(
                    str(dense_rea_predictions[prompt]["prediction"]).strip(),
                    f"<{concept}>",
                ),
                scoring_prompt=explicit_prompt,
                baseline_prompt=str(dense_rea_predictions[prompt]["prediction"]).strip().replace(
                    f"<{concept}>",
                    train_info["class"],
                ),
                tp_prefix_text=tp_prefix_text,
                conditioning_text=conditioning_text,
                reference_image_paths=reference_image_paths,
                metadata={
                    "source": "dense_rea_prediction",
                    "generation_input_mode": "text_only",
                    "dense_rea_manifest_path": str(dense_rea_manifest_path),
                    "dense_rea_item_id": dense_rea_predictions[prompt].get("item_id"),
                    "dense_rea_prediction": str(dense_rea_predictions[prompt]["prediction"]).strip(),
                    "gpt_scoring_clauses": [clause.strip() for clause in explicit_prompt.split(",") if clause.strip()],
                    "ground_truth_source": "explicit_personalized_dense_prompt_generation",
                    "clip_i_reference_image_paths": clip_i_reference_image_paths,
                    "clip_i_reference_source": "concept_test_ground_truth",
                },
            )
            for idx, (prompt, explicit_prompt) in enumerate(zip(prompts, explicit_prompts))
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
    tp_prefix_text = build_tp_prefix_text(concept, train_info)
    conditioning_text = build_ip_prefix_text(concept)
    reference_image_paths = load_reference_image_paths(data_root, concept, count=reference_image_count)
    class_name = train_info["class"]

    raw_task_name = "text_only" if task_name in {"qa", "text_only"} else task_name

    if task_name in {"rec", "vqa", "qa", "text_only"}:
        test_data = load_json(data_root / "test_data" / f"{concept}.json")
        filtered_items = [item for item in test_data if item["type"] == raw_task_name]
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
                    tp_prefix_text=tp_prefix_text,
                    conditioning_text=conditioning_text,
                    reference_image_paths=reference_image_paths,
                    prepend_system_prompt=True,
                    metadata={
                        "source": "test_data",
                        "raw_type": item["type"],
                        "evaluation_task": task_name,
                    },
                )
            )
        return examples

    test_conditions = load_json(data_root / "concept" / "test" / concept / "t2i_conditions.json")
    black_image_path = str(data_root / "black_512x512.png")

    if task_name == "rea":
        prompts = test_conditions["personalized_driven_generation"]
        explicit_prompts = test_conditions["explicit_personalized_driven_generation"]
        num_testing = int(train_info["num_testing_info"])
        prompts = prompts[-num_testing:]
        explicit_prompts = explicit_prompts[-num_testing:]
        extra_info_items = list(train_info["extra_info"])
        extra_info_text = str(extra_info_items)
        baseline_extra_info_text = str(
            [item.replace(f"<{concept}>", f"the reference {class_name}") for item in extra_info_items]
        )
        return [
            UnderstandingExampleSpec(
                item_id=f"{task_name}_{idx:03d}",
                source_prompt=prompt,
                model_prompt=(
                    f"Below is some information about <{concept}> : {extra_info_text}\n"
                    f"Please rewrite the following prompt into a more detailed prompt: {prompt.strip()}\n"
                    "If the prompt relates to a specific item from the aforementioned information list,\n"
                    "add only the relevant detail from that item into the rewritten prompt.\n"
                    "If the prompt does not relate to any item in the list,\n"
                    "return the original prompt unchanged.\n"
                    "Output only the rewritten prompt."
                ),
                scoring_query=prompt,
                ground_truth=explicit_prompt,
                image_path=black_image_path,
                baseline_prompt=(
                    f"Below is some information about the reference {class_name} : {baseline_extra_info_text}\n"
                    "Please rewrite the following prompt into a more detailed prompt: "
                    f"{prompt.replace(f'<{concept}>', f'the reference {class_name}').strip()}\n"
                    "If the prompt relates to a specific item from the aforementioned information list,\n"
                    "add only the relevant detail from that item into the rewritten prompt.\n"
                    "If the prompt does not relate to any item in the list,\n"
                    "return the original prompt unchanged.\n"
                    "Output only the rewritten prompt."
                ),
                tp_prefix_text=tp_prefix_text,
                conditioning_text=conditioning_text,
                reference_image_paths=reference_image_paths,
                prepend_system_prompt=False,
                metadata={
                    "source": "personalized_driven_generation",
                    "ground_truth_source": "explicit_personalized_driven_generation",
                    "num_testing_info": num_testing,
                },
            )
            for idx, (prompt, explicit_prompt) in enumerate(zip(prompts, explicit_prompts))
        ]

    if task_name == "dense_rea":
        prompts = test_conditions["personalized_dense_prompt_generation"]
        references = test_conditions["explicit_personalized_dense_prompt_generation"]
        if len(prompts) != len(references):
            raise ValueError(
                "personalized_dense_prompt_generation and "
                "explicit_personalized_dense_prompt_generation must have the same length"
            )

        extra_info_items = train_info.get("extra_info", [])
        if isinstance(extra_info_items, str):
            extra_info_items = [extra_info_items]
        extra_info_items = [item.strip() for item in extra_info_items if item.strip()]
        extra_info_text = str(extra_info_items)
        baseline_extra_info_text = str(
            [item.replace(f"<{concept}>", f"the reference {class_name}") for item in extra_info_items]
        )

        return [
            UnderstandingExampleSpec(
                item_id=f"{task_name}_{idx:03d}",
                source_prompt=prompt,
                model_prompt=(
                    f"Below is some information about <{concept}> : {extra_info_text}\n"
                    "Please rewrite the following prompt into a more detailed prompt: "
                    f"{prompt.strip()}\n"
                    "Use the useful details from one or more relevant items in the aforementioned "
                    "information list to make the rewritten prompt more detailed.\n"
                    "If none of the information is helpful for this prompt,\n"
                    "return the original prompt unchanged.\n"
                    "Output only the rewritten prompt."
                ),
                scoring_query=prompt,
                ground_truth=explicit_prompt,
                image_path=black_image_path,
                baseline_prompt=(
                    f"Below is some information about the reference {class_name} : {baseline_extra_info_text}\n"
                    "Please rewrite the following prompt into a more detailed prompt: "
                    f"{prompt.replace(f'<{concept}>', f'the reference {class_name}').strip()}\n"
                    "Use the useful details from one or more relevant items in the aforementioned "
                    "information list to make the rewritten prompt more detailed.\n"
                    "If none of the information is helpful for this prompt,\n"
                    "return the original prompt unchanged.\n"
                    "Output only the rewritten prompt."
                ),
                tp_prefix_text=tp_prefix_text,
                conditioning_text=conditioning_text,
                reference_image_paths=reference_image_paths,
                prepend_system_prompt=False,
                metadata={
                    "source": "personalized_dense_prompt_generation",
                    "ground_truth_source": "explicit_personalized_dense_prompt_generation",
                    "extra_info": extra_info_items,
                    "evaluation_task": task_name,
                },
            )
            for idx, (prompt, explicit_prompt) in enumerate(zip(prompts, references))
        ]

    raise ValueError(f"Unsupported understanding task: {task_name}")
