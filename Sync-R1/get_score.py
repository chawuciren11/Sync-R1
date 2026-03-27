import json
import os


def _load_score_block(data: dict, *keys: str) -> dict:
    block = data
    for key in keys:
        block = block.get(key, {})
    return block if isinstance(block, dict) else {}


def calculate_average_scores(concept_list, epoch, mode, log_dir="./logs"):
    total_rea_bleu = 0.0
    total_rea_gpt = 0.0
    total_dense_rea_gpt = 0.0
    total_weight = 0.0
    file_count = 0

    for concept in concept_list:
        path = os.path.join(log_dir, concept, f"epoch_{epoch}.json")
        if not os.path.exists(path):
            print(f"Missing score file: {path}")
            continue

        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)

            if mode == "reasoning":
                rea_score = _load_score_block(data, "rea", "score")
                dense_rea_score = _load_score_block(data, "dense-rea", "score")
                total_rea_bleu += rea_score.get("bleu", 0.0)
                total_rea_gpt += rea_score.get("gpt", rea_score.get("ds-score", 0.0))
                total_dense_rea_gpt += dense_rea_score.get(
                    "gpt",
                    dense_rea_score.get("ds-score", 0.0),
                )
            elif mode == "base":
                vqa_score = _load_score_block(data, "vqa", "score")
                qa_score = _load_score_block(data, "qa", "score") or _load_score_block(
                    data,
                    "text_only",
                    "score",
                )
                rec_score = _load_score_block(data, "rec", "score")
                total_rea_bleu += vqa_score.get("bleu", 0.0)
                total_rea_gpt += vqa_score.get("gpt", vqa_score.get("ds-score", 0.0))
                total_dense_rea_gpt += qa_score.get("gpt", qa_score.get("ds-score", 0.0))
                total_weight += rec_score.get("weight", rec_score.get("accuracy", 0.0))
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            file_count += 1
            print(f"Loaded: {path}")
        except Exception as exc:
            print(f"Failed to read {path}: {exc}")

    if file_count == 0:
        print("No score files were loaded.")
        return None

    result = {
        "rea.score.bleu_avg": total_rea_bleu / file_count,
        "rea.score.gpt_avg": total_rea_gpt / file_count,
        "dense-rea.score.gpt_avg": total_dense_rea_gpt / file_count,
    }
    if mode == "base":
        result["weight"] = total_weight / file_count
    return result


if __name__ == "__main__":
    concept_list = [
        "adrien_brody",
        "b_jordan",
        "butin",
        "coco",
        "dunpai",
        "fine_woolfhard",
        "gold_pineapple",
        "leonardo",
        "maeve_dog",
        "mam",
        "ningning",
        "pig_cup",
        "wangkai",
    ]

    averages = calculate_average_scores(
        concept_list,
        2,
        "reasoning",
        log_dir="./logs_reasoning",
    )

    if averages:
        print("\nAverage scores:")
        for key, value in averages.items():
            print(f"{key}: {value:.4f}")
