# Evaluation Pipeline

## Structure

- `eval_pipeline.py`: single entry for raw prediction generation and scoring.
- `evaluation/user_settings.py`: edit this file directly for default runtime parameters.
- `evaluation/tasks.py`: task registry and dataset parsing for both generation and understanding.
- `evaluation/adapters/`: model adapters. The working local implementation is `showo_weighted`.
- `evaluation/adapters/external_baseline.py`: shared scaffold for external TP/IP baselines.
- `evaluation/adapters/janus_base.py`: Janus-Pro runtime for multimodal understanding and text-to-image generation.
- `evaluation/scorers/`: CLIP, GPT, and text-based scoring utilities.

## Recommended workflow

1. Edit `evaluation/user_settings.py`
2. Set your `adapter`, Bagel code/model paths, GPT model, GPT key, and smoke-test defaults
3. Run `python D:\unicr1\Sync-R1\eval_pipeline.py` or `python D:\unicr1\Sync-R1\scripts\run_parallel_eval.py`
4. Use `python D:\unicr1\Sync-R1\scripts\smoke_test_preflight.py` for a dependency/path sanity check
5. Use `python D:\unicr1\Sync-R1\scripts\smoke_test_eval.py` for a minimal real run

## Two-stage flow

1. `--mode generate`
   - Generation tasks: save images and `manifest.json`
   - Understanding tasks: save text predictions and `manifest.json`
2. `--mode score`
   - Read manifests
   - Compute CLIP / GPT / BLEU / weighted recall style metrics
   - Write per-concept and per-epoch summaries

## Task names

- Generation: `pure_gen`, `dense_gen`, `rea_gen`, `dense_rea_gen`
- Understanding: `rec`, `vqa`, `qa`, `rea`, `dense_rea`
- Legacy alias: `text_only` is still accepted and maps to `qa`

## Reproducibility defaults

- Global seed defaults to `3407`
- GPT judge temperature defaults to `0.00001`
- External adapter temperature also defaults to `0.00001`
- Baseline prompts are intentionally short and plain
- GPT model version is controlled in `evaluation/user_settings.py`

## Adapters

- `showo` / `showo_weighted`: current local baseline with token weights and optional RL checkpoint
- `bagel_tp`: scaffold for Bagel text-prompt baseline
- `bagel_ip`: scaffold for Bagel image-prompt baseline
- `janus_tp`: Janus-Pro text-prompt baseline for both understanding and generation
- `janus_ip`: Janus-Pro image-prompt baseline; understanding uses reference images directly, and generation first turns the reference image(s) into a short text description because Janus generation itself is text-only

## Common commands

Show-o end-to-end:

```bash
python D:\unicr1\Sync-R1\eval_pipeline.py ^
  --mode run ^
  --adapter showo ^
  --tasks all ^
  --concepts all
```

Bagel TP baseline:

```bash
python D:\unicr1\Sync-R1\eval_pipeline.py ^
  --mode run ^
  --adapter bagel_tp ^
  --tasks all ^
  --concepts all ^
  --epoch-to-load 0 ^
  --model-epochs 0 ^
  --adapter-code-root D:\path\to\BAGEL ^
  --adapter-model-id D:\path\to\BAGEL-7B-MoT ^
  --adapter-max-memory-per-gpu 20GiB
```

Bagel IP baseline:

```bash
python D:\unicr1\Sync-R1\eval_pipeline.py ^
  --mode run ^
  --adapter bagel_ip ^
  --tasks all ^
  --concepts all ^
  --epoch-to-load 0 ^
  --model-epochs 0 ^
  --reference-image-count 1 ^
  --adapter-code-root D:\path\to\BAGEL ^
  --adapter-model-id D:\path\to\BAGEL-7B-MoT ^
  --adapter-max-memory-per-gpu 20GiB
```

Parallel Bagel generation + scoring:

```bash
python D:\unicr1\Sync-R1\scripts\run_parallel_eval.py ^
  --gpu-groups 0,1 2,3 ^
  --mode run ^
  --adapter bagel_ip ^
  --tasks all ^
  --concepts all ^
  --model-epochs 0 ^
  -- ^
  --epoch-to-load 0 ^
  --reference-image-count 1 ^
  --adapter-code-root D:\path\to\BAGEL ^
  --adapter-model-id D:\path\to\BAGEL-7B-MoT ^
  --adapter-max-memory-per-gpu 20GiB
```

Janus-Pro-1B TP baseline:

```bash
python D:\unicr1\Sync-R1\eval_pipeline.py ^
  --mode run ^
  --adapter janus_tp ^
  --tasks all ^
  --concepts all ^
  --epoch-to-load 0 ^
  --model-epochs 0 ^
  --adapter-model-id deepseek-ai/Janus-Pro-1B ^
  --adapter-image-height 384 ^
  --adapter-image-width 384
```

Janus-Pro-7B IP baseline:

```bash
python D:\unicr1\Sync-R1\eval_pipeline.py ^
  --mode run ^
  --adapter janus_ip ^
  --tasks all ^
  --concepts all ^
  --epoch-to-load 0 ^
  --model-epochs 0 ^
  --reference-image-count 1 ^
  --adapter-model-id deepseek-ai/Janus-Pro-7B ^
  --adapter-code-root D:\path\to\Janus ^
  --adapter-image-height 384 ^
  --adapter-image-width 384
```

## Important parameters

- `--seed`: one global seed for generation and understanding
- `--reference-image-count`: how many train reference images to pass to IP baselines
- `--adapter-code-root`: local BAGEL source repo root, containing `inferencer.py`, `data/`, and `modeling/`
- `--adapter-model-id`: local checkpoint directory or repo id for external baselines; for Janus use either `deepseek-ai/Janus-Pro-1B`, `deepseek-ai/Janus-Pro-7B`, or a local checkpoint path
- `--adapter-max-memory-per-gpu`: max memory string passed to `infer_auto_device_map`, such as `20GiB`
- `--adapter-offload-dir`: offload folder for multi-GPU / disk offload
- `--adapter-image-height`, `--adapter-image-width`: requested output image size for generation; Janus-Pro generation expects dimensions divisible by `16` and typically uses `384x384`
- `--adapter-cfg-text-scale`, `--adapter-cfg-img-scale`, `--adapter-cfg-interval-start`
- `--adapter-timestep-shift`, `--adapter-num-timesteps`, `--adapter-cfg-renorm-min`, `--adapter-cfg-renorm-type`
- `--adapter-use-thinking`: optional, off by default for simple baseline prompts
- `--adapter-temperature`, `--adapter-top-p`, `--adapter-top-k`, `--adapter-max-new-tokens`: future external baseline decoding controls
- `--gpt-model`, `--gpt-api-key`, `--gpt-base-url`: judge model settings
- `--gpt-temperature`, `--gpt-max-tokens`, `--gpt-timeout`: GPT scorer controls

## Add a new model later

To finish a new external adapter later:

1. Reuse one of `bagel_tp.py`, `bagel_ip.py`, `janus_tp.py`, `janus_ip.py`
2. Implement model calls in `ExternalBaselineAdapter._generate_images_for_prompt`
3. Implement model calls in `ExternalBaselineAdapter._predict_text_for_example`
4. Keep the same pipeline command and only switch `--adapter`

The task definitions and scorers do not need to change if the new adapter can:

- generate images for generation tasks
- produce text answers for understanding tasks
