# EndoVis Fine-Tuning Guide

This guide describes the complete workflow for adapting LocateAnything to the
EndoVis surgical instrument data in this repository:

1. Convert EndoVis COCO annotations and MP4 frames into LocateAnything data.
2. Fine-tune `nvidia/LocateAnything-3B` with LoRA.
3. Monitor training with offline Weights & Biases.
4. Evaluate on the held-out EndoVis validation split.

The commands assume they are run from the repository root unless noted.

---

## 1. Input Data Layout

The converter expects the EndoVis files under `data/endovis/`:

```text
data/endovis/
├── 1_fps1.mp4
├── 1_fps1_coco.json
├── 2_fps1.mp4
├── 2_fps1_coco.json
├── ...
├── 7_fps1.mp4
└── 7_fps1_coco.json
```

Each `*_coco.json` file must use COCO-style fields:

- `images`: frame id, width, height, and `file_name`
- `annotations`: `image_id`, `category_id`, and `[x, y, w, h]` boxes
- `categories`: EndoVis instrument class names

The MP4 frame index is inferred from the JPEG stem. For example,
`0000000044.jpg` is extracted from frame `44` of `1_fps1.mp4`.

---

## 2. Convert EndoVis to LocateAnything Format

For a single-class surgical-instrument-wrist model, collapse all EndoVis tool
types into one label and keep only one grounding-style sample per selected
frame:

```bash
python Embodied/tools/prepare_endovis_locany.py \
  --endovis-dir data/endovis \
  --output-dir data/endovis_locany_single \
  --collapse-class \
  --collapsed-label "surgical instrument wrist" \
  --frame-stride 5 \
  --modes grounding
```

This writes:

```text
data/endovis_locany_single/
├── annotations/
│   ├── endovis_train.jsonl       # training samples
│   ├── endovis_val.jsonl         # validation samples in training format
│   ├── endovis_val_eval.jsonl    # validation samples for evaluation scripts
│   └── endovis_val_coco.json     # validation GT for COCO AP/mAP
├── images/
│   ├── 1_fps1/*.jpg
│   ├── ...
│   └── 7_fps1/*.jpg
└── endovis_recipe.json           # pass this as META_PATH during training
```

Default split:

- Train: all videos except `7_fps1`
- Validation: `7_fps1`

Override the held-out video list if needed:

```bash
python Embodied/tools/prepare_endovis_locany.py \
  --endovis-dir data/endovis \
  --output-dir data/endovis_locany_single \
  --collapse-class \
  --frame-stride 5 \
  --modes grounding \
  --val-videos 6_fps1,7_fps1
```

### Important Options

- `--collapse-class`: map every EndoVis category to one label.
- `--collapsed-label`: collapsed label, default `surgical instrument wrist`.
- `--frame-stride N`: keep every Nth annotated frame per video. Use this to
  reduce near-duplicate neighboring frames.
- `--modes grounding`: recommended for collapsed single-class training. The
  default `detection,grounding` creates two very similar samples per image.
- `--skip-frame-extraction`: only rewrite JSONL/recipe files. Use this after
  frames have already been extracted.
- `--data-augment`: opt in to LocateAnything resize augmentation. Do not use it
  for the default EndoVis single-GPU recipe unless you also raise sequence
  limits; augmentation can create more image tokens than a 4096 context can fit.

To regenerate only annotations after frames already exist:

```bash
python Embodied/tools/prepare_endovis_locany.py \
  --endovis-dir data/endovis \
  --output-dir data/endovis_locany_single \
  --collapse-class \
  --frame-stride 5 \
  --modes grounding \
  --skip-frame-extraction
```

---

## 3. Fine-Tune Locally or Interactively

Install the LocateAnything package and dependencies from `Embodied/`:

```bash
cd Embodied
pip install -e .
```

Run LoRA fine-tuning:

```bash
cd Embodied

META_PATH=../data/endovis_locany_single/endovis_recipe.json \
GPUS=1 \
bash shell/locate-anything-lora-endovis.sh \
  1 work_dirs/locany_lora_endovis_single
```

The EndoVis launcher defaults to:

- `MODEL_PATH=nvidia/LocateAnything-3B`
- `ATTN_IMPLEMENTATION=sdpa`
- `LOCANY_VISION_ATTN=sdpa`
- `USE_LLM_LORA=64`
- `USE_BACKBONE_LORA=0`
- `FREEZE_LLM=True`
- `FREEZE_BACKBONE=True`
- `FREEZE_MLP=False`
- `MAX_SEQ_LENGTH=4096`
- `MAX_NUM_TOKENS_PER_SAMPLE=4096`
- `MAX_NUM_TOKENS=4096`
- `REPORT_TO=wandb`
- `WANDB_MODE=offline`

Useful overrides:

```bash
META_PATH=../data/endovis_locany_single/endovis_recipe.json \
MODEL_PATH=/path/to/LocateAnything-3B \
MAX_STEPS=2000 \
SAVE_STEPS=200 \
LR=2e-5 \
GPUS=1 \
bash shell/locate-anything-lora-endovis.sh \
  1 work_dirs/locany_lora_endovis_single
```

If you enabled `--data-augment` during conversion and see image-token mismatch
warnings, either disable augmentation in `endovis_recipe.json` or raise:

```bash
MAX_SEQ_LENGTH=8192 \
MAX_NUM_TOKENS_PER_SAMPLE=8192 \
MAX_NUM_TOKENS=8192
```

For the default 64 GB single-GPU Slurm recipe, disabling data augmentation is
recommended.

---

## 4. Fine-Tune with Slurm

The repository includes a Leonardo-style Slurm script:

```text
Embodied/shell/slurm-locate-anything-lora-endovis.sbatch
```

Edit or override these environment variables for your cluster:

- `WORKDIR`: repository checkout, default `/leonardo_work/IscrC_FLAC/Eagle`
- `VENV_PATH`: Python environment, default `$WORKDIR/Embodied/.venv`
- `HF_HOME`: Hugging Face cache
- `MODEL_PATH`: model id or local checkpoint
- `META_PATH`: EndoVis recipe JSON
- `OUTPUT_DIR`: training output directory under `Embodied/`

Submit:

```bash
cd /leonardo_work/IscrC_FLAC/Eagle/Embodied
mkdir -p jobs/out
sbatch shell/slurm-locate-anything-lora-endovis.sbatch
```

Override settings at submission time:

```bash
cd /leonardo_work/IscrC_FLAC/Eagle/Embodied
mkdir -p jobs/out
sbatch --export=ALL,\
WORKDIR=/leonardo_work/IscrC_FLAC/Eagle,\
META_PATH=/leonardo_work/IscrC_FLAC/Eagle/data/endovis_locany_single/endovis_recipe.json,\
OUTPUT_DIR=work_dirs/locany_lora_endovis_single,\
MAX_STEPS=2000,\
SAVE_STEPS=200 \
shell/slurm-locate-anything-lora-endovis.sbatch
```

The script creates Slurm logs in:

```text
Embodied/jobs/out/
```

Training logs and checkpoints are written under:

```text
Embodied/work_dirs/locany_lora_endovis_single/
```

---

## 5. Offline Weights & Biases Logging

The EndoVis launcher defaults to offline W&B:

```bash
REPORT_TO=wandb
WANDB_MODE=offline
WANDB_PROJECT=locany-endovis
```

Offline runs are stored under:

```text
Embodied/work_dirs/locany_lora_endovis_single/wandb/
```

To sync later from a machine with network access:

```bash
cd /leonardo_work/IscrC_FLAC/Eagle/Embodied
wandb sync work_dirs/locany_lora_endovis_single/wandb/offline-run-*
```

To disable W&B and use TensorBoard:

```bash
REPORT_TO=tensorboard \
bash shell/locate-anything-lora-endovis.sh \
  1 work_dirs/locany_lora_endovis_single
```

Or with Slurm:

```bash
sbatch --export=ALL,REPORT_TO=tensorboard \
  Embodied/shell/slurm-locate-anything-lora-endovis.sbatch
```

---

## 6. Evaluate on the EndoVis Validation Split

The converter writes an evaluation JSONL:

```text
data/endovis_locany_single/annotations/endovis_val_eval.jsonl
```

and a COCO-format validation annotation file:

```text
data/endovis_locany_single/annotations/endovis_val_coco.json
```

It uses:

- `dataset_name=EndoVis`
- `task_name=referring_object_detection`
- absolute pixel-space GT boxes
- category `surgical instrument wrist` when `--collapse-class` is used

Run validation from `Embodied/`:

```bash
cd Embodied

GPUS=1 bash evaluation/scripts/eval_grounding.sh \
  --dataset EndoVis \
  --eval_type box_eval \
  --model_path work_dirs/locany_lora_endovis_single/checkpoint-1000 \
  --image_root ../data/endovis_locany_single/images \
  --test_jsonl ../data/endovis_locany_single/annotations/endovis_val_eval.jsonl \
  --output_dir work_dirs/locany_lora_endovis_single/eval_val
```

Outputs:

```text
work_dirs/locany_lora_endovis_single/eval_val/hybrid/
├── answer.jsonl
├── eval_results.json
└── evaluation_log_*.txt
```

The most useful metrics are precision, recall, and F1 at:

- IoU `0.5`
- IoU `0.75`
- mean over IoU thresholds

The EndoVis evaluation uses `referring_object_detection`, so matching is
category-agnostic after parsing predictions. This is appropriate for a
collapsed single-class model.

### COCO-Style AP/mAP

To compute COCO-style AP/mAP, install the FastEvaluate dependency described in
`evaluation/README.md`, then run:

```bash
cd Embodied

GPUS=1 bash evaluation/scripts/eval_endovis_map.sh \
  --model_path work_dirs/locany_lora_endovis_single/checkpoint-1000 \
  --endovis_root ../data/endovis_locany_single \
  --output_dir work_dirs/locany_lora_endovis_single/eval_val_map
```

This wraps the COCO/LVIS evaluation path:

1. `inference_detection_ddp.py` generates predictions.
2. `convert_coco_lvis_to_standard_format.py` converts predictions to FastEval TSV.
3. `coco_lvis_metric.py` reports COCO-style AP/AR summaries.

Outputs:

```text
work_dirs/locany_lora_endovis_single/eval_val_map/hybrid/
├── eval_results.jsonl
├── fast_eval.tsv
├── per_iou_metrics.json
├── map_over_iou.png
└── evaluation_log_*.txt
```

For collapsed single-class EndoVis, `endovis_val_coco.json` contains one COCO
category: `surgical instrument wrist`. AP is therefore single-class AP, not an
average over the original EndoVis tool-type classes.

The AP/mAP script also prints and saves the AP curve over the standard COCO IoU
thresholds:

```text
0.50, 0.55, 0.60, 0.65, 0.70,
0.75, 0.80, 0.85, 0.90, 0.95
```

`per_iou_metrics.json` stores AP, AR, and F1 per threshold. `map_over_iou.png`
plots AP and AR over those IoU thresholds.

### Diagnose Low mAP

Use the analyzer to separate count errors from localization errors:

```bash
cd Embodied

python tools/analyze_endovis_eval_results.py
```

By default, the analyzer looks for:

```text
Embodied/work_dirs/locany_lora_endovis_single/eval_val/hybrid/eval_results.json
Embodied/work_dirs/locany_lora_endovis_single/eval_val_map/hybrid/eval_results.jsonl
data/endovis_locany_single/annotations/endovis_val_eval.jsonl
data/endovis_locany_single/images
```

Use explicit paths to analyze a different run:

```bash
python tools/analyze_endovis_eval_results.py \
  --metrics-json work_dirs/locany_lora_endovis_single/eval_val/hybrid/eval_results.json \
  --pred-jsonl work_dirs/locany_lora_endovis_single/eval_val_map/hybrid/eval_results.jsonl \
  --gt-jsonl ../data/endovis_locany_single/annotations/endovis_val_eval.jsonl \
  --image-root ../data/endovis_locany_single/images \
  --out-dir work_dirs/locany_lora_endovis_single/eval_analysis \
  --draw-overlays
```

Outputs:

```text
work_dirs/locany_lora_endovis_single/eval_analysis/
├── summary.md
├── summary.json
├── threshold_metrics.csv
├── raw_threshold_metrics.csv
├── sample_errors.csv
├── box_diagnostics.csv
├── best_iou_histogram.png
├── center_offset_scatter.png
├── scale_bias_histograms.png
├── per_video_metrics.png
├── count_error_histogram.png
└── failure_montage/
```

`sample_errors.csv` is the fastest place to inspect bad frames. It labels
failure modes such as `empty_prediction`, `missed_gt`, `false_positive`,
`count_under`, `count_over`, `duplicate_prediction`, `low_iou_localization`,
and `good_at_50_bad_at_75`.

### Evaluate Multiple Checkpoints

Example loop:

```bash
cd Embodied

for ckpt in work_dirs/locany_lora_endovis_single/checkpoint-*; do
  name=$(basename "$ckpt")
  GPUS=1 bash evaluation/scripts/eval_grounding.sh \
    --dataset EndoVis \
    --eval_type box_eval \
    --model_path "$ckpt" \
    --image_root ../data/endovis_locany_single/images \
    --test_jsonl ../data/endovis_locany_single/annotations/endovis_val_eval.jsonl \
    --output_dir "work_dirs/locany_lora_endovis_single/eval_${name}"
done
```

AP/mAP loop:

```bash
cd Embodied

for ckpt in work_dirs/locany_lora_endovis_single/checkpoint-*; do
  name=$(basename "$ckpt")
  GPUS=1 bash evaluation/scripts/eval_endovis_map.sh \
    --model_path "$ckpt" \
    --endovis_root ../data/endovis_locany_single \
    --output_dir "work_dirs/locany_lora_endovis_single/eval_map_${name}"
done
```

---

## 7. Troubleshooting

### `FlashAttention2 has been toggled on`

Use the EndoVis launcher or export:

```bash
export LOCANY_VISION_ATTN=sdpa
```

The EndoVis scripts already set this by default.

### `image token mismatch: actual=4076, expected=...`

The image tokens were truncated. Most often the generated recipe has:

```json
"data_augment": true
```

Fix it:

```bash
python - <<'PY'
import json
p = "data/endovis_locany_single/endovis_recipe.json"
d = json.load(open(p))
for v in d.values():
    v["data_augment"] = False
json.dump(d, open(p, "w"), indent=2)
PY
```

Then restart training from a fresh output directory.

### Duplicate-Looking Samples in JSONL

If you use the default:

```bash
--modes detection,grounding
```

the same frame can appear twice with near-identical prompts. For collapsed
single-class EndoVis training, use:

```bash
--modes grounding
```

### Hugging Face Offline Mode

The Slurm script defaults to:

```bash
HF_HUB_OFFLINE=1
```

Make sure `MODEL_PATH` is either a local checkpoint directory or
`nvidia/LocateAnything-3B` already exists in `HF_HOME`.
