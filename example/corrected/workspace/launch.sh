#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:?usage: launch.sh OUTPUT_DIR}"
PYTHON="${PYTHON:-python}"

mkdir -p "$OUTPUT_DIR"
if find "$OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "output directory must be empty: $OUTPUT_DIR" >&2
  exit 1
fi

export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTHONHASHSEED=108
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=source

GPU_COUNT="$(
  "$PYTHON" -c \
    'import torch; print(torch.cuda.device_count())'
)"
if ! [[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "launch.sh requires at least one visible GPU; found $GPU_COUNT" >&2
  exit 1
fi

REFERENCE_GLOBAL_BATCH=16
DOCUMENTS_PER_PACK=4
MICROBATCH_GLOBAL_BATCH=$((GPU_COUNT * DOCUMENTS_PER_PACK))
if (( REFERENCE_GLOBAL_BATCH % MICROBATCH_GLOBAL_BATCH != 0 )); then
  echo \
    "GPU_COUNT=$GPU_COUNT cannot preserve global batch $REFERENCE_GLOBAL_BATCH" \
    "with $DOCUMENTS_PER_PACK documents per rank" >&2
  exit 1
fi

"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$GPU_COUNT" \
  source/train.py \
  --base-model-dir model/base \
  --train-records data/train.jsonl \
  --output-dir "$OUTPUT_DIR/training"

"$PYTHON" source/evaluate.py \
  --base-model-dir model/base \
  --adapter-dir "$OUTPUT_DIR/training/adapter" \
  --records data/dev.jsonl \
  --output-dir "$OUTPUT_DIR/dev_validation" \
  --device cuda:0

"$PYTHON" source/evaluate.py \
  --base-model-dir model/base \
  --adapter-dir "$OUTPUT_DIR/training/adapter" \
  --records data/validation.jsonl \
  --output-dir "$OUTPUT_DIR/final_validation" \
  --device cuda:0
