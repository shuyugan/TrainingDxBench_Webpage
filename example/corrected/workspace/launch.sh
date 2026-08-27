#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)"
cd "$WORKSPACE_ROOT"

PYTHON="${PYTHON:-python3}"
GPU_SPEC=""
if (( $# == 1 )); then
  if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "usage: bash launch.sh [--gpus <id-list> | <id-list>]"
    exit 0
  fi
  GPU_SPEC="$1"
elif (( $# == 2 )) && [[ "$1" == "--gpus" ]]; then
  GPU_SPEC="$2"
elif (( $# != 0 )); then
  echo "usage: bash launch.sh [--gpus <id-list> | <id-list>]" >&2
  exit 1
fi
REQUESTED_GPU_COUNT=""
if [[ -n "$GPU_SPEC" ]]; then
  CUDA_VISIBLE_DEVICES="$(
    "$PYTHON" -c \
      'import sys; v=sys.argv[1].strip().lower().replace("cuda","").replace(":",",").replace(" ",""); p=v.split(","); ok=bool(v) and all(x.isdigit() for x in p); ids=[str(int(x)) for x in p] if ok else []; ok=ok and len(ids)==len(set(ids)); print(",".join(ids)) if ok else sys.exit(2)' \
      "$GPU_SPEC"
  )" || {
    echo "invalid GPU list: $GPU_SPEC" >&2
    exit 1
  }
  export CUDA_VISIBLE_DEVICES
  IFS=',' read -r -a GPU_IDS <<<"$CUDA_VISIBLE_DEVICES"
  REQUESTED_GPU_COUNT="${#GPU_IDS[@]}"
fi

export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTHONHASHSEED=108
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=source
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

GPU_COUNT="$(
  "$PYTHON" -c \
    'import torch; print(torch.cuda.device_count())'
)"
if [[ -n "$REQUESTED_GPU_COUNT" && "$GPU_COUNT" != "$REQUESTED_GPU_COUNT" ]]; then
  echo "requested GPUs are not all available: $GPU_SPEC" >&2
  exit 1
fi
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
  --output-dir training

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PYTHON" source/evaluate.py \
  --base-model-dir model/base \
  --records data/validation.jsonl \
  --output-dir training/base_validation \
  --device cuda:0

"$PYTHON" source/evaluate.py \
  --base-model-dir model/base \
  --adapter-dir model/adapter \
  --records data/dev.jsonl \
  --output-dir training/dev_validation \
  --device cuda:0

"$PYTHON" source/evaluate.py \
  --base-model-dir model/base \
  --adapter-dir model/adapter \
  --records data/validation.jsonl \
  --output-dir training/final_validation \
  --device cuda:0
