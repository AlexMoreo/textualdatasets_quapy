#!/bin/bash
# Sequential roberta-base embedding generation across all 9 text datasets.
# Run with: conda run -n text_embed bash generate_data_roberta.sh
set -o pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Override CUDA_VISIBLE_DEVICES from the calling shell to pick a different GPU; defaults to
# the first one. (On a machine with a driver/kernel-module version mismatch, PyTorch's
# NVML-based peer-to-peer access check can fail unless exactly one GPU is visible.)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Force the HF cache to a directory we actually own: some shared servers pre-set HF_HOME (or,
# with higher precedence, HF_HUB_CACHE/HUGGINGFACE_HUB_CACHE/HF_DATASETS_CACHE directly) to a
# system-wide cache that regular users can't write new lock files into (PermissionError on
# .locks/...). Overriding HF_HOME alone is not enough -- huggingface_hub and datasets each read
# their own, more specific var first if it's set, which the server evidently does. Unconditional
# overrides here (not just defaulting when unset), since the problem is an inherited,
# already-set-but-broken value, not a missing one.
HF_CACHE_ROOT="$(pwd)/.hf_cache"
export HF_HOME="$HF_CACHE_ROOT"
export HF_HUB_CACHE="$HF_CACHE_ROOT/hub"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_ROOT/hub"
export HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets"
export HF_ASSETS_CACHE="$HF_CACHE_ROOT/assets"
export TRANSFORMERS_CACHE="$HF_CACHE_ROOT/hub"

LOGDIR=logs
mkdir -p "$LOGDIR"

declare -A NEPOCHS
NEPOCHS[default]=5
NEPOCHS[ag_news]=1  # saturates fast at this size; more epochs overfit

FAILED=()

for dataset in yelp imdb ag_news rt sst5 20newsgroups dbpedia yahoo_answers trec ; do
  nepochs=${NEPOCHS[$dataset]:-${NEPOCHS[default]}}
  echo "=== $dataset (nepochs=$nepochs) ==="
  python train_and_embed.py \
    --nepochs "$nepochs" \
    --model_name FacebookAI/roberta-base \
    --dataset_name "$dataset" \
    --val_size_cap 10000 \
    --device cuda \
    --train_backbone \
    2>&1 | tee "$LOGDIR/${dataset}_roberta-base.log"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "!!! $dataset FAILED, continuing with the rest !!!"
    FAILED+=("$dataset")
  fi
done

if [ ${#FAILED[@]} -eq 0 ]; then
  echo "ALL DONE"
else
  echo "DONE WITH FAILURES: ${FAILED[*]}"
fi
