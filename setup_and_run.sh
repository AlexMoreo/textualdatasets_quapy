#!/bin/bash
# Self-contained setup + run for the roberta-base text embedding extraction.
# Intended for a remote GPU server: creates a dedicated conda env, installs the exact
# versions this pipeline was tested against, then runs the full 9-dataset sequence.
#
# Usage:
#   bash setup_and_run.sh [conda_env_name]
#
# Requirements: a working `conda`, a CUDA-capable GPU, and a CUDA driver recent enough
# for the cu130 PyTorch build below. If the server's driver is older, change CUDA_TAG to
# match it (check with `nvidia-smi` -> "CUDA Version" in the top-right corner) -- common
# values are cu121, cu124, cu126, cu128, cu130.
set -e -o pipefail

ENV_NAME="${1:-text_embed}"
CUDA_TAG="cu130"

echo "=== creating conda env '$ENV_NAME' (python 3.12) ==="
conda create -n "$ENV_NAME" python=3.12 -y

echo "=== installing torch (+$CUDA_TAG) ==="
conda run -n "$ENV_NAME" pip install --index-url "https://download.pytorch.org/whl/$CUDA_TAG" torch==2.14.1

echo "=== installing the rest of the stack (pinned to the tested versions) ==="
conda run -n "$ENV_NAME" pip install \
  transformers==5.22.1 \
  datasets==5.1.0 \
  accelerate==1.15.0 \
  scikit-learn==1.9.1 \
  tqdm==4.70.1

echo "=== sanity check: CUDA + model load ==="
conda run --no-capture-output -n "$ENV_NAME" python -c "
import torch
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'CUDA not available -- check the driver/CUDA_TAG match'
x = torch.zeros(10).cuda()
print('GPU alloc OK:', x.device)
"

echo "=== running the full embedding generation (sequential, all 9 datasets) ==="
conda run --no-capture-output -n "$ENV_NAME" bash generate_data_roberta.sh

echo "=== done. Outputs are under ../data/{dataset}_roberta-base/ ==="
