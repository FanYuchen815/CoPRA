#!/usr/bin/env bash
# Wrapper to run CoPRA while preventing any writes to /root.
# Usage: ./scripts/run_no_root.sh -- python run.py finetune pretune --model_config ...

set -euo pipefail

# Base dir under /root/autodl-tmp where all writes will be redirected
BASE_DIR="/root/autodl-tmp"
PROJECT_DIR="$BASE_DIR/CoPRA"

# Create isolated home and cache locations inside BASE_DIR
mkdir -p "$BASE_DIR/home" "$BASE_DIR/cache" "$BASE_DIR/tmp" "$BASE_DIR/torch_cache" "$BASE_DIR/hf_cache" "$BASE_DIR/cuda_cache" "$PROJECT_DIR/outputs" "$PROJECT_DIR/cache"
chown -R $(id -u):$(id -g) "$BASE_DIR"

# Export environment variables to redirect common cache/temp paths
export HOME="$BASE_DIR/home"
export XDG_CACHE_HOME="$BASE_DIR/cache"
export TMPDIR="$BASE_DIR/tmp"
export PYTORCH_HOME="$BASE_DIR/torch_cache"
export TORCH_HOME="$BASE_DIR/torch_cache"
export HF_HOME="$BASE_DIR/hf_cache"
export CUDA_CACHE_PATH="$BASE_DIR/cuda_cache"

# Ensure Python user base and pip cache are inside BASE_DIR
export PIP_CACHE_DIR="$BASE_DIR/cache/pip"
mkdir -p "$PIP_CACHE_DIR"

# Prevent Python from writing .local under /root
export PYTHONUSERBASE="$BASE_DIR/pyuserbase"
mkdir -p "$PYTHONUSERBASE"

echo "Redirecting HOME and caches to $BASE_DIR (no writes to /root will be performed)"
echo "HOME=$HOME"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"

# If no args provided, print help
if [ $# -eq 0 ]; then
  echo "Usage: $0 -- <command>"
  echo "Example: $0 -- python run.py finetune pretune --model_config ./config/models/copra.yml --data_config ./config/datasets/PRI30k.yml --run_config ./config/runs/pretune_struct.yml"
  exit 1
fi

# Shift to project dir and run the provided command after `--`
# The wrapper expects `--` before the target command to avoid accidental argument parsing
if [ "$1" = "--" ]; then
  shift
  cd "$PROJECT_DIR"
  exec "$@"
else
  echo "Please separate wrapper args and command with --"
  exit 1
fi
