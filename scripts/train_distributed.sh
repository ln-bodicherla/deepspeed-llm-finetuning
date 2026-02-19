#!/bin/bash
# Distributed training launcher for DeepSpeed fine-tuning.
# Supports single-node multi-GPU and multi-node configurations.

set -euo pipefail

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-4}"
NUM_NODES="${NUM_NODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training_config.yaml}"
DS_CONFIG="${DS_CONFIG:-configs/ds_config_zero2.json}"
HOSTFILE="${HOSTFILE:-}"

OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
LOG_DIR="${LOG_DIR:-./logs}"

# ──────────────────────────────────────────────────────────────
# Environment Setup
# ──────────────────────────────────────────────────────────────
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-~/.cache/huggingface}"
export HF_HOME="${HF_HOME:-~/.cache/huggingface}"

# NCCL configuration for optimal multi-GPU communication
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_TREE_THRESHOLD=0
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

# CUDA configuration
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# ──────────────────────────────────────────────────────────────
# Directory Setup
# ──────────────────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

# ──────────────────────────────────────────────────────────────
# Pre-flight Checks
# ──────────────────────────────────────────────────────────────
echo "================================================="
echo "DeepSpeed Distributed Training Launcher"
echo "================================================="
echo "Timestamp:        ${TIMESTAMP}"
echo "Num GPUs:         ${NUM_GPUS}"
echo "Num Nodes:        ${NUM_NODES}"
echo "Master Address:   ${MASTER_ADDR}"
echo "Master Port:      ${MASTER_PORT}"
echo "Training Config:  ${TRAINING_CONFIG}"
echo "DeepSpeed Config: ${DS_CONFIG}"
echo "Output Dir:       ${OUTPUT_DIR}"
echo "Log File:         ${LOG_FILE}"
echo "================================================="

# Verify configs exist
if [ ! -f "${TRAINING_CONFIG}" ]; then
    echo "ERROR: Training config not found: ${TRAINING_CONFIG}"
    exit 1
fi

if [ ! -f "${DS_CONFIG}" ]; then
    echo "ERROR: DeepSpeed config not found: ${DS_CONFIG}"
    exit 1
fi

# Verify GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. Ensure GPUs are available."
fi

# Verify DeepSpeed installation
if ! command -v deepspeed &> /dev/null; then
    echo "ERROR: deepspeed command not found. Install with: pip install deepspeed"
    exit 1
fi

echo "DeepSpeed version: $(python -c 'import deepspeed; print(deepspeed.__version__)')"
echo "PyTorch version:   $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available:    $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# ──────────────────────────────────────────────────────────────
# Build DeepSpeed Command
# ──────────────────────────────────────────────────────────────
DS_CMD="deepspeed"

if [ -n "${HOSTFILE}" ] && [ -f "${HOSTFILE}" ]; then
    DS_CMD="${DS_CMD} --hostfile=${HOSTFILE}"
    echo "Using hostfile: ${HOSTFILE}"
fi

DS_CMD="${DS_CMD} --num_gpus=${NUM_GPUS}"
DS_CMD="${DS_CMD} --num_nodes=${NUM_NODES}"
DS_CMD="${DS_CMD} --master_addr=${MASTER_ADDR}"
DS_CMD="${DS_CMD} --master_port=${MASTER_PORT}"

DS_CMD="${DS_CMD} src/trainer.py"
DS_CMD="${DS_CMD} --config ${TRAINING_CONFIG}"
DS_CMD="${DS_CMD} --deepspeed ${DS_CONFIG}"
DS_CMD="${DS_CMD} --output_dir ${OUTPUT_DIR}"

# ──────────────────────────────────────────────────────────────
# Launch Training
# ──────────────────────────────────────────────────────────────
echo "Launching training..."
echo "Command: ${DS_CMD}"
echo ""
echo "Logging to: ${LOG_FILE}"
echo "================================================="

${DS_CMD} 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "================================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Checkpoints saved to: ${OUTPUT_DIR}"
else
    echo "Training failed with exit code: ${EXIT_CODE}"
    echo "Check log file: ${LOG_FILE}"
fi
echo "================================================="

exit ${EXIT_CODE}
