#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-/home/fenrir/ubunto_data_2/worldmodel/data/piper_insert_mouse_battery_lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/piper_pi05_gr00t}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-${SCRIPT_DIR}/piper_pi05_config.py}"

USE_WANDB="${USE_WANDB:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
ACTION_HORIZON="${ACTION_HORIZON:-40}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"

export PIPER_PI05_ACTION_HORIZON="${ACTION_HORIZON}"
export USE_WANDB
export NUM_GPUS
export GLOBAL_BATCH_SIZE
export DATALOADER_NUM_WORKERS
export MAX_STEPS
export SAVE_STEPS
export ACTION_HORIZON
export SHARD_SIZE
export EPISODE_SAMPLING_RATE
export NUM_SHARDS_PER_EPOCH

cd "${REPO_ROOT}"

if [[ "${USE_WANDB}" = "1" ]]; then
  echo "[PiperPi05] wandb enabled: project=${WANDB_PROJECT:-finetune-gr00t-piper-pi05}, run=${EXPERIMENT_NAME:-piper_pi05_gr00t}"
fi

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  python "${SCRIPT_DIR}/check_training_setup.py" \
    --dataset-path "${DATASET_PATH}" \
    --embodiment-tag "${EMBODIMENT_TAG}" \
    --modality-config-path "${MODALITY_CONFIG_PATH}" \
    --model-action-horizon "${ACTION_HORIZON}"
fi

if [[ "${SKIP_STATS:-0}" != "1" ]]; then
  bash "${SCRIPT_DIR}/prepare_stats.sh"
fi

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  python "${SCRIPT_DIR}/check_training_setup.py" \
    --dataset-path "${DATASET_PATH}" \
    --embodiment-tag "${EMBODIMENT_TAG}" \
    --modality-config-path "${MODALITY_CONFIG_PATH}" \
    --model-action-horizon "${ACTION_HORIZON}" \
    --require-stats
fi

bash examples/finetune.sh \
  --base-model-path "${BASE_MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --embodiment-tag "${EMBODIMENT_TAG}" \
  --modality-config-path "${MODALITY_CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name "${EXPERIMENT_NAME:-piper_pi05_gr00t}" \
  --wandb-project "${WANDB_PROJECT:-finetune-gr00t-piper-pi05}" \
  -- \
  --action-horizon "${ACTION_HORIZON}" \
  "$@"
