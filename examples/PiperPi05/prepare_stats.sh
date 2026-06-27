#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

DATASET_PATH="${DATASET_PATH:-/home/fenrir/ubunto_data_2/worldmodel/data/piper_insert_mouse_battery_lerobot}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-${SCRIPT_DIR}/piper_pi05_config.py}"

cd "${REPO_ROOT}"

python gr00t/data/stats.py \
  --dataset-path "${DATASET_PATH}" \
  --embodiment-tag "${EMBODIMENT_TAG}" \
  --modality-config-path "${MODALITY_CONFIG_PATH}"
