#!/bin/bash
set -e

cd ~/chr/GR00T_battery_assemble

module load cuda12.2/toolkit/12.2.2
hash -r

export CUDA_HOME=/cm/shared/apps/cuda12.2/toolkit/12.2.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/ffmpeg6/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export TRITON_CACHE_DIR=/tmp/$USER/triton_cache
mkdir -p $TRITON_CACHE_DIR

export DATASET_PATH=/project/peilab/srk/wmpo_workspace/piper_insert_mouse_battery_lerobot
export WANDB_PROJECT=finetune-gr00t-piper-pi05
export EXPERIMENT_NAME=piper_pi05_gr00t

which nvcc
nvcc -V
python -c "import torch; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"

bash examples/PiperPi05/finetune_piper_pi05.sh
