# Piper pi05 -> GR00T

This folder is for the local dataset:

```text
/home/fenrir/ubunto_data_2/worldmodel/data/piper_insert_mouse_battery_lerobot
```

It does two things:

1. Fine-tune GR00T N1.7 with the Piper dataset.
2. Serve the fine-tuned GR00T checkpoint through a pi05/OpenPI-compatible websocket response:

```python
{"actions": np.ndarray(shape=(50, 14), dtype=np.float32)}
```

That is the shape consumed by:

```text
/home/fenrir/ubunto_data_2/worldmodel/kai0/deploy_client_temporal_ensembling_dictionary_save.py
```

## Data Shape

The modality config uses only the cameras that the current deploy client sends:

```text
top_head   -> cam_high
hand_left  -> cam_left_wrist
hand_right -> cam_right_wrist
```

It does not use `cam_vertical`, because the current real-machine client does not provide it.

State/action are split exactly like the dataset metadata:

```text
0:7   left_arm_joint_position
7:14  right_arm_joint_position
```

The adapter concatenates GR00T output back to pi05 order:

```text
[left_arm_joint_position(7), right_arm_joint_position(7)] -> (H, 14)
```

## Prepare Stats

The current Piper dataset does not ship `meta/stats.json`, and GR00T requires it.

Run a preflight check first:

```bash
uv run python examples/PiperPi05/check_training_setup.py
```

Then generate stats:

```bash
uv run bash examples/PiperPi05/prepare_stats.sh
```

This also writes `meta/relative_stats.json` for the relative joint-action config.

## Fine-Tune

Login to W&B once before training:

```bash
uv run wandb login
```

Default command:

```bash
uv run bash examples/PiperPi05/finetune_piper_pi05.sh
```

The finetune wrapper runs preflight checks, generates missing dataset statistics,
then calls the repo's normal `examples/finetune.sh` entrypoint. W&B is enabled
by default in this wrapper.

Common overrides:

```bash
BASE_MODEL_PATH=nvidia/GR00T-N1.7-3B \
DATASET_PATH=/home/fenrir/ubunto_data_2/worldmodel/data/piper_insert_mouse_battery_lerobot \
OUTPUT_DIR=/home/fenrir/ubunto_data_2/worldmodel/Isaac-GR00T/outputs/piper_pi05_gr00t \
MAX_STEPS=10000 \
GLOBAL_BATCH_SIZE=8 \
USE_WANDB=1 \
WANDB_PROJECT=finetune-gr00t-piper-pi05 \
EXPERIMENT_NAME=piper_pi05_gr00t \
uv run bash examples/PiperPi05/finetune_piper_pi05.sh
```

If your W&B account uses a team/entity, set it as an environment variable:

```bash
WANDB_ENTITY=<your-team-or-username> \
WANDB_PROJECT=finetune-gr00t-piper-pi05 \
uv run bash examples/PiperPi05/finetune_piper_pi05.sh
```

For a no-network dry run that still records W&B files locally:

```bash
WANDB_MODE=offline uv run bash examples/PiperPi05/finetune_piper_pi05.sh
```

The config predicts 40 GR00T action steps by default, matching the N1.7 model config in this checkout.

## Serve As pi05

Start the GR00T-as-pi05 websocket server on the same port your deploy client already uses:

```bash
uv run --with 'websockets>=15.0.1' \
  python examples/PiperPi05/serve_gr00t_as_pi05.py \
  --model-path outputs/piper_pi05_gr00t/checkpoint-10000 \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 8000 \
  --target-horizon 50
```

Then run your existing real-machine flow:

```bash
python /home/fenrir/ubunto_data_2/worldmodel/kai0/deploy_client_temporal_ensembling_dictionary_save.py
```

The deploy client still sees:

```python
self.client.infer(obs)["actions"]
```

but the server is GR00T internally.

## Horizon Note

`deploy_client_temporal_ensembling_dictionary_save.py` currently hardcodes 50 in two places:

```python
_ACTION_CHUNK_SIZE_ORIGIN = 50
self.ACTION_CHUNK_SIZE = 50
```

The facade pads GR00T's shorter chunk by repeating the last action so the existing client does not index past the end. A cleaner later version is to change both client constants to the actual GR00T action horizon and run this server with `--target-horizon 0`.
