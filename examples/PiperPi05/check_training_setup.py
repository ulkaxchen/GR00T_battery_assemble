#!/usr/bin/env python3

"""Preflight checks before fine-tuning GR00T on the local Piper/pi05 dataset."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS  # noqa: E402
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402


DEFAULT_DATASET_PATH = (
    "/home/fenrir/ubunto_data_2/worldmodel/data/piper_insert_mouse_battery_lerobot"
)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _import_modality_config(path: Path) -> None:
    if not path.is_file() or path.suffix != ".py":
        raise FileNotFoundError(f"Modality config must be a .py file: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import modality config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _failures_for_modality(config, modality_meta: dict) -> list[str]:
    failures: list[str] = []

    for video_key in config["video"].modality_keys:
        if video_key not in modality_meta.get("video", {}):
            failures.append(f"video key missing from meta/modality.json: {video_key}")

    for state_key in config["state"].modality_keys:
        if state_key not in modality_meta.get("state", {}):
            failures.append(f"state key missing from meta/modality.json: {state_key}")

    for action_key in config["action"].modality_keys:
        if action_key not in modality_meta.get("action", {}):
            failures.append(f"action key missing from meta/modality.json: {action_key}")

    for language_key in config["language"].modality_keys:
        if not language_key.startswith("annotation."):
            failures.append(f"language key must start with annotation.: {language_key}")
            continue
        annotation_key = language_key.replace("annotation.", "", 1)
        if annotation_key not in modality_meta.get("annotation", {}):
            failures.append(f"annotation key missing from meta/modality.json: {annotation_key}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--modality-config-path",
        default=str(Path(__file__).with_name("piper_pi05_config.py")),
    )
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--require-stats", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    modality_config_path = Path(args.modality_config_path)
    tag = EmbodimentTag.resolve(args.embodiment_tag)

    failures: list[str] = []
    if not dataset_path.is_dir():
        failures.append(f"dataset path does not exist: {dataset_path}")
    meta_dir = dataset_path / "meta"
    required_meta = [
        "info.json",
        "episodes.jsonl",
        "tasks.jsonl",
        "modality.json",
    ]
    for name in required_meta:
        if not (meta_dir / name).is_file():
            failures.append(f"missing meta file: {meta_dir / name}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 2

    _import_modality_config(modality_config_path)
    config = MODALITY_CONFIGS[tag.value]

    info = _load_json(meta_dir / "info.json")
    modality_meta = _load_json(meta_dir / "modality.json")
    episodes = _load_jsonl(meta_dir / "episodes.jsonl")
    tasks = _load_jsonl(meta_dir / "tasks.jsonl")
    features = info.get("features", {})

    failures.extend(_failures_for_modality(config, modality_meta))

    for video_key in config["video"].modality_keys:
        original_key = modality_meta["video"][video_key].get(
            "original_key", f"observation.images.{video_key}"
        )
        if original_key not in features:
            failures.append(f"video feature missing from meta/info.json: {original_key}")

    action_horizon = len(config["action"].delta_indices)
    max_action_horizon = Gr00tN1d7Config().action_horizon
    if action_horizon > max_action_horizon:
        failures.append(
            f"action horizon {action_horizon} exceeds default GR00T N1.7 horizon "
            f"{max_action_horizon}"
        )

    stats_path = meta_dir / "stats.json"
    rel_stats_path = meta_dir / "relative_stats.json"
    if args.require_stats:
        if not stats_path.is_file():
            failures.append(f"missing required stats file: {stats_path}")
        if not rel_stats_path.is_file():
            failures.append(f"missing required relative stats file: {rel_stats_path}")

    print(f"[OK] dataset: {dataset_path}")
    print(f"[OK] episodes: {len(episodes)}; tasks: {[task.get('task') for task in tasks]}")
    print(f"[OK] video keys: {config['video'].modality_keys}")
    print(f"[OK] state keys: {config['state'].modality_keys}")
    print(f"[OK] action keys: {config['action'].modality_keys}")
    print(f"[OK] action horizon: {action_horizon}")
    print(f"[OK] language keys: {config['language'].modality_keys}")
    print(f"[{'OK' if stats_path.exists() else 'WARN'}] stats: {stats_path}")
    print(f"[{'OK' if rel_stats_path.exists() else 'WARN'}] relative stats: {rel_stats_path}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 2
    print("[OK] training setup preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
