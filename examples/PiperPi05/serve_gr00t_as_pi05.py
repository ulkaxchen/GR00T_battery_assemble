#!/usr/bin/env python3

"""Serve a GR00T checkpoint through the pi05/OpenPI websocket shape.

The real robot client in kai0/deploy_client_temporal_ensembling_dictionary_save.py
expects:

    response["actions"] -> np.ndarray, shape (chunk, 14)

GR00T returns a dict of action groups, typically:

    left_arm_joint_position  -> (1, H, 7)
    right_arm_joint_position -> (1, H, 7)

This server converts the current pi05 payload into a GR00T observation, calls
GR00T, then converts the action dict back to {"actions": (target_horizon, 14)}.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import functools
import http
import logging
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import msgpack
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402


LOGGER = logging.getLogger("serve_gr00t_as_pi05")


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


def _as_hwc_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _batched_video(image: np.ndarray) -> np.ndarray:
    return _as_hwc_uint8(image)[None, None, ...]


def _batched_state(state: np.ndarray) -> np.ndarray:
    return np.asarray(state, dtype=np.float32)[None, None, :]


def _unbatch_action(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected GR00T action with shape (B,H,D) or (H,D), got {arr.shape}")
    return arr


def _resize_chunk(actions: np.ndarray, target_horizon: int | None) -> np.ndarray:
    if target_horizon is None or actions.shape[0] == target_horizon:
        return actions
    if actions.shape[0] > target_horizon:
        return actions[:target_horizon]
    if actions.shape[0] == 0:
        raise ValueError("Cannot pad an empty action chunk")
    pad = np.repeat(actions[-1:], target_horizon - actions.shape[0], axis=0)
    return np.concatenate([actions, pad], axis=0)


@dataclass(frozen=True)
class KeyMap:
    top_head: str = "cam_high"
    hand_left: str = "cam_left_wrist"
    hand_right: str = "cam_right_wrist"
    left_action: str = "left_arm_joint_position"
    right_action: str = "right_arm_joint_position"
    left_state: str = "left_arm_joint_position"
    right_state: str = "right_arm_joint_position"


class Gr00tAsPi05Policy:
    def __init__(
        self,
        *,
        model_path: str,
        embodiment_tag: str,
        device: str,
        target_horizon: int | None,
        default_prompt: str,
        key_map: KeyMap,
        strict: bool,
        clip_gripper: bool,
    ) -> None:
        self.policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=model_path,
            device=device,
            strict=strict,
        )
        self.target_horizon = target_horizon
        self.default_prompt = default_prompt
        self.key_map = key_map
        self.clip_gripper = clip_gripper
        self.language_key = self.policy.language_key
        self.modality_configs = self.policy.get_modality_config()
        self._validate_expected_keys()

    def _validate_expected_keys(self) -> None:
        video_keys = set(self.modality_configs["video"].modality_keys)
        state_keys = set(self.modality_configs["state"].modality_keys)
        action_keys = set(self.modality_configs["action"].modality_keys)
        needed_video = {self.key_map.top_head, self.key_map.hand_left, self.key_map.hand_right}
        needed_state = {self.key_map.left_state, self.key_map.right_state}
        needed_action = {self.key_map.left_action, self.key_map.right_action}
        missing = {
            "video": sorted(needed_video - video_keys),
            "state": sorted(needed_state - state_keys),
            "action": sorted(needed_action - action_keys),
        }
        missing = {key: value for key, value in missing.items() if value}
        if missing:
            raise ValueError(
                "Checkpoint modality config does not match this Piper/pi05 adapter. "
                f"Missing keys: {missing}. Available video={sorted(video_keys)}, "
                f"state={sorted(state_keys)}, action={sorted(action_keys)}"
            )

    def infer(self, pi05_obs: dict[str, Any]) -> dict[str, Any]:
        start = time.monotonic()
        gr00t_obs = self._pi05_obs_to_gr00t(pi05_obs)
        action_dict, info = self.policy.get_action(gr00t_obs)
        actions = self._gr00t_action_to_pi05(action_dict)
        return {
            "actions": actions,
            "policy_timing": {
                "infer_ms": (time.monotonic() - start) * 1000,
                "gr00t_action_horizon": int(actions.shape[0]),
                "gr00t_action_dim": int(actions.shape[1]),
            },
            "gr00t_info": info,
        }

    def _pi05_obs_to_gr00t(self, obs: dict[str, Any]) -> dict[str, Any]:
        images = obs["images"]
        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if state.shape[0] < 14:
            raise ValueError(f"Expected at least 14 state dims, got {state.shape}")

        prompt = str(obs.get("prompt") or self.default_prompt)
        return {
            "video": {
                self.key_map.top_head: _batched_video(images["top_head"]),
                self.key_map.hand_left: _batched_video(images["hand_left"]),
                self.key_map.hand_right: _batched_video(images["hand_right"]),
            },
            "state": {
                self.key_map.left_state: _batched_state(state[:7]),
                self.key_map.right_state: _batched_state(state[7:14]),
            },
            "language": {
                self.language_key: [[prompt]],
            },
        }

    def _gr00t_action_to_pi05(self, action_dict: dict[str, np.ndarray]) -> np.ndarray:
        left = _unbatch_action(action_dict[self.key_map.left_action])
        right = _unbatch_action(action_dict[self.key_map.right_action])
        if left.shape[-1] != 7 or right.shape[-1] != 7:
            raise ValueError(
                "Expected left/right GR00T action groups to be 7D each, "
                f"got left={left.shape}, right={right.shape}"
            )
        actions = np.concatenate([left, right], axis=-1).astype(np.float32, copy=False)
        actions = _resize_chunk(actions, self.target_horizon)
        if self.clip_gripper:
            actions[:, 6] = np.clip(actions[:, 6], 0.0, 0.1)
            actions[:, 13] = np.clip(actions[:, 13], 0.0, 0.1)
        return np.ascontiguousarray(actions)


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def _serve(args: argparse.Namespace) -> None:
    try:
        import websockets
        import websockets.asyncio.server as websocket_server
        import websockets.frames
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'websockets'. Run with:\n"
            "  uv run --with 'websockets>=15.0.1' python "
            "examples/PiperPi05/serve_gr00t_as_pi05.py ...\n"
            "or install websockets into the active environment."
        ) from exc

    policy = Gr00tAsPi05Policy(
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        device=args.device,
        target_horizon=args.target_horizon,
        default_prompt=args.default_prompt,
        key_map=KeyMap(
            top_head=args.top_head_key,
            hand_left=args.hand_left_key,
            hand_right=args.hand_right_key,
            left_action=args.left_action_key,
            right_action=args.right_action_key,
            left_state=args.left_state_key,
            right_state=args.right_state_key,
        ),
        strict=not args.no_strict,
        clip_gripper=not args.no_clip_gripper,
    )
    packer = Packer()
    metadata = {
        "server": "gr00t_as_pi05",
        "model_path": args.model_path,
        "embodiment_tag": args.embodiment_tag,
        "target_horizon": args.target_horizon,
        "action_dim": 14,
        "language_key": policy.language_key,
    }

    async def handler(websocket):
        LOGGER.info("Connection from %s opened", websocket.remote_address)
        await websocket.send(packer.pack(metadata))
        prev_total_time = None
        while True:
            try:
                start = time.monotonic()
                obs = unpackb(await websocket.recv())
                infer_start = time.monotonic()
                action = policy.infer(obs)
                action.setdefault("server_timing", {})
                action["server_timing"]["infer_ms"] = (time.monotonic() - infer_start) * 1000
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000
                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start
            except websockets.ConnectionClosed:
                LOGGER.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    LOGGER.info("Serving GR00T-as-pi05 on %s:%s", args.host, args.port)
    async with websocket_server.serve(
        handler,
        args.host,
        args.port,
        compression=None,
        max_size=None,
        process_request=_health_check,
    ) as server:
        await server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="GR00T finetuned checkpoint directory")
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=50,
        help="pi05-style chunk length returned to the existing kai0 client. Use 0 to keep GR00T horizon.",
    )
    parser.add_argument("--default-prompt", default="insert mouse battery")
    parser.add_argument("--top-head-key", default="cam_high")
    parser.add_argument("--hand-left-key", default="cam_left_wrist")
    parser.add_argument("--hand-right-key", default="cam_right_wrist")
    parser.add_argument("--left-state-key", default="left_arm_joint_position")
    parser.add_argument("--right-state-key", default="right_arm_joint_position")
    parser.add_argument("--left-action-key", default="left_arm_joint_position")
    parser.add_argument("--right-action-key", default="right_arm_joint_position")
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--no-clip-gripper", action="store_true")
    args = parser.parse_args()
    if args.target_horizon <= 0:
        args.target_horizon = None
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_serve(parse_args()))


if __name__ == "__main__":
    main()
