"""Export helpers for reconstructed hand skeletons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .coordinate_frames import default_coordinate_convention
from .human_hand_model import HumanHandSkeleton
from .schema import LANDMARK_NAMES
from .solver import HandReconstructionFrame


def skeleton_to_dict(skeleton: HumanHandSkeleton) -> dict[str, Any]:
    points_w = skeleton.keypoints_world()
    points_h = skeleton.keypoints_wrist()
    keypoints_w = _named_points(points_w)
    return {
        "hand": skeleton.hand,
        "coordinate_convention": dict(skeleton.coordinate_convention),
        "T_W_Hwrist": skeleton.transform_world_from_wrist().tolist(),
        "keypoints_21_in_Hwrist": _named_points(points_h),
        "keypoints_21_in_W": keypoints_w,
        "landmarks": keypoints_w,
    }


def solver_frame_to_dict(frame: HandReconstructionFrame) -> dict[str, Any]:
    """Return a JSON-serializable dict for a full solver frame."""
    return {
        "hand": frame.hand,
        "coordinate_convention": dict(default_coordinate_convention()),
        "joint_angles": dict(frame.joint_angles),
        "T_W_Hwrist": frame.T_W_Hwrist.tolist(),
        "keypoints_21_in_Hwrist": _named_points(frame.keypoints_21_in_Hwrist),
        "keypoints_21_in_W": _named_points(frame.keypoints_21_in_W),
        "direct_glove_keypoints_21_in_W": _named_points(
            frame.direct_glove_keypoints_21_in_W
        ),
        "direct_glove_keypoints_21_in_Hwrist": _named_points(
            frame.direct_glove_keypoints_21_in_Hwrist
        ),
        "fused_keypoints_21_in_Hwrist": _named_points(
            frame.fused_keypoints_21_in_Hwrist
        ),
        "fused_keypoints_21_in_W": _named_points(frame.fused_keypoints_21_in_W),
        "diagnostics": dict(frame.diagnostics),
    }


def _named_points(points: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "index": idx,
            "name": LANDMARK_NAMES[idx],
            "xyz": points[idx].tolist(),
        }
        for idx in range(points.shape[0])
    ]


def write_skeleton_json(skeleton: HumanHandSkeleton, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(skeleton_to_dict(skeleton), fp, indent=2)


def write_skeleton_npy(skeleton: HumanHandSkeleton, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, skeleton.to_numpy())
