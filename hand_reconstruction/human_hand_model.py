"""Simple 21-point human hand skeleton model."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np

from .coordinate_frames import (
    apply_transform,
    default_coordinate_convention,
    make_transform,
)
from .schema import FINGER_CHAINS, FINGER_ORDER, NUM_LANDMARKS


@dataclass(frozen=True)
class HumanHandSkeleton:
    landmarks: np.ndarray
    hand: str = "right"
    keypoints_21_in_Hwrist: np.ndarray | None = None
    T_W_Hwrist: np.ndarray | None = None
    coordinate_convention: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        points_w = _as_keypoints(self.landmarks, "landmarks").copy()
        if self.hand not in {"left", "right"}:
            raise ValueError("hand must be left or right")

        if self.keypoints_21_in_Hwrist is None:
            points_h = points_w.copy()
        else:
            points_h = _as_keypoints(
                self.keypoints_21_in_Hwrist, "keypoints_21_in_Hwrist"
            ).copy()

        if self.T_W_Hwrist is None:
            transform = make_transform()
        else:
            transform = np.asarray(self.T_W_Hwrist, dtype=float).copy()
            if transform.shape != (4, 4):
                raise ValueError(
                    f"T_W_Hwrist must have shape (4, 4), got {transform.shape}"
                )

        convention = (
            default_coordinate_convention()
            if self.coordinate_convention is None
            else dict(self.coordinate_convention)
        )

        object.__setattr__(self, "landmarks", points_w)
        object.__setattr__(self, "keypoints_21_in_Hwrist", points_h)
        object.__setattr__(self, "T_W_Hwrist", transform)
        object.__setattr__(self, "coordinate_convention", convention)

    @classmethod
    def default(cls, hand: str = "right") -> "HumanHandSkeleton":
        """Return a MediaPipe-style open-hand 21-point template in meters."""
        if hand not in {"left", "right"}:
            raise ValueError("hand must be 'left' or 'right'")

        x_sign = -1.0 if hand == "left" else 1.0
        points_h = _mediapipe_open_hand_template()
        points_h[:, 0] *= x_sign
        transform = make_transform()
        points_w = apply_transform(transform, points_h)
        return cls(
            points_w,
            hand=hand,
            keypoints_21_in_Hwrist=points_h,
            T_W_Hwrist=transform,
            coordinate_convention=default_coordinate_convention(),
        )

    def to_numpy(self) -> np.ndarray:
        """Return world-frame landmarks as a 21 x 3 array."""
        return self.landmarks.copy()

    def keypoints_world(self) -> np.ndarray:
        """Return keypoints_21_in_W as a 21 x 3 array."""
        return self.landmarks.copy()

    def keypoints_wrist(self) -> np.ndarray:
        """Return keypoints_21_in_Hwrist as a 21 x 3 array."""
        return self.keypoints_21_in_Hwrist.copy()

    def transform_world_from_wrist(self) -> np.ndarray:
        """Return T_W_Hwrist as a 4 x 4 homogeneous transform."""
        return self.T_W_Hwrist.copy()

    def finger_bone_lengths(self) -> dict[str, tuple[float, float, float, float]]:
        """Return wrist-to-root and intra-finger segment lengths."""
        lengths: dict[str, tuple[float, float, float, float]] = {}
        for finger, chain in FINGER_CHAINS.items():
            values = []
            for start, end in zip(chain[:-1], chain[1:]):
                values.append(
                    float(np.linalg.norm(self.landmarks[end] - self.landmarks[start]))
                )
            lengths[finger] = tuple(values)
        return lengths


def _as_keypoints(points: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.shape != (NUM_LANDMARKS, 3):
        raise ValueError(
            f"{name} must have shape ({NUM_LANDMARKS}, 3), got {points.shape}"
        )
    return points


def _mediapipe_open_hand_template() -> np.ndarray:
    return np.array(
        [
            [0.000, 0.000, 0.000],
            [0.035, 0.028, 0.000],
            [0.060, 0.052, 0.000],
            [0.080, 0.077, 0.000],
            [0.096, 0.101, 0.000],
            [0.030, 0.075, 0.000],
            [0.040, 0.117, 0.000],
            [0.047, 0.143, 0.000],
            [0.052, 0.161, 0.000],
            [0.006, 0.083, 0.000],
            [0.008, 0.132, 0.000],
            [0.007, 0.163, 0.000],
            [0.006, 0.198, 0.000],
            [-0.020, 0.076, 0.000],
            [-0.028, 0.121, 0.000],
            [-0.034, 0.149, 0.000],
            [-0.039, 0.168, 0.000],
            [-0.043, 0.062, 0.000],
            [-0.052, 0.097, 0.000],
            [-0.057, 0.124, 0.000],
            [-0.061, 0.146, 0.000],
        ],
        dtype=float,
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("cannot normalize zero-length vector")
    return vector / norm
