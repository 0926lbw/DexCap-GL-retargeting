"""Coordinate-frame helpers for reconstructed hand skeletons."""

from __future__ import annotations

from typing import Any

import numpy as np


COORDINATE_CONVENTION: dict[str, Any] = {
    "name": "hand_wrist_v0",
    "unit": "meter",
    "origin": "wrist",
    "x_axis": "thumb_side",
    "y_axis": "fingers",
    "z_axis": "palm_normal",
    "handedness": "right_hand_positive_x_to_thumb; left_hand_mirrors_x",
}


def default_coordinate_convention() -> dict[str, Any]:
    """Return a copy of the current 21-point wrist-frame convention."""
    return dict(COORDINATE_CONVENTION)


def make_transform(
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> np.ndarray:
    """Build a homogeneous transform T_W_Hwrist."""
    transform = np.eye(4, dtype=float)
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
        transform[:3, :3] = rotation
    if translation is not None:
        translation = np.asarray(translation, dtype=float)
        if translation.shape != (3,):
            raise ValueError(
                f"translation must have shape (3,), got {translation.shape}"
            )
        transform[:3, 3] = translation
    return transform


def validate_transform(transform: np.ndarray, name: str = "transform") -> np.ndarray:
    """Return a checked 4x4 homogeneous transform copy."""
    checked = np.asarray(transform, dtype=float)
    if checked.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {checked.shape}")
    if not np.all(np.isfinite(checked)):
        raise ValueError(f"{name} must contain only finite values")
    return checked.copy()


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a homogeneous transform."""
    checked = validate_transform(transform)
    rotation = checked[:3, :3]
    translation = checked[:3, 3]
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def apply_transform(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to an N x 3 point array."""
    transform = np.asarray(transform, dtype=float)
    points = np.asarray(points, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {transform.shape}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    return points @ transform[:3, :3].T + transform[:3, 3]
