"""Solver layer for DexGlove-to-human-hand keypoint reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .coordinate_frames import apply_transform, invert_transform, validate_transform
from .retargeting import GLOVE_DOF, RETARGET_LIMITS
from .schema import NUM_LANDMARKS
from .tip_locking import FINGERTIP_INDICES, fuse_tip_locked_landmarks


@dataclass(frozen=True)
class HandReconstructionFrame:
    """One reconstructed hand frame with explicit local and world keypoints."""

    hand: str
    joint_angles: dict[str, float]
    keypoints_21_in_Hwrist: np.ndarray
    keypoints_21_in_W: np.ndarray
    direct_glove_keypoints_21_in_W: np.ndarray
    direct_glove_keypoints_21_in_Hwrist: np.ndarray
    fused_keypoints_21_in_Hwrist: np.ndarray
    fused_keypoints_21_in_W: np.ndarray
    T_W_Hwrist: np.ndarray
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        if self.hand not in {"left", "right"}:
            raise ValueError("hand must be 'left' or 'right'")

        object.__setattr__(
            self,
            "joint_angles",
            {str(name): float(value) for name, value in self.joint_angles.items()},
        )
        for attr in (
            "keypoints_21_in_Hwrist",
            "keypoints_21_in_W",
            "direct_glove_keypoints_21_in_W",
            "direct_glove_keypoints_21_in_Hwrist",
            "fused_keypoints_21_in_Hwrist",
            "fused_keypoints_21_in_W",
        ):
            object.__setattr__(
                self,
                attr,
                _validate_keypoints(getattr(self, attr), attr),
            )
        object.__setattr__(
            self,
            "T_W_Hwrist",
            validate_transform(self.T_W_Hwrist, "T_W_Hwrist"),
        )
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


class JointAngleSmoother:
    """Stateful per-joint EMA plus optional per-frame max delta."""

    def __init__(self, alpha: float = 0.35, max_delta: float | None = None) -> None:
        self.alpha = float(alpha)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.max_delta = None if max_delta is None else float(max_delta)
        if self.max_delta is not None and self.max_delta <= 0.0:
            raise ValueError("max_delta must be positive")
        self._previous: dict[str, float] | None = None

    def reset(self) -> None:
        self._previous = None

    def smooth(self, joint_angles: dict[str, float]) -> tuple[dict[str, float], float]:
        current = {str(name): float(value) for name, value in joint_angles.items()}
        if self._previous is None:
            self._previous = dict(current)
            return dict(current), 0.0

        smoothed: dict[str, float] = {}
        max_delta_seen = 0.0
        for name, value in current.items():
            previous = self._previous.get(name, value)
            filtered = self.alpha * value + (1.0 - self.alpha) * previous
            delta = filtered - previous
            if self.max_delta is not None:
                delta = float(np.clip(delta, -self.max_delta, self.max_delta))
                filtered = previous + delta
            smoothed[name] = float(filtered)
            max_delta_seen = max(max_delta_seen, abs(delta))

        self._previous = dict(smoothed)
        return smoothed, float(max_delta_seen)


def _validate_keypoints(points: np.ndarray, name: str) -> np.ndarray:
    checked = np.asarray(points, dtype=float)
    expected_shape = (NUM_LANDMARKS, 3)
    if checked.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {checked.shape}")
    if not np.all(np.isfinite(checked)):
        raise ValueError(f"{name} must contain only finite values")
    return checked.copy()


class HandReconstructionSolver:
    """Coordinate one hand's retargeting, FK, frame transforms, and tip-lock fusion."""

    def __init__(
        self,
        hand: str,
        human_model: Any,
        retargeter: Any,
        glove_pipeline: Any,
        T_W_Hwrist: np.ndarray,
        *,
        smoother: Any | None = None,
        fuse=fuse_tip_locked_landmarks,
    ) -> None:
        if hand not in {"left", "right"}:
            raise ValueError("hand must be 'left' or 'right'")
        self.hand = hand
        self.human_model = human_model
        self.retargeter = retargeter
        self.glove_pipeline = glove_pipeline
        self.T_W_Hwrist = validate_transform(T_W_Hwrist, "T_W_Hwrist")
        self.T_Hwrist_W = invert_transform(self.T_W_Hwrist)
        self.smoother = smoother
        self.fuse = fuse

    def reconstruct(self, q_glove: np.ndarray) -> HandReconstructionFrame:
        q = np.asarray(q_glove, dtype=float)
        if q.shape != (GLOVE_DOF,):
            raise ValueError(f"q_glove must have shape ({GLOVE_DOF},), got {q.shape}")

        raw_joint_angles = self.retargeter.retarget(q)
        if self.smoother is None:
            joint_angles = dict(raw_joint_angles)
            max_joint_delta = 0.0
        else:
            joint_angles, max_joint_delta = self.smoother.smooth(raw_joint_angles)

        keypoints_h = self.human_model.landmarks_from_joints(joint_angles)
        direct_w = self.glove_pipeline.reconstruct_direct(q).to_numpy()
        direct_h = apply_transform(self.T_Hwrist_W, direct_w)
        keypoints_w = apply_transform(self.T_W_Hwrist, keypoints_h)
        fused_h = self.fuse(keypoints_h, direct_h)
        fused_w = apply_transform(self.T_W_Hwrist, fused_h)

        diagnostics = _diagnostics(
            joint_angles,
            keypoints_h,
            direct_h,
            fused_h,
            self.T_W_Hwrist,
            self.T_Hwrist_W,
            max_joint_delta,
        )
        return HandReconstructionFrame(
            hand=self.hand,
            joint_angles=joint_angles,
            keypoints_21_in_Hwrist=keypoints_h,
            keypoints_21_in_W=keypoints_w,
            direct_glove_keypoints_21_in_W=direct_w,
            direct_glove_keypoints_21_in_Hwrist=direct_h,
            fused_keypoints_21_in_Hwrist=fused_h,
            fused_keypoints_21_in_W=fused_w,
            T_W_Hwrist=self.T_W_Hwrist,
            diagnostics=diagnostics,
        )


def _diagnostics(
    joint_angles: dict[str, float],
    keypoints_h: np.ndarray,
    direct_h: np.ndarray,
    fused_h: np.ndarray,
    T_W_Hwrist: np.ndarray,
    T_Hwrist_W: np.ndarray,
    max_joint_delta: float,
) -> dict[str, Any]:
    before = _mean_fingertip_error(keypoints_h, direct_h)
    after = _mean_fingertip_error(fused_h, direct_h)
    roundtrip = apply_transform(T_Hwrist_W, apply_transform(T_W_Hwrist, fused_h))
    roundtrip_error = float(np.max(np.linalg.norm(roundtrip - fused_h, axis=1)))
    return {
        "fingertip_error_before_fusion": before,
        "fingertip_error_after_fusion": after,
        "transform_det": float(np.linalg.det(T_W_Hwrist[:3, :3])),
        "joint_limit_hit": _joint_limit_hit(joint_angles),
        "max_joint_delta": float(max_joint_delta),
        "roundtrip_error": roundtrip_error,
        "all_finite": bool(
            np.all(np.isfinite(keypoints_h))
            and np.all(np.isfinite(direct_h))
            and np.all(np.isfinite(fused_h))
        ),
    }


def _mean_fingertip_error(source: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.mean(
            [
                np.linalg.norm(source[idx] - target[idx])
                for idx in FINGERTIP_INDICES
            ]
        )
    )


def _joint_limit_hit(joint_angles: dict[str, float], eps: float = 1e-9) -> bool:
    for name, value in joint_angles.items():
        limit_name = _retarget_limit_name(name)
        if limit_name is None:
            continue
        lo, hi = RETARGET_LIMITS[limit_name]
        if abs(value - lo) <= eps or abs(value - hi) <= eps:
            return True
    return False


def _retarget_limit_name(joint_name: str) -> str | None:
    if joint_name.endswith("_mcp_flex"):
        return "mcp_flex"
    if joint_name.endswith("_mcp_abd"):
        return "mcp_abd"
    if joint_name.endswith("_pip"):
        return "pip"
    if joint_name.endswith("_dip"):
        return "dip"
    if joint_name in RETARGET_LIMITS:
        return joint_name
    return None
