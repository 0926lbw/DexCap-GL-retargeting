# Hand Keypoint Coordinate Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a solver layer that reconstructs DexGlove-driven human-hand keypoints in both wrist-local and world frames, performs tip-lock fusion in wrist-local coordinates, and prepares optional joint smoothing without changing the accepted display alignment.

**Architecture:** Keep `retargeting.py`, `human_hand_builder.py`, `pipeline.py`, and `tip_locking.py` focused on their existing jobs. Add `hand_reconstruction/solver.py` as the coordination layer that owns transforms, per-frame result validation, diagnostics, and optional smoothing. Update the MeshCat stream to consume solver results while keeping display-only offsets separate from reconstruction data.

**Tech Stack:** Python, NumPy, unittest, Pinocchio-backed classes only where already used by the live display path.

---

## File Structure

- Create: `hand_reconstruction/solver.py`
  - Defines `HandReconstructionFrame`, `JointAngleSmoother`, `HandReconstructionSolver`, and small pure helpers for validation and diagnostics.
- Modify: `hand_reconstruction/coordinate_frames.py`
  - Add `validate_transform()` and `invert_transform()` for shared transform handling.
- Modify: `hand_reconstruction/__init__.py`
  - Export the new solver APIs.
- Modify: `DexCap_v4/dexcap_glove_meshcat_stream.py`
  - Replace the open-coded retarget/FK/direct/tip-lock sequence in `display_human()` with solver calls.
  - Keep existing display offsets and CLI behavior.
- Modify: `hand_reconstruction/export.py`
  - Add a solver-frame export helper that includes local and world keypoints.
- Create: `tests/test_solver_frame.py`
  - Pure NumPy/unit tests for frame validation, transforms, solver fusion, smoothing, and export data shape.
- Modify: `DexCap_v4/test_dexcap_glove_meshcat_stream.py`
  - Update the display test to verify solver output is used and display offsets remain display-only.

Before implementation, run:

```bash
git rev-parse --show-toplevel
```

Expected in the current workspace: `fatal: not a git repository`. If execution happens in a git checkout, use the commit steps below. If execution happens in this workspace, skip commit commands and mention the non-git workspace in the final report.

---

### Task 1: Add Shared Transform Helpers

**Files:**
- Modify: `hand_reconstruction/coordinate_frames.py`
- Create: `tests/test_solver_frame.py`

- [ ] **Step 1: Write failing tests for transform validation and inversion**

Create `tests/test_solver_frame.py` with this initial content:

```python
import unittest

import numpy as np


class CoordinateFrameHelpersTest(unittest.TestCase):
    def test_invert_transform_round_trips_points(self):
        from hand_reconstruction.coordinate_frames import (
            apply_transform,
            invert_transform,
            make_transform,
        )

        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform = make_transform(
            rotation=rotation,
            translation=np.array([0.10, -0.20, 0.30]),
        )
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.2, -0.3],
                [-0.4, 0.5, 0.6],
            ]
        )

        world = apply_transform(transform, points)
        local = apply_transform(invert_transform(transform), world)

        np.testing.assert_allclose(local, points, atol=1e-12)

    def test_validate_transform_rejects_bad_shape_and_nonfinite_values(self):
        from hand_reconstruction.coordinate_frames import validate_transform

        with self.assertRaisesRegex(ValueError, "T_W_Hwrist must have shape"):
            validate_transform(np.eye(3), "T_W_Hwrist")

        bad = np.eye(4)
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "T_W_Hwrist must contain only finite"):
            validate_transform(bad, "T_W_Hwrist")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m unittest tests.test_solver_frame
```

Expected: FAIL with an import error for `invert_transform` or `validate_transform`.

- [ ] **Step 3: Add transform helper implementation**

Modify `hand_reconstruction/coordinate_frames.py` by adding these functions after `make_transform()` and before `apply_transform()`:

```python
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
```

- [ ] **Step 4: Run transform tests to verify they pass**

Run:

```bash
python3 -m unittest tests.test_solver_frame
```

Expected: PASS.

- [ ] **Step 5: Commit if in a git repository**

Run:

```bash
git status --short
git add hand_reconstruction/coordinate_frames.py tests/test_solver_frame.py
git commit -m "feat: add hand transform helpers"
```

Expected in this workspace: skip this step because `git rev-parse --show-toplevel` reports this is not a git repository.

---

### Task 2: Add the Solver Frame Result Object

**Files:**
- Modify: `hand_reconstruction/solver.py`
- Modify: `tests/test_solver_frame.py`
- Modify: `hand_reconstruction/__init__.py`

- [ ] **Step 1: Add failing tests for `HandReconstructionFrame` validation**

Append this test class to `tests/test_solver_frame.py` before the `if __name__ == "__main__":` block:

```python
class HandReconstructionFrameTest(unittest.TestCase):
    def test_frame_validates_keypoint_shapes_and_transform(self):
        from hand_reconstruction.solver import HandReconstructionFrame

        points = _sample_keypoints()
        frame = HandReconstructionFrame(
            hand="right",
            joint_angles={"index_mcp_flex": 0.2},
            keypoints_21_in_Hwrist=points,
            keypoints_21_in_W=points + np.array([1.0, 0.0, 0.0]),
            direct_glove_keypoints_21_in_W=points + np.array([2.0, 0.0, 0.0]),
            direct_glove_keypoints_21_in_Hwrist=points + np.array([3.0, 0.0, 0.0]),
            fused_keypoints_21_in_Hwrist=points + np.array([4.0, 0.0, 0.0]),
            fused_keypoints_21_in_W=points + np.array([5.0, 0.0, 0.0]),
            T_W_Hwrist=np.eye(4),
            diagnostics={"roundtrip_error": 0.0},
        )

        self.assertEqual(frame.hand, "right")
        np.testing.assert_allclose(frame.keypoints_21_in_Hwrist, points)
        self.assertIsNot(frame.keypoints_21_in_Hwrist, points)

    def test_frame_rejects_invalid_hand_bad_keypoints_and_bad_transform(self):
        from hand_reconstruction.solver import HandReconstructionFrame

        points = _sample_keypoints()
        kwargs = dict(
            hand="right",
            joint_angles={},
            keypoints_21_in_Hwrist=points,
            keypoints_21_in_W=points,
            direct_glove_keypoints_21_in_W=points,
            direct_glove_keypoints_21_in_Hwrist=points,
            fused_keypoints_21_in_Hwrist=points,
            fused_keypoints_21_in_W=points,
            T_W_Hwrist=np.eye(4),
            diagnostics={},
        )

        with self.assertRaisesRegex(ValueError, "hand must be 'left' or 'right'"):
            HandReconstructionFrame(**{**kwargs, "hand": "center"})

        with self.assertRaisesRegex(ValueError, "keypoints_21_in_W must have shape"):
            HandReconstructionFrame(
                **{**kwargs, "keypoints_21_in_W": np.zeros((5, 3))}
            )

        bad_points = points.copy()
        bad_points[0, 0] = np.inf
        with self.assertRaisesRegex(
            ValueError, "fused_keypoints_21_in_W must contain only finite"
        ):
            HandReconstructionFrame(
                **{**kwargs, "fused_keypoints_21_in_W": bad_points}
            )

        with self.assertRaisesRegex(ValueError, "T_W_Hwrist must have shape"):
            HandReconstructionFrame(**{**kwargs, "T_W_Hwrist": np.eye(3)})


def _sample_keypoints(offset=0.0):
    points = np.zeros((21, 3), dtype=float)
    for idx in range(21):
        points[idx] = np.array(
            [offset + idx * 0.01, idx * 0.02, -idx * 0.005],
            dtype=float,
        )
    return points
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m unittest tests.test_solver_frame
```

Expected: FAIL with `ModuleNotFoundError: No module named 'hand_reconstruction.solver'`.

- [ ] **Step 3: Create `hand_reconstruction/solver.py` with the frame dataclass**

Create `hand_reconstruction/solver.py` with:

```python
"""Solver layer for DexGlove-to-human-hand keypoint reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .coordinate_frames import validate_transform
from .schema import NUM_LANDMARKS


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


def _validate_keypoints(points: np.ndarray, name: str) -> np.ndarray:
    checked = np.asarray(points, dtype=float)
    expected_shape = (NUM_LANDMARKS, 3)
    if checked.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {checked.shape}")
    if not np.all(np.isfinite(checked)):
        raise ValueError(f"{name} must contain only finite values")
    return checked.copy()
```

- [ ] **Step 4: Export the frame class from the package**

Modify `hand_reconstruction/__init__.py`:

```python
"""Hand reconstruction utilities for DexCap glove data."""

from .human_hand_model import HumanHandSkeleton
from .pipeline import HandReconstructionPipeline, reconstruct_from_link_positions
from .solver import HandReconstructionFrame
from .tip_locking import FINGERTIP_INDICES, fuse_tip_locked_landmarks

__all__ = [
    "FINGERTIP_INDICES",
    "HandReconstructionFrame",
    "HandReconstructionPipeline",
    "HumanHandSkeleton",
    "fuse_tip_locked_landmarks",
    "reconstruct_from_link_positions",
]
```

- [ ] **Step 5: Add a package export assertion**

Append this test to `HandReconstructionFrameTest` in `tests/test_solver_frame.py`:

```python
    def test_package_exports_frame_type(self):
        from hand_reconstruction import HandReconstructionFrame

        self.assertEqual(HandReconstructionFrame.__name__, "HandReconstructionFrame")
```

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
python3 -m unittest tests.test_solver_frame
```

Expected: PASS.

- [ ] **Step 7: Commit if in a git repository**

Run:

```bash
git status --short
git add hand_reconstruction/solver.py hand_reconstruction/__init__.py tests/test_solver_frame.py
git commit -m "feat: add hand reconstruction frame"
```

Expected in this workspace: skip this step because the current directory is not a git repository.

---

### Task 3: Add the Unsmoothed Reconstruction Solver

**Files:**
- Modify: `hand_reconstruction/solver.py`
- Modify: `tests/test_solver_frame.py`

- [ ] **Step 1: Add failing tests for local-frame fusion and diagnostics**

Append this test class and helpers to `tests/test_solver_frame.py` before the `if __name__ == "__main__":` block:

```python
class HandReconstructionSolverTest(unittest.TestCase):
    def test_solver_fuses_in_wrist_local_frame_and_returns_world_outputs(self):
        from hand_reconstruction.coordinate_frames import apply_transform, make_transform
        from hand_reconstruction.schema import INDEX_TIP, THUMB_TIP
        from hand_reconstruction.solver import HandReconstructionSolver

        human_local = _sample_keypoints(0.0)
        direct_local = _sample_keypoints(1.0)
        direct_local[THUMB_TIP] = np.array([0.20, 0.05, -0.02])
        direct_local[INDEX_TIP] = np.array([0.02, 0.20, -0.03])
        transform = make_transform(translation=np.array([0.5, -0.1, 0.25]))
        direct_world = apply_transform(transform, direct_local)
        solver = HandReconstructionSolver(
            hand="right",
            human_model=_FakeHumanModel(human_local),
            retargeter=_FakeRetargeter({"index_mcp_flex": 0.2}),
            glove_pipeline=_FakeGlovePipeline(direct_world),
            T_W_Hwrist=transform,
        )

        frame = solver.reconstruct(np.arange(21, dtype=float))

        np.testing.assert_allclose(frame.keypoints_21_in_Hwrist, human_local)
        np.testing.assert_allclose(frame.direct_glove_keypoints_21_in_W, direct_world)
        np.testing.assert_allclose(frame.direct_glove_keypoints_21_in_Hwrist, direct_local)
        np.testing.assert_allclose(
            frame.fused_keypoints_21_in_Hwrist[THUMB_TIP],
            direct_local[THUMB_TIP],
        )
        np.testing.assert_allclose(
            frame.fused_keypoints_21_in_Hwrist[INDEX_TIP],
            direct_local[INDEX_TIP],
        )
        np.testing.assert_allclose(
            frame.fused_keypoints_21_in_W,
            apply_transform(transform, frame.fused_keypoints_21_in_Hwrist),
        )
        self.assertLess(frame.diagnostics["roundtrip_error"], 1e-12)
        self.assertLess(frame.diagnostics["fingertip_error_after_fusion"], 1e-12)
        self.assertGreater(frame.diagnostics["fingertip_error_before_fusion"], 0.0)
        self.assertAlmostEqual(frame.diagnostics["transform_det"], 1.0)

    def test_solver_rejects_bad_glove_q_shape(self):
        from hand_reconstruction.solver import HandReconstructionSolver

        solver = HandReconstructionSolver(
            hand="left",
            human_model=_FakeHumanModel(_sample_keypoints()),
            retargeter=_FakeRetargeter({}),
            glove_pipeline=_FakeGlovePipeline(_sample_keypoints()),
            T_W_Hwrist=np.eye(4),
        )

        with self.assertRaisesRegex(ValueError, "q_glove must have shape"):
            solver.reconstruct(np.zeros(5))


class _FakeRetargeter:
    def __init__(self, joints):
        self.joints = dict(joints)
        self.last_q = None

    def retarget(self, q):
        self.last_q = np.asarray(q, dtype=float).copy()
        return dict(self.joints)


class _FakeHumanModel:
    def __init__(self, landmarks):
        self.landmarks = np.asarray(landmarks, dtype=float)
        self.last_joints = None

    def landmarks_from_joints(self, joints):
        self.last_joints = dict(joints)
        return self.landmarks.copy()


class _FakeSkeleton:
    def __init__(self, landmarks):
        self.landmarks = np.asarray(landmarks, dtype=float)

    def to_numpy(self):
        return self.landmarks.copy()


class _FakeGlovePipeline:
    def __init__(self, landmarks):
        self.landmarks = np.asarray(landmarks, dtype=float)
        self.last_q = None

    def reconstruct_direct(self, q):
        self.last_q = np.asarray(q, dtype=float).copy()
        return _FakeSkeleton(self.landmarks)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m unittest tests.test_solver_frame
```

Expected: FAIL with an import error for `HandReconstructionSolver`.

- [ ] **Step 3: Implement `HandReconstructionSolver` and diagnostics**

Append this code to `hand_reconstruction/solver.py` after `_validate_keypoints()`:

```python
from .coordinate_frames import apply_transform, invert_transform
from .retargeting import GLOVE_DOF, RETARGET_LIMITS
from .tip_locking import FINGERTIP_INDICES, fuse_tip_locked_landmarks


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
```

- [ ] **Step 4: Run solver tests**

Run:

```bash
python3 -m unittest tests.test_solver_frame
```

Expected: PASS.

- [ ] **Step 5: Run existing retargeting and tip-locking tests**

Run:

```bash
python3 -m unittest tests.test_retargeting tests.test_tip_locking
```

Expected: PASS.

- [ ] **Step 6: Commit if in a git repository**

Run:

```bash
git status --short
git add hand_reconstruction/solver.py tests/test_solver_frame.py
git commit -m "feat: add hand reconstruction solver"
```

Expected in this workspace: skip this step because the current directory is not a git repository.

---

### Task 4: Add Optional Joint-Angle Smoothing

**Files:**
- Modify: `hand_reconstruction/solver.py`
- Modify: `tests/test_solver_frame.py`
- Modify: `hand_reconstruction/__init__.py`

- [ ] **Step 1: Add failing tests for smoothing behavior**

Append this test class to `tests/test_solver_frame.py` before the fake helper classes:

```python
class JointAngleSmootherTest(unittest.TestCase):
    def test_smoother_applies_ema_and_max_delta(self):
        from hand_reconstruction.solver import JointAngleSmoother

        smoother = JointAngleSmoother(alpha=0.5, max_delta=0.2)

        first, first_delta = smoother.smooth({"index_mcp_flex": 0.0})
        second, second_delta = smoother.smooth({"index_mcp_flex": 1.0})

        self.assertEqual(first, {"index_mcp_flex": 0.0})
        self.assertAlmostEqual(first_delta, 0.0)
        self.assertAlmostEqual(second["index_mcp_flex"], 0.2)
        self.assertAlmostEqual(second_delta, 0.2)

    def test_smoother_reset_drops_previous_state(self):
        from hand_reconstruction.solver import JointAngleSmoother

        smoother = JointAngleSmoother(alpha=0.5, max_delta=0.1)
        smoother.smooth({"index_mcp_flex": 0.0})
        smoother.reset()
        after_reset, delta = smoother.smooth({"index_mcp_flex": 1.0})

        self.assertEqual(after_reset, {"index_mcp_flex": 1.0})
        self.assertAlmostEqual(delta, 0.0)

    def test_smoother_rejects_invalid_parameters(self):
        from hand_reconstruction.solver import JointAngleSmoother

        with self.assertRaisesRegex(ValueError, "alpha must be in"):
            JointAngleSmoother(alpha=-0.1)
        with self.assertRaisesRegex(ValueError, "alpha must be in"):
            JointAngleSmoother(alpha=1.1)
        with self.assertRaisesRegex(ValueError, "max_delta must be positive"):
            JointAngleSmoother(alpha=0.5, max_delta=0.0)
```

Also append this method to `HandReconstructionSolverTest`:

```python
    def test_solver_reports_smoothing_delta(self):
        from hand_reconstruction.solver import HandReconstructionSolver, JointAngleSmoother

        smoother = JointAngleSmoother(alpha=0.5, max_delta=0.1)
        retargeter = _SequenceRetargeter(
            [
                {"index_mcp_flex": 0.0},
                {"index_mcp_flex": 1.0},
            ]
        )
        solver = HandReconstructionSolver(
            hand="right",
            human_model=_FakeHumanModel(_sample_keypoints()),
            retargeter=retargeter,
            glove_pipeline=_FakeGlovePipeline(_sample_keypoints()),
            T_W_Hwrist=np.eye(4),
            smoother=smoother,
        )

        solver.reconstruct(np.zeros(21))
        frame = solver.reconstruct(np.zeros(21))

        self.assertAlmostEqual(frame.joint_angles["index_mcp_flex"], 0.1)
        self.assertAlmostEqual(frame.diagnostics["max_joint_delta"], 0.1)
```

Add this helper near `_FakeRetargeter`:

```python
class _SequenceRetargeter:
    def __init__(self, sequence):
        self.sequence = [dict(item) for item in sequence]
        self.index = 0

    def retarget(self, q):
        value = self.sequence[self.index]
        self.index = min(self.index + 1, len(self.sequence) - 1)
        return dict(value)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m unittest tests.test_solver_frame
```

Expected: FAIL with an import error for `JointAngleSmoother`.

- [ ] **Step 3: Implement `JointAngleSmoother`**

Insert this class in `hand_reconstruction/solver.py` after `HandReconstructionFrame` and before `_validate_keypoints()`:

```python
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
```

- [ ] **Step 4: Export the smoother from the package**

Modify `hand_reconstruction/__init__.py`:

```python
"""Hand reconstruction utilities for DexCap glove data."""

from .human_hand_model import HumanHandSkeleton
from .pipeline import HandReconstructionPipeline, reconstruct_from_link_positions
from .solver import HandReconstructionFrame, HandReconstructionSolver, JointAngleSmoother
from .tip_locking import FINGERTIP_INDICES, fuse_tip_locked_landmarks

__all__ = [
    "FINGERTIP_INDICES",
    "HandReconstructionFrame",
    "HandReconstructionPipeline",
    "HandReconstructionSolver",
    "HumanHandSkeleton",
    "JointAngleSmoother",
    "fuse_tip_locked_landmarks",
    "reconstruct_from_link_positions",
]
```

Append this package export test to `JointAngleSmootherTest`:

```python
    def test_package_exports_solver_types(self):
        from hand_reconstruction import HandReconstructionSolver, JointAngleSmoother

        self.assertEqual(HandReconstructionSolver.__name__, "HandReconstructionSolver")
        self.assertEqual(JointAngleSmoother.__name__, "JointAngleSmoother")
```

- [ ] **Step 5: Run smoothing and solver tests**

Run:

```bash
python3 -m unittest tests.test_solver_frame
```

Expected: PASS.

- [ ] **Step 6: Commit if in a git repository**

Run:

```bash
git status --short
git add hand_reconstruction/solver.py hand_reconstruction/__init__.py tests/test_solver_frame.py
git commit -m "feat: add hand joint smoothing"
```

Expected in this workspace: skip this step because the current directory is not a git repository.

---

### Task 5: Switch MeshCat Overlay to Solver Output

**Files:**
- Modify: `DexCap_v4/dexcap_glove_meshcat_stream.py`
- Modify: `DexCap_v4/test_dexcap_glove_meshcat_stream.py`

- [ ] **Step 1: Add failing display test for solver-backed overlay**

Replace `test_display_human_uses_tip_locked_fused_landmarks` in `DexCap_v4/test_dexcap_glove_meshcat_stream.py` with:

```python
    def test_display_human_uses_solver_fused_local_landmarks(self):
        from hand_reconstruction.solver import HandReconstructionFrame

        left_fused_local = _sample_landmarks(0.0)
        right_fused_local = _sample_landmarks(1.0)
        display = _make_display_for_solver_test(
            left_frame=HandReconstructionFrame(
                hand="left",
                joint_angles={"unused": 1.0},
                keypoints_21_in_Hwrist=_sample_landmarks(2.0),
                keypoints_21_in_W=_sample_landmarks(3.0),
                direct_glove_keypoints_21_in_W=_sample_landmarks(4.0),
                direct_glove_keypoints_21_in_Hwrist=_sample_landmarks(5.0),
                fused_keypoints_21_in_Hwrist=left_fused_local,
                fused_keypoints_21_in_W=_sample_landmarks(6.0),
                T_W_Hwrist=np.eye(4),
                diagnostics={},
            ),
            right_frame=HandReconstructionFrame(
                hand="right",
                joint_angles={"unused": 2.0},
                keypoints_21_in_Hwrist=_sample_landmarks(7.0),
                keypoints_21_in_W=_sample_landmarks(8.0),
                direct_glove_keypoints_21_in_W=_sample_landmarks(9.0),
                direct_glove_keypoints_21_in_Hwrist=_sample_landmarks(10.0),
                fused_keypoints_21_in_Hwrist=right_fused_local,
                fused_keypoints_21_in_W=_sample_landmarks(11.0),
                T_W_Hwrist=np.eye(4),
                diagnostics={},
            ),
        )

        display.display_human(np.arange(21), np.arange(21, 42))

        expected_left = stream._offset_landmarks_for_display(
            left_fused_local,
            display.human_skeleton_display_offset,
        )
        expected_left = stream._offset_thumb_for_display(
            expected_left,
            display.thumb_display_offset,
            "left",
            left_extra_offset=display.left_thumb_extra_display_offset,
        )
        expected_right = stream._offset_landmarks_for_display(
            right_fused_local,
            display.human_skeleton_display_offset,
        )
        expected_right = stream._offset_thumb_for_display(
            expected_right,
            display.thumb_display_offset,
            "right",
            left_extra_offset=display.left_thumb_extra_display_offset,
        )

        np.testing.assert_allclose(display._left_overlay.updated, expected_left)
        np.testing.assert_allclose(display._right_overlay.updated, expected_right)
        np.testing.assert_allclose(display._left_solver.last_q, np.arange(21))
        np.testing.assert_allclose(display._right_solver.last_q, np.arange(21, 42))
        np.testing.assert_allclose(left_fused_local, _sample_landmarks(0.0))
        np.testing.assert_allclose(right_fused_local, _sample_landmarks(1.0))
```

Append these helpers near the existing fake classes:

```python
class _FakeSolver:
    def __init__(self, frame):
        self.frame = frame
        self.last_q = None

    def reconstruct(self, q):
        self.last_q = np.asarray(q, dtype=float).copy()
        return self.frame


def _make_display_for_solver_test(*, left_frame, right_frame):
    display = stream.GloveMeshcatDisplay.__new__(stream.GloveMeshcatDisplay)
    display._left_solver = _FakeSolver(left_frame)
    display._right_solver = _FakeSolver(right_frame)
    display._left_overlay = _FakeOverlay()
    display._right_overlay = _FakeOverlay()
    display.human_skeleton_display_offset = np.array([0.01, 0.02, 0.03])
    display.thumb_display_offset = np.array([0.04, 0.01, -0.03])
    display.left_thumb_extra_display_offset = np.array([0.006, 0.0, 0.0])
    display._human_overlay_runtime_error_reported = False
    return display
```

- [ ] **Step 2: Run display test to verify it fails**

Run from the repository root:

```bash
python3 -m unittest discover -s DexCap_v4 -p 'test_dexcap_glove_meshcat_stream.py'
```

Expected: FAIL because `display_human()` still checks the old `_left_human`, `_left_retar`, and `_left_glove_pipe` attributes instead of `_left_solver`.

- [ ] **Step 3: Update display object fields**

In `DexCap_v4/dexcap_glove_meshcat_stream.py`, replace the optional overlay attribute block in `GloveMeshcatDisplay.__init__` with:

```python
        self._left_solver = None
        self._right_solver = None
        self._left_overlay = None
        self._right_overlay = None
        self._human_overlay_runtime_error_reported = False
```

- [ ] **Step 4: Build solvers in `_build_human_overlays()`**

Inside `_build_human_overlays()`, update imports to include the solver:

```python
            from hand_reconstruction.solver import HandReconstructionSolver
```

Keep the existing imports for `load_human_hand`, `rigid_fit`, `default_params`,
`HandReconstructionPipeline`, `GloveToHumanRetargeter`, and
`MeshcatSkeletonOverlay`. Remove the import for `fuse_tip_locked_landmarks`.

Inside the `try:` body, keep `params`, `left_human`, `right_human`, `left_retar`,
and `right_retar` as local variables. Replace assignments to `self._left_human`,
`self._right_human`, `self._left_retar`, `self._right_retar`,
`self._left_glove_pipe`, and align inverse fields with solver construction:

```python
            params = default_params()
            left_human = load_human_hand(params, "left")
            right_human = load_human_hand(params, "right")
            left_retar = GloveToHumanRetargeter("left")
            right_retar = GloveToHumanRetargeter("right")

            for side, human, retargeter, glove_urdf, root_name in (
                ("left", left_human, left_retar, left_urdf, LEFT_ROOT_NAME),
                ("right", right_human, right_retar, right_urdf, RIGHT_ROOT_NAME),
            ):
                glove_pipe = HandReconstructionPipeline(glove_urdf, side)
                glove_nq = glove_pipe.observer.robot.model.nq
                glove_links = glove_pipe.reconstruct_direct(np.zeros(glove_nq)).landmarks
                human_rest = human.landmarks_from_q(np.zeros(human.nq))
                align = rigid_fit(human_rest, glove_links)
                solver = HandReconstructionSolver(
                    hand=side,
                    human_model=human,
                    retargeter=retargeter,
                    glove_pipeline=glove_pipe,
                    T_W_Hwrist=align,
                )
                if side == "left":
                    self._left_solver = solver
                else:
                    self._right_solver = solver
                self.viewer[root_name][HUMAN_HAND_NODE].set_transform(align)
```

In the `except Exception` cleanup block, set only:

```python
            self._left_solver = None
            self._right_solver = None
            self._left_overlay = None
            self._right_overlay = None
```

- [ ] **Step 5: Update `display_human()` to use solver results**

Replace the `required = (...)` tuple in `display_human()` with:

```python
        required = (
            self._left_overlay,
            self._right_overlay,
            self._left_solver,
            self._right_solver,
        )
```

Replace the body inside the `try:` block with:

```python
            left_q = np.asarray(left_q, dtype=float)
            right_q = np.asarray(right_q, dtype=float)
            left_frame = self._left_solver.reconstruct(left_q)
            right_frame = self._right_solver.reconstruct(right_q)
            left_display = _offset_landmarks_for_display(
                left_frame.fused_keypoints_21_in_Hwrist,
                self.human_skeleton_display_offset,
            )
            right_display = _offset_landmarks_for_display(
                right_frame.fused_keypoints_21_in_Hwrist,
                self.human_skeleton_display_offset,
            )
            left_display = _offset_thumb_for_display(
                left_display,
                self.thumb_display_offset,
                "left",
                left_extra_offset=self.left_thumb_extra_display_offset,
            )
            right_display = _offset_thumb_for_display(
                right_display,
                self.thumb_display_offset,
                "right",
                left_extra_offset=self.left_thumb_extra_display_offset,
            )
            self._left_overlay.update(left_display)
            self._right_overlay.update(right_display)
```

Keep the existing exception handling that disables both overlays after a runtime
error.

- [ ] **Step 6: Run stream tests**

Run:

```bash
python3 -m unittest discover -s DexCap_v4 -p 'test_dexcap_glove_meshcat_stream.py'
```

Expected: PASS.

- [ ] **Step 7: Run solver tests**

Run:

```bash
python3 -m unittest tests.test_solver_frame
```

Expected: PASS.

- [ ] **Step 8: Commit if in a git repository**

Run:

```bash
git status --short
git add DexCap_v4/dexcap_glove_meshcat_stream.py DexCap_v4/test_dexcap_glove_meshcat_stream.py
git commit -m "refactor: use solver for hand skeleton overlay"
```

Expected in this workspace: skip this step because the current directory is not a git repository.

---

### Task 6: Add Solver-Frame Export Support

**Files:**
- Modify: `hand_reconstruction/export.py`
- Modify: `tests/test_solver_frame.py`

- [ ] **Step 1: Add failing export test**

Append this method to `HandReconstructionFrameTest` in `tests/test_solver_frame.py`:

```python
    def test_solver_frame_export_dict_includes_local_world_and_fused_arrays(self):
        from hand_reconstruction.export import solver_frame_to_dict
        from hand_reconstruction.solver import HandReconstructionFrame

        points = _sample_keypoints()
        frame = HandReconstructionFrame(
            hand="left",
            joint_angles={"thumb_cmc_abd": 0.1},
            keypoints_21_in_Hwrist=points,
            keypoints_21_in_W=points + np.array([1.0, 0.0, 0.0]),
            direct_glove_keypoints_21_in_W=points + np.array([2.0, 0.0, 0.0]),
            direct_glove_keypoints_21_in_Hwrist=points + np.array([3.0, 0.0, 0.0]),
            fused_keypoints_21_in_Hwrist=points + np.array([4.0, 0.0, 0.0]),
            fused_keypoints_21_in_W=points + np.array([5.0, 0.0, 0.0]),
            T_W_Hwrist=np.eye(4),
            diagnostics={"all_finite": True},
        )

        payload = solver_frame_to_dict(frame)

        self.assertEqual(payload["hand"], "left")
        self.assertEqual(payload["joint_angles"], {"thumb_cmc_abd": 0.1})
        self.assertIn("coordinate_convention", payload)
        self.assertEqual(len(payload["keypoints_21_in_Hwrist"]), 21)
        self.assertEqual(len(payload["keypoints_21_in_W"]), 21)
        self.assertEqual(len(payload["fused_keypoints_21_in_Hwrist"]), 21)
        self.assertEqual(len(payload["fused_keypoints_21_in_W"]), 21)
        self.assertEqual(payload["diagnostics"], {"all_finite": True})
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_solver_frame.HandReconstructionFrameTest.test_solver_frame_export_dict_includes_local_world_and_fused_arrays
```

Expected: FAIL with an import error for `solver_frame_to_dict`.

- [ ] **Step 3: Add export helper**

Modify `hand_reconstruction/export.py` by importing the solver type:

```python
from .solver import HandReconstructionFrame
```

Append this function after `skeleton_to_dict()`:

```python
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
```

Also update imports at the top of `export.py` to include:

```python
from .coordinate_frames import default_coordinate_convention
```

- [ ] **Step 4: Run export test**

Run:

```bash
python3 -m unittest tests.test_solver_frame.HandReconstructionFrameTest.test_solver_frame_export_dict_includes_local_world_and_fused_arrays
```

Expected: PASS.

- [ ] **Step 5: Run full solver test file**

Run:

```bash
python3 -m unittest tests.test_solver_frame
```

Expected: PASS.

- [ ] **Step 6: Commit if in a git repository**

Run:

```bash
git status --short
git add hand_reconstruction/export.py tests/test_solver_frame.py
git commit -m "feat: export hand solver frames"
```

Expected in this workspace: skip this step because the current directory is not a git repository.

---

### Task 7: Final Verification and Memory Update

**Files:**
- Modify: `docs/current-memory.md`

- [ ] **Step 1: Run targeted unit tests**

Run:

```bash
python3 -m unittest tests.test_solver_frame tests.test_retargeting tests.test_tip_locking
```

Expected: PASS.

- [ ] **Step 2: Run MeshCat stream unit tests**

Run:

```bash
python3 -m unittest discover -s DexCap_v4 -p 'test_dexcap_glove_meshcat_stream.py'
```

Expected: PASS.

- [ ] **Step 3: Run existing hand reconstruction tests**

Run:

```bash
python3 -m unittest tests.test_hand_schema tests.test_human_hand_builder tests.test_glove_observation tests.test_visualize_meshcat tests.test_reconstruct_hand_frame_cli
```

Expected: PASS or skip only where the existing test suite already skips missing optional Pinocchio/MeshCat dependencies.

- [ ] **Step 4: Compile changed Python files**

Run:

```bash
python3 -m py_compile hand_reconstruction/coordinate_frames.py hand_reconstruction/solver.py hand_reconstruction/export.py DexCap_v4/dexcap_glove_meshcat_stream.py
```

Expected: no output and exit code 0.

- [ ] **Step 5: Update memory with implementation status**

Modify `docs/current-memory.md` under `## Current Implementation State` to add:

```markdown
Solver implementation:

- `hand_reconstruction/solver.py`
- `HandReconstructionFrame` stores local/world raw and fused keypoints.
- `HandReconstructionSolver` performs retargeting, human FK, direct glove FK,
  local-frame tip-lock fusion, world transform output, and diagnostics.
- `JointAngleSmoother` supports optional per-joint EMA and max-delta limiting.
- `DexCap_v4/dexcap_glove_meshcat_stream.py` consumes solver output for the
  human overlay while keeping display offsets display-only.
```

- [ ] **Step 6: Search for red-flag markers in the plan and changed docs**

Run:

```bash
rg -n "TO""DO|TB""D|place""holder|implement ""later" docs/current-memory.md docs/superpowers/plans/2026-07-02-hand-keypoint-coordinate-solver.md
```

Expected: no matches and exit code 1.

- [ ] **Step 7: Commit if in a git repository**

Run:

```bash
git status --short
git add docs/current-memory.md
git commit -m "docs: record hand solver implementation"
```

Expected in this workspace: skip this step because the current directory is not a git repository.

---

## Self-Review Notes

Spec coverage:

- Coordinate frames: Tasks 1, 2, and 3 add validated transforms, local/world arrays, and local-frame fusion.
- Solver output object: Task 2 adds `HandReconstructionFrame`.
- Tip-lock in wrist local frame: Task 3 tests and implements world-to-local transform before fusion.
- Temporal smoothing: Task 4 adds `JointAngleSmoother` and solver integration.
- Diagnostics: Task 3 adds fingertip error, transform determinant, limit hit, max delta, round-trip error, and finite checks.
- MeshCat integration: Task 5 switches `display_human()` to solver results while preserving display offsets.
- Export behavior: Task 6 adds `solver_frame_to_dict()`.
- Testing: Each task includes exact unit tests and commands.

Type consistency:

- The result object uses the exact names from the spec:
  `keypoints_21_in_Hwrist`, `keypoints_21_in_W`,
  `direct_glove_keypoints_21_in_W`, `direct_glove_keypoints_21_in_Hwrist`,
  `fused_keypoints_21_in_Hwrist`, `fused_keypoints_21_in_W`, and `T_W_Hwrist`.
- The display layer consumes `fused_keypoints_21_in_Hwrist` because the MeshCat
  human-hand node is already transformed by `T_W_Hwrist`.
- The export layer emits both wrist-local and world-frame data.

Repository note:

- `/home/lbw/DexCap-GL-retargeting` currently reports `fatal: not a git repository`,
  so commit steps are documented for future git-backed execution but should be
  skipped in this workspace.
