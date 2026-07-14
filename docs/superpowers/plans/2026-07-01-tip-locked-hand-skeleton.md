# Tip-Locked Hand Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the live 21-point human skeleton overlay hard-lock its five fingertip landmarks to the corresponding DexGlove exoskeleton fingertips while keeping intermediate joints human-looking.

**Architecture:** Add a pure NumPy fusion module under `hand_reconstruction/` that combines human FK landmarks with direct DexGlove landmarks. Store each hand's glove observation pipeline and human-to-glove alignment inverse in the MeshCat stream, convert glove fingertip observations into the overlay's local frame, then update the overlay with fused landmarks.

**Tech Stack:** Python 3, NumPy, unittest, existing `hand_reconstruction` package, existing MeshCat/Pinocchio optional runtime path.

---

## File Structure

- Create `hand_reconstruction/tip_locking.py`: pure landmark fusion logic; no MeshCat or Pinocchio dependency.
- Create `tests/test_tip_locking.py`: unit tests for fingertip hard lock, intermediate point behavior, validation, and degenerate fallback.
- Modify `hand_reconstruction/__init__.py`: export the fusion function and fingertip index constant for callers and tests.
- Modify `DexCap_v4/dexcap_glove_meshcat_stream.py`: store direct glove pipelines and alignment inverses, transform direct glove landmarks into overlay-local coordinates, and update overlays with fused landmarks.
- Modify `DexCap_v4/test_dexcap_glove_meshcat_stream.py`: add fake-based stream tests that do not require MeshCat or Pinocchio.

Repository note: `/home/lbw/DexCap-GL-retargeting` is currently not a git repository. The execution path should use verification checkpoints instead of commit steps in this workspace. If this plan is executed from a git clone, commit after each task using the task summary as the commit message.

## Task 1: Pure Tip-Locking Fusion Module

**Files:**
- Create: `hand_reconstruction/tip_locking.py`
- Create: `tests/test_tip_locking.py`
- Modify: `hand_reconstruction/__init__.py`

- [ ] **Step 1: Write the failing fusion tests**

Create `tests/test_tip_locking.py` with:

```python
import unittest

import numpy as np

from hand_reconstruction.schema import (
    FINGER_CHAINS,
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    NUM_LANDMARKS,
)


class TipLockingTest(unittest.TestCase):
    def test_fingertips_are_hard_locked_to_direct_landmarks(self):
        from hand_reconstruction.tip_locking import (
            FINGERTIP_INDICES,
            fuse_tip_locked_landmarks,
        )

        human = _human_landmarks()
        direct = _direct_landmarks()

        fused = fuse_tip_locked_landmarks(human, direct)

        self.assertEqual(fused.shape, (NUM_LANDMARKS, 3))
        for tip_idx in FINGERTIP_INDICES:
            np.testing.assert_allclose(fused[tip_idx], direct[tip_idx], atol=0.0)

    def test_intermediate_points_follow_human_shape_not_direct_links(self):
        from hand_reconstruction.tip_locking import fuse_tip_locked_landmarks

        human = _human_landmarks()
        direct = _direct_landmarks()
        direct[INDEX_PIP] = np.array([10.0, 10.0, 10.0])
        direct[INDEX_DIP] = np.array([11.0, 11.0, 11.0])

        fused = fuse_tip_locked_landmarks(human, direct)

        self.assertFalse(np.allclose(fused[INDEX_PIP], direct[INDEX_PIP]))
        self.assertFalse(np.allclose(fused[INDEX_DIP], direct[INDEX_DIP]))
        self.assertFalse(np.allclose(fused[INDEX_PIP], human[INDEX_PIP]))
        np.testing.assert_allclose(fused[INDEX_TIP], direct[INDEX_TIP])

    def test_rejects_bad_shapes_and_nonfinite_values(self):
        from hand_reconstruction.tip_locking import fuse_tip_locked_landmarks

        human = _human_landmarks()
        direct = _direct_landmarks()

        with self.assertRaisesRegex(ValueError, "human_landmarks must have shape"):
            fuse_tip_locked_landmarks(np.zeros((5, 3)), direct)

        direct_with_nan = direct.copy()
        direct_with_nan[INDEX_TIP, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "direct_landmarks must contain only finite"):
            fuse_tip_locked_landmarks(human, direct_with_nan)

    def test_degenerate_human_root_to_tip_uses_linear_fallback(self):
        from hand_reconstruction.tip_locking import fuse_tip_locked_landmarks

        human = _human_landmarks()
        direct = _direct_landmarks()
        root = human[INDEX_MCP].copy()
        human[INDEX_PIP] = root
        human[INDEX_DIP] = root
        human[INDEX_TIP] = root
        direct[INDEX_TIP] = root + np.array([0.0, 0.09, 0.0])

        fused = fuse_tip_locked_landmarks(human, direct)

        np.testing.assert_allclose(fused[INDEX_PIP], root + np.array([0.0, 0.03, 0.0]))
        np.testing.assert_allclose(fused[INDEX_DIP], root + np.array([0.0, 0.06, 0.0]))
        np.testing.assert_allclose(fused[INDEX_TIP], direct[INDEX_TIP])
        self.assertTrue(np.all(np.isfinite(fused)))

    def test_schema_is_hand_agnostic(self):
        from hand_reconstruction.tip_locking import FINGERTIP_INDICES
        from hand_reconstruction.schema import (
            INDEX_TIP,
            MIDDLE_TIP,
            PINKY_TIP,
            RING_TIP,
            THUMB_TIP,
        )

        self.assertEqual(
            FINGERTIP_INDICES,
            (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP),
        )


def _human_landmarks():
    points = np.zeros((NUM_LANDMARKS, 3), dtype=float)
    x_offsets = {
        "thumb": -0.045,
        "index": -0.020,
        "middle": 0.000,
        "ring": 0.020,
        "pinky": 0.040,
    }
    for finger, chain in FINGER_CHAINS.items():
        x = x_offsets[finger]
        points[chain[1]] = np.array([x, 0.02, 0.00])
        points[chain[2]] = np.array([x + 0.004, 0.05, -0.010])
        points[chain[3]] = np.array([x + 0.008, 0.075, -0.018])
        points[chain[4]] = np.array([x + 0.010, 0.10, -0.020])
    return points


def _direct_landmarks():
    points = np.zeros((NUM_LANDMARKS, 3), dtype=float)
    x_offsets = {
        "thumb": -0.040,
        "index": -0.015,
        "middle": 0.005,
        "ring": 0.025,
        "pinky": 0.045,
    }
    for finger, chain in FINGER_CHAINS.items():
        x = x_offsets[finger]
        points[chain[1]] = np.array([x, 0.018, 0.015])
        points[chain[2]] = np.array([x, 0.052, 0.010])
        points[chain[3]] = np.array([x, 0.078, 0.005])
        points[chain[4]] = np.array([x, 0.112, 0.000])
    return points


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing fusion tests**

Run from `/home/lbw/DexCap-GL-retargeting`:

```bash
python3 -m unittest tests/test_tip_locking.py
```

Expected: fail with `ModuleNotFoundError: No module named 'hand_reconstruction.tip_locking'`.

- [ ] **Step 3: Implement the fusion module**

Create `hand_reconstruction/tip_locking.py` with:

```python
"""Tip-locked fusion for 21-point hand skeleton landmarks."""

from __future__ import annotations

import numpy as np

from .schema import (
    FINGER_CHAINS,
    INDEX_TIP,
    MIDDLE_TIP,
    NUM_LANDMARKS,
    PINKY_TIP,
    RING_TIP,
    THUMB_TIP,
)

FINGERTIP_INDICES = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)


def fuse_tip_locked_landmarks(
    human_landmarks: np.ndarray,
    direct_landmarks: np.ndarray,
    *,
    eps: float = 1e-9,
) -> np.ndarray:
    """Return human-shaped landmarks whose fingertips equal direct landmarks.

    Inputs and output use the same coordinate frame. The wrist and each finger
    root stay anchored to the human landmarks. Finger tips are hard locked to
    direct landmarks. Intermediate joints keep the human chain's bend offsets
    after rotating and scaling them into the root-to-locked-tip segment.
    """
    human = _as_landmarks(human_landmarks, "human_landmarks")
    direct = _as_landmarks(direct_landmarks, "direct_landmarks")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    fused = human.copy()
    for chain in FINGER_CHAINS.values():
        _fuse_finger_chain(fused, human, direct, chain, eps)
    return fused


def _fuse_finger_chain(
    fused: np.ndarray,
    human: np.ndarray,
    direct: np.ndarray,
    chain: tuple[int, int, int, int, int],
    eps: float,
) -> None:
    root_idx = chain[1]
    interior = chain[2:-1]
    tip_idx = chain[-1]

    root = human[root_idx]
    human_tip = human[tip_idx]
    locked_tip = direct[tip_idx]
    human_delta = human_tip - root
    target_delta = locked_tip - root
    human_len = float(np.linalg.norm(human_delta))
    target_len = float(np.linalg.norm(target_delta))

    fused[root_idx] = root
    fused[tip_idx] = locked_tip

    if human_len <= eps or target_len <= eps:
        _linear_fallback(fused, root, locked_tip, interior)
        return

    human_axis = human_delta / human_len
    target_axis = target_delta / target_len
    rotation = _rotation_mapping(human_axis, target_axis)
    scale = target_len / human_len

    for idx in interior:
        relative = human[idx] - root
        alpha = float(np.clip(np.dot(relative, human_axis) / human_len, 0.0, 1.0))
        baseline = root + alpha * human_len * human_axis
        bend_offset = relative - (baseline - root)
        fused[idx] = root + alpha * target_len * target_axis + rotation @ (bend_offset * scale)


def _linear_fallback(
    fused: np.ndarray,
    root: np.ndarray,
    locked_tip: np.ndarray,
    interior: tuple[int, ...],
) -> None:
    count = len(interior) + 1
    for offset, idx in enumerate(interior, start=1):
        alpha = offset / count
        fused[idx] = (1.0 - alpha) * root + alpha * locked_tip


def _rotation_mapping(source_axis: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
    cross = np.cross(source_axis, target_axis)
    sin_theta = float(np.linalg.norm(cross))
    cos_theta = float(np.dot(source_axis, target_axis))

    if sin_theta < 1e-12:
        if cos_theta > 0.0:
            return np.eye(3, dtype=float)
        perp = _unit_perpendicular(source_axis)
        return 2.0 * np.outer(perp, perp) - np.eye(3, dtype=float)

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=float,
    )
    factor = (1.0 - cos_theta) / (sin_theta * sin_theta)
    return np.eye(3, dtype=float) + skew + (skew @ skew) * factor


def _unit_perpendicular(vector: np.ndarray) -> np.ndarray:
    seed = np.array([1.0, 0.0, 0.0]) if abs(vector[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp = seed - vector * float(np.dot(vector, seed))
    norm = float(np.linalg.norm(perp))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return perp / norm


def _as_landmarks(value: np.ndarray, name: str) -> np.ndarray:
    landmarks = np.asarray(value, dtype=float)
    if landmarks.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"{name} must have shape ({NUM_LANDMARKS}, 3), got {landmarks.shape}")
    if not np.all(np.isfinite(landmarks)):
        raise ValueError(f"{name} must contain only finite values")
    return landmarks.copy()
```

- [ ] **Step 4: Export the fusion API**

Modify `hand_reconstruction/__init__.py` to:

```python
"""Hand reconstruction utilities for DexCap glove data."""

from .human_hand_model import HumanHandSkeleton
from .pipeline import HandReconstructionPipeline, reconstruct_from_link_positions
from .tip_locking import FINGERTIP_INDICES, fuse_tip_locked_landmarks

__all__ = [
    "FINGERTIP_INDICES",
    "HandReconstructionPipeline",
    "HumanHandSkeleton",
    "fuse_tip_locked_landmarks",
    "reconstruct_from_link_positions",
]
```

- [ ] **Step 5: Run fusion tests**

Run from `/home/lbw/DexCap-GL-retargeting`:

```bash
python3 -m unittest tests/test_tip_locking.py
```

Expected: all 5 tests pass.

## Task 2: Stream Overlay Integration

**Files:**
- Modify: `DexCap_v4/dexcap_glove_meshcat_stream.py`
- Modify: `DexCap_v4/test_dexcap_glove_meshcat_stream.py`

- [ ] **Step 1: Add failing stream tests**

Append this test method inside `DexCapGloveMeshcatStreamTest` in `DexCap_v4/test_dexcap_glove_meshcat_stream.py`:

```python
    def test_display_human_uses_tip_locked_fused_landmarks(self):
        from hand_reconstruction.tip_locking import (
            FINGERTIP_INDICES,
            fuse_tip_locked_landmarks,
        )

        display = stream.GloveMeshcatDisplay.__new__(stream.GloveMeshcatDisplay)
        left_human = _sample_landmarks(0.0)
        right_human = _sample_landmarks(0.5)
        left_direct = _sample_landmarks(1.0)
        right_direct = _sample_landmarks(1.5)
        left_align_inv = _translation_matrix([-0.25, 0.0, 0.0])
        right_align_inv = _translation_matrix([0.25, 0.0, 0.0])

        display._left_overlay = _FakeOverlay()
        display._right_overlay = _FakeOverlay()
        display._left_retar = _FakeRetargeter()
        display._right_retar = _FakeRetargeter()
        display._left_human = _FakeHuman(left_human)
        display._right_human = _FakeHuman(right_human)
        display._left_glove_pipe = _FakePipeline(left_direct)
        display._right_glove_pipe = _FakePipeline(right_direct)
        display._left_human_align_inv = left_align_inv
        display._right_human_align_inv = right_align_inv
        display._fuse_tip_locked_landmarks = fuse_tip_locked_landmarks
        display._human_overlay_runtime_error_reported = False

        display.display_human(np.zeros(21), np.ones(21))

        left_direct_local = stream._transform_landmarks(left_align_inv, left_direct)
        right_direct_local = stream._transform_landmarks(right_align_inv, right_direct)
        expected_left = fuse_tip_locked_landmarks(left_human, left_direct_local)
        expected_right = fuse_tip_locked_landmarks(right_human, right_direct_local)
        np.testing.assert_allclose(display._left_overlay.updated, expected_left)
        np.testing.assert_allclose(display._right_overlay.updated, expected_right)
        for tip_idx in FINGERTIP_INDICES:
            np.testing.assert_allclose(display._left_overlay.updated[tip_idx], left_direct_local[tip_idx])
            np.testing.assert_allclose(display._right_overlay.updated[tip_idx], right_direct_local[tip_idx])

    def test_transform_landmarks_applies_homogeneous_transform(self):
        landmarks = np.zeros((21, 3))
        landmarks[8] = np.array([1.0, 2.0, 3.0])
        transform = _translation_matrix([0.5, -1.0, 2.0])

        transformed = stream._transform_landmarks(transform, landmarks)

        np.testing.assert_allclose(transformed[8], np.array([1.5, 1.0, 5.0]))
```

Append these helpers near the bottom of `DexCap_v4/test_dexcap_glove_meshcat_stream.py`, before `if __name__ == "__main__":`:

```python
def _sample_landmarks(offset):
    points = np.zeros((21, 3), dtype=float)
    for idx in range(21):
        points[idx] = np.array([offset + idx * 0.01, idx * 0.02, -idx * 0.005])
    return points


def _translation_matrix(xyz):
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


class _FakeRetargeter:
    def retarget(self, q):
        return {"unused": float(np.asarray(q).sum())}


class _FakeHuman:
    def __init__(self, landmarks):
        self._landmarks = np.asarray(landmarks, dtype=float)

    def landmarks_from_joints(self, joints):
        return self._landmarks.copy()


class _FakeSkeleton:
    def __init__(self, landmarks):
        self._landmarks = np.asarray(landmarks, dtype=float)

    def to_numpy(self):
        return self._landmarks.copy()


class _FakePipeline:
    def __init__(self, landmarks):
        self._landmarks = np.asarray(landmarks, dtype=float)

    def reconstruct_direct(self, q):
        return _FakeSkeleton(self._landmarks)


class _FakeOverlay:
    def __init__(self):
        self.updated = None

    def update(self, landmarks):
        self.updated = np.asarray(landmarks, dtype=float).copy()
```

- [ ] **Step 2: Run the failing stream tests**

Run from `/home/lbw/DexCap-GL-retargeting`:

```bash
python3 -m unittest DexCap_v4/test_dexcap_glove_meshcat_stream.py
```

Expected: fail because `dexcap_glove_meshcat_stream` does not define `_transform_landmarks`.

- [ ] **Step 3: Store direct pipelines and alignment inverses**

In `DexCap_v4/dexcap_glove_meshcat_stream.py`, add these attributes in `GloveMeshcatDisplay.__init__` after the existing overlay attributes:

```python
        self._left_glove_pipe = None
        self._right_glove_pipe = None
        self._left_human_align_inv = None
        self._right_human_align_inv = None
        self._fuse_tip_locked_landmarks = None
        self._human_overlay_runtime_error_reported = False
```

In `_build_human_overlays()`, add the import:

```python
            from hand_reconstruction.tip_locking import fuse_tip_locked_landmarks
```

Set the fusion function after retargeter construction:

```python
            self._fuse_tip_locked_landmarks = fuse_tip_locked_landmarks
```

Inside the existing loop that creates `glove_pipe`, after `align = rigid_fit(human_rest, glove_links)`, add:

```python
                if side == "left":
                    self._left_glove_pipe = glove_pipe
                    self._left_human_align_inv = np.linalg.inv(align)
                else:
                    self._right_glove_pipe = glove_pipe
                    self._right_human_align_inv = np.linalg.inv(align)
```

In the exception handler at the end of `_build_human_overlays()`, also reset:

```python
            self._left_glove_pipe = None
            self._right_glove_pipe = None
            self._left_human_align_inv = None
            self._right_human_align_inv = None
            self._fuse_tip_locked_landmarks = None
```

- [ ] **Step 4: Add landmark transform helper**

Add this helper near `_package_dirs()` in `DexCap_v4/dexcap_glove_meshcat_stream.py`:

```python
def _transform_landmarks(transform: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=float)
    if points.shape != (21, 3):
        raise ValueError(f"landmarks must have shape (21, 3), got {points.shape}")
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {matrix.shape}")
    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    return (matrix @ homogeneous.T).T[:, :3]
```

- [ ] **Step 5: Replace `display_human()` with fused update logic**

Replace `display_human()` in `DexCap_v4/dexcap_glove_meshcat_stream.py` with:

```python
    def display_human(self, left_q: np.ndarray, right_q: np.ndarray) -> None:
        """Retarget both glove vectors and move the tip-locked human overlays."""
        required = (
            self._left_overlay,
            self._right_overlay,
            self._left_retar,
            self._right_retar,
            self._left_human,
            self._right_human,
            self._left_glove_pipe,
            self._right_glove_pipe,
            self._left_human_align_inv,
            self._right_human_align_inv,
            self._fuse_tip_locked_landmarks,
        )
        if any(item is None for item in required):
            return

        try:
            left_joints = self._left_retar.retarget(np.asarray(left_q, dtype=float))
            right_joints = self._right_retar.retarget(np.asarray(right_q, dtype=float))
            left_human = self._left_human.landmarks_from_joints(left_joints)
            right_human = self._right_human.landmarks_from_joints(right_joints)
            left_direct = _transform_landmarks(
                self._left_human_align_inv,
                self._left_glove_pipe.reconstruct_direct(np.asarray(left_q, dtype=float)).to_numpy(),
            )
            right_direct = _transform_landmarks(
                self._right_human_align_inv,
                self._right_glove_pipe.reconstruct_direct(np.asarray(right_q, dtype=float)).to_numpy(),
            )
            self._left_overlay.update(
                self._fuse_tip_locked_landmarks(left_human, left_direct)
            )
            self._right_overlay.update(
                self._fuse_tip_locked_landmarks(right_human, right_direct)
            )
        except Exception as exc:
            if not self._human_overlay_runtime_error_reported:
                print(f"人手骨架更新失败，暂停骨架更新: {exc}", flush=True)
                self._human_overlay_runtime_error_reported = True
            self._left_overlay = None
            self._right_overlay = None
```

- [ ] **Step 6: Run stream tests**

Run from `/home/lbw/DexCap-GL-retargeting`:

```bash
python3 -m unittest DexCap_v4/test_dexcap_glove_meshcat_stream.py
```

Expected: all tests pass.

## Task 3: Verification Sweep

**Files:**
- Verify: `hand_reconstruction/tip_locking.py`
- Verify: `hand_reconstruction/__init__.py`
- Verify: `DexCap_v4/dexcap_glove_meshcat_stream.py`
- Verify: related test files

- [ ] **Step 1: Run focused hand reconstruction tests**

Run from `/home/lbw/DexCap-GL-retargeting`:

```bash
python3 -m unittest tests/test_tip_locking.py tests/test_retargeting.py tests/test_glove_observation.py tests/test_stream_overlay.py
```

Expected: all tests pass; Pinocchio-dependent tests may skip if Pinocchio is unavailable in the active Python environment.

- [ ] **Step 2: Run stream tests**

Run from `/home/lbw/DexCap-GL-retargeting`:

```bash
python3 -m unittest DexCap_v4/test_dexcap_glove_meshcat_stream.py
```

Expected: all tests pass.

- [ ] **Step 3: Compile changed Python files**

Run from `/home/lbw/DexCap-GL-retargeting`:

```bash
python3 -m py_compile hand_reconstruction/tip_locking.py hand_reconstruction/__init__.py DexCap_v4/dexcap_glove_meshcat_stream.py DexCap_v4/test_dexcap_glove_meshcat_stream.py tests/test_tip_locking.py
```

Expected: command exits with status 0 and prints no syntax errors.

- [ ] **Step 4: Record git status if the executor is in a git repo**

Run from `/home/lbw/DexCap-GL-retargeting`:

```bash
git status --short
```

Expected in the current workspace: `fatal: not a git repository`. In a git clone, expected changed files are `hand_reconstruction/tip_locking.py`, `hand_reconstruction/__init__.py`, `DexCap_v4/dexcap_glove_meshcat_stream.py`, `DexCap_v4/test_dexcap_glove_meshcat_stream.py`, and `tests/test_tip_locking.py`.
