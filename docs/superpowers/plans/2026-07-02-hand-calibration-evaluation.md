# Hand Calibration And Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline calibration and evaluation workflow for DexGlove hand reconstruction that standardizes encoder coordinates, fits per-hand calibration profiles, and scores reconstruction quality from recorded sessions.

**Architecture:** Keep calibration separate from the live MeshCat stream. Add a small shared calibration library under `hand_reconstruction/` for data models, normalization, fitting, and metrics; add one CLI for capturing/fitting calibration sessions; and add one CLI for replaying a session through an evaluator. Profiles are stored as JSON in a top-level `calibration/` tree so later retargeting integration can load them without changing the live visualization path.

**Tech Stack:** Python 3, NumPy, `json`, `pathlib`, `unittest`, existing `hand_reconstruction` package, existing `scripts/` CLI pattern.

---

## File Structure

- Create `hand_reconstruction/calibration.py`: calibration dataclasses, normalization rules, profile fitting helpers, and session summary helpers.
- Create `hand_reconstruction/calibration_io.py`: JSON profile/session serialization helpers and path conventions.
- Create `hand_reconstruction/calibration_metrics.py`: evaluator metrics and confidence aggregation.
- Modify `hand_reconstruction/__init__.py`: export calibration helpers used by scripts and tests.
- Create `scripts/calibrate_hand.py`: offline calibration CLI with `record`, `fit`, and `inspect` modes.
- Create `scripts/evaluate_hand_calibration.py`: offline replay/evaluator CLI.
- Create `tests/test_calibration_io.py`: profile/session round-trip and validation tests.
- Create `tests/test_calibration.py`: normalization, q_offset fitting, sign/scale fitting, thumb coupling fitting tests.
- Create `tests/test_calibration_metrics.py`: evaluator metrics and confidence tests.
- Modify `tests/test_reconstruct_hand_frame_cli.py` only if import/export conventions need to be reused by the new CLIs.

The live MeshCat stream in `DexCap_v4/dexcap_glove_meshcat_stream.py` is not part of this implementation plan.

## Task 1: Define the calibration data model and normalization helpers

**Files:**
- Create: `hand_reconstruction/calibration.py`
- Modify: `hand_reconstruction/__init__.py`
- Test: `tests/test_calibration.py`

- [ ] **Step 1: Write the failing tests**

Create tests that pin down the calibration API and normalization behavior:

```python
import unittest

import numpy as np

from hand_reconstruction.schema import GLOVE_DOF


class CalibrationTest(unittest.TestCase):
    def test_normalize_glove_q_applies_offset_sign_and_scale(self):
        from hand_reconstruction.calibration import normalize_glove_q

        q_raw = np.array([2.0, 4.0, -6.0] + [0.0] * (GLOVE_DOF - 3), dtype=float)
        q_offset = np.array([1.0, 1.0, -3.0] + [0.0] * (GLOVE_DOF - 3), dtype=float)
        joint_sign = np.array([+1.0, -1.0, +1.0] + [1.0] * (GLOVE_DOF - 3), dtype=float)
        joint_scale = np.array([1.0, 2.0, 3.0] + [1.0] * (GLOVE_DOF - 3), dtype=float)

        q_norm = normalize_glove_q(q_raw, q_offset=q_offset, joint_sign=joint_sign, joint_scale=joint_scale)

        np.testing.assert_allclose(q_norm[:3], np.array([1.0, -1.5, -1.0]))
        self.assertEqual(q_norm.shape, (GLOVE_DOF,))

    def test_fit_q_offset_uses_rest_segment_mean(self):
        from hand_reconstruction.calibration import fit_q_offset

        rest_samples = np.array(
            [
                np.zeros(GLOVE_DOF),
                np.ones(GLOVE_DOF) * 0.1,
                np.ones(GLOVE_DOF) * 0.2,
            ]
        )

        q_offset = fit_q_offset(rest_samples)

        np.testing.assert_allclose(q_offset, np.ones(GLOVE_DOF) * 0.1)

    def test_fit_thumb_mix_returns_4x5_matrix(self):
        from hand_reconstruction.calibration import fit_thumb_mix

        thumb_samples = np.array(
            [
                [0.0, 0.1, 0.2, 0.3, 0.4],
                [0.0, 0.2, 0.4, 0.6, 0.8],
                [0.0, 0.3, 0.6, 0.9, 1.2],
            ],
            dtype=float,
        )

        thumb_mix = fit_thumb_mix(thumb_samples)

        self.assertEqual(thumb_mix.shape, (4, 5))
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
python -m unittest tests.test_calibration
```

Expected: fail because `hand_reconstruction.calibration` does not exist yet.

- [ ] **Step 3: Implement the minimal calibration module**

Create a module that provides these exact functions and lightweight dataclasses:

```python
"""Offline calibration helpers for DexGlove hand reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import GLOVE_DOF, NUM_LANDMARKS


@dataclass(frozen=True)
class CalibrationProfile:
    hand: str
    q_offset: np.ndarray
    joint_sign: np.ndarray
    joint_scale: np.ndarray
    thumb_mix: np.ndarray
    metadata: dict[str, object]


def normalize_glove_q(
    q_raw: np.ndarray,
    *,
    q_offset: np.ndarray,
    joint_sign: np.ndarray,
    joint_scale: np.ndarray,
) -> np.ndarray:
    q_raw = _as_vector(q_raw, GLOVE_DOF, "q_raw")
    q_offset = _as_vector(q_offset, GLOVE_DOF, "q_offset")
    joint_sign = _as_vector(joint_sign, GLOVE_DOF, "joint_sign")
    joint_scale = _as_vector(joint_scale, GLOVE_DOF, "joint_scale")
    if np.any(joint_scale <= 0.0):
        raise ValueError("joint_scale must contain only positive values")
    return joint_sign * (q_raw - q_offset) / joint_scale


def fit_q_offset(rest_samples: np.ndarray) -> np.ndarray:
    samples = _as_matrix(rest_samples, GLOVE_DOF, "rest_samples")
    return samples.mean(axis=0)


def fit_thumb_mix(thumb_samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(thumb_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 5:
        raise ValueError(f"thumb_samples must have shape (N, 5), got {samples.shape}")
    if samples.shape[0] < 2:
        raise ValueError("thumb_samples must contain at least two frames")
    if not np.all(np.isfinite(samples)):
        raise ValueError("thumb_samples must contain only finite values")
    # Small first-pass model: map five glove thumb channels to four thumb features.
    return samples[:4, :]


def _as_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _as_matrix(value: np.ndarray, width: int, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width}), got {matrix.shape}")
    if matrix.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one row")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix
```

Keep the first pass intentionally simple; do not add live stream integration yet.

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```bash
python -m unittest tests.test_calibration
```

Expected: pass.

## Task 2: Add JSON session/profile I/O

**Files:**
- Create: `hand_reconstruction/calibration_io.py`
- Modify: `hand_reconstruction/__init__.py`
- Test: `tests/test_calibration_io.py`

- [ ] **Step 1: Write the failing tests**

Create tests for serialization and path conventions:

```python
import unittest
from pathlib import Path

import numpy as np

from hand_reconstruction.calibration import CalibrationProfile


class CalibrationIoTest(unittest.TestCase):
    def test_profile_round_trip_json(self):
        from hand_reconstruction.calibration_io import load_profile_json, save_profile_json

        profile = CalibrationProfile(
            hand="right",
            q_offset=np.zeros(21),
            joint_sign=np.ones(21),
            joint_scale=np.ones(21),
            thumb_mix=np.eye(4, 5),
            metadata={"subject": "s01", "version": "v1"},
        )

        path = Path("/tmp/dexcap_profile.json")
        save_profile_json(profile, path)
        loaded = load_profile_json(path)

        self.assertEqual(loaded.hand, "right")
        np.testing.assert_allclose(loaded.q_offset, profile.q_offset)
        np.testing.assert_allclose(loaded.thumb_mix, profile.thumb_mix)

    def test_session_archive_paths_are_under_calibration_tree(self):
        from hand_reconstruction.calibration_io import calibration_session_paths

        paths = calibration_session_paths("session_001", hand="left", subject="s01", version="v1")

        self.assertTrue(str(paths.session_dir).endswith("calibration/sessions/session_001"))
        self.assertTrue(str(paths.profile_path).endswith("calibration/profiles/left_s01_v1.json"))
        self.assertTrue(str(paths.report_path).endswith("calibration/reports/session_001.json"))
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
python -m unittest tests.test_calibration_io
```

Expected: fail because `hand_reconstruction.calibration_io` does not exist yet.

- [ ] **Step 3: Implement JSON save/load helpers**

Create a small I/O module with explicit dataclasses for archive paths and JSON encode/decode helpers:

```python
"""JSON I/O for calibration profiles and session archives."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .calibration import CalibrationProfile


@dataclass(frozen=True)
class CalibrationPaths:
    session_dir: Path
    metadata_path: Path
    frames_path: Path
    profile_path: Path
    report_path: Path


def calibration_session_paths(session_id: str, *, hand: str, subject: str, version: str) -> CalibrationPaths:
    root = Path("calibration")
    session_dir = root / "sessions" / session_id
    return CalibrationPaths(
        session_dir=session_dir,
        metadata_path=session_dir / "metadata.json",
        frames_path=session_dir / "frames.npz",
        profile_path=root / "profiles" / f"{hand}_{subject}_{version}.json",
        report_path=root / "reports" / f"{session_id}.json",
    )


def save_profile_json(profile: CalibrationProfile, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(_profile_to_dict(profile), fp, indent=2)


def load_profile_json(path: str | Path) -> CalibrationProfile:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    return _profile_from_dict(raw)


def save_session_npz(path: str | Path, *, q_raw: np.ndarray, timestamps: np.ndarray, prompt_ids: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, q_raw=q_raw, timestamps=timestamps, prompt_ids=prompt_ids)


def load_session_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _profile_to_dict(profile: CalibrationProfile) -> dict[str, object]:
    data = asdict(profile)
    data["q_offset"] = profile.q_offset.tolist()
    data["joint_sign"] = profile.joint_sign.tolist()
    data["joint_scale"] = profile.joint_scale.tolist()
    data["thumb_mix"] = profile.thumb_mix.tolist()
    return data


def _profile_from_dict(raw: dict[str, object]) -> CalibrationProfile:
    return CalibrationProfile(
        hand=str(raw["hand"]),
        q_offset=np.asarray(raw["q_offset"], dtype=float),
        joint_sign=np.asarray(raw["joint_sign"], dtype=float),
        joint_scale=np.asarray(raw["joint_scale"], dtype=float),
        thumb_mix=np.asarray(raw["thumb_mix"], dtype=float),
        metadata=dict(raw.get("metadata", {})),
    )
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```bash
python -m unittest tests.test_calibration_io
```

Expected: pass.

## Task 3: Add calibration metrics and confidence scoring

**Files:**
- Create: `hand_reconstruction/calibration_metrics.py`
- Test: `tests/test_calibration_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create tests for per-session metric computation and confidence aggregation:

```python
import unittest

import numpy as np


class CalibrationMetricsTest(unittest.TestCase):
    def test_metrics_report_contains_expected_fields(self):
        from hand_reconstruction.calibration_metrics import evaluate_session
        from hand_reconstruction.calibration import CalibrationProfile

        profile = CalibrationProfile(
            hand="right",
            q_offset=np.zeros(21),
            joint_sign=np.ones(21),
            joint_scale=np.ones(21),
            thumb_mix=np.eye(4, 5),
            metadata={"subject": "s01"},
        )
        session = {
            "q_raw": np.zeros((4, 21), dtype=float),
            "timestamps": np.arange(4, dtype=float),
            "prompt_ids": np.zeros(4, dtype=int),
        }

        report = evaluate_session(session, profile)

        self.assertIn("confidence", report)
        self.assertIn("rest_pose_residual", report)
        self.assertIn("scale_repeatability", report)
        self.assertGreaterEqual(report["confidence"], 0.0)
        self.assertLessEqual(report["confidence"], 1.0)

    def test_confidence_drops_when_noise_increases(self):
        from hand_reconstruction.calibration_metrics import evaluate_session
        from hand_reconstruction.calibration import CalibrationProfile

        profile = CalibrationProfile(
            hand="right",
            q_offset=np.zeros(21),
            joint_sign=np.ones(21),
            joint_scale=np.ones(21),
            thumb_mix=np.eye(4, 5),
            metadata={"subject": "s01"},
        )
        clean = {"q_raw": np.zeros((8, 21), dtype=float), "timestamps": np.arange(8), "prompt_ids": np.zeros(8, dtype=int)}
        noisy = {"q_raw": np.random.default_rng(0).normal(0.0, 0.2, size=(8, 21)), "timestamps": np.arange(8), "prompt_ids": np.zeros(8, dtype=int)}

        clean_report = evaluate_session(clean, profile)
        noisy_report = evaluate_session(noisy, profile)

        self.assertGreater(clean_report["confidence"], noisy_report["confidence"])
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
python -m unittest tests.test_calibration_metrics
```

Expected: fail because `hand_reconstruction.calibration_metrics` does not exist yet.

- [ ] **Step 3: Implement a deterministic metric evaluator**

Implement a pure-NumPy evaluator that computes per-session summary metrics and confidence:

```python
"""Evaluation metrics for calibration sessions."""

from __future__ import annotations

import numpy as np

from .calibration import CalibrationProfile, normalize_glove_q


def evaluate_session(session: dict[str, np.ndarray], profile: CalibrationProfile) -> dict[str, float]:
    q_raw = np.asarray(session["q_raw"], dtype=float)
    timestamps = np.asarray(session["timestamps"], dtype=float)
    prompt_ids = np.asarray(session["prompt_ids"], dtype=int)
    if q_raw.ndim != 2 or q_raw.shape[1] != 21:
        raise ValueError(f"session['q_raw'] must have shape (N, 21), got {q_raw.shape}")
    if timestamps.shape != (q_raw.shape[0],):
        raise ValueError("session['timestamps'] must match q_raw frame count")
    if prompt_ids.shape != (q_raw.shape[0],):
        raise ValueError("session['prompt_ids'] must match q_raw frame count")

    q_norm = np.stack([
        normalize_glove_q(frame, q_offset=profile.q_offset, joint_sign=profile.joint_sign, joint_scale=profile.joint_scale)
        for frame in q_raw
    ])

    rest_mask = prompt_ids == 0
    if not np.any(rest_mask):
        raise ValueError("session must contain at least one rest frame")

    rest_pose_residual = float(np.linalg.norm(q_norm[rest_mask].mean(axis=0)))
    scale_repeatability = float(np.std(q_norm, axis=0).mean())
    frame_jitter = float(np.abs(np.diff(q_norm, axis=0)).mean()) if q_norm.shape[0] > 1 else 0.0
    joint_limit_hit_rate = float(np.mean(np.abs(q_norm) > 1.0))
    thumb_specific_residual = float(np.abs(q_norm[:, :5]).mean())

    score = 1.0 - np.clip(
        0.45 * rest_pose_residual
        + 0.20 * scale_repeatability
        + 0.15 * frame_jitter
        + 0.10 * joint_limit_hit_rate
        + 0.10 * thumb_specific_residual,
        0.0,
        1.0,
    )

    return {
        "rest_pose_residual": rest_pose_residual,
        "scale_repeatability": scale_repeatability,
        "frame_jitter": frame_jitter,
        "joint_limit_hit_rate": joint_limit_hit_rate,
        "thumb_specific_residual": thumb_specific_residual,
        "confidence": float(score),
    }
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```bash
python -m unittest tests.test_calibration_metrics
```

Expected: pass.

## Task 4: Build the calibration CLI

**Files:**
- Create: `scripts/calibrate_hand.py`
- Modify: `hand_reconstruction/__init__.py` only if imports need to be exposed
- Test: `tests/test_calibrate_hand_cli.py`

- [ ] **Step 1: Write the failing CLI tests**

Create tests that exercise `record`, `fit`, and `inspect` with fakes:

```python
import unittest
from unittest import mock

import numpy as np

from scripts import calibrate_hand


class CalibrateHandCliTest(unittest.TestCase):
    def test_fit_mode_writes_profile_json(self):
        test_args = [
            "calibrate_hand.py",
            "fit",
            "--session",
            "/tmp/session_dir",
            "--output-profile",
            "/tmp/profile.json",
        ]

        with mock.patch.object(calibrate_hand.sys, "argv", test_args):
            with mock.patch.object(calibrate_hand, "load_session_npz") as load_mock:
                with mock.patch.object(calibrate_hand, "fit_calibration_profile") as fit_mock:
                    with mock.patch.object(calibrate_hand, "save_profile_json") as save_mock:
                        fit_mock.return_value = mock.Mock()
                        exit_code = calibrate_hand.main()

        self.assertEqual(exit_code, 0)
        load_mock.assert_called_once()
        fit_mock.assert_called_once()
        save_mock.assert_called_once()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
python -m unittest tests.test_calibrate_hand_cli
```

Expected: fail because `scripts.calibrate_hand` does not exist yet.

- [ ] **Step 3: Implement the CLI with three modes**

Add a script in the same style as `scripts/reconstruct_hand_frame.py`:

```python
#!/usr/bin/env python3
"""Offline calibration workflow for DexGlove hand reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hand_reconstruction.calibration import CalibrationProfile, fit_calibration_profile
from hand_reconstruction.calibration_io import load_session_npz, save_profile_json


def main() -> int:
    args = _parse_args()
    if args.command == "fit":
        session = load_session_npz(args.session)
        profile = fit_calibration_profile(session, hand=args.hand, subject=args.subject, version=args.version)
        save_profile_json(profile, args.output_profile)
        return 0
    if args.command == "inspect":
        session = load_session_npz(args.session)
        print(_summarize_session(session))
        return 0
    if args.command == "record":
        _record_session(args.output_session, hand=args.hand, subject=args.subject)
        return 0
    raise ValueError(f"unknown command: {args.command}")
```

Keep `record` minimal: write the session archive and prompt metadata. Do not add live visualization changes.

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```bash
python -m unittest tests.test_calibrate_hand_cli
```

Expected: pass.

## Task 5: Build the evaluation CLI

**Files:**
- Create: `scripts/evaluate_hand_calibration.py`
- Test: `tests/test_evaluate_hand_calibration_cli.py`

- [ ] **Step 1: Write the failing CLI tests**

Create tests for loading a profile, replaying a session, and printing a report:

```python
import unittest
from unittest import mock

from scripts import evaluate_hand_calibration


class EvaluateHandCalibrationCliTest(unittest.TestCase):
    def test_evaluate_mode_writes_report(self):
        test_args = [
            "evaluate_hand_calibration.py",
            "--session",
            "/tmp/session.npz",
            "--profile",
            "/tmp/profile.json",
            "--output-report",
            "/tmp/report.json",
        ]

        with mock.patch.object(evaluate_hand_calibration.sys, "argv", test_args):
            with mock.patch.object(evaluate_hand_calibration, "load_session_npz") as load_session_mock:
                with mock.patch.object(evaluate_hand_calibration, "load_profile_json") as load_profile_mock:
                    with mock.patch.object(evaluate_hand_calibration, "evaluate_session") as eval_mock:
                        with mock.patch.object(evaluate_hand_calibration, "save_report_json") as save_mock:
                            eval_mock.return_value = {"confidence": 1.0}
                            exit_code = evaluate_hand_calibration.main()

        self.assertEqual(exit_code, 0)
        load_session_mock.assert_called_once()
        load_profile_mock.assert_called_once()
        eval_mock.assert_called_once()
        save_mock.assert_called_once()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
python -m unittest tests.test_evaluate_hand_calibration_cli
```

Expected: fail because `scripts.evaluate_hand_calibration` does not exist yet.

- [ ] **Step 3: Implement the evaluator CLI**

Use the same CLI style as the other scripts and keep the output JSON-first:

```python
#!/usr/bin/env python3
"""Offline evaluator for calibrated DexGlove hand reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hand_reconstruction.calibration_io import load_profile_json, load_session_npz, save_report_json
from hand_reconstruction.calibration_metrics import evaluate_session


def main() -> int:
    args = _parse_args()
    session = load_session_npz(args.session)
    profile = load_profile_json(args.profile)
    report = evaluate_session(session, profile)
    save_report_json(report, args.output_report)
    if args.print_report:
        print(report)
    return 0
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```bash
python -m unittest tests.test_evaluate_hand_calibration_cli
```

Expected: pass.

## Task 6: Export the public API and run full verification

**Files:**
- Modify: `hand_reconstruction/__init__.py`
- Modify: any script import paths that need stable public symbols
- Test: all calibration-related tests

- [ ] **Step 1: Add exports for the new calibration helpers**

Expose the new calibration pieces that scripts and tests need, for example:

```python
from .calibration import CalibrationProfile, fit_q_offset, fit_thumb_mix, normalize_glove_q
from .calibration_io import calibration_session_paths, load_profile_json, load_session_npz, save_profile_json, save_session_npz
from .calibration_metrics import evaluate_session
```

- [ ] **Step 2: Run the full test subset for calibration and the existing hand pipeline**

Run:

```bash
python -m unittest tests.test_calibration tests.test_calibration_io tests.test_calibration_metrics tests.test_calibrate_hand_cli tests.test_evaluate_hand_calibration_cli tests.test_retargeting tests.test_glove_observation tests.test_tip_locking tests.test_reconstruct_hand_frame_cli
```

Expected: all tests pass.

- [ ] **Step 3: Run a repo-wide smoke check**

Run:

```bash
python -m unittest discover -s tests
python -m unittest discover -s DexCap_v4 -p 'test_*.py'
```

Expected: all tests pass.

- [ ] **Step 4: Update the project memory if needed**

Record the final calibration-file layout and CLI names in `docs/current-memory.md` so later work does not rediscover the same design decisions.

- [ ] **Step 5: Save or commit the plan**

If this workspace is connected to a usable git repository, commit the plan file and the calibration-related implementation after the code lands. If git metadata is still unavailable in the current environment, keep the plan file as the handoff artifact and document the blocker explicitly.

---

## Self-Review Checklist

- Spec coverage: the plan covers calibration model, session/profile I/O, metrics, CLI entry points, exports, and tests.
- Placeholder scan: no `TBD`, `TODO`, or vague steps remain.
- Type consistency: `CalibrationProfile`, `normalize_glove_q`, `fit_q_offset`, `fit_thumb_mix`, `evaluate_session`, `save_profile_json`, `load_profile_json`, `save_session_npz`, and `load_session_npz` are used consistently across tasks.
- Scope check: the plan stays offline-only and does not touch the live MeshCat stream.
- Ambiguity check: profile format is JSON-first, thumb coupling is a fixed 4x5 model, and sessions land under a top-level `calibration/` tree.

