# Hand Reconstruction MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent first-pass module that converts DexGlove joint arrays into a 21-point human hand skeleton.

**Architecture:** Keep new code under `hand_reconstruction/` and leave `DexCap.py` unchanged. Use pure NumPy for the human skeleton schema/model and optional Pinocchio for URDF observations.

**Tech Stack:** Python, NumPy, optional Pinocchio, unittest/pytest-compatible tests.

---

### Task 1: Schema And Skeleton Tests

**Files:**
- Create: `tests/test_hand_schema.py`
- Create: `hand_reconstruction/schema.py`
- Create: `hand_reconstruction/human_hand_model.py`

- [ ] Write tests for the 21-point landmark schema, finger chains, and default skeleton shape.
- [ ] Run `python3 -m unittest tests.test_hand_schema` and verify it fails because modules do not exist.
- [ ] Implement schema constants and a simple `HumanHandSkeleton`.
- [ ] Re-run `python3 -m unittest tests.test_hand_schema` and verify it passes.

### Task 2: Glove Observation Tests

**Files:**
- Create: `tests/test_glove_observation.py`
- Create: `hand_reconstruction/glove_observation.py`

- [ ] Write tests for deterministic link-name selection and observation-to-skeleton conversion using fake link positions.
- [ ] Run `python3 -m unittest tests.test_glove_observation` and verify it fails because functions do not exist.
- [ ] Implement finger link-name helpers and `SkeletonInitializer`.
- [ ] Re-run the test and verify it passes.

### Task 3: Pipeline And Export

**Files:**
- Create: `tests/test_pipeline_export.py`
- Create: `hand_reconstruction/pipeline.py`
- Create: `hand_reconstruction/export.py`
- Create: `hand_reconstruction/__init__.py`

- [ ] Write tests for a pure observation pipeline and JSON/NPY export.
- [ ] Run `python3 -m unittest tests.test_pipeline_export` and verify it fails.
- [ ] Implement pipeline and export helpers.
- [ ] Re-run the test and verify it passes.

### Task 4: CLI And Optional Visualization

**Files:**
- Create: `scripts/reconstruct_hand_frame.py`
- Create: `hand_reconstruction/visualize_meshcat.py`

- [ ] Add CLI that accepts `--hand`, `--q`, and `--output-json`.
- [ ] Add optional MeshCat renderer that imports MeshCat lazily.
- [ ] Run `python3 -m py_compile` on all new Python files.

### Task 5: Full Verification

**Files:**
- All new files

- [ ] Run `python3 -m unittest discover -s tests -v`.
- [ ] Run `python3 -m py_compile hand_reconstruction/*.py scripts/reconstruct_hand_frame.py`.
- [ ] Report any dependency limitations, especially Pinocchio availability.
