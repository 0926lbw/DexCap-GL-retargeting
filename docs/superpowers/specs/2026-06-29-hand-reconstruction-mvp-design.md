# Hand Reconstruction MVP Design

## Goal

Build a first-pass human hand reconstruction layer for DexCap glove data without changing the existing socket listener or `DexCap.py` data flow.

## Scope

This MVP converts per-hand DexGlove joint arrays into a 21-point human hand skeleton format. It treats the glove URDF as an observation source and produces a stable intermediate representation for later calibration, optimization, MANO fitting, and visualization.

Out of scope for this pass:

- Contact constraints
- Dynamics
- MANO fitting
- Learned pose regression
- Rewriting the existing socket receiver

## Architecture

The new code lives under `hand_reconstruction/` and stays independent from `DexCap.py`.

- `schema.py` defines hand landmark indices, finger chains, and lightweight data containers.
- `human_hand_model.py` provides a calibrated 21-point skeleton model and utilities for building skeletons from observed link positions.
- `glove_observation.py` loads DexGlove URDF models with Pinocchio when available and extracts link-frame observations from `q_l` or `q_r`.
- `pipeline.py` combines glove observations and the human skeleton model into one reconstruction call.
- `export.py` writes reconstructed skeleton frames to JSON or NPY.
- `visualize_meshcat.py` optionally displays the reconstructed skeleton and glove tips in MeshCat.
- `scripts/reconstruct_hand_frame.py` is a small CLI for offline arrays.

## Data Flow

```text
q_l / q_r, shape (21,)
  -> DexGlove URDF forward kinematics
  -> selected glove link positions
  -> 21-point human skeleton estimate
  -> JSON / NPY / optional MeshCat display
```

The first implementation uses a deterministic geometric initialization:

- wrist comes from the glove base frame.
- thumb uses five observed glove links.
- index, middle, ring, and pinky use four observed glove links plus the wrist.

This is intentionally simple. It gives a working intermediate format before adding user-specific optimization.

## Testing

Tests cover the pure-Python schema and skeleton behavior without requiring Pinocchio. Pinocchio-dependent code is structured so import errors are raised only when the URDF observation class is instantiated.
