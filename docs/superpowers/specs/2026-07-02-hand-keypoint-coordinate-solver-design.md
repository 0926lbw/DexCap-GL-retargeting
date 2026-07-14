# Hand Keypoint Coordinate Solver Design

## Goal

Improve the DexGlove-to-human-hand reconstruction after the current skeleton
overlay position has become acceptable. The next iteration should improve
keypoint quality, temporal stability, and coordinate-frame clarity without
switching to robot-hand retargeting or a physics simulator.

The reconstruction should keep two representations at the same time:

- wrist-local human-hand data for algorithms and anatomical constraints,
- world-frame data for visualization, export, and comparison with glove FK.

## Scope

In scope:

- a clear solver-layer output object for one reconstructed hand frame,
- consistent naming and validation for wrist-local and world-frame keypoints,
- fixed-length fingertip targeting performed in the wrist-local frame,
- joint-angle smoothing and velocity limiting,
- diagnostics for coordinate-frame and keypoint quality checks,
- integration with the existing MeshCat human skeleton overlay.

Out of scope for this pass:

- robot hand control,
- `dex-retargeting` integration,
- MANO fitting,
- MuJoCo or Isaac Sim backends,
- contact, dynamics, or grasp physics,
- per-frame bone-length fitting.

## Coordinate Frames

Use explicit frame names in data structures and function names.

- `W`: world or glove display frame used by MeshCat and direct DexGlove FK.
- `Hwrist`: canonical human wrist-local frame used by the human kinematic model.
- `D`: display-only offset layer. Display offsets must not alter exported
  reconstruction data.

The canonical data flow is:

```text
human keypoints in Hwrist
  -- T_W_Hwrist -->
human keypoints in W
  -- display offset only in MeshCat -->
displayed skeleton
```

Direct DexGlove observations arrive in `W`. Before fusion, direct glove
landmarks are transformed into `Hwrist` with `T_Hwrist_W = inv(T_W_Hwrist)`.
Fixed-length fingertip targeting then happens in `Hwrist`, and the fused
skeleton is transformed back to `W`.

## Output Model

Add a frame result object with explicit local and world arrays:

```text
HandReconstructionFrame
  hand: "left" | "right"
  joint_angles: dict[str, float]
  keypoints_21_in_Hwrist: np.ndarray
  keypoints_21_in_W: np.ndarray
  direct_glove_keypoints_21_in_W: np.ndarray
  direct_glove_keypoints_21_in_Hwrist: np.ndarray
  fused_keypoints_21_in_Hwrist: np.ndarray
  fused_keypoints_21_in_W: np.ndarray
  T_W_Hwrist: np.ndarray
  diagnostics: dict[str, float | bool | str]
```

The solver should treat `keypoints_21_in_Hwrist` as the primary algorithmic
representation. `keypoints_21_in_W` and `fused_keypoints_21_in_W` are derived
outputs for display, export, and external consumers.

## Solver Layer

Add a focused solver module, likely `hand_reconstruction/solver.py`, that
coordinates existing building blocks without moving their responsibilities:

- `GloveToHumanRetargeter` keeps the raw glove-q to human-joint mapping and
  anatomical angle rules.
- `HumanHandModel` keeps human-hand FK and fixed bone lengths.
- `HandReconstructionPipeline.reconstruct_direct()` keeps direct DexGlove FK
  observations.
- `fuse_tip_locked_landmarks()` keeps fixed human bone lengths while bending
  each finger chain toward the direct glove fingertip target.
- The solver owns frame transforms, smoothing state, and the final result
  object.

The per-frame flow is:

```text
q_glove
  -> retargeter.retarget(q_glove)
  -> optional joint smoothing / velocity limiting
  -> human_model.landmarks_from_joints(joint_angles)        # Hwrist
  -> direct glove FK                                        # W
  -> transform direct glove landmarks W -> Hwrist
  -> solve fixed-length human chains toward direct glove tips in Hwrist
  -> transform fused landmarks Hwrist -> W
  -> return HandReconstructionFrame
```

## Keypoint Quality Rules

Keep improving keypoints through the kinematic model rather than by directly
dragging every `xyz` point.

- Bone lengths remain fixed within a hand model.
- Non-thumb DIP remains coupled from PIP unless live testing proves a better
  ratio.
- MCP abduction continues to damp as MCP/PIP flexion increases.
- Thumb remains a separate model with CMC abduction and opposition/flexion.
- Fingertips target direct glove tips in the fused output, but unreachable
  targets leave residual error rather than stretching human bones.
- Display offsets are applied only after all reconstruction outputs are built.

## Temporal Stability

Smooth joint angles, not landmark positions.

The first version should use a small stateful smoother:

- exponential moving average per joint,
- optional per-frame max delta or velocity limit,
- reset behavior when hand side, model, or stream session changes,
- no smoothing for display offsets.

Smoothing should be optional and parameterized so live tuning can compare raw
and smoothed outputs.

## Diagnostics

Each frame should expose lightweight diagnostics that help catch coordinate and
quality problems during live tuning:

- fingertip target error before and after fusion,
- determinant of the wrist transform rotation,
- whether any joint hit a limit,
- max joint delta after smoothing,
- local/world round-trip error,
- whether all exported arrays are finite and shape `(21, 3)`.

Diagnostics should be cheap enough to compute every frame and safe to ignore by
callers that only need landmarks.

## MeshCat Integration

`DexCap_v4/dexcap_glove_meshcat_stream.py` should consume the solver result
instead of open-coding the same steps in `display_human()`.

The display layer should:

- use `fused_keypoints_21_in_Hwrist` or `fused_keypoints_21_in_W` from the
  solver result,
- keep existing whole-skeleton and thumb-only display offsets,
- apply those offsets only to MeshCat display landmarks,
- keep the current CLI options for display tuning.

This preserves the accepted visual alignment while separating real
reconstruction data from display-only corrections.

## Export Behavior

Exports should include both local and world keypoints:

- `keypoints_21_in_Hwrist`
- `keypoints_21_in_W`
- `fused_keypoints_21_in_Hwrist`
- `fused_keypoints_21_in_W`
- `T_W_Hwrist`
- coordinate convention metadata

If only one array is needed by a downstream consumer, the wrist-local fused
array should be treated as the algorithmic default.

## Testing

Add focused tests that do not require live hardware:

- shape and finite-value validation for `HandReconstructionFrame`,
- world-to-local-to-world round-trip consistency,
- fixed-length fingertip targeting performed in local coordinates,
- display offsets do not mutate solver outputs,
- smoothing reduces step changes while respecting joint limits,
- left/right transform handling keeps thumb mirror behavior explicit,
- export includes both local and world keypoint arrays.

Pinocchio-dependent tests should keep the existing pattern of importing heavy
dependencies only when the Pinocchio-backed objects are instantiated.

## Rollout

Implement in small steps:

1. Add the result object and pure transform helpers.
2. Add a solver that reproduces the current unsmoothed overlay behavior.
3. Switch MeshCat display to consume solver output.
4. Add optional joint smoothing and diagnostics.
5. Extend export helpers to include fused local/world data.

The first solver version should be behavior-preserving except for making frames
explicit. Visual improvements should come after the coordinate pipeline is
covered by tests.
