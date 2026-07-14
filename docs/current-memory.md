# Current Project Memory

Last updated: 2026-07-03

## User Preferences

- Respond in Chinese.
- Work efficiently and finish implementation before review; avoid per-task sub-agent review loops unless explicitly requested.
- The user is comparing output quality closely and expects direct execution, not long process overhead.

## Current Goal Context

The current visual skeleton direction is basically accepted. The skeleton should look like a human-hand skeleton driven by the glove, not like a copy of the exoskeleton linkage.

Confirmed reconstruction/display strategy:

- Keep two distinct keypoint surfaces:
  - `keypoints_21_in_Hwrist/W`: fixed-bone human FK skeleton. This is the
    reconstruction/data/export surface.
  - `fused_keypoints_21_in_Hwrist/W`: fixed-bone, tip-targeted human skeleton.
    This is for MeshCat display and fingertip residual debugging. It bends the
    human chain toward glove tips, but must not stretch bones to force an exact
    target match.
- Do not optimize by moving display points only, but also do not force one point
  surface to satisfy incompatible goals.
- Both `keypoints_21_in_*` and `fused_keypoints_*` must preserve human FK bone
  lengths. Use `fused_keypoints_*` to inspect the current displayed/targeted
  skeleton and fingertip residuals.
- Show the whole human skeleton slightly in front of the glove by translating all 21 skeleton landmarks together.
- Apply an extra thumb-only display correction because the glove thumb direction differs from the other four fingers:
  - left thumb moves toward local `+X`,
  - right thumb mirrors that toward local `-X`,
  - both thumbs move slightly backward with local `-Z`.
- Left thumb currently has an additional small inward correction on top of the mirrored thumb offset:
  - `DEFAULT_LEFT_THUMB_EXTRA_DISPLAY_OFFSET = (0.005, 0.0, 0.0)`.
- Preserve human FK bone lengths in `keypoints_21_in_*`.
- Preserve human FK bone lengths in `fused_keypoints_*`; fingertip mismatch is
  allowed when a direct glove target is outside the human finger's reachable
  workspace.
- Middle joints MCP/PIP/DIP can be adjusted in the human kinematic model, while
  the fused surface uses fixed-length chain bending for visualization.

## Current Implementation State

Main visualization file:

- `DexCap_v4/dexcap_glove_meshcat_stream.py`

MuJoCo first-pass model:

- `hand_reconstruction/mujoco_export.py`
- `scripts/export_mujoco_four_finger_hand.py`
- `mujoco/human_four_finger_hand.xml`
- `environment-dexcap_re.yml`
- Scope is intentionally four-finger only: index, middle, ring, pinky.
- The generated MJCF reuses `human_hand_params.py` for MCP attachments,
  per-finger yaw, phalanx lengths, flexion axis, abduction axis, and limits.
- Thumb, contact dynamics, tendons, and external objects are out of this first
  pass.
- DIP is modeled as passive by MuJoCo equality coupling:
  `DIP = 0.6 * PIP`.
- Active actuators are generated for `mcp_abd`, `mcp_flex`, and `pip`; no
  actuator is generated for `dip`.
- First-pass geoms are transparent capsules with `contype="0"` and
  `conaffinity="0"` so the model is kinematic/visual before contact tuning.

Full 21-point hand model (adds the thumb on top of the four-finger model):

- `hand_reconstruction/mujoco_export.py` also exposes
  `build_full_hand_mjcf(params=None, hand="right")`; the four-finger builder and
  the thumb share `_scaffold_mjcf` / `_add_four_fingers` helpers, and the thumb
  is added by `_add_thumb`.
- `scripts/export_mujoco_full_hand.py`
- `mujoco/human_full_hand.xml` (model name `human_full_hand`)
- `scripts/view_mujoco_full_hand.py`
- The full model exposes all 21 MediaPipe landmark sites (`wrist`,
  `thumb_cmc/mcp/ip/tip`, and the four-finger `mcp/pip/dip/tip` sites).
- The thumb is joints-only: 4 hinge DOFs (`thumb_cmc_abd`, `thumb_cmc_flex`,
  `thumb_mcp`, `thumb_ip`) with the CMC frame tilted by `ThumbParams.cmc_rpy`
  for opposition, plus keypoint sites and transparent capsules. It has NO
  position actuators and NO equality coupling yet; pose it directly via `qpos`.
- The four non-thumb fingers keep their actuators and DIP/PIP coupling exactly
  as in the four-finger model, so the full model is a strict superset
  (verified: nq=20, nu=12, nsite=21).
- `scripts/view_mujoco_full_hand.py` defaults to a neutral **open-hand** rest
  pose (so the skeleton reads as a hand on launch); add `--grasp` for a natural
  pinch demo. The thumb opposition direction is non-obvious in this model:
  positive `cmc_abd`/`cmc_flex` point the thumb laterally/upward (away from the
  fingers). The tuned `--grasp` thumb angles are `cmc_abd=-25, cmc_flex=40,
  mcp=30, ip=20` (negative abd adducts medially, flex curls forward into the
  palm), which brings the thumb tip onto the curled index/middle fingertips.
  The thumb is still qpos-driven (no actuators), so these are display angles.

Realistic solid-sculpt hand (added on top of the kinematic model):

- `build_four_finger_mjcf` / `build_full_hand_mjcf` now take
  `realistic: bool = True, keep_baseline: bool = False`. `realistic=True` emits a
  solid sculpted hand on viewer group 2; `realistic=False` emits the transparent
  kinematic baseline (group 1, byte-identical to the original first-pass model);
  `keep_baseline=True` emits both layers. Kinematics are identical across modes.
- The sculpt is parametric primitives only (no external mesh; no human-hand mesh
  exists in the repo). Per-finger radii come from `_finger_radii(Lp)` /
  `_thumb_radii(Ltp)` with a `clamp(k*L, lo, hi)` wrapper; each phalanx is two
  stacked tapered capsules (+ `TAPER_OVERLAP` to hide the seam); joints get
  knuckle spheres; fingertips get a bulb sphere + palmar pad + dorsal nail
  ellipsoid; the palm is a plate ellipsoid + thenar/hypothenar bulges + heel
  capsule + dorsal MCP knuckle spheres (positions derived from params so they
  mirror under `for_hand('left')`). All realistic geoms go through
  `_realistic_geom` which sets type/size/rgba/group/contype/conaffinity
  EXPLICITLY so the transparent `<default>` capsule cannot leak in.
- Tuned palm values (after a render+vision iteration): palm-plate Z semi-axis
  0.016 (~32 mm thick back-of-hand), dorsal MCP knuckle spheres at z=0.014 with
  radius `1.4 * mcp_knuckle` so they poke above the dorsal surface and read as
  a knuckle ridge.
- `scripts/export_mujoco_full_hand.py` writes the realistic sculpt by default
  (`--kinematic` for the baseline). `scripts/export_mujoco_four_finger_hand.py`
  intentionally still writes the transparent baseline (`realistic=False`) so the
  four-finger artifact stays the stable first-pass reference.
- Rendered realism ~7.5/10 for a primitive-built hand (open pose: fleshy palm,
  tapered fingers, knuckle ridge, opposed thumb; pinch pose: thumb tip meets
  curled index/middle fingertips, no clipping).
- Conda env `dexcap_re` was created with Python 3.11. Installing additional
  packages into it was blocked by the approval system; use
  `environment-dexcap_re.yml` to finish dependency installation when available.

Important current behavior:

- `DEFAULT_HUMAN_SKELETON_DISPLAY_OFFSET = (0.0, 0.0, 0.025)`
- `DEFAULT_THUMB_DISPLAY_OFFSET = (0.015, 0.0, -0.015)`
- `DEFAULT_LEFT_THUMB_EXTRA_DISPLAY_OFFSET = (0.005, 0.0, 0.0)`
- CLI option: `--human-skeleton-display-offset X Y Z`
- CLI option: `--thumb-display-offset X Y Z`; X is the left-hand value and is mirrored for the right hand.
- CLI option: `--left-thumb-extra-display-offset X Y Z`; affects only the left thumb after the base thumb offset.
- CLI option: `--frames`; shows per-joint glove coordinate frames for debugging.
  Frames are hidden by default to reduce MeshCat websocket load and live latency.
- CLI option: `--debug-human-joints`; prints per-finger q blocks, retargeted joint angles, local keypoint chord lengths, and tip-target diagnostics.
- CLI option: `--debug-human-joints-interval N`; prints the human joint diagnostics every N displayed human frames.
- TCP streaming now drains already-buffered complete packets before rendering, so
  a slow MeshCat update displays the newest available glove frame instead of
  replaying stale packets and accumulating visual delay.
- `_offset_landmarks_for_display(landmarks, offset)` moves all 21 landmarks together.
- `_offset_thumb_for_display(landmarks, offset, hand, left_extra_offset=...)` moves only landmarks `1..4`, mirrors X for the right hand, and applies the extra offset only to the left hand.
- `GloveMeshcatDisplay.display_human()`:
  - asks each per-hand `HandReconstructionSolver` for a solver frame,
  - uses `fused_keypoints_21_in_Hwrist` for the MeshCat human-hand node,
  - applies whole-skeleton display offset,
  - applies the thumb-only mirrored display correction,
  - updates the MeshCat skeleton overlay.

Solver implementation:

- `hand_reconstruction/solver.py`
- `HandReconstructionFrame` stores local/world raw and fused keypoints.
- `HandReconstructionSolver` performs retargeting, human FK, direct glove FK,
  local-frame fixed-length tip-target fusion, world transform output, and
  diagnostics.
- `JointAngleSmoother` supports optional per-joint EMA and max-delta limiting.
- `DexCap_v4/dexcap_glove_meshcat_stream.py` consumes solver output for the
  human overlay while keeping display offsets display-only.

Tip-locking implementation:

- `hand_reconstruction/tip_locking.py`
- Public API:
  - `FINGERTIP_INDICES`
  - `fuse_tip_locked_landmarks(human_landmarks, direct_landmarks, *, eps=1e-9)`
- Behavior:
  - this module creates a fixed-bone, tip-targeted human skeleton surface,
  - fingertips move toward direct glove landmarks by bending the chain,
  - reachable targets are matched within numerical tolerance,
  - unreachable targets leave fingertip residual instead of stretching bones,
  - root/MCP-side landmarks stay on the human skeleton,
  - every finger segment keeps its human FK length,
  - direct glove PIP/DIP-side intermediate landmarks choose which side of the
    root-to-tip target direction the bend should occupy, preventing the skeleton
    from folding through the palm when possible,
  - invalid shapes, non-finite values, and invalid `eps` raise `ValueError`.

Retargeting implementation:

- `hand_reconstruction/retargeting.py`
- `GloveToHumanRetargeter` now keeps the original glove-q table as the raw observation map, then applies a first-pass anatomical post-process.
- Live debug on 2026-07-02 showed that the right hand was all zeros because the
  right glove was not connected. Do not treat that as a retargeting failure.
- The same live debug showed that the connected left glove reports non-thumb
  MCP flexion mostly as negative values during grasp, while the distal channels
  `q[8]`, `q[12]`, `q[16]`, and `q[20]` carry the strongest curl signal.
- The retargeter now has a hand-specific left table:
  - left non-thumb MCP flex channels are sign-flipped into positive human curl,
  - distal curl channels are folded into the PIP estimate with
    `DISTAL_TO_PIP_RATIO = 2/3`,
  - final non-thumb DIP is still anatomically coupled from final PIP with
    `FINGER_DIP_PIP_RATIO = 0.6`.
- The right-hand table is intentionally still the previous sign convention until
  a connected right-glove debug log confirms its live signs.
- Current anatomical rules:
  - joint limits match the default human-hand URDF limits,
  - non-thumb DIP is coupled from PIP with `FINGER_DIP_PIP_RATIO = 0.6`,
  - MCP abduction is damped as MCP/PIP flexion increases, with `MCP_ABD_MIN_DAMPING = 0.25`,
  - thumb uses its own limits and keeps MCP coupled from CMC flexion with `THUMB_MCP_FROM_CMC_FLEX_RATIO = 0.5`,
  - thumb distal glove encoders still do not create a human `thumb_dip` joint.
- Temporal smoothing now lives in optional solver-layer `JointAngleSmoother`.
- Landmark-level fixed-length per-finger IK is implemented in `tip_locking.py`;
  full joint-angle constrained IK is still not implemented.

Relevant tests:

- `tests/test_tip_locking.py`
- `tests/test_solver_frame.py`
- `DexCap_v4/test_dexcap_glove_meshcat_stream.py`
- Existing tests cover fixed-length tip targeting, solver local/world frame
  handling, solver export payloads, and whole-skeleton display offset behavior.

## Important Decisions

Do not continue optimizing by only moving displayed points. Next quality improvements should happen in the human-hand kinematic model.

Coordinate-frame decision for the next keypoint optimization pass:

- Keep both wrist-local and world-frame skeleton data.
- Use wrist-local `Hwrist` keypoints as the internal algorithmic representation.
- Keep world-frame `W` keypoints for visualization, export, and comparison with direct DexGlove FK.
- Display offsets are a separate display-only layer and must not pollute exported reconstruction data.
- Fixed-length tip-target fusion should happen in wrist-local coordinates:
  transform direct glove landmarks from `W` into `Hwrist`, solve the
  fixed-length chain there, then transform fused landmarks back to `W`.

Current design spec for this direction:

- `docs/superpowers/specs/2026-07-02-hand-keypoint-coordinate-solver-design.md`

Professional direction:

- Treat glove data as observations.
- Treat the human hand as an anatomical kinematic model.
- Use a solver/constraint layer to make MCP/PIP/DIP move like a real hand while fingertips remain aligned.

Recommended kinematic constraints:

- Fingers:
  - MCP: flexion plus limited abduction/adduction.
  - PIP: mostly hinge flexion.
  - DIP: mostly hinge flexion.
  - Couple DIP to PIP, for example `DIP = 0.5 ~ 0.7 * PIP`.
  - Reduce MCP abduction as MCP/PIP flexion increases.
- Thumb:
  - Model separately.
  - CMC needs opposition behavior, not just simple flexion.
  - Give thumb CMC at least two effective DOFs: abduction/adduction and opposition/flexion.
- Bone lengths:
  - Keep bone lengths fixed per user/calibration.
  - Do not fit bone lengths per frame.
- Temporal behavior:
  - Filter joint angles, not landmark positions.
  - Add angle velocity/acceleration limits to reduce jitter and impossible snaps.

Future IK objective idea:

```text
E(q) =
  w_tip    * fingertip position error
+ w_dir    * phalanx direction error
+ w_prior  * anatomical/rest-pose prior
+ w_smooth * previous-frame angle continuity
```

Hard/strong constraints:

- fixed bone lengths,
- joint limits,
- fingertip targets as high-weight objectives with reach limits,
- no physically impossible hyperextension unless explicitly calibrated.

## Recommended Next Implementation Step

The first low-risk kinematic improvement has been implemented:

1. Joint limits and clamped outputs in `hand_reconstruction/retargeting.py`.
2. DIP/PIP coupling for four fingers.
3. MCP abduction damping based on flexion.
4. Thumb-specific limits and CMC-to-MCP coupling.
5. Fixed-length per-finger tip-target fusion in `hand_reconstruction/tip_locking.py`.

Recommended next kinematic step:

1. Add temporal smoothing on returned joint angles.
2. Tune the anatomical constants against live glove motion.
3. Add full joint-angle constrained IK only after the rules version is visually stable.

## Simulation Platform Decision

Do not switch simulation platforms now.

Current recommendation:

- Keep current MeshCat/current viewer for fast skeleton and alignment iteration.
- Add MuJoCo later as a second backend if contact, grasping, collision, friction, or physics validation become important.
- Do not move to Isaac Sim at this stage; it is too heavy for current kinematic-model iteration.

Preferred architecture:

```text
glove data -> human kinematic solver -> 21 landmarks / joint angles
                                      -> current viewer
                                      -> optional MuJoCo backend later
```

## Visualization Entry Point

Use the current stream visualizer:

```bash
cd /home/lbw/DexCap-GL-retargeting/DexCap_v4
python dexcap_glove_meshcat_stream.py
```

Useful option:

```bash
python dexcap_glove_meshcat_stream.py --human-skeleton-display-offset 0 0 0.025
```

Dependencies must include the existing DexCap visualization stack, including MeshCat and Pinocchio.

## Cautions For Future Agents

- Do not reintroduce fingertip-only display extension; it made the final fingertip segment look too short.
- Do not make every skeleton joint directly follow the exoskeleton linkage; that makes the result look mechanical.
- Do not move both thumbs in the same X direction; left thumb uses `+X`, right thumb uses mirrored `-X`.
- Do not mirror `left_thumb_extra_display_offset`; it is intentionally left-hand-only.
- Keep fingertip consistency while improving MCP/PIP/DIP through anatomical constraints.
- The next code change should likely be in `hand_reconstruction/retargeting.py` or a new solver module, not in display-only code.
- The project root `/home/lbw/DexCap-GL-retargeting` may not be a git repository; verify before assuming git operations are available.
