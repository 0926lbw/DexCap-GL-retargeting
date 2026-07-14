# Tip-Locked Hand Skeleton Design

## Goal

Improve the live human-hand skeleton overlay so each fingertip stays exactly aligned with the corresponding DexGlove exoskeleton fingertip, while the intermediate finger joints remain human-looking instead of copying the full mechanical linkage.

The scope is the 21-point skeleton shown by `DexCap_v4/dexcap_glove_meshcat_stream.py`. This does not change the DexCap TCP listener, raw glove q parsing, URDFs, or robot hand command mapping.

## Current Behavior

The MeshCat stream currently shows two layers:

- DexGlove URDF models, driven directly by the 21 glove joint vector.
- An optional in-palm human skeleton, driven by `GloveToHumanRetargeter` and `HumanHandModel.landmarks_from_joints()`.

At startup, the human hand rest pose is rigid-fit to the DexGlove link layout. During streaming, however, the overlay follows human-hand FK only. This keeps the skeleton anatomical, but the five fingertip landmarks can drift away from the actual exoskeleton fingertip link positions.

The codebase also has `HandReconstructionPipeline.reconstruct_direct(q)`, which maps all 21 skeleton landmarks directly to DexGlove link origins. That path has exact exoskeleton alignment but looks too mechanical for the desired overlay.

## Chosen Approach

Use a tip-locked fusion layer:

1. Compute the glove/direct skeleton for the current q with `reconstruct_direct(q)`.
2. Compute the human FK skeleton for the same q with `GloveToHumanRetargeter` and `HumanHandModel`.
3. For each finger, force the fingertip landmark to equal the corresponding direct glove fingertip landmark.
4. Reposition the intermediate landmarks from the human FK pose, preserving the human bend profile along a chain that ends at the locked glove fingertip.

This is a hard 3D position lock for the five fingertips only. Fingertip orientation is intentionally out of scope for this iteration because the current overlay is a 21-point skeleton, not an oriented end-effector frame display.

## Fingertip Mapping

The lock uses the existing schema in `hand_reconstruction/schema.py`:

- `THUMB_TIP` maps to `glove_link_{side}_1_5`.
- `INDEX_TIP` maps to `glove_link_{side}_2_4`.
- `MIDDLE_TIP` maps to `glove_link_{side}_3_4`.
- `RING_TIP` maps to `glove_link_{side}_4_4`.
- `PINKY_TIP` maps to `glove_link_{side}_5_4`.

The direct skeleton already applies this mapping, so the fusion layer should consume direct skeleton landmarks by schema index rather than duplicating link-name logic.

## Fusion Rule

For each finger chain from `FINGER_CHAINS`:

- Keep the wrist and finger root behavior anchored to the human FK skeleton, after the same global MeshCat alignment transform already used by the overlay.
- Set the chain tip to the direct glove skeleton tip.
- Place MCP/PIP/DIP intermediate points by using the human FK chain's cumulative bone-length fractions and bend offsets, then scale/rotate those offsets into the segment from finger root to locked fingertip.

The implementation should be deterministic and lightweight enough to run every display frame. It should avoid iterative IK in the first version. If a degenerate chain is encountered, such as near-zero root-to-tip length, fall back to linear interpolation between root and locked tip for that finger.

## Integration

Add a small module under `hand_reconstruction/`, for example `tip_locking.py`, with a focused function or class that accepts:

- human FK landmarks, shape `(21, 3)`;
- direct glove landmarks, shape `(21, 3)`.

It returns fused landmarks, shape `(21, 3)`.

The first implementation should not expose per-finger tuning weights. Keeping the fusion rule fixed makes the behavior easier to test and compare against the current overlay.

Update `DexCap_v4/dexcap_glove_meshcat_stream.py` so `display_human()`:

1. Observes or reconstructs the direct glove landmarks for each hand's current q.
2. Builds the current human FK landmarks from the retargeter.
3. Applies the existing rigid alignment to put human FK landmarks in the same local overlay frame as the glove model.
4. Runs the tip-lock fusion.
5. Sends fused landmarks to `MeshcatSkeletonOverlay.update()`.

The optional overlay remains non-fatal. If the tip-locking pipeline cannot be built because Pinocchio or hand reconstruction modules are missing, streaming still displays the glove URDFs.

## Error Handling

Validate landmark array shapes and finite numeric values before fusing.

For malformed inputs, raise clear `ValueError`s from the fusion module. In the live MeshCat stream, catch unexpected overlay exceptions at the existing optional-overlay boundary and keep the glove display alive.

Degenerate per-finger geometry should not abort the frame. The fusion function should fall back to simple root-to-tip interpolation for that finger.

## Testing

Add focused tests for the fusion module:

- The five output fingertip landmarks exactly equal the direct glove skeleton fingertips.
- Intermediate landmarks are not copied wholesale from the direct glove skeleton when human FK differs.
- Output shape is `(21, 3)` and all values are finite.
- Degenerate root-to-tip input falls back to finite interpolation.
- Left and right hands use the same schema indices without hand-specific special cases.

Add or update stream-level tests with fakes where practical, so `display_human()` calls the direct reconstruction and overlay update path without requiring a real MeshCat browser.

## Non-Goals

- No fingertip orientation lock in this iteration.
- No MANO fitting, calibration UI, or offline optimization.
- No changes to DexCap packet parsing, `DexCap.py`, or robot hand control outputs.
- No URDF geometry edits.
