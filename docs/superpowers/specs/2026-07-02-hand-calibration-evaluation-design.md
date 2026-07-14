# Hand Calibration And Evaluation Design

## Goal

Turn DexGlove exoskeleton data into a calibrated, repeatable, and measurable human-hand reconstruction input. The current live skeleton overlay is already visually acceptable, so the next step is not more display tuning. The next step is to standardize the encoder coordinate mapping, fit per-hand calibration parameters, and measure reconstruction quality with an offline evaluator.

This design focuses on the hand-reconstruction side of the pipeline, not the live MeshCat visual layer.

## Current State

The repository already has three relevant layers:

- `DexCap_v4/dexcap_glove_meshcat_stream.py` streams DexGlove q vectors into MeshCat and optionally overlays a human skeleton.
- `hand_reconstruction/retargeting.py` applies a first-pass anatomical mapping from glove q values to human joint angles.
- `hand_reconstruction/tip_locking.py` fuses human FK landmarks with direct glove fingertips so the overlay keeps fingertip positions aligned.

The current state is useful for visual inspection, but it does not yet prove that the reconstructed human-hand data are calibrated or accurate. The missing pieces are:

- a reproducible calibration workflow,
- a persistent per-hand calibration profile,
- an evaluator that can score reconstruction quality on recorded sessions.

## Scope

In scope for this design:

- zero pose capture and `q_offset` estimation,
- encoder direction and scale normalization,
- thumb-specific calibration,
- export of calibration profiles to a portable file format,
- offline replay and evaluation of recorded sessions,
- metrics for repeatability, stability, and reconstruction error.

Out of scope for this design:

- changing the live MeshCat overlay geometry,
- redesigning the tip-lock fusion layer,
- replacing the current anatomical retargeting layer with learned pose regression,
- adding dynamics or contact physics,
- MANO fitting.

The first version also keeps the current human hand geometry fixed. This design calibrates encoder mapping and thumb coupling, but it does not re-fit bone lengths or the parametric hand template.

## Recommended Architecture

Use two small offline tools plus one shared library layer:

- `hand_reconstruction/calibration.py`: calibration data structures, normalization logic, and fit helpers.
- `scripts/calibrate_hand.py`: capture or replay a calibration session, estimate per-hand calibration parameters, and write a profile file.
- `scripts/evaluate_hand_calibration.py`: replay a recorded session with a saved profile and produce a metrics report.

The online overlay can keep using the existing retargeting path until calibration profiles are wired in later. That keeps the live display stable while calibration work happens offline.

### Why This Shape

There are two other viable approaches:

1. Put calibration directly inside `retargeting.py`.
2. Build a full online auto-calibration loop.

Both are weaker here. Directly embedding calibration into `retargeting.py` mixes two concerns and makes the current visual path harder to reason about. A full online auto-calibration loop would be harder to validate and would make the first iteration brittle. A separate offline calibration pipeline is the better first step because it is easier to inspect, replay, and score.

## Calibration Model

The calibration profile should be per hand and should normalize the raw glove observation into a standard coordinate space before any anatomical reasoning is applied.

The minimal profile should contain:

- `hand`: `left` or `right`,
- `q_offset`: 21-value rest offset,
- `joint_sign`: 21-value sign vector,
- `joint_scale`: 21-value scale vector,
- `thumb_mix`: thumb-specific coupling parameters,
- `metadata`: capture date, glove side, subject label, and calibration version.

Represent the profile as JSON first. JSON is already dependency-free in the repo and matches the current export style. If human-editable YAML becomes useful later, it can be added as a compatibility layer, but it should not be a requirement for the first pass.

`thumb_mix` should be a fixed coupling model for the five thumb encoders, expressed as a 4x5 linear map or an equivalent set of named coefficients that produces the four thumb features consumed by retargeting. The stored semantics must be explicit enough that a later implementation can load the profile without guessing how the thumb channels were coupled.

### Normalization Rule

The standardized raw observation should be:

```text
q_norm[i] = joint_sign[i] * (q_raw[i] - q_offset[i]) / joint_scale[i]
```

Thumb channels may also use a small coupling model so that thumb CMC/MCP/IP behavior is not forced into the same per-channel rule as the four fingers.

The normalized vector is still an observation. It is not yet the final human-hand pose. The retargeting layer remains responsible for anatomical constraints such as joint limits, DIP/PIP coupling, and MCP abduction damping.

## Calibration Workflow

The first version should be an offline, prompt-driven workflow with a small fixed pose set.

### Session Archive

Use a per-session directory under a top-level `calibration/` folder at the repository root:

- `calibration/sessions/<session_id>/metadata.json`
- `calibration/sessions/<session_id>/frames.npz`
- `calibration/profiles/<hand>_<subject>_<version>.json`
- `calibration/reports/<session_id>.json`

The `frames.npz` archive should hold time-aligned numeric arrays such as decoded q vectors, timestamps, and any prompt labels that are easier to store as arrays than as JSON.

### Session Inputs

Each session should record:

- raw glove packets or already-decoded q vectors,
- hand side,
- timestamps or frame indices,
- optional human-readable prompt labels,
- optional tip-locked skeleton output for later inspection.

### Pose Sequence

Use a small sequence that isolates the mapping parameters:

1. Rest/open hand for zero pose.
2. Individual finger flexion for four fingers.
3. Individual MCP abduction/adduction for four fingers.
4. Thumb flexion and opposition.
5. Thumb abduction and coupled motion.

This sequence is enough to estimate zero, sign, and scale without asking the user to perform an oversized calibration routine.

### Parameter Estimation

Estimate the calibration profile from the captured segments as follows:

- `q_offset`: average rest pose over a stable open-hand segment,
- `joint_sign`: sign that makes the chosen motion positive in the normalized coordinate system,
- `joint_scale`: measured amplitude between rest and pose endpoint,
- `thumb_mix`: fit separately from thumb-only motions, because thumb axes do not behave like the other four fingers.

If the session contains repeated prompts, compute the median estimate across repeats and reject outliers with large spread.

## Evaluation Model

The evaluator should answer one question: given a recorded session and a saved profile, how trustworthy is the reconstructed hand data?

There is no external motion-capture ground truth in the current repository, so the evaluator must distinguish between two kinds of accuracy:

- **internal calibration quality**: how well the motion obeys the calibration prompts and model constraints,
- **reconstruction confidence**: how stable the solved parameters and per-frame outputs are.

That is an honest definition of accuracy for this system. It does not claim absolute physical truth without an external reference.

If a future session includes an external reference track, the evaluator may also report absolute landmark error. The first implementation does not require such a track.

### Suggested Metrics

Report these metrics per hand and per session:

- rest-pose residual,
- sign agreement score,
- scale repeatability,
- thumb-specific residual,
- joint-limit hit rate,
- frame-to-frame angle jitter,
- bone-length drift in the reconstructed skeleton,
- fingertip residual after tip-lock fusion,
- confidence score aggregated from the above.

### Confidence Interpretation

Confidence should go down when:

- repeated captures disagree,
- the solver relies heavily on clamping,
- thumb coupling residuals are large,
- the session contains many noisy or missing frames,
- the evaluator cannot match the prompt sequence cleanly.

The confidence score should be reported as a number in `[0, 1]` and be accompanied by the component metrics that produced it.

## File And CLI Layout

Proposed file structure:

- `hand_reconstruction/calibration.py`
- `hand_reconstruction/calibration_io.py`
- `hand_reconstruction/calibration_metrics.py`
- `scripts/calibrate_hand.py`
- `scripts/evaluate_hand_calibration.py`

The first script should support at least these modes:

- `record`: store a calibration session,
- `fit`: produce a JSON calibration profile,
- `inspect`: print a summary of the fitted profile.

The second script should support:

- replay of a calibration session,
- loading an existing profile,
- output of a JSON metrics report,
- optional text summary for quick review.

`scripts/reconstruct_hand_frame.py` can remain the generic one-frame reconstruction entry point. It does not need to become the calibration tool.

## Integration Points

The calibration profile should be consumable by the current retargeting layer later, but this design does not require live integration in the first implementation.

Recommended integration order:

1. Offline calibration and evaluation tools.
2. Add profile loading to `hand_reconstruction/retargeting.py`.
3. Optionally wire profile loading into `DexCap_v4/dexcap_glove_meshcat_stream.py` after the offline workflow proves useful.

This order keeps the live stream stable while the calibration math is validated.

## Error Handling

Validation should be strict and explicit:

- reject malformed session files,
- reject profiles with missing or wrong-length vectors,
- reject non-finite parameters,
- reject sessions that do not contain enough rest samples for a valid zero pose,
- reject thumb calibration if thumb-only prompts are missing.

When the evaluator cannot compute a metric, it should say so in the report rather than silently fabricating a score.

## Testing

Add focused tests for the calibration layer and evaluator:

- profile serialization and deserialization round-trip,
- zero pose estimation from repeated rest samples,
- sign and scale normalization on synthetic q vectors,
- thumb calibration using thumb-only motion segments,
- evaluator metrics on a synthetic replay session,
- confidence decreases when noise or clamping increases,
- malformed session and malformed profile rejection.

Keep the tests pure Python and NumPy where possible so calibration logic can be validated without MeshCat or Pinocchio.

## Non-Goals

- No camera-based calibration.
- No learned pose estimation.
- No real-time adaptive calibration in the first version.
- No changes to the current skeleton display layout.
- No promise of absolute physical ground truth without an external reference.
