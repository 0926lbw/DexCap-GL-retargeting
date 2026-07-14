"""Parametric human-hand model parameters — the optimizable "skeleton model".

Every field here is a calibration knob. Defaults are derived from the existing
right-hand ``_mediapipe_open_hand_template`` landmark geometry in
``human_hand_model.py`` so the rest pose already matches that validated shape;
the left hand is produced by mirroring (see :func:`HumanHandParams.for_hand`).

Coordinate convention (matches ``coordinate_frames.default_coordinate_convention``):
wrist frame  +X = thumb/radial,  +Y = fingers (distal),  +Z = palm normal (dorsal).
Flexion curls the finger toward the palm (-Z); abduction spreads radially/ulnarly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

FLEX_AXIS_RIGHT = (-1.0, 0.0, 0.0)  # positive angle curls +Y toward -Z (into palm)
ABD_AXIS_RIGHT = (0.0, 0.0, -1.0)   # positive angle spreads toward +X (radial)
ABD_AXIS_LEFT = (0.0, 0.0, 1.0)     # mirrored


@dataclass(frozen=True)
class FingerParams:
    """One non-thumb finger: palm attachment + fanning yaw + phalanx lengths (m)."""
    attach: tuple[float, float, float]  # (x, y, z) of the MCP joint in the wrist frame
    yaw: float = 0.0  # constant Z rotation so fingers fan out like a real hand
    prox: float = 0.0  # proximal phalanx (MCP->PIP)
    mid: float = 0.0  # middle phalanx (PIP->DIP)
    dist: float = 0.0  # distal phalanx (DIP->TIP)


@dataclass(frozen=True)
class ThumbParams:
    attach: tuple[float, float, float]
    cmc_rpy: tuple[float, float, float]  # constant CMC frame tilt (opposition)
    meta: float = 0.0  # metacarpal (CMC->MCP)
    prox: float = 0.0  # proximal phalanx (MCP->IP)
    dist: float = 0.0  # distal phalanx (IP->TIP)


@dataclass(frozen=True)
class HumanHandParams:
    """All tunable geometry of the parametric human hand."""

    fingers: dict[str, FingerParams] = field(default_factory=dict)
    thumb: ThumbParams = field(default_factory=ThumbParams)
    flex_axis: tuple[float, float, float] = FLEX_AXIS_RIGHT
    abd_axis: tuple[float, float, float] = ABD_AXIS_RIGHT
    # Joint limits (rad): (lower, upper). Positive flexion = curl into palm.
    limits: dict[str, tuple[float, float]] = field(default_factory=dict)

    def for_hand(self, hand: str) -> "HumanHandParams":
        """Return params adjusted for left/right (mirror palm X + abduction axis)."""
        if hand not in {"left", "right"}:
            raise ValueError("hand must be 'left' or 'right'")
        if hand == "right":
            return self

        mirrored_fingers = {
            name: replace(
                f,
                attach=(-f.attach[0], f.attach[1], f.attach[2]),
                yaw=-f.yaw,
            )
            for name, f in self.fingers.items()
        }
        mirrored_thumb = replace(
            self.thumb,
            attach=(-self.thumb.attach[0], self.thumb.attach[1], self.thumb.attach[2]),
            cmc_rpy=(-self.thumb.cmc_rpy[0], self.thumb.cmc_rpy[1], -self.thumb.cmc_rpy[2]),
        )
        return replace(
            self,
            fingers=mirrored_fingers,
            thumb=mirrored_thumb,
            abd_axis=ABD_AXIS_LEFT,
        )


def default_params() -> HumanHandParams:
    """Average adult right hand, geometry aligned to the repo's mediapipe template.

    Phalanx lengths and MCP attachments are measured from
    ``_mediapipe_open_hand_template`` (consecutive landmark distances and MCP
    positions); per-finger ``yaw`` fans the straight phalanges so the rest pose
    reproduces that template to within a few mm. The thumb is intentionally set
    to an opposed pose (out of the palm plane), which is more anatomical than the
    flat template.
    """
    fingers = {
        "index": FingerParams(
            attach=(0.030, 0.075, 0.0), yaw=-0.249,
            prox=0.0432, mid=0.0269, dist=0.0187,
        ),
        "middle": FingerParams(
            attach=(0.006, 0.083, 0.0), yaw=0.0,
            prox=0.0490, mid=0.0310, dist=0.0350,
        ),
        "ring": FingerParams(
            attach=(-0.020, 0.076, 0.0), yaw=0.204,
            prox=0.0457, mid=0.0286, dist=0.0196,
        ),
        "pinky": FingerParams(
            attach=(-0.043, 0.062, 0.0), yaw=0.211,
            prox=0.0361, mid=0.0275, dist=0.0224,
        ),
    }
    thumb = ThumbParams(
        attach=(0.035, 0.028, 0.012),
        # roll, pitch, yaw: pitch tilts the thumb palmar; yaw swings it radial.
        cmc_rpy=(0.0, 0.35, -0.70),
        meta=0.0347, prox=0.0320, dist=0.0288,
    )
    limits = {
        "mcp_flex": (0.0, 1.5708),      # 0 .. 90 deg
        "mcp_abd": (-0.3491, 0.3491),   # +/- 20 deg
        "pip": (0.0, 1.7453),           # 0 .. 100 deg
        "dip": (0.0, 1.3963),           # 0 .. 80 deg
        "thumb_cmc_flex": (-0.5236, 1.0472),
        "thumb_cmc_abd": (-0.5236, 0.5236),
        "thumb_mcp": (-0.5236, 1.0472),
        "thumb_ip": (-0.5236, 1.0472),
    }
    return HumanHandParams(
        fingers=fingers,
        thumb=thumb,
        flex_axis=FLEX_AXIS_RIGHT,
        abd_axis=ABD_AXIS_RIGHT,
        limits=limits,
    )
