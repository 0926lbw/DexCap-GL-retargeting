"""Glove -> human-hand retargeting (anatomical first-pass map).

Maps the 21 glove joint angles (``q_l`` / ``q_r``) onto the 20 human-hand joint
angles of the parametric model (see :mod:`human_hand_builder`). The output is a
``{joint_name: angle_rad}`` dict, so it is independent of the pinocchio q-vector
ordering (pinocchio reorders joints after merging fixed joints).

Glove q layout (per finger, proximal->distal after Listener parsing; thumb has 5 joints):
  thumb  q[0..4] = [CMC_abd, CMC/MCP_flex, IP_flex, DIP, tip]
  index  q[5..8] = [MCP_abd, MCP_flex, PIP, DIP]
  middle q[9..12]= [MCP_abd, MCP_flex, PIP, DIP]
  ring   q[13..16]=[MCP_abd, MCP_flex, PIP, DIP]
  pinky  q[17..20]=[MCP_abd, MCP_flex, PIP, DIP]

The live glove exposes two useful curl observations for non-thumb fingers: a
PIP-side channel and a distal channel that often carries the strongest grasp
signal. The retargeter keeps both, then applies a light anatomical post-process:
joint limits, DIP/PIP coupling, MCP abduction damping under flexion, and
thumb-specific coupling. ``q_offset`` subtracts a captured rest pose.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

FINGER_NAMES = ("index", "middle", "ring", "pinky")
FINGER_DIP_PIP_RATIO = 0.6
DISTAL_TO_PIP_RATIO = 2.0 / 3.0
MCP_ABD_MIN_DAMPING = 0.25
THUMB_MCP_FROM_CMC_FLEX_RATIO = 0.5

RETARGET_LIMITS = {
    "mcp_flex": (0.0, 1.5708),
    "mcp_abd": (-0.3491, 0.3491),
    "pip": (0.0, 1.7453),
    "dip": (0.0, 1.3963),
    "thumb_cmc_flex": (-0.5236, 1.0472),
    "thumb_cmc_abd": (-0.5236, 0.5236),
    "thumb_mcp": (-0.5236, 1.0472),
    "thumb_ip": (-0.5236, 1.0472),
}

# (human_joint_name, glove_q_index, scale). Human joint names match the URDF
# joints emitted by human_hand_builder. The pre-constraint ``*_dip`` entries are
# distal curl observations folded into the final PIP estimate; final DIP remains
# anatomically coupled from final PIP in _apply_anatomical_constraints().
RETARGET_TABLE: tuple[tuple[str, int, float], ...] = (
    ("index_mcp_flex", 6, +1.0),
    ("index_mcp_abd", 5, +1.0),
    ("index_pip", 7, +1.0),
    ("index_dip", 8, +DISTAL_TO_PIP_RATIO),
    ("middle_mcp_flex", 10, +1.0),
    ("middle_mcp_abd", 9, +1.0),
    ("middle_pip", 11, +1.0),
    ("middle_dip", 12, +DISTAL_TO_PIP_RATIO),
    ("ring_mcp_flex", 14, +1.0),
    ("ring_mcp_abd", 13, -1.0),  # DexCap negates ring/pinky MCP abduction
    ("ring_pip", 15, +1.0),
    ("ring_dip", 16, +DISTAL_TO_PIP_RATIO),
    ("pinky_mcp_flex", 18, +1.0),
    ("pinky_mcp_abd", 17, -1.0),
    ("pinky_pip", 19, +1.0),
    ("pinky_dip", 20, +DISTAL_TO_PIP_RATIO),
    ("thumb_cmc_flex", 1, +1.0),
    ("thumb_cmc_abd", 0, +1.0),
    ("thumb_mcp", 1, +0.5),  # coupled from the CMC/MCP flex encoder
    ("thumb_ip", 2, +1.0),
)

LEFT_RETARGET_TABLE: tuple[tuple[str, int, float], ...] = (
    ("index_mcp_flex", 6, -1.0),
    ("index_mcp_abd", 5, +1.0),
    ("index_pip", 7, +1.0),
    ("index_dip", 8, -DISTAL_TO_PIP_RATIO),
    ("middle_mcp_flex", 10, -1.0),
    ("middle_mcp_abd", 9, +1.0),
    ("middle_pip", 11, +1.0),
    ("middle_dip", 12, -DISTAL_TO_PIP_RATIO),
    ("ring_mcp_flex", 14, -1.0),
    ("ring_mcp_abd", 13, -1.0),
    ("ring_pip", 15, +1.0),
    ("ring_dip", 16, -DISTAL_TO_PIP_RATIO),
    ("pinky_mcp_flex", 18, -1.0),
    ("pinky_mcp_abd", 17, -1.0),
    ("pinky_pip", 19, +1.0),
    ("pinky_dip", 20, -DISTAL_TO_PIP_RATIO),
    ("thumb_cmc_flex", 1, +1.0),
    ("thumb_cmc_abd", 0, +1.0),
    ("thumb_mcp", 1, +0.5),
    ("thumb_ip", 2, +1.0),
)

HUMAN_JOINT_NAMES: tuple[str, ...] = tuple(entry[0] for entry in RETARGET_TABLE)
GLOVE_DOF = 21


class GloveToHumanRetargeter:
    """Convert a 21-dim glove joint vector into a 20-entry human-hand joint dict."""

    def __init__(
        self,
        hand: str = "right",
        q_offset: Optional[np.ndarray] = None,
        table: tuple[tuple[str, int, float], ...] | None = None,
    ) -> None:
        if hand not in {"left", "right"}:
            raise ValueError("hand must be 'left' or 'right'")
        self.hand = hand
        self.table = _default_table_for_hand(hand) if table is None else table
        self.q_offset = (
            np.zeros(GLOVE_DOF)
            if q_offset is None
            else np.asarray(q_offset, dtype=float).copy()
        )
        if self.q_offset.shape != (GLOVE_DOF,):
            raise ValueError(f"q_offset must have shape ({GLOVE_DOF},), got {self.q_offset.shape}")

    def retarget(self, q_glove: np.ndarray) -> dict[str, float]:
        """Return {human_joint_name: angle_rad} from the glove joint vector."""
        q = np.asarray(q_glove, dtype=float)
        if q.shape != (GLOVE_DOF,):
            raise ValueError(f"q_glove must have shape ({GLOVE_DOF},), got {q.shape}")
        q = q - self.q_offset
        joints = {name: float(scale * q[idx]) for name, idx, scale in self.table}
        return _apply_anatomical_constraints(joints)


def _default_table_for_hand(hand: str) -> tuple[tuple[str, int, float], ...]:
    return LEFT_RETARGET_TABLE if hand == "left" else RETARGET_TABLE


def _apply_anatomical_constraints(joints: dict[str, float]) -> dict[str, float]:
    constrained = dict(joints)

    for finger in FINGER_NAMES:
        flex_name = f"{finger}_mcp_flex"
        abd_name = f"{finger}_mcp_abd"
        pip_name = f"{finger}_pip"
        dip_name = f"{finger}_dip"

        mcp_flex = _clamp(constrained[flex_name], RETARGET_LIMITS["mcp_flex"])
        pip = _clamp(
            max(constrained[pip_name], constrained[dip_name]),
            RETARGET_LIMITS["pip"],
        )
        mcp_abd = _clamp(constrained[abd_name], RETARGET_LIMITS["mcp_abd"])
        mcp_abd *= _mcp_abduction_damping(mcp_flex, pip)
        dip = _clamp(pip * FINGER_DIP_PIP_RATIO, RETARGET_LIMITS["dip"])

        constrained[flex_name] = mcp_flex
        constrained[abd_name] = mcp_abd
        constrained[pip_name] = pip
        constrained[dip_name] = dip

    thumb_cmc_flex = _clamp(
        constrained["thumb_cmc_flex"],
        RETARGET_LIMITS["thumb_cmc_flex"],
    )
    constrained["thumb_cmc_flex"] = thumb_cmc_flex
    constrained["thumb_cmc_abd"] = _clamp(
        constrained["thumb_cmc_abd"],
        RETARGET_LIMITS["thumb_cmc_abd"],
    )
    constrained["thumb_mcp"] = _clamp(
        thumb_cmc_flex * THUMB_MCP_FROM_CMC_FLEX_RATIO,
        RETARGET_LIMITS["thumb_mcp"],
    )
    constrained["thumb_ip"] = _clamp(
        constrained["thumb_ip"],
        RETARGET_LIMITS["thumb_ip"],
    )

    return constrained


def _mcp_abduction_damping(mcp_flex: float, pip: float) -> float:
    flex_norm = max(
        _normalize_positive(mcp_flex, RETARGET_LIMITS["mcp_flex"][1]),
        _normalize_positive(pip, RETARGET_LIMITS["pip"][1]),
    )
    return 1.0 - (1.0 - MCP_ABD_MIN_DAMPING) * flex_norm


def _normalize_positive(value: float, max_value: float) -> float:
    if max_value <= 0.0:
        return 0.0
    return float(np.clip(value / max_value, 0.0, 1.0))


def _clamp(value: float, limit: tuple[float, float]) -> float:
    lo, hi = limit
    return float(np.clip(value, lo, hi))
