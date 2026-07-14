"""Parametric human-hand URDF builder + pinocchio FK landmark extraction.

Generates a 20-DOF human hand (4 DOF per finger: MCP[flex,abd] + PIP + DIP;
thumb CMC[flex,abd] + MCP + IP) from :class:`HumanHandParams`. The 21 landmark
positions come from forward kinematics of named link frames, in the exact order
of :mod:`hand_reconstruction.schema`, so they feed
:class:`hand_reconstruction.stream_overlay.MeshcatSkeletonOverlay` unchanged.

Coordinate convention (see human_hand_params): +X radial/thumb, +Y distal,
+Z dorsal. Flexion axis ``-X`` curls fingers into the palm (-Z); abduction axis
``-Z`` (right) / ``+Z`` (left) spreads radially.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .human_hand_params import HumanHandParams, default_params

# 21 landmark link-frame names in schema order (WRIST, then thumb CMC/MCP/IP/TIP,
# then index/middle/ring/pinky MCP/PIP/DIP/TIP).
LANDMARK_FRAME_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_meta_link", "thumb_prox_link", "thumb_dist_link", "thumb_tip_link",
    "index_prox_link", "index_mid_link", "index_dist_link", "index_tip_link",
    "middle_prox_link", "middle_mid_link", "middle_dist_link", "middle_tip_link",
    "ring_prox_link", "ring_mid_link", "ring_dist_link", "ring_tip_link",
    "pinky_prox_link", "pinky_mid_link", "pinky_dist_link", "pinky_tip_link",
)

_FINGER_ORDER = ("index", "middle", "ring", "pinky")



def build_human_hand_urdf(params: HumanHandParams, hand: str) -> str:
    """Return the URDF XML string for ``hand`` ('left' or 'right')."""
    params = params.for_hand(hand)
    flex = params.flex_axis
    abd = params.abd_axis

    lines: list[str] = ['<?xml version="1.0"?>', "<robot name=\"human_hand\">"]

    # wrist root link
    lines.append(_link("wrist"))
    # thumb chain (emitted first -> occupies q indices 0..3)
    lines.extend(_thumb_chain(params, flex, abd))
    # non-thumb fingers (q indices 4..7, 8..11, 12..15, 16..19)
    for f in _FINGER_ORDER:
        lines.extend(_finger_chain(f, params.fingers[f], flex, abd, params.limits))

    lines.append("</robot>")
    return "\n".join(lines) + "\n"


def _finger_chain(
    name: str,
    fp: Any,
    flex: tuple[float, float, float],
    abd: tuple[float, float, float],
    limits: dict[str, tuple[float, float]],
) -> list[str]:
    ax, ay, az = fp.attach
    out: list[str] = []
    # MCP attachment (fixed): palm offset + fanning yaw.
    out.append(_link(f"{name}_mcp_link"))
    out.append(_fixed_joint(
        f"{name}_mcp_attach", "wrist", f"{name}_mcp_link",
        origin_xyz=(ax, ay, az), origin_rpy=(0.0, 0.0, fp.yaw),
    ))
    # MCP flexion (revolute) at the MCP point.
    out.append(_link(f"{name}_flex_link"))
    out.append(_revolute_joint(
        f"{name}_mcp_flex", f"{name}_mcp_link", f"{name}_flex_link",
        axis=flex, limit=limits["mcp_flex"],
    ))
    # MCP abduction (revolute), still at the MCP point; this link starts the
    # proximal phalanx -> its frame origin is the MCP landmark.
    out.append(_link(f"{name}_prox_link"))
    out.append(_revolute_joint(
        f"{name}_mcp_abd", f"{name}_flex_link", f"{name}_prox_link",
        axis=abd, limit=limits["mcp_abd"],
    ))
    # PIP: proximal phalanx length on this joint's origin -> mid_link is PIP landmark.
    out.append(_link(f"{name}_mid_link"))
    out.append(_revolute_joint(
        f"{name}_pip", f"{name}_prox_link", f"{name}_mid_link",
        origin_xyz=(0.0, fp.prox, 0.0), axis=flex, limit=limits["pip"],
    ))
    # DIP: middle phalanx length -> dist_link is DIP landmark.
    out.append(_link(f"{name}_dist_link"))
    out.append(_revolute_joint(
        f"{name}_dip", f"{name}_mid_link", f"{name}_dist_link",
        origin_xyz=(0.0, fp.mid, 0.0), axis=flex, limit=limits["dip"],
    ))
    # Tip (fixed): distal phalanx length -> tip_link is TIP landmark.
    out.append(_link(f"{name}_tip_link"))
    out.append(_fixed_joint(
        f"{name}_tip", f"{name}_dist_link", f"{name}_tip_link",
        origin_xyz=(0.0, fp.dist, 0.0),
    ))
    return out


def _thumb_chain(
    params: HumanHandParams,
    flex: tuple[float, float, float],
    abd: tuple[float, float, float],
) -> list[str]:
    t = params.thumb
    limits = params.limits
    ax, ay, az = t.attach
    roll, pitch, yaw = t.cmc_rpy
    out: list[str] = []
    # CMC attachment (fixed): palm offset + opposition tilt.
    out.append(_link("thumb_cmc_link"))
    out.append(_fixed_joint(
        "thumb_cmc_attach", "wrist", "thumb_cmc_link",
        origin_xyz=(ax, ay, az), origin_rpy=(roll, pitch, yaw),
    ))
    # CMC flexion + abduction at the CMC point; meta_link frame = CMC landmark.
    out.append(_link("thumb_cmc_flex_link"))
    out.append(_revolute_joint(
        "thumb_cmc_flex", "thumb_cmc_link", "thumb_cmc_flex_link",
        axis=flex, limit=limits["thumb_cmc_flex"],
    ))
    out.append(_link("thumb_meta_link"))
    out.append(_revolute_joint(
        "thumb_cmc_abd", "thumb_cmc_flex_link", "thumb_meta_link",
        axis=abd, limit=limits["thumb_cmc_abd"],
    ))
    # MCP: metacarpal length -> prox_link is thumb MCP landmark.
    out.append(_link("thumb_prox_link"))
    out.append(_revolute_joint(
        "thumb_mcp", "thumb_meta_link", "thumb_prox_link",
        origin_xyz=(0.0, t.meta, 0.0), axis=flex, limit=limits["thumb_mcp"],
    ))
    # IP: proximal phalanx length -> dist_link is thumb IP landmark.
    out.append(_link("thumb_dist_link"))
    out.append(_revolute_joint(
        "thumb_ip", "thumb_prox_link", "thumb_dist_link",
        origin_xyz=(0.0, t.prox, 0.0), axis=flex, limit=limits["thumb_ip"],
    ))
    # Tip (fixed): distal phalanx length.
    out.append(_link("thumb_tip_link"))
    out.append(_fixed_joint(
        "thumb_tip", "thumb_dist_link", "thumb_tip_link",
        origin_xyz=(0.0, t.dist, 0.0),
    ))
    return out


def _link(name: str) -> str:
    # Minimal inertial so pinocchio loads every link cleanly.
    return (
        f'  <link name="{name}">\n'
        f'    <inertial>\n'
        f'      <mass value="1e-4"/>\n'
        f'      <inertia ixx="1e-9" ixy="0" ixz="0" iyy="1e-9" iyz="0" izz="1e-9"/>\n'
        f'    </inertial>\n'
        f'  </link>'
    )


def _xyz(xyz: tuple[float, float, float]) -> str:
    return f'{_f(xyz[0])} {_f(xyz[1])} {_f(xyz[2])}'


def _f(v: float) -> str:
    return f"{float(v):.6g}"


def _fixed_joint(
    name: str,
    parent: str,
    child: str,
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    return (
        f'  <joint name="{name}" type="fixed">\n'
        f'    <parent link="{parent}"/>\n'
        f'    <child link="{child}"/>\n'
        f'    <origin xyz="{_xyz(origin_xyz)}" rpy="{_xyz(origin_rpy)}"/>\n'
        f'  </joint>'
    )


def _revolute_joint(
    name: str,
    parent: str,
    child: str,
    axis: tuple[float, float, float],
    limit: tuple[float, float],
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    lo, hi = limit
    return (
        f'  <joint name="{name}" type="revolute">\n'
        f'    <parent link="{parent}"/>\n'
        f'    <child link="{child}"/>\n'
        f'    <origin xyz="{_xyz(origin_xyz)}" rpy="0 0 0"/>\n'
        f'    <axis xyz="{_xyz(axis)}"/>\n'
        f'    <limit lower="{_f(lo)}" upper="{_f(hi)}" effort="0" velocity="0"/>\n'
        f'  </joint>'
    )


class HumanHandModel:
    """Loaded pinocchio model of the human hand with 21-landmark FK extraction."""

    def __init__(
        self,
        model: Any,
        data: Any,
        frame_ids: list[int],
        joint_q_idx: dict[str, int],
        keep_alive: Any,
    ) -> None:
        self.model = model
        self._data = data
        self.frame_ids = frame_ids
        # joint name -> index in the q vector. Pinocchio does NOT preserve URDF
        # document order in q (fixed-joint merging reorders the tree), so callers
        # must address joints by name, not by position.
        self.joint_q_idx = joint_q_idx
        self._keep_alive = keep_alive  # hold the temp URDF dir for the process lifetime

    @property
    def nq(self) -> int:
        return self.model.nq

    def landmarks_from_q(self, q: np.ndarray) -> np.ndarray:
        """Forward-kinematics q (shape (nq,)) and return 21x3 landmark points."""
        import pinocchio as pin  # local; builder loaded the model already

        q = np.asarray(q, dtype=float)
        if q.shape != (self.model.nq,):
            raise ValueError(f"q must have shape ({self.model.nq},), got {q.shape}")
        pin.forwardKinematics(self.model, self._data, q)
        pin.updateFramePlacements(self.model, self._data)
        return np.stack(
            [self._data.oMf[fid].translation.copy() for fid in self.frame_ids]
        )

    def landmarks_from_joints(self, joint_angles: dict[str, float]) -> np.ndarray:
        """FK from a {joint_name: angle_rad} dict. Order-independent."""
        q = np.zeros(self.model.nq)
        for name, angle in joint_angles.items():
            idx = self.joint_q_idx.get(name)
            if idx is None:
                raise KeyError(f"unknown human-hand joint: {name}")
            q[idx] = float(angle)
        return self.landmarks_from_q(q)


def rigid_fit(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Best-fit proper rigid transform (Kabsch) mapping point set P onto Q.

    Returns a 4x4 homogeneous transform T with det(R) = +1 minimizing
    ``sum ||T[:3] @ p_i - q_i||``. Used to align the human hand's rest pose to
    the glove's actual link layout, so the skeleton inherits the glove's
    coordinate convention (direction + handedness) without hand-tuned rotations.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError(f"P and Q must have equal (N,3) shapes, got {P.shape}, {Q.shape}")
    centroid_p = P.mean(axis=0)
    centroid_q = Q.mean(axis=0)
    H = (P - centroid_p).T @ (Q - centroid_q)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = centroid_q - R @ centroid_p
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_human_hand(
    params: HumanHandParams | None = None,
    hand: str = "right",
) -> HumanHandModel:
    """Build the URDF, load it with pinocchio, and resolve the 21 landmark frames."""
    if params is None:
        params = default_params()
    if hand not in {"left", "right"}:
        raise ValueError("hand must be 'left' or 'right'")

    try:
        import pinocchio as pin
    except ImportError as exc:
        raise ImportError(
            "load_human_hand requires pinocchio. Run in the DexCap robotics env."
        ) from exc

    xml = build_human_hand_urdf(params, hand)
    tmp_dir = tempfile.mkdtemp(prefix="human_hand_")
    urdf_path = Path(tmp_dir) / f"human_hand_{hand}.urdf"
    urdf_path.write_text(xml, encoding="utf-8")

    model = pin.buildModelsFromUrdf(str(urdf_path))[0]
    data = pin.Data(model)

    frame_ids: list[int] = []
    missing = []
    for name in LANDMARK_FRAME_NAMES:
        fid = model.getFrameId(name)
        if fid >= len(model.frames):
            missing.append(name)
        frame_ids.append(int(fid))
    if missing:
        raise RuntimeError(
            f"human-hand URDF is missing landmark frames: {missing}. "
            "Pinocchio may have merged fixed-joint links."
        )

    joint_q_idx: dict[str, int] = {}
    for j in range(1, model.njoints):  # skip the 'universe' joint at index 0
        joint_q_idx[str(model.names[j])] = int(model.joints[j].idx_q)

    return HumanHandModel(model, data, frame_ids, joint_q_idx, keep_alive=tmp_dir)
