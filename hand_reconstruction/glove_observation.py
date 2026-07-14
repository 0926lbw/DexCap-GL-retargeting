"""DexGlove URDF observation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .coordinate_frames import (
    apply_transform,
    default_coordinate_convention,
    make_transform,
)
from .human_hand_model import HumanHandSkeleton
from .schema import FINGER_CHAINS, FINGER_ORDER, NUM_LANDMARKS, WRIST


@dataclass(frozen=True)
class GloveLinkObservation:
    """Selected DexGlove link-frame positions for one hand."""

    hand: str
    base_position: np.ndarray
    link_positions: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        if self.hand not in {"left", "right"}:
            raise ValueError("hand must be 'left' or 'right'")
        base_position = _as_point(self.base_position, "base_position")
        link_positions = {
            name: _as_point(position, name)
            for name, position in self.link_positions.items()
        }
        object.__setattr__(self, "base_position", base_position)
        object.__setattr__(self, "link_positions", link_positions)


class SkeletonInitializer:
    """Build a constrained 21-point skeleton from DexGlove observations."""

    def __init__(self, observation_weight: float = 0.35) -> None:
        if not 0.0 <= observation_weight <= 1.0:
            raise ValueError("observation_weight must be in [0, 1]")
        self.observation_weight = float(observation_weight)

    def from_observation(self, observation: GloveLinkObservation) -> HumanHandSkeleton:
        template = HumanHandSkeleton.default(observation.hand)
        template_points = template.keypoints_wrist()
        points_h = template_points.copy()
        names_by_finger = finger_link_names(observation.hand)

        for finger in FINGER_ORDER:
            chain = FINGER_CHAINS[finger]
            link_names = _select_landmark_links(names_by_finger[finger], len(chain) - 1)
            observed_points = _observed_points(observation, link_names)

            points_h[chain[1]] = template_points[chain[1]]
            for segment_idx in range(1, len(chain) - 1):
                start_idx = chain[segment_idx]
                end_idx = chain[segment_idx + 1]
                template_vector = template_points[end_idx] - template_points[start_idx]
                observed_vector = (
                    observed_points[segment_idx] - observed_points[segment_idx - 1]
                )
                direction = _blend_direction(
                    template_vector,
                    observed_vector,
                    self.observation_weight,
                )
                length = float(np.linalg.norm(template_vector))
                points_h[end_idx] = points_h[start_idx] + direction * length

        transform = make_transform(translation=observation.base_position)
        points_w = apply_transform(transform, points_h)
        return HumanHandSkeleton(
            points_w,
            hand=observation.hand,
            keypoints_21_in_Hwrist=points_h,
            T_W_Hwrist=transform,
            coordinate_convention=default_coordinate_convention(),
        )


def direct_skeleton_from_observation(
    observation: GloveLinkObservation,
) -> HumanHandSkeleton:
    """Build a 21-point skeleton whose landmarks are exactly the observed DexGlove
    link positions, with the wrist at ``base_link``.

    Unlike :class:`SkeletonInitializer`, there is no anatomical template and no
    direction blending: every landmark (except the wrist) is set directly to a
    glove link origin. The skeleton is therefore frame-aligned with the glove
    URDF by construction, so it inherits the exoskeleton's orientation and
    handedness exactly and never needs a coordinate-convention fix-up.

    Landmark -> link mapping (one link per non-wrist landmark):
      * thumb  : CMC/MCP/IP/TIP  <- last 4 of glove_link_{side}_1_{1..5}
      * index  : MCP/PIP/DIP/TIP <- glove_link_{side}_2_{1..4}
      * middle : MCP/PIP/DIP/TIP <- glove_link_{side}_3_{1..4}
      * ring   : MCP/PIP/DIP/TIP <- glove_link_{side}_4_{1..4}
      * pinky  : MCP/PIP/DIP/TIP <- glove_link_{side}_5_{1..4}
    """
    points = np.zeros((NUM_LANDMARKS, 3))
    points[WRIST] = observation.base_position
    names_by_finger = finger_link_names(observation.hand)
    for finger in FINGER_ORDER:
        chain = FINGER_CHAINS[finger]
        link_names = _select_landmark_links(names_by_finger[finger], len(chain) - 1)
        for landmark_idx, link_name in zip(chain[1:], link_names):
            if link_name not in observation.link_positions:
                raise KeyError(f"missing observed link position for {link_name}")
            points[landmark_idx] = observation.link_positions[link_name]
    return HumanHandSkeleton(points, hand=observation.hand)


class DexGloveObserver:
    """Pinocchio-backed observer for DexGlove URDF link positions."""

    def __init__(self, urdf_path: str | Path, hand: str) -> None:
        if hand not in {"left", "right"}:
            raise ValueError("hand must be 'left' or 'right'")
        try:
            import pinocchio as pin
            from pinocchio.robot_wrapper import RobotWrapper
        except ImportError as exc:
            raise ImportError(
                "DexGloveObserver requires pinocchio. Install/use the DexCap robotics "
                "environment, or use SkeletonInitializer with precomputed link positions."
            ) from exc

        self._pin = pin
        self.hand = hand
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.robot = RobotWrapper.BuildFromURDF(
            str(self.urdf_path),
            package_dirs=_urdf_package_dirs(self.urdf_path),
        )
        self._link_names = ("base_link",) + tuple(
            link_name
            for finger in FINGER_ORDER
            for link_name in finger_link_names(hand)[finger]
        )

    def observe(self, q: np.ndarray) -> GloveLinkObservation:
        q = np.asarray(q, dtype=float)
        if q.shape != (self.robot.model.nq,):
            raise ValueError(f"q must have shape ({self.robot.model.nq},), got {q.shape}")

        self._pin.forwardKinematics(self.robot.model, self.robot.data, q)
        self._pin.updateFramePlacements(self.robot.model, self.robot.data)

        link_positions: dict[str, np.ndarray] = {}
        for link_name in self._link_names:
            frame_id = self.robot.model.getFrameId(link_name)
            if frame_id >= len(self.robot.model.frames):
                raise KeyError(f"URDF frame not found: {link_name}")
            link_positions[link_name] = self.robot.data.oMf[frame_id].translation.copy()

        return GloveLinkObservation(
            hand=self.hand,
            base_position=link_positions["base_link"],
            link_positions=link_positions,
        )


def finger_link_names(hand: str) -> dict[str, tuple[str, ...]]:
    """Return DexGlove link names used as observations for each finger."""
    if hand == "right":
        side = "r"
    elif hand == "left":
        side = "l"
    else:
        raise ValueError("hand must be 'left' or 'right'")

    return {
        "thumb": tuple(f"glove_link_{side}_1_{idx}" for idx in range(1, 6)),
        "index": tuple(f"glove_link_{side}_2_{idx}" for idx in range(1, 5)),
        "middle": tuple(f"glove_link_{side}_3_{idx}" for idx in range(1, 5)),
        "ring": tuple(f"glove_link_{side}_4_{idx}" for idx in range(1, 5)),
        "pinky": tuple(f"glove_link_{side}_5_{idx}" for idx in range(1, 5)),
    }


def _observed_points(
    observation: GloveLinkObservation,
    link_names: tuple[str, ...],
) -> list[np.ndarray]:
    points = []
    for link_name in link_names:
        if link_name not in observation.link_positions:
            raise KeyError(f"missing observed link position for {link_name}")
        points.append(observation.link_positions[link_name])
    return points


def _blend_direction(
    template_vector: np.ndarray,
    observed_vector: np.ndarray,
    observation_weight: float,
) -> np.ndarray:
    template_direction = _unit(template_vector)
    if float(np.linalg.norm(observed_vector)) <= 1e-9:
        return template_direction
    observed_direction = _unit(observed_vector)
    blended = (
        (1.0 - observation_weight) * template_direction
        + observation_weight * observed_direction
    )
    if float(np.linalg.norm(blended)) <= 1e-9:
        return template_direction
    return _unit(blended)


def _select_landmark_links(
    link_names: tuple[str, ...],
    landmark_count: int,
) -> tuple[str, ...]:
    if len(link_names) == landmark_count:
        return link_names
    if len(link_names) < landmark_count:
        raise ValueError(
            f"need at least {landmark_count} observed links, got {len(link_names)}"
        )
    return link_names[-landmark_count:]


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("cannot normalize zero-length vector")
    return vector / norm


def _as_point(value: np.ndarray, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {point.shape}")
    return point


def _urdf_package_dirs(urdf_path: Path) -> list[str]:
    urdf_dir = urdf_path.parent
    parent = urdf_dir.parent
    grandparent = parent.parent
    candidates = [
        urdf_dir,
        urdf_dir / "DexCap_v4",
        parent,
        parent / "DexCap_v4",
        grandparent,
        grandparent / "DexCap_v4",
    ]
    return [str(path) for path in candidates if path.is_dir()]
