"""Streaming-friendly MeshCat overlay for the 21-point human hand skeleton.

Unlike :class:`hand_reconstruction.visualize_meshcat.MeshcatHandSkeletonViewer`,
this overlay renders into an *existing* meshcat viewer node (it does not create
its own server/viewer) and is designed for live streaming:

* ``build()`` creates the geometry (21 joint spheres + 21 bone cylinders) once.
* ``update()`` only changes transforms each frame.

Re-issuing ``set_object`` every frame is what makes a streamed meshcat scene
flicker (the geometry is re-sent over the websocket, and reconnects replay it).
Keeping geometry creation out of the per-frame path avoids that, matching how
pinocchio's own ``MeshcatVisualizer.display`` works.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .schema import MEDIAPIPE_HAND_CONNECTIONS, NUM_LANDMARKS

DEFAULT_SPHERE_RADIUS = 0.004
DEFAULT_BONE_RADIUS = 0.002
DEFAULT_SPHERE_COLOR = 0x2A9D8F  # teal
DEFAULT_BONE_COLOR = 0x264653  # dark teal
# meshcat ``Cylinder`` is built with height along +Y (three.js convention).
_CYLINDER_BASE_AXIS = (0.0, 1.0, 0.0)


def cylinder_transform_between(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    base_axis: tuple[float, float, float] = _CYLINDER_BASE_AXIS,
    eps: float = 1e-9,
) -> np.ndarray:
    """Return the 4x4 transform placing a unit-height (+Y) cylinder from ``p0`` to ``p1``.

    The cylinder geometry is created once with height 1 (see
    :class:`MeshcatSkeletonOverlay.build`). Per-frame segment length is produced
    by scaling the cylinder's Y basis column by ``||p1 - p0||``, so the caller
    never has to re-create the geometry just to change length.

    The degenerate case (coincident points) returns a finite transform with a
    zero-length Y column placed at ``p0`` instead of raising.
    """
    p0 = np.asarray(p0, dtype=float).reshape(3)
    p1 = np.asarray(p1, dtype=float).reshape(3)
    delta = p1 - p0
    length = float(np.linalg.norm(delta))

    transform = np.eye(4, dtype=float)
    if length <= eps:
        # Collapse to an invisible nub at p0; no NaN, no raise.
        transform[:3, 1] = 0.0
        transform[:3, 3] = p0
        return transform

    mid = 0.5 * (p0 + p1)
    axis = delta / length
    rotation = _rotation_mapping(base_axis, axis)
    transform[:3, 0] = rotation[:, 0]
    transform[:3, 1] = rotation[:, 1] * length
    transform[:3, 2] = rotation[:, 2]
    transform[:3, 3] = mid
    return transform


def _rotation_mapping(a: tuple[float, float, float], b: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps unit vector ``a`` onto unit vector ``b`` (Rodrigues)."""
    a_vec = np.asarray(a, dtype=float)
    cross = np.cross(a_vec, b)
    sin_theta = float(np.linalg.norm(cross))
    cos_theta = float(np.dot(a_vec, b))

    if sin_theta < 1e-12:
        # Parallel or anti-parallel: the cross product is undefined.
        if cos_theta > 0.0:
            return np.eye(3, dtype=float)
        # 180-degree turn: rotate about any axis perpendicular to ``a``.
        perp = _unit_perpendicular(a_vec)
        return 2.0 * np.outer(perp, perp) - np.eye(3, dtype=float)

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=float,
    )
    factor = (1.0 - cos_theta) / (sin_theta * sin_theta)
    return np.eye(3, dtype=float) + skew + (skew @ skew) * factor


def _unit_perpendicular(vec: np.ndarray) -> np.ndarray:
    """Return any unit vector perpendicular to ``vec``."""
    seed = np.array([1.0, 0.0, 0.0]) if abs(vec[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp = seed - vec * float(np.dot(vec, seed))
    norm = float(np.linalg.norm(perp))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return perp / norm


class MeshcatSkeletonOverlay:
    """Draw a 21-point skeleton into a shared meshcat node, flicker-free for streaming.

    Pass a *node handle* from the shared viewer (e.g.
    ``viewer["DexGlove_L_v4"]["human_hand"]``) and inject the lazily-imported
    ``meshcat.geometry`` / ``meshcat.transformations`` modules so the class stays
    testable without meshcat installed.
    """

    def __init__(
        self,
        root_node: Any,
        hand: str,
        *,
        geometry: Any,
        transforms: Any,
        sphere_radius: float = DEFAULT_SPHERE_RADIUS,
        bone_radius: float = DEFAULT_BONE_RADIUS,
        sphere_color: int = DEFAULT_SPHERE_COLOR,
        bone_color: int = DEFAULT_BONE_COLOR,
    ) -> None:
        if hand not in {"left", "right"}:
            raise ValueError("hand must be 'left' or 'right'")
        self._root = root_node
        self.hand = hand
        self._g = geometry
        self._tf = transforms
        self._sphere_radius = sphere_radius
        self._bone_radius = bone_radius
        self._sphere_color = sphere_color
        self._bone_color = bone_color
        self._spheres: list[Any] = []
        self._bones: list[tuple[Any, int, int]] = []
        self._built = False

    def build(self) -> None:
        """Create all joint spheres and bone cylinders exactly once."""
        if self._built:
            return
        geometry = self._g
        sphere_material = geometry.MeshLambertMaterial(color=self._sphere_color)
        bone_material = geometry.MeshLambertMaterial(color=self._bone_color)

        for i in range(NUM_LANDMARKS):
            node = self._root[f"point_{i:02d}"]
            node.set_object(geometry.Sphere(self._sphere_radius), sphere_material)
            self._spheres.append(node)

        for j, (start, end) in enumerate(MEDIAPIPE_HAND_CONNECTIONS):
            node = self._root[f"bone_{j:02d}"]
            # Unit-height cylinder; segment length is applied per-frame via
            # transform scaling in ``cylinder_transform_between``.
            node.set_object(geometry.Cylinder(1.0, self._bone_radius), bone_material)
            self._bones.append((node, start, end))

        self._built = True

    def update(self, landmarks: np.ndarray) -> None:
        """Move the existing spheres/bones to ``landmarks`` (21x3). Geometry is untouched."""
        if not self._built:
            raise RuntimeError("MeshcatSkeletonOverlay.build() must be called before update()")
        landmarks = np.asarray(landmarks, dtype=float)
        if landmarks.shape != (NUM_LANDMARKS, 3):
            raise ValueError(
                f"landmarks must have shape ({NUM_LANDMARKS}, 3), got {landmarks.shape}"
            )

        translation_matrix = self._tf.translation_matrix
        for i, node in enumerate(self._spheres):
            node.set_transform(translation_matrix(landmarks[i]))
        for node, start, end in self._bones:
            node.set_transform(cylinder_transform_between(landmarks[start], landmarks[end]))
