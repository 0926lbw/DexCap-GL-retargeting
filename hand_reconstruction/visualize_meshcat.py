"""Optional MeshCat visualization for reconstructed 21-point hand skeletons."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .human_hand_model import HumanHandSkeleton
from .schema import (
    INDEX_TIP,
    MEDIAPIPE_HAND_CONNECTIONS,
    MIDDLE_TIP,
    PINKY_TIP,
    RING_TIP,
    THUMB_TIP,
)


class MeshcatHandSkeletonViewer:
    """Draw a 21-point skeleton in MeshCat using spheres and line segments."""

    def __init__(self, root_path: str = "human_hand") -> None:
        try:
            import meshcat
            import meshcat.geometry as g
            import meshcat.transformations as tf
        except ImportError as exc:
            raise ImportError(
                "MeshcatHandSkeletonViewer requires meshcat. Install meshcat or use JSON/NPY export."
            ) from exc

        self._g = g
        self._tf = tf
        self.viewer = meshcat.Visualizer().open()
        self.root_path = root_path

    def display(self, skeleton: HumanHandSkeleton) -> None:
        root = self.viewer[self.root_path]
        render_skeleton_nodes(root, skeleton, self._g, self._tf)


def render_skeleton_nodes(root, skeleton: HumanHandSkeleton, geometry, transforms) -> None:
    points = skeleton.to_numpy()

    for idx, point in enumerate(points):
        node = root[f"point_{idx:02d}"]
        node.set_object(
            geometry.Sphere(0.004),
            geometry.MeshLambertMaterial(color=0x2A9D8F),
        )
        node.set_transform(transforms.translation_matrix(point))

    for edge_idx, (start, end) in enumerate(skeleton_edges()):
        segment = _line_segment(points[start], points[end])
        node = root[f"edge_{edge_idx:02d}"]
        node.set_object(
            geometry.Line(
                geometry.PointsGeometry(segment),
                geometry.MeshBasicMaterial(color=0x264653),
            )
        )
    for axis_name, start, end, color, label in coordinate_axes():
        segment = _line_segment(start, end)
        root[f"frame/{axis_name}"].set_object(
            geometry.Line(
                geometry.PointsGeometry(segment),
                geometry.MeshBasicMaterial(color=color),
            )
        )
        label_node = root[f"frame/{axis_name}_label"]
        _set_text_label(label_node, geometry, label, color=color)
        label_node.set_transform(transforms.translation_matrix(end * 1.12))

    for label, landmark_idx in fingertip_labels():
        label_node = root[f"label_{label}"]
        _set_text_label(label_node, geometry, label, color=0xF4A261)
        label_node.set_transform(
            transforms.translation_matrix(points[landmark_idx] + np.array([0.0, 0.010, 0.0]))
        )


def _set_text_label(node, geometry, text: str, color: int) -> None:
    if hasattr(geometry, "Text"):
        node.set_object(geometry.Text(text))
        return
    node.set_object(
        geometry.Sphere(0.003),
        geometry.MeshLambertMaterial(color=color),
    )


def coordinate_axes() -> tuple[tuple[str, np.ndarray, np.ndarray, int, str], ...]:
    origin = np.zeros(3, dtype=float)
    return (
        ("x_axis", origin, np.array([0.045, 0.000, 0.000]), 0xE63946, "+X thumb side"),
        ("y_axis", origin, np.array([0.000, 0.060, 0.000]), 0x2A9D8F, "+Y fingers"),
        ("z_axis", origin, np.array([0.000, 0.000, 0.045]), 0x457B9D, "+Z palm normal"),
    )


def fingertip_labels() -> tuple[tuple[str, int], ...]:
    return (
        ("thumb", THUMB_TIP),
        ("index", INDEX_TIP),
        ("middle", MIDDLE_TIP),
        ("ring", RING_TIP),
        ("pinky", PINKY_TIP),
    )


def skeleton_edges() -> Iterable[tuple[int, int]]:
    return iter(MEDIAPIPE_HAND_CONNECTIONS)


def _line_segment(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    return np.vstack((start, end)).T
