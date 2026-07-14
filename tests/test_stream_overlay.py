import unittest

import numpy as np

from hand_reconstruction.schema import MEDIAPIPE_HAND_CONNECTIONS, NUM_LANDMARKS
from hand_reconstruction.stream_overlay import (
    MeshcatSkeletonOverlay,
    cylinder_transform_between,
)


class CylinderTransformTest(unittest.TestCase):
    def test_axis_aligned_segment_has_midpoint_translation_and_scaled_y_column(self):
        p0 = np.zeros(3)
        p1 = np.array([0.0, 0.0, 0.1])

        transform = cylinder_transform_between(p0, p1)

        # A +Z segment needs the cylinder's +Y axis mapped to +Z, scaled to 0.1.
        self.assertTrue(np.allclose(transform[:3, 1], np.array([0.0, 0.0, 0.1])))
        # Translation sits at the midpoint.
        self.assertTrue(np.allclose(transform[:3, 3], np.array([0.0, 0.0, 0.05])))
        # The cylinder endpoints reconstruct p0 and p1 (local y in [-0.5, 0.5]).
        self.assertTrue(np.allclose(transform[:3, 3] - 0.5 * transform[:3, 1], p0))
        self.assertTrue(np.allclose(transform[:3, 3] + 0.5 * transform[:3, 1], p1))

    def test_non_axis_aligned_segment_y_column_equals_segment_vector(self):
        p0 = np.array([0.1, 0.2, 0.3])
        p1 = np.array([0.4, 0.1, 0.0])
        delta = p1 - p0

        transform = cylinder_transform_between(p0, p1)

        self.assertTrue(np.allclose(transform[:3, 1], delta))
        self.assertTrue(np.allclose(transform[:3, 3], 0.5 * (p0 + p1)))
        # The remaining basis columns stay unit length and orthogonal to the bone.
        y_axis = delta / np.linalg.norm(delta)
        self.assertTrue(np.allclose(np.linalg.norm(transform[:3, 0]), 1.0))
        self.assertTrue(np.allclose(np.linalg.norm(transform[:3, 2]), 1.0))
        self.assertTrue(np.allclose(transform[:3, 0] @ y_axis, 0.0, atol=1e-8))
        self.assertTrue(np.allclose(transform[:3, 2] @ y_axis, 0.0, atol=1e-8))

    def test_y_aligned_segment_is_identity_rotation_scaled(self):
        # Segment already along +Y (the cylinder's native axis).
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.0, 0.05, 0.0])

        transform = cylinder_transform_between(p0, p1)

        self.assertTrue(np.allclose(transform[:3, 1], np.array([0.0, 0.05, 0.0])))
        self.assertTrue(np.allclose(transform[:3, 0], np.array([1.0, 0.0, 0.0])))
        self.assertTrue(np.allclose(transform[:3, 2], np.array([0.0, 0.0, 1.0])))

    def test_antiparallel_segment_still_spans_endpoints(self):
        # Segment pointing exactly opposite to the +Y base axis (180-degree turn).
        p0 = np.array([0.0, 0.05, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])

        transform = cylinder_transform_between(p0, p1)

        self.assertFalse(np.any(np.isnan(transform)))
        self.assertTrue(np.allclose(transform[:3, 3] - 0.5 * transform[:3, 1], p0))
        self.assertTrue(np.allclose(transform[:3, 3] + 0.5 * transform[:3, 1], p1))

    def test_degenerate_coincident_points_does_not_raise(self):
        p = np.array([0.02, -0.01, 0.005])

        transform = cylinder_transform_between(p, p)

        self.assertFalse(np.any(np.isnan(transform)))
        self.assertEqual(transform.shape, (4, 4))
        # Zero-length Y column, placed at p.
        self.assertTrue(np.allclose(transform[:3, 1], 0.0))
        self.assertTrue(np.allclose(transform[:3, 3], p))


class MeshcatSkeletonOverlayTest(unittest.TestCase):
    def test_build_creates_geometry_once_and_update_only_sets_transforms(self):
        root = FakeNode("human_hand")
        geometry = FakeGeometry()
        transforms = FakeTransforms()
        overlay = MeshcatSkeletonOverlay(root, "right", geometry=geometry, transforms=transforms)

        overlay.build()

        sphere_nodes = [root.children[f"human_hand/point_{i:02d}"] for i in range(NUM_LANDMARKS)]
        bone_nodes = [root.children[f"human_hand/bone_{j:02d}"] for j in range(len(MEDIAPIPE_HAND_CONNECTIONS))]
        self.assertEqual(len(sphere_nodes), NUM_LANDMARKS)
        self.assertEqual(len(bone_nodes), len(MEDIAPIPE_HAND_CONNECTIONS))

        # build() only sets geometry, never transforms.
        for node in sphere_nodes + bone_nodes:
            self.assertEqual(len(node.objects), 1)
            self.assertEqual(len(node.transforms), 0)
        self.assertEqual(len(overlay._spheres), NUM_LANDMARKS)
        self.assertEqual(len(overlay._bones), len(MEDIAPIPE_HAND_CONNECTIONS))

        object_counts_before = {path: len(node.objects) for path, node in root.children.items()}
        landmarks = self._sample_landmarks()

        overlay.update(landmarks)

        # update() only sets transforms; geometry count is unchanged.
        for i, node in enumerate(sphere_nodes):
            self.assertEqual(len(node.objects), 1)
            self.assertEqual(len(node.transforms), 1)
            self.assertEqual(node.transforms[0], transforms.translation_matrix(landmarks[i]))
        for node in bone_nodes:
            self.assertEqual(len(node.objects), 1)
            self.assertEqual(len(node.transforms), 1)
        for path, node in root.children.items():
            self.assertEqual(len(node.objects), object_counts_before[path])

    def test_update_before_build_raises(self):
        overlay = MeshcatSkeletonOverlay(
            FakeNode("human_hand"), "right", geometry=FakeGeometry(), transforms=FakeTransforms()
        )
        with self.assertRaises(RuntimeError):
            overlay.update(np.zeros((NUM_LANDMARKS, 3)))

    def test_update_rejects_wrong_shape(self):
        root = FakeNode("human_hand")
        overlay = MeshcatSkeletonOverlay(root, "right", geometry=FakeGeometry(), transforms=FakeTransforms())
        overlay.build()
        with self.assertRaises(ValueError):
            overlay.update(np.zeros((5, 3)))

    def test_build_is_idempotent(self):
        root = FakeNode("human_hand")
        overlay = MeshcatSkeletonOverlay(root, "right", geometry=FakeGeometry(), transforms=FakeTransforms())

        overlay.build()
        overlay.build()

        self.assertEqual(len(root.children[f"human_hand/point_00"].objects), 1)
        self.assertEqual(len(overlay._spheres), NUM_LANDMARKS)

    @staticmethod
    def _sample_landmarks() -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.uniform(-0.05, 0.15, size=(NUM_LANDMARKS, 3))


class FakeNode:
    def __init__(self, path):
        self.path = path
        self.children = {}
        self.objects = []
        self.transforms = []

    def __getitem__(self, key):
        path = f"{self.path}/{key}" if self.path else key
        if path not in self.children:
            self.children[path] = FakeNode(path)
        return self.children[path]

    def set_object(self, *args):
        self.objects.append(args)

    def set_transform(self, transform):
        self.transforms.append(transform)


class FakeGeometry:
    def Sphere(self, radius):
        return ("sphere", radius)

    def Cylinder(self, height, radius=1.0, radiusTop=None, radiusBottom=None):
        return ("cylinder", height, radius)

    def MeshLambertMaterial(self, color):
        return ("lambert", color)


class FakeTransforms:
    def translation_matrix(self, point):
        return ("translate", tuple(np.asarray(point, dtype=float)))


if __name__ == "__main__":
    unittest.main()
