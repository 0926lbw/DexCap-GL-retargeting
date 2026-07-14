import unittest

import numpy as np

from hand_reconstruction.human_hand_model import HumanHandSkeleton
from hand_reconstruction.visualize_meshcat import (
    coordinate_axes,
    fingertip_labels,
    render_skeleton_nodes,
)


class MeshcatVisualizationTest(unittest.TestCase):
    def test_coordinate_axes_describe_hand_wrist_convention(self):
        axes = coordinate_axes()

        expected = (
            ("x_axis", np.array([0.0, 0.0, 0.0]), np.array([0.045, 0.0, 0.0]), 0xE63946, "+X thumb side"),
            ("y_axis", np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.060, 0.0]), 0x2A9D8F, "+Y fingers"),
            ("z_axis", np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.045]), 0x457B9D, "+Z palm normal"),
        )
        self.assertEqual(len(axes), len(expected))
        for actual, expected_axis in zip(axes, expected):
            self.assertEqual(actual[0], expected_axis[0])
            self.assertTrue(np.allclose(actual[1], expected_axis[1]))
            self.assertTrue(np.allclose(actual[2], expected_axis[2]))
            self.assertEqual(actual[3], expected_axis[3])
            self.assertEqual(actual[4], expected_axis[4])

    def test_fingertip_labels_name_each_finger_tip_landmark(self):
        self.assertEqual(
            fingertip_labels(),
            (
                ("thumb", 4),
                ("index", 8),
                ("middle", 12),
                ("ring", 16),
                ("pinky", 20),
            ),
        )


    def test_render_skeleton_nodes_does_not_draw_fake_bone_cylinders(self):
        root = FakeNode("human_hand")
        geometry = FakeGeometry()
        transforms = FakeTransforms()
        skeleton = HumanHandSkeleton.default("right")

        render_skeleton_nodes(root, skeleton, geometry, transforms)

        self.assertFalse(any("/bone_" in path for path in root.children))

    def test_render_skeleton_nodes_draws_axes_and_fingertip_labels(self):
        root = FakeNode("human_hand")
        geometry = FakeGeometry()
        transforms = FakeTransforms()
        skeleton = HumanHandSkeleton.default("right")

        render_skeleton_nodes(root, skeleton, geometry, transforms)

        self.assertIn("human_hand/frame/x_axis", root.children)
        self.assertIn("human_hand/frame/y_axis", root.children)
        self.assertIn("human_hand/frame/z_axis", root.children)
        self.assertEqual(root.children["human_hand/frame/x_axis_label"].objects[0][0], ("text", "+X thumb side"))
        self.assertEqual(root.children["human_hand/frame/y_axis_label"].objects[0][0], ("text", "+Y fingers"))
        self.assertEqual(root.children["human_hand/frame/z_axis_label"].objects[0][0], ("text", "+Z palm normal"))

        for label in ("thumb", "index", "middle", "ring", "pinky"):
            node = root.children[f"human_hand/label_{label}"]
            self.assertEqual(node.objects[0][0], ("text", label))

    def test_render_skeleton_nodes_falls_back_when_meshcat_text_is_unavailable(self):
        root = FakeNode("human_hand")
        geometry = FakeGeometryWithoutText()
        transforms = FakeTransforms()
        skeleton = HumanHandSkeleton.default("right")

        render_skeleton_nodes(root, skeleton, geometry, transforms)

        thumb_label = root.children["human_hand/label_thumb"]
        self.assertEqual(thumb_label.objects[0][0], ("sphere", 0.003))
        self.assertEqual(thumb_label.objects[0][1], ("lambert", 0xF4A261))



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

    def Line(self, geometry, material):
        return ("line", geometry, material)

    def PointsGeometry(self, segment):
        return ("points", tuple(map(tuple, segment)))

    def MeshLambertMaterial(self, color):
        return ("lambert", color)

    def MeshBasicMaterial(self, color):
        return ("basic", color)

    def Text(self, text):
        return ("text", text)


class FakeGeometryWithoutText(FakeGeometry):
    Text = None

    def __getattribute__(self, name):
        if name == "Text":
            raise AttributeError(name)
        return super().__getattribute__(name)


class FakeTransforms:
    def translation_matrix(self, point):
        return ("translate", tuple(np.asarray(point, dtype=float)))


if __name__ == "__main__":
    unittest.main()
