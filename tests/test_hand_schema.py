import unittest

import numpy as np

from hand_reconstruction import schema
from hand_reconstruction import visualize_meshcat
from hand_reconstruction.human_hand_model import HumanHandSkeleton
from hand_reconstruction.schema import (
    FINGER_CHAINS,
    INDEX_MCP,
    LANDMARK_NAMES,
    MIDDLE_MCP,
    MIDDLE_TIP,
    NUM_LANDMARKS,
    PINKY_MCP,
    PINKY_TIP,
    RING_MCP,
    THUMB_CMC,
    WRIST,
)


class HandSchemaTest(unittest.TestCase):
    def test_landmark_schema_matches_21_point_hand(self):
        self.assertEqual(NUM_LANDMARKS, 21)
        self.assertEqual(LANDMARK_NAMES[WRIST], "wrist")
        self.assertEqual(
            FINGER_CHAINS["thumb"],
            (0, 1, 2, 3, 4),
        )
        self.assertEqual(
            FINGER_CHAINS["index"],
            (0, 5, 6, 7, 8),
        )
        self.assertEqual(
            FINGER_CHAINS["pinky"],
            (0, 17, 18, 19, 20),
        )


    def test_mediapipe_connections_match_21_point_hand_topology(self):
        connections = getattr(schema, "MEDIAPIPE_HAND_CONNECTIONS", None)

        self.assertIsNotNone(connections)
        self.assertIn((WRIST, THUMB_CMC), connections)
        self.assertIn((WRIST, INDEX_MCP), connections)
        self.assertIn((INDEX_MCP, MIDDLE_MCP), connections)
        self.assertIn((MIDDLE_MCP, RING_MCP), connections)
        self.assertIn((RING_MCP, PINKY_MCP), connections)
        self.assertIn((WRIST, PINKY_MCP), connections)
        self.assertEqual(len(connections), 21)


    def test_meshcat_visualization_uses_mediapipe_connections(self):
        skeleton_edges = getattr(visualize_meshcat, "skeleton_edges", None)

        self.assertIsNotNone(skeleton_edges)
        self.assertEqual(tuple(skeleton_edges()), schema.MEDIAPIPE_HAND_CONNECTIONS)


    def test_default_skeleton_uses_mediapipe_style_open_hand_template(self):
        right = HumanHandSkeleton.default("right").to_numpy()
        left = HumanHandSkeleton.default("left").to_numpy()

        self.assertTrue(np.allclose(right[:, 2], np.zeros(NUM_LANDMARKS)))
        self.assertTrue(np.allclose(right[THUMB_CMC], [0.035, 0.028, 0.000]))
        self.assertTrue(np.allclose(right[INDEX_MCP], [0.030, 0.075, 0.000]))
        self.assertTrue(np.allclose(right[MIDDLE_TIP], [0.006, 0.198, 0.000]))
        self.assertTrue(np.allclose(right[PINKY_TIP], [-0.061, 0.146, 0.000]))
        self.assertTrue(np.allclose(right[:, 0], -left[:, 0]))
        self.assertTrue(np.allclose(right[:, 1:], left[:, 1:]))

    def test_default_skeleton_is_21_by_3_and_finite(self):
        skeleton = HumanHandSkeleton.default("right")

        points = skeleton.to_numpy()

        self.assertEqual(points.shape, (21, 3))
        self.assertTrue(np.all(np.isfinite(points)))
        self.assertTrue(np.allclose(points[WRIST], np.zeros(3)))

    def test_default_skeleton_exposes_wrist_and_world_coordinate_fields(self):
        skeleton = HumanHandSkeleton.default("right")

        keypoints_h = skeleton.keypoints_wrist()
        keypoints_w = skeleton.keypoints_world()
        transform = skeleton.transform_world_from_wrist()

        self.assertEqual(keypoints_h.shape, (NUM_LANDMARKS, 3))
        self.assertEqual(keypoints_w.shape, (NUM_LANDMARKS, 3))
        self.assertTrue(np.allclose(keypoints_h, keypoints_w))
        self.assertTrue(np.allclose(transform, np.eye(4)))
        self.assertEqual(skeleton.coordinate_convention["origin"], "wrist")
        self.assertEqual(skeleton.coordinate_convention["x_axis"], "thumb_side")
        self.assertEqual(skeleton.coordinate_convention["y_axis"], "fingers")
        self.assertEqual(skeleton.coordinate_convention["z_axis"], "palm_normal")

    def test_finger_bone_lengths_are_positive(self):
        skeleton = HumanHandSkeleton.default("right")

        lengths = skeleton.finger_bone_lengths()

        self.assertEqual(set(lengths), set(FINGER_CHAINS))
        for finger_lengths in lengths.values():
            self.assertTrue(np.all(np.asarray(finger_lengths) > 0.0))

    def test_default_skeleton_has_anatomical_palm_layout(self):
        right = HumanHandSkeleton.default("right").to_numpy()
        left = HumanHandSkeleton.default("left").to_numpy()

        self.assertGreater(right[INDEX_MCP, 0], right[MIDDLE_MCP, 0])
        self.assertGreater(right[MIDDLE_MCP, 0], right[RING_MCP, 0])
        self.assertGreater(right[RING_MCP, 0], right[PINKY_MCP, 0])
        self.assertGreater(right[MIDDLE_MCP, 1], right[PINKY_MCP, 1])
        self.assertGreater(right[THUMB_CMC, 0], right[INDEX_MCP, 0])
        self.assertLess(right[THUMB_CMC, 1], right[INDEX_MCP, 1])
        self.assertTrue(np.allclose(right[:, 2], np.zeros(NUM_LANDMARKS)))

        self.assertTrue(np.allclose(right[:, 0], -left[:, 0]))
        self.assertTrue(np.allclose(right[:, 1:], left[:, 1:]))

    def test_default_skeleton_has_human_finger_length_hierarchy(self):
        skeleton = HumanHandSkeleton.default("right")

        lengths = skeleton.finger_bone_lengths()
        phalanx_lengths = {
            finger: float(np.sum(values[1:]))
            for finger, values in lengths.items()
        }

        self.assertGreater(phalanx_lengths["middle"], phalanx_lengths["index"])
        self.assertGreater(phalanx_lengths["index"], phalanx_lengths["pinky"])
        self.assertGreater(phalanx_lengths["ring"], phalanx_lengths["pinky"])


if __name__ == "__main__":
    unittest.main()
