import unittest

import numpy as np

from hand_reconstruction.glove_observation import (
    GloveLinkObservation,
    SkeletonInitializer,
    direct_skeleton_from_observation,
    finger_link_names,
)
from hand_reconstruction.human_hand_model import HumanHandSkeleton
from hand_reconstruction.coordinate_frames import apply_transform
from hand_reconstruction.schema import (
    FINGER_CHAINS,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    NUM_LANDMARKS,
    PINKY_MCP,
    RING_MCP,
    THUMB_CMC,
    THUMB_TIP,
    WRIST,
)


class GloveObservationTest(unittest.TestCase):
    def test_finger_link_names_match_dexglove_urdf(self):
        self.assertEqual(
            finger_link_names("right")["thumb"],
            (
                "glove_link_r_1_1",
                "glove_link_r_1_2",
                "glove_link_r_1_3",
                "glove_link_r_1_4",
                "glove_link_r_1_5",
            ),
        )
        self.assertEqual(
            finger_link_names("left")["index"],
            (
                "glove_link_l_2_1",
                "glove_link_l_2_2",
                "glove_link_l_2_3",
                "glove_link_l_2_4",
            ),
        )

    def test_initializer_uses_anatomical_template_instead_of_direct_link_mapping(self):
        names = finger_link_names("right")
        base_position = np.array([0.1, 0.2, 0.3])
        link_positions = {"base_link": base_position}
        for finger_idx, finger in enumerate(names):
            for link_idx, link_name in enumerate(names[finger]):
                link_positions[link_name] = np.array(
                    [2.0 + finger_idx, -3.0 - link_idx, 4.0 + link_idx]
                )
        observation = GloveLinkObservation(
            hand="right",
            base_position=base_position,
            link_positions=link_positions,
        )

        skeleton = SkeletonInitializer().from_observation(observation)
        points = skeleton.to_numpy()
        template = HumanHandSkeleton.default("right").to_numpy() + base_position

        self.assertEqual(points.shape, (NUM_LANDMARKS, 3))
        self.assertTrue(np.allclose(points[WRIST], base_position))
        self.assertTrue(np.allclose(points[INDEX_MCP], template[INDEX_MCP]))
        self.assertFalse(
            np.allclose(points[INDEX_TIP], link_positions["glove_link_r_2_4"])
        )
        self.assertGreater(points[INDEX_MCP, 0], points[MIDDLE_MCP, 0])
        self.assertGreater(points[MIDDLE_MCP, 0], points[RING_MCP, 0])
        self.assertGreater(points[RING_MCP, 0], points[PINKY_MCP, 0])

    def test_initializer_separates_wrist_local_points_from_world_points(self):
        names = finger_link_names("right")
        base_position = np.array([0.1, 0.2, 0.3])
        link_positions = {"base_link": base_position}
        for finger in names:
            for link_idx, link_name in enumerate(names[finger]):
                link_positions[link_name] = np.array(
                    [0.2 + link_idx * 0.01, 0.3 + link_idx * 0.02, 0.4]
                )
        observation = GloveLinkObservation(
            hand="right",
            base_position=base_position,
            link_positions=link_positions,
        )

        skeleton = SkeletonInitializer().from_observation(observation)

        keypoints_h = skeleton.keypoints_wrist()
        keypoints_w = skeleton.keypoints_world()
        transform = skeleton.transform_world_from_wrist()

        self.assertTrue(np.allclose(keypoints_h[WRIST], np.zeros(3)))
        self.assertTrue(np.allclose(transform[:3, :3], np.eye(3)))
        self.assertTrue(np.allclose(transform[:3, 3], base_position))
        self.assertTrue(np.allclose(keypoints_w[WRIST], base_position))
        self.assertTrue(np.allclose(keypoints_w, apply_transform(transform, keypoints_h)))

    def test_initializer_preserves_template_bone_lengths_under_observations(self):
        names = finger_link_names("left")
        base_position = np.array([-0.2, 0.5, 0.1])
        link_positions = {"base_link": base_position}
        for finger_idx, finger in enumerate(names):
            for link_idx, link_name in enumerate(names[finger]):
                link_positions[link_name] = np.array(
                    [
                        -1.0 - finger_idx * 0.3,
                        2.0 + link_idx * 0.9,
                        -0.4 + link_idx,
                    ]
                )
        observation = GloveLinkObservation(
            hand="left",
            base_position=base_position,
            link_positions=link_positions,
        )

        skeleton = SkeletonInitializer().from_observation(observation)
        expected_lengths = HumanHandSkeleton.default("left").finger_bone_lengths()

        for finger, chain in FINGER_CHAINS.items():
            actual = []
            points = skeleton.to_numpy()
            for start, end in zip(chain[:-1], chain[1:]):
                actual.append(float(np.linalg.norm(points[end] - points[start])))
            self.assertTrue(np.allclose(actual, expected_lengths[finger]))

    def test_direct_skeleton_maps_landmarks_to_observed_link_positions(self):
        names = finger_link_names("right")
        base_position = np.array([0.1, 0.2, 0.3])
        link_positions = {"base_link": base_position}
        for finger_idx, finger in enumerate(names):
            for link_idx, link_name in enumerate(names[finger]):
                link_positions[link_name] = np.array(
                    [2.0 + finger_idx, -3.0 - link_idx, 4.0 + link_idx]
                )
        observation = GloveLinkObservation(
            hand="right",
            base_position=base_position,
            link_positions=link_positions,
        )

        skeleton = direct_skeleton_from_observation(observation)
        points = skeleton.to_numpy()

        self.assertEqual(points.shape, (NUM_LANDMARKS, 3))
        # Wrist is the glove base frame; fingers map 1:1 to observed links.
        self.assertTrue(np.allclose(points[WRIST], base_position))
        self.assertTrue(np.allclose(points[INDEX_MCP], link_positions["glove_link_r_2_1"]))
        self.assertTrue(np.allclose(points[INDEX_TIP], link_positions["glove_link_r_2_4"]))
        # Thumb has 5 links; the last 4 feed CMC/MCP/IP/TIP.
        self.assertTrue(np.allclose(points[THUMB_CMC], link_positions["glove_link_r_1_2"]))
        self.assertTrue(np.allclose(points[THUMB_TIP], link_positions["glove_link_r_1_5"]))

    def test_direct_skeleton_is_finite_for_neutral_glove_urdf(self):
        from hand_reconstruction.pipeline import HandReconstructionPipeline

        for hand, urdf in (
            ("left", "DexCap_v4/DexGlove_L_v4/urdf/DexGlove_L_v4.urdf"),
            ("right", "DexCap_v4/DexGlove_R_v4/urdf/DexGlove_R_v4.urdf"),
        ):
            pipeline = HandReconstructionPipeline(urdf, hand)
            skeleton = pipeline.reconstruct_direct(
                np.zeros(pipeline.observer.robot.model.nq)
            )
            points = skeleton.to_numpy()
            self.assertEqual(points.shape, (NUM_LANDMARKS, 3), hand)
            self.assertTrue(np.all(np.isfinite(points)), hand)
            self.assertTrue(np.allclose(points[WRIST], np.zeros(3)), hand)


if __name__ == "__main__":
    unittest.main()
