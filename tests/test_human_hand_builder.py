import unittest

import numpy as np


def _has_pinocchio() -> bool:
    try:
        import pinocchio  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_has_pinocchio(), "pinocchio not installed")
class HumanHandBuilderTest(unittest.TestCase):
    def test_loads_with_expected_dof_and_landmark_frames(self):
        from hand_reconstruction.human_hand_builder import (
            LANDMARK_FRAME_NAMES,
            load_human_hand,
        )

        model = load_human_hand(hand="right")
        self.assertEqual(model.nq, 20)
        self.assertEqual(len(model.frame_ids), 21)
        self.assertEqual(len(LANDMARK_FRAME_NAMES), 21)
        # every URDF revolute joint is addressable by name
        self.assertCountEqual(
            model.joint_q_idx.keys(),
            {
                "index_mcp_flex", "index_mcp_abd", "index_pip", "index_dip",
                "middle_mcp_flex", "middle_mcp_abd", "middle_pip", "middle_dip",
                "ring_mcp_flex", "ring_mcp_abd", "ring_pip", "ring_dip",
                "pinky_mcp_flex", "pinky_mcp_abd", "pinky_pip", "pinky_dip",
                "thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip",
            },
        )

    def test_rest_pose_matches_mediapipe_template_for_four_fingers(self):
        # The default parameter table is derived from _mediapipe_open_hand_template,
        # so the right-hand rest pose must reproduce it (thumb excepted: it is set
        # to an opposed pose, not the flat template).
        from hand_reconstruction.human_hand_builder import load_human_hand
        from hand_reconstruction.human_hand_model import HumanHandSkeleton

        model = load_human_hand(hand="right")
        rest = model.landmarks_from_q(np.zeros(model.nq))
        template = HumanHandSkeleton.default("right").to_numpy()

        non_thumb = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        for i in non_thumb:
            self.assertTrue(
                np.allclose(rest[i], template[i], atol=3e-3),
                f"landmark {i}: rest {rest[i]} != template {template[i]}",
            )

    def test_rest_pose_is_finite_and_human_scale(self):
        from hand_reconstruction.human_hand_builder import load_human_hand

        rest = load_human_hand(hand="right").landmarks_from_q(np.zeros(20))
        self.assertEqual(rest.shape, (21, 3))
        self.assertTrue(np.all(np.isfinite(rest)))
        self.assertTrue(np.allclose(rest[0], np.zeros(3)))  # wrist at origin
        # fingertips reach 0.10..0.20 m from the wrist
        for tip in (4, 8, 12, 16, 20):
            d = float(np.linalg.norm(rest[tip]))
            self.assertTrue(0.10 < d < 0.20, f"tip {tip} dist {d} out of range")
        # bone segment lengths stay in a plausible human range
        from hand_reconstruction.schema import MEDIAPIPE_HAND_CONNECTIONS
        for a, b in MEDIAPIPE_HAND_CONNECTIONS:
            seg = float(np.linalg.norm(rest[b] - rest[a]))
            self.assertTrue(0.010 < seg < 0.090, f"bone {a}-{b} len {seg}")

    def test_flexion_curls_fingertip_into_palm(self):
        # Positive flexion about the -X axis must move the tip toward -Z (ventral).
        from hand_reconstruction.human_hand_builder import load_human_hand

        model = load_human_hand(hand="right")
        rest = model.landmarks_from_q(np.zeros(model.nq))
        flexed = model.landmarks_from_joints({"index_pip": 1.0})
        self.assertLess(flexed[8, 2], rest[8, 2] - 0.01)  # INDEX_TIP.z dropped

    def test_abduction_moves_fingertip_sideways(self):
        from hand_reconstruction.human_hand_builder import load_human_hand

        model = load_human_hand(hand="right")
        rest = model.landmarks_from_q(np.zeros(model.nq))
        abd = model.landmarks_from_joints({"index_mcp_abd": 0.5})
        # abduction changes the tip X position (radial spread)
        self.assertFalse(np.allclose(abd[8, 0], rest[8, 0], atol=1e-4))

    def test_left_hand_is_right_hand_mirror(self):
        from hand_reconstruction.human_hand_builder import load_human_hand

        left = load_human_hand(hand="left").landmarks_from_q(np.zeros(20))
        right = load_human_hand(hand="right").landmarks_from_q(np.zeros(20))
        # MCP attachment X is mirrored; Y/Z unchanged for the four fingers.
        for mcp in (5, 9, 13, 17):
            self.assertAlmostEqual(left[mcp, 0], -right[mcp, 0], places=6)
            self.assertAlmostEqual(left[mcp, 1], right[mcp, 1], places=6)


class RigidFitTest(unittest.TestCase):
    def test_recovers_known_proper_transform(self):
        from hand_reconstruction.human_hand_builder import rigid_fit

        rng = np.random.default_rng(42)
        P = rng.uniform(-0.1, 0.1, size=(21, 3))
        # a known proper rotation (180° about a tilted axis) + translation
        axis = np.array([0.3, -0.2, 0.9]); axis /= np.linalg.norm(axis)
        theta = 1.234
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        t = np.array([0.05, -0.03, 0.02])
        Q = (R @ P.T).T + t

        T = rigid_fit(P, Q)
        np.testing.assert_allclose(T[:3, :3], R, atol=1e-9)
        np.testing.assert_allclose(T[:3, 3], t, atol=1e-9)
        # forward application maps P onto Q exactly
        mapped = (T[:3, :3] @ P.T).T + T[:3, 3]
        np.testing.assert_allclose(mapped, Q, atol=1e-9)

    def test_rejects_mismatched_shapes(self):
        from hand_reconstruction.human_hand_builder import rigid_fit

        with self.assertRaises(ValueError):
            rigid_fit(np.zeros((5, 3)), np.zeros((6, 3)))


if __name__ == "__main__":
    unittest.main()
