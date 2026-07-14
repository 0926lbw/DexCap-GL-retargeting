import unittest

import numpy as np

from hand_reconstruction.schema import (
    FINGER_CHAINS,
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    NUM_LANDMARKS,
    WRIST,
)


class TipLockingTest(unittest.TestCase):
    def test_reachable_fingertips_reach_direct_landmarks_without_changing_bone_lengths(self):
        from hand_reconstruction.tip_locking import (
            FINGERTIP_INDICES,
            fuse_tip_locked_landmarks,
        )

        human = _human_landmarks()
        direct = human.copy()
        for chain in FINGER_CHAINS.values():
            root = human[chain[1]]
            direct[chain[-1]] = root + np.array([0.010, 0.060, -0.010])

        fused = fuse_tip_locked_landmarks(human, direct)

        self.assertEqual(fused.shape, (NUM_LANDMARKS, 3))
        for tip_idx in FINGERTIP_INDICES:
            np.testing.assert_allclose(fused[tip_idx], direct[tip_idx], atol=1e-8)
        _assert_finger_bone_lengths_preserved(self, human, fused)

    def test_intermediate_points_follow_human_shape_not_direct_links(self):
        from hand_reconstruction.tip_locking import fuse_tip_locked_landmarks

        human = _human_landmarks()
        direct = _direct_landmarks()
        direct[INDEX_TIP] = human[INDEX_MCP] + np.array([0.0, 0.070, 0.0])
        direct[INDEX_PIP] = np.array([10.0, 10.0, 10.0])
        direct[INDEX_DIP] = np.array([11.0, 11.0, 11.0])

        fused = fuse_tip_locked_landmarks(human, direct)

        self.assertFalse(np.allclose(fused[INDEX_PIP], direct[INDEX_PIP]))
        self.assertFalse(np.allclose(fused[INDEX_DIP], direct[INDEX_DIP]))
        self.assertFalse(np.allclose(fused[INDEX_PIP], human[INDEX_PIP]))
        np.testing.assert_allclose(fused[INDEX_TIP], direct[INDEX_TIP], atol=1e-8)
        _assert_finger_bone_lengths_preserved(self, human, fused)

    def test_rejects_bad_shapes_and_nonfinite_values(self):
        from hand_reconstruction.tip_locking import fuse_tip_locked_landmarks

        human = _human_landmarks()
        direct = _direct_landmarks()

        with self.assertRaisesRegex(ValueError, "human_landmarks must have shape"):
            fuse_tip_locked_landmarks(np.zeros((5, 3)), direct)

        direct_with_nan = direct.copy()
        direct_with_nan[INDEX_TIP, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "direct_landmarks must contain only finite"):
            fuse_tip_locked_landmarks(human, direct_with_nan)

    def test_rejects_nonfinite_or_nonpositive_eps(self):
        from hand_reconstruction.tip_locking import fuse_tip_locked_landmarks

        human = _human_landmarks()
        direct = _direct_landmarks()

        for eps in (-1.0, 0.0, np.nan, np.inf):
            with self.subTest(eps=eps):
                with self.assertRaisesRegex(
                    ValueError, "eps must be a finite positive value"
                ):
                    fuse_tip_locked_landmarks(human, direct, eps=eps)

    def test_unreachable_tip_target_preserves_lengths_instead_of_scaling(self):
        from hand_reconstruction.tip_locking import fuse_tip_locked_landmarks

        human = _human_landmarks()
        direct = _direct_landmarks()
        root = np.array([0.0, 0.0, 0.0])
        human[INDEX_MCP] = root
        human[INDEX_PIP] = np.array([0.25, 0.0, 0.10])
        human[INDEX_DIP] = np.array([0.75, 0.0, 0.20])
        human[INDEX_TIP] = np.array([1.0, 0.0, 0.0])
        direct[INDEX_MCP] = root
        direct[INDEX_PIP] = np.array([0.0, 0.50, 0.10])
        direct[INDEX_DIP] = np.array([0.0, 1.50, 0.20])
        direct[INDEX_TIP] = np.array([0.0, 2.0, 0.0])

        fused = fuse_tip_locked_landmarks(human, direct)
        max_reach = _finger_length_sum(human, FINGER_CHAINS["index"])

        np.testing.assert_allclose(fused[INDEX_MCP], root, atol=1e-12)
        np.testing.assert_allclose(fused[INDEX_TIP], np.array([0.0, max_reach, 0.0]))
        self.assertLess(
            np.linalg.norm(fused[INDEX_TIP] - root),
            np.linalg.norm(direct[INDEX_TIP] - root),
        )
        _assert_finger_bone_lengths_preserved(self, human, fused)

    def test_direct_intermediate_points_choose_bend_side(self):
        from hand_reconstruction.tip_locking import fuse_tip_locked_landmarks

        human = _human_landmarks()
        direct = _direct_landmarks()
        root = np.array([0.0, 0.0, 0.0])
        human[INDEX_MCP] = root
        human[INDEX_PIP] = np.array([0.25, 0.0, 0.10])
        human[INDEX_DIP] = np.array([0.75, 0.0, 0.20])
        human[INDEX_TIP] = np.array([1.0, 0.0, 0.0])
        direct[INDEX_PIP] = np.array([0.0, 0.50, -0.25])
        direct[INDEX_DIP] = np.array([0.0, 1.20, -0.30])
        direct[INDEX_TIP] = np.array([0.0, 0.85, 0.0])

        fused = fuse_tip_locked_landmarks(human, direct)

        self.assertLess(fused[INDEX_PIP, 2], 0.0)
        self.assertLess(fused[INDEX_DIP, 2], 0.0)
        np.testing.assert_allclose(fused[INDEX_TIP], direct[INDEX_TIP], atol=1e-8)
        _assert_finger_bone_lengths_preserved(self, human, fused)

    def test_roots_and_non_finger_landmarks_stay_human(self):
        from hand_reconstruction.tip_locking import fuse_tip_locked_landmarks

        human = _human_landmarks()
        direct = _direct_landmarks()
        human[WRIST] = np.array([0.30, 0.40, 0.50])
        direct[WRIST] = np.array([9.0, 9.0, 9.0])
        direct[INDEX_MCP] = np.array([8.0, 8.0, 8.0])
        direct[MIDDLE_MCP] = np.array([7.0, 7.0, 7.0])

        fused = fuse_tip_locked_landmarks(human, direct)

        np.testing.assert_allclose(fused[WRIST], human[WRIST])
        for chain in FINGER_CHAINS.values():
            root_idx = chain[1]
            np.testing.assert_allclose(fused[root_idx], human[root_idx])

    def test_degenerate_human_root_to_tip_uses_linear_fallback(self):
        from hand_reconstruction.tip_locking import fuse_tip_locked_landmarks

        human = _human_landmarks()
        direct = _direct_landmarks()
        root = human[INDEX_MCP].copy()
        human[INDEX_PIP] = root
        human[INDEX_DIP] = root
        human[INDEX_TIP] = root
        direct[INDEX_TIP] = root + np.array([0.0, 0.09, 0.0])

        fused = fuse_tip_locked_landmarks(human, direct)

        np.testing.assert_allclose(fused[INDEX_PIP], root)
        np.testing.assert_allclose(fused[INDEX_DIP], root)
        np.testing.assert_allclose(fused[INDEX_TIP], root)
        self.assertTrue(np.all(np.isfinite(fused)))

    def test_degenerate_target_root_to_tip_uses_linear_collapse(self):
        from hand_reconstruction.tip_locking import fuse_tip_locked_landmarks

        human = _human_landmarks()
        direct = _direct_landmarks()
        root = human[INDEX_MCP].copy()
        direct[INDEX_TIP] = root

        fused = fuse_tip_locked_landmarks(human, direct)

        np.testing.assert_allclose(fused[INDEX_TIP], root, atol=1e-8)
        _assert_finger_bone_lengths_preserved(self, human, fused)
        self.assertTrue(np.all(np.isfinite(fused)))

    def test_package_exports_tip_locking_api(self):
        from hand_reconstruction import FINGERTIP_INDICES, fuse_tip_locked_landmarks

        self.assertIsInstance(FINGERTIP_INDICES, tuple)
        self.assertTrue(callable(fuse_tip_locked_landmarks))

    def test_schema_is_hand_agnostic(self):
        from hand_reconstruction.tip_locking import FINGERTIP_INDICES
        from hand_reconstruction.schema import (
            INDEX_TIP,
            MIDDLE_TIP,
            PINKY_TIP,
            RING_TIP,
            THUMB_TIP,
        )

        self.assertEqual(
            FINGERTIP_INDICES,
            (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP),
        )


def _human_landmarks():
    points = np.zeros((NUM_LANDMARKS, 3), dtype=float)
    x_offsets = {
        "thumb": -0.045,
        "index": -0.020,
        "middle": 0.000,
        "ring": 0.020,
        "pinky": 0.040,
    }
    for finger, chain in FINGER_CHAINS.items():
        x = x_offsets[finger]
        points[chain[1]] = np.array([x, 0.02, 0.00])
        points[chain[2]] = np.array([x + 0.004, 0.05, -0.010])
        points[chain[3]] = np.array([x + 0.008, 0.075, -0.018])
        points[chain[4]] = np.array([x + 0.010, 0.10, -0.020])
    return points


def _direct_landmarks():
    points = np.zeros((NUM_LANDMARKS, 3), dtype=float)
    x_offsets = {
        "thumb": -0.040,
        "index": -0.015,
        "middle": 0.005,
        "ring": 0.025,
        "pinky": 0.045,
    }
    for finger, chain in FINGER_CHAINS.items():
        x = x_offsets[finger]
        points[chain[1]] = np.array([x, 0.018, 0.015])
        points[chain[2]] = np.array([x, 0.052, 0.010])
        points[chain[3]] = np.array([x, 0.078, 0.005])
        points[chain[4]] = np.array([x, 0.112, 0.000])
    return points


def _assert_finger_bone_lengths_preserved(testcase, human, fused, atol=1e-8):
    for finger, chain in FINGER_CHAINS.items():
        for parent, child in zip(chain[1:-1], chain[2:]):
            human_length = np.linalg.norm(human[child] - human[parent])
            fused_length = np.linalg.norm(fused[child] - fused[parent])
            testcase.assertAlmostEqual(
                fused_length,
                human_length,
                delta=atol,
                msg=f"{finger} segment {parent}->{child} changed length",
            )


def _finger_length_sum(points, chain):
    return float(
        sum(
            np.linalg.norm(points[child] - points[parent])
            for parent, child in zip(chain[1:-1], chain[2:])
        )
    )


if __name__ == "__main__":
    unittest.main()
