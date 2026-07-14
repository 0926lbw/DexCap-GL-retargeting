import unittest

import numpy as np

from hand_reconstruction.retargeting import (
    DISTAL_TO_PIP_RATIO,
    FINGER_DIP_PIP_RATIO,
    GLOVE_DOF,
    HUMAN_JOINT_NAMES,
    LEFT_RETARGET_TABLE,
    MCP_ABD_MIN_DAMPING,
    RETARGET_LIMITS,
    RETARGET_TABLE,
    GloveToHumanRetargeter,
)


class RetargetingTest(unittest.TestCase):
    def _glove_with(self, idx_values):
        q = np.zeros(GLOVE_DOF)
        for idx, val in idx_values.items():
            q[idx] = val
        return q

    def test_retarget_returns_named_joints_for_known_vector(self):
        # index finger: abd=0.1 (@5), flex=0.2 (@6), PIP=0.3 (@7), DIP=0.4 (@8)
        q = self._glove_with({5: 0.1, 6: 0.2, 7: 0.3, 8: 0.4})
        out = GloveToHumanRetargeter("right").retarget(q)

        self.assertEqual(set(out.keys()), set(HUMAN_JOINT_NAMES))
        self.assertAlmostEqual(out["index_mcp_flex"], 0.2)
        self.assertAlmostEqual(out["index_pip"], 0.3)
        self.assertAlmostEqual(out["index_dip"], 0.3 * FINGER_DIP_PIP_RATIO)
        self.assertLess(out["index_mcp_abd"], 0.1)

    def test_uses_real_pip_encoder_not_dip(self):
        # Set PIP=0.3 (@7) and DIP=0.4 (@8); index_pip must equal 0.3, not 2/3*DIP.
        q = self._glove_with({7: 0.3, 8: 0.4})
        out = GloveToHumanRetargeter("right").retarget(q)
        self.assertAlmostEqual(out["index_pip"], 0.3)
        self.assertNotAlmostEqual(out["index_pip"], 0.4 * (2.0 / 3.0))

    def test_dip_is_coupled_from_final_pip_estimate(self):
        low_dip = GloveToHumanRetargeter("right").retarget(
            self._glove_with({7: 0.8, 8: 0.1})
        )
        high_dip = GloveToHumanRetargeter("right").retarget(
            self._glove_with({7: 0.8, 8: 1.2})
        )

        self.assertAlmostEqual(low_dip["index_dip"], 0.8 * FINGER_DIP_PIP_RATIO)
        self.assertAlmostEqual(high_dip["index_dip"], low_dip["index_dip"])

    def test_left_hand_negative_mcp_flex_is_positive_human_curl(self):
        out = GloveToHumanRetargeter("left").retarget(
            self._glove_with({6: -0.7})
        )

        self.assertAlmostEqual(out["index_mcp_flex"], 0.7)

    def test_distal_curl_channel_can_drive_pip_before_dip_coupling(self):
        out = GloveToHumanRetargeter("left").retarget(
            self._glove_with({7: 0.2, 8: -1.2})
        )

        expected_pip = 1.2 * DISTAL_TO_PIP_RATIO
        self.assertAlmostEqual(out["index_pip"], expected_pip)
        self.assertAlmostEqual(out["index_dip"], expected_pip * FINGER_DIP_PIP_RATIO)

    def test_mcp_abduction_damps_as_finger_flexes(self):
        straight = GloveToHumanRetargeter("right").retarget(
            self._glove_with({5: 0.2})
        )
        curled = GloveToHumanRetargeter("right").retarget(
            self._glove_with({5: 0.2, 6: RETARGET_LIMITS["mcp_flex"][1]})
        )

        self.assertAlmostEqual(straight["index_mcp_abd"], 0.2)
        self.assertAlmostEqual(curled["index_mcp_abd"], 0.2 * MCP_ABD_MIN_DAMPING)

    def test_joint_limits_clamp_unphysical_finger_angles(self):
        out = GloveToHumanRetargeter("right").retarget(
            self._glove_with({6: 5.0, 7: 5.0})
        )

        self.assertAlmostEqual(out["index_mcp_flex"], RETARGET_LIMITS["mcp_flex"][1])
        self.assertAlmostEqual(out["index_pip"], RETARGET_LIMITS["pip"][1])
        self.assertAlmostEqual(
            out["index_dip"],
            min(RETARGET_LIMITS["dip"][1], RETARGET_LIMITS["pip"][1] * FINGER_DIP_PIP_RATIO),
        )

        out = GloveToHumanRetargeter("right").retarget(
            self._glove_with({6: -1.0, 7: -1.0})
        )
        self.assertAlmostEqual(out["index_mcp_flex"], RETARGET_LIMITS["mcp_flex"][0])
        self.assertAlmostEqual(out["index_pip"], RETARGET_LIMITS["pip"][0])
        self.assertAlmostEqual(out["index_dip"], RETARGET_LIMITS["dip"][0])

    def test_ring_and_pinky_abduction_are_negated(self):
        q = self._glove_with({13: 0.2, 17: 0.2})  # ring/pinky abduction encoders
        out = GloveToHumanRetargeter("right").retarget(q)
        self.assertAlmostEqual(out["ring_mcp_abd"], -0.2)
        self.assertAlmostEqual(out["pinky_mcp_abd"], -0.2)

    def test_thumb_drops_distal_encoders_and_couples_mcp(self):
        # thumb glove: q[0]=CMC_abd, q[1]=flex, q[2]=IP, q[3]=DIP, q[4]=tip
        q = self._glove_with({0: 0.1, 1: 0.2, 2: 0.3, 3: 0.9, 4: 0.9})
        out = GloveToHumanRetargeter("right").retarget(q)
        self.assertAlmostEqual(out["thumb_cmc_abd"], 0.1)
        self.assertAlmostEqual(out["thumb_cmc_flex"], 0.2)
        self.assertAlmostEqual(out["thumb_mcp"], 0.1)  # 0.5 * q[1]
        self.assertAlmostEqual(out["thumb_ip"], 0.3)
        # q[3] (DIP) and q[4] (tip) must not drive any human joint beyond the above.
        self.assertNotIn("thumb_dip", out)

    def test_thumb_uses_thumb_specific_limits_and_coupling(self):
        out = GloveToHumanRetargeter("right").retarget(
            self._glove_with({0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0})
        )

        self.assertAlmostEqual(out["thumb_cmc_abd"], RETARGET_LIMITS["thumb_cmc_abd"][1])
        self.assertAlmostEqual(out["thumb_cmc_flex"], RETARGET_LIMITS["thumb_cmc_flex"][1])
        self.assertAlmostEqual(out["thumb_mcp"], RETARGET_LIMITS["thumb_cmc_flex"][1] * 0.5)
        self.assertAlmostEqual(out["thumb_ip"], RETARGET_LIMITS["thumb_ip"][1])
        self.assertNotIn("thumb_dip", out)

    def test_q_offset_is_subtracted_before_mapping(self):
        offset = self._glove_with({6: 0.2})
        q = self._glove_with({6: 0.5})
        out = GloveToHumanRetargeter("right", q_offset=offset).retarget(q)
        self.assertAlmostEqual(out["index_mcp_flex"], 0.3)

    def test_hand_specific_table_matches_glove_sign_convention(self):
        q = self._glove_with({6: 0.2})
        left = GloveToHumanRetargeter("left").retarget(q)
        right = GloveToHumanRetargeter("right").retarget(q)
        self.assertAlmostEqual(left["index_mcp_flex"], 0.0)
        self.assertAlmostEqual(right["index_mcp_flex"], 0.2)

    def test_table_has_twenty_entries_covering_all_dof(self):
        self.assertEqual(len(RETARGET_TABLE), 20)
        self.assertEqual(len(LEFT_RETARGET_TABLE), 20)
        self.assertEqual(len(set(HUMAN_JOINT_NAMES)), 20)


if __name__ == "__main__":
    unittest.main()
