import unittest
from unittest import mock

import numpy as np

from hand_reconstruction.human_hand_model import HumanHandSkeleton
from scripts import reconstruct_hand_frame


class ReconstructHandFrameCliTest(unittest.TestCase):
    def test_meshcat_flag_displays_reconstructed_skeleton(self):
        skeleton = HumanHandSkeleton.default("right")
        test_args = [
            "reconstruct_hand_frame.py",
            "--hand",
            "right",
            "--q",
            ",".join(["0"] * 21),
            "--meshcat",
        ]

        with mock.patch.object(reconstruct_hand_frame.sys, "argv", test_args):
            with mock.patch.object(
                reconstruct_hand_frame.HandReconstructionPipeline,
                "reconstruct",
                return_value=skeleton,
            ):
                with mock.patch.object(
                    reconstruct_hand_frame, "MeshcatHandSkeletonViewer"
                ) as viewer_cls:
                    with mock.patch("builtins.print") as print_mock:
                        exit_code = reconstruct_hand_frame.main()

        self.assertEqual(exit_code, 0)
        viewer_cls.assert_called_once()
        viewer_cls.return_value.display.assert_called_once_with(skeleton)
        print_mock.assert_not_called()


    def test_meshcat_wait_keeps_process_alive_until_keyboard_interrupt(self):
        skeleton = HumanHandSkeleton.default("right")
        test_args = [
            "reconstruct_hand_frame.py",
            "--hand",
            "right",
            "--q",
            ",".join(["0"] * 21),
            "--meshcat",
            "--meshcat-wait",
        ]

        with mock.patch.object(reconstruct_hand_frame.sys, "argv", test_args):
            with mock.patch.object(
                reconstruct_hand_frame.HandReconstructionPipeline,
                "reconstruct",
                return_value=skeleton,
            ):
                with mock.patch.object(reconstruct_hand_frame, "MeshcatHandSkeletonViewer"):
                    with mock.patch.object(
                        reconstruct_hand_frame, "_wait_for_meshcat", side_effect=KeyboardInterrupt
                    ) as wait_mock:
                        exit_code = reconstruct_hand_frame.main()

        self.assertEqual(exit_code, 0)
        wait_mock.assert_called_once()

    def test_parse_q_requires_21_values(self):
        parsed = reconstruct_hand_frame._parse_q(",".join(["0.1"] * 21))

        self.assertEqual(parsed.shape, (21,))
        self.assertTrue(np.allclose(parsed, 0.1))


if __name__ == "__main__":
    unittest.main()
