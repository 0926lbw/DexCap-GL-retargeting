import unittest

import numpy as np


class CoordinateFrameHelpersTest(unittest.TestCase):
    def test_invert_transform_round_trips_points(self):
        from hand_reconstruction.coordinate_frames import (
            apply_transform,
            invert_transform,
            make_transform,
        )

        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform = make_transform(
            rotation=rotation,
            translation=np.array([0.10, -0.20, 0.30]),
        )
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.2, -0.3],
                [-0.4, 0.5, 0.6],
            ]
        )

        world = apply_transform(transform, points)
        local = apply_transform(invert_transform(transform), world)

        np.testing.assert_allclose(local, points, atol=1e-12)

    def test_validate_transform_rejects_bad_shape_and_nonfinite_values(self):
        from hand_reconstruction.coordinate_frames import validate_transform

        with self.assertRaisesRegex(ValueError, "T_W_Hwrist must have shape"):
            validate_transform(np.eye(3), "T_W_Hwrist")

        bad = np.eye(4)
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "T_W_Hwrist must contain only finite"):
            validate_transform(bad, "T_W_Hwrist")


class HandReconstructionFrameTest(unittest.TestCase):
    def test_frame_validates_keypoint_shapes_and_transform(self):
        from hand_reconstruction.solver import HandReconstructionFrame

        points = _sample_keypoints()
        frame = HandReconstructionFrame(
            hand="right",
            joint_angles={"index_mcp_flex": 0.2},
            keypoints_21_in_Hwrist=points,
            keypoints_21_in_W=points + np.array([1.0, 0.0, 0.0]),
            direct_glove_keypoints_21_in_W=points + np.array([2.0, 0.0, 0.0]),
            direct_glove_keypoints_21_in_Hwrist=points + np.array([3.0, 0.0, 0.0]),
            fused_keypoints_21_in_Hwrist=points + np.array([4.0, 0.0, 0.0]),
            fused_keypoints_21_in_W=points + np.array([5.0, 0.0, 0.0]),
            T_W_Hwrist=np.eye(4),
            diagnostics={"roundtrip_error": 0.0},
        )

        self.assertEqual(frame.hand, "right")
        np.testing.assert_allclose(frame.keypoints_21_in_Hwrist, points)
        self.assertIsNot(frame.keypoints_21_in_Hwrist, points)

    def test_frame_rejects_invalid_hand_bad_keypoints_and_bad_transform(self):
        from hand_reconstruction.solver import HandReconstructionFrame

        points = _sample_keypoints()
        kwargs = dict(
            hand="right",
            joint_angles={},
            keypoints_21_in_Hwrist=points,
            keypoints_21_in_W=points,
            direct_glove_keypoints_21_in_W=points,
            direct_glove_keypoints_21_in_Hwrist=points,
            fused_keypoints_21_in_Hwrist=points,
            fused_keypoints_21_in_W=points,
            T_W_Hwrist=np.eye(4),
            diagnostics={},
        )

        with self.assertRaisesRegex(ValueError, "hand must be 'left' or 'right'"):
            HandReconstructionFrame(**{**kwargs, "hand": "center"})

        with self.assertRaisesRegex(ValueError, "keypoints_21_in_W must have shape"):
            HandReconstructionFrame(
                **{**kwargs, "keypoints_21_in_W": np.zeros((5, 3))}
            )

        bad_points = points.copy()
        bad_points[0, 0] = np.inf
        with self.assertRaisesRegex(
            ValueError, "fused_keypoints_21_in_W must contain only finite"
        ):
            HandReconstructionFrame(
                **{**kwargs, "fused_keypoints_21_in_W": bad_points}
            )

        with self.assertRaisesRegex(ValueError, "T_W_Hwrist must have shape"):
            HandReconstructionFrame(**{**kwargs, "T_W_Hwrist": np.eye(3)})

    def test_package_exports_frame_type(self):
        from hand_reconstruction import HandReconstructionFrame

        self.assertEqual(HandReconstructionFrame.__name__, "HandReconstructionFrame")

    def test_solver_frame_export_dict_includes_local_world_and_fused_arrays(self):
        from hand_reconstruction.export import solver_frame_to_dict
        from hand_reconstruction.solver import HandReconstructionFrame

        points = _sample_keypoints()
        frame = HandReconstructionFrame(
            hand="left",
            joint_angles={"thumb_cmc_abd": 0.1},
            keypoints_21_in_Hwrist=points,
            keypoints_21_in_W=points + np.array([1.0, 0.0, 0.0]),
            direct_glove_keypoints_21_in_W=points + np.array([2.0, 0.0, 0.0]),
            direct_glove_keypoints_21_in_Hwrist=points + np.array([3.0, 0.0, 0.0]),
            fused_keypoints_21_in_Hwrist=points + np.array([4.0, 0.0, 0.0]),
            fused_keypoints_21_in_W=points + np.array([5.0, 0.0, 0.0]),
            T_W_Hwrist=np.eye(4),
            diagnostics={"all_finite": True},
        )

        payload = solver_frame_to_dict(frame)

        self.assertEqual(payload["hand"], "left")
        self.assertEqual(payload["joint_angles"], {"thumb_cmc_abd": 0.1})
        self.assertIn("coordinate_convention", payload)
        self.assertEqual(len(payload["keypoints_21_in_Hwrist"]), 21)
        self.assertEqual(len(payload["keypoints_21_in_W"]), 21)
        self.assertEqual(len(payload["fused_keypoints_21_in_Hwrist"]), 21)
        self.assertEqual(len(payload["fused_keypoints_21_in_W"]), 21)
        self.assertEqual(payload["diagnostics"], {"all_finite": True})


class JointAngleSmootherTest(unittest.TestCase):
    def test_smoother_applies_ema_and_max_delta(self):
        from hand_reconstruction.solver import JointAngleSmoother

        smoother = JointAngleSmoother(alpha=0.5, max_delta=0.2)

        first, first_delta = smoother.smooth({"index_mcp_flex": 0.0})
        second, second_delta = smoother.smooth({"index_mcp_flex": 1.0})

        self.assertEqual(first, {"index_mcp_flex": 0.0})
        self.assertAlmostEqual(first_delta, 0.0)
        self.assertAlmostEqual(second["index_mcp_flex"], 0.2)
        self.assertAlmostEqual(second_delta, 0.2)

    def test_smoother_reset_drops_previous_state(self):
        from hand_reconstruction.solver import JointAngleSmoother

        smoother = JointAngleSmoother(alpha=0.5, max_delta=0.1)
        smoother.smooth({"index_mcp_flex": 0.0})
        smoother.reset()
        after_reset, delta = smoother.smooth({"index_mcp_flex": 1.0})

        self.assertEqual(after_reset, {"index_mcp_flex": 1.0})
        self.assertAlmostEqual(delta, 0.0)

    def test_smoother_rejects_invalid_parameters(self):
        from hand_reconstruction.solver import JointAngleSmoother

        with self.assertRaisesRegex(ValueError, "alpha must be in"):
            JointAngleSmoother(alpha=-0.1)
        with self.assertRaisesRegex(ValueError, "alpha must be in"):
            JointAngleSmoother(alpha=1.1)
        with self.assertRaisesRegex(ValueError, "max_delta must be positive"):
            JointAngleSmoother(alpha=0.5, max_delta=0.0)

    def test_package_exports_solver_types(self):
        from hand_reconstruction import HandReconstructionSolver, JointAngleSmoother

        self.assertEqual(HandReconstructionSolver.__name__, "HandReconstructionSolver")
        self.assertEqual(JointAngleSmoother.__name__, "JointAngleSmoother")


class HandReconstructionSolverTest(unittest.TestCase):
    def test_solver_fuses_in_wrist_local_frame_and_returns_world_outputs(self):
        from hand_reconstruction.coordinate_frames import apply_transform, make_transform
        from hand_reconstruction.schema import (
            FINGER_CHAINS,
            INDEX_MCP,
            INDEX_TIP,
            THUMB_CMC,
            THUMB_TIP,
        )
        from hand_reconstruction.solver import HandReconstructionSolver

        human_local = _sample_keypoints(0.0)
        direct_local = human_local.copy()
        direct_local[THUMB_TIP] = human_local[THUMB_CMC] + np.array(
            [0.015, 0.045, -0.010]
        )
        direct_local[INDEX_TIP] = human_local[INDEX_MCP] + np.array(
            [-0.005, 0.055, -0.010]
        )
        transform = make_transform(translation=np.array([0.5, -0.1, 0.25]))
        direct_world = apply_transform(transform, direct_local)
        solver = HandReconstructionSolver(
            hand="right",
            human_model=_FakeHumanModel(human_local),
            retargeter=_FakeRetargeter({"index_mcp_flex": 0.2}),
            glove_pipeline=_FakeGlovePipeline(direct_world),
            T_W_Hwrist=transform,
        )

        frame = solver.reconstruct(np.arange(21, dtype=float))

        np.testing.assert_allclose(frame.keypoints_21_in_Hwrist, human_local)
        np.testing.assert_allclose(frame.direct_glove_keypoints_21_in_W, direct_world)
        np.testing.assert_allclose(frame.direct_glove_keypoints_21_in_Hwrist, direct_local)
        np.testing.assert_allclose(
            frame.fused_keypoints_21_in_Hwrist[THUMB_TIP],
            direct_local[THUMB_TIP],
        )
        np.testing.assert_allclose(
            frame.fused_keypoints_21_in_Hwrist[INDEX_TIP],
            direct_local[INDEX_TIP],
        )
        np.testing.assert_allclose(
            frame.fused_keypoints_21_in_W,
            apply_transform(transform, frame.fused_keypoints_21_in_Hwrist),
        )
        _assert_finger_bone_lengths_preserved(
            self,
            FINGER_CHAINS,
            frame.keypoints_21_in_Hwrist,
            frame.fused_keypoints_21_in_Hwrist,
        )
        self.assertLess(frame.diagnostics["roundtrip_error"], 1e-12)
        self.assertLess(frame.diagnostics["fingertip_error_after_fusion"], 1e-8)
        self.assertGreater(frame.diagnostics["fingertip_error_before_fusion"], 0.0)
        self.assertAlmostEqual(frame.diagnostics["transform_det"], 1.0)

    def test_solver_rejects_bad_glove_q_shape(self):
        from hand_reconstruction.solver import HandReconstructionSolver

        solver = HandReconstructionSolver(
            hand="left",
            human_model=_FakeHumanModel(_sample_keypoints()),
            retargeter=_FakeRetargeter({}),
            glove_pipeline=_FakeGlovePipeline(_sample_keypoints()),
            T_W_Hwrist=np.eye(4),
        )

        with self.assertRaisesRegex(ValueError, "q_glove must have shape"):
            solver.reconstruct(np.zeros(5))

    def test_solver_reports_smoothing_delta(self):
        from hand_reconstruction.solver import HandReconstructionSolver, JointAngleSmoother

        smoother = JointAngleSmoother(alpha=0.5, max_delta=0.1)
        retargeter = _SequenceRetargeter(
            [
                {"index_mcp_flex": 0.0},
                {"index_mcp_flex": 1.0},
            ]
        )
        solver = HandReconstructionSolver(
            hand="right",
            human_model=_FakeHumanModel(_sample_keypoints()),
            retargeter=retargeter,
            glove_pipeline=_FakeGlovePipeline(_sample_keypoints()),
            T_W_Hwrist=np.eye(4),
            smoother=smoother,
        )

        solver.reconstruct(np.zeros(21))
        frame = solver.reconstruct(np.zeros(21))

        self.assertAlmostEqual(frame.joint_angles["index_mcp_flex"], 0.1)
        self.assertAlmostEqual(frame.diagnostics["max_joint_delta"], 0.1)


class _FakeRetargeter:
    def __init__(self, joints):
        self.joints = dict(joints)
        self.last_q = None

    def retarget(self, q):
        self.last_q = np.asarray(q, dtype=float).copy()
        return dict(self.joints)


class _SequenceRetargeter:
    def __init__(self, sequence):
        self.sequence = [dict(item) for item in sequence]
        self.index = 0

    def retarget(self, q):
        value = self.sequence[self.index]
        self.index = min(self.index + 1, len(self.sequence) - 1)
        return dict(value)


class _FakeHumanModel:
    def __init__(self, landmarks):
        self.landmarks = np.asarray(landmarks, dtype=float)
        self.last_joints = None

    def landmarks_from_joints(self, joints):
        self.last_joints = dict(joints)
        return self.landmarks.copy()


class _FakeSkeleton:
    def __init__(self, landmarks):
        self.landmarks = np.asarray(landmarks, dtype=float)

    def to_numpy(self):
        return self.landmarks.copy()


class _FakeGlovePipeline:
    def __init__(self, landmarks):
        self.landmarks = np.asarray(landmarks, dtype=float)
        self.last_q = None

    def reconstruct_direct(self, q):
        self.last_q = np.asarray(q, dtype=float).copy()
        return _FakeSkeleton(self.landmarks)


def _sample_keypoints(offset=0.0):
    points = np.zeros((21, 3), dtype=float)
    for idx in range(21):
        points[idx] = np.array(
            [offset + idx * 0.01, idx * 0.02, -idx * 0.005],
            dtype=float,
        )
    return points


def _assert_finger_bone_lengths_preserved(testcase, chains, human, fused, atol=1e-8):
    for finger, chain in chains.items():
        for parent, child in zip(chain[1:-1], chain[2:]):
            human_length = np.linalg.norm(human[child] - human[parent])
            fused_length = np.linalg.norm(fused[child] - fused[parent])
            testcase.assertAlmostEqual(
                fused_length,
                human_length,
                delta=atol,
                msg=f"{finger} segment {parent}->{child} changed length",
            )


if __name__ == "__main__":
    unittest.main()
