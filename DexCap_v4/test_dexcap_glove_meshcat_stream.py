import struct
import unittest
from unittest import mock

import numpy as np

import dexcap_glove_meshcat_stream as stream


class DexCapGloveMeshcatStreamTest(unittest.TestCase):
    def test_make_listener_packet_sets_left_and_right_glove_q(self):
        left = np.arange(100, 2200, 100, dtype=np.uint16)
        right = np.arange(2100, 0, -100, dtype=np.uint16)
        packet = _packet(mask=0xA000, left=left, right=right)

        listener = stream.parse_listener_packet(packet)

        expected_left = _expected_glove_q(left)
        expected_right = _expected_glove_q(right)
        self.assertTrue(np.allclose(listener.q_l, expected_left))
        self.assertTrue(np.allclose(listener.q_r, expected_right))

    def test_display_q_reuses_last_side_when_mask_omits_that_glove(self):
        left = np.arange(100, 2200, 100, dtype=np.uint16)
        first = stream.parse_listener_packet(_packet(mask=0x8000, left=left))
        previous_right = np.arange(21, dtype=float)

        left_q, right_q = stream.display_q_from_listener(
            first,
            previous_left=np.zeros(21),
            previous_right=previous_right,
        )

        self.assertTrue(np.allclose(left_q, _expected_glove_q(left)))
        self.assertTrue(np.allclose(right_q, previous_right))

    def test_display_human_uses_solver_fused_local_landmarks(self):
        from hand_reconstruction.solver import HandReconstructionFrame

        left_fused_local = _sample_landmarks(0.0)
        right_fused_local = _sample_landmarks(1.0)
        display = _make_display_for_solver_test(
            left_frame=HandReconstructionFrame(
                hand="left",
                joint_angles={"unused": 1.0},
                keypoints_21_in_Hwrist=_sample_landmarks(2.0),
                keypoints_21_in_W=_sample_landmarks(3.0),
                direct_glove_keypoints_21_in_W=_sample_landmarks(4.0),
                direct_glove_keypoints_21_in_Hwrist=_sample_landmarks(5.0),
                fused_keypoints_21_in_Hwrist=left_fused_local,
                fused_keypoints_21_in_W=_sample_landmarks(6.0),
                T_W_Hwrist=np.eye(4),
                diagnostics={},
            ),
            right_frame=HandReconstructionFrame(
                hand="right",
                joint_angles={"unused": 2.0},
                keypoints_21_in_Hwrist=_sample_landmarks(7.0),
                keypoints_21_in_W=_sample_landmarks(8.0),
                direct_glove_keypoints_21_in_W=_sample_landmarks(9.0),
                direct_glove_keypoints_21_in_Hwrist=_sample_landmarks(10.0),
                fused_keypoints_21_in_Hwrist=right_fused_local,
                fused_keypoints_21_in_W=_sample_landmarks(11.0),
                T_W_Hwrist=np.eye(4),
                diagnostics={},
            ),
        )

        display.display_human(np.arange(21), np.arange(21, 42))

        expected_left = stream._offset_landmarks_for_display(
            left_fused_local,
            display.human_skeleton_display_offset,
        )
        expected_left = stream._offset_thumb_for_display(
            expected_left,
            display.thumb_display_offset,
            "left",
            left_extra_offset=display.left_thumb_extra_display_offset,
        )
        expected_right = stream._offset_landmarks_for_display(
            right_fused_local,
            display.human_skeleton_display_offset,
        )
        expected_right = stream._offset_thumb_for_display(
            expected_right,
            display.thumb_display_offset,
            "right",
            left_extra_offset=display.left_thumb_extra_display_offset,
        )
        np.testing.assert_allclose(display._left_overlay.updated, expected_left)
        np.testing.assert_allclose(display._right_overlay.updated, expected_right)
        np.testing.assert_allclose(display._left_solver.last_q, np.arange(21))
        np.testing.assert_allclose(display._right_solver.last_q, np.arange(21, 42))
        np.testing.assert_allclose(left_fused_local, _sample_landmarks(0.0))
        np.testing.assert_allclose(right_fused_local, _sample_landmarks(1.0))

    def test_display_human_prints_debug_joint_diagnostics_when_enabled(self):
        from hand_reconstruction.solver import HandReconstructionFrame

        frame = HandReconstructionFrame(
            hand="left",
            joint_angles={
                "index_mcp_abd": 0.1,
                "index_mcp_flex": 0.2,
                "index_pip": 0.4,
                "index_dip": 0.24,
                "thumb_cmc_abd": 0.05,
                "thumb_cmc_flex": 0.3,
                "thumb_mcp": 0.15,
                "thumb_ip": 0.2,
            },
            keypoints_21_in_Hwrist=_sample_landmarks(0.0),
            keypoints_21_in_W=_sample_landmarks(1.0),
            direct_glove_keypoints_21_in_W=_sample_landmarks(2.0),
            direct_glove_keypoints_21_in_Hwrist=_sample_landmarks(3.0),
            fused_keypoints_21_in_Hwrist=_sample_landmarks(4.0),
            fused_keypoints_21_in_W=_sample_landmarks(5.0),
            T_W_Hwrist=np.eye(4),
            diagnostics={
                "fingertip_error_before_fusion": 0.1234,
                "fingertip_error_after_fusion": 0.0,
                "roundtrip_error": 1e-9,
            },
        )
        display = _make_display_for_solver_test(left_frame=frame, right_frame=frame)
        display.debug_human_joints = True
        display.debug_human_joints_interval = 1
        display._human_debug_frame_count = 0

        with mock.patch("builtins.print") as print_mock:
            display.display_human(np.arange(21), np.arange(21, 42))

        printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list)
        self.assertIn("[human-joints left]", printed)
        self.assertIn("tip_err_before=0.1234", printed)
        self.assertIn("index q[5:9]", printed)
        self.assertIn("mcp_flex=0.200", printed)
        self.assertIn("pip=0.400", printed)
        self.assertIn("dip=0.240", printed)

    def test_display_offset_moves_whole_skeleton_without_changing_bone_lengths(self):
        from hand_reconstruction.schema import INDEX_DIP, INDEX_TIP

        landmarks = np.zeros((21, 3), dtype=float)
        landmarks[INDEX_DIP] = np.array([1.0, 2.0, 3.0])
        landmarks[INDEX_TIP] = np.array([1.0, 4.0, 3.0])
        offset = np.array([0.25, -0.5, 0.75])

        moved = stream._offset_landmarks_for_display(landmarks, offset)

        np.testing.assert_allclose(moved, landmarks + offset)
        np.testing.assert_allclose(landmarks[INDEX_TIP], np.array([1.0, 4.0, 3.0]))
        original_len = np.linalg.norm(landmarks[INDEX_TIP] - landmarks[INDEX_DIP])
        moved_len = np.linalg.norm(moved[INDEX_TIP] - moved[INDEX_DIP])
        self.assertAlmostEqual(moved_len, original_len)

    def test_thumb_display_offset_mirrors_x_by_hand_and_keeps_four_fingers(self):
        from hand_reconstruction.schema import INDEX_MCP, THUMB_CMC, THUMB_TIP

        landmarks = _sample_landmarks(0.0)
        thumb_offset = np.array([0.04, 0.01, -0.03])

        left = stream._offset_thumb_for_display(landmarks, thumb_offset, "left")
        right = stream._offset_thumb_for_display(landmarks, thumb_offset, "right")

        np.testing.assert_allclose(left[THUMB_CMC], landmarks[THUMB_CMC] + thumb_offset)
        np.testing.assert_allclose(left[THUMB_TIP], landmarks[THUMB_TIP] + thumb_offset)
        np.testing.assert_allclose(
            right[THUMB_CMC],
            landmarks[THUMB_CMC] + np.array([-0.04, 0.01, -0.03]),
        )
        np.testing.assert_allclose(
            right[THUMB_TIP],
            landmarks[THUMB_TIP] + np.array([-0.04, 0.01, -0.03]),
        )
        np.testing.assert_allclose(left[INDEX_MCP], landmarks[INDEX_MCP])
        np.testing.assert_allclose(right[INDEX_MCP], landmarks[INDEX_MCP])
        np.testing.assert_allclose(landmarks[THUMB_TIP], _sample_landmarks(0.0)[THUMB_TIP])

    def test_left_thumb_extra_display_offset_only_moves_left_thumb_inward(self):
        from hand_reconstruction.schema import INDEX_MCP, THUMB_CMC, THUMB_TIP

        landmarks = _sample_landmarks(0.0)
        thumb_offset = np.array([0.04, 0.01, -0.03])
        left_extra = np.array([0.006, 0.0, 0.0])

        left = stream._offset_thumb_for_display(
            landmarks,
            thumb_offset,
            "left",
            left_extra_offset=left_extra,
        )
        right = stream._offset_thumb_for_display(
            landmarks,
            thumb_offset,
            "right",
            left_extra_offset=left_extra,
        )

        np.testing.assert_allclose(
            left[THUMB_CMC],
            landmarks[THUMB_CMC] + thumb_offset + left_extra,
        )
        np.testing.assert_allclose(
            left[THUMB_TIP],
            landmarks[THUMB_TIP] + thumb_offset + left_extra,
        )
        np.testing.assert_allclose(
            right[THUMB_CMC],
            landmarks[THUMB_CMC] + np.array([-0.04, 0.01, -0.03]),
        )
        np.testing.assert_allclose(right[INDEX_MCP], landmarks[INDEX_MCP])

    def test_transform_landmarks_applies_homogeneous_transform(self):
        landmarks = _sample_landmarks(0.5)
        transform = _translation_matrix([1.0, -2.0, 3.0])

        transformed = stream._transform_landmarks(transform, landmarks)

        np.testing.assert_allclose(transformed, landmarks + np.array([1.0, -2.0, 3.0]))

    def test_transform_landmarks_rejects_bad_shapes(self):
        with self.assertRaisesRegex(ValueError, "landmarks must have shape"):
            stream._transform_landmarks(np.eye(4), np.zeros((5, 3)))
        with self.assertRaisesRegex(ValueError, "transform must have shape"):
            stream._transform_landmarks(np.eye(3), np.zeros((21, 3)))

    def test_parse_args_accepts_human_joint_debug_flags(self):
        test_args = [
            "dexcap_glove_meshcat_stream.py",
            "--debug-human-joints",
            "--debug-human-joints-interval",
            "7",
        ]

        with mock.patch.object(stream.sys, "argv", test_args):
            args = stream._parse_args()

        self.assertTrue(args.debug_human_joints)
        self.assertEqual(args.debug_human_joints_interval, 7)

    def test_parse_args_hides_joint_frames_by_default_for_low_latency(self):
        test_args = ["dexcap_glove_meshcat_stream.py"]

        with mock.patch.object(stream.sys, "argv", test_args):
            args = stream._parse_args()

        self.assertFalse(args.show_frames)

    def test_parse_args_can_enable_joint_frames_for_debugging(self):
        test_args = ["dexcap_glove_meshcat_stream.py", "--frames"]

        with mock.patch.object(stream.sys, "argv", test_args):
            args = stream._parse_args()

        self.assertTrue(args.show_frames)

    def test_drain_ready_packets_keeps_latest_complete_packet(self):
        first = _packet(mask=0x8000, left=np.full(21, 100, dtype=np.uint16))
        second = _packet(mask=0x8000, left=np.full(21, 200, dtype=np.uint16))
        latest = _packet(mask=0x8000, left=np.full(21, 300, dtype=np.uint16))
        fake_socket = _FakeBufferedSocket(second + latest)

        packet, disconnected, drained = stream._drain_ready_packets(fake_socket, first)

        self.assertEqual(packet, latest)
        self.assertFalse(disconnected)
        self.assertEqual(drained, 2)
        self.assertEqual(fake_socket.buffer, b"")
        self.assertIsNone(fake_socket.timeout)

    def test_drain_ready_packets_leaves_partial_packet_buffered(self):
        first = _packet(mask=0x8000, left=np.full(21, 100, dtype=np.uint16))
        partial = _packet(mask=0x8000, left=np.full(21, 200, dtype=np.uint16))[:10]
        fake_socket = _FakeBufferedSocket(partial)

        packet, disconnected, drained = stream._drain_ready_packets(fake_socket, first)

        self.assertEqual(packet, first)
        self.assertFalse(disconnected)
        self.assertEqual(drained, 0)
        self.assertEqual(fake_socket.buffer, partial)


def _packet(mask, left=None, body=None, right=None):
    data = np.zeros(72, dtype=np.uint16)
    data[0] = mask
    if left is not None:
        data[1:22] = left
    if body is not None:
        data[25:48] = body
    if right is not None:
        data[48:69] = right
    return struct.pack("<72H", *data)


def _expected_glove_q(raw):
    rad = raw.astype(float) / 100.0 * np.pi / 180.0
    rad = (rad + np.pi) % (2 * np.pi) - np.pi
    return np.hstack(
        (
            np.flip(rad[:5]),
            np.flip(rad[5:9]),
            np.flip(rad[9:13]),
            np.flip(rad[13:17]),
            np.flip(rad[17:21]),
        )
    )


def _sample_landmarks(offset):
    points = np.zeros((21, 3), dtype=float)
    for idx in range(21):
        points[idx] = np.array([offset + idx * 0.01, idx * 0.02, -idx * 0.005])
    return points


def _translation_matrix(xyz):
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


class _FakeOverlay:
    def __init__(self):
        self.updated = None

    def update(self, landmarks):
        self.updated = np.asarray(landmarks, dtype=float).copy()


class _FakeSolver:
    def __init__(self, frame):
        self.frame = frame
        self.last_q = None

    def reconstruct(self, q):
        self.last_q = np.asarray(q, dtype=float).copy()
        return self.frame


class _FakeBufferedSocket:
    def __init__(self, buffer):
        self.buffer = bytes(buffer)
        self.timeout = None

    def gettimeout(self):
        return self.timeout

    def setblocking(self, flag):
        self.timeout = None if flag else 0.0

    def settimeout(self, value):
        self.timeout = value

    def recv(self, size, flags=0):
        if not self.buffer:
            raise BlockingIOError()
        chunk = self.buffer[:size]
        if flags & stream.socket.MSG_PEEK:
            return chunk
        self.buffer = self.buffer[len(chunk):]
        return chunk


def _make_display_for_solver_test(*, left_frame, right_frame):
    display = stream.GloveMeshcatDisplay.__new__(stream.GloveMeshcatDisplay)
    display._left_solver = _FakeSolver(left_frame)
    display._right_solver = _FakeSolver(right_frame)
    display._left_overlay = _FakeOverlay()
    display._right_overlay = _FakeOverlay()
    display.human_skeleton_display_offset = np.array([0.01, 0.02, 0.03])
    display.thumb_display_offset = np.array([0.04, 0.01, -0.03])
    display.left_thumb_extra_display_offset = np.array([0.006, 0.0, 0.0])
    display._human_overlay_runtime_error_reported = False
    display.debug_human_joints = False
    display.debug_human_joints_interval = 30
    display._human_debug_frame_count = 0
    return display


if __name__ == "__main__":
    unittest.main()
