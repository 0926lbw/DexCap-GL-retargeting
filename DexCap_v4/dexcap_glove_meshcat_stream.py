#!/usr/bin/env python3
"""Stream DexCap glove data into moving DexGlove MeshCat models."""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DexCap import Listener, SOCKET_BUFFER_SIZE

LEFT_MASK = 0x8000
RIGHT_MASK = 0x2000
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_LEFT_URDF = Path("DexGlove_L_v4/urdf/DexGlove_L_v4.urdf")
DEFAULT_RIGHT_URDF = Path("DexGlove_R_v4/urdf/DexGlove_R_v4.urdf")
LEFT_ROOT_NAME = "DexGlove_L_v4"
RIGHT_ROOT_NAME = "DexGlove_R_v4"
HUMAN_HAND_NODE = "human_hand"
DEFAULT_HUMAN_SKELETON_DISPLAY_OFFSET = (0.0, 0.0, 0.025)
DEFAULT_THUMB_DISPLAY_OFFSET = (0.015, 0.0, -0.015)
DEFAULT_LEFT_THUMB_EXTRA_DISPLAY_OFFSET = (0.005, 0.0, 0.0)
THUMB_DISPLAY_INDICES = (1, 2, 3, 4)
SOCKET_DRAIN_MAX_PACKETS = 2048
HUMAN_JOINT_DEBUG_FINGERS = (
    (
        "thumb",
        0,
        5,
        ("thumb_cmc_abd", "thumb_cmc_flex", "thumb_mcp", "thumb_ip"),
        (1, 2, 3, 4),
    ),
    (
        "index",
        5,
        9,
        ("index_mcp_abd", "index_mcp_flex", "index_pip", "index_dip"),
        (5, 6, 7, 8),
    ),
    (
        "middle",
        9,
        13,
        ("middle_mcp_abd", "middle_mcp_flex", "middle_pip", "middle_dip"),
        (9, 10, 11, 12),
    ),
    (
        "ring",
        13,
        17,
        ("ring_mcp_abd", "ring_mcp_flex", "ring_pip", "ring_dip"),
        (13, 14, 15, 16),
    ),
    (
        "pinky",
        17,
        21,
        ("pinky_mcp_abd", "pinky_mcp_flex", "pinky_pip", "pinky_dip"),
        (17, 18, 19, 20),
    ),
)


def parse_listener_packet(packet: bytes) -> Listener:
    """Parse one SDK stream packet with DexCap.py's existing Listener parser."""
    listener = Listener(port=0)
    listener.last_mask = int.from_bytes(packet[:2], byteorder="little", signed=False)
    listener.last_device_flags = listener._parse_data_packet(packet)
    return listener


def display_q_from_listener(
    listener: Listener,
    previous_left: np.ndarray,
    previous_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return display q arrays, preserving the last value for omitted gloves."""
    mask = getattr(listener, "last_mask", 0)
    left_q = listener.q_l if mask & LEFT_MASK else previous_left
    right_q = listener.q_r if mask & RIGHT_MASK else previous_right
    return left_q.copy(), right_q.copy()


def _transform_landmarks(transform: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to 21 xyz landmarks."""
    points = np.asarray(landmarks, dtype=float)
    if points.shape != (21, 3):
        raise ValueError(f"landmarks must have shape (21, 3), got {points.shape}")
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {transform.shape}")
    homogeneous = np.ones((points.shape[0], 4), dtype=float)
    homogeneous[:, :3] = points
    return (transform @ homogeneous.T).T[:, :3]


def _offset_landmarks_for_display(
    landmarks: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """Move the displayed skeleton as one rigid overlay offset."""
    points = np.asarray(landmarks, dtype=float)
    if points.shape != (21, 3):
        raise ValueError(f"landmarks must have shape (21, 3), got {points.shape}")
    offset = np.asarray(offset, dtype=float)
    if offset.shape != (3,):
        raise ValueError(f"offset must have shape (3,), got {offset.shape}")
    if not np.all(np.isfinite(offset)):
        raise ValueError("offset must contain only finite values")
    return points + offset


def _offset_thumb_for_display(
    landmarks: np.ndarray,
    offset: np.ndarray,
    hand: str,
    *,
    left_extra_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the thumb-only display correction, mirrored for the right hand."""
    if hand not in {"left", "right"}:
        raise ValueError("hand must be 'left' or 'right'")
    points = np.asarray(landmarks, dtype=float)
    if points.shape != (21, 3):
        raise ValueError(f"landmarks must have shape (21, 3), got {points.shape}")
    offset = np.asarray(offset, dtype=float)
    if offset.shape != (3,):
        raise ValueError(f"offset must have shape (3,), got {offset.shape}")
    if not np.all(np.isfinite(offset)):
        raise ValueError("offset must contain only finite values")
    if left_extra_offset is None:
        left_extra = np.zeros(3, dtype=float)
    else:
        left_extra = np.asarray(left_extra_offset, dtype=float)
        if left_extra.shape != (3,):
            raise ValueError(
                f"left_extra_offset must have shape (3,), got {left_extra.shape}"
            )
        if not np.all(np.isfinite(left_extra)):
            raise ValueError("left_extra_offset must contain only finite values")

    hand_offset = offset.copy()
    if hand == "right":
        hand_offset[0] *= -1.0
    else:
        hand_offset += left_extra

    moved = points.copy()
    moved[list(THUMB_DISPLAY_INDICES)] += hand_offset
    return moved


def _human_joint_debug_lines(hand: str, q: np.ndarray, frame) -> list[str]:
    """Format per-finger reconstruction diagnostics for live tuning."""
    q = np.asarray(q, dtype=float)
    diagnostics = frame.diagnostics
    lines = [
        (
            f"[human-joints {hand}] "
            f"tip_err_before={_debug_metric(diagnostics, 'fingertip_error_before_fusion')} "
            f"tip_err_after={_debug_metric(diagnostics, 'fingertip_error_after_fusion')} "
            f"roundtrip={_debug_metric(diagnostics, 'roundtrip_error')}"
        )
    ]
    for finger, start, stop, joint_names, chain in HUMAN_JOINT_DEBUG_FINGERS:
        q_values = np.array2string(
            q[start:stop],
            precision=3,
            suppress_small=True,
            separator=",",
        )
        joints = " ".join(
            f"{_short_joint_name(name)}={frame.joint_angles.get(name, float('nan')):.3f}"
            for name in joint_names
        )
        chord = _debug_chord_lengths(frame.keypoints_21_in_Hwrist, chain)
        tip_idx = chain[-1]
        tip_target_delta = float(
            np.linalg.norm(
                frame.fused_keypoints_21_in_Hwrist[tip_idx]
                - frame.direct_glove_keypoints_21_in_Hwrist[tip_idx]
            )
        )
        lines.append(
            f"[human-joints {hand}] {finger} q[{start}:{stop}]={q_values} "
            f"joints {joints} chord={chord} tip_target_delta={tip_target_delta:.4f}"
        )
    return lines


def _debug_metric(diagnostics: dict, key: str) -> str:
    value = diagnostics.get(key)
    if value is None:
        return "nan"
    return f"{float(value):.4f}"


def _short_joint_name(name: str) -> str:
    for prefix in ("thumb_", "index_", "middle_", "ring_", "pinky_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _debug_chord_lengths(points: np.ndarray, chain: tuple[int, int, int, int]) -> str:
    root = points[chain[0]]
    lengths = [
        float(np.linalg.norm(points[idx] - root))
        for idx in chain[1:]
    ]
    return "[" + ",".join(f"{value:.4f}" for value in lengths) + "]"


class GloveMeshcatDisplay:
    """MeshCat display for the left and right DexGlove URDF models."""

    def __init__(
        self,
        left_urdf: Path = DEFAULT_LEFT_URDF,
        right_urdf: Path = DEFAULT_RIGHT_URDF,
        show_frames: bool = False,
        frame_axis_length: float = 0.02,
        show_human_skeleton: bool = True,
        human_skeleton_display_offset: tuple[float, float, float] = DEFAULT_HUMAN_SKELETON_DISPLAY_OFFSET,
        thumb_display_offset: tuple[float, float, float] = DEFAULT_THUMB_DISPLAY_OFFSET,
        left_thumb_extra_display_offset: tuple[float, float, float] = DEFAULT_LEFT_THUMB_EXTRA_DISPLAY_OFFSET,
        debug_human_joints: bool = False,
        debug_human_joints_interval: int = 30,
    ) -> None:
        try:
            import meshcat.geometry as mg
            import meshcat.transformations as mtf
            import pinocchio as pin
            from pinocchio.visualize import MeshcatVisualizer
        except ImportError as exc:
            raise ImportError(
                "需要 pinocchio 和 meshcat 才能显示手套模型。请在已安装 DexCap "
                "机器人依赖的 Python 环境运行。"
            ) from exc

        self._mtf = mtf
        self._meshcat_geometry = mg
        self._pin = pin
        self._meshcat_visualizer = MeshcatVisualizer
        self.show_frames = show_frames
        self.frame_axis_length = frame_axis_length
        self.human_skeleton_display_offset = np.asarray(
            human_skeleton_display_offset, dtype=float
        )
        self.thumb_display_offset = np.asarray(thumb_display_offset, dtype=float)
        self.left_thumb_extra_display_offset = np.asarray(
            left_thumb_extra_display_offset, dtype=float
        )
        if debug_human_joints_interval <= 0:
            raise ValueError("debug_human_joints_interval must be positive")
        self.debug_human_joints = bool(debug_human_joints)
        self.debug_human_joints_interval = int(debug_human_joints_interval)
        self._human_debug_frame_count = 0
        self.left_viz, self.right_viz = self._load_models(left_urdf, right_urdf)
        self.left_q = np.zeros(self.left_viz.model.nq)
        self.right_q = np.zeros(self.right_viz.model.nq)

        # Optional in-palm 21-point human skeleton overlay. Built non-fatally so
        # glove streaming still works even if the hand reconstruction modules
        # or URDF observation hit a problem.
        self._left_solver = None
        self._right_solver = None
        self._left_overlay = None
        self._right_overlay = None
        self._human_overlay_runtime_error_reported = False
        if show_human_skeleton:
            self._build_human_overlays(left_urdf, right_urdf)

    def _build_human_overlays(self, left_urdf: Path, right_urdf: Path) -> None:
        """Build the in-palm 21-point human-skeleton overlays. Non-fatal on failure.

        Uses a parametric human-hand URDF (human_hand_builder) driven by a
        faithful glove->hand retargeting (retargeting.GloveToHumanRetargeter),
        so the skeleton moves like a human hand rather than tracing the
        exoskeleton mechanism. Each hand is rigid-fit (Kabsch) to the glove's
        actual 21-link layout, so the skeleton inherits the glove's coordinate
        convention (direction + handedness) automatically -- no hand-tuned
        rotation, and left/right comes out correct per hand.
        """
        try:
            from hand_reconstruction.human_hand_builder import load_human_hand, rigid_fit
            from hand_reconstruction.human_hand_params import default_params
            from hand_reconstruction.pipeline import HandReconstructionPipeline
            from hand_reconstruction.retargeting import GloveToHumanRetargeter
            from hand_reconstruction.solver import HandReconstructionSolver
            from hand_reconstruction.stream_overlay import MeshcatSkeletonOverlay
        except ImportError as exc:
            print(f"人手骨架模块不可用，跳过掌内骨架: {exc}", flush=True)
            return

        try:
            params = default_params()
            left_human = load_human_hand(params, "left")
            right_human = load_human_hand(params, "right")
            left_retar = GloveToHumanRetargeter("left")
            right_retar = GloveToHumanRetargeter("right")

            # Fit each human rest pose onto that glove's actual link positions
            # (the direct-link ground truth). The mirrored left model keeps each
            # fit a proper rigid transform, so direction + handedness match.
            for side, human, retargeter, glove_urdf, root_name in (
                ("left", left_human, left_retar, left_urdf, LEFT_ROOT_NAME),
                ("right", right_human, right_retar, right_urdf, RIGHT_ROOT_NAME),
            ):
                glove_pipe = HandReconstructionPipeline(glove_urdf, side)
                glove_nq = glove_pipe.observer.robot.model.nq
                glove_links = glove_pipe.reconstruct_direct(np.zeros(glove_nq)).landmarks
                human_rest = human.landmarks_from_q(np.zeros(human.nq))
                align = rigid_fit(human_rest, glove_links)
                solver = HandReconstructionSolver(
                    hand=side,
                    human_model=human,
                    retargeter=retargeter,
                    glove_pipeline=glove_pipe,
                    T_W_Hwrist=align,
                )
                if side == "left":
                    self._left_solver = solver
                else:
                    self._right_solver = solver
                self.viewer[root_name][HUMAN_HAND_NODE].set_transform(align)

            self._left_overlay = MeshcatSkeletonOverlay(
                self.viewer[LEFT_ROOT_NAME][HUMAN_HAND_NODE],
                "left",
                geometry=self._meshcat_geometry,
                transforms=self._mtf,
            )
            self._right_overlay = MeshcatSkeletonOverlay(
                self.viewer[RIGHT_ROOT_NAME][HUMAN_HAND_NODE],
                "right",
                geometry=self._meshcat_geometry,
                transforms=self._mtf,
            )
            self._left_overlay.build()
            self._right_overlay.build()
            print("已在掌内加载参数化人手骨架 (对齐手套坐标系 + retargeting)", flush=True)
        except Exception as exc:  # 骨架是可选叠加，不能让手套显示整体失败
            print(f"人手骨架加载失败，仅显示手套: {exc}", flush=True)
            self._left_solver = None
            self._right_solver = None
            self._left_overlay = None
            self._right_overlay = None

    def display_human(self, left_q: np.ndarray, right_q: np.ndarray) -> None:
        """Retarget both glove vectors and move the in-palm human-hand overlays."""
        required = (
            self._left_overlay,
            self._right_overlay,
            self._left_solver,
            self._right_solver,
        )
        if any(item is None for item in required):
            return

        try:
            left_q = np.asarray(left_q, dtype=float)
            right_q = np.asarray(right_q, dtype=float)
            left_frame = self._left_solver.reconstruct(left_q)
            right_frame = self._right_solver.reconstruct(right_q)
            self._print_human_joint_debug(left_q, right_q, left_frame, right_frame)
            left_display = _offset_landmarks_for_display(
                left_frame.fused_keypoints_21_in_Hwrist,
                self.human_skeleton_display_offset,
            )
            right_display = _offset_landmarks_for_display(
                right_frame.fused_keypoints_21_in_Hwrist,
                self.human_skeleton_display_offset,
            )
            left_display = _offset_thumb_for_display(
                left_display,
                self.thumb_display_offset,
                "left",
                left_extra_offset=self.left_thumb_extra_display_offset,
            )
            right_display = _offset_thumb_for_display(
                right_display,
                self.thumb_display_offset,
                "right",
                left_extra_offset=self.left_thumb_extra_display_offset,
            )
            self._left_overlay.update(left_display)
            self._right_overlay.update(right_display)
        except Exception as exc:
            if not self._human_overlay_runtime_error_reported:
                print(f"人手骨架更新失败，暂停骨架更新: {exc}", flush=True)
                self._human_overlay_runtime_error_reported = True
            self._left_overlay = None
            self._right_overlay = None

    def _print_human_joint_debug(
        self,
        left_q: np.ndarray,
        right_q: np.ndarray,
        left_frame,
        right_frame,
    ) -> None:
        if not self.debug_human_joints:
            return

        self._human_debug_frame_count += 1
        if (self._human_debug_frame_count - 1) % self.debug_human_joints_interval != 0:
            return

        for line in _human_joint_debug_lines("left", left_q, left_frame):
            print(line, flush=True)
        for line in _human_joint_debug_lines("right", right_q, right_frame):
            print(line, flush=True)

    def _load_models(self, left_urdf: Path, right_urdf: Path):
        viewer = None
        visualizers = []
        configs = (
            (
                "DexGlove_L_v4",
                _resolve_path(left_urdf),
                np.array([0.0, -0.12, 0.0]),
                [0.72, 0.84, 1.0, 1.0],
            ),
            (
                "DexGlove_R_v4",
                _resolve_path(right_urdf),
                np.array([0.0, 0.12, 0.0]),
                [1.0, 0.80, 0.55, 1.0],
            ),
        )

        for root_name, urdf, offset, color in configs:
            model, collision_model, visual_model = self._pin.buildModelsFromUrdf(
                str(urdf),
                package_dirs=_package_dirs(urdf),
            )
            viz = self._meshcat_visualizer(model, collision_model, visual_model)
            viz.initViewer(viewer=viewer, open=True)
            if viewer is None:
                viewer = viz.viewer
                viewer.delete()
            viz.loadViewerModel(rootNodeName=root_name, visual_color=color)
            if self.show_frames:
                self._enable_joint_frames(viz, model)
            viz.display(self._pin.neutral(model))
            viewer[root_name].set_transform(self._mtf.translation_matrix(offset))
            visualizers.append(viz)

        print(f"MeshCat URL: {viewer.url()}", flush=True)
        self.viewer = viewer
        return visualizers[0], visualizers[1]

    def display(self, left_q: np.ndarray, right_q: np.ndarray) -> None:
        if left_q.shape != (self.left_viz.model.nq,):
            raise ValueError(f"left_q must have shape ({self.left_viz.model.nq},)")
        if right_q.shape != (self.right_viz.model.nq,):
            raise ValueError(f"right_q must have shape ({self.right_viz.model.nq},)")
        self.left_q = left_q.copy()
        self.right_q = right_q.copy()
        self.left_viz.display(self.left_q)
        self.right_viz.display(self.right_q)

    def _enable_joint_frames(self, viz, model) -> None:
        """Render an RGB axis triad at every glove joint frame in MeshCat.

        Pinocchio exposes one JOINT frame per revolute ``glove_joint_*`` joint;
        selecting only those keeps the view free of body/inertial frames. The
        triads track the glove as it streams because ``display(q)`` refreshes
        every frame placement while ``display_frames`` is enabled.
        """
        frame_ids = [
            index
            for index, frame in enumerate(model.frames)
            if frame.type == self._pin.FrameType.JOINT
        ]
        viz.displayFrames(
            True,
            frame_ids=frame_ids,
            axis_length=self.frame_axis_length,
            axis_width=2,
        )


def run_tcp_stream(
    host: str,
    port: int,
    display: GloveMeshcatDisplay,
    fps_limit: float,
) -> None:
    listener_state = Listener(port=port)
    last_left = np.zeros(21)
    last_right = np.zeros(21)
    min_interval = 0.0 if fps_limit <= 0 else 1.0 / fps_limit
    last_display_time = 0.0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"等待 DexCap SDK TCP 数据流: {host}:{port}", flush=True)

        while True:
            client_socket, addr = server_socket.accept()
            print(f"SDK 已连接: {addr}", flush=True)
            with client_socket:
                while True:
                    packet = listener_state._recv_exact(client_socket, SOCKET_BUFFER_SIZE)
                    if packet is None:
                        print("SDK 连接断开，继续等待下一次连接", flush=True)
                        break
                    packet, disconnected, _ = _drain_ready_packets(
                        client_socket,
                        packet,
                    )

                    listener_state.last_mask = int.from_bytes(
                        packet[:2], byteorder="little", signed=False
                    )
                    listener_state.last_device_flags = listener_state._parse_data_packet(packet)
                    last_left, last_right = display_q_from_listener(
                        listener_state,
                        last_left,
                        last_right,
                    )

                    now = time.monotonic()
                    if now - last_display_time >= min_interval:
                        display.display(last_left, last_right)
                        display.display_human(last_left, last_right)
                        last_display_time = now
                    if disconnected:
                        print("SDK 连接断开，继续等待下一次连接", flush=True)
                        break


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--left-urdf", type=Path, default=DEFAULT_LEFT_URDF)
    parser.add_argument("--right-urdf", type=Path, default=DEFAULT_RIGHT_URDF)
    parser.add_argument(
        "--fps-limit",
        type=float,
        default=60.0,
        help="MeshCat refresh limit. Use 0 to display every packet.",
    )
    parser.add_argument(
        "--frames",
        dest="show_frames",
        action="store_true",
        default=False,
        help="Show an RGB axis triad at every glove joint frame for debugging.",
    )
    parser.add_argument(
        "--no-frames",
        dest="show_frames",
        action="store_false",
        help="Hide the glove joint coordinate frames (default, lower latency).",
    )
    parser.add_argument(
        "--frame-axis-length",
        type=float,
        default=0.02,
        help="Triad axis length (m) for the joint frames. Default 0.02.",
    )
    parser.add_argument(
        "--no-human-skeleton",
        dest="show_human_skeleton",
        action="store_false",
        default=True,
        help="不在掌内显示 21 点人手骨架。",
    )
    parser.add_argument(
        "--human-skeleton-display-offset",
        nargs=3,
        type=float,
        default=DEFAULT_HUMAN_SKELETON_DISPLAY_OFFSET,
        metavar=("X", "Y", "Z"),
        help="Display offset for the whole human skeleton in local meters. Default 0 0 0.025.",
    )
    parser.add_argument(
        "--thumb-display-offset",
        nargs=3,
        type=float,
        default=DEFAULT_THUMB_DISPLAY_OFFSET,
        metavar=("X", "Y", "Z"),
        help=(
            "Thumb-only display offset in local meters. X is mirrored for the "
            "right hand. Default 0.015 0 -0.015."
        ),
    )
    parser.add_argument(
        "--left-thumb-extra-display-offset",
        nargs=3,
        type=float,
        default=DEFAULT_LEFT_THUMB_EXTRA_DISPLAY_OFFSET,
        metavar=("X", "Y", "Z"),
        help=(
            "Extra left-thumb-only display offset in local meters. "
            "Default 0.005 0 0."
        ),
    )
    parser.add_argument(
        "--debug-human-joints",
        action="store_true",
        default=False,
        help=(
            "Print per-finger q, retargeted human joint angles, local keypoint "
            "chords, and tip-target diagnostics while streaming."
        ),
    )
    parser.add_argument(
        "--debug-human-joints-interval",
        type=int,
        default=30,
        help="Print human joint diagnostics every N displayed human frames. Default 30.",
    )
    return parser.parse_args()


def _drain_ready_packets(
    client_socket: socket.socket,
    first_packet: bytes,
    *,
    packet_size: int = SOCKET_BUFFER_SIZE,
    max_packets: int = SOCKET_DRAIN_MAX_PACKETS,
) -> tuple[bytes, bool, int]:
    """Drop already-buffered stale TCP packets and return the newest complete packet."""
    latest = first_packet
    drained = 0
    disconnected = False
    previous_timeout = client_socket.gettimeout()

    try:
        client_socket.setblocking(False)
        while drained < max_packets:
            try:
                available = client_socket.recv(packet_size, socket.MSG_PEEK)
            except (BlockingIOError, InterruptedError):
                break

            if not available:
                disconnected = True
                break
            if len(available) < packet_size:
                break

            packet = client_socket.recv(packet_size)
            if not packet:
                disconnected = True
                break
            if len(packet) < packet_size:
                break

            latest = packet
            drained += 1
    finally:
        client_socket.settimeout(previous_timeout)

    return latest, disconnected, drained


def _resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _package_dirs(urdf_path: Path) -> list[str]:
    urdf_dir = urdf_path.parent
    parent = urdf_dir.parent
    grandparent = parent.parent
    candidates = [
        urdf_dir,
        parent,
        grandparent,
        Path.cwd(),
        REPO_ROOT,
        REPO_ROOT / "DexCap_v4",
    ]
    return [str(path) for path in candidates if path.is_dir()]


def main() -> int:
    args = _parse_args()
    display = GloveMeshcatDisplay(
        args.left_urdf,
        args.right_urdf,
        show_frames=args.show_frames,
        frame_axis_length=args.frame_axis_length,
        show_human_skeleton=args.show_human_skeleton,
        human_skeleton_display_offset=tuple(args.human_skeleton_display_offset),
        thumb_display_offset=tuple(args.thumb_display_offset),
        left_thumb_extra_display_offset=tuple(args.left_thumb_extra_display_offset),
        debug_human_joints=args.debug_human_joints,
        debug_human_joints_interval=args.debug_human_joints_interval,
    )
    run_tcp_stream(args.host, args.port, display, args.fps_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
