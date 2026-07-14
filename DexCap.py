"""DexCap exoskeleton and data glove interface module.

This module provides interfaces for DexCap exoskeleton upper body tracking,
data glove finger joint mapping, and socket-based sensor data communication.
It handles forward kinematics, coordinate transformations, and real-time
data streaming from the exoskeleton hardware.

Classes:
    DexCap_UL: Upper body exoskeleton controller with forward kinematics.
    DexCap_GL: Data glove controller with finger joint mapping.
    Listener: Socket server for receiving sensor data from hardware.
    MappingType: Enum for glove mapping strategies.
"""

import os
import socket
import struct
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple
from scipy.spatial.transform import Rotation

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_file(path: str) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())

    cwd_candidate = candidate.resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    return str((PROJECT_ROOT / candidate).resolve())


def _urdf_package_dirs(urdf_path: str) -> list[str]:
    """Return package search roots for source-tree and resource-package URDF layouts."""
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    parent = os.path.dirname(urdf_dir)
    grandparent = os.path.dirname(parent)
    candidates = [
        urdf_dir,
        os.path.join(urdf_dir, "DexCap_v4"),
        parent,
        os.path.join(parent, "DexCap_v4"),
        grandparent,
        os.path.join(grandparent, "DexCap_v4"),
    ]

    package_dirs = []
    for candidate in candidates:
        if os.path.isdir(candidate) and candidate not in package_dirs:
            package_dirs.append(candidate)
    return package_dirs


# Exoskeleton constants
EXOSKELETON_DOF = 23  # Degrees of freedom
EXOSKELETON_ARM_DOF = 18  # Arm degrees of freedom
EXOSKELETON_Q_START_INDEX = 5  # Starting index for arm joints
EXOSKELETON_ANGLE_WRAP_THRESHOLD = 18000  # Angle wrapping threshold
EXOSKELETON_ANGLE_RANGE = 36000  # Full angle range

# Frame names for left and right arm endpoints
FRAME_NAME_LEFT_ARM = "left_arm_joint_9"
FRAME_NAME_RIGHT_ARM = "right_arm_joint_9"
FRAME_NAME_WAIST = "waist_joint_5"

# Low-pass filter coefficient for exoskeleton data
EXOSKELETON_FILTER_ALPHA = 0.25  # New data weight

# Angle conversion constants
ENCODER_TO_DEGREES = 100.0
DEGREES_TO_RADIANS = np.pi / 180.0
ANGLE_DECIMAL_PRECISION = 3

# Coordinate system transformation matrices
ROTATION_PLACEMENT_L = np.array([[1, 0, 0],
                                  [0, 0, -1],
                                  [0, 1, 0]])
ROTATION_PLACEMENT_R = np.array([[1, 0, 0],
                                  [0, 0, -1],
                                  [0, 1, 0]])

# Data glove constants
GLOVE_DOF = 21  # Degrees of freedom per glove
GLOVE_HAND_DOF = 20  # Output hand joint DOF
NUM_FINGERS = 5  # Number of fingers
JOINTS_PER_FINGER = 4  # Joints per finger

# Joint mapping scale factors
FINGER_TIP_SCALE = 1.0 / 3.0
FINGER_MID_SCALE = 2.0 / 3.0
THUMB_BASE_SCALE = 0.5

# Finger joint indices to negate
LEFT_HAND_NEGATE_INDICES = [12, 16]
RIGHT_HAND_NEGATE_INDICES = [12, 16]

# Visualization colors
VIZ_COLOR_SEMI_TRANSPARENT = [0.5, 0.5, 0.5, 0.5]
VIZ_COLOR_OPAQUE = [0.5, 0.5, 0.5, 1.0]

# Socket communication constants
DEFAULT_SOCKET_PORT = 8080
SOCKET_LISTEN_HOST = '127.0.0.1'
SOCKET_BACKLOG = 5  # Max queued connections
SOCKET_BUFFER_SIZE = 144  # 2 * 72 bytes

# Data packet constants
PACKET_SIZE = 72  # Expected packet size in uint16 elements
PACKET_HEADER_SIZE = 2  # Header size in bytes
DATA_TYPE = np.uint16
BINARY_STRING_WIDTH = 16

# Data array slices
LEFT_GLOVE_START = 1
LEFT_GLOVE_END = 22
BACKPLATE_START = 25
BACKPLATE_END = 48
RIGHT_GLOVE_START = 48
RIGHT_GLOVE_END = 69

# Arm data indices
LEFT_ARM_SIZE = 9
RIGHT_ARM_START = 9
RIGHT_ARM_END = 18


class MappingType(Enum):
    """Enumeration of hand joint mapping strategies.

    Attributes:
        joint_matching: Direct joint-to-joint mapping strategy.
        adaptive_mapping: Adaptive mapping strategy (not yet implemented).
    """
    joint_matching = 1
    adaptive_mapping = 2


class DexCap_UL:
    """DexCap upper body exoskeleton controller.

    This class manages the upper body exoskeleton, computing forward kinematics
    and providing arm endpoint transforms for teleoperation. It handles sensor
    data processing, calibration, and coordinate frame transformations.

    Attributes:
        exoskeleton: Pinocchio robot model for the exoskeleton.
        q: Current joint positions (23 DOF).
        q_offset: Joint position offsets for initialization.
        original_pose: Original pose at calibration.
        offset: Calibration offset values.
        received_array: Raw sensor data array.
        display: Whether to display visualization.
        state: Calibration state dictionary.
        initial_pose_l: Initial left arm endpoint pose.
        initial_pose_r: Initial right arm endpoint pose.
    """

    def __init__(
        self,
        urdf_path_exoskeleton: str = "urdf/DexCap_UL_v4.urdf",
        state_: Optional[Dict[str, bool]] = None,
        display: bool = False,
    ) -> None:
        """Initializes the exoskeleton controller.

        Args:
            urdf_path_exoskeleton: Path to the exoskeleton URDF file.
            state_: Calibration state dictionary.
            display: Whether to enable visualization.
        """
        if state_ is None:
            state_ = {'calibration': False}
        urdf_path_exoskeleton = _resolve_project_file(urdf_path_exoskeleton)

        # Build exoskeleton model from URDF
        self.exoskeleton = RobotWrapper.BuildFromURDF(
            urdf_path_exoskeleton,
            package_dirs=_urdf_package_dirs(urdf_path_exoskeleton),
        )

        # Initialize joint positions and parameters
        self.q = np.zeros(EXOSKELETON_DOF)
        self.q_offset = np.zeros(EXOSKELETON_DOF)
        self.original_pose = np.zeros(EXOSKELETON_DOF)
        self.offset = np.zeros(EXOSKELETON_DOF)
        self.received_array = np.zeros(EXOSKELETON_DOF)
        self.display = display
        self.state = state_
        self._idx = 0
        self._filter_alpha = EXOSKELETON_FILTER_ALPHA
        self._q_vector_prev: Optional[np.ndarray] = None
        self.q_offset[:5]=[-np.pi/4,0,-np.pi/4*3,np.pi/2,0]
        # Align exoskeleton and robot arm coordinate systems
        self._frame_id_left = self.exoskeleton.model.getFrameId(
            FRAME_NAME_LEFT_ARM
        )
        self.exoskeleton.model.frames[
            self._frame_id_left
        ].placement.rotation = ROTATION_PLACEMENT_L

        self._frame_id_right = self.exoskeleton.model.getFrameId(
            FRAME_NAME_RIGHT_ARM
        )
        self.exoskeleton.model.frames[
            self._frame_id_right
        ].placement.rotation = ROTATION_PLACEMENT_R
        self._frame_id_waist = self.exoskeleton.model.getFrameId(FRAME_NAME_WAIST)

        # Compute initial transforms
        self.compute_transform()

        # Setup visualization if enabled
        if self.display:
            self._viz = MeshcatVisualizer(
                self.exoskeleton.model,
                self.exoskeleton.collision_model,
                self.exoskeleton.visual_model,
            )
            self._viz.initViewer(open=True)
            self._viz.loadViewerModel(color=VIZ_COLOR_SEMI_TRANSPARENT)
            self._viz.display(self.q)

    def compute_transform(self) -> None:

        """Computes initial forward kinematics for arm endpoints.

        Calculates and stores the initial poses of left and right arm
        endpoints for later use in computing relative transformations.
        """
        q = self.q.copy()
        # 腰部位姿保留真实腰部关节，用于 G1 腰部相对姿态映射。
        pin.forwardKinematics(
            self.exoskeleton.model, self.exoskeleton.data, q
        )
        pin.updateFramePlacements(self.exoskeleton.model, self.exoskeleton.data)
        if not hasattr(self, 'initial_pose_waist'):
            self.initial_pose_waist = self.exoskeleton.data.oMf[
                self._frame_id_waist
            ].copy()

        # 与 get_Arm_transform 保持一致：手臂相对变换仍把腰部关节置零后再做 FK。
        q[:EXOSKELETON_Q_START_INDEX] = 0.0
        pin.forwardKinematics(
            self.exoskeleton.model, self.exoskeleton.data, q
        )
        pin.updateFramePlacements(self.exoskeleton.model, self.exoskeleton.data)
        # 记录标定时刻外骨骼两端的绝对位姿，作为后续相对运动计算的参考零点
        self.initial_pose_l = self.exoskeleton.data.oMf[
            self._frame_id_left
        ].copy()
        self.initial_pose_r = self.exoskeleton.data.oMf[
            self._frame_id_right
        ].copy()


    # 外骨骼传感器数据 → 手臂相对变换
    def get_Arm_transform(
        self, received_array: np.ndarray
    ) -> Tuple[pin.SE3, pin.SE3, pin.SE3]:
        """Computes arm endpoint transforms from sensor data.

        Processes raw sensor data, applies calibration and filtering,
        computes forward kinematics, and returns the relative transforms
        of left and right arm endpoints.

        Args:
            received_array: Raw sensor data array from exoskeleton.

        Returns:
            Tuple of (left_transform, right_transform) as Pinocchio SE3 objects
            representing the relative poses of the arm endpoints.

        Raises:
            ValueError: If received_array has insufficient data.
        """

        calibrating = self.state.get('calibration', False)

        # Convert to int32 for angle wrapping
        q_vector = received_array.astype(np.int32)

        # Handle angle wrapping (convert from 0-36000 to -18000 to 18000)
        for i in range(len(q_vector)):
            if q_vector[i] > EXOSKELETON_ANGLE_WRAP_THRESHOLD:
                q_vector[i] =  q_vector[i] -EXOSKELETON_ANGLE_RANGE

        # Convert to float and check size
        q_vector = q_vector.astype(np.float64)
        if len(q_vector) < EXOSKELETON_DOF:
            raise ValueError(
                f"Insufficient data: expected {EXOSKELETON_DOF}, "
                f"got {len(q_vector)}"
            )

        # Convert from encoder units to radians
        q_vector = q_vector / ENCODER_TO_DEGREES * DEGREES_TO_RADIANS

        # Handle calibration
        if calibrating:
            self.original_pose = self.q.copy()
            self.offset = q_vector.copy()
            self.q = np.zeros(EXOSKELETON_DOF)
            self._idx = 0

            # Update initial poses
            self.initial_pose_l = self.exoskeleton.data.oMf[
                self._frame_id_left
            ].copy()
            self.initial_pose_r = self.exoskeleton.data.oMf[
                self._frame_id_right
            ].copy()
            self.state['calibration'] = False

        # Apply calibration offset
        q_vector -= self.offset

        # Wrap angles to [-pi, pi] and apply low-pass filter
        q_vector = self._normalize_angles(q_vector)

        if self._idx != 0 and self._q_vector_prev is not None:
            # Apply low-pass filter
            q_vector = (
                (1 - self._filter_alpha) * q_vector +
                self._filter_alpha * self._q_vector_prev
            )

        if self._idx == 0:
            self._idx = 1

        self._q_vector_prev = q_vector.copy()

        # Reapply offset and normalize for configuration
        q_vector += self.offset
        q_vector = self._normalize_angles(q_vector)
        q_vector = np.round(q_vector, decimals=ANGLE_DECIMAL_PRECISION)
        q_vector = np.clip(q_vector, -np.pi, np.pi)

        # Update exoskeleton joint positions
        self.q[EXOSKELETON_Q_START_INDEX:EXOSKELETON_DOF] = (
            q_vector[:EXOSKELETON_ARM_DOF] + self.q_offset[EXOSKELETON_Q_START_INDEX:EXOSKELETON_DOF]
        )
        self.q[0:EXOSKELETON_Q_START_INDEX] = (
            q_vector[EXOSKELETON_ARM_DOF:23] + self.q_offset[0:EXOSKELETON_Q_START_INDEX]
        )
        
        pin.forwardKinematics(self.exoskeleton.model,self.exoskeleton.data, self.q)
        pin.updateFramePlacements(self.exoskeleton.model, self.exoskeleton.data)
        current_pose_waist = self.exoskeleton.data.oMf[self._frame_id_waist].copy()
        if calibrating:
            self.initial_pose_waist = current_pose_waist.copy()
        transform_waist = self.initial_pose_waist.inverse() * current_pose_waist
        
        if self.display:
            self._viz.display(self.q)

        self.q[:5]=np.zeros(5)
        # Compute forward kinematics
        pin.forwardKinematics(
            self.exoskeleton.model, self.exoskeleton.data, self.q
        )
        pin.updateFramePlacements(self.exoskeleton.model, self.exoskeleton.data)

        # Compute relative transforms
        current_pose_left = self.exoskeleton.data.oMf[self._frame_id_left]
        transform_left = self.initial_pose_l.inverse() * current_pose_left

        current_pose_right = self.exoskeleton.data.oMf[self._frame_id_right]
        transform_right = self.initial_pose_r.inverse() * current_pose_right

        return transform_left, transform_right, transform_waist

    @staticmethod
    def _normalize_angles(angles: np.ndarray) -> np.ndarray:
        """Normalizes angles to [-pi, pi] range.

        Args:
            angles: Array of angles in radians.

        Returns:
            Normalized angles in [-pi, pi] range.
        """
        normalized = angles.copy()
        for i in range(len(normalized)):
            while normalized[i] > np.pi:
                normalized[i] -= 2 * np.pi
            while normalized[i] < -np.pi:
                normalized[i] += 2 * np.pi
        return normalized


class DexCap_GL:
    """Data glove controller for finger joint mapping.

    This class manages data gloves for both hands, processing raw glove sensor
    data and mapping it to robot hand joint positions using different strategies.

    Attributes:
        mapping_type: Type of mapping strategy to use.
        q_hand_l: Mapped joint positions for left hand.
        q_hand_r: Mapped joint positions for right hand.
        visualize: Whether to display visualization.
        state: Calibration state dictionary.
        q_l: Raw joint positions for left glove.
        q_r: Raw joint positions for right glove.
    """

    def __init__(
        self,
        q_hand_l: Optional[np.ndarray] = None,
        q_hand_r: Optional[np.ndarray] = None,
        state: Optional[Dict[str, bool]] = None,
        display: bool = False,
        mapping_type: MappingType = MappingType.joint_matching,
    ) -> None:
        """Initializes the data glove controller.

        Args:
            q_hand_l: Initial left hand joint positions.
            q_hand_r: Initial right hand joint positions.
            state: Calibration state dictionary.
            display: Whether to enable visualization.
            mapping_type: Mapping strategy to use.
        """
        if q_hand_l is None:
            q_hand_l = np.zeros(GLOVE_HAND_DOF)
        if q_hand_r is None:
            q_hand_r = np.zeros(GLOVE_HAND_DOF)
        if state is None:
            state = {'calibration_hand_l': False, 'calibration_hand_r': False}

        self.mapping_type = mapping_type
        self.q_hand_l = q_hand_l
        self.q_hand_r = q_hand_r
        self.visualize = display
        self.state = state
        self._offset_l = np.zeros(GLOVE_DOF)
        self._offset_r = np.zeros(GLOVE_DOF)

        # Load URDF models for gloves
        urdf_path_left = _resolve_project_file("urdf/DexGlove_L_v4.urdf")
        self._glove_l = RobotWrapper.BuildFromURDF(
            urdf_path_left,
            package_dirs=_urdf_package_dirs(urdf_path_left),
        )

        urdf_path_right = _resolve_project_file("urdf/DexGlove_R_v4.urdf")
        self._glove_r = RobotWrapper.BuildFromURDF(
            urdf_path_right,
            package_dirs=_urdf_package_dirs(urdf_path_right),
        )

        self.q_l = np.zeros(GLOVE_DOF)
        self.q_r = np.zeros(GLOVE_DOF)
        self.q_l_original = np.zeros(GLOVE_DOF)
        self.q_r_original = np.zeros(GLOVE_DOF)

        # Setup visualization if enabled
        if self.visualize:
            self._setup_visualization()

    def _setup_visualization(self) -> None:
        """Sets up MeshCat visualization for both gloves."""
        # Left glove visualizer
        self._viz_l = MeshcatVisualizer(
            self._glove_l.model,
            self._glove_l.collision_model,
            self._glove_l.visual_model,
        )
        self._viz_l.initViewer(open=True)
        self._viz_l.loadViewerModel(color=VIZ_COLOR_SEMI_TRANSPARENT)
        pin.forwardKinematics(self._glove_l.model, self._glove_l.data, self.q_l_original)
        pin.updateFramePlacements(self._glove_l.model, self._glove_l.data)
        self._viz_l.display(self.q_l_original)

        # Right glove visualizer
        self._viz_r = MeshcatVisualizer(
            self._glove_r.model,
            self._glove_r.collision_model,
            self._glove_r.visual_model,
        )
        self._viz_r.initViewer(open=True)
        self._viz_r.loadViewerModel(color=VIZ_COLOR_SEMI_TRANSPARENT)
        pin.forwardKinematics(self._glove_r.model, self._glove_r.data, self.q_r_original)
        pin.updateFramePlacements(self._glove_r.model, self._glove_r.data)
        self._viz_r.display(self.q_r_original)

    def get_finger_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the pose of all finger tips for both hands.

        Computes forward kinematics and extracts the position and orientation
        of each finger tip in the form of translation (x,y,z) and quaternion
        (x,y,z,w).

        Returns:
            Tuple of (left_finger_poses, right_finger_poses) where each is a
            numpy array of shape (5, 7) containing [x, y, z, qx, qy, qz, qw]
            for each of the 5 fingers.
        """
        # Compute forward kinematics for left glove
        pin.forwardKinematics(self._glove_l.model, self._glove_l.data, self.q_l_original)
        pin.updateFramePlacements(self._glove_l.model, self._glove_l.data)

        # Compute forward kinematics for right glove
        pin.forwardKinematics(self._glove_r.model, self._glove_r.data, self.q_r_original)
        pin.updateFramePlacements(self._glove_r.model, self._glove_r.data)

        # Get left hand finger tip transforms
        frame_id=self._glove_l.model.getFrameId("glove_link_l_1_5")
        finger_l_1_trans = self._glove_l.data.oMf[frame_id]
        frame_id=self._glove_l.model.getFrameId("glove_link_l_2_4")
        finger_l_2_trans = self._glove_l.data.oMf[frame_id]
        frame_id=self._glove_l.model.getFrameId("glove_link_l_3_4")
        finger_l_3_trans = self._glove_l.data.oMf[frame_id]
        frame_id=self._glove_l.model.getFrameId("glove_link_l_4_4")
        finger_l_4_trans = self._glove_l.data.oMf[frame_id]
        frame_id=self._glove_l.model.getFrameId("glove_link_l_5_4")
        finger_l_5_trans = self._glove_l.data.oMf[frame_id]

        # Get right hand finger tip transforms
        frame_id=self._glove_r.model.getFrameId("glove_link_r_1_5")
        finger_r_1_trans = self._glove_r.data.oMf[frame_id]
        frame_id=self._glove_r.model.getFrameId("glove_link_r_2_4")
        finger_r_2_trans = self._glove_r.data.oMf[frame_id]
        frame_id=self._glove_r.model.getFrameId("glove_link_r_3_4")
        finger_r_3_trans = self._glove_r.data.oMf[frame_id]
        frame_id=self._glove_r.model.getFrameId("glove_link_r_4_4")
        finger_r_4_trans = self._glove_r.data.oMf[frame_id]
        frame_id=self._glove_r.model.getFrameId("glove_link_r_5_4")
        finger_r_5_trans = self._glove_r.data.oMf[frame_id]

        # Convert left hand transforms to position + quaternion format
        left_finger_poses = np.zeros((5, 7))
        for i, trans in enumerate([finger_l_1_trans, finger_l_2_trans, finger_l_3_trans,
                                     finger_l_4_trans, finger_l_5_trans]):
            # Extract position (x, y, z)
            left_finger_poses[i, 0:3] = trans.translation
            # Extract rotation as quaternion (x, y, z, w)
            rotation = Rotation.from_matrix(trans.rotation)
            quat = rotation.as_quat()  # Returns [x, y, z, w]
            left_finger_poses[i, 3:7] = quat

        # Convert right hand transforms to position + quaternion format
        right_finger_poses = np.zeros((5, 7))
        for i, trans in enumerate([finger_r_1_trans, finger_r_2_trans, finger_r_3_trans,
                                     finger_r_4_trans, finger_r_5_trans]):
            # Extract position (x, y, z)
            right_finger_poses[i, 0:3] = trans.translation
            # Extract rotation as quaternion (x, y, z, w)
            rotation = Rotation.from_matrix(trans.rotation)
            quat = rotation.as_quat()  # Returns [x, y, z, w]
            right_finger_poses[i, 3:7] = quat

        return left_finger_poses, right_finger_poses

    def update_frame(self, q_l: np.ndarray, q_r: np.ndarray) -> None:
        """Updates finger joint angles from glove sensor data.

        Processes raw glove data, applies calibration offsets, and maps
        the data to robot hand joint positions using the selected strategy.

        Args:
            q_l: Left glove raw joint positions.
            q_r: Right glove raw joint positions.
        """
        self.q_r = q_r.copy()
        self.q_l = q_l.copy()
        self.q_r_original = q_r.copy()
        self.q_l_original = q_l.copy()
        # Handle calibration
        
    
        # Apply mapping strategy
        if self.mapping_type == MappingType.joint_matching:
            self._apply_joint_matching()
        elif self.mapping_type == MappingType.adaptive_mapping:
            self._apply_adaptive_mapping()
    def _apply_adaptive_mapping(self) -> None:
        return 0

    def _apply_joint_matching(self) -> None:
        """Applies joint-to-joint matching mapping strategy.

        Maps glove joint positions to robot hand joints using a direct
        joint matching approach with specific scaling factors for each joint.
        """
        if self.state.get('calibration_hand_r', False):
            self._offset_r = self.q_r.copy()
            self.state['calibration_hand_r'] = False

        if self.state.get('calibration_hand_l', False):
            self._offset_l = self.q_l.copy()
            self.state['calibration_hand_l'] = False
            
        # Apply calibration offsets
        self.q_l -= self._offset_l
        self.q_r -= self._offset_r
        # Map left hand
        for i in range(NUM_FINGERS):
            base_idx = i * JOINTS_PER_FINGER
            glove_idx = i * JOINTS_PER_FINGER + 1

            # Distribute distal joint angle across finger segments
            self.q_hand_l[base_idx + 3] = (
                self.q_l[glove_idx + 3] * FINGER_TIP_SCALE
            )
            self.q_hand_l[base_idx + 2] = (
                self.q_l[glove_idx + 3] * FINGER_MID_SCALE
            )
            self.q_hand_l[base_idx + 1] = self.q_l[glove_idx + 1]
            self.q_hand_l[base_idx] = self.q_l[glove_idx]

        # Apply thumb base adjustment
        self.q_hand_l[0] += self.q_l[0] * THUMB_BASE_SCALE

        # Negate specific joints
        for idx in LEFT_HAND_NEGATE_INDICES:
            self.q_hand_l[idx] = -self.q_hand_l[idx]

        self.q_hand_l[:GLOVE_HAND_DOF] = np.round(
            self.q_hand_l, decimals=ANGLE_DECIMAL_PRECISION
        )[:GLOVE_HAND_DOF]

        # Map right hand (mirrored)
        for i in range(NUM_FINGERS):
            base_idx = i * JOINTS_PER_FINGER
            glove_idx = i * JOINTS_PER_FINGER + 1

            self.q_hand_r[base_idx + 3] = (
                -self.q_r[glove_idx + 3] * FINGER_TIP_SCALE
            )
            self.q_hand_r[base_idx + 2] = (
                -self.q_r[glove_idx + 3] * FINGER_MID_SCALE
            )
            self.q_hand_r[base_idx + 1] = -self.q_r[glove_idx + 1]
            self.q_hand_r[base_idx] = -self.q_r[glove_idx]

        # Apply thumb base adjustment
        self.q_hand_r[0] -= self.q_r[0] * THUMB_BASE_SCALE

        # Negate specific joints
        for idx in RIGHT_HAND_NEGATE_INDICES:
            self.q_hand_r[idx] = -self.q_hand_r[idx]

        self.q_hand_r[:GLOVE_HAND_DOF] = np.round(
            self.q_hand_r, decimals=ANGLE_DECIMAL_PRECISION
        )[:GLOVE_HAND_DOF]
        
        # Update visualization if enabled
        if self.visualize:
            self._viz_l.display(self.q_l_original)
            self._viz_r.display(self.q_r_original)


class Listener:
    """Socket server for receiving exoskeleton and glove sensor data.

    This class implements a TCP socket server that receives binary sensor data
    from the DexCap hardware, parses it, and makes it available as numpy arrays.

    Attributes:
        q_l: Left glove joint positions (21 DOF).
        q_r: Right glove joint positions (21 DOF).
        received_array: Exoskeleton joint positions (23 DOF).
        vibe: Vibration feedback data (currently unused).
        port: TCP port number for socket server.
        vibration: Vibration feedback enabled flag (currently unused).
    """

    def __init__(self, port: int = DEFAULT_SOCKET_PORT) -> None:
        """Initializes the socket listener.

        Args:
            port: TCP port number to listen on.
        """
        self.q_l = np.zeros(GLOVE_DOF)
        self.q_r = np.zeros(GLOVE_DOF)
        self.received_array = np.zeros(EXOSKELETON_DOF)
        self.vibe = np.zeros(10)
        self.port = port
        self.vibration = False

    def listener(self) -> None:
        """Runs the socket server loop.

        Creates a TCP socket server, accepts client connections, and
        continuously receives and parses sensor data packets. This method
        runs indefinitely until the program is terminated.

        The data packet format:
        - Header (2 bytes): Device flags indicating which devices sent data
        - Left glove data (21 uint16 values)
        - Backplate/exoskeleton data (23 uint16 values)
        - Right glove data (21 uint16 values)
        """
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((SOCKET_LISTEN_HOST, self.port))
        server_socket.listen(SOCKET_BACKLOG)
        print(f"遥操作数据：服务器已启动，正在监听端口 {self.port}...")

        while True:
            client_socket, addr = server_socket.accept()
            print(f"与 {addr} 已建立连接。")

            try:
                self._handle_client(client_socket)
            except Exception as e:
                print(f"客户端处理错误: {e}")
            finally:
                client_socket.close()

    def _handle_client(self, client_socket: socket.socket) -> None:
        """Handles data reception from a connected client.

        Args:
            client_socket: Connected client socket.
        """
        while True:
            data = self._recv_exact(client_socket, SOCKET_BUFFER_SIZE)

            if not data:
                break

            try:
                dev_flags=self._parse_data_packet(data)
                if self.vibration:
                
                    data=np.ones(11,dtype=np.uint8)*0
                    data[0] = 0xC0          #振动启动标志位 0x80=左手 0x40=右手 0xC0=双手
                    data[1:11]=self.vibe.copy() #振动数据位
                    packed_data = struct.pack('<11B', *data)
                            
                    client_socket.sendall(packed_data)
                    print(f"成功发送数据: {data}")
                    print(f"成功发送数据: {packed_data}")
                    self.vibration=False  
            except Exception as e:
                print(f"数据解析错误: {e}")
                continue

    @staticmethod
    def _recv_exact(client_socket: socket.socket, size: int) -> Optional[bytes]:
        """从 TCP 流中读满一帧，避免半包导致解析错帧。"""
        chunks = []
        remaining = size
        while remaining > 0:
            chunk = client_socket.recv(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)

    def _parse_data_packet(self, data: bytes) -> np.ndarray:
        """Parses a binary data packet from the hardware.

        Args:
            data: Raw binary data packet.

        Raises:
            ValueError: If packet size is incorrect.
        """
        # Check packet size
        n = len(data) // DATA_TYPE().nbytes
        if n != PACKET_SIZE:
            raise ValueError(f"数据丢帧: 期望 {PACKET_SIZE}，收到 {n}")

        # Parse header to determine which devices sent data
        header_value = struct.unpack('<H', data[:PACKET_HEADER_SIZE])[0]
        device_flags = bin(header_value)[2:].zfill(BINARY_STRING_WIDTH)

        # Parse full data array
        array = np.frombuffer(
            data[:n * DATA_TYPE().nbytes], dtype=DATA_TYPE
        )

        # Parse left glove data
        if device_flags[0] == '1':
            self._parse_left_glove(array[LEFT_GLOVE_START:LEFT_GLOVE_END])

        # Parse backplate/exoskeleton data
        if device_flags[1] == '1':
            self._parse_backplate(array[BACKPLATE_START:BACKPLATE_END])

        # Parse right glove data
        if device_flags[2] == '1':
            self._parse_right_glove(array[RIGHT_GLOVE_START:RIGHT_GLOVE_END])

        return device_flags
       
    


    def _parse_left_glove(self, array_l: np.ndarray) -> None:
        """Parses left glove sensor data.

        Args:
            array_l: Raw left glove data array.
        """
        angles = array_l / ENCODER_TO_DEGREES
        rad = angles * DEGREES_TO_RADIANS
        rad = DexCap_UL._normalize_angles(rad)

        # Rearrange finger data (data comes in reverse order)
        finger_1 = np.flip(rad[:5])
        finger_2 = np.flip(rad[5:9])
        finger_3 = np.flip(rad[9:13])
        finger_4 = np.flip(rad[13:17])
        finger_5 = np.flip(rad[17:21])

        self.q_l = np.hstack((finger_1, finger_2, finger_3, finger_4, finger_5))

    def _parse_backplate(self, array_m: np.ndarray) -> None:
        """Parses backplate/exoskeleton sensor data.

        Args:
            array_m: Raw backplate data array.
        """
        # Make a writable copy since np.frombuffer creates read-only array
        array_m = array_m.copy()

        # V3 wiring order is from distal to proximal, so flip
        # Left arm
        array_m[:LEFT_ARM_SIZE] = np.flip(array_m[:LEFT_ARM_SIZE])

        # Right arm
        array_m[RIGHT_ARM_START:RIGHT_ARM_END] = np.flip(
            array_m[RIGHT_ARM_START:RIGHT_ARM_END]
        )

        self.received_array = array_m

    def _parse_right_glove(self, array_r: np.ndarray) -> None:
        """Parses right glove sensor data.

        Args:
            array_r: Raw right glove data array.
        """
        angles = array_r / ENCODER_TO_DEGREES
        rad = angles * DEGREES_TO_RADIANS
        rad = DexCap_UL._normalize_angles(rad)

        # Rearrange finger data (data comes in reverse order)
        finger_1 = np.flip(rad[:5])
        finger_2 = np.flip(rad[5:9])
        finger_3 = np.flip(rad[9:13])
        finger_4 = np.flip(rad[13:17])
        finger_5 = np.flip(rad[17:21])

        self.q_r = np.hstack((finger_1, finger_2, finger_3, finger_4, finger_5))
