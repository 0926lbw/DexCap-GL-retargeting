"""Fixed-length fingertip targeting for 21-point hand landmarks."""

from __future__ import annotations

import numpy as np

from .schema import (
    FINGER_CHAINS,
    INDEX_TIP,
    MIDDLE_TIP,
    NUM_LANDMARKS,
    PINKY_TIP,
    RING_TIP,
    THUMB_TIP,
)

FINGERTIP_INDICES = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

__all__ = ["FINGERTIP_INDICES", "fuse_tip_locked_landmarks"]


def fuse_tip_locked_landmarks(human_landmarks, direct_landmarks, *, eps=1e-9):
    """Fuse human-shaped landmarks with fixed-length fingertip targeting.

    Roots and non-finger landmarks stay on the human skeleton. Finger segments
    keep the human FK bone lengths. Fingertips move toward the direct glove
    targets by bending the chain; targets outside the human finger's reach leave
    a residual instead of stretching the bones.
    """

    eps = _validate_eps(eps)

    human = _validate_landmarks("human_landmarks", human_landmarks)
    direct = _validate_landmarks("direct_landmarks", direct_landmarks)

    fused = human.copy()
    for chain in FINGER_CHAINS.values():
        chain_indices = list(chain[1:])
        solved = _solve_fixed_length_chain(
            human[chain_indices],
            direct[chain_indices],
            eps,
        )
        for idx, point in zip(chain_indices, solved):
            fused[idx] = point

    return fused


def _validate_eps(eps):
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a finite positive value")
    return eps


def _validate_landmarks(name, landmarks):
    points = np.asarray(landmarks, dtype=float)
    expected_shape = (NUM_LANDMARKS, 3)
    if points.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}; got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    return points


def _solve_fixed_length_chain(human_chain, direct_chain, eps):
    root = human_chain[0].copy()
    target = direct_chain[-1].copy()
    lengths = _segment_lengths(human_chain)
    total_length = float(np.sum(lengths))

    if total_length <= eps:
        return np.repeat(root[None, :], len(human_chain), axis=0)

    target_vec = target - root
    target_distance = float(np.linalg.norm(target_vec))
    preferred_axis = _preferred_axis(human_chain, eps)

    if target_distance >= total_length - eps:
        direction = _unit_or_fallback(target_vec, preferred_axis, eps)
        return _straight_chain(root, lengths, direction)

    initial = _initial_guided_chain(
        human_chain,
        direct_chain,
        target_vec,
        preferred_axis,
        eps,
    )
    return _fabrik(initial, lengths, root, target, eps)


def _segment_lengths(points):
    return np.asarray(
        [
            np.linalg.norm(points[i + 1] - points[i])
            for i in range(len(points) - 1)
        ],
        dtype=float,
    )


def _straight_chain(root, lengths, direction):
    solved = np.empty((len(lengths) + 1, 3), dtype=float)
    solved[0] = root
    for i, length in enumerate(lengths, start=1):
        solved[i] = solved[i - 1] + direction * length
    return solved


def _initial_guided_chain(
    human_chain,
    direct_chain,
    target_vec,
    preferred_axis,
    eps,
):
    root = human_chain[0]
    source_vec = human_chain[-1] - root
    source_axis = _unit_or_fallback(source_vec, preferred_axis, eps)
    target_axis = _unit_or_fallback(target_vec, source_axis, eps)
    bend_dir = _direct_bend_direction(direct_chain, target_axis, eps)

    if bend_dir is None:
        rotation = _rotation_between(source_axis, target_axis, eps)
        initial = np.asarray(
            [root + rotation.dot(point - root) for point in human_chain],
            dtype=float,
        )
    else:
        initial = np.empty_like(human_chain, dtype=float)
        for i, point in enumerate(human_chain):
            relative = point - root
            axial_distance = float(np.dot(relative, source_axis))
            bend_offset = relative - source_axis * axial_distance
            bend_magnitude = float(np.linalg.norm(bend_offset))
            initial[i] = (
                root
                + target_axis * axial_distance
                + bend_dir * bend_magnitude
            )

    initial[0] = root
    return initial


def _direct_bend_direction(direct_chain, target_axis, eps):
    direct_root = direct_chain[0]
    best = None
    best_norm = 0.0
    for point in direct_chain[1:-1]:
        relative = point - direct_root
        perpendicular = relative - target_axis * float(np.dot(relative, target_axis))
        norm = float(np.linalg.norm(perpendicular))
        if norm > best_norm:
            best = perpendicular
            best_norm = norm

    if best is None or best_norm <= eps:
        return None
    return best / best_norm


def _fabrik(initial, lengths, root, target, eps, max_iterations=80):
    solved = np.asarray(initial, dtype=float).copy()
    fallback_dirs = _fallback_directions(solved, target - root, eps)
    tolerance = max(10.0 * eps, 1e-10)

    for _ in range(max_iterations):
        solved[-1] = target
        for i in range(len(solved) - 2, -1, -1):
            length = lengths[i]
            if length <= eps:
                solved[i] = solved[i + 1]
                continue
            direction = _unit_or_fallback(
                solved[i] - solved[i + 1],
                -fallback_dirs[i],
                eps,
            )
            solved[i] = solved[i + 1] + direction * length

        solved[0] = root
        for i in range(1, len(solved)):
            length = lengths[i - 1]
            if length <= eps:
                solved[i] = solved[i - 1]
                continue
            direction = _unit_or_fallback(
                solved[i] - solved[i - 1],
                fallback_dirs[i - 1],
                eps,
            )
            solved[i] = solved[i - 1] + direction * length

        if np.linalg.norm(solved[-1] - target) <= tolerance:
            break

    return solved


def _fallback_directions(points, target_vec, eps):
    target_axis = _unit_or_fallback(target_vec, np.array([0.0, 1.0, 0.0]), eps)
    directions = []
    for i in range(len(points) - 1):
        directions.append(
            _unit_or_fallback(points[i + 1] - points[i], target_axis, eps)
        )
    return directions


def _preferred_axis(points, eps):
    full = points[-1] - points[0]
    if np.linalg.norm(full) > eps:
        return full / np.linalg.norm(full)
    for i in range(len(points) - 1):
        segment = points[i + 1] - points[i]
        norm = np.linalg.norm(segment)
        if norm > eps:
            return segment / norm
    return np.array([0.0, 1.0, 0.0])


def _unit_or_fallback(vector, fallback, eps):
    norm = float(np.linalg.norm(vector))
    if norm > eps:
        return vector / norm
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm > eps:
        return fallback / fallback_norm
    return np.array([0.0, 1.0, 0.0])


def _rotation_between(source_vec, target_vec, eps):
    source = source_vec / np.linalg.norm(source_vec)
    target = target_vec / np.linalg.norm(target_vec)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))

    if dot >= 1.0 - eps:
        return np.eye(3)

    if dot <= -1.0 + eps:
        axis = _orthogonal_unit(source)
        return _rotation_from_axis_angle(axis, np.pi)

    axis = np.cross(source, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= eps:
        return np.eye(3)

    axis = axis / axis_norm
    angle = np.arccos(dot)
    return _rotation_from_axis_angle(axis, angle)


def _orthogonal_unit(vector):
    basis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(vector, basis)) > 0.9:
        basis = np.array([0.0, 1.0, 0.0])
    orthogonal = basis - vector * np.dot(vector, basis)
    return orthogonal / np.linalg.norm(orthogonal)


def _rotation_from_axis_angle(axis, angle):
    x, y, z = axis
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )
    identity = np.eye(3)
    return identity + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew.dot(skew)
