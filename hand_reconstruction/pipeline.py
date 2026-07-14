"""High-level hand reconstruction pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .glove_observation import (
    DexGloveObserver,
    GloveLinkObservation,
    SkeletonInitializer,
    direct_skeleton_from_observation,
)
from .human_hand_model import HumanHandSkeleton


class HandReconstructionPipeline:
    """Convert DexGlove joint arrays into a first-pass human hand skeleton."""

    def __init__(self, urdf_path: str | Path, hand: str) -> None:
        self.observer = DexGloveObserver(urdf_path=urdf_path, hand=hand)
        self.initializer = SkeletonInitializer()

    def reconstruct(self, q: np.ndarray) -> HumanHandSkeleton:
        observation = self.observer.observe(q)
        return self.initializer.from_observation(observation)

    def reconstruct_direct(self, q: np.ndarray) -> HumanHandSkeleton:
        """Direct link-position skeleton: landmarks equal glove link origins.

        No template, no blending, so the skeleton is frame-aligned with the
        glove URDF (correct orientation and handedness) by construction.
        """
        observation = self.observer.observe(q)
        return direct_skeleton_from_observation(observation)


def reconstruct_from_link_positions(
    hand: str,
    link_positions: dict[str, np.ndarray],
) -> HumanHandSkeleton:
    """Build a skeleton from already-computed DexGlove link positions."""
    if "base_link" not in link_positions:
        raise KeyError("link_positions must include base_link")
    observation = GloveLinkObservation(
        hand=hand,
        base_position=link_positions["base_link"],
        link_positions=link_positions,
    )
    return SkeletonInitializer().from_observation(observation)
