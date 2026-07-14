#!/usr/bin/env python3
"""Reconstruct one 21-point hand skeleton frame from a DexGlove q array."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hand_reconstruction.export import write_skeleton_json, write_skeleton_npy
from hand_reconstruction.pipeline import HandReconstructionPipeline
from hand_reconstruction.visualize_meshcat import MeshcatHandSkeletonViewer


DEFAULT_URDFS = {
    "left": "DexCap_v4/DexGlove_L_v4/urdf/DexGlove_L_v4.urdf",
    "right": "DexCap_v4/DexGlove_R_v4/urdf/DexGlove_R_v4.urdf",
}


def main() -> int:
    args = _parse_args()
    urdf_path = Path(args.urdf) if args.urdf else REPO_ROOT / DEFAULT_URDFS[args.hand]
    q = _parse_q(args.q)

    pipeline = HandReconstructionPipeline(urdf_path=urdf_path, hand=args.hand)
    skeleton = pipeline.reconstruct(q)

    if args.output_json:
        write_skeleton_json(skeleton, args.output_json)
    if args.output_npy:
        write_skeleton_npy(skeleton, args.output_npy)
    if args.meshcat:
        MeshcatHandSkeletonViewer().display(skeleton)
        if args.meshcat_wait:
            try:
                _wait_for_meshcat()
            except KeyboardInterrupt:
                pass
    if not args.output_json and not args.output_npy and not args.meshcat:
        print(skeleton.to_numpy())
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=("left", "right"), required=True)
    parser.add_argument(
        "--q",
        required=True,
        help="Comma-separated 21-DOF DexGlove joint array in radians.",
    )
    parser.add_argument("--urdf", help="Optional DexGlove URDF path.")
    parser.add_argument("--output-json", help="Output skeleton JSON path.")
    parser.add_argument("--output-npy", help="Output skeleton NPY path.")
    parser.add_argument(
        "--meshcat",
        action="store_true",
        help="Display the reconstructed 21-point skeleton in MeshCat.",
    )
    parser.add_argument(
        "--meshcat-wait",
        action="store_true",
        help="Keep the MeshCat process alive until Ctrl+C. Implies --meshcat.",
    )
    args = parser.parse_args()
    if args.meshcat_wait:
        args.meshcat = True
    return args


def _parse_q(value: str) -> np.ndarray:
    items = [item.strip() for item in value.split(",") if item.strip()]
    q = np.asarray([float(item) for item in items], dtype=float)
    if q.shape != (21,):
        raise ValueError(f"--q must contain 21 comma-separated values, got {q.shape[0]}")
    return q


def _wait_for_meshcat() -> None:
    while True:
        time.sleep(3600.0)


if __name__ == "__main__":
    raise SystemExit(main())
