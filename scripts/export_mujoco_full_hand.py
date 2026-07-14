#!/usr/bin/env python3
"""Export the full 21-point human-hand MuJoCo MJCF model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hand_reconstruction.mujoco_export import build_full_hand_mjcf

DEFAULT_OUTPUT = REPO_ROOT / "mujoco" / "human_full_hand.xml"


def export_full_hand_mjcf(output_path: Path = DEFAULT_OUTPUT, *, kinematic: bool = False) -> Path:
    """Write the generated full-hand MJCF to ``output_path``.

    ``kinematic=False`` (default) writes the realistic solid sculpt;
    ``kinematic=True`` writes the transparent kinematic baseline.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_full_hand_mjcf(realistic=not kinematic), encoding="utf-8")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output MJCF path. Default: mujoco/human_full_hand.xml",
    )
    parser.add_argument(
        "--kinematic",
        action="store_true",
        help="Write the transparent kinematic baseline instead of the realistic sculpt.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_path = export_full_hand_mjcf(args.output, kinematic=args.kinematic)
    print(f"Wrote {output_path} ({'kinematic baseline' if args.kinematic else 'realistic sculpt'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
