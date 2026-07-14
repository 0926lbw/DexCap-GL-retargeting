#!/usr/bin/env python3
"""Export the four-finger human-hand MuJoCo MJCF model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hand_reconstruction.mujoco_export import build_four_finger_mjcf

DEFAULT_OUTPUT = REPO_ROOT / "mujoco" / "human_four_finger_hand.xml"


def export_four_finger_mjcf(output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Write the generated four-finger MJCF to ``output_path``.

    The four-finger artifact stays the transparent kinematic baseline
    (``realistic=False``) so it remains a stable first-pass reference; the full
    21-point hand is where the realistic sculpt lives.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_four_finger_mjcf(realistic=False), encoding="utf-8")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output MJCF path. Default: mujoco/human_four_finger_hand.xml",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_path = export_four_finger_mjcf(args.output)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
