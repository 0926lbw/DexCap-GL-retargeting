#!/usr/bin/env python3
"""View the generated four-finger MuJoCo hand model."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = REPO_ROOT / "mujoco" / "human_four_finger_hand.xml"
FINGERS = ("index", "middle", "ring", "pinky")


def main() -> int:
    args = _parse_args()
    mujoco = _import_mujoco()

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    _apply_demo_pose(mujoco, model, data)

    _launch_viewer(mujoco, model, data, args)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML,
        help="MJCF XML path. Default: mujoco/human_four_finger_hand.xml",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Interactive viewer duration in seconds. 0 keeps it open.",
    )
    parser.add_argument("--azimuth", type=float, default=-135.0)
    parser.add_argument("--elevation", type=float, default=-35.0)
    parser.add_argument("--distance", type=float, default=0.24)
    return parser.parse_args()


def _import_mujoco():
    import mujoco

    if not hasattr(mujoco, "MjModel"):
        raise RuntimeError(
            "Imported a non-official 'mujoco' namespace. Run this script as "
            "`python scripts/view_mujoco_four_finger_hand.py`, or run it from "
            "outside the repo root so the local mujoco/ output directory does "
            "not shadow the package."
        )
    return mujoco


def _apply_demo_pose(mujoco, model, data) -> None:
    flex_pose = {
        "index": (6.0, 52.0, 66.0),
        "middle": (0.0, 58.0, 72.0),
        "ring": (-5.0, 55.0, 68.0),
        "pinky": (-10.0, 48.0, 62.0),
    }
    for finger in FINGERS:
        abd_deg, mcp_deg, pip_deg = flex_pose[finger]
        _set_joint_qpos(mujoco, model, data, f"{finger}_mcp_abd", math.radians(abd_deg))
        _set_joint_qpos(mujoco, model, data, f"{finger}_mcp_flex", math.radians(mcp_deg))
        _set_joint_qpos(mujoco, model, data, f"{finger}_pip", math.radians(pip_deg))
        _set_joint_qpos(mujoco, model, data, f"{finger}_dip", math.radians(pip_deg * 0.6))
        _set_actuator_ctrl(mujoco, model, data, f"{finger}_mcp_abd_act", math.radians(abd_deg))
        _set_actuator_ctrl(mujoco, model, data, f"{finger}_mcp_flex_act", math.radians(mcp_deg))
        _set_actuator_ctrl(mujoco, model, data, f"{finger}_pip_act", math.radians(pip_deg))
    mujoco.mj_forward(model, data)


def _set_joint_qpos(mujoco, model, data, joint_name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Unknown joint: {joint_name}")
    data.qpos[model.jnt_qposadr[joint_id]] = value


def _set_actuator_ctrl(mujoco, model, data, actuator_name: str, value: float) -> None:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if actuator_id < 0:
        raise ValueError(f"Unknown actuator: {actuator_name}")
    data.ctrl[actuator_id] = value


def _launch_viewer(mujoco, model, data, args: argparse.Namespace) -> None:
    import mujoco.viewer

    started_at = time.monotonic()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.075, -0.015)
        viewer.cam.distance = args.distance
        viewer.cam.azimuth = args.azimuth
        viewer.cam.elevation = args.elevation
        while viewer.is_running():
            viewer.sync()
            if args.duration > 0.0 and time.monotonic() - started_at >= args.duration:
                break
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    raise SystemExit(main())
