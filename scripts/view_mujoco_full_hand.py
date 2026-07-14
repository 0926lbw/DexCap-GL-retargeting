#!/usr/bin/env python3
"""View the generated full 21-point MuJoCo hand model."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = REPO_ROOT / "mujoco" / "human_full_hand.xml"
FINGERS = ("index", "middle", "ring", "pinky")
# The thumb has joints + sites only (no actuators), so it is posed via qpos.
THUMB_JOINTS = ("thumb_cmc_abd", "thumb_cmc_flex", "thumb_mcp", "thumb_ip")


def main() -> int:
    args = _parse_args()
    mujoco = _import_mujoco()

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    _apply_pose(mujoco, model, data, grasp=args.grasp)

    _launch_viewer(mujoco, model, data, args)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML,
        help="MJCF XML path. Default: mujoco/human_full_hand.xml",
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
    parser.add_argument(
        "--grasp",
        action="store_true",
        help="Apply a natural pinch demo pose (curled fingers + thumb opposing "
        "the fingertips). Without this flag the hand is shown in a neutral "
        "open pose so the skeleton reads clearly as a hand.",
    )
    return parser.parse_args()


def _import_mujoco():
    import mujoco

    if not hasattr(mujoco, "MjModel"):
        raise RuntimeError(
            "Imported a non-official 'mujoco' namespace. Run this script as "
            "`python scripts/view_mujoco_full_hand.py`, or run it from "
            "outside the repo root so the local mujoco/ output directory does "
            "not shadow the package."
        )
    return mujoco


def _apply_pose(mujoco, model, data, *, grasp: bool) -> None:
    if not grasp:
        # Neutral open-hand rest pose: nothing to set, just finalize kinematics
        # so the skeleton reads clearly as a hand on first launch.
        mujoco.mj_forward(model, data)
        return

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

    # Thumb: natural opposition toward the curled fingertips. The thumb is
    # joints-only, so it is driven through qpos. Angles were tuned so the thumb
    # tip meets the curled index/middle fingertips: adduction (negative cmc_abd)
    # pulls it medially and flexion curls it forward into the palm -- positive
    # cmc_abd/cmc_flex point the thumb laterally/upward instead of opposing.
    thumb_pose = {
        "thumb_cmc_abd": -25.0,
        "thumb_cmc_flex": 40.0,
        "thumb_mcp": 30.0,
        "thumb_ip": 20.0,
    }
    for joint_name in THUMB_JOINTS:
        _set_joint_qpos(mujoco, model, data, joint_name, math.radians(thumb_pose[joint_name]))

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
