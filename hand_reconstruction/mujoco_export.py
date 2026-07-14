"""MuJoCo MJCF export helpers for the human hand skeleton."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np

from .human_hand_params import HumanHandParams, default_params

FOUR_FINGER_NAMES = ("index", "middle", "ring", "pinky")
DIP_PIP_COUPLING_RATIO = 0.6
DEFAULT_SITE_SIZE = 0.004
DEFAULT_PHALANX_RADIUS = 0.006
DEFAULT_EMPTY_JOINT_BODY_MASS = 0.001
DEFAULT_EMPTY_JOINT_BODY_DIAGINERTIA = 1e-6

# Viewer geom groups. Group 2 = realistic solid sculpt; group 1 = transparent
# kinematic baseline sticks; group 0 reserved for a future collision proxy.
GROUP_REALISTIC = 2
GROUP_BASELINE = 1

# Realistic solid-sculpt color palette (alpha 1).
SKIN_RGBA = "0.88 0.69 0.56 1"
JOINT_RGBA = "0.82 0.60 0.50 1"
PALM_RGBA = "0.85 0.64 0.53 1"
THENAR_RGBA = "0.86 0.65 0.54 1"
PAD_RGBA = "0.85 0.62 0.52 1"
NAIL_RGBA = "0.93 0.82 0.74 1"
# Transparent kinematic baseline colors (match the original first-pass model).
BASELINE_GEOM_RGBA = "0.76 0.82 0.88 0.55"
BASELINE_PALM_RGBA = "0.55 0.62 0.70 0.35"

# Overlap the two tapered-capsule halves so the 0.5L seam hides inside both.
TAPER_OVERLAP = 0.0015


def build_four_finger_mjcf(
    params: HumanHandParams | None = None,
    *,
    hand: str = "right",
    realistic: bool = True,
    keep_baseline: bool = False,
) -> str:
    """Return a four-finger-only human-hand MJCF model.

    The model intentionally omits the thumb and contact dynamics. It is a
    kinematic first pass for validating joint axes, limits, passive DIP/PIP
    coupling, and keypoint site extraction in MuJoCo. For the complete 21-point
    hand (with thumb), see :func:`build_full_hand_mjcf`.

    ``realistic=True`` emits a solid sculpted hand (group 2); ``realistic=False``
    emits the transparent kinematic baseline (group 1) and is byte-identical to
    the original first-pass model. ``keep_baseline=True`` additionally keeps the
    transparent baseline visible alongside the sculpt.
    """
    params = _resolve_params(params, hand)
    root, wrist = _scaffold_mjcf(
        model_name="human_four_finger_hand",
        params=params,
        realistic=realistic,
        keep_baseline=keep_baseline,
    )
    _add_four_fingers(wrist, root, params, realistic=realistic, keep_baseline=keep_baseline)
    return _pretty_xml(root)


def build_full_hand_mjcf(
    params: HumanHandParams | None = None,
    *,
    hand: str = "right",
    realistic: bool = True,
    keep_baseline: bool = False,
) -> str:
    """Return the full 21-point human-hand MJCF model.

    This is the four-finger model (:func:`build_four_finger_mjcf`) extended with
    an articulated thumb (CMC -> MCP -> IP) so the kinematic skeleton carries all
    21 MediaPipe keypoints as sites. The four non-thumb fingers keep their
    position actuators and passive DIP/PIP coupling exactly as in the four-finger
    model. The thumb is joints-only for now (no actuators, no coupling) -- a
    geometric baseline that can be posed directly through ``qpos``.

    ``realistic`` / ``keep_baseline`` select the visual layer as in
    :func:`build_four_finger_mjcf`; kinematics are identical across all modes.
    """
    params = _resolve_params(params, hand)
    root, wrist = _scaffold_mjcf(
        model_name="human_full_hand",
        params=params,
        realistic=realistic,
        keep_baseline=keep_baseline,
    )
    _add_four_fingers(wrist, root, params, realistic=realistic, keep_baseline=keep_baseline)
    _add_thumb(wrist, params, realistic=realistic, keep_baseline=keep_baseline)
    return _pretty_xml(root)


def _resolve_params(params: HumanHandParams | None, hand: str) -> HumanHandParams:
    if hand not in {"left", "right"}:
        raise ValueError("hand must be 'left' or 'right'")
    params = default_params() if params is None else params
    return params.for_hand(hand)


def _scaffold_mjcf(
    *,
    model_name: str,
    params: HumanHandParams,
    realistic: bool,
    keep_baseline: bool,
) -> tuple[ET.Element, ET.Element]:
    """Return ``(root, wrist)`` with compiler/option/visual/defaults and the wrist body."""
    root = ET.Element("mujoco", {"model": model_name})
    ET.SubElement(root, "compiler", {"angle": "degree", "autolimits": "true"})
    ET.SubElement(root, "option", {"timestep": "0.002", "gravity": "0 0 0"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "960"})
    _add_default(root)

    worldbody = ET.SubElement(root, "worldbody")
    wrist = ET.SubElement(worldbody, "body", {"name": "wrist", "pos": "0 0 0"})
    if realistic:
        _add_palm_sculpt(wrist, params)
    if not realistic or keep_baseline:
        _baseline_palm(wrist)
    _site(wrist, "wrist", (0.0, 0.0, 0.0), rgba="0.15 0.45 0.70 1")
    return root, wrist


def _add_four_fingers(
    wrist: ET.Element,
    root: ET.Element,
    params: HumanHandParams,
    *,
    realistic: bool,
    keep_baseline: bool,
) -> None:
    """Append the four non-thumb finger chains plus their DIP/PIP coupling and actuators."""
    for finger in FOUR_FINGER_NAMES:
        _add_finger(wrist, finger, params, realistic=realistic, keep_baseline=keep_baseline)

    equality = ET.SubElement(root, "equality")
    for finger in FOUR_FINGER_NAMES:
        ET.SubElement(
            equality,
            "joint",
            {
                "name": f"{finger}_dip_coupling",
                "joint1": f"{finger}_dip",
                "joint2": f"{finger}_pip",
                "polycoef": _polycoef((0.0, DIP_PIP_COUPLING_RATIO, 0.0, 0.0, 0.0)),
            },
        )

    actuator = ET.SubElement(root, "actuator")
    for finger in FOUR_FINGER_NAMES:
        _position_actuator(actuator, f"{finger}_mcp_abd", kp=8.0)
        _position_actuator(actuator, f"{finger}_mcp_flex", kp=20.0)
        _position_actuator(actuator, f"{finger}_pip", kp=20.0)


def _add_thumb(
    wrist: ET.Element,
    params: HumanHandParams,
    *,
    realistic: bool,
    keep_baseline: bool,
) -> None:
    """Append the articulated thumb chain (CMC -> MCP -> IP) to ``wrist``.

    Joints and keypoint sites only -- no actuators and no equality coupling. The
    CMC body is tilted by ``ThumbParams.cmc_rpy`` (roll, pitch, yaw in radians,
    emitted as degrees under the default ``xyz`` euler sequence) so the thumb
    sits in opposition to the other fingers.
    """
    thumb = params.thumb
    limits = params.limits

    cmc_body = ET.SubElement(
        wrist,
        "body",
        {
            "name": "thumb_cmc",
            "pos": _vec(thumb.attach),
            "euler": _vec(np.degrees(value) for value in thumb.cmc_rpy),
        },
    )
    _joint(cmc_body, "thumb_cmc_abd", params.abd_axis, limits["thumb_cmc_abd"])
    _inertial(cmc_body, mass=DEFAULT_EMPTY_JOINT_BODY_MASS)

    cmc_flex_body = ET.SubElement(
        cmc_body, "body", {"name": "thumb_cmc_flex_body", "pos": "0 0 0"}
    )
    _joint(cmc_flex_body, "thumb_cmc_flex", params.flex_axis, limits["thumb_cmc_flex"])
    _site(cmc_flex_body, "thumb_cmc", (0.0, 0.0, 0.0))
    radii = _thumb_radii(thumb.prox) if realistic else None
    if realistic:
        _sphere(cmc_flex_body, "thumb_cmc_base", (0.0, 0.0, 0.0), radii["cmc_base"], JOINT_RGBA)
        _tapered_phalanx(cmc_flex_body, "thumb_meta", thumb.meta, radii["meta_thick"], radii["meta_thin"], SKIN_RGBA)
    if not realistic or keep_baseline:
        _capsule(cmc_flex_body, "thumb_meta_geom", (0.0, 0.0, 0.0), (0.0, thumb.meta, 0.0))

    mcp_body = ET.SubElement(
        cmc_flex_body,
        "body",
        {"name": "thumb_mcp_body", "pos": _vec((0.0, thumb.meta, 0.0))},
    )
    _joint(mcp_body, "thumb_mcp", params.flex_axis, limits["thumb_mcp"])
    _site(mcp_body, "thumb_mcp", (0.0, 0.0, 0.0))
    if realistic:
        _sphere(mcp_body, "thumb_mcp_knuckle", (0.0, 0.0, 0.0), radii["mcp_knuckle"], JOINT_RGBA)
        _r_capsule(mcp_body, "thumb_prox", (0.0, 0.0, 0.0), (0.0, thumb.prox, 0.0), radii["prox_shaft"], SKIN_RGBA)
    if not realistic or keep_baseline:
        _capsule(mcp_body, "thumb_prox_geom", (0.0, 0.0, 0.0), (0.0, thumb.prox, 0.0))

    ip_body = ET.SubElement(
        mcp_body,
        "body",
        {"name": "thumb_ip_body", "pos": _vec((0.0, thumb.prox, 0.0))},
    )
    _joint(ip_body, "thumb_ip", params.flex_axis, limits["thumb_ip"])
    _site(ip_body, "thumb_ip", (0.0, 0.0, 0.0))
    if realistic:
        _sphere(ip_body, "thumb_ip_knuckle", (0.0, 0.0, 0.0), radii["ip_knuckle"], JOINT_RGBA)
        _r_capsule(ip_body, "thumb_dist_shaft", (0.0, 0.0, 0.0), (0.0, 0.7 * thumb.dist, 0.0), radii["dist_shaft"], SKIN_RGBA)
        _sphere(ip_body, "thumb_tip_bulb", (0.0, thumb.dist, 0.0), radii["tip_bulb"], SKIN_RGBA)
        _sphere(ip_body, "thumb_pad", (0.0, thumb.dist - 0.004, -0.002), radii["pad"], PAD_RGBA)
        _ellipsoid(
            ip_body,
            "thumb_nail",
            (0.0, 0.4 * thumb.dist, radii["dist_shaft"]),
            (0.0055, 0.30 * thumb.dist, 0.0010),
            NAIL_RGBA,
        )
    if not realistic or keep_baseline:
        _capsule(ip_body, "thumb_dist_geom", (0.0, 0.0, 0.0), (0.0, thumb.dist, 0.0))
    _site(ip_body, "thumb_tip", (0.0, thumb.dist, 0.0))


def _add_default(root: ET.Element) -> None:
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "joint",
        {
            "type": "hinge",
            "limited": "true",
            "damping": "0.02",
            "armature": "0.0001",
        },
    )
    ET.SubElement(
        default,
        "geom",
        {
            "type": "capsule",
            "size": _fmt(DEFAULT_PHALANX_RADIUS),
            "rgba": BASELINE_GEOM_RGBA,
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        default,
        "site",
        {
            "type": "sphere",
            "size": _fmt(DEFAULT_SITE_SIZE),
            "rgba": "0.10 0.55 0.45 1",
        },
    )


def _add_finger(
    parent: ET.Element,
    finger: str,
    params: HumanHandParams,
    *,
    realistic: bool,
    keep_baseline: bool,
) -> None:
    finger_params = params.fingers[finger]
    limits = params.limits
    attach_body = ET.SubElement(
        parent,
        "body",
        {
            "name": f"{finger}_mcp",
            "pos": _vec(finger_params.attach),
            "euler": _vec((0.0, 0.0, np.degrees(finger_params.yaw))),
        },
    )
    _joint(
        attach_body,
        f"{finger}_mcp_abd",
        params.abd_axis,
        limits["mcp_abd"],
    )
    _inertial(attach_body, mass=DEFAULT_EMPTY_JOINT_BODY_MASS)

    radii = _finger_radii(finger_params.prox) if realistic else None
    emit_baseline = (not realistic) or keep_baseline

    flex_body = ET.SubElement(
        attach_body,
        "body",
        {"name": f"{finger}_mcp_flex_body", "pos": "0 0 0"},
    )
    _joint(
        flex_body,
        f"{finger}_mcp_flex",
        params.flex_axis,
        limits["mcp_flex"],
    )
    _site(flex_body, f"{finger}_mcp", (0.0, 0.0, 0.0))
    if realistic:
        _sphere(flex_body, f"{finger}_mcp_knuckle", (0.0, 0.0, 0.0), radii["mcp_knuckle"], JOINT_RGBA)
        _tapered_phalanx(flex_body, f"{finger}_prox", finger_params.prox, radii["prox_thick"], radii["prox_thin"], SKIN_RGBA)
    if emit_baseline:
        _capsule(flex_body, f"{finger}_prox_geom", (0.0, 0.0, 0.0), (0.0, finger_params.prox, 0.0))

    pip_body = ET.SubElement(
        flex_body,
        "body",
        {"name": f"{finger}_pip_body", "pos": _vec((0.0, finger_params.prox, 0.0))},
    )
    _joint(pip_body, f"{finger}_pip", params.flex_axis, limits["pip"])
    _site(pip_body, f"{finger}_pip", (0.0, 0.0, 0.0))
    if realistic:
        _sphere(pip_body, f"{finger}_pip_knuckle", (0.0, 0.0, 0.0), radii["pip_knuckle"], JOINT_RGBA)
        _tapered_phalanx(pip_body, f"{finger}_mid", finger_params.mid, radii["mid_thick"], radii["mid_thin"], SKIN_RGBA)
    if emit_baseline:
        _capsule(pip_body, f"{finger}_mid_geom", (0.0, 0.0, 0.0), (0.0, finger_params.mid, 0.0))

    dip_body = ET.SubElement(
        pip_body,
        "body",
        {"name": f"{finger}_dip_body", "pos": _vec((0.0, finger_params.mid, 0.0))},
    )
    _joint(dip_body, f"{finger}_dip", params.flex_axis, limits["dip"])
    _site(dip_body, f"{finger}_dip", (0.0, 0.0, 0.0))
    if realistic:
        _sphere(dip_body, f"{finger}_dip_knuckle", (0.0, 0.0, 0.0), radii["dip_knuckle"], JOINT_RGBA)
        _r_capsule(dip_body, f"{finger}_dist_shaft", (0.0, 0.0, 0.0), (0.0, 0.7 * finger_params.dist, 0.0), radii["dist_shaft"], SKIN_RGBA)
        _sphere(dip_body, f"{finger}_tip_bulb", (0.0, finger_params.dist, 0.0), radii["tip_bulb"], SKIN_RGBA)
        _sphere(dip_body, f"{finger}_pad", (0.0, finger_params.dist - 0.003, -0.0015), radii["pad"], PAD_RGBA)
        _ellipsoid(
            dip_body,
            f"{finger}_nail",
            (0.0, 0.4 * finger_params.dist, radii["dist_shaft"]),
            (0.30 * radii["tip_bulb"], 0.30 * finger_params.dist, 0.0009),
            NAIL_RGBA,
        )
    if emit_baseline:
        _capsule(dip_body, f"{finger}_dist_geom", (0.0, 0.0, 0.0), (0.0, finger_params.dist, 0.0))
    _site(dip_body, f"{finger}_tip", (0.0, finger_params.dist, 0.0))


def _add_palm_sculpt(wrist: ET.Element, params: HumanHandParams) -> None:
    """Emit the solid sculpted palm (group 2) on the wrist body.

    Positions are derived from params so the sculpt tracks attach/yaw changes
    and mirrors correctly under ``for_hand('left')``.
    """
    _ellipsoid(wrist, "palm_plate", (0.0, 0.055, -0.006), (0.050, 0.058, 0.016), PALM_RGBA)

    thumb_attach = params.thumb.attach
    _ellipsoid(
        wrist,
        "thenar_bulge",
        (thumb_attach[0], thumb_attach[1] + 0.007, -0.002),
        (0.018, 0.026, 0.013),
        THENAR_RGBA,
    )

    pinky_attach = params.fingers["pinky"].attach
    _ellipsoid(
        wrist,
        "hypothenar_bulge",
        (pinky_attach[0] + 0.005, pinky_attach[1] - 0.017, -0.002),
        (0.014, 0.030, 0.012),
        THENAR_RGBA,
    )

    _r_capsule(wrist, "palm_heel", (-0.028, 0.012, -0.003), (0.028, 0.012, -0.003), 0.013, PALM_RGBA)

    for finger in FOUR_FINGER_NAMES:
        fp = params.fingers[finger]
        knuckle_r = _finger_radii(fp.prox)["mcp_knuckle"] * 1.4
        _sphere(
            wrist,
            f"dorsal_{finger}_knuckle",
            (fp.attach[0], fp.attach[1], 0.014),
            knuckle_r,
            JOINT_RGBA,
        )


def _finger_radii(lp: float) -> dict[str, float]:
    """Per-finger geom radii (m), parametric on the proximal phalanx length ``lp``.

    Every entry is ``clamp(k*lp, lo, hi)`` so sizes stay sane for unusually
    short/long fingers and survive left-hand mirroring with no special-casing.
    """
    c = lambda k, lo, hi: _clamp(k * lp, lo, hi)
    return {
        "mcp_knuckle": c(0.21, 0.0075, 0.0110),
        "prox_thick": c(0.18, 0.0065, 0.0095),
        "prox_thin": c(0.16, 0.0058, 0.0085),
        "pip_knuckle": c(0.175, 0.0065, 0.0095),
        "mid_thick": c(0.155, 0.0055, 0.0085),
        "mid_thin": c(0.14, 0.0050, 0.0075),
        "dip_knuckle": c(0.15, 0.0055, 0.0080),
        "dist_shaft": c(0.13, 0.0048, 0.0070),
        "tip_bulb": c(0.145, 0.0052, 0.0075),
        "pad": c(0.125, 0.0045, 0.0065),
    }


def _thumb_radii(ltp: float) -> dict[str, float]:
    """Thumb geom radii (m), parametric on the thumb proximal phalanx ``ltp``.

    The thumb is the stockiest digit; scaling off its proximal phalanx keeps it
    visibly thicker than the other fingers.
    """
    c = lambda k, lo, hi: _clamp(k * ltp, lo, hi)
    return {
        "cmc_base": c(0.34, 0.0095, 0.0125),
        "meta_thick": c(0.30, 0.0085, 0.0110),
        "meta_thin": c(0.275, 0.0080, 0.0100),
        "mcp_knuckle": c(0.30, 0.0085, 0.0110),
        "prox_shaft": c(0.26, 0.0075, 0.0095),
        "ip_knuckle": c(0.26, 0.0075, 0.0095),
        "dist_shaft": c(0.225, 0.0065, 0.0085),
        "tip_bulb": c(0.26, 0.0075, 0.0095),
        "pad": c(0.225, 0.0065, 0.0085),
    }


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def _tapered_phalanx(
    parent: ET.Element,
    name_prefix: str,
    length: float,
    r_thick: float,
    r_thin: float,
    rgba: str,
) -> None:
    """Two stacked capsules along +Y (thick proximal half, thin distal half)."""
    mid = length * 0.5
    _r_capsule(parent, f"{name_prefix}_thick", (0.0, 0.0, 0.0), (0.0, mid + TAPER_OVERLAP, 0.0), r_thick, rgba)
    _r_capsule(parent, f"{name_prefix}_thin", (0.0, mid - TAPER_OVERLAP, 0.0), (0.0, length, 0.0), r_thin, rgba)


def _realistic_geom(
    parent: ET.Element,
    name: str,
    gtype: str,
    rgba: str,
    size_str: str,
    *,
    pos=None,
    fromto=None,
    group: int = GROUP_REALISTIC,
) -> None:
    """Emit a geom with EXPLICIT type/size/rgba/group/contype/conaffinity.

    Every attribute is set so the transparent capsule <default> (size 0.006 +
    alpha-0.55 rgba) cannot leak into the realistic sculpt.
    """
    attrs = {
        "name": name,
        "type": gtype,
        "rgba": rgba,
        "group": str(group),
        "contype": "0",
        "conaffinity": "0",
        "size": size_str,
    }
    if pos is not None:
        attrs["pos"] = pos if isinstance(pos, str) else _vec(pos)
    if fromto is not None:
        attrs["fromto"] = fromto
    ET.SubElement(parent, "geom", attrs)


def _sphere(
    parent: ET.Element, name: str, pos, radius: float, rgba: str, *, group: int = GROUP_REALISTIC
) -> None:
    _realistic_geom(parent, name, "sphere", rgba, _fmt(radius), pos=pos, group=group)


def _ellipsoid(
    parent: ET.Element, name: str, pos, semiaxes, rgba: str, *, group: int = GROUP_REALISTIC
) -> None:
    _realistic_geom(parent, name, "ellipsoid", rgba, _vec(semiaxes), pos=pos, group=group)


def _r_capsule(
    parent: ET.Element,
    name: str,
    start,
    end,
    radius: float,
    rgba: str,
    *,
    group: int = GROUP_REALISTIC,
) -> None:
    _realistic_geom(
        parent,
        name,
        "capsule",
        rgba,
        _fmt(radius),
        fromto=f"{_vec(start)} {_vec(end)}",
        group=group,
    )


def _baseline_palm(wrist: ET.Element) -> None:
    """Transparent baseline palm box (group-default); byte-identical to the original model."""
    ET.SubElement(
        wrist,
        "geom",
        {
            "name": "palm",
            "type": "box",
            "pos": "0 0.045 -0.004",
            "size": "0.055 0.055 0.006",
            "rgba": BASELINE_PALM_RGBA,
            "contype": "0",
            "conaffinity": "0",
        },
    )


def _joint(
    parent: ET.Element,
    name: str,
    axis: tuple[float, float, float],
    limit: tuple[float, float],
) -> None:
    ET.SubElement(
        parent,
        "joint",
        {
            "name": name,
            "axis": _vec(axis),
            "range": _limit_degrees(limit),
        },
    )


def _capsule(
    parent: ET.Element,
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> None:
    ET.SubElement(
        parent,
        "geom",
        {
            "name": name,
            "fromto": f"{_vec(start)} {_vec(end)}",
        },
    )


def _inertial(parent: ET.Element, *, mass: float) -> None:
    inertia = DEFAULT_EMPTY_JOINT_BODY_DIAGINERTIA
    ET.SubElement(
        parent,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": _fmt(mass),
            "diaginertia": _vec((inertia, inertia, inertia)),
        },
    )


def _site(
    parent: ET.Element,
    name: str,
    pos: tuple[float, float, float],
    *,
    rgba: str | None = None,
) -> None:
    attrs = {"name": name, "pos": _vec(pos)}
    if rgba is not None:
        attrs["rgba"] = rgba
    ET.SubElement(parent, "site", attrs)


def _position_actuator(parent: ET.Element, joint: str, *, kp: float) -> None:
    ET.SubElement(
        parent,
        "position",
        {
            "name": f"{joint}_act",
            "joint": joint,
            "kp": _fmt(kp),
        },
    )


def _limit_degrees(limit: tuple[float, float]) -> str:
    lo, hi = limit
    return f"{_fmt(np.degrees(lo))} {_fmt(np.degrees(hi))}"


def _polycoef(values: tuple[float, float, float, float, float]) -> str:
    return " ".join(_fmt(value) for value in values)


def _vec(values) -> str:
    return " ".join(_fmt(value) for value in values)


def _fmt(value: float) -> str:
    return f"{float(value):.6g}"


def _pretty_xml(root: ET.Element) -> str:
    rough = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ")
