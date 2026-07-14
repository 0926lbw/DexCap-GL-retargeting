import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


class MujocoFourFingerExportTest(unittest.TestCase):
    def test_build_four_finger_mjcf_has_expected_structure_without_thumb(self):
        from hand_reconstruction.human_hand_params import default_params
        from hand_reconstruction.mujoco_export import (
            FOUR_FINGER_NAMES,
            build_four_finger_mjcf,
        )

        xml_text = build_four_finger_mjcf(default_params(), realistic=False)
        root = ET.fromstring(xml_text)

        self.assertEqual(root.tag, "mujoco")
        self.assertEqual(root.attrib["model"], "human_four_finger_hand")
        visual_global = root.find("./visual/global")
        self.assertIsNotNone(visual_global)
        self.assertEqual(visual_global.attrib["offwidth"], "1280")
        self.assertEqual(visual_global.attrib["offheight"], "960")
        self.assertNotIn("thumb", xml_text)

        names = _named_elements(root)
        for finger in FOUR_FINGER_NAMES:
            for suffix in ("mcp", "pip", "dip", "tip"):
                self.assertIn(f"{finger}_{suffix}", names["site"])
            for suffix in ("mcp_abd", "mcp_flex", "pip", "dip"):
                self.assertIn(f"{finger}_{suffix}", names["joint"])
            for suffix in ("mcp_abd_act", "mcp_flex_act", "pip_act"):
                self.assertIn(f"{finger}_{suffix}", names["position"])
            self.assertNotIn(f"{finger}_dip_act", names["position"])
            self.assertIn(f"{finger}_dip_coupling", names["equality_joint"])

    def test_joint_axes_limits_and_dip_coupling_match_four_finger_model(self):
        from hand_reconstruction.human_hand_params import default_params
        from hand_reconstruction.mujoco_export import (
            FOUR_FINGER_NAMES,
            build_four_finger_mjcf,
        )

        params = default_params()
        root = ET.fromstring(build_four_finger_mjcf(params, realistic=False))
        joints = {
            elem.attrib["name"]: elem
            for elem in root.findall(".//joint")
            if "name" in elem.attrib
        }
        equality_joints = {
            elem.attrib["name"]: elem for elem in root.findall("./equality/joint")
        }

        for finger in FOUR_FINGER_NAMES:
            self.assertEqual(
                joints[f"{finger}_mcp_abd"].attrib["axis"],
                _vec(params.abd_axis),
            )
            for suffix in ("mcp_flex", "pip", "dip"):
                self.assertEqual(
                    joints[f"{finger}_{suffix}"].attrib["axis"],
                    _vec(params.flex_axis),
                )

            self.assertEqual(
                joints[f"{finger}_mcp_abd"].attrib["range"],
                _limit_degrees(params.limits["mcp_abd"]),
            )
            self.assertEqual(
                joints[f"{finger}_mcp_flex"].attrib["range"],
                _limit_degrees(params.limits["mcp_flex"]),
            )
            self.assertEqual(
                joints[f"{finger}_pip"].attrib["range"],
                _limit_degrees(params.limits["pip"]),
            )
            self.assertEqual(
                joints[f"{finger}_dip"].attrib["range"],
                _limit_degrees(params.limits["dip"]),
            )

            coupling = equality_joints[f"{finger}_dip_coupling"]
            self.assertEqual(coupling.attrib["joint1"], f"{finger}_dip")
            self.assertEqual(coupling.attrib["joint2"], f"{finger}_pip")
            self.assertEqual(coupling.attrib["polycoef"], "0 0.6 0 0 0")

    def test_empty_joint_holder_bodies_have_inertial_and_no_empty_tip_bodies(self):
        from hand_reconstruction.human_hand_params import default_params
        from hand_reconstruction.mujoco_export import (
            FOUR_FINGER_NAMES,
            build_four_finger_mjcf,
        )

        root = ET.fromstring(build_four_finger_mjcf(default_params(), realistic=False))

        for finger in FOUR_FINGER_NAMES:
            mcp_body = root.find(f".//body[@name='{finger}_mcp']")
            self.assertIsNotNone(mcp_body)
            self.assertIsNotNone(mcp_body.find("joint"))
            self.assertIsNone(mcp_body.find("geom"))

            inertial = mcp_body.find("inertial")
            self.assertIsNotNone(inertial)
            self.assertGreater(float(inertial.attrib["mass"]), 0.0)
            for value in inertial.attrib["diaginertia"].split():
                self.assertGreater(float(value), 0.0)

            self.assertIsNone(root.find(f".//body[@name='{finger}_tip_body']"))

    def test_export_script_writes_parseable_xml_to_requested_path(self):
        from scripts.export_mujoco_four_finger_hand import export_four_finger_mjcf

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "hand.xml"

            result = export_four_finger_mjcf(output_path)

            self.assertEqual(result, output_path)
            root = ET.parse(output_path).getroot()
            self.assertEqual(root.attrib["model"], "human_four_finger_hand")


class MujocoFullHandExportTest(unittest.TestCase):
    def test_build_full_hand_mjcf_exposes_all_21_landmark_sites(self):
        from hand_reconstruction.mujoco_export import build_full_hand_mjcf
        from hand_reconstruction.schema import LANDMARK_NAMES

        root = ET.fromstring(build_full_hand_mjcf())

        self.assertEqual(root.attrib["model"], "human_full_hand")
        named_sites = {
            elem.attrib["name"]
            for elem in root.findall(".//site")
            if "name" in elem.attrib
        }
        self.assertEqual(named_sites, set(LANDMARK_NAMES))

    def test_full_hand_has_articulated_thumb_without_actuators_or_coupling(self):
        from hand_reconstruction.mujoco_export import (
            FOUR_FINGER_NAMES,
            build_full_hand_mjcf,
        )

        root = ET.fromstring(build_full_hand_mjcf())
        names = _named_elements(root)

        for suffix in ("cmc", "mcp", "ip", "tip"):
            self.assertIn(f"thumb_{suffix}", names["site"])
        for joint in ("thumb_cmc_abd", "thumb_cmc_flex", "thumb_mcp", "thumb_ip"):
            self.assertIn(joint, names["joint"])

        # Thumb is joints-only: no actuators and no equality coupling.
        self.assertFalse(any("thumb" in n for n in names["position"]))
        self.assertFalse(any("thumb" in n for n in names["equality_joint"]))

        # The four non-thumb fingers are unchanged: still actuated + coupled.
        for finger in FOUR_FINGER_NAMES:
            self.assertIn(f"{finger}_mcp_flex_act", names["position"])
            self.assertIn(f"{finger}_pip_act", names["position"])
            self.assertIn(f"{finger}_dip_coupling", names["equality_joint"])

    def test_full_hand_thumb_joints_match_param_limits_and_flex_axis(self):
        from hand_reconstruction.human_hand_params import default_params
        from hand_reconstruction.mujoco_export import build_full_hand_mjcf

        params = default_params()
        root = ET.fromstring(build_full_hand_mjcf(params))
        joints = {
            elem.attrib["name"]: elem
            for elem in root.findall(".//joint")
            if "name" in elem.attrib
        }

        self.assertEqual(joints["thumb_cmc_abd"].attrib["axis"], _vec(params.abd_axis))
        for suffix in ("cmc_flex", "mcp", "ip"):
            self.assertEqual(
                joints[f"thumb_{suffix}"].attrib["axis"], _vec(params.flex_axis)
            )
        self.assertEqual(
            joints["thumb_cmc_abd"].attrib["range"],
            _limit_degrees(params.limits["thumb_cmc_abd"]),
        )
        self.assertEqual(
            joints["thumb_cmc_flex"].attrib["range"],
            _limit_degrees(params.limits["thumb_cmc_flex"]),
        )
        self.assertEqual(
            joints["thumb_mcp"].attrib["range"],
            _limit_degrees(params.limits["thumb_mcp"]),
        )
        self.assertEqual(
            joints["thumb_ip"].attrib["range"],
            _limit_degrees(params.limits["thumb_ip"]),
        )

    def test_full_hand_thumb_cmc_body_is_opposed_with_inertial(self):
        import numpy as np

        from hand_reconstruction.human_hand_params import default_params
        from hand_reconstruction.mujoco_export import build_full_hand_mjcf

        params = default_params()
        root = ET.fromstring(build_full_hand_mjcf(params))
        cmc_body = root.find(".//body[@name='thumb_cmc']")
        self.assertIsNotNone(cmc_body)

        expected_euler = " ".join(
            f"{float(np.degrees(v)):.6g}" for v in params.thumb.cmc_rpy
        )
        self.assertEqual(cmc_body.attrib["euler"], expected_euler)
        self.assertEqual(cmc_body.attrib["pos"], _vec(params.thumb.attach))

        inertial = cmc_body.find("inertial")
        self.assertIsNotNone(inertial)
        self.assertGreater(float(inertial.attrib["mass"]), 0.0)

    def test_export_full_hand_script_writes_parseable_xml_to_requested_path(self):
        from scripts.export_mujoco_full_hand import export_full_hand_mjcf

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "full_hand.xml"

            result = export_full_hand_mjcf(output_path)

            self.assertEqual(result, output_path)
            root = ET.parse(output_path).getroot()
            self.assertEqual(root.attrib["model"], "human_full_hand")


class MujocoRealisticSculptTest(unittest.TestCase):
    def test_realistic_full_hand_emits_solid_group2_sculpt_without_baseline(self):
        from hand_reconstruction.mujoco_export import FOUR_FINGER_NAMES, build_full_hand_mjcf

        root = ET.fromstring(build_full_hand_mjcf(realistic=True))
        geoms = _geoms(root)

        for sculpt_name in ("palm_plate", "thenar_bulge", "hypothenar_bulge", "palm_heel"):
            self.assertIn(sculpt_name, geoms)
        for finger in FOUR_FINGER_NAMES:
            for n in (
                f"{finger}_mcp_knuckle", f"{finger}_prox_thick", f"{finger}_prox_thin",
                f"{finger}_pip_knuckle", f"{finger}_mid_thick", f"{finger}_mid_thin",
                f"{finger}_dip_knuckle", f"{finger}_dist_shaft", f"{finger}_tip_bulb",
                f"{finger}_pad", f"{finger}_nail", f"dorsal_{finger}_knuckle",
            ):
                self.assertIn(n, geoms)
        for n in (
            "thumb_cmc_base", "thumb_meta_thick", "thumb_meta_thin", "thumb_mcp_knuckle",
            "thumb_prox", "thumb_ip_knuckle", "thumb_dist_shaft", "thumb_tip_bulb",
            "thumb_pad", "thumb_nail",
        ):
            self.assertIn(n, geoms)

        # Every group-2 geom is solid (alpha 1) and collision-free.
        for el in geoms.values():
            if el.attrib.get("group") == "2":
                self.assertEqual(el.attrib["contype"], "0")
                self.assertEqual(el.attrib["conaffinity"], "0")
                self.assertTrue(el.attrib["rgba"].endswith(" 1"))

        # Default keep_baseline=False -> transparent baseline sticks are gone.
        for finger in FOUR_FINGER_NAMES:
            self.assertNotIn(f"{finger}_prox_geom", geoms)
        self.assertNotIn("palm", geoms)

    def test_baseline_full_hand_has_only_transparent_sticks(self):
        from hand_reconstruction.mujoco_export import build_full_hand_mjcf

        root = ET.fromstring(build_full_hand_mjcf(realistic=False))
        geoms = _geoms(root)
        self.assertIn("palm", geoms)
        self.assertIn("index_prox_geom", geoms)
        self.assertNotIn("palm_plate", geoms)
        self.assertNotIn("index_mcp_knuckle", geoms)
        self.assertNotIn("index_nail", geoms)

    def test_keep_baseline_emits_both_layers(self):
        from hand_reconstruction.mujoco_export import build_full_hand_mjcf

        root = ET.fromstring(build_full_hand_mjcf(realistic=True, keep_baseline=True))
        geoms = _geoms(root)
        self.assertIn("palm_plate", geoms)        # group-2 sculpt
        self.assertIn("palm", geoms)              # baseline palm box
        self.assertIn("index_prox_geom", geoms)   # baseline capsule
        self.assertIn("index_mcp_knuckle", geoms)  # sculpt knuckle

    def test_realistic_does_not_change_kinematics(self):
        from hand_reconstruction.mujoco_export import build_full_hand_mjcf

        base = _named_elements(ET.fromstring(build_full_hand_mjcf(realistic=False)))
        real = _named_elements(ET.fromstring(build_full_hand_mjcf(realistic=True)))
        for key in ("site", "joint", "position", "equality_joint"):
            self.assertEqual(base[key], real[key], key)

    def test_four_finger_realistic_has_palm_sculpt(self):
        from hand_reconstruction.mujoco_export import build_four_finger_mjcf

        root = ET.fromstring(build_four_finger_mjcf(realistic=True))
        geoms = _geoms(root)
        self.assertIn("palm_plate", geoms)
        self.assertIn("index_mcp_knuckle", geoms)
        self.assertNotIn("thumb_nail", geoms)  # four-finger model has no thumb

    def test_finger_radii_are_parametric_and_clamped(self):
        from hand_reconstruction.mujoco_export import _finger_radii

        radii = _finger_radii(0.0432)
        self.assertAlmostEqual(radii["prox_thick"], min(0.0095, max(0.0065, 0.18 * 0.0432)))
        self.assertAlmostEqual(_finger_radii(0.01)["prox_thick"], 0.0065)   # clamped to lo
        self.assertAlmostEqual(_finger_radii(0.20)["mcp_knuckle"], 0.0110)  # clamped to hi


def _geoms(root):
    return {
        elem.attrib["name"]: elem
        for elem in root.findall(".//geom")
        if "name" in elem.attrib
    }


def _named_elements(root):
    result = {
        "site": set(),
        "joint": set(),
        "position": set(),
        "equality_joint": set(),
    }
    for elem in root.findall(".//site"):
        if "name" in elem.attrib:
            result["site"].add(elem.attrib["name"])
    for elem in root.findall(".//joint"):
        if "name" in elem.attrib:
            result["joint"].add(elem.attrib["name"])
    for elem in root.findall("./actuator/position"):
        result["position"].add(elem.attrib["name"])
    for elem in root.findall("./equality/joint"):
        result["equality_joint"].add(elem.attrib["name"])
    return result


def _vec(values):
    return " ".join(f"{float(value):.6g}" for value in values)


def _limit_degrees(limit):
    import numpy as np

    lo, hi = limit
    return f"{np.degrees(lo):.6g} {np.degrees(hi):.6g}"


if __name__ == "__main__":
    unittest.main()
