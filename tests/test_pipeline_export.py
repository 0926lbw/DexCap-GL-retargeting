import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hand_reconstruction.export import skeleton_to_dict, write_skeleton_json, write_skeleton_npy
from hand_reconstruction.glove_observation import GloveLinkObservation, finger_link_names
from hand_reconstruction.pipeline import reconstruct_from_link_positions
from hand_reconstruction.schema import LANDMARK_NAMES, NUM_LANDMARKS


class PipelineExportTest(unittest.TestCase):
    def test_reconstruct_from_link_positions_returns_named_21_point_skeleton(self):
        names = finger_link_names("left")
        positions = {"base_link": np.array([1.0, 2.0, 3.0])}
        for finger_idx, finger in enumerate(names):
            for link_idx, link_name in enumerate(names[finger]):
                positions[link_name] = np.array(
                    [1.0 + finger_idx, 2.0 + link_idx, 3.0]
                )

        skeleton = reconstruct_from_link_positions("left", positions)
        payload = skeleton_to_dict(skeleton)

        self.assertEqual(payload["hand"], "left")
        self.assertEqual(len(payload["landmarks"]), NUM_LANDMARKS)
        self.assertEqual(payload["landmarks"][0]["name"], LANDMARK_NAMES[0])
        self.assertEqual(payload["landmarks"][0]["xyz"], [1.0, 2.0, 3.0])
        self.assertIn("coordinate_convention", payload)
        self.assertIn("T_W_Hwrist", payload)
        self.assertIn("keypoints_21_in_Hwrist", payload)
        self.assertIn("keypoints_21_in_W", payload)
        self.assertEqual(len(payload["keypoints_21_in_Hwrist"]), NUM_LANDMARKS)
        self.assertEqual(len(payload["keypoints_21_in_W"]), NUM_LANDMARKS)
        self.assertEqual(payload["keypoints_21_in_Hwrist"][0]["xyz"], [0.0, 0.0, 0.0])
        self.assertEqual(payload["keypoints_21_in_W"][0]["xyz"], [1.0, 2.0, 3.0])

    def test_exports_json_and_npy(self):
        observation = GloveLinkObservation(
            hand="right",
            base_position=np.zeros(3),
            link_positions={
                link_name: np.ones(3) * idx
                for idx, link_name in enumerate(_all_right_link_names(), start=1)
            },
        )
        skeleton = reconstruct_from_link_positions(
            observation.hand,
            {"base_link": observation.base_position, **observation.link_positions},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            json_path = tmp_path / "skeleton.json"
            npy_path = tmp_path / "skeleton.npy"

            write_skeleton_json(skeleton, json_path)
            write_skeleton_npy(skeleton, npy_path)

            with json_path.open("r", encoding="utf-8") as fp:
                loaded_json = json.load(fp)
            loaded_npy = np.load(npy_path)

        self.assertEqual(loaded_json["hand"], "right")
        self.assertEqual(loaded_npy.shape, (NUM_LANDMARKS, 3))


def _all_right_link_names():
    names = finger_link_names("right")
    for finger in names:
        yield from names[finger]


if __name__ == "__main__":
    unittest.main()
