# Copyright 2024 Google LLC
#
# TartanAir V1 Stereo Challenge adapter for Splat-SLAM.
# Splat-SLAM itself is RGB-only, so the left camera is used as the image stream.
# The right-camera paths are retained for validation/future stereo experiments.

import glob
import os

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import src.utils.datasets as datasets_module
from src.utils.datasets import BaseDataset


def _frame_index(path: str) -> int:
    """Extract the zero-padded frame id from e.g. 000123_left.png."""
    stem = os.path.basename(path).split("_")[0]
    return int(stem)


class TartanAirV1StereoChallenge(BaseDataset):
    """TartanAir V1 Visual SLAM Stereo Challenge sequence.

    Expected layout:

      stereo_root/SE000/image_left/000000_left.png
      stereo_root/SE000/image_right/000000_right.png
      gt_root/SE000.txt

    Ground-truth rows are:
      tx ty tz qx qy qz qw

    The GT poses are converted to 4x4 camera-to-world matrices and expressed
    relative to the first selected frame. Splat-SLAM uses the left RGB stream;
    GT is used only for trajectory evaluation.
    """

    def __init__(self, cfg, device="cuda:0"):
        # BaseDataset expects input_folder to exist in cfg. The dedicated runner
        # sets it to the selected sequence before constructing the dataset.
        super().__init__(cfg, device)

        self.sequence = cfg["scene"]
        self.input_folder = os.path.join(cfg["data"]["dataset_root"], self.sequence)
        self.gt_root = cfg["data"]["gt_root"]

        split_cfg = cfg.get("evaluation", {})
        self.test_every = int(split_cfg.get("test_every", 5))
        self.test_offset = int(split_cfg.get("test_offset", 4))
        if self.test_every < 2:
            raise ValueError(f"evaluation.test_every must be >= 2, got {self.test_every}")
        if not (0 <= self.test_offset < self.test_every):
            raise ValueError(
                f"evaluation.test_offset must be in [0, {self.test_every - 1}], "
                f"got {self.test_offset}"
            )

        left_dir = os.path.join(self.input_folder, "image_left")
        right_dir = os.path.join(self.input_folder, "image_right")
        gt_path = os.path.join(self.gt_root, f"{self.sequence}.txt")

        all_left_paths = sorted(
            glob.glob(os.path.join(left_dir, "*_left.png")), key=_frame_index
        )
        all_right_paths = sorted(
            glob.glob(os.path.join(right_dir, "*_right.png")), key=_frame_index
        )

        if not all_left_paths:
            raise FileNotFoundError(
                f"No left images found for {self.sequence}: {left_dir}"
            )
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"Ground-truth pose file not found: {gt_path}")

        # Validate the configured TartanAir V1 camera model against the images.
        first = cv2.imread(all_left_paths[0], cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"Failed to read image: {all_left_paths[0]}")
        h, w = first.shape[:2]
        if (h, w) != (self.H, self.W):
            raise ValueError(
                f"Configured camera size is {self.W}x{self.H}, but "
                f"{all_left_paths[0]} is {w}x{h}."
            )

        raw_poses = np.loadtxt(gt_path, dtype=np.float64)
        if raw_poses.ndim == 1:
            raw_poses = raw_poses[None, :]
        if raw_poses.shape[1] != 7:
            raise ValueError(
                f"Expected GT rows 'tx ty tz qx qy qz qw', got shape "
                f"{raw_poses.shape} from {gt_path}."
            )
        if len(raw_poses) < len(all_left_paths):
            raise ValueError(
                f"GT has {len(raw_poses)} poses but left stream has "
                f"{len(all_left_paths)} images for {self.sequence}."
            )

        all_poses = []
        for row in raw_poses[: len(all_left_paths)]:
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = Rotation.from_quat(row[3:7]).as_matrix()
            pose[:3, 3] = row[:3]
            all_poses.append(pose)

        stride = int(cfg.get("stride", 1))
        max_frames = int(cfg.get("max_frames", -1))
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        stop = None if max_frames < 0 else max_frames

        selected_indices = list(range(len(all_left_paths)))[:stop:stride]
        self.color_paths = [all_left_paths[i] for i in selected_indices]
        self.right_paths = [all_right_paths[i] for i in selected_indices if i < len(all_right_paths)]
        selected_poses = [all_poses[i] for i in selected_indices]
        self.source_frame_ids = [_frame_index(all_left_paths[i]) for i in selected_indices]

        self.n_img = len(self.color_paths)
        if self.n_img == 0:
            raise ValueError(
                f"No frames selected for {self.sequence}; check max_frames/stride."
            )

        # Express GT relative to the first selected frame. This removes the
        # arbitrary global coordinate origin but preserves metric scale. The
        # evaluator later applies rigid SE(3), never Sim(3), alignment.
        first_inv = np.linalg.inv(selected_poses[0])
        self.poses = [first_inv @ pose for pose in selected_poses]

        # TartanAir Stereo Challenge does not provide sensor depth in this path.
        # Mapper has a legacy RGB-D-shaped interface and calls .numpy()/.to() on
        # frame_reader[idx][2], although that value is not used to initialize the
        # RGB-only tracker. Return zeros as an interface placeholder and disable
        # sensor-depth metrics in TartanAirV1SLAM.
        self.depth_paths = None
        self.has_sensor_depth = False
        self._dummy_depth = torch.zeros(
            (self.H_out, self.W_out), dtype=torch.float32
        )
        self.w2c_first_pose = np.linalg.inv(self.poses[0])

        if all_right_paths and len(self.right_paths) != self.n_img:
            print(
                f"WARNING: {self.sequence}: {self.n_img} selected left frames but "
                f"{len(self.right_paths)} selected right frames. Right images are not "
                "used by RGB-only Splat-SLAM."
            )

        test_count = sum(self.is_test_frame(i) for i in range(self.n_img))
        train_count = self.n_img - test_count
        print(
            f"INFO: TartanAir V1 {self.sequence}: {self.n_img} left RGB frames, "
            f"train={train_count}, test={test_count}, "
            f"split=source_frame_id % {self.test_every} == {self.test_offset}, "
            f"GT={gt_path}, sensor_depth=False"
        )

    def is_test_frame(self, index: int) -> bool:
        """True when the original source frame belongs to the held-out 20%."""
        source_id = int(self.source_frame_ids[int(index)])
        return source_id % self.test_every == self.test_offset

    def split_name(self, index: int) -> str:
        return "test" if self.is_test_frame(index) else "train"

    def __getitem__(self, index):
        # BaseDataset.get_color performs resize/crop and BGR->RGB conversion.
        color_data = self.get_color(index)
        pose = torch.from_numpy(self.poses[index]).float()
        return index, color_data, self._dummy_depth.clone(), pose


# Mapper imports get_dataset() from src.utils.datasets. Register this adapter in
# that module's dataset_dict so mapper-side frame_reader construction works in
# both the parent and multiprocessing-spawned child processes.
datasets_module.dataset_dict["tartanair_v1_challenge"] = TartanAirV1StereoChallenge
