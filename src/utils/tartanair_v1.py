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
    relative to the first frame. GT is used for trajectory evaluation only;
    it is never supplied to the tracker or mapper.
    """

    def __init__(self, cfg, device="cuda:0"):
        super().__init__(cfg, device)

        self.sequence = cfg["scene"]
        self.input_folder = os.path.join(cfg["data"]["dataset_root"], self.sequence)
        self.gt_root = cfg["data"]["gt_root"]

        left_dir = os.path.join(self.input_folder, "image_left")
        right_dir = os.path.join(self.input_folder, "image_right")
        gt_path = os.path.join(self.gt_root, f"{self.sequence}.txt")

        self.color_paths = sorted(
            glob.glob(os.path.join(left_dir, "*_left.png")), key=_frame_index
        )
        self.right_paths = sorted(
            glob.glob(os.path.join(right_dir, "*_right.png")), key=_frame_index
        )

        if not self.color_paths:
            raise FileNotFoundError(
                f"No left images found for {self.sequence}: {left_dir}"
            )
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"Ground-truth pose file not found: {gt_path}")

        # Validate the configured TartanAir V1 camera model against the images.
        first = cv2.imread(self.color_paths[0], cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"Failed to read image: {self.color_paths[0]}")
        h, w = first.shape[:2]
        if (h, w) != (self.H, self.W):
            raise ValueError(
                f"Configured camera size is {self.W}x{self.H}, but "
                f"{self.color_paths[0]} is {w}x{h}."
            )

        raw_poses = np.loadtxt(gt_path, dtype=np.float64)
        if raw_poses.ndim == 1:
            raw_poses = raw_poses[None, :]
        if raw_poses.shape[1] != 7:
            raise ValueError(
                f"Expected GT rows 'tx ty tz qx qy qz qw', got shape "
                f"{raw_poses.shape} from {gt_path}."
            )
        if len(raw_poses) < len(self.color_paths):
            raise ValueError(
                f"GT has {len(raw_poses)} poses but left stream has "
                f"{len(self.color_paths)} images for {self.sequence}."
            )

        poses = []
        for row in raw_poses[: len(self.color_paths)]:
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = Rotation.from_quat(row[3:7]).as_matrix()
            pose[:3, 3] = row[:3]
            poses.append(pose)

        # Put GT in a first-frame-relative coordinate system, matching the other
        # loaders in Splat-SLAM. The evaluator later performs Sim(3) alignment.
        first_inv = np.linalg.inv(poses[0])
        poses = [first_inv @ pose for pose in poses]

        stride = int(cfg.get("stride", 1))
        max_frames = int(cfg.get("max_frames", -1))
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        stop = None if max_frames < 0 else max_frames

        self.color_paths = self.color_paths[:stop:stride]
        self.right_paths = self.right_paths[:stop:stride]
        self.poses = poses[:stop:stride]
        self.n_img = len(self.color_paths)
        self.depth_paths = None
        self.has_sensor_depth = False
        self.w2c_first_pose = np.linalg.inv(self.poses[0])

        if self.right_paths and len(self.right_paths) != self.n_img:
            print(
                f"WARNING: {self.sequence}: {self.n_img} selected left frames but "
                f"{len(self.right_paths)} selected right frames. Right images are not "
                "used by RGB-only Splat-SLAM."
            )

        print(
            f"INFO: TartanAir V1 {self.sequence}: {self.n_img} left RGB frames, "
            f"GT={gt_path}, sensor_depth=False"
        )

    def __getitem__(self, index):
        # BaseDataset.get_color performs resize/crop and BGR->RGB conversion.
        color_data = self.get_color(index)
        pose = torch.from_numpy(self.poses[index]).float()
        return index, color_data, None, pose
