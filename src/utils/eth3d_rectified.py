# ETH3D rectified stereo adapter for the Splat-SLAM benchmark.
# Splat-SLAM itself is RGB-only, so image_left (rectified ETH3D camera 2)
# is used as the monocular RGB stream. image_right is retained for validation.

import glob
import json
import os

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

import src.utils.datasets as datasets_module
from src.utils.datasets import BaseDataset


def _timestamp_from_path(path: str) -> float:
    return float(os.path.splitext(os.path.basename(path))[0])


def _pose_from_row(row: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
    pose[:3, 3] = row[1:4]
    return pose


def _interpolate_gt(image_timestamps, gt_rows, max_gt_gap_sec):
    """Return one compatibility pose per image plus a reliable-GT mask.

    ETH3D image and GT timestamps are not assumed to be one-to-one. Translation
    is linearly interpolated and rotation is interpolated with SLERP. A pose is
    marked valid for ATE only when the bracketing GT interval is no larger than
    max_gt_gap_sec. For larger GT outages we still provide an interpolated/nearest
    pose to satisfy the legacy dataset interface, but exclude it from ATE.
    """
    gt_rows = np.asarray(gt_rows, dtype=np.float64)
    gt_t = gt_rows[:, 0]
    if len(gt_t) < 2:
        raise ValueError("ETH3D groundtruth_left.txt must contain at least 2 poses")
    if np.any(np.diff(gt_t) <= 0):
        order = np.argsort(gt_t)
        gt_rows = gt_rows[order]
        gt_t = gt_rows[:, 0]

    gt_trans = gt_rows[:, 1:4]
    gt_rots = Rotation.from_quat(gt_rows[:, 4:8])
    poses = []
    valid = []
    gaps = []

    for t in image_timestamps:
        pos = int(np.searchsorted(gt_t, t, side="left"))

        # Exact (or effectively exact) GT timestamp.
        exact_idx = None
        if pos < len(gt_t) and abs(gt_t[pos] - t) <= 1e-9:
            exact_idx = pos
        elif pos > 0 and abs(gt_t[pos - 1] - t) <= 1e-9:
            exact_idx = pos - 1

        if exact_idx is not None:
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = gt_rots[exact_idx].as_matrix()
            pose[:3, 3] = gt_trans[exact_idx]
            poses.append(pose)
            valid.append(True)
            gaps.append(0.0)
            continue

        if pos == 0:
            pose = _pose_from_row(gt_rows[0])
            poses.append(pose)
            valid.append(False)
            gaps.append(float("inf"))
            continue
        if pos >= len(gt_t):
            pose = _pose_from_row(gt_rows[-1])
            poses.append(pose)
            valid.append(False)
            gaps.append(float("inf"))
            continue

        i0, i1 = pos - 1, pos
        t0, t1 = float(gt_t[i0]), float(gt_t[i1])
        gap = t1 - t0
        alpha = (float(t) - t0) / gap

        translation = (1.0 - alpha) * gt_trans[i0] + alpha * gt_trans[i1]
        slerp = Slerp([t0, t1], Rotation.from_quat([gt_rows[i0, 4:8], gt_rows[i1, 4:8]]))
        rotation = slerp([float(t)])[0].as_matrix()

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        poses.append(pose)
        valid.append(gap <= float(max_gt_gap_sec))
        gaps.append(gap)

    return poses, np.asarray(valid, dtype=bool), np.asarray(gaps, dtype=np.float64)


class ETH3DRectifiedStereo(BaseDataset):
    """Rectified ETH3D stereo sequence with camera-2 as the left reference.

    Expected layout:
      <root>/<scene>/image_left/*.png
      <root>/<scene>/image_right/*.png
      <root>/<scene>/calibration.json
      <root>/<scene>/groundtruth_left.txt
      <root>/<scene>/timestamps.txt

    Splat-SLAM consumes only image_left. The right stream and rectified baseline
    are retained in the dataset object for validation/metadata only.
    """

    def __init__(self, cfg, device="cuda:0"):
        super().__init__(cfg, device)

        self.sequence = cfg["scene"]
        self.input_folder = os.path.join(cfg["data"]["dataset_root"], self.sequence)
        split_cfg = cfg.get("evaluation", {})
        self.test_every = int(split_cfg.get("test_every", 5))
        self.test_offset = int(split_cfg.get("test_offset", 4))
        self.max_gt_gap_sec = float(split_cfg.get("max_gt_gap_sec", 0.1))
        self.save_test_renders = bool(split_cfg.get("save_test_renders", True))

        if self.test_every < 2:
            raise ValueError("evaluation.test_every must be >= 2")
        if not (0 <= self.test_offset < self.test_every):
            raise ValueError("evaluation.test_offset is outside the split period")

        left_dir = os.path.join(self.input_folder, "image_left")
        right_dir = os.path.join(self.input_folder, "image_right")
        calibration_path = os.path.join(self.input_folder, "calibration.json")
        gt_path = os.path.join(self.input_folder, "groundtruth_left.txt")

        all_left = sorted(glob.glob(os.path.join(left_dir, "*.png")), key=_timestamp_from_path)
        all_right = sorted(glob.glob(os.path.join(right_dir, "*.png")), key=_timestamp_from_path)
        if not all_left:
            raise FileNotFoundError(f"No ETH3D rectified left images: {left_dir}")
        if len(all_right) != len(all_left):
            raise ValueError(
                f"ETH3D stereo count mismatch for {self.sequence}: "
                f"left={len(all_left)}, right={len(all_right)}"
            )
        if not os.path.isfile(calibration_path):
            raise FileNotFoundError(calibration_path)
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(gt_path)

        with open(calibration_path, "r", encoding="utf-8") as f:
            calibration = json.load(f)
        self.calibration = calibration
        self.baseline_m = float(calibration["baseline_rectified_m"])

        first = cv2.imread(all_left[0], cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"Failed to read {all_left[0]}")
        h, w = first.shape[:2]
        if (h, w) != (self.H, self.W):
            raise ValueError(
                f"Configured ETH3D size={self.W}x{self.H}, image={w}x{h}. "
                "Run through run_eth3d_rectified.py so calibration.json is applied."
            )

        image_timestamps_all = np.asarray([_timestamp_from_path(p) for p in all_left], dtype=np.float64)
        gt_rows = np.loadtxt(gt_path, dtype=np.float64)
        if gt_rows.ndim == 1:
            gt_rows = gt_rows[None, :]
        if gt_rows.shape[1] != 8:
            raise ValueError(
                f"Expected timestamp tx ty tz qx qy qz qw in {gt_path}, got {gt_rows.shape}"
            )

        all_poses, all_gt_valid, all_gt_gap = _interpolate_gt(
            image_timestamps_all, gt_rows, self.max_gt_gap_sec
        )

        stride = int(cfg.get("stride", 1))
        max_frames = int(cfg.get("max_frames", -1))
        if stride < 1:
            raise ValueError(f"stride must be >=1, got {stride}")
        stop = None if max_frames < 0 else max_frames
        selected = list(range(len(all_left)))[:stop:stride]

        self.color_paths = [all_left[i] for i in selected]
        self.right_paths = [all_right[i] for i in selected]
        self.source_frame_indices = [int(i) for i in selected]
        self.source_timestamps = [float(image_timestamps_all[i]) for i in selected]
        # Keep source_frame_ids integer for compatibility with existing CSV code.
        self.source_frame_ids = list(self.source_frame_indices)
        selected_poses = [all_poses[i] for i in selected]
        self.gt_valid_mask = np.asarray([all_gt_valid[i] for i in selected], dtype=bool)
        self.gt_bracket_gap_sec = np.asarray([all_gt_gap[i] for i in selected], dtype=np.float64)

        self.n_img = len(self.color_paths)
        if self.n_img == 0:
            raise ValueError("No ETH3D frames selected")

        # Remove the arbitrary world origin while preserving metric scale.
        first_inv = np.linalg.inv(selected_poses[0])
        self.poses = [first_inv @ pose for pose in selected_poses]

        self.depth_paths = None
        self.has_sensor_depth = False
        self._dummy_depth = torch.zeros((self.H_out, self.W_out), dtype=torch.float32)
        self.w2c_first_pose = np.linalg.inv(self.poses[0])

        test_count = sum(self.is_test_frame(i) for i in range(self.n_img))
        train_count = self.n_img - test_count
        gt_valid_count = int(self.gt_valid_mask.sum())
        self.split_rule_text = (
            f"source ordinal % {self.test_every} == {self.test_offset} -> test"
        )

        print(
            f"INFO: ETH3D {self.sequence}: frames={self.n_img}, "
            f"train={train_count}, test={test_count}, baseline={self.baseline_m:.6f} m, "
            f"GT-valid={gt_valid_count}/{self.n_img} "
            f"(max bracket gap={self.max_gt_gap_sec:.3f}s), sensor_depth=False"
        )
        if gt_valid_count < self.n_img:
            print(
                f"INFO: ETH3D {self.sequence}: {self.n_img - gt_valid_count} image frames "
                "fall inside/around larger GT gaps; they remain in SLAM/rendering but are excluded from ATE."
            )

    def is_test_frame(self, index: int) -> bool:
        source_index = int(self.source_frame_indices[int(index)])
        return source_index % self.test_every == self.test_offset

    def split_name(self, index: int) -> str:
        return "test" if self.is_test_frame(index) else "train"

    def frame_label(self, index: int) -> str:
        return f"{self.source_frame_indices[int(index)]:06d}_{self.source_timestamps[int(index)]:.9f}"

    def __getitem__(self, index):
        color_data = self.get_color(index)
        pose = torch.from_numpy(self.poses[index]).float()
        return index, color_data, self._dummy_depth.clone(), pose


datasets_module.dataset_dict["eth3d_rectified"] = ETH3DRectifiedStereo
