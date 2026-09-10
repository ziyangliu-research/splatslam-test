import os

import numpy as np

from src.tartanair_slam import TartanAirV1SLAM
from src.utils.eth3d_eval import evaluate_ate_eth3d, evaluate_split_rendering_eth3d
from src.utils.tartanair_eval import extract_full_estimated_c2w, save_evaluation_summary


class ETH3DRectifiedSLAM(TartanAirV1SLAM):
    """Reuse the one-run ONLINE->FULL pipeline with ETH3D-specific GT handling."""

    def _write_failure_summary(self, stage, exitcode):
        total = len(self.stream)
        processed = int(self.processed_frames.value)
        ratio = (processed / total) if total else 0.0
        online_time = float(self.online_elapsed.value)
        summary = {
            "status": "failed",
            "mode": "FAILED",
            "failure_stage": stage,
            "exitcode": int(exitcode),
            "sequence": self.cfg["scene"],
            "processed_frames": processed,
            "total_frames": total,
            "maxmap_ratio": ratio,
            "maxmap_percent": ratio * 100.0,
            "online_time_sec": online_time if online_time > 0 else None,
            "offline_time_sec": None,
            "total_time_sec": online_time if online_time > 0 else None,
            "split_rule": getattr(self.stream, "split_rule_text", "every fifth frame -> test"),
        }
        save_evaluation_summary(self.save_dir, summary)

    def _evaluate_current_state(self, mode, include_fps, trajectory_filename):
        traj_est = extract_full_estimated_c2w(self)
        gt = np.asarray(self.stream.poses)
        processed = int(self.processed_frames.value)

        ate = evaluate_ate_eth3d(
            traj_est,
            gt,
            processed_frames=processed,
            gt_valid_mask=self.stream.gt_valid_mask,
        )

        np.savez(
            os.path.join(self.save_dir, trajectory_filename),
            estimated_c2w=traj_est,
            gt_c2w=gt,
            gt_valid_mask=np.asarray(self.stream.gt_valid_mask, dtype=bool),
            image_timestamps=np.asarray(self.stream.source_timestamps, dtype=np.float64),
        )

        if (
            (not self.only_tracking)
            and ate["maxmap_start"] is not None
            and ate["maxmap_frames"] > 0
        ):
            rendering_metrics = evaluate_split_rendering_eth3d(
                self.mapper,
                self.stream,
                traj_est,
                ate["maxmap_start"],
                ate["maxmap_end"],
                self.save_dir,
                tag=mode.lower(),
                save_test_renders=bool(self.stream.save_test_renders),
            )
            gaussian_count = int(self.mapper.gaussians.get_xyz.shape[0])
        else:
            rendering_metrics = {
                "train_psnr": float("nan"),
                "train_ssim": float("nan"),
                "train_lpips": float("nan"),
                "test_psnr": float("nan"),
                "test_ssim": float("nan"),
                "test_lpips": float("nan"),
                "train_frames_evaluated": 0,
                "test_frames_evaluated": 0,
                "test_render_dir": None,
            }
            gaussian_count = None

        online_time = float(self.online_elapsed.value)
        online_fps = (processed / online_time) if online_time > 0 else float("nan")
        valid_gt = int(np.asarray(self.stream.gt_valid_mask, dtype=bool).sum())

        return {
            "status": "ok" if not self.only_tracking else "ok_tracking_only",
            "mode": mode,
            "dataset": "ETH3D_rectified",
            "sequence": self.cfg["scene"],
            "processed_frames": processed,
            "total_frames": len(self.stream),
            "online_wall_sec": online_time,
            "online_time_sec": online_time,
            "fps": online_fps if include_fps else None,
            "online_fps_reference": online_fps,
            "gaussians": gaussian_count,
            "trajectory_alignment_primary": "SE3 (rigid; scale fixed)",
            "trajectory_alignment_reference": "Sim3 (scale fitted; monocular reference)",
            "gt_valid_frames": valid_gt,
            "gt_valid_fraction": valid_gt / len(self.stream) if len(self.stream) else 0.0,
            "max_gt_gap_sec": float(self.stream.max_gt_gap_sec),
            "baseline_m": float(self.stream.baseline_m),
            "split_rule": self.stream.split_rule_text,
            "test_frames_used_for_pose": True,
            "test_frames_used_for_droid_keyframe_selection": True,
            "test_frames_used_for_gaussian_mapping": False,
            "test_frames_used_for_map_optimization": False,
            "test_renders_saved": bool(self.stream.save_test_renders),
            **ate,
            **rendering_metrics,
        }
