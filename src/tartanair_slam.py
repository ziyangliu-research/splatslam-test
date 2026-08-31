import json
import os

import numpy as np
import torch
import torch.multiprocessing as mp

from src.slam import SLAM
from src.tartanair_tracker import TartanAirSplitTracker
from src.utils.Printer import FontColor
from src.utils.tartanair_eval import (
    evaluate_ate_se3,
    evaluate_split_rendering,
    extract_full_estimated_c2w,
    save_evaluation_summary,
)
from src.utils.tartanair_sim3 import evaluate_ate_sim3


def _no_sensor_depth_eval(*args, **kwargs):
    """Return NaN depth metrics when the dataset has no sensor depth."""
    return np.nan, np.nan, np.nan


class TartanAirV1SLAM(SLAM):
    """Splat-SLAM wrapper for the TartanAir V1 challenge benchmark.

    The challenge package contains stereo RGB and GT poses, but Splat-SLAM is a
    monocular RGB-only method. The left stream is used for SLAM. Every frame is
    sent through DROID pose estimation; held-out test frames are blocked only at
    the Gaussian mapper boundary.
    """

    def __init__(self, cfg, stream):
        super().__init__(cfg, stream)

        self.processed_frames = mp.Value("i", 0)
        self.online_elapsed = mp.Value("d", 0.0)
        self.video.eval_depth_l1 = _no_sensor_depth_eval

    def tracking(self, pipe):
        self.tracker = TartanAirSplitTracker(self, pipe)
        self.printer.print("Tracking Triggered!", FontColor.TRACKER)
        self.all_trigered += 1

        os.makedirs(f"{self.save_dir}/mono_priors/depths", exist_ok=True)

        while self.all_trigered < self.num_running_thread:
            pass
        self.printer.pbar_ready()

        self.tracker.run(self.stream)
        self.printer.print("Tracking Done!", FontColor.TRACKER)

        if self.only_tracking:
            self.terminate()

    def _write_failure_summary(self, stage, exitcode):
        total = len(self.stream)
        processed = int(self.processed_frames.value)
        ratio = (processed / total) if total else 0.0
        summary = {
            "status": "failed",
            "failure_stage": stage,
            "exitcode": int(exitcode),
            "sequence": self.cfg["scene"],
            "processed_frames": processed,
            "total_frames": total,
            "maxmap_ratio": ratio,
            "maxmap_percent": ratio * 100.0,
            "split_rule": "source_frame_id % 5 == 4 -> test",
        }
        save_evaluation_summary(self.save_dir, summary)

    def terminate(self):
        """TartanAir-specific finalization and benchmark evaluation.

        SE(3) and Sim(3) ATE are both reported on the exact same MaxMap interval.
        SE(3) is the metric-scale-fixed benchmark value used in our comparison
        table; Sim(3) matches the original monocular Splat-SLAM protocol.
        """
        if self.only_tracking:
            self.video.save_video(f"{self.save_dir}/video.npz")
            traj_est = extract_full_estimated_c2w(self)
            gt = np.asarray(self.stream.poses)
            ate = evaluate_ate_se3(
                traj_est, gt, processed_frames=int(self.processed_frames.value)
            )
            ate_sim3 = evaluate_ate_sim3(
                traj_est, gt, ate["maxmap_start"], ate["maxmap_end"]
            )
            elapsed = float(self.online_elapsed.value)
            processed = int(self.processed_frames.value)
            summary = {
                "status": "ok_tracking_only",
                "sequence": self.cfg["scene"],
                "processed_frames": processed,
                "total_frames": len(self.stream),
                "online_wall_sec": elapsed,
                "fps": (processed / elapsed) if elapsed > 0 else float("nan"),
                "split_rule": (
                    f"source_frame_id % {self.stream.test_every} == "
                    f"{self.stream.test_offset} -> test"
                ),
                **ate,
                **ate_sim3,
            }
            save_evaluation_summary(self.save_dir, summary)
            self.printer.print(
                "Tracking-only trajectory evaluation:\n"
                f"  MaxMap: {summary['maxmap_percent']:.2f}%\n"
                f"  ATE RMSE SE(3):  {summary['ate_rmse_se3_m']:.6f} m\n"
                f"  ATE RMSE Sim(3): {summary['ate_rmse_sim3_m']:.6f} m",
                FontColor.EVAL,
            )
            return

        # Final global BA is pose/depth-only and may use train and test frames.
        if self.cfg["tracking"]["backend"]["final_ba"]:
            self.backend()

        self.video.save_video(f"{self.save_dir}/video.npz")

        # Test RGB frames were never sent to Mapper, so they cannot enter final
        # Gaussian refinement.
        final_refine_iters = int(self.cfg["mapping"]["final_refine_iters"])
        if final_refine_iters > 0:
            self.mapper.final_refine(iters=final_refine_iters)

        traj_est = extract_full_estimated_c2w(self)
        gt = np.asarray(self.stream.poses)
        processed = int(self.processed_frames.value)
        ate = evaluate_ate_se3(traj_est, gt, processed_frames=processed)
        ate_sim3 = evaluate_ate_sim3(
            traj_est, gt, ate["maxmap_start"], ate["maxmap_end"]
        )

        np.savez(
            os.path.join(self.save_dir, "tartanair_eval_trajectories.npz"),
            estimated_c2w=traj_est,
            gt_c2w=gt,
        )

        if ate["maxmap_start"] is not None and ate["maxmap_frames"] > 0:
            rendering_metrics = evaluate_split_rendering(
                self.mapper,
                self.stream,
                traj_est,
                ate["maxmap_start"],
                ate["maxmap_end"],
                self.save_dir,
            )
        else:
            rendering_metrics = {
                "train_psnr": float("nan"),
                "train_ssim": float("nan"),
                "test_psnr": float("nan"),
                "test_ssim": float("nan"),
                "train_frames_evaluated": 0,
                "test_frames_evaluated": 0,
            }

        elapsed = float(self.online_elapsed.value)
        gaussian_count = int(self.mapper.gaussians.get_xyz.shape[0])
        summary = {
            "status": "ok",
            "sequence": self.cfg["scene"],
            "processed_frames": processed,
            "total_frames": len(self.stream),
            "online_wall_sec": elapsed,
            # Online tracking+mapping FPS. Final global BA, final 3DGS refine,
            # and metric rendering are excluded.
            "fps": (processed / elapsed) if elapsed > 0 else float("nan"),
            "gaussians": gaussian_count,
            "final_refine_iters": final_refine_iters,
            "trajectory_alignment_primary": "SE3 (rigid; scale fixed)",
            "trajectory_alignment_reference": "Sim3 (scale fitted; original monocular protocol)",
            "split_rule": (
                f"source_frame_id % {self.stream.test_every} == "
                f"{self.stream.test_offset} -> test"
            ),
            "test_frames_used_for_pose": True,
            "test_frames_used_for_droid_keyframe_selection": True,
            "test_frames_used_for_gaussian_mapping": False,
            "test_frames_used_for_map_optimization": False,
            **ate,
            **ate_sim3,
            **rendering_metrics,
        }
        summary_path = save_evaluation_summary(self.save_dir, summary)

        self.printer.print(
            "Final TartanAir evaluation:\n"
            f"  MaxMap: {summary['maxmap_percent']:.2f}%\n"
            f"  ATE RMSE SE(3):  {summary['ate_rmse_se3_m']:.6f} m\n"
            f"  ATE RMSE Sim(3): {summary['ate_rmse_sim3_m']:.6f} m\n"
            f"  Train PSNR/SSIM: {summary['train_psnr']:.4f}/{summary['train_ssim']:.6f}\n"
            f"  Test  PSNR/SSIM: {summary['test_psnr']:.4f}/{summary['test_ssim']:.6f}\n"
            f"  Online FPS: {summary['fps']:.4f}\n"
            f"  Gaussians: {summary['gaussians']}\n"
            f"  Summary: {summary_path}",
            FontColor.EVAL,
        )

    def run(self):
        """Run tracker/mapper with fail-fast child-process handling."""
        m_pipe, t_pipe = mp.Pipe()
        tracking_process = mp.Process(target=self.tracking, args=(t_pipe,))
        mapping_process = mp.Process(target=self.mapping, args=(m_pipe,))
        processes = [tracking_process, mapping_process]

        self.num_running_thread[0] += len(processes)
        for process in processes:
            process.start()

        try:
            while True:
                for process in processes:
                    process.join(timeout=0.2)

                if tracking_process.exitcode not in (None, 0):
                    if mapping_process.is_alive():
                        mapping_process.terminate()
                    self._write_failure_summary("tracking", tracking_process.exitcode)
                    raise RuntimeError(
                        f"TartanAir tracking process failed with exit code "
                        f"{tracking_process.exitcode}."
                    )

                if mapping_process.exitcode not in (None, 0):
                    if tracking_process.is_alive():
                        tracking_process.terminate()
                    self._write_failure_summary("mapping", mapping_process.exitcode)
                    raise RuntimeError(
                        f"TartanAir mapping process failed with exit code "
                        f"{mapping_process.exitcode}."
                    )

                if all(process.exitcode is not None for process in processes):
                    break
        except BaseException:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            raise
        finally:
            self.printer.terminate()
