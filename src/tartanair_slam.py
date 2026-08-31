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
    return np.nan, np.nan, np.nan


class TartanAirV1SLAM(SLAM):
    """TartanAir benchmark wrapper with one-run ONLINE + FULL evaluation.

    ONLINE is evaluated immediately after the input stream ends, before any
    end-of-sequence global BA or final Gaussian refinement. It is therefore the
    exact state corresponding to final_ba=False and final_refine_iters=0.

    FULL continues from that same state with the original Splat-SLAM final
    global BA and final Gaussian refinement, then evaluates again.
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
            "mode": "FAILED",
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

    def _evaluate_current_state(self, mode, include_fps, trajectory_filename):
        """Evaluate the current pose/map state without modifying it."""
        traj_est = extract_full_estimated_c2w(self)
        gt = np.asarray(self.stream.poses)
        processed = int(self.processed_frames.value)

        ate = evaluate_ate_se3(traj_est, gt, processed_frames=processed)
        ate_sim3 = evaluate_ate_sim3(
            traj_est, gt, ate["maxmap_start"], ate["maxmap_end"]
        )

        np.savez(
            os.path.join(self.save_dir, trajectory_filename),
            estimated_c2w=traj_est,
            gt_c2w=gt,
        )

        if (not self.only_tracking) and ate["maxmap_start"] is not None and ate["maxmap_frames"] > 0:
            rendering_metrics = evaluate_split_rendering(
                self.mapper,
                self.stream,
                traj_est,
                ate["maxmap_start"],
                ate["maxmap_end"],
                self.save_dir,
                tag=mode.lower(),
            )
            gaussian_count = int(self.mapper.gaussians.get_xyz.shape[0])
        else:
            rendering_metrics = {
                "train_psnr": float("nan"),
                "train_ssim": float("nan"),
                "test_psnr": float("nan"),
                "test_ssim": float("nan"),
                "train_frames_evaluated": 0,
                "test_frames_evaluated": 0,
            }
            gaussian_count = None

        elapsed = float(self.online_elapsed.value)
        online_fps = (processed / elapsed) if elapsed > 0 else float("nan")

        summary = {
            "status": "ok" if not self.only_tracking else "ok_tracking_only",
            "mode": mode,
            "sequence": self.cfg["scene"],
            "processed_frames": processed,
            "total_frames": len(self.stream),
            "online_wall_sec": elapsed,
            # Only ONLINE gets an FPS entry. FULL deliberately reports no FPS
            # because final BA/refinement are post-processing.
            "fps": online_fps if include_fps else None,
            "online_fps_reference": online_fps,
            "gaussians": gaussian_count,
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
        return summary

    def _print_summary(self, title, summary, path):
        fps_text = "-" if summary.get("fps") is None else f"{summary['fps']:.4f}"
        gaussian_text = "-" if summary.get("gaussians") is None else str(summary["gaussians"])
        self.printer.print(
            f"{title}:\n"
            f"  MaxMap: {summary['maxmap_percent']:.2f}%\n"
            f"  ATE RMSE SE(3):  {summary['ate_rmse_se3_m']:.6f} m\n"
            f"  ATE RMSE Sim(3): {summary['ate_rmse_sim3_m']:.6f} m\n"
            f"  Train PSNR/SSIM: {summary['train_psnr']:.4f}/{summary['train_ssim']:.6f}\n"
            f"  Test  PSNR/SSIM: {summary['test_psnr']:.4f}/{summary['test_ssim']:.6f}\n"
            f"  FPS: {fps_text}\n"
            f"  Gaussians: {gaussian_text}\n"
            f"  Summary: {path}",
            FontColor.EVAL,
        )

    def terminate(self):
        """Evaluate ONLINE state, then continue to FULL post-processed state."""
        # ---------------- ONLINE checkpoint ----------------
        # At this exact point no end-of-sequence BA/refinement has run. This is
        # equivalent to final_ba=False + final_refine_iters=0.
        self.video.save_video(f"{self.save_dir}/video_online.npz")
        online_summary = self._evaluate_current_state(
            mode="ONLINE",
            include_fps=True,
            trajectory_filename="tartanair_eval_trajectories_online.npz",
        )
        online_summary.update(
            {
                "final_ba_applied": False,
                "final_refine_iters_applied": 0,
                "fps_definition": "input frames / online tracking+mapping wall time; excludes final BA, final refinement, and metric rendering",
            }
        )
        online_path = save_evaluation_summary(
            self.save_dir, online_summary, basename="evaluation_online"
        )
        self._print_summary("ONLINE evaluation (no final BA / no final refine)", online_summary, online_path)

        if self.only_tracking:
            # Keep the conventional summary alias for tracking-only runs.
            save_evaluation_summary(self.save_dir, online_summary)
            return

        # Evaluation rendering is not part of the timed online pipeline. Free
        # temporary allocations before continuing the original post-processing.
        torch.cuda.empty_cache()

        # ---------------- FULL post-processing ----------------
        final_ba_enabled = bool(self.cfg["tracking"]["backend"]["final_ba"])
        final_refine_iters = int(self.cfg["mapping"]["final_refine_iters"])

        if final_ba_enabled:
            self.backend()

        self.video.save_video(f"{self.save_dir}/video_full.npz")

        if final_refine_iters > 0:
            self.mapper.final_refine(iters=final_refine_iters)

        full_summary = self._evaluate_current_state(
            mode="FULL",
            include_fps=False,
            trajectory_filename="tartanair_eval_trajectories_full.npz",
        )
        full_summary.update(
            {
                "final_ba_applied": final_ba_enabled,
                "final_refine_iters_applied": final_refine_iters,
                "fps_definition": None,
                "fps_note": "FULL result includes end-of-sequence post-processing; FPS is intentionally not reported.",
            }
        )
        full_path = save_evaluation_summary(
            self.save_dir, full_summary, basename="evaluation_full"
        )
        # Backward-compatible alias: evaluation_summary.* refers to FULL.
        save_evaluation_summary(self.save_dir, full_summary)
        self._print_summary(
            f"FULL evaluation (final BA={final_ba_enabled}, final refine={final_refine_iters})",
            full_summary,
            full_path,
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
                        f"TartanAir tracking process failed with exit code {tracking_process.exitcode}."
                    )

                if mapping_process.exitcode not in (None, 0):
                    if tracking_process.is_alive():
                        tracking_process.terminate()
                    self._write_failure_summary("mapping", mapping_process.exitcode)
                    raise RuntimeError(
                        f"TartanAir mapping process failed with exit code {mapping_process.exitcode}."
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
