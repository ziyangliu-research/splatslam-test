import os
import time

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
    end-of-sequence global BA or final Gaussian refinement. FULL continues from
    exactly that state with the original final global BA + 3DGS refinement.

    Runtime metrics exclude PSNR/SSIM/LPIPS rendering/evaluation itself:
      online_time = tracking + online BA + online Gaussian mapping
      offline_time = final global BA + final Gaussian refinement
      total_time = online_time + offline_time
    """

    def __init__(self, cfg, stream):
        super().__init__(cfg, stream)
        self.processed_frames = mp.Value("i", 0)
        self.online_start_time = mp.Value("d", 0.0)
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
            "split_rule": "source_frame_id % 5 == 4 -> test",
        }
        save_evaluation_summary(self.save_dir, summary)

    def _evaluate_current_state(self, mode, include_fps, trajectory_filename):
        """Evaluate current pose/map state without modifying map parameters."""
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

        if (
            (not self.only_tracking)
            and ate["maxmap_start"] is not None
            and ate["maxmap_frames"] > 0
        ):
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
                "train_lpips": float("nan"),
                "test_psnr": float("nan"),
                "test_ssim": float("nan"),
                "test_lpips": float("nan"),
                "train_frames_evaluated": 0,
                "test_frames_evaluated": 0,
            }
            gaussian_count = None

        online_time = float(self.online_elapsed.value)
        online_fps = (processed / online_time) if online_time > 0 else float("nan")

        summary = {
            "status": "ok" if not self.only_tracking else "ok_tracking_only",
            "mode": mode,
            "sequence": self.cfg["scene"],
            "processed_frames": processed,
            "total_frames": len(self.stream),
            "online_wall_sec": online_time,
            "online_time_sec": online_time,
            # Only ONLINE reports FPS. FULL is post-processed and deliberately '-'.
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
        gaussian_text = (
            "-" if summary.get("gaussians") is None else str(summary["gaussians"])
        )

        def fmt_time(key):
            value = summary.get(key)
            return "-" if value is None else f"{float(value):.2f} s"

        self.printer.print(
            f"{title}:\n"
            f"  MaxMap: {summary['maxmap_percent']:.2f}%\n"
            f"  ATE RMSE SE(3):  {summary['ate_rmse_se3_m']:.6f} m\n"
            f"  ATE RMSE Sim(3): {summary['ate_rmse_sim3_m']:.6f} m\n"
            f"  Train PSNR/SSIM/LPIPS: {summary['train_psnr']:.4f}/"
            f"{summary['train_ssim']:.6f}/{summary['train_lpips']:.6f}\n"
            f"  Test  PSNR/SSIM/LPIPS: {summary['test_psnr']:.4f}/"
            f"{summary['test_ssim']:.6f}/{summary['test_lpips']:.6f}\n"
            f"  FPS: {fps_text}\n"
            f"  Online time:  {fmt_time('online_time_sec')}\n"
            f"  Offline time: {fmt_time('offline_time_sec')}\n"
            f"  Total time:   {fmt_time('total_time_sec')}\n"
            f"  Gaussians: {gaussian_text}\n"
            f"  Summary: {path}",
            FontColor.EVAL,
        )

    def terminate(self):
        """Evaluate ONLINE, time offline optimization, then evaluate FULL."""
        # For mapping runs this method is called after Mapper.run() receives EOF.
        # Synchronize the mapper CUDA context so the online endpoint includes all
        # outstanding map work. Tracker synchronized its own context before EOF.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.online_start_time.value > 0:
            self.online_elapsed.value = (
                time.perf_counter() - float(self.online_start_time.value)
            )

        # ---------------- ONLINE checkpoint ----------------
        # Equivalent algorithmic state to final_ba=False + final_refine_iters=0.
        self.video.save_video(f"{self.save_dir}/video_online.npz")
        online_summary = self._evaluate_current_state(
            mode="ONLINE",
            include_fps=True,
            trajectory_filename="tartanair_eval_trajectories_online.npz",
        )
        online_time = float(self.online_elapsed.value)
        online_summary.update(
            {
                "final_ba_applied": False,
                "final_refine_iters_applied": 0,
                "offline_time_sec": 0.0,
                "final_ba_time_sec": 0.0,
                "final_refine_time_sec": 0.0,
                "total_time_sec": online_time,
                "fps_definition": (
                    "input frames / online tracking+mapping wall time; excludes "
                    "final BA, final refinement, and all metric rendering"
                ),
                "time_definition": (
                    "algorithm processing only; PSNR/SSIM/LPIPS evaluation and "
                    "result-file I/O are excluded"
                ),
            }
        )
        online_path = save_evaluation_summary(
            self.save_dir, online_summary, basename="evaluation_online"
        )
        self._print_summary(
            "ONLINE evaluation (no final BA / no final refine)",
            online_summary,
            online_path,
        )

        if self.only_tracking:
            save_evaluation_summary(self.save_dir, online_summary)
            return

        # ONLINE rendering is evaluation only. Finish it completely and clear
        # temporary allocations before starting the OFFLINE optimization timer.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # ---------------- OFFLINE post-processing ----------------
        final_ba_enabled = bool(self.cfg["tracking"]["backend"]["final_ba"])
        final_refine_iters = int(self.cfg["mapping"]["final_refine_iters"])

        final_ba_time = 0.0
        if final_ba_enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.backend()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            final_ba_time = time.perf_counter() - t0

        final_refine_time = 0.0
        if final_refine_iters > 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.mapper.final_refine(iters=final_refine_iters)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            final_refine_time = time.perf_counter() - t0

        offline_time = final_ba_time + final_refine_time
        total_time = online_time + offline_time

        # Saving and metric rendering happen after timing is complete.
        self.video.save_video(f"{self.save_dir}/video_full.npz")
        full_summary = self._evaluate_current_state(
            mode="FULL",
            include_fps=False,
            trajectory_filename="tartanair_eval_trajectories_full.npz",
        )
        full_summary.update(
            {
                "final_ba_applied": final_ba_enabled,
                "final_refine_iters_applied": final_refine_iters,
                "final_ba_time_sec": final_ba_time,
                "final_refine_time_sec": final_refine_time,
                "offline_time_sec": offline_time,
                "total_time_sec": total_time,
                "fps_definition": None,
                "fps_note": (
                    "FULL includes end-of-sequence post-processing; FPS is "
                    "intentionally not reported."
                ),
                "time_definition": (
                    "total_time = online_time + final_BA_time + final_refine_time; "
                    "metric rendering and result-file I/O excluded"
                ),
            }
        )
        full_path = save_evaluation_summary(
            self.save_dir, full_summary, basename="evaluation_full"
        )
        # Backward-compatible alias: evaluation_summary.* refers to FULL.
        save_evaluation_summary(self.save_dir, full_summary)
        self._print_summary(
            f"FULL evaluation (final BA={final_ba_enabled}, "
            f"final refine={final_refine_iters})",
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
