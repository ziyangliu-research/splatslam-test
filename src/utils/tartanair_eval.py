import csv
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from evo.core import metrics
from evo.core.trajectory import PoseTrajectory3D
from lietorch import SE3

from thirdparty.gaussian_splatting.gaussian_renderer import render
from thirdparty.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from thirdparty.gaussian_splatting.utils.image_utils import psnr
from thirdparty.gaussian_splatting.utils.loss_utils import ssim
from thirdparty.monogs.utils.camera_utils import Camera


def extract_full_estimated_c2w(slam) -> np.ndarray:
    """Fill non-keyframe poses and return one raw c2w pose per input frame."""
    traj_est_inv = slam.traj_filler(slam.stream)
    traj_est = traj_est_inv.inv().matrix().data.cpu().numpy()

    kf_num = slam.video.counter.value
    if kf_num > 0:
        kf_timestamps = slam.video.timestamp[:kf_num].detach().cpu().int().numpy()
        kf_poses = (
            SE3(slam.video.poses[:kf_num].clone())
            .inv()
            .matrix()
            .data.cpu()
            .numpy()
        )
        valid = (kf_timestamps >= 0) & (kf_timestamps < len(traj_est))
        traj_est[kf_timestamps[valid]] = kf_poses[valid]

    return traj_est


def longest_true_segment(mask: np.ndarray) -> Tuple[Optional[int], Optional[int], int]:
    """Return inclusive [start, end] of the longest contiguous True run."""
    best_start = None
    best_end = None
    best_len = 0
    run_start = None

    for i, value in enumerate(mask.astype(bool)):
        if value and run_start is None:
            run_start = i
        if run_start is not None and ((not value) or i == len(mask) - 1):
            run_end = i if value and i == len(mask) - 1 else i - 1
            run_len = run_end - run_start + 1
            if run_len > best_len:
                best_start, best_end, best_len = run_start, run_end, run_len
            run_start = None

    return best_start, best_end, best_len


def evaluate_ate_se3(
    traj_est_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    processed_frames: int,
) -> Dict:
    """Evaluate translation ATE after rigid SE(3) alignment (no scale fit)."""
    n = min(len(traj_est_c2w), len(gt_c2w))
    finite = np.isfinite(traj_est_c2w[:n]).all(axis=(1, 2))
    finite &= np.isfinite(gt_c2w[:n]).all(axis=(1, 2))

    processed_frames = max(0, min(int(processed_frames), n))
    finite[np.arange(n) >= processed_frames] = False

    start, end, length = longest_true_segment(finite)
    maxmap_ratio = (length / n) if n > 0 else 0.0

    result = {
        "maxmap_ratio": float(maxmap_ratio),
        "maxmap_percent": float(maxmap_ratio * 100.0),
        "maxmap_start": start,
        "maxmap_end": end,
        "maxmap_frames": int(length),
        "ate_rmse_se3_m": float("nan"),
        "ate_statistics_se3": {},
    }

    if start is None or length < 2:
        return result

    indices = np.arange(start, end + 1)
    timestamps = indices.astype(np.float64)

    traj_est = PoseTrajectory3D(
        poses_se3=list(traj_est_c2w[indices]), timestamps=timestamps
    )
    traj_ref = PoseTrajectory3D(
        poses_se3=list(gt_c2w[indices]), timestamps=timestamps
    )

    r_a, t_a, s_a = traj_est.align(traj_ref, correct_scale=False)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((traj_ref, traj_est))
    stats = ape.get_all_statistics()

    result.update(
        {
            "ate_rmse_se3_m": float(stats["rmse"]),
            "ate_statistics_se3": {k: float(v) for k, v in stats.items()},
            "se3_alignment_scale": float(s_a),
            "se3_alignment_rotation": np.asarray(r_a).tolist(),
            "se3_alignment_translation": np.asarray(t_a).tolist(),
        }
    )
    return result


def evaluate_split_rendering(
    mapper,
    stream,
    traj_est_c2w: np.ndarray,
    segment_start: int,
    segment_end: int,
    save_dir: str,
    tag: str = "final",
) -> Dict:
    """Render every frame in the valid map segment and report 8:2 metrics.

    The evaluation itself performs no optimization. Test frames are ephemeral
    cameras at the estimated DROID poses and never become mapper viewpoints.
    """
    device = torch.device(mapper.config["device"])
    projection_matrix = getProjectionMatrix2(
        znear=0.01,
        zfar=100.0,
        fx=stream.fx,
        fy=stream.fy,
        cx=stream.cx,
        cy=stream.cy,
        W=stream.W_out,
        H=stream.H_out,
    ).transpose(0, 1).to(device=device)

    rows = []
    train_psnr, train_ssim = [], []
    test_psnr, test_ssim = [], []
    dummy_depth = np.zeros((stream.H_out, stream.W_out), dtype=np.float32)

    with torch.no_grad():
        for idx in range(int(segment_start), int(segment_end) + 1):
            _, color, _, _ = stream[idx]
            gt_image = color.squeeze(0).to(device=device, dtype=torch.float32)

            c2w = torch.as_tensor(
                traj_est_c2w[idx], dtype=torch.float32, device=device
            )
            w2c = torch.linalg.inv(c2w)

            data = {
                "gt_color": gt_image,
                "glorie_depth": dummy_depth,
                "glorie_pose": w2c,
                "idx": idx,
            }
            camera = Camera.init_from_dataset(stream, data, projection_matrix)
            camera.update_RT(camera.R_gt, camera.T_gt)

            rendered = render(
                camera,
                mapper.gaussians,
                mapper.pipeline_params,
                mapper.background,
            )["render"].detach()
            rendered = torch.clamp(rendered, 0.0, 1.0)

            psnr_value = float(
                psnr(rendered.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()
            )
            ssim_value = float(
                ssim(rendered.unsqueeze(0), gt_image.unsqueeze(0)).item()
            )

            split = "test" if stream.is_test_frame(idx) else "train"
            source_id = (
                int(stream.source_frame_ids[idx])
                if hasattr(stream, "source_frame_ids")
                else int(idx)
            )
            rows.append(
                {
                    "frame_index": int(idx),
                    "source_frame_id": source_id,
                    "split": split,
                    "psnr": psnr_value,
                    "ssim": ssim_value,
                }
            )

            if split == "test":
                test_psnr.append(psnr_value)
                test_ssim.append(ssim_value)
            else:
                train_psnr.append(psnr_value)
                train_ssim.append(ssim_value)

            del camera, rendered, gt_image, c2w, w2c

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"split_render_metrics_{tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_index", "source_frame_id", "split", "psnr", "ssim"],
        )
        writer.writeheader()
        writer.writerows(rows)

    def mean_or_nan(values):
        return float(np.mean(values)) if values else float("nan")

    return {
        "train_psnr": mean_or_nan(train_psnr),
        "train_ssim": mean_or_nan(train_ssim),
        "test_psnr": mean_or_nan(test_psnr),
        "test_ssim": mean_or_nan(test_ssim),
        "train_frames_evaluated": len(train_psnr),
        "test_frames_evaluated": len(test_psnr),
        "per_frame_metrics_csv": csv_path,
    }


def save_evaluation_summary(
    save_dir: str,
    summary: Dict,
    basename: str = "evaluation_summary",
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{basename}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    text_path = os.path.join(save_dir, f"{basename}.txt")
    fps_value = summary.get("fps")
    fps_text = "-" if fps_value is None else f"{float(fps_value):.4f}"
    gaussians = summary.get("gaussians")
    gaussians_text = "-" if gaussians is None else str(int(gaussians))

    with open(text_path, "w", encoding="utf-8") as f:
        f.write(
            "Mode      Sequence      MaxMap       ATE SE3(m)    ATE Sim3(m)   "
            "Train PSNR/SSIM       Test PSNR/SSIM        FPS       Gaussians\n"
        )
        f.write(
            f"{summary.get('mode', '-'):8s}  "
            f"{summary.get('sequence', '-'):10s}  "
            f"{summary.get('maxmap_percent', float('nan')):8.2f}%  "
            f"{summary.get('ate_rmse_se3_m', float('nan')):12.6f}  "
            f"{summary.get('ate_rmse_sim3_m', float('nan')):12.6f}  "
            f"{summary.get('train_psnr', float('nan')):7.4f}/"
            f"{summary.get('train_ssim', float('nan')):.6f}    "
            f"{summary.get('test_psnr', float('nan')):7.4f}/"
            f"{summary.get('test_ssim', float('nan')):.6f}    "
            f"{fps_text:>8s}  {gaussians_text:>10s}\n"
        )
    return path
