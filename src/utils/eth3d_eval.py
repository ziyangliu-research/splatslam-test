import csv
import os

import cv2
import numpy as np
import torch
from evo.core import metrics
from evo.core.trajectory import PoseTrajectory3D
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from thirdparty.gaussian_splatting.gaussian_renderer import render
from thirdparty.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from thirdparty.gaussian_splatting.utils.image_utils import psnr
from thirdparty.gaussian_splatting.utils.loss_utils import ssim
from thirdparty.monogs.utils.camera_utils import Camera
from src.utils.tartanair_eval import longest_true_segment


def _trajectory_indices(traj_est_c2w, gt_c2w, processed_frames, gt_valid_mask):
    n = min(len(traj_est_c2w), len(gt_c2w))
    est_finite = np.isfinite(traj_est_c2w[:n]).all(axis=(1, 2))
    processed_frames = max(0, min(int(processed_frames), n))
    est_finite[np.arange(n) >= processed_frames] = False

    # MaxMap describes the SLAM trajectory, not ETH3D GT availability.
    start, end, length = longest_true_segment(est_finite)
    maxmap_ratio = (length / n) if n > 0 else 0.0

    eval_mask = est_finite & np.isfinite(gt_c2w[:n]).all(axis=(1, 2))
    if gt_valid_mask is not None:
        valid = np.asarray(gt_valid_mask[:n], dtype=bool)
        eval_mask &= valid
    if start is not None:
        range_mask = np.zeros(n, dtype=bool)
        range_mask[int(start): int(end) + 1] = True
        eval_mask &= range_mask

    indices = np.flatnonzero(eval_mask)
    return start, end, length, maxmap_ratio, indices


def evaluate_ate_eth3d(traj_est_c2w, gt_c2w, processed_frames, gt_valid_mask):
    """Report SE(3) and Sim(3) ATE on identical reliable ETH3D GT frames."""
    start, end, length, ratio, indices = _trajectory_indices(
        traj_est_c2w, gt_c2w, processed_frames, gt_valid_mask
    )
    result = {
        "maxmap_ratio": float(ratio),
        "maxmap_percent": float(ratio * 100.0),
        "maxmap_start": start,
        "maxmap_end": end,
        "maxmap_frames": int(length),
        "ate_gt_frames_evaluated": int(len(indices)),
        "ate_rmse_se3_m": float("nan"),
        "ate_rmse_sim3_m": float("nan"),
        "ate_statistics_se3": {},
        "ate_statistics_sim3": {},
        "sim3_alignment_scale": float("nan"),
    }
    if len(indices) < 2:
        return result

    timestamps = indices.astype(np.float64)

    # SE(3): metric scale fixed.
    est_se3 = PoseTrajectory3D(poses_se3=list(traj_est_c2w[indices]), timestamps=timestamps)
    ref_se3 = PoseTrajectory3D(poses_se3=list(gt_c2w[indices]), timestamps=timestamps)
    r_se3, t_se3, s_se3 = est_se3.align(ref_se3, correct_scale=False)
    ape_se3 = metrics.APE(metrics.PoseRelation.translation_part)
    ape_se3.process_data((ref_se3, est_se3))
    stats_se3 = ape_se3.get_all_statistics()

    # Sim(3): original monocular Splat-SLAM style scale fitting.
    est_sim3 = PoseTrajectory3D(poses_se3=list(traj_est_c2w[indices]), timestamps=timestamps)
    ref_sim3 = PoseTrajectory3D(poses_se3=list(gt_c2w[indices]), timestamps=timestamps)
    r_sim3, t_sim3, s_sim3 = est_sim3.align(ref_sim3, correct_scale=True)
    ape_sim3 = metrics.APE(metrics.PoseRelation.translation_part)
    ape_sim3.process_data((ref_sim3, est_sim3))
    stats_sim3 = ape_sim3.get_all_statistics()

    result.update(
        {
            "ate_rmse_se3_m": float(stats_se3["rmse"]),
            "ate_statistics_se3": {k: float(v) for k, v in stats_se3.items()},
            "se3_alignment_scale": float(s_se3),
            "se3_alignment_rotation": np.asarray(r_se3).tolist(),
            "se3_alignment_translation": np.asarray(t_se3).tolist(),
            "ate_rmse_sim3_m": float(stats_sim3["rmse"]),
            "ate_statistics_sim3": {k: float(v) for k, v in stats_sim3.items()},
            "sim3_alignment_scale": float(s_sim3),
            "sim3_alignment_rotation": np.asarray(r_sim3).tolist(),
            "sim3_alignment_translation": np.asarray(t_sim3).tolist(),
        }
    )
    return result


def _save_test_render(save_dir, tag, stream, idx, gt_image, rendered):
    render_dir = os.path.join(save_dir, "test_renders", tag)
    os.makedirs(render_dir, exist_ok=True)
    label = stream.frame_label(idx) if hasattr(stream, "frame_label") else f"{idx:06d}"

    gt = (
        torch.clamp(gt_image, 0.0, 1.0)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )
    pred = (
        torch.clamp(rendered, 0.0, 1.0)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )
    gt_u8 = np.round(gt * 255.0).astype(np.uint8)
    pred_u8 = np.round(pred * 255.0).astype(np.uint8)
    diff_u8 = np.abs(gt_u8.astype(np.int16) - pred_u8.astype(np.int16)).astype(np.uint8)
    comparison = np.concatenate([gt_u8, pred_u8, diff_u8], axis=1)

    cv2.imwrite(
        os.path.join(render_dir, f"{label}_render.png"),
        cv2.cvtColor(pred_u8, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(render_dir, f"{label}_comparison.png"),
        cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR),
    )
    return render_dir


def evaluate_split_rendering_eth3d(
    mapper,
    stream,
    traj_est_c2w,
    segment_start,
    segment_end,
    save_dir,
    tag="final",
    save_test_renders=True,
):
    """Evaluate all tracked frames; optionally save every held-out test render."""
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

    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device).eval()

    rows = []
    train_psnr, train_ssim, train_lpips = [], [], []
    test_psnr, test_ssim, test_lpips = [], [], []
    dummy_depth = np.zeros((stream.H_out, stream.W_out), dtype=np.float32)
    test_render_dir = None

    with torch.no_grad():
        for idx in range(int(segment_start), int(segment_end) + 1):
            _, color, _, _ = stream[idx]
            gt_image = color.squeeze(0).to(device=device, dtype=torch.float32)
            c2w = torch.as_tensor(traj_est_c2w[idx], dtype=torch.float32, device=device)
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
                camera, mapper.gaussians, mapper.pipeline_params, mapper.background
            )["render"].detach()
            rendered = torch.clamp(rendered, 0.0, 1.0)

            psnr_value = float(psnr(rendered.unsqueeze(0), gt_image.unsqueeze(0)).mean().item())
            ssim_value = float(ssim(rendered.unsqueeze(0), gt_image.unsqueeze(0)).item())
            lpips_value = float(lpips_metric(rendered.unsqueeze(0), gt_image.unsqueeze(0)).item())
            lpips_metric.reset()

            split = "test" if stream.is_test_frame(idx) else "train"
            timestamp = float(stream.source_timestamps[idx])
            source_index = int(stream.source_frame_indices[idx])
            rows.append(
                {
                    "frame_index": int(idx),
                    "source_frame_index": source_index,
                    "timestamp": f"{timestamp:.9f}",
                    "split": split,
                    "psnr": psnr_value,
                    "ssim": ssim_value,
                    "lpips": lpips_value,
                }
            )

            if split == "test":
                test_psnr.append(psnr_value)
                test_ssim.append(ssim_value)
                test_lpips.append(lpips_value)
                if save_test_renders:
                    test_render_dir = _save_test_render(
                        save_dir, tag, stream, idx, gt_image, rendered
                    )
            else:
                train_psnr.append(psnr_value)
                train_ssim.append(ssim_value)
                train_lpips.append(lpips_value)

            del camera, rendered, gt_image, c2w, w2c

    del lpips_metric
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"split_render_metrics_{tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "source_frame_index",
                "timestamp",
                "split",
                "psnr",
                "ssim",
                "lpips",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    def mean_or_nan(values):
        return float(np.mean(values)) if values else float("nan")

    return {
        "train_psnr": mean_or_nan(train_psnr),
        "train_ssim": mean_or_nan(train_ssim),
        "train_lpips": mean_or_nan(train_lpips),
        "test_psnr": mean_or_nan(test_psnr),
        "test_ssim": mean_or_nan(test_ssim),
        "test_lpips": mean_or_nan(test_lpips),
        "train_frames_evaluated": len(train_psnr),
        "test_frames_evaluated": len(test_psnr),
        "per_frame_metrics_csv": csv_path,
        "test_render_dir": test_render_dir,
    }
