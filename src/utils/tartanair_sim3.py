import numpy as np
from evo.core import metrics
from evo.core.trajectory import PoseTrajectory3D


def evaluate_ate_sim3(traj_est_c2w, gt_c2w, segment_start, segment_end):
    """Evaluate translation ATE after Sim(3) alignment on a fixed frame segment.

    The segment is supplied by the SE(3)/MaxMap evaluator so SE(3) and Sim(3)
    are evaluated on exactly the same frames. correct_scale=True matches the
    original monocular Splat-SLAM trajectory evaluation protocol.
    """
    result = {
        "ate_rmse_sim3_m": float("nan"),
        "ate_statistics_sim3": {},
        "sim3_alignment_scale": float("nan"),
    }

    if segment_start is None or segment_end is None:
        return result

    start = int(segment_start)
    end = int(segment_end)
    if end - start + 1 < 2:
        return result

    indices = np.arange(start, end + 1)
    finite = np.isfinite(traj_est_c2w[indices]).all(axis=(1, 2))
    finite &= np.isfinite(gt_c2w[indices]).all(axis=(1, 2))
    indices = indices[finite]
    if len(indices) < 2:
        return result

    timestamps = indices.astype(np.float64)
    traj_est = PoseTrajectory3D(
        poses_se3=list(traj_est_c2w[indices]), timestamps=timestamps
    )
    traj_ref = PoseTrajectory3D(
        poses_se3=list(gt_c2w[indices]), timestamps=timestamps
    )

    r_a, t_a, s_a = traj_est.align(traj_ref, correct_scale=True)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((traj_ref, traj_est))
    stats = ape.get_all_statistics()

    result.update(
        {
            "ate_rmse_sim3_m": float(stats["rmse"]),
            "ate_statistics_sim3": {k: float(v) for k, v in stats.items()},
            "sim3_alignment_scale": float(s_a),
            "sim3_alignment_rotation": np.asarray(r_a).tolist(),
            "sim3_alignment_translation": np.asarray(t_a).tolist(),
        }
    )
    return result
