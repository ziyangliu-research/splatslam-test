import argparse
import json
import os
import random
import warnings
from time import gmtime, strftime

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
)

import numpy as np
import torch
from colorama import Fore, Style

from thirdparty.glorie_slam import config
from src.eth3d_slam import ETH3DRectifiedSLAM
from src.utils.eth3d_rectified import ETH3DRectifiedStereo


SEQUENCES = ["mannequin_face_1", "einstein_1", "sofa_3", "plant_scene_3"]
EXPECTED_FRAMES = {
    "mannequin_face_1": 421,
    "einstein_1": 487,
    "sofa_3": 533,
    "plant_scene_3": 618,
}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _configure_camera_from_rectification(cfg, sequence):
    seq_root = os.path.join(cfg["data"]["dataset_root"], sequence)
    calibration_path = os.path.join(seq_root, "calibration.json")
    if not os.path.isfile(calibration_path):
        raise FileNotFoundError(calibration_path)
    with open(calibration_path, "r", encoding="utf-8") as f:
        calibration = json.load(f)

    width, height = [int(x) for x in calibration["rectified_size"]]
    K = np.asarray(calibration["K_rectified_left"], dtype=np.float64)

    cfg["cam"]["H"] = height
    cfg["cam"]["W"] = width
    cfg["cam"]["fx"] = float(K[0, 0])
    cfg["cam"]["fy"] = float(K[1, 1])
    cfg["cam"]["cx"] = float(K[0, 2])
    cfg["cam"]["cy"] = float(K[1, 2])
    cfg["cam"]["png_depth_scale"] = 1.0

    # DROID's feature pyramid is cleanest at dimensions divisible by 8. When
    # possible, preserve native pixels by symmetrically cropping only the small
    # remainder instead of resizing the rectified image.
    def axis_settings(size):
        remainder = size % 8
        if remainder == 0:
            return size, 0, "native"
        target = size - remainder
        if remainder % 2 == 0:
            return target, remainder // 2, "symmetric_crop"
        return target, 0, "resize"

    h_out, h_edge, h_mode = axis_settings(height)
    w_out, w_edge, w_mode = axis_settings(width)
    cfg["cam"]["H_out"] = h_out
    cfg["cam"]["W_out"] = w_out
    cfg["cam"]["H_edge"] = h_edge
    cfg["cam"]["W_edge"] = w_edge

    return calibration, (w_out, h_out), (w_mode, h_mode)


def main():
    parser = argparse.ArgumentParser(
        description="Run RGB-only Splat-SLAM on rectified ETH3D stereo data."
    )
    parser.add_argument("sequence", choices=SEQUENCES)
    parser.add_argument(
        "--config",
        default="configs/ETH3D/eth3d_rectified.yaml",
    )
    parser.add_argument("--only_tracking", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--final_refine_iters", type=int, default=None)
    parser.add_argument("--test_every", type=int, default=None)
    parser.add_argument("--test_offset", type=int, default=None)
    parser.add_argument("--max_gt_gap_sec", type=float, default=None)
    parser.add_argument(
        "--no_save_test_renders",
        action="store_true",
        help="Do not save held-out test render PNGs.",
    )
    args = parser.parse_args()

    torch.multiprocessing.set_start_method("spawn")

    cfg = config.load_config(args.config, "./configs/splat_slam.yaml")
    cfg["scene"] = args.sequence
    cfg["data"]["input_folder"] = args.sequence

    if args.only_tracking:
        cfg["only_tracking"] = True
        cfg["mono_prior"]["predict_online"] = True
    if args.max_frames is not None:
        cfg["max_frames"] = args.max_frames
    if args.stride is not None:
        cfg["stride"] = args.stride
    if args.final_refine_iters is not None:
        cfg["mapping"]["final_refine_iters"] = args.final_refine_iters
    if args.test_every is not None:
        cfg["evaluation"]["test_every"] = args.test_every
    if args.test_offset is not None:
        cfg["evaluation"]["test_offset"] = args.test_offset
    if args.max_gt_gap_sec is not None:
        cfg["evaluation"]["max_gt_gap_sec"] = args.max_gt_gap_sec
    if args.no_save_test_renders:
        cfg["evaluation"]["save_test_renders"] = False

    calibration, output_size, resize_modes = _configure_camera_from_rectification(
        cfg, args.sequence
    )
    setup_seed(cfg["setup_seed"])

    output_dir = os.path.join(cfg["data"]["output"], cfg["scene"])
    os.makedirs(output_dir, exist_ok=True)

    start_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    start_info = (
        "-" * 30
        + Fore.LIGHTRED_EX
        + f"\nStart Splat-SLAM at {start_time},\n"
        + Style.RESET_ALL
        + "   dataset: ETH3D rectified stereo (Splat-SLAM uses left RGB only),\n"
        + f"   scene: {cfg['scene']}, expected_frames: {EXPECTED_FRAMES[cfg['scene']]},\n"
        + f"   rectified input: {cfg['cam']['W']}x{cfg['cam']['H']}, "
        + f"SLAM output: {output_size[0]}x{output_size[1]} ({resize_modes[0]}/{resize_modes[1]}),\n"
        + f"   baseline: {float(calibration['baseline_rectified_m']):.6f} m,\n"
        + f"   only_tracking: {cfg['only_tracking']}, max_frames: {cfg['max_frames']}, stride: {cfg['stride']},\n"
        + f"   split: source ordinal % {cfg['evaluation']['test_every']} == "
        + f"{cfg['evaluation']['test_offset']} -> test,\n"
        + f"   max GT bracket gap for ATE: {cfg['evaluation']['max_gt_gap_sec']} s,\n"
        + f"   save test renders: {cfg['evaluation']['save_test_renders']},\n"
        + f"   final BA: {cfg['tracking']['backend']['final_ba']}, "
        + f"final refine: {cfg['mapping']['final_refine_iters']},\n"
        + f"   output: {output_dir}\n"
        + "-" * 30
    )
    print(start_info)

    config.save_config(cfg, os.path.join(output_dir, "cfg.yaml"))
    dataset = ETH3DRectifiedStereo(cfg)

    expected = EXPECTED_FRAMES[args.sequence]
    if args.max_frames is None and int(cfg.get("stride", 1)) == 1 and len(dataset) != expected:
        print(
            f"WARNING: expected {expected} frames for {args.sequence}, but loader selected {len(dataset)}."
        )

    slam = ETH3DRectifiedSLAM(cfg, dataset)
    slam.run()

    end_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print(
        "-" * 30
        + Fore.LIGHTRED_EX
        + "\nSplat-SLAM ETH3D finishes!\n"
        + Style.RESET_ALL
        + f"{end_time}\n"
        + "-" * 30
    )


if __name__ == "__main__":
    main()
