import argparse
import os
import random
from time import gmtime, strftime

import numpy as np
import torch
from colorama import Fore, Style

from thirdparty.glorie_slam import config
from src.tartanair_slam import TartanAirV1SLAM
from src.utils.tartanair_v1 import TartanAirV1StereoChallenge


SEQUENCES = [f"SE{i:03d}" for i in range(8)] + [f"SH{i:03d}" for i in range(8)]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(
        description="Run RGB-only Splat-SLAM on TartanAir V1 Stereo Challenge data."
    )
    parser.add_argument("sequence", choices=SEQUENCES, help="SE000-SE007 or SH000-SH007")
    parser.add_argument(
        "--config",
        default="configs/TartanAir/tartanair_v1.yaml",
        help="TartanAir dataset config.",
    )
    parser.add_argument(
        "--only_tracking",
        action="store_true",
        help="Run tracking only (no Gaussian mapping/rendering).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Override max_frames for a smoke test, e.g. 40. Omit for the full sequence.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override frame stride. Omit to use the original stride=1.",
    )
    parser.add_argument(
        "--final_refine_iters",
        type=int,
        default=None,
        help="Override final 3DGS refinement iterations for quick tests.",
    )
    args = parser.parse_args()

    torch.multiprocessing.set_start_method("spawn")

    cfg = config.load_config(args.config, "./configs/splat_slam.yaml")
    cfg["scene"] = args.sequence
    # BaseDataset expects input_folder. The TartanAir loader maps it to the
    # sequence directory under dataset_root.
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

    setup_seed(cfg["setup_seed"])

    output_dir = os.path.join(cfg["data"]["output"], cfg["scene"])
    os.makedirs(output_dir, exist_ok=True)

    start_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    start_info = (
        "-" * 30
        + Fore.LIGHTRED_EX
        + f"\nStart Splat-SLAM at {start_time},\n"
        + Style.RESET_ALL
        + f"   dataset: TartanAir V1 Stereo Challenge (left RGB),\n"
        + f"   scene: {cfg['scene']},\n"
        + f"   only_tracking: {cfg['only_tracking']},\n"
        + f"   max_frames: {cfg['max_frames']}, stride: {cfg['stride']},\n"
        + f"   output: {output_dir}\n"
        + "-" * 30
    )
    print(start_info)

    config.save_config(cfg, os.path.join(output_dir, "cfg.yaml"))

    dataset = TartanAirV1StereoChallenge(cfg)
    slam = TartanAirV1SLAM(cfg, dataset)
    slam.run()

    end_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print(
        "-" * 30
        + Fore.LIGHTRED_EX
        + "\nSplat-SLAM finishes!\n"
        + Style.RESET_ALL
        + f"{end_time}\n"
        + "-" * 30
    )


if __name__ == "__main__":
    main()
