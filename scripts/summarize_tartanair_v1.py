#!/usr/bin/env python3
import argparse
import json
import math
import os


def fmt(value, pattern):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(value):
        return "-"
    return format(value, pattern)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_root", default="output/TartanAir_V1", help="Benchmark output root"
    )
    parser.add_argument(
        "sequences", nargs="*", default=["SH000", "SH001", "SH002", "SH003"]
    )
    args = parser.parse_args()

    header = (
        f"{'Sequence':<10} {'Status':<10} {'MaxMap':>9} {'ATE SE3(m)':>12} "
        f"{'Train PSNR/SSIM':>20} {'Test PSNR/SSIM':>20} "
        f"{'FPS':>9} {'Gaussians':>12}"
    )
    print(header)
    print("-" * len(header))

    for seq in args.sequences:
        path = os.path.join(args.output_root, seq, "evaluation_summary.json")
        if not os.path.isfile(path):
            print(f"{seq:<10} {'missing':<10} {'-':>9} {'-':>12} {'-':>20} {'-':>20} {'-':>9} {'-':>12}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            x = json.load(f)

        train = f"{fmt(x.get('train_psnr'), '.4f')}/{fmt(x.get('train_ssim'), '.6f')}"
        test = f"{fmt(x.get('test_psnr'), '.4f')}/{fmt(x.get('test_ssim'), '.6f')}"
        maxmap = f"{fmt(x.get('maxmap_percent'), '.2f')}%" if x.get("maxmap_percent") is not None else "-"
        gaussians = str(x.get("gaussians", "-"))

        print(
            f"{seq:<10} {str(x.get('status', '-')):<10} {maxmap:>9} "
            f"{fmt(x.get('ate_rmse_se3_m'), '.6f'):>12} "
            f"{train:>20} {test:>20} "
            f"{fmt(x.get('fps'), '.4f'):>9} {gaussians:>12}"
        )


if __name__ == "__main__":
    main()
