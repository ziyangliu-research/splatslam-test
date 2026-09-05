#!/usr/bin/env python3
import argparse
import csv
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


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_root", default="output/TartanAir_V1", help="Benchmark output root"
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV output path. Defaults to <output_root>/summary.csv.",
    )
    parser.add_argument(
        "sequences",
        nargs="*",
        default=["SE000", "SE001", "SE002", "SE003", "SH000", "SH001", "SH002", "SH003"],
    )
    args = parser.parse_args()

    header = (
        f"{'Sequence':<8} {'Mode':<6} {'MaxMap':>8} "
        f"{'ATE SE3':>10} {'ATE Sim3':>10} "
        f"{'Train P/S/L':>24} {'Test P/S/L':>24} "
        f"{'FPS':>8} {'Online(s)':>10} {'Offline(s)':>11} {'Total(s)':>9} {'G':>10}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for seq in args.sequences:
        seq_root = os.path.join(args.output_root, seq)
        for mode, filename in [
            ("ONLINE", "evaluation_online.json"),
            ("FULL", "evaluation_full.json"),
        ]:
            x = load_json(os.path.join(seq_root, filename))
            if x is None:
                print(
                    f"{seq:<8} {mode:<6} {'-':>8} {'-':>10} {'-':>10} "
                    f"{'-':>24} {'-':>24} {'-':>8} {'-':>10} {'-':>11} {'-':>9} {'-':>10}"
                )
                rows.append({"sequence": seq, "mode": mode, "status": "missing"})
                continue

            train = (
                f"{fmt(x.get('train_psnr'), '.3f')}/"
                f"{fmt(x.get('train_ssim'), '.4f')}/"
                f"{fmt(x.get('train_lpips'), '.4f')}"
            )
            test = (
                f"{fmt(x.get('test_psnr'), '.3f')}/"
                f"{fmt(x.get('test_ssim'), '.4f')}/"
                f"{fmt(x.get('test_lpips'), '.4f')}"
            )
            maxmap = (
                f"{fmt(x.get('maxmap_percent'), '.2f')}%"
                if x.get("maxmap_percent") is not None
                else "-"
            )
            gaussians = str(x.get("gaussians")) if x.get("gaussians") is not None else "-"

            print(
                f"{seq:<8} {mode:<6} {maxmap:>8} "
                f"{fmt(x.get('ate_rmse_se3_m'), '.5f'):>10} "
                f"{fmt(x.get('ate_rmse_sim3_m'), '.5f'):>10} "
                f"{train:>24} {test:>24} "
                f"{fmt(x.get('fps'), '.4f'):>8} "
                f"{fmt(x.get('online_time_sec'), '.2f'):>10} "
                f"{fmt(x.get('offline_time_sec'), '.2f'):>11} "
                f"{fmt(x.get('total_time_sec'), '.2f'):>9} "
                f"{gaussians:>10}"
            )

            rows.append(
                {
                    "sequence": seq,
                    "mode": mode,
                    "status": x.get("status"),
                    "maxmap_percent": x.get("maxmap_percent"),
                    "ate_rmse_se3_m": x.get("ate_rmse_se3_m"),
                    "ate_rmse_sim3_m": x.get("ate_rmse_sim3_m"),
                    "train_psnr": x.get("train_psnr"),
                    "train_ssim": x.get("train_ssim"),
                    "train_lpips": x.get("train_lpips"),
                    "test_psnr": x.get("test_psnr"),
                    "test_ssim": x.get("test_ssim"),
                    "test_lpips": x.get("test_lpips"),
                    "fps": x.get("fps"),
                    "online_time_sec": x.get("online_time_sec"),
                    "final_ba_time_sec": x.get("final_ba_time_sec"),
                    "final_refine_time_sec": x.get("final_refine_time_sec"),
                    "offline_time_sec": x.get("offline_time_sec"),
                    "total_time_sec": x.get("total_time_sec"),
                    "gaussians": x.get("gaussians"),
                    "processed_frames": x.get("processed_frames"),
                    "total_frames": x.get("total_frames"),
                }
            )

    csv_path = args.csv or os.path.join(args.output_root, "summary.csv")
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames = [
        "sequence",
        "mode",
        "status",
        "maxmap_percent",
        "ate_rmse_se3_m",
        "ate_rmse_sim3_m",
        "train_psnr",
        "train_ssim",
        "train_lpips",
        "test_psnr",
        "test_ssim",
        "test_lpips",
        "fps",
        "online_time_sec",
        "final_ba_time_sec",
        "final_refine_time_sec",
        "offline_time_sec",
        "total_time_sec",
        "gaussians",
        "processed_frames",
        "total_frames",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"\nSaved summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
