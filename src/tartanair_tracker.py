import time

import torch

from thirdparty.glorie_slam.backend import Backend
from thirdparty.glorie_slam.frontend import Frontend
from thirdparty.glorie_slam.motion_filter import MotionFilter
from src.utils.Printer import FontColor, Printer
from src.utils.datasets import BaseDataset


class TartanAirSplitTracker:
    """DROID tracker with a mapping-only train/test split.

    Every input frame is processed by MotionFilter / Frontend and therefore can
    participate in pose estimation, local BA, loop closure and online/global BA.
    Frames designated as test by the dataset are only prevented from being sent
    to the Gaussian mapper. This keeps the rendering test split isolated from
    map construction and map optimization without weakening pose estimation.
    """

    def __init__(self, slam, pipe):
        self.cfg = slam.cfg
        self.device = self.cfg["device"]
        self.net = slam.droid_net
        self.video = slam.video
        self.verbose = slam.verbose
        self.pipe = pipe
        self.only_tracking = slam.only_tracking
        self.output = slam.save_dir

        self.frontend_window = self.cfg["tracking"]["frontend"]["window"]
        filter_thresh = self.cfg["tracking"]["motion_filter"]["thresh"]
        self.motion_filter = MotionFilter(
            self.net, self.video, self.cfg, thresh=filter_thresh, device=self.device
        )
        self.enable_online_ba = self.cfg["tracking"]["frontend"]["enable_online_ba"]
        self.every_kf = self.cfg["mapping"]["every_keyframe"]
        self.frontend = Frontend(self.net, self.video, self.cfg)
        self.online_ba = Backend(self.net, self.video, self.cfg)
        self.ba_freq = self.cfg["tracking"]["backend"]["ba_freq"]
        self.printer: Printer = slam.printer

        self.processed_frames = slam.processed_frames
        self.online_elapsed = slam.online_elapsed
        self.online_start_time = slam.online_start_time

    def run(self, stream: BaseDataset):
        prev_kf_idx = 0
        prev_ba_idx = 0
        number_of_kf = 0
        intrinsic = stream.get_intrinsic()

        # Start after model/process initialization. Synchronization ensures no
        # setup CUDA work leaks into the measured online processing interval.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.online_start_time.value = time.perf_counter()

        for i in range(len(stream)):
            timestamp, image, _, _ = stream[i]
            with torch.no_grad():
                self.motion_filter.track(timestamp, image, intrinsic)
                self.frontend()

            curr_kf_idx = self.video.counter.value - 1

            if curr_kf_idx != prev_kf_idx and self.frontend.is_initialized:
                number_of_kf += 1

                # Pose-only optimization is intentionally allowed to use test
                # frames: the split applies to Gaussian map construction only.
                if self.enable_online_ba and curr_kf_idx >= prev_ba_idx + self.ba_freq:
                    self.printer.print(
                        f"Online BA at {curr_kf_idx}th keyframe, frame index: {timestamp}",
                        FontColor.TRACKER,
                    )
                    self.online_ba.dense_ba(2)
                    prev_ba_idx = curr_kf_idx

                should_map = (
                    (not self.only_tracking)
                    and (number_of_kf % self.every_kf == 0)
                    and (not stream.is_test_frame(timestamp))
                )

                if should_map:
                    self.pipe.send(
                        {
                            "is_keyframe": True,
                            "video_idx": curr_kf_idx,
                            "timestamp": timestamp,
                            "end": False,
                        }
                    )
                    self.pipe.recv()
                elif (
                    not self.only_tracking
                    and stream.is_test_frame(timestamp)
                    and self.verbose
                ):
                    self.printer.print(
                        f"Frame {timestamp} is TEST: pose tracked/optimized, Gaussian mapping skipped.",
                        FontColor.INFO,
                    )

            prev_kf_idx = curr_kf_idx
            self.processed_frames.value = i + 1
            self.printer.update_pbar()

        # Finish all tracker-context CUDA work before EOF. For mapping runs the
        # mapping process performs its own final CUDA synchronize and determines
        # the authoritative online end time after receiving this EOF.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if self.only_tracking:
            self.online_elapsed.value = time.perf_counter() - self.online_start_time.value
        else:
            self.pipe.send(
                {"is_keyframe": True, "video_idx": None, "timestamp": None, "end": True}
            )
