import numpy as np

from src.slam import SLAM


def _no_sensor_depth_eval(*args, **kwargs):
    """Return NaN depth metrics when the dataset has no sensor depth."""
    return np.nan, np.nan, np.nan


class TartanAirV1SLAM(SLAM):
    """Splat-SLAM wrapper for TartanAir V1 challenge data.

    The challenge package used here contains RGB stereo images and GT poses,
    but no sensor depth. Splat-SLAM is RGB-only and estimates depth internally,
    so this is sufficient for tracking/mapping. The original terminate() routine
    also evaluates estimated depth against sensor depth; patch that evaluator to
    return NaN for this dataset rather than attempting to read unavailable depth.
    """

    def __init__(self, cfg, stream):
        super().__init__(cfg, stream)
        # Keep the replacement at module scope so the SLAM object remains
        # picklable with torch.multiprocessing's spawn start method.
        self.video.eval_depth_l1 = _no_sensor_depth_eval
