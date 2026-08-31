import numpy as np

from src.slam import SLAM


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

        def _no_sensor_depth_eval(*args, **kwargs):
            return np.nan, np.nan, np.nan

        self.video.eval_depth_l1 = _no_sensor_depth_eval
