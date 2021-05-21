# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
from vedacore.misc import registry
from .base_segment_coder import BaseSegmentCoder


@registry.register_module('segment_coder')
class PseudoSegmentCoder(BaseSegmentCoder):
    """Pseudo segment coder."""

    def __init__(self, **kwargs):
        super(BaseSegmentCoder, self).__init__(**kwargs)

    def encode(self, segments, gt_segments):
        """torch.Tensor: return the given ``segments``"""
        return gt_segments

    def decode(self, segments, pred_segments):
        """torch.Tensor: return the given ``pred_segments``"""
        return pred_segments
