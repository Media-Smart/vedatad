# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
from abc import ABCMeta, abstractmethod


class BaseSegmentCoder(metaclass=ABCMeta):
    """Base segment coder."""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def encode(self, segments, gt_segments):
        """Encode deltas between segments and ground truth segments."""
        pass

    @abstractmethod
    def decode(self, segments, segments_pred):
        """Decode the predicted segments according to prediction and base
        segments."""
        pass
