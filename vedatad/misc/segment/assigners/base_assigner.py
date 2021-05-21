# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
from abc import ABCMeta, abstractmethod


class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns segments to ground truth segments."""

    @abstractmethod
    def assign(self,
               segments,
               gt_segments,
               gt_segments_ignore=None,
               gt_labels=None):
        """Assign segments to either a ground truth segment or a negative
        segment."""
        pass
