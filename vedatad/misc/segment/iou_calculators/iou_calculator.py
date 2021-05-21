# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
from vedacore.misc import registry
from ..segment import segment_overlaps


@registry.register_module('iou_calculator')
class SegmentOverlaps(object):
    """IoU Calculator."""

    def __call__(self, segments1, segments2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            segments1 (Tensor): segments have shape (m, 2) in <start, end>
                format, or shape (m, 3) in <start, end, score> format.
            segments2 (Tensor): segments have shape (m, 2) in <start, end>
                format, shape (m, 3) in <start, end, score> format, or be
                empty. If is_aligned is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union) or iof (intersection
                over foreground).

        Returns:
            ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
        """
        assert segments1.size(-1) in [0, 2, 3]
        assert segments2.size(-1) in [0, 2, 3]
        if segments1.size(-1) == 3:
            segments1 = segments1[..., :2]
        if segments2.size(-1) == 3:
            segments2 = segments2[..., :4]
        return segment_overlaps(segments1, segments2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str
