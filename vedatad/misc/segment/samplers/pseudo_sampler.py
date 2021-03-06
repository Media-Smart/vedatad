# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import torch

from vedacore.misc import registry
from .base_sampler import BaseSampler
from .sampling_result import SamplingResult


@registry.register_module('segment_sampler')
class PseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, segments, gt_segments, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            segments (torch.Tensor): Segments
            gt_segments (torch.Tensor): Ground truth segments

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = segments.new_zeros(segments.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(pos_inds, neg_inds, segments,
                                         gt_segments, assign_result, gt_flags)
        return sampling_result
