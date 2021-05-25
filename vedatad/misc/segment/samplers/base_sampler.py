# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
from abc import ABCMeta, abstractmethod

import torch

from .sampling_result import SamplingResult


class BaseSampler(metaclass=ABCMeta):
    """Base class of samplers."""

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    @abstractmethod
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Sample positive samples."""
        pass

    @abstractmethod
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Sample negative samples."""
        pass

    def sample(self,
               assign_result,
               segments,
               gt_segments,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative segments.

        This is a simple implementation of segment sampling given candidates,
        assigning results and ground truth segments.

        Args:
            assign_result (:obj:`AssignResult`): Segment assigning results.
            segments (Tensor): Segments to be sampled from.
            gt_segments (Tensor): Ground truth segments.
            gt_labels (Tensor, optional): Class labels of ground truth
                segments.

        Returns:
            :obj:`SamplingResult`: Sampling result.

        Example:
            >>> from vedatad.misc.segment import RandomSampler
            >>> from vedatad.misc.segment.assigners import AssignResult
            >>> from vedatad.misc.segment.demodata import ensure_rng, random_segments
            >>> rng = ensure_rng(None)
            >>> assign_result = AssignResult.random(rng=rng)
            >>> segments = random_segments(assign_result.num_preds, rng=rng)
            >>> gt_segments = random_segments(assign_result.num_gts, rng=rng)
            >>> gt_labels = None
            >>> self = RandomSampler(num=32, pos_fraction=0.5, neg_pos_ub=-1,
            >>>                      add_gt_as_proposals=False)
            >>> self = self.sample(assign_result, segments, gt_segments, gt_labels) # noqa: E501
        """
        if len(segments.shape) < 2:
            segments = segments[None, :]

        segments = segments[:, :2]

        gt_flags = segments.new_zeros((segments.shape[0], ), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_segments) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            segments = torch.cat([gt_segments, segments], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = segments.new_ones(
                gt_segments.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, segments=segments, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, segments=segments, **kwargs)
        neg_inds = neg_inds.unique()

        sampling_result = SamplingResult(pos_inds, neg_inds, segments,
                                         gt_segments, assign_result, gt_flags)
        return sampling_result
