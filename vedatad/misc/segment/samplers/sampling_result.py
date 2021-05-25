# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import torch

from ..util_mixins import NiceRepr


class SamplingResult(NiceRepr):
    """Segment sampling result.

    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> self = SamplingResult.random(rng=10)
        >>> print(f'self = {self}')
        self = <SamplingResult({
            'neg_segments': torch.Size([12, 2]),
            'neg_inds': tensor([ 0,  1,  2,  4,  5,  6,  7,  8,  9, 10, 11, 12]), # noqa: E501
            'num_gts': 4,
            'pos_assigned_gt_inds': tensor([], dtype=torch.int64),
            'pos_segments': torch.Size([0, 2]),
            'pos_inds': tensor([], dtype=torch.int64),
            'pos_is_gt': tensor([], dtype=torch.uint8)
        })>
    """

    def __init__(self, pos_inds, neg_inds, segments, gt_segments,
                 assign_result, gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_segments = segments[pos_inds]
        self.neg_segments = segments[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_segments.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_segments.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_segments = torch.empty_like(gt_segments).view(-1, 2)
        else:
            if len(gt_segments.shape) < 2:
                gt_segments = gt_segments.view(-1, 2)

            self.pos_gt_segments = gt_segments[self.pos_assigned_gt_inds, :]

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def segments(self):
        """torch.Tensor: concatenated positive and negative segments"""
        return torch.cat([self.pos_segments, self.neg_segments])

    def to(self, device):
        """Change the device of the data inplace.

        Example:
            >>> self = SamplingResult.random()
            >>> print(f'self = {self.to(None)}')
            >>> # xdoctest: +REQUIRES(--gpu)
            >>> print(f'self = {self.to(0)}')
        """
        _dict = self.__dict__
        for key, value in _dict.items():
            if isinstance(value, torch.Tensor):
                _dict[key] = value.to(device)
        return self

    def __nice__(self):
        data = self.info.copy()
        data['pos_segments'] = data.pop('pos_segments').shape
        data['neg_segments'] = data.pop('neg_segments').shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = '    ' + ',\n    '.join(parts)
        return '{\n' + body + '\n}'

    @property
    def info(self):
        """Returns a dictionary of info about the object."""
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_segments': self.pos_segments,
            'neg_segments': self.neg_segments,
            'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
        }

    @classmethod
    def random(cls, rng=None, **kwargs):
        """
        Args:
            rng (None | int | numpy.random.RandomState): seed or state.
            kwargs (keyword arguments):
                - num_preds: number of predicted segments
                - num_gts: number of true segments
                - p_ignore (float): probability of a predicted segment assinged
                    to an ignored truth.
                - p_assigned (float): probability of a predicted segment not
                    being assigned.
                - p_use_label (float | bool): with labels or not.

        Returns:
            :obj:`SamplingResult`: Randomly generated sampling result.

        Example:
            >>> self = SamplingResult.random()
            >>> print(self.__dict__)
        """
        from .. import demodata
        from ..assigners.assign_result import AssignResult
        from .random_sampler import RandomSampler
        rng = demodata.ensure_rng(rng)

        # make probabalistic?
        num = 32
        pos_fraction = 0.5
        neg_pos_ub = -1

        assign_result = AssignResult.random(rng=rng, **kwargs)

        # Note we could just compute an assignment
        segments = demodata.random_segments(assign_result.num_preds, rng=rng)
        gt_segments = demodata.random_segments(assign_result.num_gts, rng=rng)

        if rng.rand() > 0.2:
            # sometimes algorithms squeeze their data, be robust to that
            gt_segments = gt_segments.squeeze()
            segments = segments.squeeze()

        if assign_result.labels is None:
            gt_labels = None
        else:
            gt_labels = None  # todo

        if gt_labels is None:
            add_gt_as_proposals = False
        else:
            add_gt_as_proposals = True  # make probabalistic?

        sampler = RandomSampler(
            num,
            pos_fraction,
            neg_pos_ub=neg_pos_ub,
            add_gt_as_proposals=add_gt_as_proposals,
            rng=rng)
        self = sampler.sample(assign_result, segments, gt_segments, gt_labels)
        return self
