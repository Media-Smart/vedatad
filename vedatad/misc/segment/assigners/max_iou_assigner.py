# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import torch

from vedacore.misc import registry
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@registry.register_module('segment_assigner')
class MaxIoUAssigner(BaseAssigner):
    """Assign a corresponding gt segment or background to each segment.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive segments.
        neg_iou_thr (float or tuple): IoU threshold for negative segments.
        min_pos_iou (float): Minimum iou for a segment to be considered as a
            positive segment. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all segments with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring segments (if
            `gt_segments_ignore` is specified). Negative values mean not
            ignoring any segments.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `segments` and `gt_segments_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonetrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 match_low_quality=True,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(typename='SegmentOverlaps')):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self,
               segments,
               gt_segments,
               gt_segments_ignore=None,
               gt_labels=None):
        """Assign gt to segments.

        This method assign a gt segment to every segment (proposal/anchor),
        each segment will be assigned with -1, or a semi-positive number. -1
        means negative sample, semi-positive number is the index (0-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every segment to the background
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each segment, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that segment
        4. for each gt segment, assign its nearest proposals (may be more than
           one) to itself

        Args:
            segments (Tensor): Segments to be assigned, shape(n, 2).
            gt_segments (Tensor): Groundtruth segments, shape (k, 2).
            gt_segments_ignore (Tensor, optional): Ground truth segments that
                are labelled as `ignored`.
            gt_labels (Tensor, optional): Label of gt_segments, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> segments = torch.Tensor([[0., 10.], [10., 20.]])
            >>> gt_segments = torch.Tensor([[0., 10.]])
            >>> assign_result = self.assign(segments, gt_segments)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_segments.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = segments.device
            segments = segments.cpu()
            gt_segments = gt_segments.cpu()
            if gt_segments_ignore is not None:
                gt_segments_ignore = gt_segments_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        overlaps = self.iou_calculator(gt_segments, segments)

        if (self.ignore_iof_thr > 0 and gt_segments_ignore is not None
                and gt_segments_ignore.numel() > 0 and segments.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    segments, gt_segments_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_segments_ignore, segments, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of segments with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_segments and n segments,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_segments, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_segments = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_segments, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_segments == 0:
            # No ground truth or segments, return empty assignment
            max_overlaps = overlaps.new_zeros((num_segments, ))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_segments, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        # the negative inds are set to be 0
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        if self.match_low_quality:
            # Low-quality matching will overwirte the assigned_gt_inds assigned
            # in Step 3. Thus, the assigned gt might not be the best one for
            # prediction.
            # For example, if segment A has 0.9 and 0.8 iou with GT segment
            # 1 & 2, segment 1 will be assigned as the best target for segment
            # A in step 3.
            # However, if GT segment 2's gt_argmax_overlaps = A, segment A's
            # assigned_gt_inds will be overwritten to be segment B.
            # This might be the reason that it is not used in ROI Heads.
            for i in range(num_gts):
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    if self.gt_max_assign_all:
                        max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_segments, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
