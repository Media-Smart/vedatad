import torch

from vedacore.misc import multi_apply, registry, unmap
from vedatad.bridge import build_meshgrid
from vedatad.misc.anchor import anchor_inside_flags, videos_to_levels
from vedatad.misc.segment import (build_assigner, build_sampler,
                                  build_segment_coder)
from .base_criterion import BaseCriterion
from .losses import build_loss


@registry.register_module('criterion')
class SegmentAnchorCriterion(BaseCriterion):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        segment_coder (dict): Config of segment coder.
        reg_decoded_segment (bool): If true, the regression loss would be
            applied on decoded segment. Default: False
        background_label (int | None): Label ID of background, set as 0 for
            RPN and num_classes for other heads. It will automatically set as
            num_classes if None is given.
        loss_cls (dict): Config of classification loss.
        loss_seg (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 meshgrid,
                 segment_coder=dict(
                     typename='DeltaSegmentCoder',
                     target_means=(.0, .0),
                     target_stds=(1.0, 1.0)),
                 reg_decoded_segment=False,
                 loss_cls=dict(
                     typename='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_segment=dict(
                     typename='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 train_cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        # TODO better way to determine whether sample or not
        self.sampling = loss_cls['typename'] not in [
            'FocalLoss', 'GHMC', 'QualityFocalLoss'
        ]
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self.reg_decoded_segment = reg_decoded_segment
        self.background_label = num_classes
        self.segment_coder = build_segment_coder(segment_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_segment = build_loss(loss_segment)
        self.train_cfg = train_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(typename='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg)

        self.meshgrid = build_meshgrid(meshgrid)
        # usually the numbers of anchors for each level are the same
        # except SSD detectors
        # self.num_anchors = self.anchor_generator.num_base_anchors[0]

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            video_meta,
                            gt_segments,
                            gt_labels,
                            gt_segments_ignore,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single video.

        Adapted from https://github.com/open-mmlab/mmdetection

        Args:
            flat_anchors (Tensor): Multi-level anchors of the video, which are
                concatenated into a single tensor of shape (num_anchors, 2)
            valid_flags (Tensor): Multi level valid flags of the video, which
                are concatenated into a single tensor of shape (num_anchors,).
            video_meta (dict): Meta info of the video.
            gt_segments (Tensor): Ground truth segments of the video,
                shape (num_gts, 2).
            gt_labels (Tensor): Ground truth labels of each segment,
                shape (num_gts,).
            gt_segments_ignore (Tensor): Ground truth segments to be
                ignored, shape (num_ignored_gts, 2).
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                segment_targets_list (list[Tensor]): Segment targets of each
                    level
                segment_weights_list (list[Tensor]): Segment weights of each
                    level
                num_total_pos (int): Number of positive samples in all videos
                num_total_neg (int): Number of negative samples in all videos
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           video_meta['tsize'],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.assigner.assign(
            anchors, gt_segments, gt_segments_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_segments)

        num_valid_anchors = anchors.shape[0]
        segment_targets = torch.zeros_like(anchors)
        segment_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.background_label,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_segment:
                pos_segment_targets = self.segment_coder.encode(
                    sampling_result.pos_segments,
                    sampling_result.pos_gt_segments)
            else:
                pos_segment_targets = sampling_result.pos_gt_segments
            segment_targets[pos_inds, :] = pos_segment_targets
            segment_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # only rpn gives gt_labels as None, this time FG is 1
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels,
                num_total_anchors,
                inside_flags,
                fill=self.background_label)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            segment_targets = unmap(segment_targets, num_total_anchors,
                                    inside_flags)
            segment_weights = unmap(segment_weights, num_total_anchors,
                                    inside_flags)

        return (labels, label_weights, segment_targets, segment_weights,
                pos_inds, neg_inds, sampling_result)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    video_metas,
                    gt_segments_list,
                    gt_labels_list,
                    gt_segments_ignore_list=None,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple videos.

        Adapted from https://github.com/open-mmlab/mmdetection

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                video. The outer list indicates videos, and the inner list
                corresponds to feature levels of the video. Each element of
                the inner list is a tensor of shape (num_anchors, 2).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each video. The outer list indicates videos, and the inner list
                corresponds to feature levels of the video. Each element of
                the inner list is a tensor of shape (num_anchors, )
            video_metas (list[dict]): Meta info of each video.
            gt_segments_list (list[Tensor]): Ground truth segments of each
                image.
            gt_labels_list (list[Tensor]): Ground truth labels of each segment.
            gt_segments_ignore_list (None | list[Tensor]): Ground truth
                segments to be ignored.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                segment_targets_list (list[Tensor]): Segment targets of each
                    level
                segment_weights_list (list[Tensor]): Segment weights of each
                    level.
                num_total_pos (int): Number of positive samples in all videos
                num_total_neg (int): Number of negative samples in all videos
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having T dimension).
                The results will be concatenated after the end
        """
        num_videos = len(video_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_videos

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_videos):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_segments_ignore_list is None:
            gt_segments_ignore_list = [None for _ in range(num_videos)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_videos)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            video_metas,
            gt_segments_list,
            gt_labels_list,
            gt_segments_ignore_list,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_segment_targets,
         all_segment_weights, pos_inds_list, neg_inds_list,
         sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = videos_to_levels(all_labels, num_level_anchors)
        label_weights_list = videos_to_levels(all_label_weights,
                                              num_level_anchors)
        segment_targets_list = videos_to_levels(all_segment_targets,
                                                num_level_anchors)
        segment_weights_list = videos_to_levels(all_segment_weights,
                                                num_level_anchors)
        res = (labels_list, label_weights_list, segment_targets_list,
               segment_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = videos_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    def loss_single(self, cls_score, segment_pred, anchors, labels,
                    label_weights, segment_targets, segment_weights,
                    num_total_samples):
        """Compute loss of a single scale level.

        Adapted from https://github.com/open-mmlab/mmdetection

        Args:
            cls_score (Tensor): Segment scores for each scale level
                Has shape (N, num_anchors * num_classes, T).
            segment_pred (Tensor): Segment energies / deltas for each scale
                level with shape (N, num_anchors * 2, T).
            anchors (Tensor): Segment reference for each scale level with shape
                (N, num_total_anchors, 2).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            segment_targets (Tensor): Segment regression targets of each anchor
                with shape (N, num_total_anchors, 2).
            segment_weights (Tensor): Segment regression loss weights of each
                anchor with shape (N, num_total_anchors, 2).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 1).reshape(-1,
                                                       self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        segment_targets = segment_targets.reshape(-1, 2)
        segment_weights = segment_weights.reshape(-1, 2)
        segment_pred = segment_pred.permute(0, 2, 1).reshape(-1, 2)
        if self.reg_decoded_segment:
            anchors = anchors.reshape(-1, 2)
            segment_pred = self.segment_coder.decode(anchors, segment_pred)
        loss_segment = self.loss_segment(
            segment_pred,
            segment_targets,
            segment_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_segment

    def loss(self,
             feats,
             video_metas,
             gt_segments,
             gt_labels,
             gt_segments_ignore=None):
        """Compute losses of the head.

        Adapted from https://github.com/open-mmlab/mmdetection

        Args:
            feats (list[list[Tensor], list[Tensor]]): Features containing cls
                and seg features.
                cls_scores (list[Tensor]): Segment scores for each scale level
                    Has shape (N, num_anchors * num_classes, T)
                seg_preds (list[Tensor]): Segment energies / deltas for each
                    scale level with shape (N, num_anchors * 2, T)
            video_metas (list[dict]): Meta information of each video, e.g.,
                video shape, etc.
            gt_segments (list[Tensor]): Ground truth segments for each video
                with shape (num_gts, 2) in [start, end] format.
            gt_labels (list[Tensor]): class indices corresponding to each
                segment
            gt_segments_ignore (None | list[Tensor]): specify which segments
                can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        cls_scores, segment_preds = feats
        featmap_tsizes = [featmap.size()[2] for featmap in cls_scores]

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.meshgrid.gen_anchor_mesh(
            featmap_tsizes, video_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            video_metas,
            gt_segments,
            gt_labels,
            gt_segments_ignore_list=gt_segments_ignore)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, segment_targets_list,
         segment_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = videos_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_segment = multi_apply(
            self.loss_single,
            cls_scores,
            segment_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            segment_targets_list,
            segment_weights_list,
            num_total_samples=num_total_samples)
        loss, log_vars = self._parse_losses(
            dict(loss_cls=losses_cls, loss_segment=losses_segment))
        return dict(loss=loss, log_vars=log_vars)
