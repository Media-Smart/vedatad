# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import torch.nn as nn

from vedacore.misc import multi_apply, registry
from vedacore.modules import normal_init
from .base_dense_head import BaseDenseHead


@registry.register_module('head')
class AnchorHead(BaseDenseHead):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 num_anchors,
                 in_channels,
                 feat_channels=256,
                 use_sigmoid=True):
        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        if use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.conv_cls = nn.Conv1d(self.in_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv1d(self.in_channels, self.num_anchors * 2, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                segment_pred (Tensor): Segment energies / deltas for a single
                    scale level, the channels number is num_anchors * 2.
        """
        cls_score = self.conv_cls(x)
        segment_pred = self.conv_reg(x)
        return cls_score, segment_pred

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 3D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and segment
                prediction.
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 3D-tensor, the channels number is
                    num_anchors * num_classes.
                segment_preds (list[Tensor]): Segment energies / deltas for all
                    scale levels, each is a 3D-tensor, the channels number is
                    num_anchors * 2.
        """
        return multi_apply(self.forward_single, feats)
