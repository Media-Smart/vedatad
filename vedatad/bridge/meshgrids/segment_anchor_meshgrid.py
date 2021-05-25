# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import numpy as np
import torch

from vedacore.misc import registry
from .base_anchors import build_base_anchor
from .base_meshgrid import BaseMeshGrid


@registry.register_module('meshgrid')
class SegmentAnchorMeshGrid(BaseMeshGrid):

    def __init__(self, strides, base_anchor):
        super().__init__(strides)
        self.base_anchors = build_base_anchor(base_anchor).generate()

    def gen_anchor_mesh(self,
                        featmap_tsizes,
                        video_metas,
                        dtype=torch.float,
                        device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_tsizes (list[int]): Multi-level feature map temporal sizes.
            video_metas (list[dict]): Video meta info.
            device (torch.device | str): Device for returned tensors
        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each video.
                valid_flag_list (list[Tensor]): Valid flags of each video.
        """
        num_videos = len(video_metas)

        # since feature map temporal sizes of all videos are the same, we only
        # compute anchors for one time
        multi_level_anchors = self._gen_anchor_mesh(featmap_tsizes, dtype,
                                                    device)
        anchor_list = [multi_level_anchors for _ in range(num_videos)]

        # for each video, we compute valid flags of multi level anchors
        valid_flag_list = []
        for video_id, video_meta in enumerate(video_metas):
            multi_level_flags = self.valid_flags(featmap_tsizes,
                                                 video_meta['pad_tsize'],
                                                 device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def _gen_anchor_mesh(self, featmap_tsizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_tsizes (list[int]): Multi-level feature map temporal sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.
        Returns:
            tuple: points of each image.
        """
        assert self.num_levels == len(featmap_tsizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self._single_level_anchor_mesh(
                self.base_anchors[i].to(device).to(dtype),
                featmap_tsizes[i],
                self.strides[i],
                device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def _single_level_anchor_mesh(self, base_anchors, featmap_tsize, stride,
                                  device):
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_anchors``.

        Args:
            base_anchors (torch.Tensor): The base anchors of a feature grid.
            featmap_tsize (int): Temporal size of the feature maps.
            stride (int, optional): Stride of the feature map.
                Defaults to .
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """
        shifts = torch.arange(0, featmap_tsize, device=device) * stride
        shifts = shifts.type_as(base_anchors)
        # add A anchors (1, A, 2) to K shifts (K, 1, 1) to get
        # shifted anchors (K, A, 2), reshape to (K*A, 2)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, None]
        all_anchors = all_anchors.view(-1, 2)
        # first A rows correspond to A anchors of 0 in feature map,
        # then 1, 2, ...
        return all_anchors

    def valid_flags(self, featmap_tsizes, pad_tsize, device='cuda'):
        """Generate valid flags of anchors in multiple feature levels.

        Args:
            featmap_tsizes (list(tuple)): List of feature map temporal sizes in
                multiple feature levels.
            pad_tsize (int): The padded temporal size of the video.
            device (str): Device where the anchors will be put on.

        Return:
            list(torch.Tensor): Valid flags of anchors in multiple levels.
        """
        assert self.num_levels == len(featmap_tsizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_tsize = featmap_tsizes[i]
            valid_feat_tsize = min(
                int(np.ceil(pad_tsize / anchor_stride)), feat_tsize)
            flags = self._single_level_valid_flags(
                feat_tsize,
                valid_feat_tsize,
                self.num_base_anchors[i],
                device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def _single_level_valid_flags(self,
                                  featmap_tsize,
                                  valid_tsize,
                                  num_base_anchors,
                                  device='cuda'):
        """Generate the valid flags of anchor in a single feature map.

        Args:
            featmap_tsize (int): The temporal size of feature maps.
            valid_tsize (int): The valid temporal size of the feature
                maps.
            num_base_anchors (int): The number of base anchors.
            device (str, optional): Device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each anchor in a single level
                feature map.
        """
        assert valid_tsize <= featmap_tsize
        valid = torch.zeros(featmap_tsize, dtype=torch.bool, device=device)
        valid[:valid_tsize] = 1
        valid = valid[:, None].expand(valid.size(0),
                                      num_base_anchors).contiguous().view(-1)
        return valid

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return [base_anchors.size(0) for base_anchors in self.base_anchors]
