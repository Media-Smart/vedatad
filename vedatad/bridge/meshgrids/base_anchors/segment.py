# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import numpy as np
import torch

from vedacore.misc import registry


@registry.register_module('base_anchor')
class SegmentBaseAnchor:

    def __init__(self,
                 base_sizes,
                 scales=None,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 centers=None,
                 center_offset=0.):
        """Generate base anchors.

        list(torch.Tensor): Base anchors of a feature grid in multiple
            feature levels.
        """
        if center_offset != 0:
            assert centers is None, (
                f'center cannot be set when center_offset '
                f'!=0, {centers} is given.')
        if not (0 <= center_offset <= 1):
            raise ValueError('center_offset should be in range [0, 1], '
                             f'{center_offset} is given.')
        if centers is not None:
            assert len(centers) == len(base_sizes), (
                f'The number of base_sizes should be the same as centers, '
                f'got {base_sizes} and {centers}')

        # calculate scales of anchors
        assert (
            (octave_base_scale is not None and scales_per_octave is not None) ^
            (scales is not None)
        ), ('scales and octave_base_scale with scales_per_octave cannot be '
            'set at the same time')
        if scales is not None:
            self.scales = torch.Tensor(scales)
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = np.array(
                [2**(i / scales_per_octave) for i in range(scales_per_octave)])
            scales = octave_scales * octave_base_scale
            self.scales = torch.Tensor(scales)
        else:
            raise ValueError('Either scales or octave_base_scale with '
                             'scales_per_octave should be set')

        self.base_sizes = base_sizes
        self.centers = centers
        self.center_offset = center_offset

    def generate(self):
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self._single_level_base_anchors(base_size, self.scales,
                                                self.center_offset, center))
        return multi_level_base_anchors

    @staticmethod
    def _single_level_base_anchors(base_size,
                                   scales,
                                   center_offset,
                                   center=None):
        """Generate base anchors of a single level.

        Args:
            base_size (int | float): Basic temporal size of an anchor.
            scales (torch.Tensor): Scales of the anchor.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature maps
        """
        if center is None:
            center = center_offset * (base_size - 1)

        intervals = base_size * scales

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [center - 0.5 * intervals, center + 0.5 * intervals]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors
