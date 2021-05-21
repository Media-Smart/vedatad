# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import torch


def videos_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_video0, target_video1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


def anchor_inside_flags(flat_anchors, valid_flags, tsize, allowed_border=0):
    """Check whether the anchors are inside the border.

    Args:
        flat_anchors (torch.Tensor): Flatten anchors, shape (n, 2).
        valid_flags (torch.Tensor): An existing valid flags of anchors.
        tsize (int): Temporal size of current video.
        allowed_border (int, optional): The border to allow the valid anchor.
            Defaults to 0.

    Returns:
        torch.Tensor: Flags indicating whether the anchors are inside a
            valid range.
    """
    if allowed_border >= 0:
        inside_flags = (
            valid_flags & (flat_anchors[:, 0] >= -allowed_border) &
            (flat_anchors[:, 1] < tsize + allowed_border))
    else:
        inside_flags = valid_flags
    return inside_flags
