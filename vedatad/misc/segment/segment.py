# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import numpy as np
import torch

from vedatad.ops import batched_nms


def segment_overlaps(segments1,
                     segments2,
                     mode='iou',
                     is_aligned=False,
                     eps=1e-6):
    """Calculate overlap between two set of segments.
    If ``is_aligned`` is ``False``, then calculate the ious between each
    segment of segments1 and segments2, otherwise the ious between each aligned
     pair of segments1 and segments2.
    Args:
        segments1 (Tensor): shape (m, 2) in <t1, t2> format or empty.
        segments2 (Tensor): shape (n, 2) in <t1, t2> format or empty.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).
    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    Example:
        >>> segments1 = torch.FloatTensor([
        >>>     [0, 10],
        >>>     [10, 20],
        >>>     [32, 38],
        >>> ])
        >>> segments2 = torch.FloatTensor([
        >>>     [0, 20],
        >>>     [0, 19],
        >>>     [10, 20],
        >>> ])
        >>> segment_overlaps(segments1, segments2)
        tensor([[0.5000, 0.5263, 0.0000],
                [0.0000, 0.4500, 1.0000],
                [0.0000, 0.0000, 0.0000]])
    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 9],
        >>> ])
        >>> assert tuple(segment_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(segment_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(segment_overlaps(empty, empty).shape) == (0, 0)
    """

    is_numpy = False
    if isinstance(segments1, np.ndarray):
        segments1 = torch.from_numpy(segments1)
        is_numpy = True
    if isinstance(segments2, np.ndarray):
        segments2 = torch.from_numpy(segments2)
        is_numpy = True

    assert mode in ['iou', 'iof']
    # Either the segments are empty or the length of segments's last dimenstion
    # is 2
    assert (segments1.size(-1) == 2 or segments1.size(0) == 0)
    assert (segments2.size(-1) == 2 or segments2.size(0) == 0)

    rows = segments1.size(0)
    cols = segments2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return segments1.new(rows, 1) if is_aligned else segments2.new(
            rows, cols)

    if is_aligned:
        start = torch.max(segments1[:, 0], segments2[:, 0])  # [rows]
        end = torch.min(segments1[:, 1], segments2[:, 1])  # [rows]

        overlap = (end - start).clamp(min=0)  # [rows, 2]
        area1 = segments1[:, 1] - segments1[:, 0]

        if mode == 'iou':
            area2 = segments2[:, 1] - segments2[:, 0]
            union = area1 + area2 - overlap
        else:
            union = area1
    else:
        start = torch.max(segments1[:, None, 0], segments2[:,
                                                           0])  # [rows, cols]
        end = torch.min(segments1[:, None, 1], segments2[:, 1])  # [rows, cols]

        overlap = (end - start).clamp(min=0)  # [rows, cols]
        area1 = segments1[:, 1] - segments1[:, 0]

        if mode == 'iou':
            area2 = segments2[:, 1] - segments2[:, 0]
            union = area1[:, None] + area2 - overlap
        else:
            union = area1[:, None]

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union

    if is_numpy:
        ious = ious.numpy()

    return ious


def multiclass_nms(multi_segments,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):

    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.get('typename', 'nms')
    if nms_type.startswith('lb_'):
        nms_cfg_['typename'] = nms_type[3:]
        return _lb_multiclass_nms(multi_segments, multi_scores, score_thr,
                                  nms_cfg_, max_num, score_factors)
    else:
        return _multiclass_nms(multi_segments, multi_scores, score_thr,
                               nms_cfg_, max_num, score_factors)


def _multiclass_nms(multi_segments,
                    multi_scores,
                    score_thr,
                    nms_cfg,
                    max_num=-1,
                    score_factors=None):
    """NMS for multi-class segments.

    Args:
        multi_segments (Tensor): shape (n, #class*2) or (n, 2)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): segment threshold, segments with scores lower than
            it will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num segments after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (segments, labels), tensors of shape (k, 3) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_segments.shape[1] > 2:
        segments = multi_segments.view(multi_scores.size(0), -1, 2)
    else:
        segments = multi_segments[:, None].expand(-1, num_classes, 2)
    scores = multi_scores[:, :-1]

    # filter out segments with low scores
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    valid_mask = scores > score_thr
    segments = segments[valid_mask]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if segments.numel() == 0:
        segments = multi_segments.new_zeros((0, 3))
        labels = multi_segments.new_zeros((0, ), dtype=torch.long)
        return segments, labels

    dets, keep = batched_nms(segments, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]


def _lb_multiclass_nms(multi_segments,
                       multi_scores,
                       score_thr,
                       nms_cfg,
                       max_num=-1,
                       score_factors=None):
    """NMS for multi-class segments.

    Args:
        multi_segments (Tensor): shape (n, #class*2) or (n, 2)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): segment threshold, segments with scores lower than
            it will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num segments after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (segments, labels), tensors of shape (k, 3) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_segments.shape[1] > 2:
        segments = multi_segments.view(multi_scores.size(0), -1, 2)
    else:
        segments = multi_segments[:, None].expand(-1, num_classes, 2)
    scores = multi_scores[:, :-1]

    # filter out segments with low scores
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    valid_mask = scores > score_thr
    segments = segments[valid_mask]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if segments.numel() == 0:
        segments = multi_segments.new_zeros((0, 3))
        labels = multi_segments.new_zeros((0, ), dtype=torch.long)
        return segments, labels

    inds = scores.argsort(descending=True)
    segments = segments[inds]
    scores = scores[inds]
    labels = labels[inds]

    batch_segments = torch.empty((0, 2),
                                 dtype=segments.dtype,
                                 device=segments.device)
    batch_scores = torch.empty((0, ), dtype=scores.dtype, device=scores.device)
    batch_labels = torch.empty((0, ), dtype=labels.dtype, device=labels.device)
    while segments.shape[0] > 0:
        num = min(10000, segments.shape[0])
        batch_segments = torch.cat([batch_segments, segments[:num]])
        batch_scores = torch.cat([batch_scores, scores[:num]])
        batch_labels = torch.cat([batch_labels, labels[:num]])
        segments = segments[num:]
        scores = scores[num:]
        labels = labels[num:]

        _, keep = batched_nms(batch_segments, batch_scores, batch_labels,
                              nms_cfg)
        batch_segments = batch_segments[keep]
        batch_scores = batch_scores[keep]
        batch_labels = batch_labels[keep]

        if max_num > 0 and batch_segments.shape[0] > max_num:
            break

    dets = torch.cat([batch_segments, batch_scores[:, None]], dim=-1)
    labels = batch_labels

    if max_num > 0:
        dets = dets[:max_num]
        labels = labels[:max_num]

    return dets, labels


def distance2segment(points, distance, max_t=None):
    """Decode distance prediction to segment.
    Args:
        points (Tensor): Shape (n,), [center].
        distance (Tensor): Distance from the given point to 2
            boundaries (left, right).
        max_shape (tuple): Shape of the image.
    Returns:
        Tensor: Decoded segments.
    """
    start = points[:, 0] - distance[:, 0]
    end = points[:, 0] + distance[:, 1]
    if max_t is not None:
        start = start.clamp(min=0, max=max_t)
        end = end.clamp(min=0, max=max_t)
    return torch.stack([start, end], -1)


def segment2distance(points, segment, max_dis=None, eps=0.1):
    """Encode segment based on distances.
    Args:
        points (Tensor): Shape (n,), [center].
        segment (Tensor): Shape (n, 2), "start, end" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=
    Returns:
        Tensor: Decoded distances.
    """
    left = points[:, 0] - segment[:, 0]
    right = segment[:, 1] - points[:, 0]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, right], -1)


def segment2roi(segment_list):
    """Convert a list of segments to roi format.

    Args:
        segment_list (list[Tensor]): a list of segments corresponding to a
            batch of videos.

    Returns:
        Tensor: shape (n, 3), [batch_ind, start, end]
    """
    rois_list = []
    for video_id, segments in enumerate(segment_list):
        if segments.size(0) > 0:
            video_inds = segments.new_full((segments.size(0), 1), video_id)
            rois = torch.cat([video_inds, segments[:, :2]], dim=-1)
        else:
            rois = segments.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def segment2result(segments, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        segments (Tensor): shape (n, 3)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): segment results of each class
    """
    if segments.shape[0] == 0:
        return [np.zeros((0, 3), dtype=np.float32) for _ in range(num_classes)]
    else:
        segments = segments.cpu().numpy()
        labels = labels.cpu().numpy()
        return [segments[labels == i, :] for i in range(num_classes)]
