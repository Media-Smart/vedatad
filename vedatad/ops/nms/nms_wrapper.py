# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import numpy as np
import torch

from . import nms_ext


def nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): segments with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept segments and indice, which is always the same data type as
            the input.

    Example:
        >>> dets = np.array([[49.1, 51.0, 0.95],
        >>>                  [49.3, 51.0, 0.9],
        >>>                  [49.2, 51.0, 0.55],
        >>>                  [35.1, 39.1, 0.53],
        >>>                  [35.6, 39.3, 0.5],
        >>>                  [35.3, 39.9, 0.4],
        >>>                  [35.2, 39.7, 0.3]], dtype=np.float32)
        >>> iou_thr = 0.6
        >>> suppressed, inds = nms(dets, iou_thr)
        >>> assert len(inds) == len(suppressed) == 3
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else f'cuda:{device_id}'
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError('dets must be either a Tensor or numpy array, '
                        f'but got {type(dets)}')

    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        if dets_th.is_cuda:
            inds = nms_ext.nms(dets_th, iou_thr)
        else:
            inds = nms_ext.nms(dets_th, iou_thr)

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


def soft_nms(dets, iou_thr, method='linear', sigma=0.5, min_score=1e-3):
    """Dispatch to only CPU Soft NMS implementations.

    The input can be either a torch tensor or numpy array.
    The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): segments with scores.
        iou_thr (float): IoU threshold for Soft NMS.
        method (str): either 'linear' or 'gaussian'
        sigma (float): hyperparameter for gaussian method
        min_score (float): score filter threshold

    Returns:
        tuple: new det segments and indice, which is always the same
        data type as the input.

    Example:
        >>> dets = np.array([[4., 5., 0.95],
        >>>                  [4., 5., 0.9],
        >>>                  [3., 4., 0.55],
        >>>                  [3., 4., 0.5],
        >>>                  [3., 4., 0.4],
        >>>                  [3., 4., 0.0]], dtype=np.float32)
        >>> iou_thr = 0.6
        >>> new_dets, inds = soft_nms(dets, iou_thr, sigma=0.5)
        >>> assert len(inds) == len(new_dets) == 2
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_tensor = True
        dets_t = dets.detach().cpu()
    elif isinstance(dets, np.ndarray):
        is_tensor = False
        dets_t = torch.from_numpy(dets)
    else:
        raise TypeError('dets must be either a Tensor or numpy array, '
                        f'but got {type(dets)}')

    method_codes = {'linear': 1, 'gaussian': 2}
    if method not in method_codes:
        raise ValueError(f'Invalid method for SoftNMS: {method}')
    results = nms_ext.soft_nms(dets_t, iou_thr, method_codes[method], sigma,
                               min_score)

    new_dets = results[:, :3]
    inds = results[:, 3]

    if is_tensor:
        return new_dets.to(
            device=dets.device, dtype=dets.dtype), inds.to(
                device=dets.device, dtype=torch.long)
    else:
        return new_dets.numpy().astype(dets.dtype), inds.numpy().astype(
            np.int64)


def batched_nms(segments, scores, inds, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.

    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the segments. The offset is dependent only on the class idx, and is large
    enough so that segments from different classes do not overlap.

    Arguments:
        segments (torch.Tensor): segments in shape (N, 2).
        scores (torch.Tensor): scores in shape (N, ).
        inds (torch.Tensor): each index value correspond to a segment cluster,
            and NMS will not be applied between elements of different inds,
            shape (N, ).
        nms_cfg (dict): specify nms type and class_agnostic as well as other
            parameters like iou_thr.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all segments,
            regardless of the predicted class

    Returns:
        tuple: kept segments and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        segments_for_nms = segments
    else:
        max_coordinate = segments.max()
        offsets = inds.to(segments) * (max_coordinate + 1)
        segments_for_nms = segments + offsets[:, None]
    nms_type = nms_cfg_.pop('typename', 'nms')
    nms_op = eval(nms_type)
    dets, keep = nms_op(
        torch.cat([segments_for_nms, scores[:, None]], -1), **nms_cfg_)
    segments = segments[keep]
    scores = dets[:, -1]
    return torch.cat([segments, scores[:, None]], -1), keep


def nms_match(dets, thresh):
    """Matched dets into different groups by NMS.

    NMS match is Similar to NMS but when a segment is suppressed, nms match
    will record the indice of supporessed segment and form a group with the
    indice of kept segment. In each group, indice is sorted as score order.

    Arguments:
        dets (torch.Tensor | np.ndarray): Det segments with scores, shape
            (N, 3).
        iou_thr (float): IoU thresh for NMS.

    Returns:
        List[Tensor | ndarray]: The outer list corresponds different matched
            group, the inner Tensor corresponds the indices for a group in
            score order.
    """
    if dets.shape[0] == 0:
        matched = []
    else:
        assert dets.shape[-1] == 3, 'inputs dets.shape should be (N, 3), ' \
                                    f'but get {dets.shape}'
        if isinstance(dets, torch.Tensor):
            dets_t = dets.detach().cpu()
        else:
            dets_t = torch.from_numpy(dets)
        matched = nms_ext.nms_match(dets_t, thresh)

    if isinstance(dets, torch.Tensor):
        return [dets.new_tensor(m, dtype=torch.long) for m in matched]
    else:
        return [np.array(m, dtype=np.int) for m in matched]
