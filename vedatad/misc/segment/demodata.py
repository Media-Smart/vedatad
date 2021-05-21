# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import numpy as np
import torch


def ensure_rng(rng=None):
    """Simple version of the ``kwarray.ensure_rng``

    Args:
        rng (int | numpy.random.RandomState | None):
            if None, then defaults to the global rng. Otherwise this can be an
            integer or a RandomState class
    Returns:
        (numpy.random.RandomState) : rng -
            a numpy random number generator

    References:
        https://gitlab.kitware.com/computer-vision/kwarray/blob/master/kwarray/util_random.py#L270
    """

    if rng is None:
        rng = np.random.mtrand._rand
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)
    else:
        rng = rng
    return rng


def random_segments(num=1, scale=1, rng=None):
    """
    Returns:
        Tensor: shape (n, 2) in start, end format.

    References:
        https://gitlab.kitware.com/computer-vision/kwimage/blob/master/kwimage/structs/boxes.py#L1390

    Example:
        >>> num = 3
        >>> scale = 512
        >>> rng = 0
        >>> segments = random_segments(num, scale, rng)
        >>> print(segments)
        tensor([[280.9925, 366.1769],
                [278.9802, 308.6148],
                [216.9113, 330.6978]])
    """
    rng = ensure_rng(rng)

    segments = rng.rand(num, 2).astype(np.float32)

    start = np.minimum(segments[:, 0], segments[:, 1])
    end = np.maximum(segments[:, 0], segments[:, 1])

    segments[:, 0] = start * scale
    segments[:, 1] = end * scale

    segments = torch.from_numpy(segments)
    return segments
