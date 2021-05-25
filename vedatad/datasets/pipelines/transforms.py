# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import numpy as np
from numpy import random

import vedacore.image as image
from vedacore.misc import registry
from vedatad.misc.segment import segment_overlaps


@registry.register_module('pipeline')
class SpatialRandomFlip(object):
    """Spatially flip images.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction
        if flip_ratio is not None:
            assert 0 <= flip_ratio <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        """Call function to flip images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if np.random.rand() < self.flip_ratio:
            for key in results.get('img_fields', ['imgs']):
                if self.direction == 'horizontal':
                    results[key] = np.flip(results[key], axis=2)
                else:
                    results[key] = np.flip(results[key], axis=1)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'


@registry.register_module('pipeline')
class Pad(object):
    """Pad images.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_imgs(self, results):
        """Pad images according to ``self.size``."""
        for key in results.get('img_fields', ['imgs']):
            if self.size is not None:
                padded_imgs = image.impad(
                    results[key], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                padded_imgs = image.impad_to_multiple(
                    results[key], self.size_divisor, pad_val=self.pad_val)
            results[key] = padded_imgs
        results['pad_tsize'] = padded_imgs.shape[0]

    def __call__(self, results):
        """Call function to pad images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_imgs(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@registry.register_module('pipeline')
class Normalize(object):
    """Normalize images.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert images from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['imgs']):
            results[key] = image.imnormalize(results[key], self.mean, self.std,
                                             self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@registry.register_module('pipeline')
class SpatialRandomCrop(object):
    """Spatially random crop images.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).

    Notes:
        - If the image is smaller than the crop size, return the original image
    """

    def __init__(self, crop_size):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    def __call__(self, results):
        """Call function to randomly crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'imgs_shape' key in result dict
                is updated according to crop size.
        """

        for key in results.get('img_fields', ['imgs']):
            imgs = results[key]
            margin_h = max(imgs.shape[1] - self.crop_size[0], 0)
            margin_w = max(imgs.shape[2] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            # crop images
            imgs = imgs[:, crop_y1:crop_y2, crop_x1:crop_x2, ...]
            results[key] = imgs

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@registry.register_module('pipeline')
class SpatialCenterCrop(object):
    """Spatially center crop images.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).

    Notes:
        - If the image is smaller than the crop size, return the original image
    """

    def __init__(self, crop_size):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    def __call__(self, results):
        """Call function to center crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'imgs_shape' key in result dict
                is updated according to crop size.
        """

        for key in results.get('img_fields', ['imgs']):
            imgs = results[key]
            margin_h = max(imgs.shape[1] - self.crop_size[0], 0)
            margin_w = max(imgs.shape[2] - self.crop_size[1], 0)
            offset_h = int(margin_h / 2)
            offset_w = int(margin_w / 2)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            # crop images
            imgs = imgs[:, crop_y1:crop_y2, crop_x1:crop_x2, ...]
            results[key] = imgs

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@registry.register_module('pipeline')
class PhotoMetricDistortion(object):
    """Apply photometric distortion to images sequentially, every
    transformation is applied with a probability of 0.5. The position of random
    contrast is in second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 p=0.5):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.p = p

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == [
                'imgs'
            ], ('Only single img_fields is allowed')
        imgs = results['imgs']
        assert imgs.dtype == np.float32, (
            'PhotoMetricDistortion needs the input imgs of dtype np.float32'
            ', please set "to_float32=True" in "LoadFrames" pipeline')

        def _filter(img):
            img[img < 0] = 0
            img[img > 255] = 255
            return img

        if random.uniform(0, 1) <= self.p:

            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                       self.brightness_delta)
                imgs += delta
                imgs = _filter(imgs)

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    imgs *= alpha
                    imgs = _filter(imgs)

            # convert color from BGR to HSV
            imgs = np.array([image.bgr2hsv(img) for img in imgs])

            # random saturation
            if random.randint(2):
                imgs[..., 1] *= random.uniform(self.saturation_lower,
                                               self.saturation_upper)

            # random hue
            # if random.randint(2):
            if True:
                imgs[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                imgs[..., 0][imgs[..., 0] > 360] -= 360
                imgs[..., 0][imgs[..., 0] < 0] += 360

            # convert color from HSV to BGR
            imgs = np.array([image.hsv2bgr(img) for img in imgs])
            imgs = _filter(imgs)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    imgs *= alpha
                    imgs = _filter(imgs)

            # randomly swap channels
            if random.randint(2):
                imgs = imgs[..., random.permutation(3)]

            results['imgs'] = imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@registry.register_module('pipeline')
class TemporalRandomCrop(object):
    """Temporally crop.

    Args:
        num_frames (int, optional): The cropped frame num. Default: 768.
        iof_th(float, optional): The minimal iof threshold to crop. Default: 0
    """

    def __init__(self, num_frames=768, iof_th=0):
        self.num_frames = num_frames
        self.iof_th = iof_th
        self.segment2label = dict(
            gt_segments='gt_labels', gt_segments_ignore='gt_labels_ignore')

    def get_valid_mask(self, segments, patch, iof_th):
        gt_iofs = segment_overlaps(segments, patch, mode='iof')[:, 0]
        patch_iofs = segment_overlaps(patch, segments, mode='iof')[0, :]
        iofs = np.maximum(gt_iofs, patch_iofs)
        mask = iofs > iof_th

        return mask

    def __call__(self, results):
        """Call function to random temporally crop video frame.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Temporally cropped results, 'img_ids' is updated in
                result dict.
        """

        total_frames = results['tsize']
        patch_num_frames = min(self.num_frames, total_frames)
        while True:
            start = np.random.randint(0, total_frames - patch_num_frames + 1)
            end = start + patch_num_frames
            patch = np.array([[start, end]], dtype=np.float32)

            mask = self.get_valid_mask(results['gt_segments'], patch,
                                       self.iof_th)
            if np.count_nonzero(mask) == 0:
                continue

            for key in results.get('segment_fields', []):
                segments = results[key]
                mask = self.get_valid_mask(segments, patch, self.iof_th)
                segments = segments[mask]
                segments[:, 0] = segments[:, 0].clip(min=start)
                segments[:, 1] = segments[:, 1].clip(max=end)
                segments -= start
                results[key] = segments

                label_key = self.segment2label[key]
                if label_key in results:
                    results[label_key] = results[label_key][mask]
            results['img_ids'] = results['img_ids'][start:end]
            results['tsize'] = end - start
            results['tshift'] = start

            return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(num_frames={self.num_frames},'
        repr_str += f'iof_th={self.iof_th})'

        return repr_str


@registry.register_module('pipeline')
class Rotate(object):
    """Spatially rotate images.

    Args:
        limit (int, list or tuple): Angle range, (min_angle, max_angle).
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".
            Default: bilinear
        border_mode (str): Border mode, accepted values are "constant",
            "isolated", "reflect", "reflect101", "replicate", "transparent",
            "wrap". Default: constant
        border_value (int): Border value. Default: 0
    """

    def __init__(self,
                 limit,
                 interpolation='bilinear',
                 border_mode='constant',
                 border_value=0,
                 p=0.5):
        if isinstance(limit, int):
            limit = (-limit, limit)
        self.limit = limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.border_value = border_value
        self.p = p

    def __call__(self, results):
        """Call function to random rotate images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Spatially rotated results.
        """

        if random.uniform(0, 1) <= self.p:
            angle = random.uniform(*self.limit)
            for key in results.get('img_fields', ['imgs']):
                imgs = [
                    image.imrotate(
                        img,
                        angle=angle,
                        interpolation=self.interpolation,
                        border_mode=self.border_mode,
                        border_value=self.border_value) for img in results[key]
                ]
                results[key] = np.array(imgs)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(limit={self.limit},'
        repr_str += f'interpolation={self.interpolation},'
        repr_str += f'border_mode={self.border_mode},'
        repr_str += f'border_value={self.border_value},'
        repr_str += f'p={self.p})'

        return repr_str


@registry.register_module('pipeline')
class TemporalCrop(object):
    """Temporally crop."""

    def __init__(self):
        self.segment2label = dict(
            gt_segments='gt_labels', gt_segments_ignore='gt_labels_ignore')

    def __call__(self, results):
        """Call function to temporally crop video frame.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Temporally cropped results, 'img_ids', 'tsize', 'tshift' is
                updated in result dict.
        """

        start, end = results['patch']
        patch = np.array([start, end], dtype=np.float32)
        for key in results.get('segment_fields', []):
            segments = results[key]
            iofs = segment_overlaps(segments, patch[None, :], mode='iof')[:, 0]
            mask = iofs > 0
            segments = segments[mask]
            segments[:, 0] = segments[:, 0].clip(min=start)
            segments[:, 1] = segments[:, 1].clip(max=end)
            segments -= start
            results[key] = segments

            label_key = self.segment2label[key]
            if label_key in results:
                labels = results[label_key]
                labels = labels[mask]
                results[label_key] = labels
        results['img_ids'] = results['img_ids'][start:end]
        results['tsize'] = end - start
        results['tshift'] = start

        return results
