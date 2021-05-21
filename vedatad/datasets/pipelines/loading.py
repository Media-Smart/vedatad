# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
import numpy as np
import os.path as osp
import glob

import vedacore.fileio as fileio
import vedacore.image as image
from vedacore.misc import registry


@registry.register_module('pipeline')
class LoadMetaInfo(object):

    def __call__(self, results):
        results['img_ids'] = list(range(results['video_info']['frames']))
        results['ori_tsize'] = results['video_info']['frames']
        results['tsize'] = results['video_info']['frames']
        results['fps'] = results['video_info']['fps']
        results['duration'] = results['video_info']['duration']

        return results


@registry.register_module('pipeline')
class LoadFrames(object):
    """Load video frames.

    Required keys are "video_prefix" and "video_info" (a dict that must contain
    the key "video_name"). Added or updated keys are "video_name", "imgs",
    "imgs_shape", "ori_shape" (same as `imgs_shape`).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`image.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`fileio.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load frames and get frames meta information.

        Args:
            results (dict): Result dict from :obj:`.CustomDataset`.

        Returns:
            dict: The dict contains loaded frames and meta information.
        """

        if self.file_client is None:
            self.file_client = fileio.FileClient(**self.file_client_args)

        if results['video_prefix'] is not None:
            video_name = osp.join(results['video_prefix'],
                                  results['video_info']['video_name'])
        else:
            video_name = results['video_info']['video_name']

        imgfiles = sorted(glob.glob(osp.join(video_name, '*')))
        assert len(imgfiles) == results['video_info']['frames']

        imgs = []
        for img_id in results['img_ids']:
            filename = imgfiles[img_id]
            img_bytes = self.file_client.get(filename)
            img = image.imfrombytes(img_bytes, flag=self.color_type)
            if self.to_float32:
                img = img.astype(np.float32)
            imgs.append(img)
        imgs = np.array(imgs)

        results['video_name'] = video_name
        results['ori_video_name'] = results['video_info']['video_name']
        results['imgs'] = imgs
        results['tsize'] = imgs.shape[0]
        results['pad_tsize'] = imgs.shape[0]
        results['img_fields'] = ['imgs']

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@registry.register_module('pipeline')
class LoadAnnotations(object):
    """Load mutiple types of annotations.

    Args:
        with_segment (bool): Whether to parse and load the segment annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`fileio.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_segment=True,
                 with_label=True,
                 file_client_args=dict(backend='disk')):
        self.with_segment = with_segment
        self.with_label = with_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_segments(self, results):
        """Private function to load segment annotations.

        Args:
            results (dict): Result dict from :obj:`.CustomDataset`.

        Returns:
            dict: The dict contains loaded segment annotations.
        """

        ann_info = results['ann_info']
        results['gt_segments'] = ann_info['segments'].copy()

        gt_segments_ignore = ann_info.get('segments_ignore', None)
        if gt_segments_ignore is not None:
            results['gt_segments_ignore'] = gt_segments_ignore.copy()
            results['segment_fields'].append('gt_segments_ignore')
        results['segment_fields'].append('gt_segments')
        return results

    def _load_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`.CustomDataset`.

        Returns:
            dict: The dict contains loaded segment, label annotations.
        """

        if self.with_segment:
            results = self._load_segments(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_segment={self.with_segment}, '
        repr_str += f'with_label={self.with_label})'
        return repr_str


@registry.register_module('pipeline')
class Time2Frame(object):
    """Switch time point to frame index.
    """

    def __call__(self, results):
        """Call function to switch time point to frame index.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Switch results.
        """

        for key in results.get('segment_fields', []):
            results[key] = results[key] * results['fps']

        return results
