import os.path as osp
import tempfile
from collections import defaultdict
from copy import deepcopy

import numpy as np
from torch.utils.data import Dataset

import vedacore.fileio as fileio
from vedacore.misc import registry
from vedatad.misc.evaluation import eval_map
from .pipelines import Compose


@registry.register_module('dataset')
class CustomDataset(Dataset):
    """Custom dataset for temporal action detection.

    The annotation format is shown as follows. The `ann` field is optional for
    testing.

    .. code-block:: none

        [
            {
                'video_name': 'abc',
                'width': 320,
                'height': 180,
                'duration': 14.93,
                'frames': 1024,
                'fps': 30,
                'ann': {
                    'segments': <np.ndarray> (n, 2),
                    'labels': <np.ndarray> (n, ),
                    'segments_ignore': <np.ndarray> (k, 2), (optional field)
                    'labels_ignore': <np.ndarray> (k, 2) (optional field)
                }
            },
            ...
        ]

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        video_prefix (str, optional): Video frames root. Default: "".
        test_mode (bool, optional): If set True, annotation will not be loaded.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 video_prefix='',
                 proposal_file=None,
                 test_mode=False):
        self.ann_file = ann_file
        self.video_prefix = video_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.CLASSES = self.get_classes(classes)

        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)
        # filter data infos if classes are customized
        if self.custom_classes:
            self.data_infos = self.get_subset_by_classes()

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
        # filter videos too small
        if not test_mode:
            valid_inds = self._filter_videos()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        return fileio.load(ann_file)

    def load_proposals(self, proposal_file):
        """Load proposal from proposal file."""
        return fileio.load(proposal_file)

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.data_infos[idx]['ann']

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.data_infos[idx]['ann']['labels'].astype(np.int).tolist()

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['video_prefix'] = self.video_prefix
        results['proposal_file'] = self.proposal_file
        results['segment_fields'] = []

    def _filter_videos(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, video_info in enumerate(self.data_infos):
            if min(video_info['width'], video_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            video_info = self.data_infos[i]
            if video_info['width'] / video_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                True).
        """

        if self.test_mode:
            data = self.prepare_test_video(idx)
            return data

        while True:
            data = self.prepare_train_video(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data

    def prepare_train_video(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        video_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        if len(ann_info['segments']) == 0:
            return None

        results = dict(video_info=video_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_video(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
            piepline.
        """

        video_info = self.data_infos[idx]
        results = dict(video_info=video_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
        """
        if classes is None:
            cls.custom_classes = False
            return cls.CLASSES

        cls.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = fileio.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def get_subset_by_classes(self):
        return self.data_infos

    def _det2json(self, results):
        """Convert detection results to ActivityNet json style."""
        json_results = dict(
            version='', results=defaultdict(list), external_data=dict())
        for idx in range(len(self)):
            video_name = self.data_infos[idx]['video_name']
            for label, segments in enumerate(results[idx]):
                for segment in segments:
                    start, end, score = segment.tolist()
                    label_name = self.CLASSES[label]
                    res = dict(
                        segment=[start, end], score=score, label=label_name)
                    json_results['results'][video_name].append(res)
        return json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to an ActivityNet style json file.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.json"

        Returns:
            str: output filename.
        """

        json_results = self._det2json(results)
        result_file = f'{outfile_prefix}.json'
        fileio.dump(json_results, result_file, indent=4)

        return result_file

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for ActivityNet
        evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_file, tmp_dir), result_file is saved file name,
                tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_file = self.results2json(results, jsonfile_prefix)
        return result_file, tmp_dir

    def convert2time(self, results):
        time_results = []
        _results = deepcopy(results)
        for idx, video_results in enumerate(_results):
            fps = self.data_infos[idx]['fps']
            res = []
            for class_result in video_results:
                class_result[:, :2] /= fps
                res.append(class_result)
            time_results.append(res)

        return time_results

    def evaluate(self,
                 results,
                 mode='anet',
                 logger=None,
                 jsonfile_prefix=None,
                 iou_thr=0.5,
                 scale_ranges=None,
                 convert2time=True):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            mode (str | None): Mode name, there are minor differences in
                metrics for different modes, e.g. "anet", "voc07", "voc12" etc.
                Default: anet.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json file. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            convert2time (bool): Whether convert results to time. Default: True
        """
        if convert2time:
            results = self.convert2time(results)

        if jsonfile_prefix is not None:
            self.format_results(results, jsonfile_prefix)

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        assert isinstance(iou_thr, float)
        mean_ap, _ = eval_map(
            results,
            annotations,
            scale_ranges=scale_ranges,
            iou_thr=iou_thr,
            mode=mode,
            logger=logger,
            label_names=self.CLASSES)
        eval_results['mAP'] = mean_ap
        return eval_results
