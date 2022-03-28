import glob
import os.path as osp

import numpy as np

from vedacore import fileio
from vedacore.image import imread
from vedacore.misc import registry
from .custom import CustomDataset



@registry.register_module('dataset')
class MultiThumosDataset(CustomDataset):
    """MultiThumos dataset for temporal action detection."""

    CLASSES = ('BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
        'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch',
        'GolfSwing', 'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump',
        'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing', 'ThrowDiscus',
        'VolleyballSpiking', 'Walk', 'Sit', 'StandUp', 'Stand', 'Run', 'Jump',
        'Fall', 'PickUp', 'Hug', 'PatPerson', 'HighFive', 'FistPump', 'ClapHands',
        'OneRaisedArmCelebrate', 'TwoRaisedArmCelebrate', 'Throw', 'Drop',
        'OneHandedCatch', 'TwoHandedCatch', 'BodyRoll', 'BodyTurn', 'BodyBend',
        'BodyContract', 'Squat', 'BasketballDribble', 'BasketballPass',
        'BasketballBlock', 'BasketballShot', 'BasketballGuard', 'VolleyballSet',
        'VolleyballBump', 'VolleyballBlock', 'VolleyballServe', 'WeightliftingClean',
        'WeightliftingJerk', 'HammerThrowWindUp', 'HammerThrowSpin', 'HammerThrowRelease',
        'PoleVaultPlantPole', 'ShotPutBend', 'DiscusWindUp', 'DiscusRelease', 'TalkToCamera',
        'CloseUpTalkToCamera', 'NoHuman')

    def __init__(self, **kwargs):
        super(MultiThumosDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from MultiThumos json ann_file.

        Args:
            ann_file (str): Path of JSON file.

        Returns:
            list[dict]: Annotation info from JSON file.
        """

        data_infos = []
        data = fileio.load(ann_file)
        for video_name, video_info in data['database'].items():
            data_info = dict()
            data_info['video_name'] = video_name
            data_info['duration'] = float(video_info['duration'])
            imgfiles = glob.glob(osp.join(self.video_prefix, video_name, '*'))
            num_imgs = len(imgfiles)
            data_info['frames'] = num_imgs
            data_info['fps'] = int(round(num_imgs / video_info['duration']))
            img = imread(imgfiles[0])
            data_info['height'], data_info['width'] = img.shape[:2]

            segments = []
            labels = []
            segments_ignore = []
            for ann in video_info['annotations']:
                label = ann['label']
                segment = ann['segment']

                if not self.test_mode:
                    segment[0] = min(video_info['duration'],
                                     max(0, segment[0]))
                    segment[1] = min(video_info['duration'],
                                     max(0, segment[1]))
                    if segment[0] >= segment[1]:
                        continue

                if label == 'Ambiguous':
                    segments_ignore.append(segment)
                elif label in self.CLASSES:
                    segments.append(segment)
                    labels.append(self.CLASSES.index(label))
                else:
                    continue
            if not segments:
                segments = np.zeros((0, 2))
                labels = np.zeros((0, ))
            else:
                segments = np.array(segments)
                labels = np.array(labels)
            if not segments_ignore:
                segments_ignore = np.zeros((0, 2))
            else:
                segments_ignore = np.array(segments_ignore)
            data_info['ann'] = dict(
                segments=segments.astype(np.float32),
                labels=labels.astype(np.int64),
                segments_ignore=segments_ignore.astype(np.float32))

            data_infos.append(data_info)

        return data_infos
