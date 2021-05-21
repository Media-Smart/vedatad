from vedacore.misc import registry
from .compose import Compose


@registry.register_module('pipeline')
class OverlapCropAug(object):
    """Test-time augmentation with overlapped crop.

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        num_frames (int): The cropped frame num.
        overlap_ratio (int, optional): Overlap ratio for crop.
    """

    def __init__(self, transforms, num_frames, overlap_ratio=0.5):
        assert 0 <= overlap_ratio < 1

        self.transforms = Compose(transforms)
        self.num_frames = num_frames
        self.overlap_ratio = overlap_ratio

    def __call__(self, results):
        """Call function to apply test time augment transforms on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """

        stride = int(self.num_frames * (1 - self.overlap_ratio))
        tsize = results['tsize']
        start = 0
        aug_data = []
        while True:
            end = min(start + self.num_frames, tsize)
            _results = results.copy()
            _results['patch'] = [start, end]
            aug_data.append(self.transforms(_results))

            if end == tsize:
                break
            else:
                start += stride

        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'num_frames={self.num_frames}, '
        repr_str += f'overlap_ratio={self.overlap_ratio})'
        return repr_str
