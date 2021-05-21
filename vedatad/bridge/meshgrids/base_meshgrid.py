from abc import ABCMeta, abstractmethod


class BaseMeshGrid(metaclass=ABCMeta):

    def __init__(self, strides):
        assert len(strides) > 0
        self.strides = strides

    @abstractmethod
    def gen_anchor_mesh(self):
        pass
