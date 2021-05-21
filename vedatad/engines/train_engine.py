from vedacore.misc import registry
from vedacore.optimizers import build_optimizer
from vedatad.criteria import build_criterion
from .base_engine import BaseEngine


@registry.register_module('engine')
class TrainEngine(BaseEngine):

    def __init__(self, model, criterion, optimizer):
        super().__init__(model)
        self.criterion = build_criterion(criterion)
        self.optimizer = build_optimizer(self.model, optimizer)

    def extract_feats(self, img):
        feats = self.model(img, train=True)
        return feats

    def forward(self, data):
        return self.forward_impl(**data)

    def forward_impl(self,
                     imgs,
                     video_metas,
                     gt_segments,
                     gt_labels,
                     gt_segments_ignore=None):
        feats = self.extract_feats(imgs)
        losses = self.criterion.loss(feats, video_metas, gt_segments,
                                     gt_labels, gt_segments_ignore)
        return losses
