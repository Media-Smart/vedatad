import torch

from vedacore.misc import registry
from vedatad.bridge import build_converter, build_meshgrid
from vedatad.misc.segment import segment2result, multiclass_nms
from .base_engine import BaseEngine


@registry.register_module('engine')
class InferEngine(BaseEngine):

    def __init__(self, model, meshgrid, converter, num_classes, use_sigmoid,
                 test_cfg):
        super().__init__(model)
        self.meshgrid = build_meshgrid(meshgrid)
        self.converter = build_converter(converter)
        if use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.test_cfg = test_cfg

    def extract_feats(self, img):
        feats = self.model(img, train=False)
        return feats

    def _get_raw_dets(self, imgs, video_metas):
        """
        Args:
            imgs (torch.Tensor): shape N*3*T*H*W, N is batch size
            video_metas (list): len(video_metas) = N
        Returns:
            dets(list): len(dets) is the batch size, len(dets[ii]) = #classes,
                dets[ii][jj] is an np.array whose shape is N*3
        """
        feats = self.extract_feats(imgs)

        featmap_tsizes = [feat.shape[2] for feat in feats[0]]
        dtype = feats[0][0].dtype
        device = feats[0][0].device
        anchor_mesh = self.meshgrid.gen_anchor_mesh(featmap_tsizes,
                                                    video_metas, dtype, device)
        # segments, scores, score_factor
        dets = self.converter.get_segments(anchor_mesh, video_metas, *feats)

        return dets

    def _simple_infer(self, imgs, video_metas):
        """
        Args:
            imgs (torch.Tensor): shape N*3*T*H*W, N is batch size
            video_metas (list): len(video_metas) = N
        Returns:
            dets(list): len(dets) is the batch size, len(dets[ii]) = #classes,
                dets[ii][jj] is an np.array whose shape is N*3
        """
        dets = self._get_raw_dets(imgs, video_metas)
        batch_size = len(dets)

        result_list = []
        for ii in range(batch_size):
            segments, scores, centerness = dets[ii]
            det_segments, det_labels = multiclass_nms(
                segments,
                scores,
                self.test_cfg.score_thr,
                self.test_cfg.nms,
                self.test_cfg.max_per_video,
                score_factors=centerness)
            segment_result = segment2result(det_segments, det_labels,
                                            self.cls_out_channels)
            result_list.append(segment_result)

        return result_list

    def _aug_infer(self, imgs_list, video_metas_list):
        assert len(imgs_list) == len(video_metas_list)
        dets = []
        ntransforms = len(imgs_list)
        for idx in range(len(imgs_list)):
            imgs = imgs_list[idx]
            video_metas = video_metas_list[idx]
            tdets = self._get_raw_dets(imgs, video_metas)
            dets.append(tdets)
        batch_size = len(dets[0])
        nclasses = len(dets[0][0])
        merged_dets = []
        for ii in range(batch_size):
            single_video = []
            for kk in range(nclasses):
                single_class = []
                for jj in range(ntransforms):
                    single_class.append(dets[jj][ii][kk])
                single_video.append(torch.cat(single_class, axis=0))
            merged_dets.append(single_video)

        result_list = []
        for ii in range(batch_size):
            segments, scores, centerness = merged_dets[ii]
            det_segments, det_labels = multiclass_nms(
                segments,
                scores,
                self.test_cfg.score_thr,
                self.test_cfg.nms,
                self.test_cfg.max_per_video,
                score_factors=centerness)
            segment_result = segment2result(det_segments, det_labels,
                                            self.cls_out_channels)
            result_list.append(segment_result)

        return result_list

    def infer(self, imgs, video_metas):
        if len(imgs) == 1:
            return self._simple_infer(imgs[0], video_metas[0])
        else:
            return self._aug_infer(imgs, video_metas)
