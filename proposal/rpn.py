import os
import torch
import torch.nn as nn
from libs.model.rpn.rpn import _RPN
from libs.model.rpn.anchor_target_layer import _AnchorTargetLayer
from libs.model.rpn.proposal_layer import _ProposalLayer
from libs.nms.pth_nms import pth_nms as nms
from configparser import ConfigParser


class RPN(nn.Module):
    """Region Proposal Network Wrapper support single map and pyramid maps.   
    
    """

    def __init__(self, dim, use_fpn):
        """
        Args: 
            dim: depth of input feature map, e.g. FPN:256, resnet C4:1024
        """
        super(RPN, self).__init__()
        self.config = ConfigParser()
        self.config.read(os.path.abspath(os.path.join(__file__, "../../", 'config.ini')))
        self.rpn = _RPN(dim)
        self.use_fpn = use_fpn
        if self.use_fpn:
            feature_strides = (4, 8, 16, 32, 64)
            anchor_areas = [int(i) for i in self.config['RPN']['ANCHOR_AREAS_FPN'].split()]
            self.anchor_scales = []
            for i in range(len(anchor_areas)):
                assert anchor_areas[i] % feature_strides[i] == 0, (
                    "anchor area must be multiple of feat stride.")
                self.anchor_scales.append(anchor_areas[i] // feature_strides[i])

            self.anchor_ratios = [float(i) for i in self.config['RPN']['ANCHOR_RATIOS'].split()]

            ################################################
            # monkey patches adapt libs/RPN to support FPN.
            ################################################

            # define bg/fg classification score layer
            self.rpn.nc_score_out = len(self.anchor_ratios) * 2  # 2(bg/fg) * 3 (anchors)
            self.rpn.RPN_cls_score = nn.Conv2d(512, self.rpn.nc_score_out, 1, 1, 0)

            # define anchor box offset prediction layer
            self.rpn.nc_bbox_out = len(self.anchor_ratios) * 4  # 4(coords) * 3 (anchors)
            self.rpn.RPN_bbox_pred = nn.Conv2d(512, self.rpn.nc_bbox_out, 1, 1, 0)

            self.RPN_anchor_targets = [_AnchorTargetLayer(feat_stride=feature_strides[idx],
                                                          scales=[self.anchor_scales[idx]],
                                                          ratios=self.anchor_ratios)
                                       for idx, scale in enumerate(self.anchor_scales)]
            self.RPN_proposals = [_ProposalLayer(feat_stride=feature_strides[idx],
                                                 scales=[self.anchor_scales[idx]],
                                                 ratios=self.anchor_ratios)
                                  for idx, scale in enumerate(self.anchor_scales)]
            #################################################

    def forward(self, feature_maps, gt_bboxes=None, img_shape=None):
        """
        
        Args:
            feature_maps: [p2, p3, p4, p5, p6] or [c5], feature pyramid or single feature map.
            gt_bboxes: [N, M, (x1, y1, x2, y2)].
            img_shape: [height, width], Image shape. 
        Returns:
             rois(Tensor): [N, M, (idx, x1, y1, x2, y2)] N: batch size, M: number of roi after 
                nms, idx: bbox index in mini-batch.
             rpn_loss_cls(Tensor): Classification loss
             rpn_loss_bbo(Tensor)x: Bounding box regression loss
        """
        batch_size = feature_maps[0].size(0)
        assert batch_size == 1, "batch_size > 1 will add support later."

        if self.use_fpn:
            if self.training:
                post_nms_top_n = int(self.config['FPN']['TRAIN_FPN_POST_NMS_TOP_N'])
                nms_thresh = float(self.config['FPN']['TRAIN_FPN_NMS_THRESH'])
            else:
                post_nms_top_n = int(self.config['FPN']['TEST_FPN_POST_NMS_TOP_N'])
                nms_thresh = float(self.config['FPN']['TEST_FPN_NMS_THRESH'])
            rois_pre_nms = []
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            for idx, feature in enumerate(feature_maps):
                self.rpn.RPN_anchor_target = self.RPN_anchor_targets[idx]
                self.rpn.RPN_proposal = self.RPN_proposals[idx]
                rpn_result = self.rpn(feature, img_shape, gt_bboxes, None)
                roi_single, loss_cls_single, loss_bbox_single = rpn_result
                rpn_loss_cls += loss_cls_single
                rpn_loss_bbox += loss_bbox_single

                rois_pre_nms.append(roi_single)

            rois_pre_nms = torch.cat(rois_pre_nms, 1)  # [N, M, 5], torch.cat() at dim 'M'.
            # Apply nms to result of all pyramid rois.
            keep_idx = nms(rois_pre_nms[0, :, 1:], nms_thresh)
            keep_idx = keep_idx[:post_nms_top_n]
            rois_per_img = torch.cat([rois_pre_nms[:, idx, :] for idx in keep_idx])
            rois = rois_per_img[:, :5]  # remove roi_score
            rois = rois.unsqueeze(0)
            rpn_loss_cls /= len(feature_maps)
            rpn_loss_bbox /= len(feature_maps)
        else:
            rpn_result = self.rpn(feature_maps[0], img_shape, gt_bboxes, None)
            rois_rpn, rpn_loss_cls, rpn_loss_bbox = rpn_result
            rois = rois_rpn[:, :, [0, 2, 3, 4, 5]]  # remove roi_score

        return rois, rpn_loss_cls, rpn_loss_bbox
