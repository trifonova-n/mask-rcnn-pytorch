import torch
import torch.nn as nn
from libs.model.rpn.rpn import _RPN
from libs.model.rpn.anchor_target_layer import _AnchorTargetLayer
from libs.model.rpn.proposal_layer import _ProposalLayer
from libs.model.rpn.config import cfg
from libs.nms.pth_nms import pth_nms as nms


class RPN(nn.Module):
    """Region Proposal Network Wrapper.   
    
    """

    def __init__(self, dim, use_fpn):
        """
        Args: 
            dim: depth of input feature map, e.g., 512
        """
        super(RPN, self).__init__()
        self.rpn = _RPN(dim)
        self.use_fpn = use_fpn
        if self.use_fpn:
            self.anchor_ratios = cfg.ANCHOR_RATIOS
            self.anchor_scales = [8, 16, 32, 64, 128]
            self.feat_strides = [4, 8, 16, 32, 32]
            self.RPN_anchor_targets = [_AnchorTargetLayer(feat_stride=self.feat_strides[idx],
                                                          scales=[scale],
                                                          ratios=self.anchor_ratios)
                                       for idx, scale in enumerate(self.anchor_scales)]
            self.RPN_proposals = [_ProposalLayer(feat_stride=self.feat_strides[idx],
                                                 scales=[scale],
                                                 ratios=self.anchor_ratios)
                                  for idx, scale in enumerate(self.anchor_scales)]

    def forward(self, feature_maps, gt_bboxes=None, img_shape=None):
        """
        
        Args:
            feature_maps: [p2, p3, p4, p5, p6] or [c5], feature pyramid or single feature map.
            gt_bboxes: [N, M, (x1, y1, x2, y2)].
            img_shape: [height, width], Image shape. 
        Returns:
             rois(Tensor): [N, M, (idx, x1, y1, x2, y2)] N: batch size, M: number of roi after nms, 
                 idx: bbox index in mini-batch.
             rpn_loss_cls(Tensor): Classification loss
             rpn_loss_bbo(Tensor)x: Bounding box regression loss
        """
        batch_size = feature_maps[0].size(0)
        nms_output_num = cfg.TEST.RPN_POST_NMS_TOP_N
        if self.training:
            nms_output_num = cfg.TRAIN.RPN_POST_NMS_TOP_N

        if self.use_fpn:
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
                roi_score = roi_single[:, :, 1]
                roi_bbox = roi_single[:, :, 2:]
                roi_score.unsqueeze_(-1)
                rois_pre_nms.append(torch.cat((roi_bbox, roi_score), 2))

            rois_pre_nms = torch.cat(rois_pre_nms, 1)  # [N, M, 5], torch.cat() at dim 'M'.
            rois = feature_maps[0].data.new(batch_size, nms_output_num, 5).zero_()
            # Apply nms to result of all pyramid rois.
            for i in range(batch_size):
                keep_idx = nms(rois_pre_nms[i], cfg.TRAIN.RPN_NMS_THRESH)
                keep_idx = keep_idx[:nms_output_num]
                rois_per_img = torch.cat([rois_pre_nms[i, idx, :].unsqueeze(0) for idx in keep_idx])
                rois[i, :, 0] = i
                rois[i, :rois_per_img.size(0), 1:] = rois_per_img[:, :4]  # remove roi_score
        else:
            rpn_result = self.rpn(feature_maps[0], img_shape, gt_bboxes, None)
            rois_rpn, rpn_loss_cls, rpn_loss_bbox = rpn_result
            rois = feature_maps[0].data.new(batch_size, nms_output_num, 5).zero_()
            rois[:, :, 0] = 0
            rois[:, :, 1:] = rois_rpn[:, :, 2:]  # remove roi_score
        return rois, rpn_loss_cls, rpn_loss_bbox
