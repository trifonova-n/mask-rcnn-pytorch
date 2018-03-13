from backbone.resnet_101_fpn import ResNet_101_FPN
from proposal.rpn import RPN
from head.cls_bbox import ClsBBoxHead_fc as ClsBBoxHead
from head.mask import MaskHead
from pooling.roi_align import RoiAlign

import torch
import torch.nn.functional as F
import torch.nn as nn
import random


class MaskRCNN(nn.Module):
    """Mask R-CNN
    
    References: https://arxiv.org/pdf/1703.06870.pdf
    
    Notes: In comments below, we assume N: batch size, M: number of roi,
        C: feature map channel, H: image height, W: image width
        
    """

    def __init__(self, num_classes):
        super(MaskRCNN, self).__init__()
        self.num_classes = num_classes
        self.fpn = ResNet_101_FPN()
        self.rpn = RPN(dim=512)
        self.roi_align = RoiAlign(grid_size=(14, 14))
        self.cls_box_head = ClsBBoxHead(depth=512, pool_size=14, num_classes=num_classes)
        self.mask_head = MaskHead(depth=512, pool_size=14, num_classes=num_classes)
        # Todo: initialize weights.

    def forward(self, x, gt_classes=None, gt_bboxes=None, gt_masks=None):
        """
        
        Args:
            x: image data. NxCxHxW.  
            gt_classes: NxMx1, ground truth class ids.
            gt_bboxes: NxMx4(x1, y1, x2, y2), ground truth bounding boxes.
            gt_masks: NxMxHxW, ground truth masks.
            
        Returns:
            prob_cls: NxMx(num_classes), probability of classification. 
            reg_bbox: NxMx(x1, y1, x2, y2), regression of bounding-box. 
            prob_cls:  NxMx2(num_classes), probability of mask.
            
        """
        if self.training:
            assert gt_classes is not None
            assert gt_bboxes is not None
            assert gt_masks is not None

        self.batch_size = x.size(0)
        p2, p3, p4, p5, p6 = self.fpn(x)
        rpn_features_rpn = [p2, p3, p4, p5, p6]
        fpn_features = [p2, p3, p4, p5]
        proposals, rpn_loss_cls, rpn_loss_bbox = self.rpn(rpn_features_rpn, gt_bboxes)
        gen_result = self._generate_targets(proposals, gt_classes, gt_bboxes, gt_masks)
        rois, cls_targets, bbox_targets, mask_targets = gen_result
        rois_pooling = self._roi_align_fpn(fpn_features, rois, img_height=x.size(2),
                                           img_width=x.size(3))
        rois_pooling = torch.cat(rois_pooling, 0)

        prob_cls, reg_bbox = self.cls_box_head(rois_pooling)
        prob_mask = self.mask_head(rois_pooling)

        self._calc_maskrcnn_loss()
        mask_loss = rpn_loss_cls + rpn_loss_bbox
        return prob_cls, reg_bbox, prob_mask, mask_loss

    def _generate_targets(self, proposals, gt_classes, gt_bboxes, gt_masks):
        """Process proposals from RPN to generate rois to feed predict heads, and
            corresponding head targets.
        
        Args:
            proposals: NxMx4, proposals from RPN. 
            gt_classes: NxRxNum_Classes, ground truth class ids.
            gt_bboxes: NxRx4(x1, y1, x2, y2), ground truth bounding boxes.
            gt_masks: NxRxHxW, ground truth masks.  
              
        Returns: 
            rois: NxSx4(x1, y1, x2, y2), rois to feed predict heads. 
            cls_targets: NxSxNum_Classes, train targets for classification.
            bbox_targets: NxSx4(x1, y1, x2, y2), train targets for bounding box regression.
            mask_targets: NxSxHxW, train targets for mask prediction.
            
        Notes: In above, R: number of ground truth objects, S: number of rois to train.

        """
        self.train_rois_num = 1024
        num_proposals = proposals.size(1)
        height = gt_bboxes.size(2)
        width = gt_bboxes.size(3)
        num_gt_bboxes = gt_bboxes.size(1)
        rois = torch.zeros(self.batch_size, num_proposals, num_gt_bboxes, 2, 4)
        cls_targets = torch.zeros(self.batch_size, num_proposals, num_gt_bboxes, 2,
                                  self.num_classes)
        bbox_targets = torch.zeros(self.batch_size, num_proposals, num_gt_bboxes, 2, 4)
        mask_targets = torch.zeros(self.batch_size, num_proposals, num_gt_bboxes, 2, height, width)
        for i in range(self.batch_size):
            for j in range(num_proposals):
                for k in range(num_gt_bboxes):
                    iou = MaskRCNN._calc_iou(proposals[i, j, :], gt_bboxes[i, k, :])
                    if iou < 0.5:
                        rois[i, j, k, 0, :] = proposals[i, j, :]
                        cls_targets[i, j, k, 0, :] = gt_classes[i, k, :]
                        bbox_targets[i, j, k, 0, :] = gt_bboxes[i, k, :]
                        mask_targets[i, j, k, 0, :] = gt_masks[i, k, :]
                    else:
                        rois[i, j, k, 1, :] = proposals[i, j, :]
                        cls_targets[i, j, k, 1, :] = gt_classes[i, k, :]
                        bbox_targets[i, j, k, 1, :] = gt_bboxes[i, k, :]
                        mask_targets[i, j, k, 1, :] = gt_masks[i, k, :]

        rois = rois.view(self.batch_size, num_proposals * num_gt_bboxes, 2, -1)
        cls_targets = cls_targets.view(self.batch_size, num_proposals * num_gt_bboxes, 2, -1)
        bbox_targets = bbox_targets.view(self.batch_size, num_proposals * num_gt_bboxes, 2, -1)
        mask_targets = mask_targets.view(self.batch_size, num_proposals * num_gt_bboxes, 2,
                                         height, width)
        # train_rois should have 1:3 positive negative ratio, see Mask R-CNN paper.
        rois_neg = rois[:, :, 0, :]
        rois_pos = rois[:, :, 1, :]
        rois_neg.squeeze_()
        rois_pos.squeeze_()

        cls_targets_neg = cls_targets[:, :, 0, :]
        cls_targets_pos = cls_targets[:, :, 1, :]
        cls_targets_neg.squeeze_()
        cls_targets_pos.squeeze_()

        bbox_targets_neg = bbox_targets[:, :, 0, :]
        bbox_targets_pos = bbox_targets[:, :, 1, :]
        bbox_targets_neg.squeeze_()
        bbox_targets_pos.squeeze_()

        mask_targets_neg = mask_targets[:, :, 0, :, :]
        mask_targets_pos = mask_targets[:, :, 1, :, :]
        mask_targets_neg.squeeze_()
        mask_targets_pos.squeeze_()

        sample_size_neg = int(0.75 * num_proposals * num_gt_bboxes)
        sample_size_pos = self.train_rois_num - sample_size_neg

        sample_indexes_neg = random.sample(range(num_proposals * num_gt_bboxes), sample_size_neg)
        sample_indexes_pos = random.sample(range(num_proposals * num_gt_bboxes), sample_size_pos)

        rois_neg_sampled = rois_neg[:, sample_indexes_neg, :]
        rois_pos_sampled = rois_pos[:, sample_indexes_pos, :]

        cls_targets_neg_sampled = cls_targets_neg[:, sample_indexes_neg, :]
        cls_targets_pos_sampled = cls_targets_pos[:, sample_indexes_pos, :]

        bbox_targets_neg_sampled = bbox_targets_neg[:, sample_indexes_neg, :]
        bbox_targets_pos_sampled = bbox_targets_pos[:, sample_indexes_pos, :]

        mask_targets_pos_sampled = mask_targets_pos[:, sample_indexes_pos, :, :]

        # Mask target only define on positive roi, and is the intersection of roi and
        # ground truth mask.
        mask_targets = mask_targets_pos_sampled
        # Todo: modify mask_targets to be intersection with roi.
        rois = torch.cat([rois_neg_sampled, rois_pos_sampled], 1)
        cls_targets = torch.cat([cls_targets_neg_sampled, cls_targets_pos_sampled], 1)
        bbox_targets = torch.cat([bbox_targets_neg_sampled, bbox_targets_pos_sampled], 1)

        return rois, cls_targets, bbox_targets, mask_targets

    @staticmethod
    def _calc_iou(bbox1, bbox2):
        """ Calculate IoU(Intersection over union) of two bounding boxes.
        
        Args:
            bbox1: (x1, y1, x2, y2), bounding box 1.
            bbox2: (x1, y1, x2, y2), bounding box 2.

        Returns:
            iou: IoU of two bounding boxes.
        """
        area_1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1])
        area_2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1])

        inter_x1 = max(bbox1[0], bbox2[1])
        inter_y1 = max(bbox1[1], bbox2[1])
        inter_x2 = min(bbox1[2], bbox2[2])
        inter_y2 = min(bbox1[3], bbox2[3])

        inter_w = max(0, inter_x2 - inter_x1 + 1)
        inter_h = max(0, inter_y2 - inter_y1 + 1)

        intersection = inter_w * inter_h
        iou = intersection / (area_1 + area_2 + intersection)

        return iou

    @staticmethod
    def _calc_maskrcnn_loss(cls_prob, bbox_reg, mask_prob, cls_targets, bbox_targets, mask_targets):
        """ Calculate Mask R-CNN loss.
        
        Args:
            cls_prob: NxSxNum_Classes, classification predict probability.
            bbox_reg: NxSx4(x1, y1, x2, y2), bounding box prediction.
            mask_prob: NxSxHxW, mask prediction.
            cls_targets: NxSxNum_classes, classification targets.
            bbox_targets: NxSx4(x1, y1, x2, y2), bounding box targets.
            mask_targets: NxSxHxW, mask targets.

        Returns:
            maskrcnn_loss: Total loss of Mask R-CNN predict heads.
            
        Notes: In above, S: number of rois feed to prediction heads.
        
        """
        cls_loss = F.nll_loss(cls_prob, cls_targets)
        bbox_loss = F.smooth_l1_loss(bbox_reg, bbox_targets)
        mask_loss = F.binary_cross_entropy(mask_prob, mask_targets)
        maskrcnn_loss = cls_loss + bbox_loss + mask_loss
        return maskrcnn_loss

    def _roi_align_fpn(self, fpn_features, rois, img_width, img_height):
        """When use fpn backbone, set RoiAlign use different levels of fpn feature pyramid
         according to RoI size.
         
        Args:
            fpn_features: [p2, p3, p4, p5], 
            rois: NxMx5(n ,x1, y1, x2, y2), RPN proposals.
            img_width: Input image width.
            img_height: Input image height.

        Returns:
            roi_pools: RoI after use RoIAlign.
            
        """
        # flatten NxMx4 to (NxM)x4
        rois_reshape = rois.view(-1, rois.size(-1))
        bboxes = rois_reshape[:, 1:]
        bbox_indexes = rois_reshape[:, 0]
        roi_pools = []
        for idx, bbox in enumerate(bboxes):
            # In feature pyramid network paper, alpha is 224 and image short side 800 pixels,
            # for using of small image input, like maybe short side 256, here alpha is
            # parameterized by image short side size.
            alpha = 224 // 800 * (img_width if img_width <= img_height else img_height)
            bbox_width = torch.abs(bbox[0] - bbox[2])
            bbox_height = torch.abs(bbox[1] - bbox[3])
            log2 = torch.log(torch.sqrt(bbox_height * bbox_width)) / torch.log(2) / alpha
            level = torch.floor(4 + log2) - 2  # minus 2 to make level 0 indexed
            # Rois small or big enough may get level below 0 or above 3.
            level = torch.clamp(level, 0, 3)
            bbox = torch.unsqueeze(bbox, 0)
            roi_pool_per_box = self.roi_align(fpn_features[level], bbox, bbox_indexes[idx])
            roi_pools.append(roi_pool_per_box)
        return roi_pools
