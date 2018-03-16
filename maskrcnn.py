from backbone.resnet_101_fpn import ResNet_101_FPN
from proposal.rpn import RPN
from head.cls_bbox import ClsBBoxHead_fc as ClsBBoxHead
from head.mask import MaskHead
from pooling.roi_align import RoiAlign
from util.utils import calc_iou, calc_maskrcnn_loss, coord_corner2center

import random
import torch
import torch.nn as nn
from configparser import ConfigParser


class MaskRCNN(nn.Module):
    """Mask R-CNN model.
    
    References: https://arxiv.org/pdf/1703.06870.pdf
    
    Notes: In comments below, we assume N: batch size, M: number of roi,
        C: feature map channel, H: image height, W: image width,
        (x1, y1, x2, y2) stands for top-left and bottom-right coord of bounding box, 
        without normalization, (x, y, w, h) stands for center coord, height and 
        width of bounding box.
        
    """

    def __init__(self, num_classes):
        super(MaskRCNN, self).__init__()
        self.config = ConfigParser()
        self.config.read("./config.ini")
        self.num_classes = num_classes
        self.fpn = ResNet_101_FPN()
        self.rpn = RPN(dim=256)
        self.roi_align = RoiAlign(grid_size=(14, 14))
        self.cls_box_head = ClsBBoxHead(depth=256, pool_size=14, num_classes=num_classes)
        self.mask_head = MaskHead(depth=256, pool_size=14, num_classes=num_classes)

    def forward(self, x, gt_classes=None, gt_bboxes=None, gt_masks=None):
        """
        
        Args:
            x: image data. NxCxHxW.  
            gt_classes: NxM, ground truth class ids.
            gt_bboxes: NxMx4(x1, y1, x2, y2), ground truth bounding boxes.
            gt_masks: NxMxHxW, ground truth masks.
            
        Returns:
            result(list of lists of dict): Outer list composed of mini-batch, inner list 
                composed of detected objects per image, dict composed of "cls_pred": class id,
                "bbox_pred" : bounding-box with tuple (x1, y1, x2, y2), "mask_pred" : mask 
                prediction with tuple (H,W).
                
                So, result[0][0]['cls_pred'] stands for class id of the first detected objects
                in first image of mini-batch.
            
        """

        p2, p3, p4, p5, p6 = self.fpn(x)
        rpn_features_rpn = [p2, p3, p4, p5, p6]
        fpn_features = [p2, p3, p4, p5]
        # img_shape = torch.x.size(2), x.size(3)
        rois, rpn_loss_cls, rpn_loss_bbox = self.rpn(rpn_features_rpn, gt_bboxes, img_shape)
        cls_targets, bbox_targets, mask_targets = None, None, None

        if self.training:
            assert gt_classes is not None
            assert gt_bboxes is not None
            assert gt_masks is not None
            gen_result = self._generate_targets(rois, gt_classes, gt_bboxes, gt_masks)
            rois, cls_targets, bbox_targets, mask_targets = gen_result

        rois_pooling = self._roi_align_fpn(fpn_features, rois, img_height=x.size(2),
                                           img_width=x.size(3))

        cls_prob, bbox_reg = self.cls_box_head(rois_pooling)
        mask_prob = self.mask_head(rois_pooling)

        maskrcnn_loss = calc_maskrcnn_loss(cls_prob, bbox_reg, mask_prob, cls_targets,
                                           bbox_targets, mask_targets)
        loss = rpn_loss_cls + rpn_loss_bbox + maskrcnn_loss
        result = self._process_result(cls_prob, bbox_reg, mask_prob)
        return result, loss

    def _process_result(self, cls_prob, bbox_reg, mask_prob):
        """Process heads output to get the final result.
        
        """
        result = []
        cls_id_prob, cls_id = torch.max(cls_prob, 2)
        cls_threshold = float(self.config['Test']['cls-threshold'])
        # remove background and predicted ids whose probability below threshold.
        keep_index = (cls_id > 0) & (cls_id_prob >= cls_threshold)
        for i in range(cls_prob.size(0)):
            objects = []
            for j in range(cls_prob.size(1)):
                pred_dict = {}
                if keep_index[i, j]:
                    pred_dict['cls_pred'] = cls_id[i, j]
                    pred_dict['bbox_pred'] = coord_corner2center(bbox_reg[i, j, :])
                    mask_threshold = self.config['Test']['mask_threshold']
                    pred_dict['mask_prob'] = mask_prob[i, j] >= mask_threshold
                objects.append(pred_dict)
            result.append(objects)

        return result

    def _generate_targets(self, proposals, gt_classes, gt_bboxes, gt_masks):
        """Process proposals from RPN to generate rois to feed predict heads, and
            corresponding head targets.
        
        Args:
            proposals: NxMx5(idx, x1, y1, x2, y2), proposals from RPN. 
            gt_classes: NxR, ground truth class ids.
            gt_bboxes: NxRx4(x1, y1, x2, y2), ground truth bounding boxes.
            gt_masks: NxRxHxW, ground truth masks.  
              
        Returns: 
            rois: NxSx5(idx, x1, y1, x2, y2), rois to feed RoIAlign. 
            cls_targets: NxSxNum_Classes, train targets for classification.
            bbox_targets: NxSx4(x, y, w, h), train targets for bounding box regression.  
            mask_targets: NxSxHxW, train targets for mask prediction.
            
        Notes: In above, M: number of rois from FRN, R: number of ground truth objects,
            S: number of rois to train.

        """
        train_rois_num = self.config['Train']['train-rois-num']
        batch_size = proposals.size(0)
        num_proposals = proposals.size(1)
        num_gt_bboxes = gt_bboxes.size(1)
        height = gt_bboxes.size(2)
        width = gt_bboxes.size(3)
        rois = torch.zeros(batch_size, num_proposals, num_gt_bboxes, 2, 5)
        cls_targets = gt_classes.new(batch_size, num_proposals, num_gt_bboxes, 2,
                                     self.num_classes).zero_()
        bbox_targets = gt_bboxes.new(batch_size, num_proposals, num_gt_bboxes, 2, 4).zero_()
        mask_targets = gt_masks.new(batch_size, num_proposals, num_gt_bboxes, 2, height,
                                    width).zero_()
        for i in range(batch_size):
            for j in range(num_proposals):
                for k in range(num_gt_bboxes):
                    iou = calc_iou(proposals[i, j, :1], gt_bboxes[i, k, :])
                    pos_neg_idx = 1
                    if iou < 0.5:
                        pos_neg_idx = 0
                    rois[i, j, k, pos_neg_idx, :] = proposals[i, j, :]
                    cls_targets[i, j, k, pos_neg_idx, :] = gt_classes[i, k, :]
                    # Transform bbox coord from (x1, y1, x2, y2) to(x, y, w, h).
                    x, y, w, h = coord_corner2center(gt_bboxes[i, k, :])
                    bbox_targets[i, j, k, pos_neg_idx, :] = x, y, w, h
                    mask_targets[i, j, k, pos_neg_idx, :, :] = gt_masks[i, k, :, :]

        rois = rois.view(batch_size, num_proposals * num_gt_bboxes, 2, -1)
        cls_targets = cls_targets.view(batch_size, num_proposals * num_gt_bboxes, 2, -1)
        bbox_targets = bbox_targets.view(batch_size, num_proposals * num_gt_bboxes, 2, -1)
        mask_targets = mask_targets.view(batch_size, num_proposals * num_gt_bboxes, 2,
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

        neg_num = rois_neg.size(1)
        pos_num = rois_pos.size(1)
        sample_size_neg = int(0.75 * train_rois_num)
        sample_size_pos = train_rois_num - sample_size_neg
        sample_size_neg = sample_size_neg if sample_size_neg <= neg_num else neg_num
        sample_size_pos = sample_size_pos if sample_size_pos <= pos_num else pos_num

        sample_index_neg = random.sample(range(neg_num), sample_size_neg)
        sample_index_pos = random.sample(range(pos_num), sample_size_pos)

        rois_neg_sampled = rois_neg[:, sample_index_neg, :]
        rois_pos_sampled = rois_pos[:, sample_index_pos, :]

        cls_targets_neg_sampled = cls_targets_neg[:, sample_index_neg, :]
        cls_targets_pos_sampled = cls_targets_pos[:, sample_index_pos, :]

        bbox_targets_neg_sampled = bbox_targets_neg[:, sample_index_neg, :]
        bbox_targets_pos_sampled = bbox_targets_pos[:, sample_index_pos, :]

        mask_targets_pos_sampled = mask_targets_pos[:, sample_index_pos, :, :]

        # Mask targets only define on positive rois, and are the intersection of
        # roi with ground truth mask.
        mask_targets = mask_targets_pos_sampled
        for i in range(mask_targets.size(0)):
            for j in range(mask_targets.size(1)):
                x1, y1, x2, y2 = rois_pos_sampled[i, j, :]
                mask_targets[i, j, :x1, :] = 0
                mask_targets[i, j, :, :y1] = 0
                mask_targets[i, j, x2:, :] = 0
                mask_targets[i, j, :, y2:] = 0

        rois = torch.cat([rois_neg_sampled, rois_pos_sampled], 1)
        cls_targets = torch.cat([cls_targets_neg_sampled, cls_targets_pos_sampled], 1)
        bbox_targets = torch.cat([bbox_targets_neg_sampled, bbox_targets_pos_sampled], 1)

        return rois, cls_targets, bbox_targets, mask_targets

    def _roi_align_fpn(self, fpn_features, rois, img_width, img_height):
        """When use fpn backbone, set RoiAlign use different levels of fpn feature pyramid
            according to RoI size.
         
        Args:
            fpn_features: (p2, p3, p4, p5), 
            rois: NxMx5(n, x1, y1, x2, y2), RPN proposals.
            img_width: Input image width.
            img_height: Input image height.

        Returns:
            rois_pooling: NxMxHxW, rois after use RoIAlign.
            
        """
        # Flatten NxMx4 to (NxM)x4
        rois_reshape = rois.view(-1, rois.size(-1))
        bboxes = rois_reshape[:, 1:]
        bbox_indexes = rois_reshape[:, 0]
        rois_pooling = []
        # Iterate bbox to find which level of pyramid features to feed.
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
            rois_pooling.append(roi_pool_per_box)

        rois_pooling = torch.cat(rois_pooling, 0)
        rois_pooling = rois_pooling.view(fpn_features[0].size(0), -1, rois_pooling.size(1),
                                         rois_pooling.size(2))
        return rois_pooling
