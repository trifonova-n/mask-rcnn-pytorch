from backbone.resnet_101_fpn import ResNet_101_FPN
from head.cls_bbox import ClsBBoxHead_fc as ClsBBoxHead
from head.mask import MaskHead
from tools.utils import calc_iou, coord_corner2center, coord_center2corner

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from configparser import ConfigParser

if not __debug__:
    # when not doing unittest import below module.
    from proposal.rpn import RPN
    from pooling.roi_align import RoiAlign


# TODO: optimize GPU memory consumption

class MaskRCNN(nn.Module):
    """Mask R-CNN model.
    
    References: https://arxiv.org/pdf/1703.06870.pdf
    
    Notes: In comments below, we assume N: batch size C: feature map channel, H: image height, 
        W: image width.

    """

    def __init__(self, num_classes):
        super(MaskRCNN, self).__init__()
        self.config = ConfigParser()
        self.config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.ini"))
        self.pooling_size = (14, 14)
        self.depth = 256
        self.num_classes = num_classes
        self.fpn = ResNet_101_FPN()
        self.rpn = RPN(dim=self.depth)
        self.roi_align = RoiAlign(grid_size=self.pooling_size)
        self.cls_box_head = ClsBBoxHead(depth=self.depth, pool_size=self.pooling_size,
                                        num_classes=num_classes)
        self.mask_head = MaskHead(depth=self.depth, pool_size=self.pooling_size,
                                  num_classes=num_classes)

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
        self.image_size = (x.size(2), x.size(3))
        p2, p3, p4, p5, p6 = self.fpn(x)
        rpn_features_rpn = [p2, p3, p4, p5, p6]
        fpn_features = [p2, p3, p4, p5]
        img_shape = x.data.new(x.size(0), 2).zero_()
        img_shape[:, 0] = x.size(2)
        img_shape[:, 1] = x.size(3)
        rois, rpn_loss_cls, rpn_loss_bbox = self.rpn(rpn_features_rpn, gt_bboxes, img_shape)

        cls_targets, bbox_targets, mask_targets = None, None, None
        if self.training:
            assert gt_classes is not None
            assert gt_bboxes is not None
            assert gt_masks is not None
            gen_result = self._generate_targets(rois, gt_classes, gt_bboxes, gt_masks)
            rois, cls_targets, bbox_targets, mask_targets = gen_result

        rois_pooling = self._roi_align_fpn(fpn_features, rois, x.size(2), x.size(3))
        cls_prob, bbox_reg = self.cls_box_head(rois_pooling)
        mask_prob = self.mask_head(rois_pooling)

        if self.training:
            # reshape back to (NxM) from NxM
            cls_targets = cls_targets.view(-1)
            bbox_targets = bbox_targets.view(-1, bbox_targets.size(2))
            mask_targets = mask_targets.view(-1, mask_targets.size(2), mask_targets.size(3))
            maskrcnn_loss = MaskRCNN._calc_maskrcnn_loss(cls_prob, bbox_reg, mask_prob, cls_targets,
                                                         bbox_targets, mask_targets)
            loss = rpn_loss_cls + rpn_loss_bbox + maskrcnn_loss
            return loss
        else:
            result = self._process_result(x.size(0), rois, cls_prob, bbox_reg, mask_prob)
            return result

    def _generate_targets(self, proposals, gt_classes, gt_bboxes, gt_masks, mask_size=(28, 28)):
        """ Generate Mask R-CNN targets, and corresponding rois.

        Args:
            proposals(Variable): [N, a, (idx, x1, y1, x2, y2)], proposals from RPN, idx is batch
                size index. 
            gt_classes(Tensor): [N, b], ground truth class ids.
            gt_bboxes(Tensor): [N, b, (x1, y1, x2, y2)], ground truth bounding boxes.
            gt_masks(Tensor): [N, b, H, W], ground truth masks, H and W for origin image height 
                and width.  

        Returns: 
            rois(Variable): [N, c, (idx, x1, y1, x2, y2)], proposals after process to feed 
              RoIAlign. 
            cls_targets(Variable): [N, c], train targets for classification.
            bbox_targets(Variable): [N, c, (dx, dy, dw, dh)], train targets for bounding box 
                regression, see R-CNN paper for meaning details.  
            mask_targets(Variable): [N, c, 28, 28], train targets for mask prediction.

        Notes: N: batch_size, a: number of proposals from FRN, b: number of ground truth objects,
            c: number of rois to train.

        """
        train_rois_num = int(self.config['Train']['train_rois_num'])
        batch_size = proposals.size(0)

        # Todo: add support to use batch_size >= 1
        assert batch_size == 1, "batch_size >= 2 will add support later."

        proposals = proposals.squeeze(0)
        gt_classes = gt_classes.squeeze(0)
        gt_bboxes = gt_bboxes.squeeze(0)
        gt_masks = gt_masks.squeeze(0)

        iou = calc_iou(proposals[:, 1:], gt_bboxes[:, :])
        neg_index = torch.nonzero(iou < 0.5)
        pos_index = torch.nonzero(iou >= 0.5)
        # check if neg_index is empty
        neg_num = neg_index.size(0) if neg_index.size() != torch.LongTensor([]).size() else 0
        pos_num = pos_index.size(0) if pos_index.size() != torch.LongTensor([]).size() else 0
        sample_size_neg = int(0.75 * train_rois_num)
        sample_size_pos = train_rois_num - sample_size_neg
        sample_size_neg = sample_size_neg if sample_size_neg <= neg_num else neg_num
        sample_size_pos = sample_size_pos if sample_size_pos <= pos_num else pos_num
        sample_index_neg = random.sample(range(neg_num), sample_size_neg)
        sample_index_pos = random.sample(range(pos_num), sample_size_pos)
        neg_index_sampled = neg_index[sample_index_neg, :]

        # if there is no positive iou, take some of negative.
        if sample_size_pos != 0:
            pos_index_sampled = pos_index[sample_index_pos, :]
        else:
            pos_index_sampled = neg_index[0:2, :]

        neg_index_proposal = neg_index_sampled[:, 0]
        pos_index_proposal = pos_index_sampled[:, 0]
        neg_index_gt = neg_index_sampled[:, 1]
        pos_index_gt = pos_index_sampled[:, 1]
        index_gt = torch.cat([neg_index_gt, pos_index_gt])

        rois = torch.cat([proposals[neg_index_proposal, :], proposals[pos_index_proposal, :]])
        cls_targets = gt_classes[index_gt]
        proposals = proposals[:, 1:]
        bbox_targets_neg = MaskRCNN._get_bbox_targets(proposals[neg_index_proposal, :],
                                                      gt_bboxes[neg_index_gt, :])
        bbox_targets_pos = MaskRCNN._get_bbox_targets(proposals[pos_index_proposal, :],
                                                      gt_bboxes[pos_index_gt, :])
        bbox_targets = torch.cat([bbox_targets_neg, bbox_targets_pos])
        # mask targets define on positive proposals.
        mask_targets = MaskRCNN._get_mask_targets(proposals[pos_index_proposal, :],
                                                  gt_masks[pos_index_gt, :, :], mask_size)
        rois = rois.unsqueeze(0)
        cls_targets = cls_targets.unsqueeze(0)
        bbox_targets = bbox_targets.unsqueeze(0)
        mask_targets = mask_targets.unsqueeze(0)

        return rois, Variable(cls_targets), Variable(bbox_targets), Variable(mask_targets)

    def _roi_align_fpn(self, fpn_features, rois, img_width, img_height):
        """When use fpn backbone, set RoiAlign use different levels of fpn feature pyramid
            according to RoI size.
         
        Args:
            fpn_features: (p2, p3, p4, p5), 
            rois: NxMx5(n, x1, y1, x2, y2), RPN proposals.
            img_width: Input image width.
            img_height: Input image height.

        Returns:
            rois_pooling: (NxM)xCxHxW, rois after use RoIAlign.
            
        """
        # Flatten NxMx4 to (NxM)x4
        rois_reshape = rois.view(-1, rois.size(-1))
        bboxes = rois_reshape[:, 1:]
        bbox_indexes = rois_reshape[:, 0]
        rois_pooling_batches = [[] for _ in range(rois.size(0))]
        bbox_levels = [[] for _ in range(len(fpn_features))]
        bbox_idx_levels = [[] for _ in range(len(fpn_features))]
        # iterate bbox to find which level of pyramid features to feed.
        for idx, bbox in enumerate(bboxes):
            # in feature pyramid network paper, alpha is 224 and image short side 800 pixels,
            # for using of small image input, like maybe short side 256, here alpha is
            # parameterized by image short side size.
            alpha = 224 * (img_width if img_width <= img_height else img_height) / 800
            bbox_width = torch.abs(rois.new([bbox[0] - bbox[2]]).float())
            bbox_height = torch.abs(rois.new([bbox[1] - bbox[3]]).float())
            log2 = torch.log(torch.sqrt(bbox_height * bbox_width)) / torch.log(
                rois.new([2]).float()) / alpha
            level = torch.floor(4 + log2) - 2  # minus 2 to make level 0 indexed
            # rois small or big enough may get level below 0 or above 3.
            level = int(torch.clamp(level, 0, 3))
            bbox = bbox.type_as(bboxes).unsqueeze(0)
            bbox_idx = rois.new([bbox_indexes[idx]]).int()
            bbox_levels[level].append(bbox)
            bbox_idx_levels[level].append(bbox_idx)
        for level in range(len(fpn_features)):
            if len(bbox_levels[level]) != 0:
                bbox = Variable(torch.cat(bbox_levels[level]))
                bbox_idx = Variable(torch.cat(bbox_idx_levels[level]))
                roi_pool_per_level = self.roi_align(fpn_features[level], bbox, bbox_idx)
                for idx, batch_idx in enumerate(bbox_idx_levels[level]):
                    rois_pooling_batches[int(batch_idx)].append(roi_pool_per_level[idx])

        rois_pooling = torch.cat([torch.cat(i) for i in rois_pooling_batches])
        rois_pooling = rois_pooling.view(-1, fpn_features[0].size(1), rois_pooling.size(1),
                                         rois_pooling.size(2))
        return rois_pooling

    def _process_result(self, batch_size, proposals, cls_prob, bbox_reg, mask_prob):
        """ Process heads output to get the final result.
        Args:
            batch_size:
            proposals: [(NxM), (x1, y1, x2, y2)]
            cls_prob: [(NxM),  num_classes]
            bbox_reg: [(NxM), num_classes, (x1, y1, x2, y2)]
            mask_prob: [(NxM), num_classes, 28, 28]
        """

        result = []
        # reshape back to NxM from (NxM)
        cls_prob = cls_prob.view(batch_size, -1, cls_prob.size(1))
        bbox_reg = bbox_reg.view(batch_size, -1, bbox_reg.size(1), bbox_reg.size(2))
        mask_prob = mask_prob.view(batch_size, -1, mask_prob.size(1), mask_prob.size(2),
                                   mask_prob.size(3))
        cls_id_prob, cls_id = torch.max(cls_prob, 2)
        # remove background and predicted ids whose probability below threshold.
        keep_index = (cls_id > 0)
        # Todo: support batch_size > 1.
        assert batch_size == 1, "batch_size > 1 will support later"

        keep_index = keep_index.squeeze(0)
        proposals = proposals.squeeze(0)
        cls_prob = cls_prob.squeeze(0)
        bbox_reg = bbox_reg.squeeze(0)
        mask_prob = mask_prob.squeeze(0)

        objects = []
        for i in range(cls_prob.size(1)):
            pred_dict = {'cls_pred': None, 'bbox_pred': None, 'mask_pred': None}
            if keep_index[i].all():
                pred_dict['cls_pred'] = cls_id[i]
                print("cls_id[i]: ", cls_id[i].size())
                bbox_reg_per_roi = bbox_reg[i, :, :]
                print("bbox_reg_per_roi :", bbox_reg_per_roi.size())
                print("bbox_reg_per_roi[cls_id[i], :] :", bbox_reg_per_roi[cls_id[i], :].size())
                dx, dy, dw, dh = bbox_reg_per_roi[cls_id[i], :]
                x, y, w, h = coord_corner2center(proposals[i, :])
                px, py = w * dx + x, h * dy + y
                pw, ph = w * torch.exp(dw), h * torch.exp(dh)
                px1, py1, px2, py2 = coord_center2corner((px, py, pw, ph))[0, :]
                pred_dict['bbox_pred'] = (px1, py1, px2, py2)
                mask_threshold = self.config['Test']['mask_threshold']
                mask_prob_per_roi = mask_prob[i, :, :, :]
                mask = Variable(mask_prob_per_roi[cls_id[i], :, :] >= mask_threshold,
                                require_grad=False)
                mask_height, mask_width = py2 - py1, px2 - px1
                mask_resize = F.upsample(mask, (mask_height, mask_width)).data
                mask_pred = mask_resize.new(self.image_size).zero_()
                mask_pred[px1:px2, py1:py2] = mask_resize
                pred_dict['mask_pred'] = mask_pred
            objects.append(pred_dict)
        result.append(objects)

        return result

    @staticmethod
    def _get_bbox_targets(proposals, gt_bboxes):
        """ Calculate bounding box targets, input coord format is (left, top, right, bottom),
            see R-CNN paper for the formula.

        Args:
            proposals(Tensor): [n, 4]
            gt_bboxes(Tensor): [n, 4]

        Returns:
            bbox_targets(Tensor): [n, 4]
        """
        proposals = coord_corner2center(proposals)
        gt_bboxes = coord_corner2center(gt_bboxes)

        x, y = ((gt_bboxes[:, :2] - proposals[:, :2]) / proposals[:, 2:]).chunk(2, dim=1)
        w, h = torch.log(gt_bboxes[:, 2:] / proposals[:, 2:]).chunk(2, dim=1)
        bbox_targets = torch.cat([x, y, w, h], dim=1)

        return bbox_targets

    @staticmethod
    def _get_mask_targets(proposals, gt_masks, mask_size):
        """ Get mask targets, mask target is intersection between proposal and ground
            truth mask, input coord format is (left, top, right, bottom).

        Args:
            proposals(Tensor): [n, 4]
            gt_masks(Tensor): [n, H, W]
            mask_size(tuple): (mask_height, mask_width)
        Returns:
            mask_targets(Tensor): [n, 28, 28]
        """
        n = proposals.size(0)
        mask_targets = gt_masks.new(n, mask_size[0], mask_size[1]).zero_()
        for i in range(n):
            x1, y1, x2, y2 = proposals[i, :]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x1 < x2 and y1 < y2:
                mask = Variable(gt_masks[i, x1:x2, y1:y2].unsqueeze(0), requires_grad=False)
                mask_resize = F.adaptive_avg_pool2d(mask, output_size=mask_size)
                mask_targets[i, :, :] = mask_resize.data

        return mask_targets

    @staticmethod
    def _calc_maskrcnn_loss(cls_prob, bbox_reg, mask_prob, cls_targets, bbox_targets, mask_targets):
        """ Calculate Mask R-CNN loss.
    
        Args:
            cls_prob: (NxS)x num_classes, classification predict probability.
            bbox_reg: (NxS)x num_classes x 4(dx, dy, dw, dh), bounding box regression.
            mask_prob: (NxS)x num_classes x HxW, mask prediction.
            cls_targets: (NxS), classification targets.
            bbox_targets: (NxS)x4(dx, dy, dw, dh), bounding box regression targets.
            mask_targets: (NxS)xHxW, mask targets.
    
        Returns:
            maskrcnn_loss: Total loss of Mask R-CNN predict heads.
    
        Notes: In above, S: number of rois feed to prediction heads.
    
        """
        cls_loss = F.nll_loss(cls_prob, cls_targets)
        _, cls_pred = torch.max(cls_prob, 1)
        # Only predicted class masks contribute to bbox and mask loss.
        bbox_loss, mask_loss = 0, 0
        for i in range(cls_prob.size(0)):
            cls_id = int(cls_pred[i])
            bbox_loss += F.smooth_l1_loss(bbox_reg[i, cls_id, :], bbox_targets[i, :])
        # last part is positive roi, contribute to mask loss.
        for i in range(mask_targets.size(0)):
            start = cls_pred.size(0) - mask_targets.size(0)
            cls_id = int(cls_pred[start + i])
            mask_loss += F.binary_cross_entropy(mask_prob[start + i, cls_id, :, :],
                                                mask_targets[i, :, :])
        maskrcnn_loss = cls_loss + bbox_loss + mask_loss

        return maskrcnn_loss
