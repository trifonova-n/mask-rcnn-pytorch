from backbone.resnet_101_fpn import ResNet_101_FPN
from backbone.resnet_101 import ResNet_101
from head.cls_bbox import ClsBBoxHead_fc as ClsBBoxHead
from head.mask import MaskHead
from tools.utils import calc_iou, coord_corner2center, coord_center2corner
from proposal.rpn import RPN
from pooling.roi_align import RoiAlign

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from configparser import ConfigParser


class MaskRCNN(nn.Module):
    """Mask R-CNN model.
    
    References: https://arxiv.org/pdf/1703.06870.pdf
    
    Notes: In comments below, we assume N: batch size C: number of feature map channel, H: image 
        height, W: image width.

    """

    def __init__(self, num_classes):
        super(MaskRCNN, self).__init__()
        self.config = ConfigParser()
        self.config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.ini"))
        self.num_classes = num_classes
        self.pooling_size_clsbbox = (7, 7)
        self.pooling_size_mask = (14, 14)
        self.use_fpn = False
        self.train_rpn_only = False

        if self.use_fpn:
            self.backbone_fpn = ResNet_101_FPN()
            self.depth = 256
        else:
            self.backbone = ResNet_101()
            self.depth = 1024

        self.rpn = RPN(dim=self.depth, use_fpn=self.use_fpn)

        if not self.train_rpn_only:
            # RoiAlign for cls and bbox head, pooling size 7x7
            self.roi_align_clsbbox = RoiAlign(grid_size=self.pooling_size_clsbbox)
            # RoiAlign for mask head, pooling size 14x14
            self.roi_align_mask = RoiAlign(grid_size=self.pooling_size_mask)
            self.clsbbox_head = ClsBBoxHead(depth=self.depth, pool_size=self.pooling_size_clsbbox,
                                            num_classes=num_classes)
            self.mask_head = MaskHead(depth=self.depth, pool_size=self.pooling_size_mask,
                                      num_classes=num_classes)
        self.img_height = None
        self.img_width = None
        self.batch_size = None

    def forward(self, image, gt_classes=None, gt_bboxes=None, gt_masks=None):
        """
        
        Args:
            image(Variable): image data. [N, C, H, W]  
            gt_classes(Tensor): [N, M], ground truth class ids.
            gt_bboxes(Tensor): [N, M, (x1, y1, x2, y2)], ground truth bounding boxes, coord is 
                left, top, right, bottom.
            gt_masks(Tensor): [N, M, 1, H, W], ground truth masks.
            
        Returns:
            result: list of lists of dict, outer list is mini-batch, inner list is detected objects,
                dict contains stuff below.
                
                dict_key:
                    'proposal': (x1, y1, x2, y2), course bbox from RPN proposal.
                    'cls_pred': predicted class id.
                    'bbox_pred': (x1, y1, x2, y2), refined bbox from prediction head.
                    'mask_pred': [1, 28, 28], predicted mask.
                    
                e.g. result[0][0]['mask_pred'] stands for the first object's mask prediction of
                    the first image of mini-batch.
        """

        self.img_height, self.img_width = image.size(2), image.size(3)
        self.batch_size = image.size(0)
        img_shape = image.data.new(self.batch_size, 2).zero_()
        img_shape[:, 0] = self.img_height
        img_shape[:, 1] = self.img_width
        result, maskrcnn_loss = None, None
        if self.use_fpn:
            p2, p3, p4, p5, p6 = self.backbone_fpn(image)
            # feature maps to feed RPN to generate proposals.
            proposal_features = [p2, p3, p4, p5, p6]
            # feature maps to feed prediction heads to refine bbox and predict class and mask.
            refine_features = [p2, p3, p4, p5]
        else:
            feature_map = self.backbone(image)
            proposal_features = [feature_map]
            refine_features = [feature_map]

        rois, rpn_loss_cls, rpn_loss_bbox = self.rpn(proposal_features, gt_bboxes, img_shape)
        rois = rois.view(-1, 5)  # [N, M, 5] -> [(NxM), 5]

        if self.train_rpn_only:  # only train RPN.
            rpn_loss = None
            if self.training:
                rpn_loss = rpn_loss_cls + rpn_loss_bbox
            else:
                result = self._process_result(self.batch_size, rois)

            return result, rpn_loss
        else:  # train RPN + Predict heads together.
            if self.training:
                assert gt_classes is not None
                assert gt_bboxes is not None
                assert gt_masks is not None
                rois = rois.view(self.batch_size, -1, 5)  # [(NxM), 5] -> [N, M, 5]
                gen_targets = self._generate_targets(rois, gt_classes, gt_bboxes, gt_masks)
                rois_sampled, cls_targets, bbox_targets, mask_targets = gen_targets

                cls_prob, bbox_reg, mask_prob = self._run_predict_head(refine_features,
                                                                       rois_sampled)
                head_loss = MaskRCNN._calc_head_loss(cls_prob, bbox_reg, mask_prob,
                                                     cls_targets, bbox_targets, mask_targets)
                maskrcnn_loss = rpn_loss_cls + rpn_loss_bbox + head_loss
            else:
                cls_prob, bbox_reg, mask_prob = self._run_predict_head(refine_features, rois)
                result = self._process_result(self.batch_size, rois, cls_prob, bbox_reg, mask_prob)

            return result, maskrcnn_loss

    def _run_predict_head(self, refine_features, rois):
        """Run classification, bounding box regression and mask prediction heads.
        
        Args:
            refine_features(list of Variable): 
            rois(Tensor):

        Returns:
            cls_prob:
            bbox_reg:
            mask_prob:
        """
        if self.use_fpn:
            rois_pooling_clsbbox = self._roi_align_fpn(refine_features, rois)
            rois_pooling_mask = self._roi_align_fpn(refine_features, rois)
        else:
            bbox_idx = rois[:, 0]
            bboxes = rois[:, 1:]
            rois_pooling_clsbbox = self.roi_align_clsbbox(refine_features[0],
                                                          Variable(bboxes),
                                                          Variable(bbox_idx.int()))
            rois_pooling_mask = self.roi_align_mask(refine_features[0],
                                                    Variable(bboxes),
                                                    Variable(bbox_idx.int()))
        # run cls, bbox, mask prediction head.
        cls_prob, bbox_reg = self.clsbbox_head(rois_pooling_clsbbox)
        mask_prob = self.mask_head(rois_pooling_mask)

        return cls_prob, bbox_reg, mask_prob

    def _generate_targets(self, proposals, gt_classes, gt_bboxes, gt_masks, mask_size=(28, 28)):
        """Generate Mask R-CNN targets, and corresponding rois.

        Args:
            proposals(Tensor): [N, a, (idx, x1, y1, x2, y2)], proposals from RPN, idx is batch
                size index. 
            gt_classes(Tensor): [N, b], ground truth class ids.
            gt_bboxes(Tensor): [N, b, (x1, y1, x2, y2)], ground truth bounding boxes.
            gt_masks(Tensor): [(N, b, 1, H, W], ground truth masks, H and W for origin image height 
                and width.  

        Returns: 
            sampled_rois(Tensor): [(Nxc), (idx, x1, y1, x2, y2)], proposals after sampled to feed 
                RoIAlign. 
            cls_targets(Variable): [(Nxc)], train targets for classification.
            bbox_targets(Variable): [(Nxc), (dx, dy, dw, dh)], train targets for bounding box 
                regression, see R-CNN paper for meaning details.  
            mask_targets(Variable): [(Nxc), 28, 28], train targets for mask prediction.

        Notes: N: batch_size, a: number of proposals from FRN, b: number of ground truth objects,
            c: number of rois to train.

        """
        rois_sample_size = int(self.config['Train']['rois_sample_size'])
        rois_positive_portion = float(self.config['Train']['rois_positive_portion'])

        batch_size = proposals.size(0)
        # Todo: add support to use batch_size >= 1
        assert batch_size == 1, "batch_size >= 2 will add support later."

        # get rid of batch size dim, need change when support batch_size >= 1.
        proposals = proposals.squeeze(0)
        gt_classes = gt_classes.squeeze(0)
        gt_bboxes = gt_bboxes.squeeze(0)
        gt_masks = gt_masks.squeeze(0)

        iou = calc_iou(proposals[:, 1:], gt_bboxes[:, :])
        max_iou, max_iou_idx_gt = torch.max(iou, dim=1)
        pos_index_prop = torch.nonzero(max_iou >= 0.5).view(-1)
        neg_index_prop = torch.nonzero(max_iou < 0.5).view(-1)

        # if pos_index_prop or neg_index_prop is empty, return an background.
        if pos_index_prop.numel() == 0 or neg_index_prop.numel() == 0:
            cls_targets = gt_classes.new([0])
            bbox_targets = MaskRCNN._get_bbox_targets(proposals[:1, 1:],
                                                      proposals[:1, 1:])
            mask_targets = gt_masks.new(1, mask_size[0], mask_size[1]).zero_()
            sampled_rois = proposals[:1, :]

            return sampled_rois, Variable(cls_targets), Variable(bbox_targets), Variable(
                mask_targets)

        pos_index_gt = max_iou_idx_gt[pos_index_prop]
        assert pos_index_prop.size() == pos_index_gt.size()

        sample_size_pos = int(rois_positive_portion * rois_sample_size)

        pos_num = pos_index_prop.size(0)
        neg_num = neg_index_prop.size(0)
        sample_size_pos = min(sample_size_pos, pos_num)
        # keep the ratio of positive and negative rois, if there are not enough positives.
        sample_size_neg = int((pos_num / rois_positive_portion) * (1 - rois_positive_portion) + 1)
        sample_size_neg = min(sample_size_neg, neg_num)

        sample_index_pos = random.sample(range(pos_num), sample_size_pos)
        sample_index_neg = random.sample(range(neg_num), sample_size_neg)

        pos_index_sampled_prop = pos_index_prop[sample_index_pos]
        neg_index_sampled_prop = neg_index_prop[sample_index_neg]
        pos_index_sampled_gt = pos_index_gt[sample_index_pos]

        index_proposal = torch.cat([pos_index_sampled_prop, neg_index_sampled_prop])
        sampled_rois = proposals[index_proposal, :]

        # targets for classification, positive rois use gt_class id, negative use 0 as background.
        cls_targets_pos = gt_classes[pos_index_sampled_gt]
        cls_targets_neg = gt_classes.new([0 for _ in range(sample_size_neg)])
        cls_targets = torch.cat([cls_targets_pos, cls_targets_neg])

        # bbox regression target define on define on positive proposals.
        bboxes = proposals[:, 1:]
        bbox_targets = MaskRCNN._get_bbox_targets(bboxes[pos_index_sampled_prop, :],
                                                  gt_bboxes[pos_index_sampled_gt, :])
        # mask targets define on positive proposals.
        mask_targets = MaskRCNN._get_mask_targets(bboxes[pos_index_sampled_prop, :],
                                                  gt_masks[pos_index_sampled_gt, :, :], mask_size)

        return sampled_rois, Variable(cls_targets), Variable(bbox_targets), Variable(mask_targets)

    def _roi_align_fpn(self, fpn_features, rois):
        """When use fpn backbone, set RoiAlign use different levels of fpn feature pyramid
            according to RoI size.
         
        Args:
            fpn_features(list of Variable): [p2, p3, p4, p5], 
            rois: NxMx5(n, x1, y1, x2, y2), RPN proposals.

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
            alpha = 224 * min(self.img_height, self.img_width) / 800
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
                roi_pool_per_level = self.roi_align_mask(fpn_features[level], bbox, bbox_idx)
                for idx, batch_idx in enumerate(bbox_idx_levels[level]):
                    rois_pooling_batches[int(batch_idx)].append(roi_pool_per_level[idx])

        rois_pooling = torch.cat([torch.cat(i) for i in rois_pooling_batches])
        rois_pooling = rois_pooling.view(-1, fpn_features[0].size(1), rois_pooling.size(1),
                                         rois_pooling.size(2))

        return rois_pooling

    def _process_result(self, batch_size, proposals, cls_prob=None, bbox_reg=None, mask_prob=None):
        """Process heads output to get the final result.
        Args:
            batch_size(int): mini-batch size.
            proposals(Tensor): [(NxM), (idx, x1, y1, x2, y2)]
            cls_prob(Variable): [(NxM),  num_classes]
            bbox_reg(Variable): [(NxM), num_classes, (x1, y1, x2, y2)]
            mask_prob(Variable): [(NxM), num_classes, 28, 28]
            
        Returns:
            result: list of lists of dict, outer list is mini-batch, inner list is detected objects,
                dict contains stuff below.
                
                dict_key:
                    'proposal': (x1, y1, x2, y2), course bbox from RPN proposal.
                    'cls_pred': predicted class id.
                    'bbox_pred': (x1, y1, x2, y2), refined bbox from prediction head.
                    'mask_pred': [1, 28, 28], predicted mask.
                    
                e.g. result[0][0]['mask_pred'] stands for the first object's mask prediction of
                    the first image of mini-batch.
        """
        # Todo: support batch_size > 1.
        assert batch_size == 1, "batch_size > 1 will support later"

        proposals = proposals.view(batch_size, -1, 5)
        proposals = proposals.squeeze(0)
        num_rois = proposals.size(0)

        result = []

        if self.train_rpn_only:
            obj_detected = []
            for i in range(num_rois):
                pred_dict = {'proposal': proposals[i, 1:]}
                obj_detected.append(pred_dict)
            result.append(obj_detected)

            return result

        else:
            # reshape back to NxM from (NxM)
            cls_prob = cls_prob.view(batch_size, -1, cls_prob.size(1)).data
            bbox_reg = bbox_reg.view(batch_size, -1, bbox_reg.size(1), bbox_reg.size(2)).data
            mask_prob = mask_prob.view(batch_size, -1, mask_prob.size(1), mask_prob.size(2),
                                       mask_prob.size(3)).data

            cls_prob = cls_prob.squeeze(0)
            bbox_reg = bbox_reg.squeeze(0)
            mask_prob = mask_prob.squeeze(0)

            cls_pred = torch.max(cls_prob, 1)[1]
            # remove background and predicted ids.
            keep_index = (cls_pred > 0)
            obj_detected = []
            for i in range(num_rois):
                if int(keep_index[i]):
                    pred_dict = {'proposal': None, 'cls_pred': None, 'bbox_pred': None,
                                 'mask_pred': None}

                    pred_dict['proposal'] = proposals[i, 1:]
                    cls_id = cls_pred[i]
                    pred_dict['cls_pred'] = int(cls_id)

                    bbox_reg_keep = bbox_reg[i, :, :]
                    dx, dy, dw, dh = bbox_reg_keep[cls_id, :].chunk(4)
                    x, y, w, h = coord_corner2center(proposals[i, 1:]).chunk(4)
                    px, py = w * dx + x, h * dy + y
                    pw, ph = w * torch.exp(dw), h * torch.exp(dh)
                    bbox_pred = coord_center2corner(torch.cat([px, py, pw, ph]))

                    px1, py1, px2, py2 = bbox_pred.chunk(4)
                    px1 = int(torch.clamp(px1, max=self.img_width - 1, min=0))
                    px2 = int(torch.clamp(px2, max=self.img_width - 1, min=0))
                    py1 = int(torch.clamp(py1, max=self.img_height - 1, min=0))
                    py2 = int(torch.clamp(py2, max=self.img_height - 1, min=0))
                    mask_height, mask_width = py2 - py1 + 1, px2 - px1 + 1
                    # leave malformed bbox alone, will met this in training.
                    if mask_height == 0 or mask_width == 0 or py1 > py2 or px1 > px2:
                        continue

                    pred_dict['bbox_pred'] = (px1, py1, px2, py2)
                    mask_threshold = float(self.config['Test']['mask_threshold'])
                    mask_prob_keep = mask_prob[i, :, :, :]
                    mask = (mask_prob_keep[cls_id, :, :] >= mask_threshold).float()
                    mask = Variable(mask.unsqueeze(0), requires_grad=False)
                    mask_resize = F.adaptive_avg_pool2d(mask, (mask_height, mask_width)).data
                    mask_pred = mask_prob.new(self.img_height, self.img_width).zero_()
                    mask_pred[py1:py2 + 1, px1:px2 + 1] = mask_resize
                    pred_dict['mask_pred'] = mask_pred.cpu()
                    obj_detected.append(pred_dict)
            result.append(obj_detected)

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
        xy = (gt_bboxes[:, :2] - proposals[:, :2]) / proposals[:, 2:]
        wh = torch.log(gt_bboxes[:, 2:] / proposals[:, 2:])
        x, y = xy.chunk(2, dim=1)
        w, h = wh.chunk(2, dim=1)
        bbox_targets = torch.cat([x, y, w, h], dim=1)

        return bbox_targets

    @staticmethod
    def _get_mask_targets(proposals, gt_masks, mask_size):
        """ Get mask targets, mask target is intersection between proposal and ground
            truth mask, input coord format is (left, top, right, bottom).

        Args:
            proposals(Tensor): [num_rois, 4]
            gt_masks(Tensor): [num_rois, 1, H, W]
            mask_size(tuple): (mask_height, mask_width)
        Returns:
            mask_targets(Tensor): [num_rois, mask_height, mask_width]
        """
        num_rois = proposals.size(0)
        img_height = gt_masks.size(2)
        img_width = gt_masks.size(3)
        mask_targets = gt_masks.new(num_rois, mask_size[0], mask_size[1]).zero_()
        for i in range(num_rois):
            x1, y1, x2, y2 = proposals[i, :]
            x1 = int(max(min(img_width - 1, x1), 0))
            x2 = int(max(min(img_width - 1, x2), 0))
            y1 = int(max(min(img_height - 1, y1), 0))
            y2 = int(max(min(img_height - 1, y2), 0))
            mask = Variable(gt_masks[i, :, y1:y2, x1:x2], requires_grad=False)
            mask_resize = F.adaptive_avg_pool2d(mask, output_size=mask_size)
            mask_targets[i, :, :] = mask_resize.data[0, :, :]

        return mask_targets

    @staticmethod
    def _calc_head_loss(cls_prob, bbox_reg, mask_prob, cls_targets, bbox_targets, mask_targets):
        """ Calculate Mask R-CNN loss.
    
        Args:
            cls_prob(Variable): [(NxS), num_classes], classification predict probability.
            bbox_reg(Variable): [(NxS), num_classes, (dx, dy, dw, dh)], bounding box regression.
            mask_prob(Variable): [(NxS), num_classes, H, W], mask prediction.
            
            cls_targets(Variable): [(NxS)], classification targets.
            bbox_targets(Variable): [(NxPositive), (dx, dy, dw, dh)], bounding box regression targets.
            mask_targets(Variable): [(NxPositive), H, W], mask targets.
    
        Returns:
            maskrcnn_loss: Total loss of Mask R-CNN predict heads.
    
        Notes: In above, S: number of sampled rois feed to prediction heads.
    
        """
        # calculate classification head loss.
        cls_loss = F.nll_loss(cls_prob, cls_targets)
        # cls_pred = torch.max(cls_prob, 1)[1]

        # calculate bbox regression and mask head loss.
        bbox_loss, mask_loss = 0, 0
        # num_all = cls_prob.size(0)
        num_foreground = bbox_targets.size(0)

        for i in range(num_foreground):
            cls_id = int(cls_targets[i])
            # Only corresponding class prediction contribute to bbox and mask loss.
            bbox_loss += F.smooth_l1_loss(bbox_reg[i, cls_id, :], bbox_targets[i, :])
            mask_loss += F.binary_cross_entropy(mask_prob[i, cls_id, :, :], mask_targets[i, :, :])

        bbox_loss /= num_foreground
        mask_loss /= num_foreground

        head_loss = cls_loss + bbox_loss + mask_loss
        return head_loss
