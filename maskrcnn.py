from backbones.resnet_fpn import ResNetFPN
from backbones.resnet import ResNet
from heads.cls_bbox import ClsBBoxHead_fc as ClsBBoxHead
from heads.mask import MaskHead
from tools.detect_utils import calc_iou, bbox_corner2center, bbox_center2corner
from proposal.rpn import RPN
from pooling.roi_align import RoiAlign
from third_party.nms.pth_nms import pth_nms as nms

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from configparser import ConfigParser


class MaskRCNN(nn.Module):
    """Mask R-CNN model.
    
    References: Mask R-CNN: https://arxiv.org/pdf/1703.06870.pdf
    
    Notes: In docstring, N: batch size, M: number of ground-truth objects,
            C: number of feature map channels, H: image height, W: image width.
    """

    def __init__(self, num_classes, pretrained=None):
        """
        
        Args:
            num_classes(int): number of classes, background should be counted in.
                e.g: there are 100 foreground objects, num_classes should be 101.                    
            pretrained(str): 'imagenet' or 'coco', set 'imagenet' indicate just
                backbone use imagenet pretrained weights, 'coco' indicate whole
                Mask R-CNN model use pretrained weights on COCO dataset.
        """

        super(MaskRCNN, self).__init__()
        if pretrained is not None:
            assert pretrained in ['imagenet', 'coco']
            assert pretrained not in ['coco'], "COCO pretrained weights is not available yet."
        self.config = ConfigParser()
        self.config.read(os.path.abspath(os.path.join(__file__, "../", "config.ini")))
        self.num_classes = num_classes
        self.pooling_size_clsbbox = (7, 7)
        self.pooling_size_mask = (14, 14)
        self.validating = False  # when True output loss and predict results.
        use_fpn = bool(int(self.config['BACKBONE']['USE_FPN']))
        self.use_fpn = use_fpn
        self.train_rpn_only = bool(int(self.config['TRAIN']['TRAIN_RPN_ONLY']))
        resnet_layer = int(self.config['BACKBONE']['RESNET_LAYER'])
        if self.use_fpn:
            self.backbone_fpn = ResNetFPN(resnet_layer, pretrained=pretrained)
            self.depth = 256
        else:
            self.backbone = ResNet(resnet_layer, pretrained=pretrained)
            self.depth = 1024

        self.rpn = RPN(self.depth, self.use_fpn)

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
            image(Tensor): [N, C, H, W], image data.  
            gt_classes(Tensor): [N, M], ground truth class ids.
            gt_bboxes(Tensor): [N, M, (x1, y1, x2, y2)], ground truth bounding
                boxes, coord is in format (left, top, right, bottom).
            gt_masks(Tensor): [N, M, H, W], ground truth masks.
            
        Returns:
            result(list of lists of dict): the outer list is mini-batch, the
                inner list is detected objects, the dict contains keys below.
                
            |------------------------------------------------------------------|
            |keys in result dict:                                              |
            |    'proposal': (x1, y1, x2, y2), course bbox from RPN proposal.  |
            |    'cls_pred': predicted class id.                               |
            |    'bbox_pred': (x1, y1, x2, y2), refined bbox from head.        |
            |    'mask_pred': [H, W], predicted mask.                          |                 
            |                                                                  |
            |e.g. result[0][0]['mask_pred'] stands for the first object's mask |
            |    prediction of the first image in mini-batch.                  |
            |------------------------------------------------------------------|
        """

        if not self.training and (gt_classes is not None
                                  and gt_bboxes is not None
                                  and gt_masks is not None):
            self.validating = True
        else:
            self.validating = False

        self._check_input(image, gt_classes, gt_bboxes, gt_masks)

        self.img_height, self.img_width = image.size(2), image.size(3)
        self.batch_size = image.size(0)
        img_shape = image.new(self.batch_size, 2).zero_()
        img_shape[:, 0] = self.img_height
        img_shape[:, 1] = self.img_width
        result, maskrcnn_loss = None, 0
        if self.use_fpn:
            p2, p3, p4, p5, p6 = self.backbone_fpn(Variable(image, requires_grad=False))
            # feature maps to feed RPN to generate proposals.
            rpn_features = [p2, p3, p4, p5, p6]
            # feature maps to feed prediction heads to refine bbox and predict class and mask.
            head_features = [p2, p3, p4, p5]
        else:
            feature_map = self.backbone(Variable(image, requires_grad=False))
            rpn_features = [feature_map]
            head_features = [feature_map]

        rois, rpn_loss_cls, rpn_loss_bbox = self.rpn(rpn_features, gt_bboxes, img_shape)
        if self.train_rpn_only:  # only train RPN.
            result = self._process_result(self.batch_size, head_features, rois)
            rpn_loss = rpn_loss_cls + rpn_loss_bbox

            return result, rpn_loss
        else:  # train RPN + Predict heads together.
            if self.training or self.validating:
                gen_targets = self._generate_targets(rois, gt_classes, gt_bboxes, gt_masks)
                rois_sampled, cls_targets, bbox_targets, mask_targets = gen_targets
                cls_prob, bbox_reg, mask_prob = self._run_predict_head(head_features,
                                                                       rois_sampled)
                head_loss = MaskRCNN._calc_head_loss(cls_prob, bbox_reg, mask_prob,
                                                     cls_targets, bbox_targets, mask_targets)
                maskrcnn_loss = rpn_loss_cls + rpn_loss_bbox + head_loss

            if not self.training:  # valid or test phase
                # rois value will be changed in _run_predict_head(), so make two copy.
                rois_head, rois_result = rois.clone(), rois.clone()
                cls_prob, bbox_reg, _ = self._run_predict_head(head_features, rois_head)
                result = self._process_result(self.batch_size, head_features, rois_result,
                                              cls_prob, bbox_reg)

            return result, maskrcnn_loss

    def _check_input(self, image, gt_classes=None, gt_bboxes=None, gt_masks=None):
        """check model input. 
        """
        assert image.dim() == 4 and image.size(1) == 3

        if self.training or self.validating:
            assert gt_classes.dim() == 2
            assert gt_bboxes.dim() == 3 and gt_bboxes.size(-1) == 4
            assert gt_masks.dim() == 4

    def _run_predict_head(self, features, rois):
        """Run classification, bounding box regression and mask prediction heads.
        
        Args:
            features(list of Variable): extracted features from backbone
            rois(Tensor): [N, M (idx, score, x1, y1, x2, y2)]

        Returns:
            cls_prob(Variable): [(NxM), num_classes]
            bbox_reg(Variable): [(NxM), num_classes, (dx, dy, dw, dh)]
            mask_prob(Variable or None): [(NxM), num_classes, 28, 28] when training, None when 
                testing, in test stage mask head use refined bbox, self._process_result() will 
                handle this.
        """

        mask_prob = None

        rois = rois.view(-1, 6)  # [N, M, 6] -> [(NxM), 6]
        rois_bbox, rois_mask = rois.clone(), rois.clone()
        if self.use_fpn:
            rois_pooling_clsbbox = self._roi_align_fpn(features, rois_bbox, mode='clsbbox')
            cls_prob, bbox_reg = self.clsbbox_head(rois_pooling_clsbbox)
            if self.training or self.validating:
                rois_pooling_mask = self._roi_align_fpn(features, rois_mask, mode='mask')
                mask_prob = self.mask_head(rois_pooling_mask)
        else:
            rois_pooling_clsbbox = self.roi_align_clsbbox(features[0], rois_bbox, self.img_height)
            cls_prob, bbox_reg = self.clsbbox_head(rois_pooling_clsbbox)
            if self.training or self.validating:
                rois_pooling_mask = self.roi_align_mask(features[0], rois_mask, self.img_height)
                mask_prob = self.mask_head(rois_pooling_mask)

        return cls_prob, bbox_reg, mask_prob

    def _generate_targets(self, proposals, gt_classes, gt_bboxes, gt_masks, mask_size=(28, 28)):
        """Generate Mask R-CNN targets, and corresponding rois.

        Args:
            proposals(Tensor): [N, a, (idx, score, x1, y1, x2, y2)], proposals from RPN, 
                idx is batch size index. 
            gt_classes(Tensor): [N, b], ground truth class ids.
            gt_bboxes(Tensor): [N, b, (x1, y1, x2, y2)], ground truth bounding boxes.
            gt_masks(Tensor): [(N, b, H, W], ground truth masks.

        Returns: 
            sampled_rois(Tensor): [N, c, (idx, score, x1, y1, x2, y2)], proposals after sampled to 
                feed RoIAlign. 
            cls_targets(Variable): [(Nxc)], train targets for classification.
            bbox_targets(Variable): [(Nxc), (dx, dy, dw, dh)], train targets for bounding box 
                regression, see R-CNN paper for meaning details.  
            mask_targets(Variable): [(Nxc), 28, 28], train targets for mask prediction.

        Notes: a: number of proposals from FRN, b: number of ground truth objects, c: number 
            of rois to train.
        """

        rois_sample_size = int(self.config['TRAIN']['ROIS_SAMPLE_SIZE'])
        rois_pos_fraction = float(self.config['TRAIN']['ROIS_POS_FRACTION'])
        rois_pos_thresh = float(self.config['TRAIN']['ROIS_POS_THRESH'])
        rois_neg_thresh = float(self.config['TRAIN']['ROIS_NEG_THRESH'])

        batch_size = proposals.size(0)
        # Todo: add support to use batch_size >= 1
        assert batch_size == 1, "batch_size >= 2 will add support later."

        # get rid of batch size dim, need change when support batch_size >= 1.
        proposals = proposals.squeeze(0)
        gt_classes = gt_classes.squeeze(0)
        gt_bboxes = gt_bboxes.squeeze(0)
        gt_masks = gt_masks.squeeze(0)

        iou = calc_iou(proposals[:, 2:], gt_bboxes[:, :])
        max_iou, max_iou_idx_gt = torch.max(iou, dim=1)
        pos_index_prop = torch.nonzero(max_iou >= rois_pos_thresh).view(-1)
        neg_index_prop = torch.nonzero(max_iou < rois_neg_thresh).view(-1)

        if pos_index_prop.numel() == 0:  # no positive roi.
            neg_num = neg_index_prop.size(0)
            sample_size_neg = min(rois_sample_size, neg_num)
            sample_index_neg = random.sample(range(neg_num), sample_size_neg)
            neg_index_sampled_prop = neg_index_prop[sample_index_neg]
            sampled_rois = proposals[neg_index_sampled_prop, :]
            sampled_rois = sampled_rois.view(batch_size, -1, 6)
            cls_targets = gt_classes.new([0 for _ in range(sample_size_neg)])

            return sampled_rois, Variable(cls_targets), None, None

        pos_index_gt = max_iou_idx_gt[pos_index_prop]
        sample_size_pos = int(rois_pos_fraction * rois_sample_size)

        pos_num = pos_index_prop.size(0)
        neg_num = neg_index_prop.size(0)
        sample_size_pos = min(sample_size_pos, pos_num)

        # keep the ratio of positive and negative rois, if there are not enough positives.
        sample_size_neg = int((sample_size_pos / rois_pos_fraction) * (1 - rois_pos_fraction) + 1)
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
        bboxes = proposals[:, 2:]
        bbox_targets = MaskRCNN._get_bbox_targets(bboxes[pos_index_sampled_prop, :],
                                                  gt_bboxes[pos_index_sampled_gt, :])
        # mask targets define on positive proposals.
        mask_targets = MaskRCNN._get_mask_targets(bboxes[pos_index_sampled_prop, :],
                                                  gt_masks[pos_index_sampled_gt, :, :],
                                                  mask_size)
        sampled_rois = sampled_rois.view(batch_size, -1, 6)

        return sampled_rois, Variable(cls_targets), Variable(bbox_targets), Variable(mask_targets)

    def _refine_proposal(self, proposal, bbox_reg):
        """Refine proposal bbox with the result of bbox regression.
        
        Args:
            proposal(Tensor): (x1, y1, x2, y2), bbox proposal from RPN.
            bbox_reg(Tensor): (dx, dy, dw, dh), bbox regression value.

        Returns:
            bbox_refined(Tensor): (x1, y1, x2, y2)
        """

        x, y, w, h = bbox_corner2center(proposal).chunk(4)
        dx, dy, dw, dh = bbox_reg.chunk(4)
        px, py = w * dx + x, h * dy + y
        pw, ph = w * torch.exp(dw), h * torch.exp(dh)
        bbox_refined = bbox_center2corner(torch.cat([px, py, pw, ph]))

        px1, py1, px2, py2 = bbox_refined.chunk(4)
        px1 = torch.clamp(px1, max=self.img_width, min=0)
        px2 = torch.clamp(px2, max=self.img_width, min=0)
        py1 = torch.clamp(py1, max=self.img_height, min=0)
        py2 = torch.clamp(py2, max=self.img_height, min=0)

        bbox_refined = torch.cat([px1, py1, px2, py2])

        return bbox_refined

    def _roi_align_fpn(self, fpn_features, rois, mode):
        """When use fpn backbone, set RoiAlign use different levels of fpn feature pyramid
            according to RoI size.
         
        Args:
            fpn_features(list of Variable): [p2, p3, p4, p5]], 
            rois(Tensor): [(NxM), (n, score, x1, y1, x2, y2)], RPN proposals.
            mode(str): 'clsbbox': roi_align for cls and bbox head, 'mask': roi_align for mask head. 
        Returns:
            rois_pooling: [(NxM), C, pool_size, pool_size], rois after use RoIAlign.
            
        """
        assert mode in ['clsbbox', 'mask']

        rois_levels = [[] for _ in range(len(fpn_features))]
        rois_pool_result = []
        # iterate bbox to find which level of pyramid features to feed.
        for roi in rois:
            bbox = roi[2:]
            # in feature pyramid network paper, alpha is 224 and image short side 800 pixels,
            # for using of small image input, like maybe short side 256, here alpha is
            # parameterized by image short side size.
            alpha = 224 * min(self.img_height, self.img_width) / 800
            bbox_width = torch.abs(rois.new([bbox[2] - bbox[0]]).float())
            bbox_height = torch.abs(rois.new([bbox[3] - bbox[1]]).float())
            log2 = torch.log(torch.sqrt(bbox_height * bbox_width) / alpha) / torch.log(
                rois.new([2]).float())
            level = torch.floor(4 + log2) - 2  # 4 stands for C4, minus 2 to make level 0 indexed
            # rois small or big enough may get level below 0 or above 3.
            level = int(torch.clamp(level, 0, 3))
            roi.unsqueeze_(0)
            rois_levels[level].append(roi)

        for level in range(len(fpn_features)):
            if len(rois_levels[level]) != 0:
                if mode == 'clsbbox':
                    roi_pool_per_level = self.roi_align_clsbbox(fpn_features[level],
                                                                torch.cat(rois_levels[level]),
                                                                self.img_height)
                else:
                    roi_pool_per_level = self.roi_align_mask(fpn_features[level],
                                                             torch.cat(rois_levels[level]),
                                                             self.img_height)
                rois_pool_result.append(roi_pool_per_level)
        rois_pooling = torch.cat(rois_pool_result)

        return rois_pooling

    def _bbox_nms(self, refined_props_nms):
        """Non-maximum suppression on each class of refined proposals.
        
        Args:
            refined_props_nms(Tensor): [N, M, (cls, x1, y1, x2, y2, cls_score)].

        Returns:
            keep_idx(LongTensor): keep index after NMS for final bounding box output.
        """
        assert refined_props_nms.size(0) == 1, "batch size >=2 is not supported yet."
        refined_props_nms.squeeze_(0)

        nms_thresh = float(self.config['TEST']['NMS_THRESH'])
        props_indexed = {}  # indexed by class
        # record position in input refined_props
        for pos, prop in enumerate(refined_props_nms):
            props_indexed.setdefault(prop[0], []).append((pos, prop[1:]))

        keep_idx = []
        for cls, pos_props in props_indexed.items():
            pos = [i[0] for i in pos_props]
            prop = [i[1].unsqueeze(0) for i in pos_props]
            pos = refined_props_nms.new(pos).long()
            prop = torch.cat(prop)
            score = prop[:, 4]
            order = torch.sort(score, dim=0, descending=True)[1]
            pos_ordered = pos[order]
            prop_ordered = prop[order]
            keep_idx_per_cls = nms(prop_ordered, nms_thresh)
            keep_idx.append(pos_ordered[keep_idx_per_cls])

        keep_idx = torch.cat(keep_idx).long()

        return keep_idx

    def _process_result(self, batch_size, features, proposals, cls_prob=None, bbox_reg=None):
        """Get the final result in test stage.
        Args:
            batch_size(int): mini-batch size.
            features(list of Variable): extracted features from backbone
            proposals(Tensor): [N, M, (idx, score, x1, y1, x2, y2)]
            cls_prob(Variable): [(NxM),  num_classes]
            bbox_reg(Variable): [(NxM), num_classes, (x1, y1, x2, y2)]
            
        Returns:
            result: list of lists of dict, outer list is mini-batch, inner list is detected objects,
                dict contains stuff below.
                
                dict_key:
                    'proposal'(Tensor): (x1, y1, x2, y2), course bbox from RPN proposal.
                    'cls_pred'(int): predicted class id.
                    'bbox_pred'(Tensor): (x1, y1, x2, y2), refined bbox from prediction head.
                    'mask_pred'(Tensor): [H, W], predicted mask.
                    
                e.g. result[0][0]['mask_pred'] stands for the first object's mask prediction of
                    the first image of mini-batch.
        """

        # Todo: support batch_size > 1.
        assert batch_size == 1, "batch_size > 1 will add support later"
        proposals = proposals.squeeze(0)
        result = []

        if self.train_rpn_only:
            obj_detected = []
            for i in range(proposals.size(0)):
                pred_dict = {'proposal': proposals[i, 2:].cpu()}
                obj_detected.append(pred_dict)
            result.append(obj_detected)

            return result
        else:
            props = []
            bboxes = []
            cls_scores = []
            cls_ids = []
            for idx, roi in enumerate(proposals):
                cls_score, cls_id = torch.max(cls_prob[idx], dim=0)
                if int(cls_id) > 0:  # remove background
                    # refine proposal bbox with bbox regression result.
                    bbox = self._refine_proposal(roi[2:],
                                                 bbox_reg[idx, :, :][cls_id, :].squeeze(0).data)
                    px1, py1, px2, py2 = bbox
                    # leave malformed bbox alone
                    if py1 >= py2 or px1 >= px2:
                        continue
                    props.append(roi.unsqueeze(0))
                    bboxes.append(bbox.unsqueeze(0))
                    # class score is in log space, turn it to range in 0~1
                    cls_scores.append(torch.exp(cls_score.data))
                    cls_ids.append(cls_id.data)

            if len(props) != 0:
                cls_ids = torch.cat(cls_ids)
                props_origin = torch.cat(props)
                props_refined = props_origin.clone()
                props_refined[:, 2:] = torch.cat(bboxes)

                props_refined_nms = props_origin.new(len(cls_ids), 6)
                props_refined_nms[:, 0] = cls_ids
                props_refined_nms[:, 1:5] = torch.cat(bboxes)
                props_refined_nms[:, 5] = torch.cat(cls_scores)
            else:
                result.append([])

                return result

            props_refined_nms = props_refined_nms.unsqueeze(0)  # dummy batch size == 1
            keep_idx = self._bbox_nms(props_refined_nms)
            cls_ids = cls_ids[keep_idx]
            props_origin = props_origin[keep_idx]
            props_refined = props_refined[keep_idx]

            if self.use_fpn:
                rois_pooling_mask = self._roi_align_fpn(features, props_refined.clone(),
                                                        mode='mask')
                mask_prob = self.mask_head(rois_pooling_mask).data
            else:
                rois_pooling_mask = self.roi_align_mask(features[0], props_refined.clone(),
                                                        self.img_height)
                mask_prob = self.mask_head(rois_pooling_mask).data

            obj_detected = []
            for i in range(len(props_origin)):
                pred_dict = {'proposal': props_origin[i, 2:].cpu(), 'cls_pred': cls_ids[i],
                             'bbox_pred': props_refined[i, 2:].cpu(), 'mask_pred': None}

                px1, py1, px2, py2 = props_refined[i, 2:].int()
                mask_height, mask_width = py2 - py1, px2 - px1
                mask = mask_prob[i, :, :, :][cls_ids[i], :, :]
                mask = Variable(mask.unsqueeze(0), requires_grad=False)
                mask_resize = F.adaptive_avg_pool2d(mask, (mask_height, mask_width)).data
                mask_threshold = float(self.config['TEST']['MASK_THRESH'])
                mask_resize = mask_resize >= mask_threshold
                mask_pred = mask_prob.new(self.img_height, self.img_width).zero_()
                mask_pred[py1:py2, px1:px2] = mask_resize
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
        proposals = bbox_corner2center(proposals)
        gt_bboxes = bbox_corner2center(gt_bboxes)
        xy = (gt_bboxes[:, :2] - proposals[:, :2]) / proposals[:, 2:]
        wh = torch.log(gt_bboxes[:, 2:] / proposals[:, 2:])
        x, y = xy.chunk(2, dim=1)
        w, h = wh.chunk(2, dim=1)
        bbox_targets = torch.cat([x, y, w, h], dim=1)

        return bbox_targets

    @staticmethod
    def _get_mask_targets(proposals, gt_masks, mask_size):
        """Get mask targets, mask target is intersection between proposal and ground
            truth mask, input coord format is (left, top, right, bottom).

        Args:
            proposals(Tensor): [num_rois, 4]
            gt_masks(Tensor): [N, num_rois, H, W]
            mask_size(tuple): (mask_height, mask_width)
        Returns:
            mask_targets(Tensor): [num_rois, mask_height, mask_width]
        """
        num_rois = proposals.size(0)
        img_height = gt_masks.size(1)
        img_width = gt_masks.size(2)
        mask_targets = gt_masks.new(num_rois, mask_size[0], mask_size[1]).zero_()

        for i in range(num_rois):
            x1, y1, x2, y2 = proposals[i, :]
            x1 = int(max(min(img_width, x1), 0))
            x2 = int(max(min(img_width, x2), 0))
            y1 = int(max(min(img_height, y1), 0))
            y2 = int(max(min(img_height, y2), 0))
            mask = Variable(gt_masks[i, y1:y2, x1:x2], requires_grad=False)
            # mask.unsqueeze(0) work around F.adaptive_avg_pool2d silent crash.
            mask_resize = F.adaptive_avg_pool2d(mask.unsqueeze(0), output_size=mask_size)
            mask_targets[i, :, :] = mask_resize.data[0]

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

        # calculate bbox regression and mask head loss.
        bbox_loss, mask_loss = 0, 0
        if bbox_targets is not None and mask_targets is not None:
            num_foreground = bbox_targets.size(0)
            for i in range(num_foreground):
                cls_id = int(cls_targets[i])
                # Only corresponding class prediction contribute to bbox and mask loss.
                bbox_loss += F.smooth_l1_loss(bbox_reg[i, cls_id, :], bbox_targets[i, :])
                mask_loss += F.binary_cross_entropy(mask_prob[i, cls_id, :, :],
                                                    mask_targets[i, :, :])
            bbox_loss /= num_foreground
            mask_loss /= num_foreground

        head_loss = cls_loss + bbox_loss + mask_loss

        return head_loss
