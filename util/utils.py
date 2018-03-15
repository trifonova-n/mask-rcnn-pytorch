import torch
from torch.nn import functional as F


def calc_iou(bbox1, bbox2):
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


def calc_maskrcnn_loss(cls_prob, bbox_reg, mask_prob, cls_targets, bbox_targets,
                       mask_targets):
    """ Calculate Mask R-CNN loss.

    Args:
        cls_prob: NxSxNum_classes, classification predict probability.
        bbox_reg: NxSxNum_classes*4(dx, dy, dw, dh), bounding box regression.
        mask_prob: NxSxNUM_CLASSESxHxW, mask prediction.
        cls_targets: NxSxNum_classes, classification targets.
        bbox_targets: NxSxNum_classes*4(dx, dy, dw, dh), bounding box regression targets.
        mask_targets: NxSxHxW, mask targets.

    Returns:
        maskrcnn_loss: Total loss of Mask R-CNN predict heads.

    Notes: In above, S: number of rois feed to prediction heads.

    """
    cls_loss = F.nll_loss(cls_prob, cls_targets)
    bbox_loss = F.smooth_l1_loss(bbox_reg, bbox_targets)
    _, cls_pred = torch.max(cls_prob, 2)
    # Only predicted class masks contribute to mask loss.
    mask_loss = 0
    for i in range(cls_prob.size(0)):
        for j in range(cls_prob.size(1)):
            cls_id = cls_pred[i, j]
            mask_loss += F.binary_cross_entropy(mask_prob[i, j, cls_id, :, :],
                                                mask_targets[i, j, :, :])
    maskrcnn_loss = cls_loss + bbox_loss + mask_loss
    return maskrcnn_loss


def coord_corner2center(bbox):
    """
    
    Args:
        bbox: (x1, y1, x2, y2), bounding box in corner coord, (x1, y1) stands for bbox 
            top-left, (x2, y2) stands for bbox bottom-right.

    Returns: (x, y, w, h), bounding box in center coord, (x, y) stands for bbox center,
        (w, h) stands for bbox width and height.
    """
    x1, y1, x2, y2 = bbox
    x = (x2 - x1 + 1) // 2 + x1
    y = (y2 - y1 + 1) // 2 + y1
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    return x, y, w, h


def coord_center2corner():
    pass
