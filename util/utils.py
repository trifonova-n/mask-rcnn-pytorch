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

    inter_x1 = torch.max(torch.cat([bbox1.new([bbox1[0]]), bbox1.new([bbox2[0]])]))
    inter_x2 = torch.max(torch.cat([bbox1.new([bbox1[1]]), bbox1.new([bbox2[1]])]))
    inter_y1 = torch.max(torch.cat([bbox1.new([bbox1[2]]), bbox1.new([bbox2[2]])]))
    inter_y2 = torch.max(torch.cat([bbox1.new([bbox1[3]]), bbox1.new([bbox2[3]])]))

    inter_w = torch.max(torch.cat([bbox1.new([0]), bbox1.new([inter_x2 - inter_x1 + 1])]))
    inter_h = torch.max(torch.cat([bbox1.new([0]), bbox1.new([inter_y2 - inter_y1 + 1])]))

    intersection = inter_w * inter_h
    iou = intersection / (area_1 + area_2 + intersection)

    return iou


def calc_maskrcnn_loss(cls_prob, bbox_reg, mask_prob, cls_targets, bbox_targets,
                       mask_targets):
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


def coord_corner2center(bbox):
    """ Transform corner style coord (x1, y1, x2, y2) to center style (x, y, w, h). 
    
    Args:
        bbox: (x1, y1, x2, y2), bounding box in corner coord, (x1, y1) stands for bbox 
            top-left, (x2, y2) stands for bbox bottom-right.

    Returns: (x, y, w, h), bounding box in center coord, (x, y) stands for bbox center,
        (w, h) stands for bbox width and height.
    """
    x1, y1 = bbox.new([bbox[0]]), bbox.new([bbox[1]])
    x2, y2 = bbox.new([bbox[2]]), bbox.new([bbox[3]])
    x = torch.floor((x2 - x1 + 1) / 2) + x1
    y = torch.floor((y2 - y1 + 1) / 2) + y1
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    return x, y, w, h


def coord_center2corner(bbox):
    """ Transform center style coord (x, y, w, h) to corner style (x1, y1, x2, y2). 

    Args:
        bbox: (x, y, w, h), bounding box in center coord, (x, y) stands for bbox center,
        (w, h) stands for bbox width and height.

    Returns: 
        bbox: (x1, y1, x2, y2), bounding box in corner coord, (x1, y1) stands for bbox 
            top-left, (x2, y2) stands for bbox bottom-right.
    """

    x, y = bbox.new([bbox[0]]), bbox.new([bbox[1]])
    w, h = bbox.new([bbox[2]]), bbox.new([bbox[3]])
    x1 = x - torch.floor(w / 2)
    y1 = y - torch.floor(h / 2)
    x2 = x + torch.floor(w / 2)
    y2 = y + torch.floor(h / 2)

    return x1, y1, x2, y2
