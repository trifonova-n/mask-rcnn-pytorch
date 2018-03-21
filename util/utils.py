import torch
from torch.nn import functional as F


def calc_iou(bbox_a, bbox_b):
    """ Calculate IoU(Intersection over union) of two batch of bounding box, bbox coord format
        is (x1, y1, x2, y2), stand for left, top, right, bottom.
    Args:
        bbox_a(Tensor): shape: [a, 4], bounding box batch a.
        bbox_b(Tensor): shape: [b, 4], bounding box batch b.

    Returns:
        iou(Tensor): shape: [a, b], IoU of two bounding boxes.
    """

    a = bbox_a.size(0)
    b = bbox_b.size(0)

    area_a = (bbox_a[:, 2] - bbox_a[:, 0] + 1) * (bbox_a[:, 3] - bbox_a[:, 1] + 1)
    area_a = area_a.unsqueeze(1).expand(a, b)  # [a, b]
    area_b = (bbox_b[:, 2] - bbox_b[:, 0] + 1) * (bbox_b[:, 3] - bbox_b[:, 1] + 1)
    area_b = area_b.unsqueeze(0).expand(a, b)  # [a, b]

    bbox_a = bbox_a.unsqueeze(1).expand(a, b, 4)  # [a, b, 4]
    bbox_b = bbox_b.unsqueeze(0).expand(a, b, 4)  # [a, b, 4]
    inter_x1_y1 = torch.max(bbox_a[:, :, :2], bbox_b[:, :, :2])
    inter_x2_y2 = torch.min(bbox_a[:, :, 2:], bbox_b[:, :, 2:])
    inter_w_h = torch.clamp(inter_x2_y2 - inter_x1_y1, min=0)

    intersection = inter_w_h[:, :, 0] * inter_w_h[:, :, 1]  # [a, b] width * height
    iou = intersection / (area_a + area_b - intersection)

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
    """ Transform corner format coord (x1, y1, x2, y2) to center format (x, y, w, h). 
        (x1, y1, x2, y2) stands for left, top, right, bottom, (x, y, w, h) stands for
        center x, center y, width, height.
        
    Args:
        bbox(Tensor): shape: [n, 4] 

    Returns: 
        bbox_trans(Tensor): shape: [n, 4]
        
    """
    x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    x = torch.floor((x2 - x1 + 1) / 2.0) + x1
    y = torch.floor((y2 - y1 + 1) / 2.0) + y1

    w = (x2 - x1) + 1
    h = (y2 - y1) + 1

    x.unsqueeze_(1), y.unsqueeze_(1), w.unsqueeze_(1), h.unsqueeze_(1)
    bbox_trans = torch.cat([x, y, w, h], dim=1)
    return bbox_trans


def coord_center2corner(bbox):
    """ Transform center format (x, y, w, h) to corner format coord (x1, y1, x2, y2). 
            (x1, y1, x2, y2) stands for left, top, right, bottom, (x, y, w, h) stands for
            center x, center y, width, height.

    Args:
        bbox(Tensor): shape: [n, 4] 

    Returns: 
        bbox_trans(Tensor): shape: [n, 4]

    """
    x, y, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    x1 = x - torch.floor(w / 2)
    y1 = y - torch.floor(h / 2)
    x2 = x + torch.floor(w / 2)
    y2 = y + torch.floor(h / 2)

    x1.unsqueeze_(1), y1.unsqueeze_(1), x2.unsqueeze_(1), y2.unsqueeze_(1)

    bbox_trans = torch.cat([x1, y1, x2, y2], dim=1)

    return bbox_trans
