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
