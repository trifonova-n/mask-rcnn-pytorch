"""
---------------------------------------------------------
Useful PyTorch utils for CV detection task.

Copyright (c) 2018 Geeshang Xu
Licensed under the MIT License (see LICENSE for details)
---------------------------------------------------------

In code below:

coordinate corner format: 
    (left, top, right, bottom) == (x1, y1, x2, y2), x2 and y2 are excluded.
coordinate center format:
    (center x, center y, width, height) == (x, y, w, h)
coordinate coco format:
    (left, top, width, height) == (x1, y1, w, h)

"""

import torch


def calc_iou(bbox_a, bbox_b):
    """Calculate IoU(Intersection over union) of two batch of bounding box
        efficiently by using vectorized operations, bbox coord format is
        in (left, top, right, bottom).
    Args:
        bbox_a(Tensor): shape: [a, 4], bounding box batch a.
        bbox_b(Tensor): shape: [b, 4], bounding box batch b.

    Returns:
        iou(Tensor): shape: [a, b], IoU of two bounding boxes.    
            
    """

    _check_bbox_input(bbox_a)
    _check_bbox_input(bbox_b)

    a = bbox_a.size(0)
    b = bbox_b.size(0)

    area_a = (bbox_a[:, 2] - bbox_a[:, 0]) * (bbox_a[:, 3] - bbox_a[:, 1])
    area_a = area_a.unsqueeze(1).expand(a, b)  # [a, b]
    area_b = (bbox_b[:, 2] - bbox_b[:, 0]) * (bbox_b[:, 3] - bbox_b[:, 1])
    area_b = area_b.unsqueeze(0).expand(a, b)  # [a, b]

    bbox_a = bbox_a.unsqueeze(1).expand(a, b, 4)  # [a, b, 4]
    bbox_b = bbox_b.unsqueeze(0).expand(a, b, 4)  # [a, b, 4]
    inter_x1_y1 = torch.max(bbox_a[:, :, :2], bbox_b[:, :, :2])
    inter_x2_y2 = torch.min(bbox_a[:, :, 2:], bbox_b[:, :, 2:])
    inter_w_h = torch.clamp(inter_x2_y2 - inter_x1_y1, min=0)

    intersection = inter_w_h[:, :, 0] * inter_w_h[:, :, 1]  # [a, b] width * height
    iou = intersection / (area_a + area_b - intersection)

    return iou


def bbox_corner2center(bbox):
    """Transform bounding box in corner format (left, top, right, bottom) to center
        format (center x, center y, width, height).
        
    Args:
        bbox(Tensor): [*, 4], where * means, any number of additional dimensions.

    Returns: 
        bbox_trans(Tensor): [*, 4], same shape as the input.
        
    """
    _check_bbox_input(bbox)

    cat_dim = bbox.dim() - 1
    x1, y1, x2, y2 = bbox.chunk(4, dim=cat_dim)

    x = torch.floor((x2 - x1) / 2) + x1
    y = torch.floor((y2 - y1) / 2) + y1

    w = x2 - x1
    h = y2 - y1

    bbox_trans = torch.cat([x, y, w, h], dim=cat_dim)

    return bbox_trans


def bbox_center2corner(bbox):
    """Transform bounding box in center format (center x, center y, width, height) 
        to corner format (left, top, right, bottom). 

    Args:
        bbox(Tensor): [*, 4], where * means, any number of additional dimensions.

    Returns: 
        bbox_trans(Tensor): [*, 4], same shape as the input.

    """

    _check_bbox_input(bbox)

    cat_dim = bbox.dim() - 1
    x, y, w, h = bbox.chunk(4, dim=cat_dim)

    x1 = x - torch.floor(w / 2)
    y1 = y - torch.floor(h / 2)
    x2 = x + torch.floor(w / 2)
    y2 = y + torch.floor(h / 2)

    bbox_trans = torch.cat([x1, y1, x2, y2], dim=cat_dim)

    return bbox_trans


def bbox_coco2corner(bbox):
    """Transform bounding box in coco coord format (left, top, width, height) to corner
        format (left, top, right, bottom).

    Args:
        bbox(Tensor): [*, 4], where * means, any number of additional dimensions.

    Returns: 
        bbox_trans(Tensor): [*, 4], same shape as the input.

    Notes: In code, using (x1, y1, w, h) as (left, top, width, height),
        (x1, y1, x2, y2) as (left, top, right, bottom).
        
    """

    _check_bbox_input(bbox)

    cat_dim = bbox.dim() - 1
    x1, y1, w, h = bbox.chunk(4, dim=cat_dim)

    x2 = x1 + w
    y2 = y1 + h

    bbox_trans = torch.cat([x1, y1, x2, y2], dim=cat_dim)

    return bbox_trans


def bbox_corner2coco(bbox):
    """Transform bounding box in corner format (left, top, right, bottom) to
        coco coord format (left, top, width, height).

    Args:
        bbox(Tensor): [*, 4], where * means, any number of additional dimensions.

    Returns: 
        bbox_trans(Tensor): [*, 4], same shape as the input.

    Notes: In code, using (x1, y1, w, h) as (left, top, width, height),
        (x1, y1, x2, y2) as (left, top, right, bottom).

    """

    _check_bbox_input(bbox)

    cat_dim = bbox.dim() - 1
    x1, y1, x2, y2 = bbox.chunk(4, dim=cat_dim)

    w = x2 - x1
    h = y2 - y1

    bbox_trans = torch.cat([x1, y1, w, h], dim=cat_dim)

    return bbox_trans


def _check_bbox_input(bbox):
    assert bbox.size(-1) == 4, "Input bounding box coordinate must have 4 elements."
