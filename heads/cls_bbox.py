"""
Cls and bbox head for resnet-c4 feature map.
"""

import torch.nn as nn


class ClsBBoxHead(nn.Module):
    """Classification and bounding box regression head using fully-connected style.
    """

    def __init__(self, depth, num_classes):
        super(ClsBBoxHead, self).__init__()
        self.num_classes = num_classes
        self.fc_cls = nn.Linear(depth, num_classes)
        self.fc_bbox = nn.Linear(depth, num_classes * 4)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        
        Args:
            x: (NxS)xCxHxW, roi fixed dimensional representation after pooling like RoIAlign,
                HxW: fixed size, like 7x7.

        Returns:
            cls_prob: (NxS)x num_classes, probability of class.
            bbox_reg: (NxS)x num_classes x 4(dx, dy, dw, dh), defined in R-CNN paper.
        
        Notes: In above, S: number of rois per image feed to predict heads
            
        """

        fc_out_cls = self.fc_cls(x)
        cls_prob = self.log_softmax(fc_out_cls)
        bbox_reg = self.fc_bbox(x)
        bbox_reg = bbox_reg.view(bbox_reg.size(0), self.num_classes, 4)

        return cls_prob, bbox_reg
