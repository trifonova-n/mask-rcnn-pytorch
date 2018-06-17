"""
Cls and bbox head for FPN feature maps.
"""

import torch.nn as nn


class ClsBBoxHead_fc(nn.Module):
    """Classification and bounding box regression head using fully-connected style.
    """

    def __init__(self, depth, pool_size, num_classes):
        super(ClsBBoxHead_fc, self).__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.fc_0 = nn.Linear(self.depth * pool_size[0] * pool_size[1], 1024)
        self.fc_1 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.fc_cls = nn.Linear(1024, num_classes)
        self.fc_bbox = nn.Linear(1024, num_classes * 4)
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
        x = x.view(x.size(0), -1)
        x = self.fc_0(x)
        x = self.relu(x)
        x = self.fc_1(x)
        x = self.relu(x)

        fc_out_cls = self.fc_cls(x)
        cls_prob = self.log_softmax(fc_out_cls)
        bbox_reg = self.fc_bbox(x)
        bbox_reg = bbox_reg.view(-1, self.num_classes, 4)

        return cls_prob, bbox_reg


class ClsBBoxHead_conv(nn.Module):
    """Classification and bounding box regression head using Conv style.
    """

    def __init__(self, depth, pool_size, num_classes):
        super(ClsBBoxHead_conv, self).__init__()
        self.num_classes = num_classes
        self.depth = depth
        self.conv1 = nn.Conv2d(depth, 1024, kernel_size=pool_size, stride=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.fc_cls = nn.Linear(1024, num_classes)
        self.fc_bbox = nn.Linear(1024, num_classes * 4)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)

    def forward(self, x):
        """

        Args:
            x: NxSxHxW, rois fixed dimensional representation after pooling like RoIAlign,
                HxW: fixed size, like 14x14.

        Returns:
            cls_prob: NxSxNum_classes, probability of class.
            bbox_reg: NxSxNum_classes*4(dx, dy, dw, dh), defined in R-CNN paper.
        
        Notes: In above, S: number of rois per image feed to predict heads

        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x.view(-1, 1024)
        fc_out_cls = self.fc_cls(x)
        cls_prob = self.log_softmax(fc_out_cls)
        bbox_reg = self.fc_bbox(x)
        bbox_reg = bbox_reg.view(-1, self.num_classes, 4)

        return cls_prob, bbox_reg
