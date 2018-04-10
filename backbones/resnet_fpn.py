import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet


class ResNetFPN(nn.Module):
    """ResNet FPN backbone feature map extractor.
    
    """

    def __init__(self, resnet_layer, pretrained=None):
        super(ResNetFPN, self).__init__()
        self.lateral_conv1 = nn.Conv2d(2048, 256, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(1024, 256, kernel_size=1)
        self.lateral_conv3 = nn.Conv2d(512, 256, kernel_size=1)
        self.lateral_conv4 = nn.Conv2d(256, 256, kernel_size=1)

        self.anti_aliasing_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.anti_aliasing_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.anti_aliasing_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self._init_parameters()

        assert resnet_layer in [18, 34, 50, 101, 152]
        pretrained = True if pretrained is not None else False
        if resnet_layer == 18:
            self.resnet = resnet.resnet18(pretrained)
        elif resnet_layer == 34:
            self.resnet = resnet.resnet34(pretrained)
        elif resnet_layer == 50:
            self.resnet = resnet.resnet50(pretrained)
        elif resnet_layer == 101:
            self.resnet = resnet.resnet101(pretrained)
        elif resnet_layer == 152:
            self.resnet = resnet.resnet152(pretrained)

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)

    def forward(self, x):
        """
        Args:
            x(Variable): input image 
        Returns:
            feature_pyramid(tuple): feature pyramid contains 5 feature maps.
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        c2 = self.resnet.layer1(x)
        c3 = self.resnet.layer2(c2)
        c4 = self.resnet.layer3(c3)
        c5 = self.resnet.layer4(c4)

        p5 = self.lateral_conv1(c5)
        p4_tmp = self.lateral_conv2(c4) + F.upsample(p5, scale_factor=2, mode='nearest')
        p4 = self.anti_aliasing_conv1(p4_tmp)
        p3_tmp = self.lateral_conv3(c3) + F.upsample(p4, scale_factor=2, mode='nearest')
        p3 = self.anti_aliasing_conv2(p3_tmp)
        p2_tmp = self.lateral_conv4(c2) + F.upsample(p3, scale_factor=2, mode='nearest')
        p2 = self.anti_aliasing_conv3(p2_tmp)

        # Detectron style, use max pooling to simulate stride 2 subsampling
        p6 = F.max_pool2d(p5, kernel_size=1, stride=2)
        feature_pyramid = (p2, p3, p4, p5, p6)

        return feature_pyramid
