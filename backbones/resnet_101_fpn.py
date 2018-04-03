import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101


class ResNet_101_FPN(nn.Module):
    """ResNet 101 FPN backbone feature map extractor.
    
    """
    def __init__(self, pretrained=None):
        super(ResNet_101_FPN, self).__init__()
        if pretrained is not None:
            self.resnet = resnet101(pretrained=True)
        else:
            self.resnet = resnet101()

        self.lateral_conv1 = nn.Conv2d(2048, 256, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(1024, 256, kernel_size=1)
        self.lateral_conv3 = nn.Conv2d(512, 256, kernel_size=1)
        self.lateral_conv4 = nn.Conv2d(256, 256, kernel_size=1)

        self.anti_aliasing_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.anti_aliasing_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.anti_aliasing_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

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
