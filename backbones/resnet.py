from torchvision.models import resnet
import torch.nn as nn


class ResNet(nn.Module):
    """ResNet 101 FPN backbone feature map extractor..
    
    """

    def __init__(self, resnet_layer, pretrained=None):
        super(ResNet, self).__init__()
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

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        c4 = self.resnet.layer3(x)
        # c5 = self.resnet.layer4(c4)

        return c4
