from torchvision.models.resnet import resnet101
import torch.nn as nn


class ResNet_101(nn.Module):
    """ResNet 101 FPN backbone feature map extractor..
    
    """

    def __init__(self, pretrained=None):
        super(ResNet_101, self).__init__()
        if pretrained == 'imagenet':
            self.resnet = resnet101(pretrained=True)
        else:
            self.resnet = resnet101()

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
