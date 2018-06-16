import torch.nn as nn
from torchvision.models.vgg import vgg16


class VGG16(nn.Module):
    def __init__(self, pretrained):
        super(VGG16, self).__init__()
        vgg = vgg16(pretrained=True if pretrained is not None else False)
        # Fix the layers before conv3:
        for layer in range(10):
            for p in vgg.features[layer].parameters():
                p.requires_grad = False

        self.feat_extractor = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    def forward(self, x):
        return self.feat_extractor(x)
