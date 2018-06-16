"""
Common module for all head.
"""

from configparser import ConfigParser
from torchvision.models import resnet
from torchvision.models.vgg import vgg16
import torch.nn as nn
import torch.nn.functional as F
import os


class HeadCommon(nn.Module):
    def __init__(self, pretrained=None):
        super(HeadCommon, self).__init__()
        self.config = ConfigParser()
        config_path = os.path.abspath(os.path.join(__file__, "../../", "config.ini"))
        assert os.path.exists(config_path), "config.ini not exists!"
        self.config.read(config_path)
        self.backbone_type = self.config['BACKBONE']['BACKBONE_TYPE']
        _pretrained = True if pretrained is not None else False
        assert self.backbone_type in ['resnet', 'vgg16']
        if self.backbone_type == 'resnet':
            resnet_layer = int(self.config['BACKBONE']['RESNET_LAYER'])
            assert resnet_layer in [18, 34, 50, 101, 152]
            if resnet_layer == 18:
                _resnet = resnet.resnet18(_pretrained)
            elif resnet_layer == 34:
                _resnet = resnet.resnet34(_pretrained)
            elif resnet_layer == 50:
                _resnet = resnet.resnet50(_pretrained)
            elif resnet_layer == 101:
                _resnet = resnet.resnet101(_pretrained)
            else:
                _resnet = resnet.resnet152(_pretrained)
            # using resnet_c5 the last bottle neck of resnet
            _resnet.layer4[0].conv2.stride = (1, 1)
            _resnet.layer4[0].downsample[0].stride = (1, 1)
            self.resnet_c5 = _resnet.layer4
            self.resnet_c5_avg = _resnet.avgpool
        elif self.backbone_type == 'vgg16':
            assert not bool(int(self.config['HEAD']['MASK_HEAD_ON'])), (
                "When mask head on, not support vgg16 backbone.")
            vgg = vgg16(pretrained=True)
            self.vgg_fc = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    def forward(self, x, is_mask=False):
        if self.backbone_type == 'resnet':
            x = self.resnet_c5(x)
            if not is_mask:
                x = self.resnet_c5_avg(x)
                x = F.relu(x)
                x = x.view(x.size(0), -1)
        elif self.backbone_type == 'vgg16':
            x = x.view(x.size(0), -1)
            x = self.vgg_fc(x)
            x = F.relu(x)

        return x
