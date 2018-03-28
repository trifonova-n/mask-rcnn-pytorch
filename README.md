# Mask R-CNN implementation in PyTorch

![maskrcnn-result](http://chuantu.biz/t6/250/1520606201x-1404795469.png)

## Usage
(Usage and examples will update soon.)

```python
from maskrcnn import MaskRCNN
from torch.utils.data import Dataset

# use pretrained weights: 
# 1) "imagenet", just backbone feature map extractor trained on ImageNet.
# 2) "coco", whole maskrcnn model pretrained on COCO.

mask_rcnn = MaskRCNN(num_classes=1000, pretrained="imagenet") 

class OneDataset(Dataset):
    def __init__(self):
        pass
    def __getitem__(self, item):
        pass
    def __len__(self):
        pass

def train():
    pass
def test():
    pass
```

## Source directory

Source directories are arranged according to internal models or execution process of Mask R-CNN 
model, trying to decouple these models or processes to make it easy for adding experimental 
variants.

![dirs-relationship](http://chuantu.biz/t6/267/1522230494x-1404795469.jpg)

#### 1. backbones: 

Several feature map extractor backbones support Mask R-CNN, like ResNet-101-FPN.

#### 2. proposal:

RoI(Region of Interest) proposal model, like RPN and variants.

#### 3. pooling:

Pooling for fixed dimensional representation, like RoIAlign and some variants.

#### 4. heads:
Predict heads include classification head, bounding box head, mask head and their variants.

#### 5. tools:
Some utils like function to calculate iou, and visualization tools.

#### 6. tests:
Unittests and sanity checks.

#### 6. libs:

Some third-party libs this project based on.


## Reference:

1. [Kaiming He et al. Mask R-CNN](https://arxiv.org/abs/1703.06870)
2. [Shaoqing Ren et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
3. [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
4. [ruotianluo/pytorch-faster-rcnn](ruotianluo/pytorch-faster-rcnn)
5. [TuSimple/mx-maskrcnn](https://github.com/TuSimple/mx-maskrcnn)
6. [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
