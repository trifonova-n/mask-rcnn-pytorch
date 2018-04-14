# Mask R-CNN implementation in PyTorch 

![maskrcnn-result](http://chuantu.biz/t6/250/1520606201x-1404795469.png)

This implementation is verified with some custom datasets, achieved good speed and result, quantitative results on
some standard datasets like PASCAL VOC and COCO will release soon.

There is still some work to be done.
- [ ] ImageNet pretrained weights of backbone is ok, works need on COCO pretrained weights of the
whole model 
- [ ] support batch size >= 2.
- [ ] refined documentation and examples
- [ ] replace third party libs nms and roi_align with pure PyTorch, but need wait NMS natively 
supported version of PyTorch, which is coming soon.  

## Usage

### Installation

#### 1. Download this repo
 `git clone git@github.com:GeeshangXu/mask-rcnn-pytorch.git`
 
#### 2. Install python package dependencies

`pip install cffi pillow easydict`

#### 3. Install libs
Choose your GPU architecture, e.g. sm_62 for Titan XP , then run

`python .\libs\build_libs.py sm_62`

| architectures | capabilities  |  example GPU|
| :------------- |:-------------| :-----|
| sm_30, sm_32 | Basic features + Keplersupport +Unified memory programming |  |
| sm_35	      | + Dynamic parallelism support |  |
| sm_50, sm_52, sm_53 | + Maxwell support | M40 |
| sm_60, sm_61, sm_62 | + Pascal support |Titan XP, 1080(Ti), 1070 |
| sm_70 | + Volta support|V100|

### Using MaskRCNN

**NOTE: Two examples are under working, one show how to train custom dataset using PyTorch 
typical pipeline (dataset, dataloader, transform, train, test), the other show how to train COCO
dataset. Actually, you may figure out how to use this implementation by reading docstring 
(\*^\_\_^\*), I am trying my best to 
document this project.**

```python
# Take a look at config.ini, config some hyper-parameters.

import sys
# add this project's root directory to PATH
sys.path.append("/ANY_DIR_YOU_CLONE_AT/mask-rcnn-pytorch/")
from maskrcnn import MaskRCNN
from torch.utils.data import Dataset, DataLoader

# use pretrained weights: 
# 1) "imagenet", just backbone feature map extractor trained on ImageNet.
# 2) "coco", whole maskrcnn model pretrained on COCO.
mask_rcnn = MaskRCNN(num_classes=80, pretrained="imagenet")
``` 
 
##### Example 1: Custom Dataset Using PyTorch Typical Pipeline.
I will create a simple dummy dataset using as example, and give the
download link soon. 
```python
class DummyDataset(Dataset):
    def __init__(self):
        pass
    def __getitem__(self, index):
        pass
    def __len__(self):
        pass

# make sure your dataloader return below as the docstring says. 
"""
image(Tensor): image data. [N, C, H, W]  
gt_classes(Tensor): [N, M], ground truth class ids.
gt_bboxes(Tensor): [N, M, (x1, y1, x2, y2)], ground truth bounding boxes, coord is 
    left, top, right, bottom.
gt_masks(Tensor): [N, M, 1, H, W], ground truth masks.
"""
dummy_dataloader = DataLoader(dataset=DummyDataset, batch_size=1)

def train_epoch():
    pass
    
def val_epoch():
    pass

def train():
    pass

def test():
    pass
```
##### Example 2: Train COCO Dataset.
```python
    pass(release later)

```

## Result on Standard Dataset 
(release later)

| dataset | train memory(GB) | train time (hr/epoch) |inference time(s/img) |box AP| mask AP |
| :---------------|:--------|---|:-----|----|----|
| PASCAL VOC 2007 |  |  | | | |
| PASCAL VOC 2012 |  |  | | | |
| COCO 2017       |  |  | | | |


## Source Directory

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
