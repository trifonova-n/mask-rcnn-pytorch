# Mask R-CNN implementation in PyTorch 

Read this in other languages: [English](./README.md) [中文](./README.zh.md) 

This implementation is verified with some custom datasets, achieved good speed and result, quantitative results on
some standard datasets like PASCAL VOC and COCO will release soon.

![maskrcnn-result](http://chuantu.biz/t6/250/1520606201x-1404795469.png)

There is still some work to be done.
- [ ] support batch size >= 2.
- [ ] COCO dataset training example and pre-trained weights.
- [ ] fix performance problem using FPN. 
- [ ] replace third-party libs NMS and roi_align with pure PyTorch, NMS in torchvision is under developing, need to wait the version coming out.
- [ ] keep up with PyTorch version 0.4 and the exciting version 1.0 that is about to be released.

## Usage

### Supported PyTorch version
PyTorch 0.4 is not supported yet, versions below 0.3.1 are not guaranteed to work. 

Tested version: torch == 0.3.1 torchvision == 0.2.0

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

```python
# Take a look at config.ini, config some hyper-parameters.

import sys
# add this project's root directory to PATH
sys.path.append("/ANY_DIR_YOU_CLONE_AT/mask-rcnn-pytorch/")
from maskrcnn import MaskRCNN
mask_rcnn = MaskRCNN(num_classes=81, pretrained="imagenet")
``` 
 
### Examples
#### 1: Train Custom Dataset with PyTorch Typical Pipeline.
1. Download the tiny (25MB) dataset  CST-Dataset

    Download link: [CST-Dataset](https://github.com/GeeshangXu/cst-dataset)

2. replace `config.ini` with `examples/cst-dataset/config.ini`

3. see Jupyter Notebook [example-cst-dataset.ipynb](./examples/cst-dataset/example-cst-dataset.ipynb)

##### 2: Train COCO Dataset.

release later

## Result on Standard Dataset 
(release later)

| dataset | train memory(GB) | train time (hr/epoch) |inference time(s/img) |box AP| mask AP |
| :---------------|:--------|---|:-----|----|----|
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

Pooling for fixed dimensional representation(e.g. 14x14 pixels), like RoIAlign and some variants.

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
