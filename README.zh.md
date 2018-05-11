# Mask R-CNN PyTorch实现 

阅读其他语言版本: [中文](./README.zh.md) [English](./README.md) 

这个实现版本在一些非标准测试集上进行了验证，取得了不错效果，速度也较快，在标准数据集，如PASCAL VOC与COCO的结果很快会公布出来。

![maskrcnn-result](http://chuantu.biz/t6/250/1520606201x-1404795469.png)
还有一些工作正在进行当中：

- [ ] 支持batch size >= 2.
- [ ] 在COCO数据集上的训练代码样例，以及预训练权重.
- [ ] 修复下使用FPN的问题
- [ ] 用纯PyTorch代码替换第三方libs，如NMS与roi_align，torchvision对NMS的支持正在开发当中，需要等一下这个版本.
- [ ] 支持PyTorch 0.4 还有令人激动的、很快就要发布的1.0版本.

## 使用用法

### 支持的PyTorch版本
PyTorch 0.4还未支持, 低于0.3.1的版本不保证可用. 

经过测试环境:

Linux ubuntu 16.04

CUDA == 8.0

python == 3.5.2

torch == 0.3.1

torchvision == 0.2.0

### 安装

#### 1. 下载这个项目
 `git clone git@github.com:GeeshangXu/mask-rcnn-pytorch.git`
 
#### 2. 安装一些Python库依赖

`pip install cffi pillow easydict`

#### 3. 安装第三方libs
选择你的CUDA版本，`cuda8` 或者 `cuda9`

`python .\third_party\build_libs.py cuda8`

### 使用 MaskRCNN

```python
# 首先看一下config.ini，可能需要针对数据集调整一些超参数

import sys
# 将此工程的根目录加入到PATH
sys.path.append("/ANY_DIR_YOU_CLONE_AT/mask-rcnn-pytorch/")
from maskrcnn import MaskRCNN
mask_rcnn = MaskRCNN(num_classes=81, pretrained="imagenet")
``` 
 
## 例子
### 1: 使用典型的PyTorch流程训练定制训练集.
1. 下载这个名为CST-Dataset的小数据集，只有25MB大小。

    下载链接: [CST-Dataset](https://github.com/GeeshangXu/cst-dataset)

2. 将 `config.ini` 替换为 `examples/cst-dataset/config.ini`

3. 查看 Jupyter Notebook [example-cst-dataset.ipynb](./examples/cst-dataset/example-cst-dataset.ipynb)

### 2: 训练COCO训练集.

过段时间放出。

## 在标准数据上的结果 
(过段时间放出)

| dataset | train memory(GB) | train time (hr/epoch) |inference time(s/img) |box AP| mask AP |
| :---------------|:--------|---|:-----|----|----|
| PASCAL VOC 2012 |  |  | | | |
| COCO 2017       |  |  | | | |


## 源码文件夹结构

源码文件夹按照Mask R-CNN模型的执行过程与内部子模型进行组织，希望达到解耦的目的，更容易进行子模型变种的试验。

![dirs-relationship](http://chuantu.biz/t6/267/1522230494x-1404795469.jpg)

#### 1. backbones: 
一些特征图提取器（feature map extractor），用于支持Mask R-CNN，像ResNet-101等等。

#### 2. proposal:
提取RoI(Region of Interest)的模型，像RPN等。

#### 3. pooling:
池化操作，来得到RoI的固定大小（如14x14像素）的表达，如RoIAlign及其变种。

#### 4. heads:
预测用的head，包括分类的head，预测边框的head，以及预测mask的head。

#### 5. tools:
一些常用的工具方法，如iou计算及可视化等工具。

#### 6. tests:
单元测试及初步完整性检查（sanity check）代码。

#### 6. libs:
依赖的第三方的代码。


## 参考资料:

1. [Kaiming He et al. Mask R-CNN](https://arxiv.org/abs/1703.06870)
2. [Shaoqing Ren et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
3. [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
4. [ruotianluo/pytorch-faster-rcnn](ruotianluo/pytorch-faster-rcnn)
5. [TuSimple/mx-maskrcnn](https://github.com/TuSimple/mx-maskrcnn)
6. [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
