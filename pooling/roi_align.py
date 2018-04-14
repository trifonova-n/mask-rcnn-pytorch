import torch.nn as nn
from torch.autograd import Variable
from libs.roi_align.roi_align import RoIAlign as _RoiAlign


class RoiAlign(nn.Module):
    """RoiAlign wrapper
    
    """

    def __init__(self, grid_size):
        """
        Args:
            grid_size(tuple): grid pooling size apply to roi, e.g., (14, 14).  
        """
        super(RoiAlign, self).__init__()
        assert isinstance(grid_size, tuple)
        self.roi_align = _RoiAlign(crop_width=grid_size[0], crop_height=grid_size[1])

    def forward(self, feature_map, rois, img_height):
        """
        
        Args:
            feature_map: [N, C, H, W]
            rois(Tensor): [(NxM), (n, score, x1, y1, x2, y2)], n is mini-batch index, coord is in 
                origin image scale.
            img_height(int): origin image height
        Returns:
            roi_pool: MxCxoHxoW  M: number of roi in all mini-batch.
            
        """
        bbox_idx = rois[:, 0]
        bboxes = rois[:, 2:]
        # transform origin image coord to feature map coord.
        stride = img_height / feature_map.size(2)  # stride of feature map, e.g. C4:16
        bboxes /= stride

        bboxes = Variable(bboxes, requires_grad=False)
        bbox_idx = Variable(bbox_idx.int(), requires_grad=False)
        roi_pool = self.roi_align(feature_map, bboxes, bbox_idx)

        return roi_pool
