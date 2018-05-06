import unittest
import torch
from maskrcnn import MaskRCNN


class TestUtils(unittest.TestCase):
    def test__generate_targets(self):
        pass

    def test__roi_align_fpn(self):
        pass

    def test__process_result(self):
        pass

    def test__get_bbox_targets(self):
        proposals = torch.zeros(3, 4)
        proposals[:, :2] = 0
        proposals[:, 2:] = 100
        gt_bboxes = torch.zeros(3, 4)
        gt_bboxes[:, :2] = 5
        gt_bboxes[:, 2:] = 105
        bbox_targets = torch.zeros(3, 4)
        bbox_targets[:, :2] = 0.05
        bbox_targets[:, 2:] = 0

        self.assertTrue(MaskRCNN._get_bbox_targets(proposals, gt_bboxes).equal(bbox_targets))

    def test__get_mask_targets(self):
        pass

    def test__calc_maskrcnn_loss(self):
        pass


if __name__ == '__main__':
    unittest.main()
