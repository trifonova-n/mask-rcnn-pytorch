import unittest
import torch
from maskrcnn import MaskRCNN


class TestMaskRCNN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("loading MaskRCNN model ...")
        cls.model = MaskRCNN(81, 'imagenet')

    def test__generate_targets(self):
        pass

    def test__roi_align_fpn(self):
        pass

    def test__process_result(self):
        pass

    def test__get_mask_targets(self):
        pass

    def test__calc_maskrcnn_loss(self):
        pass

    def test__get_bbox_targets(self):
        proposal0 = torch.zeros(3, 4)
        proposal0[:, :2] = 0
        proposal0[:, 2:] = 100
        gt_bbox0 = torch.zeros(3, 4)
        gt_bbox0[:, :2] = 5
        gt_bbox0[:, 2:] = 105
        bbox_target0 = torch.zeros(3, 4)
        bbox_target0[:, :2] = 0.05
        bbox_target0[:, 2:] = 0

        proposal1 = torch.zeros(3, 4)
        proposal1[:, :2] = 50
        proposal1[:, 2:] = 100
        gt_bbox1 = torch.zeros(3, 4)
        gt_bbox1[:, :2] = 0
        gt_bbox1[:, 2:] = 200
        bbox_target1 = torch.zeros(3, 4)
        bbox_target1[:, :2] = 0.5
        bbox_target1[:, 2:] = 1.3863

        target0 = MaskRCNN._get_bbox_targets(proposal0, gt_bbox0)
        target1 = MaskRCNN._get_bbox_targets(proposal1, gt_bbox1)

        self.assertTrue(torch.abs(target0 - bbox_target0).le(1e-4).all())
        self.assertTrue(torch.abs(target1 - bbox_target1).le(1e-4).all())

    def test__bbox_nms(self):
        props0 = torch.zeros(1, 9, 6)
        props0[0, 0, :] = torch.FloatTensor([1, 0, 0, 100, 100, 1])
        props0[0, 1, :] = torch.FloatTensor([1, 0, 0, 95, 95, 0.9])
        props0[0, 2, :] = torch.FloatTensor([1, 0, 0, 90, 90, 0.8])

        props0[0, 3, :] = torch.FloatTensor([2, 0, 0, 100, 100, 1])
        props0[0, 4, :] = torch.FloatTensor([2, 0, 0, 95, 95, 0.9])
        props0[0, 5, :] = torch.FloatTensor([2, 0, 0, 90, 90, 0.8])

        props0[0, 6, :] = torch.FloatTensor([3, 0, 0, 100, 100, 1])
        props0[0, 7, :] = torch.FloatTensor([3, 0, 0, 95, 95, 0.9])
        props0[0, 8, :] = torch.FloatTensor([3, 0, 0, 90, 90, 0.8])

        index0 = self.model._bbox_nms(props0)

        props1 = torch.zeros(1, 9, 6)
        props1[0, 0, :] = torch.FloatTensor([1, 0, 0, 100, 100, 0.9])
        props1[0, 1, :] = torch.FloatTensor([1, 0, 0, 95, 95, 1])
        props1[0, 2, :] = torch.FloatTensor([1, 0, 0, 90, 90, 0.8])

        props1[0, 3, :] = torch.FloatTensor([2, 0, 0, 100, 100, 0.9])
        props1[0, 4, :] = torch.FloatTensor([2, 0, 0, 95, 95, 0.99])
        props1[0, 5, :] = torch.FloatTensor([2, 0, 0, 90, 90, 0.8])

        props1[0, 6, :] = torch.FloatTensor([3, 0, 0, 100, 100, 0.8])
        props1[0, 7, :] = torch.FloatTensor([3, 0, 0, 95, 95, 0.8])
        props1[0, 8, :] = torch.FloatTensor([3, 0, 0, 90, 90, 0.9])

        index1 = self.model._bbox_nms(props1)

        self.assertTrue(index0.equal(torch.LongTensor([0, 3, 6])))
        self.assertTrue(index1.equal(torch.LongTensor([1, 4, 8])))


if __name__ == '__main__':
    unittest.main()
