import unittest
import torch
from tools.detect_utils import (calc_iou, bbox_corner2center, bbox_center2corner,
                                bbox_coco2corner, bbox_corner2coco)


class TestUtils(unittest.TestCase):
    def test_calc_iou(self):
        a1 = torch.zeros(3, 4)
        a1[:, :2] = 0
        a1[:, 2:] = 100
        b1 = torch.zeros(4, 4)
        b1[:, :2] = 0
        b1[:, 2:] = 100
        c1 = torch.zeros(3, 4)
        c1[:, :] = 1

        a2 = torch.zeros(3, 4)
        a2[:, :2] = 0
        a2[:, 2:] = 100
        b2 = torch.zeros(4, 4)
        b2[:, :2] = 0
        b2[:, 2:] = 50
        c2 = torch.zeros(3, 4)
        c2[:, :] = 0.25

        a3 = torch.zeros(3, 4)
        a3[:, :2] = 0
        a3[:, 2:] = 100
        b3 = torch.zeros(4, 4)
        b3[:, :2] = 0
        b3[:, 2:] = 200
        c3 = torch.zeros(3, 4)
        c3[:, :] = 0.25

        a4 = torch.zeros(3, 4)
        a4[:, :2] = 50
        a4[:, 2:] = 150
        b4 = torch.zeros(4, 4)
        b4[:, 0] = 100
        b4[:, 1] = 0
        b4[:, 2:] = 200
        c4 = torch.zeros(3, 4)
        c4[:, :] = 0.2

        a5 = torch.zeros(3, 4)
        a5[:1, :2] = 0
        a5[:1, 2:] = 100
        b5 = torch.zeros(4, 4)
        b5[:1, :2] = 0
        b5[:1, 2:] = 100
        c5 = torch.zeros(3, 4)
        c5[:, :1] = 1

        a5[1:, :2] = 0
        a5[1:, 2:] = 100
        b5[1:, :2] = 0
        b5[1:, 2:] = 50
        c5[:, 1:] = 0.25

        self.assertTrue(calc_iou(a1, b1).equal(c1))
        self.assertTrue(calc_iou(a2, b2).equal(c2))
        self.assertTrue(calc_iou(a3, b3).equal(c3))
        self.assertTrue(calc_iou(a4, b4).equal(c4))
        self.assertTrue(calc_iou(a5, b5).equal(c5))

    def test_bbox_corner2center(self):
        a1 = torch.FloatTensor([0, 0, 100, 100])
        b1 = torch.FloatTensor([50, 50, 100, 100])
        a2 = torch.FloatTensor([[0, 0, 100, 100]])
        b2 = torch.FloatTensor([[50, 50, 100, 100]])
        a3 = torch.FloatTensor([[50, 50, 100, 100]])
        b3 = torch.FloatTensor([[75, 75, 50, 50]])

        self.assertTrue(bbox_corner2center(a1).equal(b1))
        self.assertTrue(bbox_corner2center(a2).equal(b2))
        self.assertTrue(bbox_corner2center(a3).equal(b3))

    def test_bbox_center2corner(self):
        a1 = torch.FloatTensor([0, 0, 100, 100])
        b1 = torch.FloatTensor([50, 50, 100, 100])
        a2 = torch.FloatTensor([[0, 0, 100, 100]])
        b2 = torch.FloatTensor([[50, 50, 100, 100]])
        a3 = torch.FloatTensor([[50, 50, 100, 100]])
        b3 = torch.FloatTensor([[75, 75, 50, 50]])

        self.assertTrue(bbox_center2corner(b1).equal(a1))
        self.assertTrue(bbox_center2corner(b2).equal(a2))
        self.assertTrue(bbox_center2corner(b3).equal(a3))

    def test_bbox_coco2corner(self):
        a1 = torch.FloatTensor([50, 50, 100, 100])
        b1 = torch.FloatTensor([50, 50, 150, 150])
        a2 = torch.FloatTensor([[50, 50, 100, 100]])
        b2 = torch.FloatTensor([[50, 50, 150, 150]])

        self.assertTrue(bbox_coco2corner(a1).equal(b1))
        self.assertTrue(bbox_coco2corner(a2).equal(b2))

    def test_bbox_corner2coco(self):
        a1 = torch.FloatTensor([50, 50, 100, 100])
        b1 = torch.FloatTensor([50, 50, 150, 150])
        a2 = torch.FloatTensor([[50, 50, 100, 100]])
        b2 = torch.FloatTensor([[50, 50, 150, 150]])

        self.assertTrue(bbox_corner2coco(b1).equal(a1))
        self.assertTrue(bbox_corner2coco(b2).equal(a2))

if __name__ == '__main__':
    unittest.main()
