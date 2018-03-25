import unittest
import torch
from tools.utils import calc_iou, coord_corner2center, coord_center2corner


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
        a2[:, 2:] = 99
        b2 = torch.zeros(4, 4)
        b2[:, :2] = 0
        b2[:, 2:] = 49
        c2 = torch.zeros(3, 4)
        c2[:, :] = 0.25

        a3 = torch.zeros(3, 4)
        a3[:, :2] = 0
        a3[:, 2:] = 99
        b3 = torch.zeros(4, 4)
        b3[:, :2] = 0
        b3[:, 2:] = 199
        c3 = torch.zeros(3, 4)
        c3[:, :] = 0.25

        a4 = torch.zeros(3, 4)
        a4[:, :2] = 50
        a4[:, 2:] = 149
        b4 = torch.zeros(4, 4)
        b4[:, 0] = 100
        b4[:, 1] = 0
        b4[:, 2:] = 199
        c4 = torch.zeros(3, 4)
        c4[:, :] = 0.2

        self.assertTrue(calc_iou(a1, b1).equal(c1))
        self.assertTrue(calc_iou(a2, b2).equal(c2))
        self.assertTrue(calc_iou(a3, b3).equal(c3))
        self.assertTrue(calc_iou(a4, b4).equal(c4))

    def test_coord_corner2center(self):
        a1 = torch.zeros(3, 4)
        a1[:, :2] = 0
        a1[:, 2:] = 99
        b1 = torch.zeros(3, 4)
        b1[:, :2] = 50
        b1[:, 2:] = 100

        a2 = torch.zeros(3, 4)
        a2[:, 0] = 50
        a2[:, 1] = 100
        a2[:, 2] = 99
        a2[:, 3] = 199
        b2 = torch.zeros(3, 4)
        b2[:, 0] = 75
        b2[:, 1] = 150
        b2[:, 2] = 50
        b2[:, 3] = 100

        self.assertTrue(coord_corner2center(a1).equal(b1))
        self.assertTrue(coord_corner2center(a2).equal(b2))

    def test_coord_center2corner(self):
        a1 = torch.zeros(3, 4)
        a1[:, :2] = 0
        a1[:, 2:] = 99
        b1 = torch.zeros(3, 4)
        b1[:, :2] = 50
        b1[:, 2:] = 100

        a2 = torch.zeros(3, 4)
        a2[:, 0] = 50
        a2[:, 1] = 100
        a2[:, 2] = 99
        a2[:, 3] = 199
        b2 = torch.zeros(3, 4)
        b2[:, 0] = 75
        b2[:, 1] = 150
        b2[:, 2] = 50
        b2[:, 3] = 100

        self.assertTrue(coord_center2corner(b1).equal(a1))
        self.assertTrue(coord_center2corner(b2).equal(a2))


if __name__ == '__main__':
    unittest.main()
