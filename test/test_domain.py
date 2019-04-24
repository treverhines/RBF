import numpy as np
import unittest

from rbf.pde import geometry as geo
from rbf.pde.domain import Domain, circle, sphere

class Test(unittest.TestCase):
    def test_intersection_count_2d(self):
        vert, smp = circle()
        pnt1 = np.random.normal(0.0, 2.0, (1000, 2))
        pnt2 = np.random.normal(0.0, 2.0, (1000, 2))

        out1 = geo.intersection_count(pnt1, pnt2, vert, smp)

        dom = Domain(vert, smp, orient_simplices=False, use_tree=False)
        out2 = dom.intersection_count(pnt1, pnt2)

        dom = Domain(vert, smp, orient_simplices=False, use_tree=True)
        out3 = dom.intersection_count(pnt1, pnt2)

        self.assertTrue(np.all(out1 == out2))
        self.assertTrue(np.all(out1 == out3))

    def test_intersection_count_3d(self):
        vert, smp = sphere()
        pnt1 = np.random.normal(0.0, 2.0, (1000, 3))
        pnt2 = np.random.normal(0.0, 2.0, (1000, 3))

        out1 = geo.intersection_count(pnt1, pnt2, vert, smp)

        dom = Domain(vert, smp, orient_simplices=False, use_tree=False)
        out2 = dom.intersection_count(pnt1, pnt2)

        dom = Domain(vert, smp, orient_simplices=False, use_tree=True)
        out3 = dom.intersection_count(pnt1, pnt2)

        self.assertTrue(np.all(out1 == out2))
        self.assertTrue(np.all(out1 == out3))

    def test_contains_2d(self):
        vert, smp = circle()
        pnt = np.random.normal(0.0, 2.0, (1000, 2))

        out1 = geo.contains(pnt, vert, smp)

        dom = Domain(vert, smp, orient_simplices=False, use_tree=False)
        out2 = dom.contains(pnt)

        dom = Domain(vert, smp, orient_simplices=False, use_tree=True)
        out3 = dom.contains(pnt)

        self.assertTrue(np.all(out1 == out2))
        self.assertTrue(np.all(out1 == out3))

    def test_contains_3d(self):
        vert, smp = sphere()
        pnt = np.random.normal(0.0, 2.0, (1000, 3))

        out1 = geo.contains(pnt, vert, smp)

        dom = Domain(vert, smp, orient_simplices=False, use_tree=False)
        out2 = dom.contains(pnt)

        dom = Domain(vert, smp, orient_simplices=False, use_tree=True)
        out3 = dom.contains(pnt)

        self.assertTrue(np.all(out1 == out2))
        self.assertTrue(np.all(out1 == out3))

    def test_snap_2d(self):
        vert, smp = circle()
        pnt = np.random.normal(0.0, 2.0, (1000, 2))

        dom = Domain(vert, smp, orient_simplices=False, use_tree=False)
        out1a,out1b = dom.snap(pnt)

        dom = Domain(vert, smp, orient_simplices=False, use_tree=True)
        out2a, out2b = dom.snap(pnt)

        self.assertTrue(np.all(out1a == out2a))
        self.assertTrue(np.all(out1b == out2b))

    def test_snap_3d(self):
        vert, smp = sphere()
        pnt = np.random.normal(0.0, 2.0, (1000, 3))

        dom = Domain(vert, smp, orient_simplices=False, use_tree=False)
        out1a, out1b = dom.snap(pnt)

        dom = Domain(vert, smp, orient_simplices=False, use_tree=True)
        out2a, out2b = dom.snap(pnt)

        self.assertTrue(np.all(out1a == out2a))
        self.assertTrue(np.all(out1b == out2b))

    def test_orient_simplices_2d(self):
        vert, smp = circle()
        # reverse the ordering of the simplices
        smp = smp[:, ::-1]

        out1 = geo.oriented_simplices(vert, smp)
        out2 = Domain(vert, smp, orient_simplices=True).simplices
        self.assertTrue(np.all(out1 == out2))

    def test_orient_simplices_3d(self):
        vert, smp = sphere()
        # reverse the ordering of the simplices
        smp = smp[:, ::-1]

        out1 = geo.oriented_simplices(vert, smp)
        out2 = Domain(vert, smp, orient_simplices=True).simplices
        self.assertTrue(np.all(out1 == out2))
