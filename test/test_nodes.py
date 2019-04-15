import numpy as np
import unittest
from rbf.pde import geometry as geo
from rbf.pde.nodes import (intersection_count, 
                           snap_to_boundary, 
                           build_rtree)
from rbf.pde.domain import circle, sphere


class Test(unittest.TestCase):
    def test_snap_to_boundary_2d(self):
        vert, smp = circle(5)
        pnt = np.random.uniform(-1.0, 1.0, (10000, 2))

        out1a, out1b = snap_to_boundary(pnt, vert, smp)
        tree = build_rtree(vert, smp)
        out2a, out2b = snap_to_boundary(pnt, vert, smp, tree=tree)
        
        self.assertTrue(np.all(out1a == out2a))
        self.assertTrue(np.all(out1b == out2b))

    def test_snap_to_boundary_3d(self):
        vert, smp = sphere(5)
        pnt = np.random.uniform(-1.0, 1.0, (1000, 3))

        out1a, out1b = snap_to_boundary(pnt, vert, smp)
        tree = build_rtree(vert, smp)
        out2a, out2b = snap_to_boundary(pnt, vert, smp, tree=tree)
        
        self.assertTrue(np.all(out1a == out2a))
        self.assertTrue(np.all(out1b == out2b))

    def test_intersection_count_2d(self):
        vert, smp = circle(5)
        pnt1 = np.random.uniform(-1.0, 1.0, (10000, 2))
        pnt2 = pnt1 + np.random.normal(0.0, 0.01, (10000, 2)) 

        out1 = geo.intersection_count(pnt1, pnt2, vert, smp)
        out2 = intersection_count(pnt1, pnt2, vert, smp)
        tree = build_rtree(vert, smp)
        out3 = intersection_count(pnt1, pnt2, vert, smp, tree=tree)
        self.assertTrue(np.all(out1 == out2))
        self.assertTrue(np.all(out1 == out3))

    def test_intersection_count_3d(self):
        vert, smp = sphere(5)
        pnt1 = np.random.uniform(-1.0, 1.0, (1000, 3))
        pnt2 = pnt1 + np.random.normal(0.0, 0.01, (1000, 3)) 

        out1 = intersection_count(pnt1, pnt2, vert, smp)
        out2 = geo.intersection_count(pnt1, pnt2, vert, smp)
        tree = build_rtree(vert, smp)
        out3 = intersection_count(pnt1, pnt2, vert, smp, tree=tree)
        self.assertTrue(np.all(out1 == out2))
        self.assertTrue(np.all(out1 == out3))
