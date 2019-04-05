import unittest

import numpy as np
from rbf.pde.quadtree import QuadTree
from rbf.pde.octtree import OctTree
try:
    from rtree.index import Index, Property
    HAS_RTREE = True
except ImportError:
    HAS_RTREE = False    
    print('The package `rtree` should be installed to perform a '
          'more thorough test of `QuadTree` and `OctTree`')


class Test(unittest.TestCase):
    def test_quad_tree_against_rtree(self):
        if not HAS_RTREE:
            # do not perform this test if rtree is not installed
            return

        # test quad tree against RTree with randomly generated boxes
        lb = np.random.randint(-10, 10, (100, 2))
        ub = lb + np.random.randint(0, 5, (100, 2))
        boxes = np.hstack((lb, ub)).astype(float)

        query_lb = np.random.randint(-10, 10, (1000, 2))
        query_ub = query_lb + np.random.randint(0, 3, (1000, 2))
        query_boxes = np.hstack((query_lb, query_ub)).astype(float)

        bounds = np.array([-15.0, -15.0, 15.0, 15.0])
        tree = QuadTree(bounds, max_depth=5)
        tree.add_boxes(boxes)
        tree.prune()

        p = Property()
        p.dimension = 2
        rtree = Index(properties=p)
        for i, b in enumerate(boxes):
            rtree.add(i, b)

        for b in query_boxes:
            out1 = set(tree.intersections(b))
            out2 = set(rtree.intersection(b))
            self.assertTrue(out1 == out2)

    def test_oct_tree_against_rtree(self):
        if not HAS_RTREE:
            # do not perform this test if rtree is not installed
            return
            
        # test quad tree against RTree with randomly generated boxes
        lb = np.random.randint(-10, 10, (100, 3))
        ub = lb + np.random.randint(0, 5, (100, 3))
        boxes = np.hstack((lb, ub)).astype(float)

        query_lb = np.random.randint(-10, 10, (1000, 3))
        query_ub = query_lb + np.random.randint(0, 3, (1000, 3))
        query_boxes = np.hstack((query_lb, query_ub)).astype(float)

        bounds = np.array([-15.0, -15.0, -15.0, 15.0, 15.0, 15.0])
        tree = OctTree(bounds, max_depth=5)
        tree.add_boxes(boxes)
        tree.prune()

        p = Property()
        p.dimension = 3
        rtree = Index(properties=p)
        for i, b in enumerate(boxes):
            rtree.add(i, b)

        for b in query_boxes:
            out1 = set(tree.intersections(b))
            out2 = set(rtree.intersection(b))
            self.assertTrue(out1 == out2)

