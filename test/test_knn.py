import numpy as np
import rbf.pde.knn
import unittest

class Test(unittest.TestCase):
  def test_nearest_1d_1(self):
    pop = np.array([[1.0],
                    [2.0],
                    [2.5]])
                          
    query = np.array([[2.1]])
    soln_idx = np.array([[1, 2]])
    soln_dist = np.array([[0.1, 0.4]])
    idx, dist = rbf.pde.knn.k_nearest_neighbors(query, pop, 2)
    self.assertTrue(np.all(soln_idx==idx))
    self.assertTrue(np.allclose(soln_dist, dist))

  def test_nearest_2d_1(self):
    pop = np.array([[0.4, 0.5],
                    [0.6, 0.6],
                    [0.4, 2.0]])
                          
    query = np.array([[0.4, 0.6]])

    soln_idx = np.array([[0, 1]])
    soln_dist = np.array([[0.1, 0.2]])
    
    idx, dist = rbf.pde.knn.k_nearest_neighbors(query, pop, 2)
    self.assertTrue(np.all(soln_idx==idx))
    self.assertTrue(np.allclose(soln_dist, dist))

  def test_nearest_2d_2(self):
    vert = np.array([[0.5, 0.0],
                     [0.5, 1.0]])
    smp = np.array([[0, 1]])
     
    pop = np.array([[0.4, 0.5],
                    [0.6, 0.6],
                    [0.4, 2.0]])
                          
    query = np.array([[0.4, 0.6]])

    soln_idx = np.array([[0, 2]])
    soln_dist = np.array([[0.1, 1.4]])
    idx, dist = rbf.pde.knn.k_nearest_neighbors(query, pop, 2, vert, smp)
    self.assertTrue(np.all(soln_idx==idx))
    self.assertTrue(np.allclose(soln_dist, dist))
