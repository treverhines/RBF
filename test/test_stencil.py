import numpy as np
import rbf.stencil
import unittest

class Test(unittest.TestCase):
  def test_nearest_1d_1(self):
    pop = np.array([[1.0],
                    [2.0],
                    [2.5]])
                          
    query = np.array([[2.1]])
    n = np.array([[1,2]])
    nest = rbf.stencil.stencil_network(query,pop,2)
    self.assertTrue(np.all(n==nest))

  def test_nearest_1d_2(self):
    vert = np.array([[2.4]])
    smp = np.array([[0]])
     
    pop = np.array([[1.0],
                    [2.0],
                    [2.5]])
                          
    query = np.array([[2.1]])

    n = np.array([[1,0]])
    nest = rbf.stencil.stencil_network(query,pop,2,vert,smp)
    self.assertTrue(np.all(n==nest))

  def test_nearest_2d_1(self):
    pop = np.array([[0.4,0.5],
                    [0.6,0.6],
                    [0.4,2.0]])
                          
    query = np.array([[0.4,0.6]])

    n = np.array([[0,1]])
    nest = rbf.stencil.stencil_network(query,pop,2)
    self.assertTrue(np.all(n==nest))

  def test_nearest_2d_2(self):
    vert = np.array([[0.5,0.0],
                     [0.5,1.0]])
    smp = np.array([[0,1]])
     
    pop = np.array([[0.4,0.5],
                    [0.6,0.6],
                    [0.4,2.0]])
                          
    query = np.array([[0.4,0.6]])

    n = np.array([[0,2]])
    nest = rbf.stencil.stencil_network(query,pop,2,vert,smp)
    self.assertTrue(np.all(n==nest))


