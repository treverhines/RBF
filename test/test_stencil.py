#!/usr/bin/env python
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
    d = np.array([[0.1,0.4]])
    nest,dest = rbf.stencil.nearest(query,pop,2)
    self.assertTrue(np.all(n==nest))
    self.assertTrue(np.all(np.isclose(d,dest)))

  def test_nearest_1d_2(self):
    vert = np.array([[2.4]])
    smp = np.array([[0]])
     
    pop = np.array([[1.0],
                    [2.0],
                    [2.5]])
                          
    query = np.array([[2.1]])

    n = np.array([[1,0]])
    d = np.array([[0.1,1.1]])
    nest,dest = rbf.stencil.nearest(query,pop,2,vert,smp)
    self.assertTrue(np.all(n==nest))
    self.assertTrue(np.all(np.isclose(d,dest)))

  def test_nearest_2d_1(self):
    pop = np.array([[0.4,0.5],
                    [0.6,0.6],
                    [0.4,2.0]])
                          
    query = np.array([[0.4,0.6]])

    n = np.array([[0,1]])
    d = np.array([[0.1,0.2]])
    nest,dest = rbf.stencil.nearest(query,pop,2)
    self.assertTrue(np.all(n==nest))
    self.assertTrue(np.all(np.isclose(d,dest)))

  def test_nearest_2d_2(self):
    vert = np.array([[0.5,0.0],
                     [0.5,1.0]])
    smp = np.array([[0,1]])
     
    pop = np.array([[0.4,0.5],
                    [0.6,0.6],
                    [0.4,2.0]])
                          
    query = np.array([[0.4,0.6]])

    n = np.array([[0,2]])
    d = np.array([[0.1,1.4]])
    nest,dest = rbf.stencil.nearest(query,pop,2,vert,smp)
    self.assertTrue(np.all(n==nest))
    self.assertTrue(np.all(np.isclose(d,dest)))

  def test_nearest_and_naive_nearest_2d(self):
    np.random.seed(1)
    vert = np.random.random((10,2))
    smp = np.arange(10).reshape((5,2))
    
    pnt1 = np.random.random((100,2))
    pnt2 = np.random.random((100,2))
    n1,d1 = rbf.stencil.nearest(pnt1,pnt2,3,vert,smp)
    n2,d2 = rbf.stencil._naive_nearest(pnt1,pnt2,3,vert,smp)
    n1 = np.sort(n1,axis=1)
    n2 = np.sort(n2,axis=1)
    d1 = np.sort(d1,axis=1)
    d2 = np.sort(d2,axis=1)
    self.assertTrue(np.all(n1==n2))
    self.assertTrue(np.all(np.isclose(n1,n2)))

  def test_nearest_and_naive_nearest_3d(self):
    np.random.seed(1)
    vert = np.random.random((12,3))
    smp = np.arange(12).reshape((4,3))
    
    pnt1 = np.random.random((100,3))
    pnt2 = np.random.random((100,3))
    n1,d1 = rbf.stencil.nearest(pnt1,pnt2,3,vert,smp)
    n2,d2 = rbf.stencil._naive_nearest(pnt1,pnt2,3,vert,smp)
    n1 = np.sort(n1,axis=1)
    n2 = np.sort(n2,axis=1)
    d1 = np.sort(d1,axis=1)
    d2 = np.sort(d2,axis=1)
    self.assertTrue(np.all(n1==n2))
    self.assertTrue(np.all(np.isclose(n1,n2)))

#unittest.main()
