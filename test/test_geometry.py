#!usr/bin/env python
from __future__ import division
import numpy as np
import rbf.geometry
import rbf.halton
import unittest
import rbf.domain
import matplotlib.pyplot as plt

class Test(unittest.TestCase):
  def test_intersection_count_1d(self):
    vert = np.array([[1.0],
                     [2.0],
                     [3.0]])
    smp = np.array([[0],
                    [1],
                    [2]])                     
    start = np.array([[0.5],
                      [1.5],
                      [2.5],
                      [3.5]])
    end = np.array([[2.5],
                    [2.5],
                    [4.0],
                    [4.0]])
    
    soln = np.array([2,1,1,0])
    out = rbf.geometry.intersection_count(start,end,vert,smp)
    self.assertTrue(np.all(soln==out))

  def test_intersection_index_1d(self):
    vert = np.array([[1.0],
                     [2.0],
                     [3.0]])
    smp = np.array([[0],
                    [1],
                    [2]])                     
    start = np.array([[1.5],
                      [2.5]])
    end = np.array([[2.5],
                    [3.5]])

    soln = np.array([1,2])
    out = rbf.geometry.intersection_index(start,end,vert,smp)
    self.assertTrue(np.all(soln==out))

  def test_intersection_count_2d(self):
    # unit square
    vert = np.array([[0.0,0.0],
                     [1.0,0.0],
                     [1.0,1.0],
                     [0.0,1.0]])
    smp = np.array([[0,1],
                    [1,2],
                    [2,3],
                    [3,0]])                     
                    
    start = np.array([[0.5,0.5],
                      [0.5,0.5],
                      [0.5,-1.0],
                      [0.5,-1.0]])
    end = np.array([[0.5,1.5],
                    [0.9,0.9],
                    [0.5,1.5],
                    [1.0,-1.0]])
    
    soln = np.array([1,0,2,0])
    out = rbf.geometry.intersection_count(start,end,vert,smp)
    self.assertTrue(np.all(soln==out))

  def test_intersection_index_2d(self):
    # unit square
    vert = np.array([[0.0,0.0],
                     [1.0,0.0],
                     [1.0,1.0],
                     [0.0,1.0]])

    smp = np.array([[0,1],
                    [1,2],
                    [2,3],
                    [3,0]])                     
                    
    start = np.array([[0.5,0.5],
                      [0.5,-1.0]])

    end = np.array([[0.5,1.5],
                    [0.5,0.9]])
    
    soln = np.array([2,0])
    out = rbf.geometry.intersection_index(start,end,vert,smp)
    self.assertTrue(np.all(soln==out))
    
  def test_contains_1d(self):
    vert = np.array([[1.0],
                     [2.0]])
    smp = np.array([[0],
                    [1]])
    pnts = np.array([[0.5],
                     [1.5],
                     [2.5]])
    soln = np.array([False,True,False])                     
    out = rbf.geometry.contains(pnts,vert,smp)
    self.assertTrue(np.all(out == soln))

  def test_contains_2d(self):
    vert = np.array([[0.0,0.0],
                     [1.0,0.0],
                     [1.0,1.0],
                     [0.0,1.0]])
    smp = np.array([[0,1],
                    [1,2],
                    [2,3],
                    [3,0]])                     

    pnts = np.array([[0.5,0.5],
                     [-0.5,0.5],
                     [0.5,-0.5],
                     [0.5,1.5],
                     [1.5,0.5]])
    soln = np.array([True,False,False,False,False])                     
    out = rbf.geometry.contains(pnts,vert,smp)
    self.assertTrue(np.all(out == soln))
    
  def test_pi1(self):
    # calculate pi through monte carlo simulations
    P = 100000
    vert,smp = rbf.domain.circle(5)    
    pnts = 2*(rbf.halton.halton(P,2) - 0.5)
    is_inside = rbf.geometry.contains(pnts,vert,smp)
    pi_est = 4*sum(is_inside)/P
    self.assertTrue(np.isclose(pi_est,np.pi,atol=1e-2))    

  def test_pi2(self):
    # calculate the area of a circle with the enclosure function
    P = 100000
    vert,smp = rbf.domain.circle(5)    
    pi_est = rbf.geometry.enclosure(vert,smp)
    self.assertTrue(np.isclose(pi_est,np.pi,atol=1e-2))    

  def test_sphere1(self):
    # calculate area of sphere from oriented simplices
    vert,smp = rbf.domain.sphere(5)
    vol = 4*np.pi/3
    vol_est = rbf.geometry.enclosure(vert,smp,orient=False)
    self.assertTrue(np.isclose(vol_est,vol,atol=1e-2))    

  def test_sphere2(self):
    # calculate area of sphere from unoriented simplices
    vert,smp = rbf.domain.sphere(5)
    # mix up simplices
    smp = [s[np.random.choice(range(3),3,replace=False)] for s in smp]
    vol = 4*np.pi/3
    vol_est = rbf.geometry.enclosure(vert,smp,orient=True)
    self.assertTrue(np.isclose(vol_est,vol,atol=1e-2))    

  def test_sphere3(self):
    # calculate area of sphere using monte carlo simulations
    P = 10000
    vert,smp = rbf.domain.sphere(5)
    pnts = 2*(rbf.halton.halton(P,3) - 0.5)
    is_inside = rbf.geometry.contains(pnts,vert,smp)
    vol_est = 8*sum(is_inside)/P
    vol = 4*np.pi/3
    self.assertTrue(np.isclose(vol_est,vol,atol=1e-2))    

unittest.main()    
