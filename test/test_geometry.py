#!usr/bin/env python
from __future__ import division
import numpy as np
import rbf.geometry
import rbf.halton
import unittest
import matplotlib.pyplot as plt

def refine(vert,smp):
  new_vert = np.copy(vert)
  new_smp = np.zeros((0,3),dtype=int)
  for s in smp:
    a,b,c = vert[s,:]
    new_vert = np.vstack((new_vert,[a+b]))
    new_vert = np.vstack((new_vert,[b+c]))
    new_vert = np.vstack((new_vert,[a+c]))

    i = new_vert.shape[0] - 3
    j = i + 1
    k = i + 2
    new_smp = np.vstack((new_smp,[   i,   j,   k]))
    new_smp = np.vstack((new_smp,[s[0],   i,   k]))
    new_smp = np.vstack((new_smp,[   i,s[1],   j]))
    new_smp = np.vstack((new_smp,[   k,   j,s[2]]))

  new_vert = new_vert / np.linalg.norm(new_vert,axis=1)[:,None]
  return new_vert,new_smp


def make_icosphere(refinement):
  f = np.sqrt(2.0)/2.0
  vert = np.array([[ 0.0,-1.0, 0.0],
                   [  -f, 0.0,   f],
                   [   f, 0.0,   f],
                   [   f, 0.0,  -f],
                   [  -f, 0.0,  -f],
                   [ 0.0, 1.0, 0.0]])
  smp = np.array([[0,2,1],
                  [0,3,2],
                  [0,4,3],
                  [0,1,4],
                  [5,1,2],
                  [5,2,3],
                  [5,3,4],
                  [5,4,1]])

  for i in range(refinement):
    vert,smp = refine(vert,smp)

  return vert,smp


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
    N = 1000
    P = 100000
    t = np.linspace(0.0,2*np.pi,N)
    x = np.sin(t)
    y = np.cos(t)
    vert = np.array([x,y]).T
    smp = np.array([np.arange(N),np.roll(np.arange(N),-1)]).T
    
    pnts = 2*(rbf.halton.halton(P,2) - 0.5)
    is_inside = rbf.geometry.contains(pnts,vert,smp)
    pi_est = 4*sum(is_inside)/P
    self.assertTrue(np.isclose(pi_est,np.pi,atol=1e-2))    

  def test_pi2(self):
    # calculate the area of a circle with the enclosure function
    N = 1000
    P = 100000
    t = np.linspace(0.0,2*np.pi,N)
    x = np.sin(t)
    y = np.cos(t)
    vert = np.array([x,y]).T
    smp = np.array([np.arange(N),np.roll(np.arange(N),-1)]).T
    pi_est = rbf.geometry.enclosure(vert,smp)
    self.assertTrue(np.isclose(pi_est,np.pi,atol=1e-2))    

  def test_sphere1(self):
    # calculate area of sphere from oriented simplices
    vert,smp = make_icosphere(5)
    vol = 4*np.pi/3
    vol_est = rbf.geometry.enclosure(vert,smp,orient=False)
    self.assertTrue(np.isclose(vol_est,vol,atol=1e-2))    

  def test_sphere2(self):
    # calculate area of sphere from unoriented simplices
    vert,smp = make_icosphere(5)
    # mix up simplices
    smp = [s[np.random.choice(range(3),3,replace=False)] for s in smp]
    vol = 4*np.pi/3
    vol_est = rbf.geometry.enclosure(vert,smp,orient=True)
    self.assertTrue(np.isclose(vol_est,vol,atol=1e-2))    

  def test_sphere3(self):
    # calculate area of sphere using monte carlo simulations
    P = 10000
    vert,smp = make_icosphere(5)
    pnts = 2*(rbf.halton.halton(P,3) - 0.5)
    is_inside = rbf.geometry.contains(pnts,vert,smp)
    vol_est = 8*sum(is_inside)/P
    vol = 4*np.pi/3
    self.assertTrue(np.isclose(vol_est,vol,atol=1e-2))    

unittest.main()    
