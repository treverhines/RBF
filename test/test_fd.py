import numpy as np
import rbf.basis
import rbf.pde.fd
import rbf.pde.halton
import unittest

def test_func2d(x):
  return np.sin(2*np.pi*x[...,0])*np.cos(2*np.pi*x[...,1])

def test_func2d_diffx(x):
  dx = 1e-6
  xpert = np.copy(x)
  xpert[...,0] += dx
  return (test_func2d(xpert) - test_func2d(x))/dx

def test_func2d_diffy(x):
  dy = 1e-6
  xpert = np.copy(x)
  xpert[...,1] += dy
  return (test_func2d(xpert) - test_func2d(x))/dy

class Test(unittest.TestCase):
  def test_2d_rbf_weight_diffx1(self):
    # estimate derivative of a plane given three points. This should 
    # be exact with polynomial order > 1 
    x = np.array([0.25,0.25])
    nodes = np.array([[0.0,0.0],
                      [1.0,0.0],
                      [0.0,1.0]])

    u = np.array([0.0,2.0,3.0])
    w = rbf.pde.fd.weights(x,nodes,(1,0),
                       basis=rbf.basis.phs2)
    self.assertTrue(np.isclose(u.dot(w),2.0))

    w = rbf.pde.fd.weights(x,nodes,(1,0),
                       basis=rbf.basis.phs3)
    self.assertTrue(np.isclose(u.dot(w),2.0))

    w = rbf.pde.fd.weights(x,nodes,(1,0),
                       basis=rbf.basis.phs4)
    self.assertTrue(np.isclose(u.dot(w),2.0))

    w = rbf.pde.fd.weights(x,nodes,(1,0),
                       basis=rbf.basis.phs5)
    self.assertTrue(np.isclose(u.dot(w),2.0))

    w = rbf.pde.fd.weights(x,nodes,(1,0),
                       basis=rbf.basis.phs6)
    self.assertTrue(np.isclose(u.dot(w),2.0))

    w = rbf.pde.fd.weights(x,nodes,(1,0),
                       basis=rbf.basis.phs7)
    self.assertTrue(np.isclose(u.dot(w),2.0))

    w = rbf.pde.fd.weights(x,nodes,(1,0),
                       basis=rbf.basis.phs8)
    self.assertTrue(np.isclose(u.dot(w),2.0))


  def test_2d_rbf_weight_diffy1(self):
    # estimate derivative of a plane given three points. This should 
    # be exact with polynomial order > 1 
    x = np.array([0.25,0.25])
    nodes = np.array([[0.0,0.0],
                      [1.0,0.0],
                      [0.0,1.0]])

    u = np.array([0.0,2.0,3.0])
    w = rbf.pde.fd.weights(x,nodes,(0,1),
                       basis=rbf.basis.phs2)
    self.assertTrue(np.isclose(u.dot(w),3.0))

    w = rbf.pde.fd.weights(x,nodes,(0,1),
                       basis=rbf.basis.phs3)
    self.assertTrue(np.isclose(u.dot(w),3.0))

    w = rbf.pde.fd.weights(x,nodes,(0,1),
                       basis=rbf.basis.phs4)
    self.assertTrue(np.isclose(u.dot(w),3.0))

    w = rbf.pde.fd.weights(x,nodes,(0,1),
                       basis=rbf.basis.phs5)
    self.assertTrue(np.isclose(u.dot(w),3.0))

    w = rbf.pde.fd.weights(x,nodes,(0,1),
                       basis=rbf.basis.phs5)
    self.assertTrue(np.isclose(u.dot(w),3.0))

    w = rbf.pde.fd.weights(x,nodes,(0,1),
                       basis=rbf.basis.phs6)
    self.assertTrue(np.isclose(u.dot(w),3.0))

    w = rbf.pde.fd.weights(x,nodes,(0,1),
                       basis=rbf.basis.phs7)
    self.assertTrue(np.isclose(u.dot(w),3.0))

    w = rbf.pde.fd.weights(x,nodes,(0,1),
                       basis=rbf.basis.phs8)
    self.assertTrue(np.isclose(u.dot(w),3.0))
  
  def test_2d_rbf_weight_diffx2(self):
    # estimate derivative in f(x,y) = sin(2*pi*x)*cos(2*pi*y). The 
    # accuracy should improve with increasing polynomial order  
    x = np.array([0.5,0.5])
    H = rbf.pde.halton.HaltonSequence(2)
    nodes = H(500)
    
    u = test_func2d(nodes)
    diff_true = test_func2d_diffx(x)

    # estimate derivates at x 
    w = rbf.pde.fd.weights(x,nodes,(1,0),
                       basis=rbf.basis.phs2)
    self.assertTrue(np.isclose(u.dot(w),diff_true,atol=1e-2))

    w = rbf.pde.fd.weights(x,nodes,(1,0),
                       basis=rbf.basis.phs3)
    self.assertTrue(np.isclose(u.dot(w),diff_true,atol=1e-2))

    w = rbf.pde.fd.weights(x,nodes,(1,0),
                       basis=rbf.basis.phs4)
    self.assertTrue(np.isclose(u.dot(w),diff_true,atol=1e-2))

    w = rbf.pde.fd.weights(x,nodes,(1,0),
                       basis=rbf.basis.phs5)
    self.assertTrue(np.isclose(u.dot(w),diff_true,atol=1e-2))

    w = rbf.pde.fd.weights(x,nodes,(1,0),
                       basis=rbf.basis.phs6)
    self.assertTrue(np.isclose(u.dot(w),diff_true,atol=1e-2))
    
    w = rbf.pde.fd.weights(x,nodes,(1,0),
                       basis=rbf.basis.phs7)
    self.assertTrue(np.isclose(u.dot(w),diff_true,atol=1e-2))

    w = rbf.pde.fd.weights(x,nodes,(1,0),
                       basis=rbf.basis.phs8)
    self.assertTrue(np.isclose(u.dot(w),diff_true,atol=1e-2))

  def test_2d_rbf_weight_diffy2(self):
    # estimate derivative in f(x,y) = sin(2*pi*x)*cos(2*pi*y). The 
    # accuracy should improve with increasing polynomial order test 
    # test d/dy
    x = np.array([0.7,0.6])
    H = rbf.pde.halton.HaltonSequence(2)
    nodes = H(500)
    
    u = test_func2d(nodes)
    diff_true = test_func2d_diffy(x)

    # estimate derivates at x 
    w = rbf.pde.fd.weights(x,nodes,(0,1),
                       basis=rbf.basis.phs2)
    self.assertTrue(np.isclose(u.dot(w),diff_true,atol=1e-2))

    w = rbf.pde.fd.weights(x,nodes,(0,1),
                       basis=rbf.basis.phs3)
    self.assertTrue(np.isclose(u.dot(w),diff_true,atol=1e-2))

    w = rbf.pde.fd.weights(x,nodes,(0,1),
                       basis=rbf.basis.phs4)
    self.assertTrue(np.isclose(u.dot(w),diff_true,atol=1e-2))

    w = rbf.pde.fd.weights(x,nodes,(0,1),
                       basis=rbf.basis.phs5)
    self.assertTrue(np.isclose(u.dot(w),diff_true,atol=1e-2))

    w = rbf.pde.fd.weights(x,nodes,(0,1),
                       basis=rbf.basis.phs6)
    self.assertTrue(np.isclose(u.dot(w),diff_true,atol=1e-2))
    
    w = rbf.pde.fd.weights(x,nodes,(0,1),
                       basis=rbf.basis.phs7)
    self.assertTrue(np.isclose(u.dot(w),diff_true,atol=1e-2))

    w = rbf.pde.fd.weights(x,nodes,(0,1),
                       basis=rbf.basis.phs8)
    self.assertTrue(np.isclose(u.dot(w),diff_true,atol=1e-2))
    
                         
