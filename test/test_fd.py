#!/usr/bin/env python
import numpy as np
import rbf.fd
import rbf.basis
import rbf.halton
import unittest
rbf.basis.set_sym_to_num('numpy')

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
  def test_poly_weight_diff0(self):
    x = np.array([1.0])
    nodes = np.array([[1.0],[2.0]])
    diff = (0,)
    w_true = np.array([1.0,0.0])
    w = rbf.fd.poly_diff_weights(x,nodes,diff=diff)
    self.assertTrue(np.all(w==w_true))

  def test_poly_weight_diff1(self):
    x = np.array([1.0])
    nodes = np.array([[1.0],[2.0]])
    diff = (1,)
    w_true = np.array([-1.0,1.0])
    w = rbf.fd.poly_diff_weights(x,nodes,diff=diff)
    self.assertTrue(np.all(w==w_true))

    x = np.array([1.0])
    nodes = np.array([[0.0],[1.0],[2.0]])
    diff = (1,)
    w_true = np.array([-0.5,0.0,0.5])
    w = rbf.fd.poly_diff_weights(x,nodes,diff=diff)
    self.assertTrue(np.all(w==w_true))

  def test_poly_weight_diff2(self):
    x = np.array([1.0])
    nodes = np.array([[0.0],[1.0],[2.0]])
    diff = (2,)
    w_true = np.array([1.0,-2.0,1.0])
    w = rbf.fd.poly_diff_weights(x,nodes,diff=diff)
    self.assertTrue(np.all(w==w_true))

  def test_1d_rbf_weight_diff0(self):
    x = np.array([1.5])
    nodes = np.array([[0.2],[1.2],[2.1]])

    diff = (0,)
    w1 = rbf.fd.poly_diff_weights(x,nodes,diff=diff)
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs1)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs2)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs3)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs4)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs5)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs6)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs7)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs8)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.ga)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.iq)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.imq)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.mq)
    self.assertTrue(np.all(np.isclose(w1,w2)))

  def test_1d_rbf_weight_diff1(self):
    x = np.array([1.5])
    nodes = np.array([[0.2],[1.2],[2.1]])

    diff = (1,)
    w1 = rbf.fd.poly_diff_weights(x,nodes,diff=diff)
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs1)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs2)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs3)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs4)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs5)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs6)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs7)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs8)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.ga)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.iq)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.imq)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.mq)
    self.assertTrue(np.all(np.isclose(w1,w2)))

  def test_1d_rbf_weight_diff1(self):
    x = np.array([1.5])
    nodes = np.array([[0.2],[1.2],[2.1]])

    diff = (2,)
    w1 = rbf.fd.poly_diff_weights(x,nodes,diff=diff)
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs1)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs2)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs3)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs4)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs5)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs6)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs7)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.phs8)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.ga)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.iq)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.imq)
    self.assertTrue(np.all(np.isclose(w1,w2)))
    w2 = rbf.fd.diff_weights(x,nodes,diff=diff,order='max',basis=rbf.basis.mq)
    self.assertTrue(np.all(np.isclose(w1,w2)))
  
  def test_2d_rbf_weight_diffx1(self):
    # estimate derivative of a plane given three points. This should 
    # be exact with polynomial order > 1 
    x = np.array([0.25,0.25])
    nodes = np.array([[0.0,0.0],
                      [1.0,0.0],
                      [0.0,1.0]])

    u = np.array([0.0,2.0,3.0])
    w = rbf.fd.diff_weights(x,nodes,diff=(1,0),
                             basis=rbf.basis.phs1,order=1)
    self.assertTrue(np.isclose(u.dot(w),2.0))

    w = rbf.fd.diff_weights(x,nodes,diff=(1,0),
                             basis=rbf.basis.phs2,order=1)
    self.assertTrue(np.isclose(u.dot(w),2.0))

    w = rbf.fd.diff_weights(x,nodes,diff=(1,0),
                             basis=rbf.basis.phs3,order=1)
    self.assertTrue(np.isclose(u.dot(w),2.0))

    w = rbf.fd.diff_weights(x,nodes,diff=(1,0),
                             basis=rbf.basis.phs4,order=1)
    self.assertTrue(np.isclose(u.dot(w),2.0))

    w = rbf.fd.diff_weights(x,nodes,diff=(1,0),
                             basis=rbf.basis.phs5,order=1)
    self.assertTrue(np.isclose(u.dot(w),2.0))

    w = rbf.fd.diff_weights(x,nodes,diff=(1,0),
                             basis=rbf.basis.phs5,order=1)
    self.assertTrue(np.isclose(u.dot(w),2.0))

    w = rbf.fd.diff_weights(x,nodes,diff=(1,0),
                             basis=rbf.basis.phs6,order=1)
    self.assertTrue(np.isclose(u.dot(w),2.0))

    w = rbf.fd.diff_weights(x,nodes,diff=(1,0),
                             basis=rbf.basis.phs7,order=1)
    self.assertTrue(np.isclose(u.dot(w),2.0))

    w = rbf.fd.diff_weights(x,nodes,diff=(1,0),
                             basis=rbf.basis.phs8,order=1)
    self.assertTrue(np.isclose(u.dot(w),2.0))


  def test_2d_rbf_weight_diffy1(self):
    # estimate derivative of a plane given three points. This should 
    # be exact with polynomial order > 1 
    x = np.array([0.25,0.25])
    nodes = np.array([[0.0,0.0],
                      [1.0,0.0],
                      [0.0,1.0]])

    u = np.array([0.0,2.0,3.0])
    w = rbf.fd.diff_weights(x,nodes,diff=(0,1),
                             basis=rbf.basis.phs1,order=1)
    self.assertTrue(np.isclose(u.dot(w),3.0))

    w = rbf.fd.diff_weights(x,nodes,diff=(0,1),
                             basis=rbf.basis.phs2,order=1)
    self.assertTrue(np.isclose(u.dot(w),3.0))

    w = rbf.fd.diff_weights(x,nodes,diff=(0,1),
                             basis=rbf.basis.phs3,order=1)
    self.assertTrue(np.isclose(u.dot(w),3.0))

    w = rbf.fd.diff_weights(x,nodes,diff=(0,1),
                             basis=rbf.basis.phs4,order=1)
    self.assertTrue(np.isclose(u.dot(w),3.0))

    w = rbf.fd.diff_weights(x,nodes,diff=(0,1),
                             basis=rbf.basis.phs5,order=1)
    self.assertTrue(np.isclose(u.dot(w),3.0))

    w = rbf.fd.diff_weights(x,nodes,diff=(0,1),
                             basis=rbf.basis.phs5,order=1)
    self.assertTrue(np.isclose(u.dot(w),3.0))

    w = rbf.fd.diff_weights(x,nodes,diff=(0,1),
                             basis=rbf.basis.phs6,order=1)
    self.assertTrue(np.isclose(u.dot(w),3.0))

    w = rbf.fd.diff_weights(x,nodes,diff=(0,1),
                             basis=rbf.basis.phs7,order=1)
    self.assertTrue(np.isclose(u.dot(w),3.0))

    w = rbf.fd.diff_weights(x,nodes,diff=(0,1),
                             basis=rbf.basis.phs8,order=1)
    self.assertTrue(np.isclose(u.dot(w),3.0))
  
  def test_2d_rbf_weight_diffx2(self):
    # estimate derivative in f(x,y) = sin(2*pi*x)*cos(2*pi*y). The 
    # accuracy should improve with increasing polynomial order  
    x = np.array([0.5,0.5])
    H = rbf.halton.Halton(2)
    nodes = H(500)
    centers = H(500)
    
    u = test_func2d(nodes)
    diff_true = test_func2d_diffx(x)

    # estimate derivates at x 
    w = rbf.fd.diff_weights(x,nodes,centers=centers,diff=(1,0),
                            basis=rbf.basis.phs1,order='max')
    self.assertTrue(np.isclose(u.dot(w),diff_true))

    w = rbf.fd.diff_weights(x,nodes,centers=centers,diff=(1,0),
                            basis=rbf.basis.phs2,order='max')
    self.assertTrue(np.isclose(u.dot(w),diff_true))

    w = rbf.fd.diff_weights(x,nodes,centers=centers,diff=(1,0),
                            basis=rbf.basis.phs3,order='max')
    self.assertTrue(np.isclose(u.dot(w),diff_true))

    w = rbf.fd.diff_weights(x,nodes,centers=centers,diff=(1,0),
                            basis=rbf.basis.phs4,order='max')
    self.assertTrue(np.isclose(u.dot(w),diff_true))

    w = rbf.fd.diff_weights(x,nodes,centers=centers,diff=(1,0),
                            basis=rbf.basis.phs5,order='max')
    self.assertTrue(np.isclose(u.dot(w),diff_true))

    w = rbf.fd.diff_weights(x,nodes,centers=centers,diff=(1,0),
                            basis=rbf.basis.phs6,order='max')
    self.assertTrue(np.isclose(u.dot(w),diff_true))
    
    w = rbf.fd.diff_weights(x,nodes,centers=centers,diff=(1,0),
                            basis=rbf.basis.phs7,order='max')
    self.assertTrue(np.isclose(u.dot(w),diff_true))

    w = rbf.fd.diff_weights(x,nodes,centers=centers,diff=(1,0),
                            basis=rbf.basis.phs8,order='max')
    self.assertTrue(np.isclose(u.dot(w),diff_true))

  def test_2d_rbf_weight_diffy2(self):
    # estimate derivative in f(x,y) = sin(2*pi*x)*cos(2*pi*y). The 
    # accuracy should improve with increasing polynomial order test 
    # test d/dy
    x = np.array([0.7,0.6])
    H = rbf.halton.Halton(2)
    nodes = H(500)
    centers = H(500) 
    
    u = test_func2d(nodes)
    diff_true = test_func2d_diffy(x)

    # estimate derivates at x 
    w = rbf.fd.diff_weights(x,nodes,centers=centers,diff=(0,1),
                            basis=rbf.basis.phs1,order='max')
    self.assertTrue(np.isclose(u.dot(w),diff_true))

    w = rbf.fd.diff_weights(x,nodes,centers=centers,diff=(0,1),
                            basis=rbf.basis.phs2,order='max')
    self.assertTrue(np.isclose(u.dot(w),diff_true))

    w = rbf.fd.diff_weights(x,nodes,centers=centers,diff=(0,1),
                            basis=rbf.basis.phs3,order='max')
    self.assertTrue(np.isclose(u.dot(w),diff_true))

    w = rbf.fd.diff_weights(x,nodes,centers=centers,diff=(0,1),
                            basis=rbf.basis.phs4,order='max')
    self.assertTrue(np.isclose(u.dot(w),diff_true))

    w = rbf.fd.diff_weights(x,nodes,centers=centers,diff=(0,1),
                            basis=rbf.basis.phs5,order='max')
    self.assertTrue(np.isclose(u.dot(w),diff_true))

    w = rbf.fd.diff_weights(x,nodes,centers=centers,diff=(0,1),
                            basis=rbf.basis.phs6,order='max')
    self.assertTrue(np.isclose(u.dot(w),diff_true))
    
    w = rbf.fd.diff_weights(x,nodes,centers=centers,diff=(0,1),
                            basis=rbf.basis.phs7,order='max')
    self.assertTrue(np.isclose(u.dot(w),diff_true))

    w = rbf.fd.diff_weights(x,nodes,centers=centers,diff=(0,1),
                            basis=rbf.basis.phs8,order='max')
    self.assertTrue(np.isclose(u.dot(w),diff_true))
    
  def test_diff_matrix(self):
    x = np.arange(4.0)[:,None]
    diff_mat = rbf.fd.diff_matrix(x,(2,)).toarray()
    true_mat = np.array([[ 1., -2.,  1.,  0.],
                         [ 1., -2.,  1.,  0.],
                         [ 0.,  1., -2.,  1.],
                         [ 0.,  1., -2.,  1.]])
    self.assertTrue(np.all(np.isclose(diff_mat,true_mat)))
                         
  def test_poly_diff_matrix(self):    
    x = np.arange(4.0)[:,None]
    diff_mat = rbf.fd.poly_diff_matrix(x,(1,)).toarray()
    true_mat = np.array([[-1.,  1.,  0.,  0.],
                         [ 0., -1.,  1.,  0.],
                         [ 0.,  0., -1.,  1.],
                         [ 0.,  0., -1.,  1.]])
    self.assertTrue(np.all(np.isclose(diff_mat,true_mat)))
                         
unittest.main()
