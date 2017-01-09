#!/usr/bin/env python
import rbf.basis
import numpy as np
import matplotlib.pyplot as plt
import sympy
import unittest

def test_odd_phs(func):
  ''' 
  makes sure the function value is 1.0 at 1.0 radial distance
  '''
  x = np.array([[1.0,3.0]])
  c = np.array([[1.0,2.0]])
  u = func(x,c)
  return u[0,0]==1.0

def test_even_phs(func):
  ''' 
  makes sure the function value is 0.0 at 1.0 radial distance
  '''
  x = np.array([[1.0,3.0]])
  c = np.array([[1.0,2.0]])
  u = func(x,c)
  return u[0,0]==0.0

def test_x_derivative(func):
  dx = 1e-8
  x1 = np.array([[1.2,3.3]])
  x2 = np.array([[1.2+dx,3.3]])
  c = np.array([[0.5,2.1]])
  eps = np.array([3.0])

  u1 = func(x1,c,eps=eps)
  u2 = func(x2,c,eps=eps)
  diff_num = (u2 - u1)/dx
  diff_true = func(x1,c,eps=eps,diff=(1,0))
  return np.isclose(diff_num[0,0],diff_true[0,0])

def test_y_derivative(func):
  dy = 1e-8
  x1 = np.array([[1.2,3.3]])
  x2 = np.array([[1.2,3.3+dy]])
  c = np.array([[0.5,2.1]])
  eps = np.array([3.0])
  u1 = func(x1,c,eps=eps)
  u2 = func(x2,c,eps=eps)
  diff_num = (u2 - u1)/dy
  diff_true = func(x1,c,eps=eps,diff=(0,1))
  return np.isclose(diff_num[0,0],diff_true[0,0])

class Test(unittest.TestCase):
  # test 1
  def test_even_phs(self):
    self.assertTrue(test_even_phs(rbf.basis.phs2))
    self.assertTrue(test_even_phs(rbf.basis.phs4))
    self.assertTrue(test_even_phs(rbf.basis.phs6))
    self.assertTrue(test_even_phs(rbf.basis.phs8))

  def test_odd_phs(self):
    self.assertTrue(test_odd_phs(rbf.basis.phs1))
    self.assertTrue(test_odd_phs(rbf.basis.phs3))
    self.assertTrue(test_odd_phs(rbf.basis.phs5))
    self.assertTrue(test_odd_phs(rbf.basis.phs7))
    
  def test_x_derivative(self):  
    self.assertTrue(test_x_derivative(rbf.basis.phs1))
    self.assertTrue(test_x_derivative(rbf.basis.phs2))
    self.assertTrue(test_x_derivative(rbf.basis.phs3))
    self.assertTrue(test_x_derivative(rbf.basis.phs4))
    self.assertTrue(test_x_derivative(rbf.basis.phs5))
    self.assertTrue(test_x_derivative(rbf.basis.phs6))
    self.assertTrue(test_x_derivative(rbf.basis.phs7))
    self.assertTrue(test_x_derivative(rbf.basis.phs8))
    self.assertTrue(test_x_derivative(rbf.basis.ga))
    self.assertTrue(test_x_derivative(rbf.basis.imq))
    self.assertTrue(test_x_derivative(rbf.basis.mq))
    self.assertTrue(test_x_derivative(rbf.basis.iq))

  def test_y_derivative(self):  
    self.assertTrue(test_y_derivative(rbf.basis.phs1))
    self.assertTrue(test_y_derivative(rbf.basis.phs2))
    self.assertTrue(test_y_derivative(rbf.basis.phs3))
    self.assertTrue(test_y_derivative(rbf.basis.phs4))
    self.assertTrue(test_y_derivative(rbf.basis.phs5))
    self.assertTrue(test_y_derivative(rbf.basis.phs6))
    self.assertTrue(test_y_derivative(rbf.basis.phs7))
    self.assertTrue(test_y_derivative(rbf.basis.phs8))
    self.assertTrue(test_y_derivative(rbf.basis.ga))
    self.assertTrue(test_y_derivative(rbf.basis.imq))
    self.assertTrue(test_y_derivative(rbf.basis.mq))
    self.assertTrue(test_y_derivative(rbf.basis.iq))

  def test_make_rbf(self):
    r = rbf.basis.get_r()
    # define the imq function and make sure it is equal to 
    # rbf.basis.imq
    imq = rbf.basis.RBF(1/sympy.sqrt(1 + r**2))
    np.random.seed(1)
    x = np.random.random((5,2))
    c = np.random.random((3,2))
    eps = np.random.random(3,)
    out1 = imq(x,c,eps=eps)
    out2 = rbf.basis.imq(x,c,eps=eps)
    check = np.all(np.isclose(out1,out2))
    self.assertTrue(check)


#unittest.main()
