#!/usr/bin/env python
import numpy as np
import rbf.interpolate
import rbf.halton
import unittest
import matplotlib.pyplot as plt

def allclose(a,b,**kwargs):
  return np.all(np.isclose(a,b,**kwargs))

def test_func2d(x):
  return np.sin(x[...,0])*np.cos(x[...,1])

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
  def test_interp(self):
    N = 1000
    P = 1000
    H = rbf.halton.Halton(2)
    obs = H(N)
    itp = H(P)
    val = test_func2d(obs)

    I = rbf.interpolate.RBFInterpolant(obs,val,basis=rbf.basis.phs3,order=3)
    valitp_est = I(itp)
    valitp_true = test_func2d(itp)
    self.assertTrue(allclose(valitp_est,valitp_true,atol=1e-2))

  def test_interp_chunk(self):
    # make sure the interpolation value does not change depending on 
    # the chunk size
    N = 1000
    P = 1000
    H = rbf.halton.Halton(2)
    obs = H(N)
    itp = H(P)
    val = test_func2d(obs)

    I = rbf.interpolate.RBFInterpolant(obs,val,basis=rbf.basis.phs3,order=3)
    valitp1 = I(itp,max_chunk=10*P)
    valitp2 = I(itp,max_chunk=P)
    valitp3 = I(itp,max_chunk=100)
    valitp4 = I(itp,max_chunk=33)
    valitp5 = I(itp,max_chunk=1)
    
    self.assertTrue(np.all(valitp1==valitp2))
    self.assertTrue(np.all(valitp1==valitp3))
    self.assertTrue(np.all(valitp1==valitp4))
    self.assertTrue(np.all(valitp1==valitp5))

  def test_interp_diffx(self):
    N = 1000
    P = 1000
    H = rbf.halton.Halton(2)
    obs = H(N)
    itp = H(P)
    val = test_func2d(obs)

    I = rbf.interpolate.RBFInterpolant(obs,val,basis=rbf.basis.phs3,order=3)
    valitp_est = I(itp,diff=(1,0))
    valitp_true = test_func2d_diffx(itp)
    self.assertTrue(allclose(valitp_est,valitp_true,atol=1e-2))

  def test_interp_diffy(self):
    N = 1000
    P = 1000
    H = rbf.halton.Halton(2)
    obs = H(N)
    itp = H(P)
    val = test_func2d(obs)

    I = rbf.interpolate.RBFInterpolant(obs,val,basis=rbf.basis.phs3,order=3)
    valitp_est = I(itp,diff=(0,1))
    valitp_true = test_func2d_diffy(itp)
    self.assertTrue(allclose(valitp_est,valitp_true,atol=1e-2))

  def test_interp_smooth1(self):
    # make sure that smoothing does not hinder the ability to 
    # reproduce a polynomial
    N = 1000
    P = 1000
    H = rbf.halton.Halton(2)
    obs = H(N)
    itp = H(P)
    # I am adding a zeroth order polynomial and so I should be able to 
    # reproduce a zeroth order function despite the penalty parameter
    val = 4.0 + 0*obs[:,0]
    I = rbf.interpolate.RBFInterpolant(obs,val,basis=rbf.basis.phs3,order=0,
                                       penalty=10000.0)
    valitp_est = I(itp)
    valitp_true = 4.0 + 0.0*itp[:,1] 
    self.assertTrue(allclose(valitp_est,valitp_true))

  def test_interp_smooth2(self):
    # make sure that smoothing does not hinder the ability to 
    # reproduce a polynomial
    N = 1000
    P = 1000
    H = rbf.halton.Halton(2)
    obs = H(N)
    itp = H(P)
    # I am adding a first order polynomial and so I should be able to 
    # reproduce a first order function despite the penalty parameter
    val = 4.0 + 2.0*obs[:,1] + 3.0*obs[:,0]
    I = rbf.interpolate.RBFInterpolant(obs,val,basis=rbf.basis.phs3,order=1,
                                       penalty=10000.0)
    valitp_est = I(itp)
    valitp_true = 4.0 + 2.0*itp[:,1] + 3.0*itp[:,0]
    self.assertTrue(allclose(valitp_est,valitp_true))

  def test_interp_smooth3(self):
    # smooth noisy data
    N = 1000
    P = 1000
    H = rbf.halton.Halton(2)
    obs = H(N)
    itp = H(P)
    val = test_func2d(obs)
    np.random.seed(1)
    val += np.random.normal(0.0,0.1,val.shape)
    
    I = rbf.interpolate.RBFInterpolant(obs,val,basis=rbf.basis.phs3,order=1,
                                       penalty=3.0)
    valitp_est = I(itp)
    valitp_true = test_func2d(itp)
    misfit = np.max(np.abs(valitp_est-valitp_true))
    self.assertTrue(allclose(valitp_est,valitp_true,atol=1e-1))
    
  def test_extrapolate(self):
    # make sure that the extrapolate key word is working    
    N = 1000
    H = rbf.halton.Halton(2)
    
    obs = H(N)
    val = test_func2d(obs)
    itp = np.array([[0.5,0.5],
                    [0.5,1.5],
                    [0.5,-0.5],
                    [1.5,0.5],
                    [-0.5,0.5]])
    I = rbf.interpolate.RBFInterpolant(obs,val,basis=rbf.basis.phs3,order=1,
                                       extrapolate=False,fill=np.nan)
    out = I(itp)
    soln_true = np.array([False,True,True,True,True])
    self.assertTrue(np.all(np.isnan(out) == soln_true))
    
  def test_weight(self):
    # give an outlier zero weight and make sure the interpolant is not 
    # affected. add a slight penalty to ensure a non-singular matrix
    N = 1000
    P = 1000
    H = rbf.halton.Halton(2)
    obs = H(N)
    itp = H(P)
    val = test_func2d(obs)
    val[0] += 100.0
    
    weight = np.ones(N)
    weight[0] = 0.0
    I = rbf.interpolate.RBFInterpolant(obs,val,weight=weight,penalty=0.01,
                                       basis=rbf.basis.phs3,order=1)
    
    valitp_est = I(itp)
    valitp_true = test_func2d(itp)
    self.assertTrue(allclose(valitp_est,valitp_true,atol=1e-2))
    

unittest.main()    
    

