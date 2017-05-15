#!/usr/bin/env python
import numpy as np
import rbf
import unittest
import matplotlib.pyplot as plt
import scipy.sparse as sp
np.random.seed(1)

def allclose(a,b,**kwargs):
  return np.all(np.isclose(a,b,**kwargs))

def test_func1d(x):
  return np.sin(x[:,0])

def test_func1d_diffx(x):
  return np.cos(x[:,0])

def test_func1d(x):
  return np.sin(x[:,0])

def test_func2d(x):
  return np.sin(x[:,0])*np.cos(x[:,1])

def test_func2d_diffx(x):
  return np.cos(x[:,0])*np.cos(x[:,1])

def test_func2d_diffy(x):
  return -np.sin(x[:,0])*np.sin(x[:,1])


class Test(unittest.TestCase):
  def test_run_mean(self):
    # make sure the mean method runs without failure
    gp = rbf.gauss.gpiso(rbf.basis.se,(1.0,2.0,3.0))  
    gp.mean([[0.0],[1.0],[2.0]],diff=(0,))
    return 
    
  def test_run_covariance(self):
    # make sure the covariance method runs without failure
    gp = rbf.gauss.gpiso(rbf.basis.se,(0.0,1.0,1.0))  
    gp.covariance([[0.0],[1.0],[2.0]],[[0.0],[1.0],[2.0]],
                  diff1=(0,),diff2=(0,))
    return 

  def test_run_meansd(self):
    # make sure the test_mean_and_sigma methods runs without 
    # failure
    gp = rbf.gauss.gpiso(rbf.basis.se,(0.0,1.0,1.0))  
    gp.meansd([[0.0],[1.0],[2.0]])
    return 
  
  def test_run_add(self):
    # make sure the add methods runs without failure 
    gp1 = rbf.gauss.gpiso(rbf.basis.se,(0.0,1.0,1.0))  
    gp2 = rbf.gauss.gpiso(rbf.basis.se,(0.0,1.0,1.0))  
    out = gp1 + gp2
    return 
    
  def test_run_subtract(self):
    # make sure the subtract method runs without failure 
    gp1 = rbf.gauss.gpiso(rbf.basis.se,(0.0,1.0,1.0))  
    gp2 = rbf.gauss.gpiso(rbf.basis.se,(0.0,1.0,1.0))  
    out = gp1 - gp2
    return 

  def test_run_scale(self):
    # make sure the scale method runs without failure 
    gp = rbf.gauss.gpiso(rbf.basis.se,(0.0,1.0,1.0))  
    out = 1.0*gp
    return 

  def test_run_differentiate(self):
    # make sure the differentiate method runs without failure 
    gp = rbf.gauss.gpiso(rbf.basis.se,(0.0,1.0,1.0))  
    out = gp.differentiate((1,))
    return 

  def test_run_condition(self):
    # make sure the condition method runs without failure 
    gp = rbf.gauss.gpiso(rbf.basis.se,(0.0,1.0,1.0))  
    out = gp.condition([[0.0]],[1.0],sigma=[1.0],obs_diff=(0,))
    return 

  def test_run_sample(self):
    # make sure the sample method runs without failure 
    gp = rbf.gauss.gpiso(rbf.basis.se,(0.0,1.0,1.0))  
    gp.sample([[0.0],[1.0],[2.0]])
    return 

  def test_run_is_positive_definite(self):
    # make sure the is_positive_definite method runs without failure 
    gp = rbf.gauss.gpiso(rbf.basis.se,(0.0,1.0,1.0))  
    gp.is_positive_definite([[0.0],[1.0],[2.0]])
    return 
    
  def test_condition(self):  
    # make sure that condition produces an accurate interpolant
    gp = rbf.gauss.gpiso(rbf.basis.se,(0.0,2.0,0.5))  
    n = 500
    m = 1000
    y = np.linspace(0.0,10.0,n)[:,None]
    x = np.linspace(0.0,10.0,m)[:,None]
    sigma = 0.1*np.ones(n)
    d = test_func1d(y) + np.random.normal(0.0,sigma)
    gp = gp.condition(y,d,sigma) 
    mean,std = gp(x)
    res = (mean - test_func1d(x))/std
    mean_chi2 = np.sqrt(res.T.dot(res)/m)
    # the mean chi squared should be close to 1
    self.assertTrue(np.abs(np.log10(mean_chi2)) < 0.2)
    #plt.plot(x,mean,'b-')
    #plt.fill_between(x[:,0],mean-std,mean+std,color='b',alpha=0.2)
    #plt.plot(x,test_func1d(x),'k-')
    #plt.show()

  def test_condition_and_differentiate(self):  
    # make sure that condition produces an accurate estimate of the 
    # derivative
    n = 500
    m = 1000
    y = np.linspace(0.0,10.0,n)[:,None]
    x = np.linspace(0.0,10.0,m)[:,None]
    sigma = 0.1*np.ones(n)
    d = test_func1d(y) + np.random.normal(0.0,sigma)
    gp = rbf.gauss.gpiso(rbf.basis.se,(0.0,2.0,0.5))  
    gp = gp.condition(y,d,sigma) 
    gp = gp.differentiate((1,))
    mean,std = gp(x)
    res = (mean - test_func1d_diffx(x))/std
    mean_chi2 = np.sqrt(res.T.dot(res)/m)
    # the mean chi squared should be close to 1
    self.assertTrue(np.abs(np.log10(mean_chi2)) < 0.2)
    #plt.plot(x,mean,'b-')
    #plt.fill_between(x[:,0],mean-std,mean+std,color='b',alpha=0.2)
    #plt.plot(x,test_func1d_diffx(x),'k-')
    #plt.show()

  def test_condition_with_derivatives(self):  
    # make sure that conditioning with derivative constraints produces 
    # an accurate interpolant
    n = 500
    m = 1000
    y = np.linspace(0.0,10.0,n)[:,None]
    x = np.linspace(0.0,10.0,m)[:,None]
    sigma = 0.1*np.ones(n)
    d = test_func1d(y) + np.random.normal(0.0,sigma)
    d_diff = test_func1d_diffx(y) + np.random.normal(0.0,sigma)    
    gp = rbf.gauss.gpiso(rbf.basis.se,(0.0,2.0,0.5))  
    gp = gp.condition(y,d,sigma) 
    gp = gp.condition(y,d_diff,sigma,obs_diff=(1,)) 
    mean,std = gp(x)
    res = (mean - test_func1d(x))/std
    mean_chi2 = np.sqrt(res.T.dot(res)/m)
    # the mean chi squared should be close to 1
    self.assertTrue(np.abs(np.log10(mean_chi2)) < 0.2)
    #plt.plot(x,mean,'b-')
    #plt.fill_between(x[:,0],mean-std,mean+std,color='b',alpha=0.2)
    #plt.plot(x,test_func1d(x),'k-')
    #plt.show()

  def test_sparse_factor_solve_A(self):
    n = 100
    A = sp.rand(n,n,density=0.2)
    A = A.T.dot(A).tocsc()
    b = np.random.random((n,))
    factor = rbf.gauss._SparseFactor(A)
    x1 = factor.solve_A(b)
    x2 = np.linalg.solve(A.A,b)
    self.assertTrue(np.all(np.isclose(x1,x2)))
  
  def test_sparse_factor_solve_L(self):
    n = 100
    A = sp.rand(n,n,density=0.2)
    A = A.T.dot(A).tocsc()
    b = np.random.random((n,))
    factor = rbf.gauss._SparseFactor(A)
    x1 = factor.solve_L(b)
    x2 = np.linalg.solve(factor.L().A,b)
    self.assertTrue(np.all(np.isclose(x1,x2)))

  def test_sparse_factor_L(self):
    n = 100
    A = sp.rand(n,n,density=0.2)
    A = A.T.dot(A).tocsc()
    factor = rbf.gauss._SparseFactor(A)
    L = factor.L()
    A2 = L.dot(L.T)
    self.assertTrue(np.all(np.isclose(A.A,A2.A)))
  
  def test_sparse_factor_log_det_A(self):
    n = 100
    A = sp.rand(n,n,density=0.2)
    A = A.T.dot(A).tocsc()
    factor = rbf.gauss._SparseFactor(A)
    x1 = factor.log_det_A()
    x2 = np.log(np.linalg.det(A.A))
    self.assertTrue(np.isclose(x1,x2))
  
  def test_dense_factor_solve_A(self):
    n = 100
    A = np.random.random((n,n))
    A = A.T.dot(A)
    b = np.random.random((n,))
    factor = rbf.gauss._DenseFactor(A)
    x1 = factor.solve_A(b)
    x2 = np.linalg.solve(A,b)
    self.assertTrue(np.all(np.isclose(x1,x2)))
  
  def test_dense_factor_solve_L(self):
    n = 100
    A = np.random.random((n,n))
    A = A.T.dot(A)
    b = np.random.random((n,))
    factor = rbf.gauss._DenseFactor(A)
    x1 = factor.solve_L(b)
    x2 = np.linalg.solve(factor.L(),b)
    self.assertTrue(np.all(np.isclose(x1,x2)))
  
  def test_dense_factor_L(self):
    n = 100
    A = np.random.random((n,n))
    A = A.T.dot(A)
    factor = rbf.gauss._DenseFactor(A)
    L = factor.L()
    A2 = L.dot(L.T)
    self.assertTrue(np.all(np.isclose(A,A2)))

  def test_dense_factor_log_det_A(self):
    n = 100
    A = np.random.random((n,n))
    A = A.T.dot(A)
    factor = rbf.gauss._DenseFactor(A)
    x1 = factor.log_det_A()
    x2 = np.log(np.linalg.det(A))
    self.assertTrue(np.isclose(x1,x2))

  def test_partitioned_factor(self):    
    A = np.random.random((4,4))
    A = A.T.dot(A) # A is now P.D.
    B = np.random.random((4,2))
    a = np.random.random((4,))
    b = np.random.random((2,))
    Cfact = rbf.gauss._PartitionedFactor(A,B)
    soln1a,soln1b = Cfact.solve(a,b)
    soln1 = np.hstack((soln1a,soln1b))
    Cinv = np.linalg.inv(
             np.vstack(
               (np.hstack((A,B)),
                np.hstack((B.T,np.zeros((2,2)))))))
    soln2 = Cinv.dot(np.hstack((a,b)))
    self.assertTrue(np.all(np.isclose(soln1,soln2)))

  def test_partitioned_factor_sparse(self):    
    A = np.random.random((4,4))
    A = A.T.dot(A) # A is now P.D.
    A = sp.csc_matrix(A) # A is now sparse
    B = np.random.random((4,2))
    a = np.random.random((4,))
    b = np.random.random((2,))
    Cfact = rbf.gauss._PartitionedFactor(A,B)
    soln1a,soln1b = Cfact.solve(a,b)
    soln1 = np.hstack((soln1a,soln1b))
    Cinv = np.linalg.inv(
             np.vstack(
               (np.hstack((A.A,B)),
                np.hstack((B.T,np.zeros((2,2)))))))
    soln2 = Cinv.dot(np.hstack((a,b)))
    self.assertTrue(np.all(np.isclose(soln1,soln2)))
    
                  
