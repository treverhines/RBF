#!/usr/bin/env python
import numpy as np
import rbf
import unittest
import matplotlib.pyplot as plt
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
    gp = rbf.gpr.PriorGaussianProcess((1.0,2.0,3.0))  
    gp.mean([[0.0],[1.0],[2.0]],diff=(0,))
    return 
    
  def test_run_covariance(self):
    # make sure the covariance method runs without failure
    gp = rbf.gpr.PriorGaussianProcess((0.0,1.0,1.0))  
    gp.covariance([[0.0],[1.0],[2.0]],[[0.0],[1.0],[2.0]],
                  diff1=(0,),diff2=(0,))
    return 

  def test_run_mean_and_uncertainty(self):
    # make sure the test_mean_and_uncertainty methods runs without 
    # failure
    gp = rbf.gpr.PriorGaussianProcess((0.0,1.0,1.0))  
    gp.mean_and_uncertainty([[0.0],[1.0],[2.0]])
    return 
  
  def test_run_add(self):
    # make sure the add methods runs without failure 
    gp1 = rbf.gpr.PriorGaussianProcess((0.0,1.0,1.0))  
    gp2 = rbf.gpr.PriorGaussianProcess((0.0,1.0,1.0))  
    out = gp1 + gp2
    return 
    
  def test_run_subtract(self):
    # make sure the subtract method runs without failure 
    gp1 = rbf.gpr.PriorGaussianProcess((0.0,1.0,1.0))  
    gp2 = rbf.gpr.PriorGaussianProcess((0.0,1.0,1.0))  
    out = gp1 - gp2
    return 

  def test_run_scale(self):
    # make sure the scale method runs without failure 
    gp = rbf.gpr.PriorGaussianProcess((0.0,1.0,1.0))  
    out = 1.0*gp
    return 

  def test_run_differentiate(self):
    # make sure the differentiate method runs without failure 
    gp = rbf.gpr.PriorGaussianProcess((0.0,1.0,1.0))  
    out = gp.differentiate((1,))
    return 

  def test_run_condition(self):
    # make sure the condition method runs without failure 
    gp = rbf.gpr.PriorGaussianProcess((0.0,1.0,1.0))  
    out = gp.condition([[0.0]],[1.0],sigma=[1.0],obs_diff=(0,))
    return 

  def test_run_recursive_condition(self):
    # make sure the condition method runs without failure 
    gp = rbf.gpr.PriorGaussianProcess((0.0,1.0,1.0))  
    out = gp.recursive_condition(
            [[0.0],[1.0]],[1.0,2.0],sigma=[1.0,1.0],obs_diff=(0,),
            max_chunk=1)
    return 

  def test_run_draw_sample(self):
    # make sure the draw_sample method runs without failure 
    gp = rbf.gpr.PriorGaussianProcess((0.0,1.0,1.0))  
    gp.draw_sample([[0.0],[1.0],[2.0]])
    return 

  def test_run_is_positive_definite(self):
    # make sure the is_positive_definite method runs without failure 
    gp = rbf.gpr.PriorGaussianProcess((0.0,1.0,1.0))  
    gp.is_positive_definite([[0.0],[1.0],[2.0]])
    return 
    
  def test_condition_and_recursive_condition(self):  
    # make sure condition and recursive_condition produce the same 
    # results
    gp = rbf.gpr.PriorGaussianProcess((0.0,1.0,1.0))  
    y = np.array([[0.0],[1.0],[2.0]])
    d = np.array([2.0,3.0,4.0])
    sigma = np.array([1.0,1.5,2.0])
    gp1 = gp.condition(y,d,sigma)
    gp2 = gp.recursive_condition(y,d,sigma,max_chunk=1)
    x = np.linspace(-1.0,3.0,20)[:,None]
    mean1,std1 = gp1(x)
    mean2,std2 = gp2(x)
    self.assertTrue(allclose(mean1,mean2))    
    self.assertTrue(allclose(std1,std2))    

  def test_condition(self):  
    # make sure that condition produces an accurate interpolant
    gp = rbf.gpr.PriorGaussianProcess((0.0,2.0,0.5),order=-1)  
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
    gp = rbf.gpr.PriorGaussianProcess((0.0,2.0,0.5),order=-1)  
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
    gp = rbf.gpr.PriorGaussianProcess((0.0,2.0,0.5),order=-1)  
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
