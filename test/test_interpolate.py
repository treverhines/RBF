import numpy as np
import rbf.interpolate
import rbf.pde.halton
import unittest


def test_func2d(x):
  return np.sin(x[...,0])*np.cos(x[...,1])


def test_func2d_diffx(x):
  return np.cos(x[:,0])*np.cos(x[:,1])


def test_func2d_diffy(x):
  return -np.sin(x[:,0])*np.sin(x[:,1])


class Test(unittest.TestCase):
  def test_interp(self):
    N = 1000
    P = 1000
    H = rbf.pde.halton.HaltonSequence(2)
    obs = H(N)
    itp = H(P)
    val = test_func2d(obs)

    I = rbf.interpolate.RBFInterpolant(obs,val,phi=rbf.basis.phs3,order=3)
    valitp_est = I(itp)
    valitp_true = test_func2d(itp)
    self.assertTrue(np.allclose(valitp_est,valitp_true,atol=1e-2))

  def test_interp_chunk(self):
    # make sure the interpolation value does not change depending on 
    # the chunk size
    N = 1000
    P = 1000
    H = rbf.pde.halton.HaltonSequence(2)
    obs = H(N)
    itp = H(P)
    val = test_func2d(obs)

    I = rbf.interpolate.RBFInterpolant(obs,val,phi=rbf.basis.phs3,order=3)
    valitp1 = I(itp,chunk_size=10*P)
    valitp2 = I(itp,chunk_size=P)
    valitp3 = I(itp,chunk_size=100)
    valitp4 = I(itp,chunk_size=33)
    valitp5 = I(itp,chunk_size=1)
    
    self.assertTrue(np.all(np.isclose(valitp1,valitp2)))
    self.assertTrue(np.all(np.isclose(valitp1,valitp3)))
    self.assertTrue(np.all(np.isclose(valitp1,valitp4)))
    self.assertTrue(np.all(np.isclose(valitp1,valitp5)))

  def test_interp_diffx(self):
    N = 1000
    P = 1000
    H = rbf.pde.halton.HaltonSequence(2)
    obs = H(N)
    itp = H(P)
    val = test_func2d(obs)

    I = rbf.interpolate.RBFInterpolant(obs,val,phi=rbf.basis.phs3,order=3)
    valitp_est = I(itp,diff=(1,0))
    valitp_true = test_func2d_diffx(itp)
    self.assertTrue(np.allclose(valitp_est,valitp_true,atol=1e-2))

  def test_interp_diffy(self):
    N = 1000
    P = 1000
    H = rbf.pde.halton.HaltonSequence(2)
    obs = H(N)
    itp = H(P)
    val = test_func2d(obs)

    I = rbf.interpolate.RBFInterpolant(obs,val,phi=rbf.basis.phs3,order=3)
    valitp_est = I(itp,diff=(0,1))
    valitp_true = test_func2d_diffy(itp)
    self.assertTrue(np.allclose(valitp_est,valitp_true,atol=1e-2))

  def test_interp_smooth1(self):
    # make sure that smoothing does not hinder the ability to 
    # reproduce a polynomial
    N = 1000
    P = 1000
    H = rbf.pde.halton.HaltonSequence(2)
    obs = H(N)
    itp = H(P)
    # I am adding a zeroth order polynomial and so I should be able to 
    # reproduce a zeroth order function despite the penalty parameter
    val = 4.0 + 0*obs[:,0]
    I = rbf.interpolate.RBFInterpolant(obs,val,sigma=10000.0,
                                       phi=rbf.basis.phs1,order=0)
    valitp_est = I(itp)
    valitp_true = 4.0 + 0.0*itp[:,1] 
    self.assertTrue(np.allclose(valitp_est,valitp_true))

  def test_interp_smooth2(self):
    # make sure that smoothing does not hinder the ability to 
    # reproduce a polynomial
    N = 1000
    P = 1000
    H = rbf.pde.halton.HaltonSequence(2)
    obs = H(N)
    itp = H(P)
    # I am adding a first order polynomial and so I should be able to 
    # reproduce a first order function despite the penalty parameter
    val = 4.0 + 2.0*obs[:,1] + 3.0*obs[:,0]
    I = rbf.interpolate.RBFInterpolant(obs,val,sigma=10000.0,
                                       phi=rbf.basis.phs3,order=1)
    valitp_est = I(itp)
    valitp_true = 4.0 + 2.0*itp[:,1] + 3.0*itp[:,0]
    self.assertTrue(np.allclose(valitp_est,valitp_true))

  def test_interp_smooth3(self):
    # smooth noisy data
    N = 1000
    P = 1000
    H = rbf.pde.halton.HaltonSequence(2)
    obs = H(N)
    itp = H(P)
    val = test_func2d(obs)
    np.random.seed(1)
    val += np.random.normal(0.0,0.1,val.shape)
    
    I = rbf.interpolate.RBFInterpolant(obs,val,sigma=3.0,
                                       phi=rbf.basis.phs3,order=1)
    valitp_est = I(itp)
    valitp_true = test_func2d(itp)
    self.assertTrue(np.allclose(valitp_est,valitp_true,atol=1e-1))
    
  def test_extrapolate(self):
    # make sure that the extrapolate key word is working    
    N = 1000
    H = rbf.pde.halton.HaltonSequence(2)
    obs = H(N)
    val = test_func2d(obs)
    itp = np.array([[0.5,0.5],
                    [0.5,1.5],
                    [0.5,-0.5],
                    [1.5,0.5],
                    [-0.5,0.5]])
    I = rbf.interpolate.RBFInterpolant(obs,val,phi=rbf.basis.phs3,order=1,
                                       extrapolate=False)
    out = I(itp)
    soln_true = np.array([False,True,True,True,True])
    self.assertTrue(np.all(np.isnan(out) == soln_true))
    
  def test_weight(self):
    # give an outlier zero weight and make sure the interpolant is not
    # affected.
    N = 1000
    P = 1000
    H = rbf.pde.halton.HaltonSequence(2)
    obs = H(N)
    itp = H(P)
    val = test_func2d(obs)
    val[0] += 100.0
    sigma = np.zeros(N)
    sigma[0] = np.inf
    I = rbf.interpolate.RBFInterpolant(obs,val,sigma=sigma,
                                       phi=rbf.basis.phs3,order=1)    
    valitp_est = I(itp)
    valitp_true = test_func2d(itp)
    self.assertTrue(np.allclose(valitp_est,valitp_true,atol=1e-2))
    
  def test_sparse(self):
    # make sure the RBFInterpolant works with sparse RBFs
    N = 1000
    P = 1000
    H = rbf.pde.halton.HaltonSequence(2)
    obs = H(N)
    itp = H(P)
    val = test_func2d(obs)
    I = rbf.interpolate.RBFInterpolant(obs,val,
                                       phi=rbf.basis.spwen31,order=1,
                                       eps=0.5)    
    valitp_est = I(itp)
    valitp_true = test_func2d(itp)
    self.assertTrue(np.allclose(valitp_est,valitp_true,atol=1e-2))

#unittest.main()    
    

