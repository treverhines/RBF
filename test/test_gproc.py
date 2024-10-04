import numpy as np
import rbf.gproc
import unittest
np.random.seed(1)

def allclose(a,b,**kwargs):
  return np.all(np.isclose(a,b,**kwargs))

def test_func1d(x):
  return np.sin(x[:,0])

def test_func1d_diffx(x):
  return np.cos(x[:,0])

def test_func2d(x):
  return np.sin(x[:,0])*np.cos(x[:,1])

def test_func2d_diffx(x):
  return np.cos(x[:,0])*np.cos(x[:,1])

def test_func2d_diffy(x):
  return -np.sin(x[:,0])*np.sin(x[:,1])


class Test(unittest.TestCase):
  def test_run_mean(self):
    # make sure the mean method runs without failure
    gp = rbf.gproc.gpiso('se', var=1.0, eps=1.0)  
    gp.mean([[0.0], [1.0], [2.0]], diff=(0,))
    return 
    
  def test_run_covariance(self):
    # make sure the covariance method runs without failure
    gp = rbf.gproc.gpiso('se', var=1.0, eps=1.0)  
    gp.covariance([[0.0], [1.0], [2.0]], [[0.0], [1.0], [2.0]],
                  diff1=(0,), diff2=(0,))
    return 

  def test_run_meansd(self):
    # make sure the test_mean_and_sigma methods runs without 
    # failure
    gp = rbf.gproc.gpiso('se', var=1.0, eps=1.0)
    gp([[0.0], [1.0], [2.0]])
    return 
  
  def test_run_add(self):
    # make sure the add methods runs without failure 
    gp1 = rbf.gproc.gpiso('se', var=1.0, eps=1.0) 
    gp2 = rbf.gproc.gpiso('se', var=1.0, eps=1.0)
    out = gp1 + gp2
    return 
    
  def test_run_subtract(self):
    # make sure the subtract method runs without failure 
    gp1 = rbf.gproc.gpiso('se', var=1.0, eps=1.0)
    gp2 = rbf.gproc.gpiso('se', var=1.0, eps=1.0)  
    out = gp1 - gp2
    return 

  def test_run_scale(self):
    # make sure the scale method runs without failure 
    gp = rbf.gproc.gpiso('se', var=1.0, eps=1.0)
    out = 1.0*gp
    return 

  def test_run_differentiate(self):
    # make sure the differentiate method runs without failure 
    gp = rbf.gproc.gpiso('se', var=1.0, eps=1.0)
    out = gp.differentiate((1,))
    return 

  def test_run_condition(self):
    # make sure the condition method runs without failure 
    gp = rbf.gproc.gpiso('se', var=1.0, eps=1.0)
    out = gp.condition([[0.0]], [1.0], dcov=[[1.0]], ddiff=(0,))
    return 

  def test_run_sample(self):
    # make sure the sample method runs without failure 
    gp = rbf.gproc.gpiso('se', var=1.0, eps=1.0)
    gp.sample([[0.0], [1.0], [2.0]])
    return 

  def test_run_is_positive_definite(self):
    # make sure the is_positive_definite method runs without failure 
    gp = rbf.gproc.gpiso('se', var=1.0, eps=1.0)
    gp.is_positive_definite([[0.0], [1.0], [2.0]])
    return 
    
  def test_condition(self):  
    # make sure that condition produces an accurate interpolant
    gp = rbf.gproc.gpiso('se', var=2.0, eps=0.5)
    n = 500
    m = 1000
    y = np.linspace(0.0, 10.0, n)[:,None]
    x = np.linspace(0.0, 10.0, m)[:,None]
    sigma = 0.1*np.ones(n)
    d = test_func1d(y) + np.random.normal(0.0,sigma)
    gp = gp.condition(y, d, np.diag(sigma**2))
    mean, std = gp(x)
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
    y = np.linspace(0.0, 10.0, n)[:,None]
    x = np.linspace(0.0, 10.0, m)[:,None]
    sigma = 0.1*np.ones(n)
    d = test_func1d(y) + np.random.normal(0.0, sigma)
    gp = rbf.gproc.gpiso('se', var=2.0, eps=0.5)
    gp = gp.condition(y, d, np.diag(sigma**2)) 
    gp = gp.differentiate((1,))
    mean, std = gp(x)
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
    y = np.linspace(0.0, 10.0, n)[:,None]
    x = np.linspace(0.0, 10.0, m)[:,None]
    sigma = 0.1*np.ones(n)
    d = test_func1d(y) + np.random.normal(0.0, sigma)
    d_diff = test_func1d_diffx(y) + np.random.normal(0.0, sigma)    
    gp = rbf.gproc.gpiso('se', var=2.0, eps=0.5)
    gp = gp.condition(y, d, np.diag(sigma**2)) 
    gp = gp.condition(y, d_diff, np.diag(sigma**2), ddiff=(1,)) 
    mean, std = gp(x)
    res = (mean - test_func1d(x))/std
    mean_chi2 = np.sqrt(res.T.dot(res)/m)
    # the mean chi squared should be close to 1
    self.assertTrue(np.abs(np.log10(mean_chi2)) < 0.2)
    #plt.plot(x,mean,'b-')
    #plt.fill_between(x[:,0],mean-std,mean+std,color='b',alpha=0.2)
    #plt.plot(x,test_func1d(x),'k-')
    #plt.show()

  def test_differentiator(self):
    # make sure the differentiaion decorator works
    @rbf.gproc.differentiator(1e-5)
    def func1(x):
      return np.sin(x[:, 0])*np.cos(x[:, 1])

    def func1_dx0(x):    
      return np.cos(x[:, 0])*np.cos(x[:, 1])

    def func1_dx1(x):    
      return -np.sin(x[:, 0])*np.sin(x[:, 1])

    x = np.random.random((5, 2))    
    dudx0 = func1(x, (1, 0))
    dudx1 = func1(x, (0, 1))
    true_dudx0 = func1_dx0(x)
    true_dudx1 = func1_dx1(x)
    self.assertTrue(np.allclose(dudx0, true_dudx0, atol=1e-4, rtol=1e-4))
    self.assertTrue(np.allclose(dudx1, true_dudx1, atol=1e-4, rtol=1e-4))

  def test_covariance_differentiator_0(self):
    # make sure the covariance differentiaion decorator works
    @rbf.gproc.covariance_differentiator(1e-5)
    def func2(x, y):
      return np.sin(x[:, 0])*np.cos(y[:, 0])

    def func2_dx0(x, y):    
      return np.cos(x[:, 0])*np.cos(y[:, 0])

    def func2_dy0(x, y):    
      return -np.sin(x[:, 0])*np.sin(y[:, 0])

    x = np.random.random((5, 1))    
    y = np.random.random((5, 1))    
    dudx0 = func2(x, y, (1,), (0,))
    dudy0 = func2(x, y, (0,), (1,))
    true_dudx0 = func2_dx0(x, y)
    true_dudy0 = func2_dy0(x, y)
    self.assertTrue(np.allclose(dudx0, true_dudx0, atol=1e-4, rtol=1e-4))
    self.assertTrue(np.allclose(dudy0, true_dudy0, atol=1e-4, rtol=1e-4))

  def test_covariance_differentiator_1(self):
    # make sure the covariance differentiator works for sparse covariance
    # functions
    @rbf.gproc.covariance_differentiator(1e-5)
    def sparse_cov(x,y):
      gp = rbf.gproc.gpiso('spwen12', var=1.0, eps=1.0)
      return gp._covariance(x, y, np.array([0]), np.array([0]))

    def sparse_cov_dx0(x,y):
      gp = rbf.gproc.gpiso('spwen12', var=1.0, eps=1.0)
      return gp._covariance(x, y, np.array([1]), np.array([0]))

    def sparse_cov_dy0(x,y):
      gp = rbf.gproc.gpiso('spwen12', var=1.0, eps=1.0)
      return gp._covariance(x, y, np.array([0]), np.array([1]))

    x = np.random.random((3, 1))
    y = np.random.random((2, 1))
    dudx0 = sparse_cov(x, y, (1,), (0,)).toarray()
    dudy0 = sparse_cov(x, y, (0,), (1,)).toarray()
    true_dudx0 = sparse_cov_dx0(x, y).toarray()
    true_dudy0 = sparse_cov_dy0(x, y).toarray()
    self.assertTrue(np.allclose(dudx0, true_dudx0, atol=1e-4, rtol=1e-4))
    self.assertTrue(np.allclose(dudy0, true_dudy0, atol=1e-4, rtol=1e-4))

  def test_variance_0(self):
    # make sure the variance is equal to the covariance diagonal for dense
    # isotropic GPs
    x = np.random.random((3, 1))
    gp = rbf.gproc.gpiso('se', var=1.0, eps=1.0)
    var1 = gp.variance(x)
    var2 = np.diagonal(gp.covariance(x, x))
    self.assertTrue(np.allclose(var1, var2))

  def test_variance_1(self):
    # make sure the variance is equal to the covariance diagonal for sparse
    # isotropic GPs
    x = np.random.random((3, 1))
    gp = rbf.gproc.gpiso('spwen12', var=1.0, eps=1.0)
    var1 = gp.variance(x)
    var2 = gp.covariance(x, x).diagonal()
    self.assertTrue(np.allclose(var1, var2))

  def test_variance_2(self):
    # make sure the variance is equal to the covariance diagonal for
    # conditioned GPs
    x = np.random.random((3, 1))
    y = np.random.random((2, 1))
    d = np.random.random((2,))
    s = np.random.random((2,))
    gp = rbf.gproc.gpiso('se', var=1.0, eps=1.0)
    gp = gp.condition(y, d, np.diag(s**2))
    var1 = gp.variance(x)
    var2 = gp.covariance(x, x).diagonal()
    self.assertTrue(np.allclose(var1, var2))
  
