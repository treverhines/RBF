import rbf.basis
import rbf.poly
import numpy as np
import sympy
import unittest

def test_positive_definite(phi, order=None, dim=2, ntests=100):
    # generate a random vector to test if the RBF is (conditionally) positive
    # definite
    for _ in range(ntests):
        x = np.random.uniform(0.0, 1.0, (50, dim))
        c = np.random.normal(0.0, 1.0, (50,))
        if order is not None:
            # modify the random vector c so that P^Tc = 0
            M = np.eye(50)
            P = rbf.poly.mvmonos(x, rbf.poly.monomial_powers(order - 1, dim))
            M[:P.shape[1]] = P.T
            c[:P.shape[1]] = 0.0
            c = np.linalg.solve(M, c)

        if c.dot(phi(x, x)).dot(c) < 0.0:
            return False

    return True

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
  def test_conditional_positive_definite(self):
    self.assertTrue(test_positive_definite(rbf.basis.phs1, order=1))
    self.assertTrue(test_positive_definite(rbf.basis.phs2, order=2))
    self.assertTrue(test_positive_definite(rbf.basis.phs3, order=2))
    self.assertTrue(test_positive_definite(rbf.basis.phs4, order=3))
    self.assertTrue(test_positive_definite(rbf.basis.phs5, order=3))
    self.assertTrue(test_positive_definite(rbf.basis.phs6, order=4))
    self.assertTrue(test_positive_definite(rbf.basis.phs7, order=4))
    self.assertTrue(test_positive_definite(rbf.basis.phs8, order=5))
    self.assertTrue(test_positive_definite(rbf.basis.mq, order=1))

  def test_positive_definite(self):
    self.assertTrue(test_positive_definite(rbf.basis.imq))
    self.assertTrue(test_positive_definite(rbf.basis.iq))
    self.assertTrue(test_positive_definite(rbf.basis.ga))
    self.assertTrue(test_positive_definite(rbf.basis.exp))
    self.assertTrue(test_positive_definite(rbf.basis.se))
    self.assertTrue(test_positive_definite(rbf.basis.mat32))
    self.assertTrue(test_positive_definite(rbf.basis.mat52))

  def test_wendland_positive_definite(self):
    self.assertTrue(test_positive_definite(rbf.basis.wen10, dim=1))
    self.assertTrue(test_positive_definite(rbf.basis.wen11, dim=1))
    self.assertTrue(test_positive_definite(rbf.basis.wen12, dim=1))

    self.assertTrue(test_positive_definite(rbf.basis.wen30, dim=1))
    self.assertTrue(test_positive_definite(rbf.basis.wen31, dim=1))
    self.assertTrue(test_positive_definite(rbf.basis.wen32, dim=1))

    self.assertTrue(test_positive_definite(rbf.basis.wen30, dim=2))
    self.assertTrue(test_positive_definite(rbf.basis.wen31, dim=2))
    self.assertTrue(test_positive_definite(rbf.basis.wen32, dim=2))

    self.assertTrue(test_positive_definite(rbf.basis.wen30, dim=3))
    self.assertTrue(test_positive_definite(rbf.basis.wen31, dim=3))
    self.assertTrue(test_positive_definite(rbf.basis.wen32, dim=3))

  def test_x_derivative(self):
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

  def test_wendland_limits(self):
    # make sure the provided limits for the centers of the wendland
    # functions are correct
    phis = [rbf.basis.wen10,
            rbf.basis.wen11,
            rbf.basis.wen12,
            rbf.basis.wen30,
            rbf.basis.wen31,
            rbf.basis.wen32,
            rbf.basis.spwen30,
            rbf.basis.spwen31,
            rbf.basis.spwen32]
    eps = 2.0
    for phi in phis:
      tol = phi.tol
      dx = 1.1*float(tol.subs({rbf.basis.get_eps():eps}))
      for k in phi.limits.keys():
        dim = len(k)
        c = np.zeros((1, dim))
        center_val = phi(c, c, diff=k, eps=eps)[0,0]
        center_plus_dx_val = phi(c, c+dx, diff=k, eps=eps)[0,0]
        diff = np.abs(center_val - center_plus_dx_val)
        self.assertTrue(diff < 1.0e-4)

  def test_precompiled(self):
    # make sure the precompiled numerical functions give the same results as
    # the ones produced at runtime.
    rbf.basis.clear_rbf_caches()
    rbf.basis.add_precompiled_to_rbf_caches()
    try:
        for inst in rbf.basis._PREDEFINED.values():
            for diff in inst._cache.keys():
                x = np.random.random((5, len(diff)))
                y = np.random.random((10, len(diff)))
                out1 = inst(x, y)
                inst._add_diff_to_cache(diff)
                out2 = inst(x, y)
                if isinstance(inst, rbf.basis.SparseRBF):
                    self.assertTrue(np.allclose(out1.A, out2.A))
                else:
                    self.assertTrue(np.allclose(out1, out2))
    finally:
        # make sure it returns to the initial state
        rbf.basis.add_precompiled_to_rbf_caches()
    
  def test_matern_limits(self):
    # make sure the provided limits for the centers of the matern
    # functions are correct
    phis = [rbf.basis.mat32,
            rbf.basis.mat52]
    eps = 2.0
    for phi in phis:
      tol = phi.tol
      dx = 1.1*float(tol.subs({rbf.basis.get_eps():eps}))
      for k in phi.limits.keys():
        dim = len(k)
        c = np.zeros((1, dim))
        center_val = phi(c, c, diff=k, eps=eps)[0,0]
        center_plus_dx_val = phi(c, c+dx, diff=k, eps=eps)[0,0]
        diff = np.abs(center_val - center_plus_dx_val)
        self.assertTrue(diff < 1.0e-4)

#unittest.main()
