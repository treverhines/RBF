import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import rbf.sputils
import unittest

class Test(unittest.TestCase):
  def test_row_norms(self):
    A = sp.rand(100, 100, 0.1)
    A = A.tocsc()

    n1 = spla.norm(A, axis=1, ord=2)
    n2 = rbf.sputils.row_norms(A, order=2)
    self.assertTrue(np.allclose(n1, n2))

    n1 = spla.norm(A, axis=1, ord=1)
    n2 = rbf.sputils.row_norms(A, order=1)
    self.assertTrue(np.allclose(n1, n2))

  def test_divide_rows(self):
    A = sp.rand(100, 10, 0.1)
    
    n = np.random.uniform(1.0, 2.0, (100,))
    B1 = rbf.sputils.divide_rows(A, n)
    B2 = sp.diags(1.0/n).dot(A)
    
    self.assertTrue(np.allclose(B1.A, B2.A))

  def test_divide_rows_inplace(self):
    A = sp.rand(100, 10, 0.1)
    A = A.tocsc()

    n = np.random.uniform(1.0, 2.0, (100,))
    B = sp.diags(1.0/n).dot(A)
    # This should modify A inplace as long as A is csc, csr, or coo
    rbf.sputils.divide_rows(A, n, inplace=True)
    self.assertTrue(np.allclose(A.A, B.A))

  def test_add_rows(self):
    A = sp.rand(100, 10, 1.0)
    B = sp.rand(20, 10, 1.0)
    idx = np.random.choice(range(100), size=(20,), replace=False)
    C = rbf.sputils.add_rows(A, B, idx)

    # this should be equivalent to `add_rows`
    D = A.tocsc(copy=True)
    D[idx, :] += B
    
    self.assertTrue(np.allclose(C.A, D.A))
