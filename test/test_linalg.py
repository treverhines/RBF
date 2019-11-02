import numpy as np
import rbf.linalg
import unittest
import scipy.sparse as sp
np.random.seed(1)


class Test(unittest.TestCase):
  def test_sparse_solver(self):
    n = 100
    A = sp.rand(n, n, density=0.2)
    A = A.tocsc()
    
    b = np.random.random((n,))
    x1 = np.linalg.solve(A.A, b)
    x2 = rbf.linalg._SparseSolver(A).solve(b)
    self.assertTrue(np.allclose(x1, x2))

  def test_dense_solver(self):
    n = 100
    A = np.random.random((n, n))
    b = np.random.random((n,))
    x1 = np.linalg.solve(A, b)
    x2 = rbf.linalg._DenseSolver(A).solve(b)
    self.assertTrue(np.allclose(x1, x2))
      
  def test_sparse_pos_def_solve(self):
    if not rbf.linalg.HAS_CHOLMOD:
      # dont bother with this test if cholmod doesnt exist
      return
      
    n = 100
    A = sp.rand(n,n,density=0.2)
    A = A.T.dot(A).tocsc()
    b = np.random.random((n,))
    factor = rbf.linalg._SparsePosDefSolver(A)
    x1 = factor.solve(b)
    x2 = np.linalg.solve(A.A,b)
    self.assertTrue(np.allclose(x1,x2))
  
  def test_sparse_pos_def_solve_L(self):
    if not rbf.linalg.HAS_CHOLMOD:
      # dont bother with this test if cholmod doesnt exist
      return
      
    n = 100
    A = sp.rand(n,n,density=0.2)
    A = A.T.dot(A).tocsc()
    b = np.random.random((n,))
    factor = rbf.linalg._SparsePosDefSolver(A)
    x1 = factor.solve_L(b)
    x2 = np.linalg.solve(factor.L().A,b)
    self.assertTrue(np.allclose(x1,x2))

  def test_sparse_pos_def_L(self):
    if not rbf.linalg.HAS_CHOLMOD:
      # dont bother with this test if cholmod doesnt exist
      return
      
    n = 100
    A = sp.rand(n,n,density=0.2)
    A = A.T.dot(A).tocsc()
    factor = rbf.linalg._SparsePosDefSolver(A)
    L = factor.L()
    A2 = L.dot(L.T)
    self.assertTrue(np.allclose(A.A,A2.A))
  
  def test_sparse_pos_def_log_det(self):
    if not rbf.linalg.HAS_CHOLMOD:
      # dont bother with this test if cholmod doesnt exist
      return
      
    n = 100
    A = sp.rand(n,n,density=0.2)
    A = A.T.dot(A).tocsc()
    factor = rbf.linalg._SparsePosDefSolver(A)
    x1 = factor.log_det()
    x2 = np.log(np.linalg.det(A.A))
    self.assertTrue(np.isclose(x1,x2))
  
  def test_dense_pos_def_solve(self):
    n = 100
    A = np.random.random((n,n))
    A = A.T.dot(A)
    b = np.random.random((n,))
    factor = rbf.linalg._DensePosDefSolver(A)
    x1 = factor.solve(b)
    x2 = np.linalg.solve(A,b)
    self.assertTrue(np.allclose(x1,x2))
  
  def test_dense_pos_def_solve_L(self):
    n = 100
    A = np.random.random((n,n))
    A = A.T.dot(A)
    b = np.random.random((n,))
    factor = rbf.linalg._DensePosDefSolver(A)
    x1 = factor.solve_L(b)
    x2 = np.linalg.solve(factor.L(),b)
    self.assertTrue(np.allclose(x1,x2))
  
  def test_dense_pos_def_L(self):
    n = 100
    A = np.random.random((n,n))
    A = A.T.dot(A)
    factor = rbf.linalg._DensePosDefSolver(A)
    L = factor.L()
    A2 = L.dot(L.T)
    self.assertTrue(np.allclose(A,A2))

  def test_dense_pos_def_log_det(self):
    n = 100
    A = np.random.random((n,n))
    A = A.T.dot(A)
    factor = rbf.linalg._DensePosDefSolver(A)
    x1 = factor.log_det()
    x2 = np.log(np.linalg.det(A))
    self.assertTrue(np.isclose(x1,x2))

  def test_solver_dense_build_inv(self):
    A = np.random.random((4, 4))
    d = np.random.random((4,))
    solver1 = rbf.linalg.Solver(A, build_inverse=False)
    solver2 = rbf.linalg.Solver(A, build_inverse=True)
    soln1 = solver1.solve(d)
    soln2 = solver2.solve(d)
    self.assertTrue(np.allclose(soln1,soln2))

  def test_pos_def_solver_dense_build_inv(self):
    A = np.random.random((4, 4))
    A = A.T.dot(A)
    d = np.random.random((4,))
    solver1 = rbf.linalg.PosDefSolver(A, build_inverse=False)
    solver2 = rbf.linalg.PosDefSolver(A, build_inverse=True)
    soln1 = solver1.solve(d)
    soln2 = solver2.solve(d)
    self.assertTrue(np.allclose(soln1,soln2))
  
  def test_partitioned_solver_dense(self):    
    A = np.random.random((4,4))
    A = A.T + A # A is now symmetric
    B = np.random.random((4,2))
    a = np.random.random((4,))
    b = np.random.random((2,))
    Cfact = rbf.linalg.PartitionedSolver(A,B)
    soln1a,soln1b = Cfact.solve(a,b)
    soln1 = np.hstack((soln1a,soln1b))
    Cinv = np.linalg.inv(
             np.vstack(
               (np.hstack((A,B)),
                np.hstack((B.T,np.zeros((2,2)))))))
    soln2 = Cinv.dot(np.hstack((a,b)))
    self.assertTrue(np.allclose(soln1,soln2))

  def test_partitioned_solver_dense_build_inv(self):    
    A = np.random.random((4,4))
    A = A.T + A # A is now symmetric
    B = np.random.random((4,2))
    a = np.random.random((4,))
    b = np.random.random((2,))
    Cfact = rbf.linalg.PartitionedSolver(A,B, build_inverse=True)
    soln1a,soln1b = Cfact.solve(a,b)
    soln1 = np.hstack((soln1a,soln1b))
    Cinv = np.linalg.inv(
             np.vstack(
               (np.hstack((A,B)),
                np.hstack((B.T,np.zeros((2,2)))))))
    soln2 = Cinv.dot(np.hstack((a,b)))
    self.assertTrue(np.allclose(soln1,soln2))

  def test_partitioned_solver_dense_pos_def(self):    
    A = np.random.random((4,4))
    A = A.T.dot(A) # A is now P.D.
    B = np.random.random((4,2))
    a = np.random.random((4,))
    b = np.random.random((2,))
    Cfact = rbf.linalg.PartitionedPosDefSolver(A,B)
    soln1a,soln1b = Cfact.solve(a,b)
    soln1 = np.hstack((soln1a,soln1b))
    Cinv = np.linalg.inv(
             np.vstack(
               (np.hstack((A,B)),
                np.hstack((B.T,np.zeros((2,2)))))))
    soln2 = Cinv.dot(np.hstack((a,b)))
    self.assertTrue(np.allclose(soln1,soln2))

  def test_partitioned_solver_dense_pos_def_build_inv(self):    
    A = np.random.random((4,4))
    A = A.T.dot(A) # A is now P.D.
    B = np.random.random((4,2))
    a = np.random.random((4,))
    b = np.random.random((2,))
    Cfact = rbf.linalg.PartitionedPosDefSolver(A,B, build_inverse=True)
    soln1a,soln1b = Cfact.solve(a,b)
    soln1 = np.hstack((soln1a,soln1b))
    Cinv = np.linalg.inv(
             np.vstack(
               (np.hstack((A,B)),
                np.hstack((B.T,np.zeros((2,2)))))))
    soln2 = Cinv.dot(np.hstack((a,b)))
    self.assertTrue(np.allclose(soln1,soln2))

  def test_partitioned_solver_sparse(self):    
    A = np.random.random((4,4))
    A = A.T + A # A is now symmetric
    A = sp.csc_matrix(A) # A is now sparse
    B = np.random.random((4,2))
    a = np.random.random((4,))
    b = np.random.random((2,))
    Cfact = rbf.linalg.PartitionedSolver(A,B)
    soln1a,soln1b = Cfact.solve(a,b)
    soln1 = np.hstack((soln1a,soln1b))
    Cinv = np.linalg.inv(
             np.vstack(
               (np.hstack((A.A,B)),
                np.hstack((B.T,np.zeros((2,2)))))))
    soln2 = Cinv.dot(np.hstack((a,b)))
    self.assertTrue(np.allclose(soln1,soln2))

  def test_partitioned_solver_sparse_build_inv(self):    
    A = np.random.random((4,4))
    A = A.T + A # A is now symmetric
    A = sp.csc_matrix(A) # A is now sparse
    B = np.random.random((4,2))
    a = np.random.random((4,))
    b = np.random.random((2,))
    Cfact = rbf.linalg.PartitionedSolver(A,B, build_inverse=True)
    soln1a,soln1b = Cfact.solve(a,b)
    soln1 = np.hstack((soln1a,soln1b))
    Cinv = np.linalg.inv(
             np.vstack(
               (np.hstack((A.A,B)),
                np.hstack((B.T,np.zeros((2,2)))))))
    soln2 = Cinv.dot(np.hstack((a,b)))
    self.assertTrue(np.allclose(soln1,soln2))

  def test_partitioned_solver_sparse_pos_def(self):    
    A = np.random.random((4,4))
    A = A.T.dot(A) # A is now P.D.
    A = sp.csc_matrix(A) # A is now sparse
    B = np.random.random((4,2))
    a = np.random.random((4,))
    b = np.random.random((2,))
    Cfact = rbf.linalg.PartitionedPosDefSolver(A,B)
    soln1a,soln1b = Cfact.solve(a,b)
    soln1 = np.hstack((soln1a,soln1b))
    Cinv = np.linalg.inv(
             np.vstack(
               (np.hstack((A.A,B)),
                np.hstack((B.T,np.zeros((2,2)))))))
    soln2 = Cinv.dot(np.hstack((a,b)))
    self.assertTrue(np.allclose(soln1,soln2))

  def test_partitioned_solver_sparse_pos_def_build_inv(self):    
    A = np.random.random((4,4))
    A = A.T.dot(A) # A is now P.D.
    A = sp.csc_matrix(A) # A is now sparse
    B = np.random.random((4,2))
    a = np.random.random((4,))
    b = np.random.random((2,))
    Cfact = rbf.linalg.PartitionedPosDefSolver(A,B, build_inverse=True)
    soln1a,soln1b = Cfact.solve(a,b)
    soln1 = np.hstack((soln1a,soln1b))
    Cinv = np.linalg.inv(
             np.vstack(
               (np.hstack((A.A,B)),
                np.hstack((B.T,np.zeros((2,2)))))))
    soln2 = Cinv.dot(np.hstack((a,b)))
    self.assertTrue(np.allclose(soln1,soln2))
