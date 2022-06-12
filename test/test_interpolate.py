import numpy as np
import scipy.linalg
import rbf.interpolate
from rbf.interpolate import RBFInterpolant
import rbf.pde.halton
import rbf.basis
from rbf.poly import mvmonos
import unittest


def gml_slow(d, K, P):
    '''
    A literal implementation of the GML expression from section 4.8 of [1].

    Parameters
    ----------
    d : (N, S) array

    K : (N, N) array

    P : (N, M) array

    Returns
    -------
    float

    References
    ----------
    [1] Wahba, G., 1990. Spline Models for Observational Data. SIAM.

    '''
    n, s = d.shape
    m = P.shape[1]
    Q, _ = np.linalg.qr(P, mode='complete')
    Q2 = Q[:, m:]
    K_proj = Q2.T.dot(K).dot(Q2)
    d_proj = Q2.T.dot(d)
    # flatten vector-valued input
    d_proj = np.hstack(d_proj.T)
    K_proj = scipy.linalg.block_diag(*((K_proj,)*s))
    num = d_proj.dot(np.linalg.inv(K_proj).dot(d_proj))
    den = np.linalg.det(np.linalg.inv(K_proj))**(1/(s*(n - m)))
    out = num / den
    return out


def loocv_slow(d, S, K, P):
    '''
    Naive implementation of leave-one-out cross validation
    '''
    n, s = d.shape
    m = P.shape[1]
    Z = np.zeros((m, m))
    z = np.zeros((m, s))
    out = 0.0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        d_hat = d[mask]
        K_hat = K[np.ix_(mask, mask)]
        S_hat = S[np.ix_(mask, mask)]
        P_hat = P[mask]
        lhs = np.block([[K_hat + S_hat, P_hat], [P_hat.T, Z]])
        rhs = np.vstack((d_hat, z))
        soln = np.linalg.solve(lhs, rhs)
        pred = np.hstack((K[i, mask], P[i])).dot(soln)
        out += np.sum((d[i] - pred)**2)

    return np.sqrt(out)


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

        I = RBFInterpolant(obs,val,phi=rbf.basis.phs3,order=3)
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

        I = RBFInterpolant(obs,val,phi=rbf.basis.phs3,order=3)
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

        I = RBFInterpolant(obs,val,phi=rbf.basis.phs3,order=3)
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

        I = RBFInterpolant(obs,val,phi=rbf.basis.phs3,order=3)
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
        I = RBFInterpolant(obs,val,sigma=10000.0, phi=rbf.basis.phs1,order=0)
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
        I = RBFInterpolant(obs,val,sigma=10000.0, phi=rbf.basis.phs3,order=1)
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

        I = RBFInterpolant(obs,val,sigma=3.0, phi=rbf.basis.phs3,order=1)
        valitp_est = I(itp)
        valitp_true = test_func2d(itp)
        self.assertTrue(np.allclose(valitp_est,valitp_true,atol=1e-1))

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
        sigma[0] = 1e3
        I = RBFInterpolant(obs,val,sigma=sigma, phi=rbf.basis.phs3,order=1)
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
        I = RBFInterpolant(obs,val, phi=rbf.basis.spwen31,order=1, eps=0.5)
        valitp_est = I(itp)
        valitp_true = test_func2d(itp)
        self.assertTrue(np.allclose(valitp_est,valitp_true,atol=1e-2))

    def test_gml(self):
        n = 20
        x = np.random.random((n, 2))
        d = np.column_stack([np.sin(x[:, 0]), np.cos(x[:, 1]), x[:, 0]**2])
        d += np.random.normal(0.0, 0.1, d.shape)
        K = rbf.basis.ga(x, x, eps=0.1)
        P = mvmonos(x, 1)
        S = np.diag(np.full(n, 0.1))

        soln1 = gml_slow(d, K + S, P)
        soln2 = rbf.interpolate._gml(d, K + S, P)
        self.assertTrue(np.isclose(soln1, soln2))

    def test_loocv(self):
        n = 20
        x = np.random.random((n, 2))
        d = np.column_stack([np.sin(x[:, 0]), np.cos(x[:, 1]), x[:, 0]**2])
        d += np.random.normal(0.0, 0.1, d.shape)
        K = rbf.basis.ga(x, x, eps=0.1)
        P = mvmonos(x, 1)
        S = np.diag(np.full(n, 0.1))

        soln1 = loocv_slow(d, S, K, P)
        soln2 = rbf.interpolate._loocv(d, K + S, P)
        self.assertTrue(np.isclose(soln1, soln2))


    def test_loocv_full(self):
        # test LOOCV with the end-user functions
        n = 20
        x = np.random.random((n, 2))
        d = np.column_stack([np.sin(x[:, 0]), np.cos(x[:, 1]), x[:, 0]**2])
        d += np.random.normal(0.0, 0.1, d.shape)

        value1 = RBFInterpolant.loocv(x, d, phi='ga', sigma=0.1, eps=2.0, order=2)
        errors = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            interp = RBFInterpolant(x[mask], d[mask], phi='ga', sigma=0.1, eps=2.0, order=2)
            errors.append(interp(x[[i]])[0] - d[i])

        value2 = np.linalg.norm(errors)
        self.assertTrue(np.isclose(value1, value2))

#unittest.main()


