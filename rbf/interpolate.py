''' 
This module provides a class for RBF interpolation, `RBFInterpolant`. 
This function has numerous features that are lacking in 
`scipy.interpolate.rbf`. They include:
  
* variable weights on the data (when creating a smoothed interpolant)
* more choices of basis functions (you can also easily make your own)
* analytical differentiation of the interpolant 
* added polynomial terms for improved accuracy
* prevent extrapolation by masking data that is outside of the 
  convex hull defined by the data points

RBF Interpolation
-----------------
The RBF interpolant :math:`\mathbf{f(x)}` is defined as
    
.. math::
  \mathbf{f(x)} = \mathbf{K(x,y)a} + \mathbf{P(x)b}
  
where :math:`\mathbf{K(x,y)}` consists of the RBFs with centers at 
:math:`\mathbf{y}` evaluated at the interpolation points 
:math:`\mathbf{x}`. :math:`\mathbf{P(x)}` is a polynomial matrix
where each column is a monomial basis function evaluated at the 
interpolation points. The monomial basis functions span the space of 
all polynomials with a user specified order. :math:`\mathbf{a}` and 
:math:`\mathbf{b}` are coefficients that need to be estimated. The 
coefficients are found by solving the linear system of equations
  
.. math::
  (\mathbf{K(y,y)} + \mathbf{C_d})\mathbf{a}  
  + \mathbf{P(y)b} = \mathbf{d}

.. math::
  \mathbf{P^T(y)a} = \mathbf{0} 

where :math:`\mathbf{C_d}` is the data covariance matrix, 
:math:`\mathbf{d}` are the observations at :math:`\mathbf{y}`. If the
data has no uncertainty, :math:`\mathbf{C_d}=\mathbf{0}`, the
observations are fit perfectly by the interpolant. Increasing the
uncertainty degrades the fit while improving the smoothness of the
interpolant. This formulation closely follows chapter 19.4 of [1] and
chapter 13.2.1 of [2].
    
References
----------
[1] Fasshauer, G., Meshfree Approximation Methods with Matlab. World 
Scientific Publishing Co, 2007.
    
[2] Schimek, M., Smoothing and Regression: Approaches, Computations, 
and Applications. John Wiley & Sons, 2000.
    
'''
import numpy as np
import scipy.sparse
import scipy.spatial
import rbf.basis
import rbf.poly 
from rbf.utils import assert_shape
from rbf.linalg import PartitionedSolver


def _in_hull(p, hull):
  ''' 
  Tests if points in `p` are in the convex hull made up by `hull`
  '''
  dim = p.shape[1]
  # if there are not enough points in `hull` to form a simplex then 
  # return False for each point in `p`.
  if hull.shape[0] <= dim:
    return np.zeros(p.shape[0], dtype=bool)
  
  if dim >= 2:
    hull = scipy.spatial.Delaunay(hull)
    return hull.find_simplex(p)>=0
  else:
    # one dimensional points
    min = np.min(hull)
    max = np.max(hull)
    return (p[:, 0] >= min) & (p[:, 0] <= max)


class RBFInterpolant(object):
  ''' 
  Regularized radial basis function interpolant  

  Parameters 
  ---------- 
  y : (N, D) array
    Observation points.

  d : (N,) array
    Observed values at `y`.

  sigma : float or (N,) array, optional
    One standard deviation uncertainty for the observations. This
    defaults to zeros (i.e., the observations are perfectly known).
    Increasing this values will decrease the size of the RBF
    coefficients and leave the polynomial terms undamped. 
        
  eps : float or (N,) array, optional
    Shape parameters for each RBF. This has no effect for odd
    order polyharmonic splines. Defaults to 1.0.

  basis : rbf.basis.RBF instance, optional
    Radial basis function to use.
 
  extrapolate : bool, optional
    Whether to allows points to be extrapolated outside of a convex 
    hull formed by `y`. If False, then np.nan is returned for outside 
    points.

  order : int, optional
    Order of added polynomial terms.

  Notes
  -----
  This function does not make any estimates of the uncertainties on 
  the interpolated values.  See `rbf.gauss` for interpolation with 
  uncertainties.
    
  With certain choices of basis functions and polynomial orders this 
  interpolant is equivalent to a thin-plate spline.  For example, if the 
  observation space is one-dimensional then a thin-plate spline can be 
  obtained with the arguments `basis` = `rbf.basis.phs3` and `order` = 
  1.  For two-dimensional observation space a thin-plate spline can be 
  obtained with the arguments `basis` = `rbf.basis.phs2` and `order` = 
  1. See [2] for additional details on thin-plate splines.

  References
  ----------
  [1] Fasshauer, G., Meshfree Approximation Methods with Matlab, World 
  Scientific Publishing Co, 2007.
    
  [2] Schimek, M., Smoothing and Regression: Approaches, Computations, 
  and Applications. John Wiley & Sons, 2000.
  '''
  def __init__(self, y, d,
               sigma=None,
               eps=1.0,
               basis=rbf.basis.phs3,
               order=1,
               extrapolate=True):
    y = np.asarray(y) 
    assert_shape(y, (None, None), 'y')
    
    d = np.asarray(d)
    assert_shape(d, (y.shape[0],), 'd')
    
    q,dim = y.shape

    if sigma is None:
      # if sigma is not specified then it is zeros
      sigma = np.zeros(q)

    elif np.isscalar(sigma):
      # if a float is specified then use it as the uncertainties for
      # all observations
      sigma = np.repeat(sigma,q)  

    else:
      sigma = np.asarray(sigma)
      assert_shape(sigma, (y.shape[0],), 'sigma')
      
    # form block consisting of the RBF and uncertainties on the
    # diagonal
    K = basis(y, y, eps=eps) 
    Cd = scipy.sparse.diags(sigma**2)
    # form the block consisting of the monomials
    powers = rbf.poly.powers(order, dim)
    P = rbf.poly.mvmonos(y, powers)
    # create zeros vector for the right-hand-side
    z = np.zeros((powers.shape[0],))
    # solve for the RBF and mononomial coefficients
    basis_coeff, poly_coeff = PartitionedSolver(K + Cd, P).solve(d, z) 

    self._y = y
    self._basis = basis
    self._order = order 
    self._eps = eps
    self._basis_coeff = basis_coeff
    self._poly_coeff = poly_coeff
    self._powers = powers 
    self.extrapolate = extrapolate

  def __call__(self, x, diff=None, chunk_size=1000):
    ''' 
    Evaluates the interpolant at `x`

    Parameters 
    ---------- 
    x : (N, D) array
      Target points.

    diff : (D,) int array, optional
      Derivative order for each spatial dimension.
        
    chunk_size : int, optional  
      Break `x` into chunks with this size and evaluate the
      interpolant for each chunk.  Smaller values result in decreased
      memory usage but also decreased speed.

    Returns
    -------
    out : (N,) array
      Values of the interpolant at `x`
      
    '''
    x = np.asarray(x,dtype=float) 
    assert_shape(x, (None, self._y.shape[1]), 'x')
    
    xlen = x.shape[0]
    # allocate output array
    out = np.zeros(xlen, dtype=float)
    count = 0
    while count < xlen:
      start, stop = count, count + chunk_size
      K = self._basis(
        x[start:stop], 
        self._y, 
        eps=self._eps, 
        diff=diff)
      P = rbf.poly.mvmonos(
        x[start:stop], 
        self._powers, 
        diff=diff)
      out[start:stop] = (K.dot(self._basis_coeff) + 
                         P.dot(self._poly_coeff))
      count += chunk_size

    # return zero for points outside of the convex hull if 
    # extrapolation is not allowed
    if not self.extrapolate:
      out[~_in_hull(x, self._y)] = np.nan

    return out


