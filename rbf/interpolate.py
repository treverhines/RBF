''' 
This module provides a class for RBF interpolation, `RBFInterpolant`.
This function has numerous features that are lacking in
`scipy.interpolate.rbf`. They include:
  
* variable weights on the data (when creating a smoothed interpolant)
* more choices of basis functions (you can also easily make your own)
* analytical differentiation of the interpolant 
* added polynomial terms for improved accuracy

RBF Interpolation
-----------------
Consider a vector of :math:`n` observations :math:`\mathbf{d}` made at
locations :math:`\mathbf{y}`. These observations could potentially
have normally distributed noise described by the covariance matrix
:math:`\Sigma`. An RBF interpolant :math:`\mathbf{f(x)}` for these
observations is parameterized as
    
.. math::
  \mathbf{f(x)} = \mathbf{K(x,y) a} + \mathbf{P(x) b}
  
where :math:`\mathbf{K(x,y)}` consists of the RBFs with centers at 
:math:`\mathbf{y}` evaluated at the interpolation points
:math:`\mathbf{x}`. :math:`\mathbf{P(x)}` is a polynomial matrix
containing :math:`m` monomial basis function evaluated at the
interpolation points. The monomial basis functions span the space of
all polynomials with a user specified order. The coefficients
:math:`\mathbf{a}` and :math:`\mathbf{b}` are chosen to minimize the
objective function

.. math::
  \mathcal{L}(\mathbf{a, b}) = 
  \mathbf{a^T K(y,y) a} + 
  \mathbf{(f(y; a, b) - d)^T \Sigma^{-1} (f(y; a, b) - d)}.

The second term on the right side of the above expression is a norm
measuring the misfit between the interpolant and the observations. The
first term on the right side is a norm that essentially penalizes the
roughness of the interpolant (technically, it is the norm associated
with the reproducing kernel Hilbert space for the chosen radial basis
function). This problem formulation is inspired by the formulation
used in section 19.2 and 6.3 of [1].

To determine :math:`\mathbf{a}` and :math:`\mathbf{b}`, we need
:math:`n+m` linear constraints. We can differentiate
:math:`\mathcal{L}` with respect to :math:`\mathbf{a}` and set it
equal to zero to get :math:`n` constraints

.. math::
  (\mathbf{K(y,y)} + \mathbf{\Sigma}) \mathbf{a}  
  + \mathbf{P(y) b} = \mathbf{d}.

For the remaining :math:`m` constraints, we differentiate
:math:`\mathcal{L}` with respect to :math:`\mathbf{b}` and
substitute the above equation in for :math:`\mathbf{d}` to get

.. math::
  \mathbf{P(y)^T a} = \mathbf{0}.

For most purposes, the above two equations provide the linear
constrains that are needed to solve for :math:`\mathbf{a}` and
:math:`\mathbf{b}`. However, there are cases where the system cannot
be solved due to, for example, too few data points or a polynomial
order that is too high. There are also some radial basis functions
which are conditionally positive definite, which means that the norm
:math:`\mathbf{a^T K(y, y) a}` is only guaranteed to be positive when
the order of the added polynomial is sufficiently high. The user is
refered to [1] for details on when the interpolation problem is
well-posed. For the default settings in `RBFInterpolant`, the
interpolation problem is well-posed as long as :math:`n \geq D+1`,
where `D` is the number of spatial dimensions.

If the data has no noise, :math:`\mathbf{\Sigma}=\mathbf{0}`, the
observations are fit perfectly by the interpolant. Increasing the
uncertainty degrades the fit while improving the smoothness of the
interpolant.
    
References
----------
[1] Fasshauer, G., Meshfree Approximation Methods with Matlab. World 
Scientific Publishing Co, 2007.
    
'''
import numpy as np
import scipy.sparse
import scipy.spatial

from rbf.poly import powers, mvmonos
from rbf.basis import phs3, get_rbf
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

  phi : rbf.basis.RBF instance or str, optional
    Radial basis function to use. This can be an RBF instance (e.g,
    `rbf.basis.phs3`) or a string for a predefined RBF (e.g., 'phs3').
    See `rbf.basis` for all the available options.
 
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
  interpolant is equivalent to a thin-plate spline.  For example, if
  the observation space is one-dimensional then a thin-plate spline
  can be obtained with the arguments `phi` = `rbf.basis.phs3` and
  `order` = 1.  For two-dimensional observation space a thin-plate
  spline can be obtained with the arguments `phi` = `rbf.basis.phs2`
  and `order` = 1. 

  References
  ----------
  [1] Fasshauer, G., Meshfree Approximation Methods with Matlab, World 
  Scientific Publishing Co, 2007.
    
  '''
  def __init__(self, y, d,
               sigma=None,
               eps=1.0,
               phi=phs3,
               order=1,
               extrapolate=True):
    y = np.asarray(y) 
    assert_shape(y, (None, None), 'y')
    nobs, dim = y.shape
    
    d = np.asarray(d)
    assert_shape(d, (nobs,), 'd')
    
    if sigma is None:
      # if sigma is not specified then it is zeros
      sigma = np.zeros(nobs)

    elif np.isscalar(sigma):
      # if a float is specified then use it as the uncertainties for
      # all observations
      sigma = np.repeat(sigma, nobs)  

    else:
      sigma = np.asarray(sigma)
      assert_shape(sigma, (nobs,), 'sigma')
      
    phi = get_rbf(phi)
    
    # form block consisting of the RBF and uncertainties on the
    # diagonal
    K = phi(y, y, eps=eps) 
    Cd = scipy.sparse.diags(sigma**2)
    # form the block consisting of the monomials
    pwr = powers(order, dim)
    P = mvmonos(y, pwr)
    # create zeros vector for the right-hand-side
    z = np.zeros((pwr.shape[0],))
    # solve for the RBF and mononomial coefficients
    phi_coeff, poly_coeff = PartitionedSolver(K + Cd, P).solve(d, z) 

    self._y = y
    self._phi = phi
    self._order = order 
    self._eps = eps
    self._phi_coeff = phi_coeff
    self._poly_coeff = poly_coeff
    self._pwr = pwr
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
      K = self._phi(
        x[start:stop], 
        self._y, 
        eps=self._eps, 
        diff=diff)
      P = mvmonos(
        x[start:stop], 
        self._pwr, 
        diff=diff)
      out[start:stop] = (K.dot(self._phi_coeff) + 
                         P.dot(self._poly_coeff))
      count += chunk_size

    # return zero for points outside of the convex hull if 
    # extrapolation is not allowed
    if not self.extrapolate:
      out[~_in_hull(x, self._y)] = np.nan

    return out


