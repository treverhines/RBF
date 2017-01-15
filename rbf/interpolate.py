''' 
This module provides a class for RBF interpolation, *RBFInterpolant*. 
This function has numerous features that are lacking in 
*scipy.interpolate.rbf*. They include:
  
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
  (\mathbf{K(y,y)} + p\mathbf{C_d})\mathbf{a}  
  + \mathbf{P(y)b} = \mathbf{d}

.. math::
  \mathbf{P^T(y)a} = \mathbf{0} 

where :math:`\mathbf{C_d}` is the data covariance matrix, 
:math:`\mathbf{d}` are the observations at :math:`\mathbf{y}`, 
and :math:`p` is a penalty parameter. With :math:`p=0` the 
observations are fit perfectly by the interpolant.  Increasing
:math:`p` degrades the fit while improving the smoothness of the 
interpolant. This formulation closely follows chapter 19.4 of [1] 
and chapter 13.2.1 of [2].
    
References
----------
[1] Fasshauer, G., Meshfree Approximation Methods with Matlab. World 
Scientific Publishing Co, 2007.
    
[2] Schimek, M., Smoothing and Regression: Approaches, Computations, 
and Applications. John Wiley & Sons, 2000.
    
'''
import numpy as np
import scipy.optimize
import scipy.spatial
import rbf.basis
import rbf.poly
import rbf.geometry

def _coefficient_matrix(x,eps,sigma,basis,order):
  ''' 
  returns the matrix used to compute the radial basis function 
  coefficients
  '''
  # number of observation points and spatial dimensions
  N,D = x.shape

  # powers for the additional polynomials
  powers = rbf.poly.powers(order,D)
  # number of polynomial terms
  P = powers.shape[0]
  # data covariance matrix
  Cd = np.diag(sigma**2)
  # allocate array 
  A = np.zeros((N+P,N+P))
  A[:N,:N] = basis(x,x,eps=eps) + Cd
  Ap = rbf.poly.mvmonos(x,powers)
  A[N:,:N] = Ap.T
  A[:N,N:] = Ap
  return A  


def _interpolation_matrix(xitp,x,diff,eps,basis,order):
  ''' 
  returns the matrix that maps the coefficients to the function values 
  at the interpolation points
  '''
  # number of interpolation points and spatial dimensions
  I,D = xitp.shape
  # number of observation points
  N = x.shape[0]
  # powers for the additional polynomials
  powers = rbf.poly.powers(order,D)
  # number of polynomial terms
  P = powers.shape[0]
  # allocate array 
  A = np.zeros((I,N+P))
  A[:,:N] = basis(xitp,x,eps=eps,diff=diff)
  A[:,N:] = rbf.poly.mvmonos(xitp,powers,diff=diff)
  return A


def _in_hull(p, hull):
  ''' 
  Tests if points in *p* are in the convex hull made up by *hull*
  '''
  dim = p.shape[1]
  # if there are not enough points in *hull* to form a simplex then 
  # return False for each point in *p*.
  if hull.shape[0] <= dim:
    return np.zeros(p.shape[0],dtype=bool)
  
  if dim >= 2:
    hull = scipy.spatial.Delaunay(hull)
    return hull.find_simplex(p)>=0
  else:
    # one dimensional points
    min = np.min(hull)
    max = np.max(hull)
    return (p[:,0] >= min) & (p[:,0] <= max)


class RBFInterpolant(object):
  ''' 
  Regularized radial basis function interpolant  

  Parameters 
  ---------- 
  y : (N,D) array
    Observation points.

  d : (N,) array
    Observed values at *y*.

  sigma : (N,) array, optional
    One standard deviation uncertainty on the observations. This 
    defaults to ones.
        
  eps : (N,) array, optional
    Shape parameters for each RBF. this has no effect for odd
    order polyharmonic splines.

  basis : rbf.basis.RBF instance, optional
    Radial basis function to use.
 
  extrapolate : bool, optional
    Whether to allows points to be extrapolated outside of a convex 
    hull formed by *y*. If False, then np.nan is returned for outside 
    points.

  order : int, optional
    Order of added polynomial terms.
        
  penalty : float, optional
    The smoothing parameter. This parameter merely scales *sigma*. 
    Increasing this values will decrease the size of the RBF 
    coefficients and leave the polynomial terms undamped. Thus the 
    endmember for a large penalty parameter will be equivalent to 
    polynomial regression. 

  Notes
  -----
  This function involves solving a dense system of equations, which 
  will be prohibitive for large data sets. See *rbf.filter* for 
  smoothing large data sets. 
  
  This function does not make any estimates of the uncertainties on 
  the interpolated values.  See *rbf.gpr* for interpolation with 
  uncertainties.
    
  With certain choices of basis functions and polynomial orders this 
  interpolant is equivalent to a thin-plate spline.  For example, if the 
  observation space is one-dimensional then a thin-plate spline can be 
  obtained with the arguments *basis* = *rbf.basis.phs3* and *order* = 
  1.  For two-dimensional observation space a thin-plate spline can be 
  obtained with the arguments *basis* = *rbf.basis.phs2* and *order* = 
  1. See [2] for additional details on thin-plate splines.

  References
  ----------
  [1] Fasshauer, G., Meshfree Approximation Methods with Matlab, World 
  Scientific Publishing Co, 2007.
    
  [2] Schimek, M., Smoothing and Regression: Approaches, Computations, 
  and Applications. John Wiley & Sons, 2000.
  '''
  def __init__(self,y,d,sigma=None,eps=None,basis=rbf.basis.phs3,
               order=1,extrapolate=True,penalty=0.0):
    y = np.asarray(y) 
    d = np.asarray(d)
    q,dim = y.shape
    p = rbf.poly.count(order,dim)

    if eps is None:
      eps = np.ones(q)
    else:
      eps = np.asarray(eps)

    if sigma is None:
      sigma = np.ones(q)
    else:
      sigma = np.asarray(sigma)
      
    # form matrix for the LHS
    A = _coefficient_matrix(y,eps,penalty*sigma,basis,order)
    # add zeros to the RHS for the polynomial constraints
    d = np.concatenate((d,np.zeros(p)))
    # find the radial basis function coefficients
    coeff = np.linalg.solve(A,d)

    self.y = y
    self.coeff = coeff
    self.basis = basis
    self.order = order 
    self.eps = eps
    self.extrapolate = extrapolate

  def __call__(self,x,diff=None,max_chunk=100000):
    ''' 
    Evaluates the interpolant at *x*

    Parameters 
    ---------- 
    x : (N,D) array
      Target points.

    diff : (D,) int array, optional
      Derivative order for each spatial dimension.
        
    max_chunk : int, optional  
      Break *x* into chunks with this size and evaluate the 
      interpolant for each chunk.  Smaller values result in decreased 
      memory usage but also decreased speed.

    Returns
    -------
    out : (N,) array
      Values of the interpolant at *x*
      
    '''
    n = 0
    x = np.asarray(x) 
    q = x.shape[0]
    # allocate output array
    out = np.zeros(q)
    while n < q:
      # xitp indices for this chunk
      idx = range(n,min(n+max_chunk,q))
      A = _interpolation_matrix(x[idx],self.y,
                                diff,self.eps,
                                self.basis,self.order)
      out[idx] = A.dot(self.coeff) 
      n += max_chunk

    # return zero for points outside of the convex hull if 
    # extrapolation is not allowed
    if not self.extrapolate:
      out[~_in_hull(x,self.y)] = np.nan

    return out


