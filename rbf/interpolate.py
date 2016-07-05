#!/usr/bin/env python
import numpy as np
from numpy.linalg import pinv
import scipy.optimize
import scipy.spatial
import rbf.basis
import rbf.poly
import rbf.geometry

def _coefficient_matrix(x,eps=None,basis=rbf.basis.phs3,order=0):
  ''' 
  returns the matrix that maps the coefficients to the function values 
  at the observation points
  '''
  # number of observation points and spatial dimensions
  N,D = x.shape

  # powers for the additional polynomials
  powers = rbf.poly.monomial_powers(order,D)
  # number of polynomial terms
  P = powers.shape[0]

  # allocate array 
  A = np.zeros((N+P,N+P))
  A[:N,:N] = basis(x,x,eps=eps)
  Ap = rbf.poly.mvmonos(x,powers)
  A[N:,:N] = Ap.T
  A[:N,N:] = Ap
  return A  


def _interpolation_matrix(xitp,x,diff=None,eps=None,basis=rbf.basis.phs3,order=0):
  ''' 
  returns the matrix that maps the coefficients to the function values 
  at the interpolation points
  '''
  # number of interpolation points and spatial dimensions
  I,D = xitp.shape

  # number of observation points
  N = x.shape[0]

  # powers for the additional polynomials
  powers = rbf.poly.monomial_powers(order,D)
  # number of polynomial terms
  P = powers.shape[0]

  A = np.zeros((I,N+P))
  A[:,:N] = basis(xitp,x,eps=eps,diff=diff)
  A[:,N:] = rbf.poly.mvmonos(xitp,powers,diff=diff)
  return A


def _in_hull(p, hull):
  ''' 
  Tests if points in p are in the convex hull made up by hull
  '''
  dim = p.shape[1]
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
  
  This function has numerous features that are lacking in 
  scipy.interpolate.rbf. They include:
  
    * variable weights on the data (when creating a smoothed interpolant)

    * more choices of basis functions (you can also easily make your own)
    
    * analytical differentiation of the interpolant 
    
    * added polynomial terms for improved accuracy
    
    * prevent extrapolation by masking data that is outside of the 
      convex hull defined by the data points

  Notes
  -----
    Regularization is imposed using the ridge regression method 
    described in chapter 19.4 of [1].

  
  References
  ----------
    [1] Fasshauer, G., Meshfree Approximation Methods with Matlab, 
      World Scientific Publishing Co, 2007.
    
  '''
  def __init__(self,
               x,
               value, 
               weight=None,
               eps=None, 
               basis=rbf.basis.phs3,
               order=1,  
               extrapolate=True,
               fill=np.nan,
               penalty=0.0,**kwargs):
    ''' 
    Initiates the RBF interpolant

    Parameters 
    ---------- 
      x: (N,D) array
        observation points which make up the rbf centers

      value : (N,) array
        function values at the observation points

      weight : (N,) array, optional
        weights to put on each observation point
        
      eps : (N,) array, optional
        shape parameters for each RBF. this has no effect for odd
        order polyharmonic splines

      basis : rbf.basis.RBF instance, optional
        radial basis function to use
 
      extrapolate : bool, optional
        whether to allows points to be extrapolated outside of a 
        convex hull formed by x.

      fill : optional
        if extrapolate is False then points outside of the convex hull 
        will be assigned this value
 
      order : int, optional
        order of added polynomial terms
        
      penalty : float, optional
        the smoothing parameter. This decreases the size of the RBF 
        coefficients while leaving the polynomial terms undamped

    '''
    x = np.asarray(x) 
    value = np.asarray(value)

    if eps is None:
      eps = np.ones(x.shape[0])
    else:
      eps = np.asarray(eps)

    if weight is None:
      weight = np.ones(x.shape[0])
    else:
      weight = np.asarray(weight)
      
    # number of observation points
    N,D = x.shape

    # number of polynomial terms
    P = rbf.poly.monomial_count(order,D)

    # form matrix for the LHS
    A = _coefficient_matrix(x,eps=eps,basis=basis,order=order)

    # scale RHS and LHS by weight
    A[:N,:] *= weight[:,None]
    value = value*weight

    # form regularization matrix
    L = np.zeros((N,N+P))  
    L[range(N),range(N)] = penalty

    # extend values to have a consistent size as A
    value = np.concatenate((value,np.zeros(P)))

    # stack coefficient matrix and regularization matrix
    A_ext = np.vstack((A,L))
    value_ext = np.concatenate((value,np.zeros(N)))

    # solve for RBF coefficients
    coeff = np.linalg.lstsq(A_ext,value_ext)[0] 

    self.x = x
    self.coeff = coeff
    self.basis = basis
    self.order = order 
    self.eps = eps
    self.extrapolate = extrapolate
    self.fill = fill

  def __call__(self,xitp,diff=None,max_chunk=100000):
    ''' 
    Returns the interpolant evaluated at xitp

    Parameters 
    ---------- 
      xitp: (N,D) array
        points where the interpolant is to be evaluated

      diff: (D,) int array, optional
        derivative order for each spatial dimension
        
      max_chunk : int, optional  
        break xitp into chunks with this size and evaluate the 
        interpolant for each chunk.  Smaller values result in 
        decreased memory usage but also decreased speed

    '''
    n = 0
    xitp = np.asarray(xitp) 
    #xitp = self.norm(xitp)
    
    Nitp = xitp.shape[0]
    # allocate output array
    out = np.zeros(Nitp)
    while n < Nitp:
      # xitp indices for this chunk
      idx = range(n,min(n+max_chunk,Nitp))
      A = _interpolation_matrix(xitp[idx],self.x,
                                diff=diff,eps=self.eps,
                                basis=self.basis,order=self.order)
      out[idx] = np.einsum('ij,j...->i...',A,self.coeff)
      n += max_chunk

    # return zero for points outside of the convex hull if 
    # extrapolation is not allowed
    if not self.extrapolate:
      out[~_in_hull(xitp,self.x)] = self.fill

    return out


