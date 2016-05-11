#!/usr/bin/env python
import numpy as np
from scipy.linalg import lstsq
import scipy.spatial
import rbf.basis
import rbf.poly
import rbf.geometry
import modest

def in_hull(p, hull):
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
    return (p >= min) & (p <= max)


class RBFInterpolant(object):
  ''' 
  A callable RBF interpolant
  '''
  def __init__(self,
               x,
               value, 
               eps=None, 
               basis=rbf.basis.phs3,
               order=0,
               extrapolate=True,
               fill=np.nan,
               damping=0.0):
    ''' 
    Initiates the RBF interpolant

    Parameters 
    ---------- 
      x: (N,D) array of interpolation points which make up the rbf
        centers

      value: (N,...) function values at the interpolation points. This
        may have any shape as long as the first axis has length N.

      eps: (N,) array of shape parameters for each RBF. this has no 
        effect for the default basis function, which is scale 
        invariant. 

      basis: (default:phs3) type of basis function to use

      order: (default:0) order of added polynomial terms
 
      extrapolate: (default:True) whether to allows points to be
        extrapolated outside of a convex hull formed by x. 

      fill: (default:np.nan) if extrapolate is False then points
        outside of the convex hull will be assigned this value

      damping: (default:0.0) damping coefficient. This is used if the 
        interpolation problem is ill-posed or if the data is noisy

    '''
    x = np.asarray(x)

    # number of RBFs and dimensions                                            
    Ns,Ndim = x.shape

    # number of monomial terms                                          
    Np = rbf.poly.monomial_count(order,Ndim)

    # number of regularization lines
    Nr = Np + Ns

    value = np.asarray(value)

    
    # extend value to include a zero for each added polynomial term
    # and for each regularization constraint
    added_zeros = np.repeat(np.zeros((1,)+value.shape[1:]),Np+Nr,axis=0)
  
    value = np.concatenate((value,added_zeros))

    if eps is None:
      eps = np.ones(Ns)

    eps = np.asarray(eps)

    # make system matrix
    A = np.zeros((Ns+Np,Ns+Np))
    A[:Ns,:Ns] = basis(x,x,eps)
    powers = rbf.poly.monomial_powers(order,Ndim)
    diff = np.zeros(Ns,dtype=int)
    Ap = rbf.poly.mvmonos(x,powers,diff)
    A[Ns:,:Ns] = Ap.T
    A[:Ns,Ns:] = Ap

    # make regularization matrix
    L = np.zeros((Ns+Np,Ns+Np))
    L[:Ns,:Ns] = sum(basis(x,x,eps,diff=i.astype(int)) for i in modest.Perturb(np.zeros(Ndim),2))
    powers = rbf.poly.monomial_powers(order,Ndim)
    Lp = sum(rbf.poly.mvmonos(x,powers,diff=i.astype(int)) for i in modest.Perturb(np.zeros(Ndim),2))
    L[Ns:,:Ns] = Ap.T
    L[:Ns,Ns:] = Ap

    A = np.vstack((A,damping*L))
 
    #try:
    print(A.shape)
    print(value.shape)
    coeff = lstsq(A,value)[0]
    #except np.linalg.LinAlgError:
    #  print('Encountered singular matrix when finding the coefficients '
    #        'for the RBF interpolant. Attempting to solve with Tikhonov '
    #        'regularization')
    #  coeff = solve(A+1e-10*np.eye(Ns+Np),value)      

    self.x = x
    self.coeff = coeff
    self.basis = basis
    self.powers = powers
    self.eps = eps
    self.Ns = Ns
    self.Np = Np
    self.Ndim = Ndim
    self.fill = fill
    self.extrapolate = extrapolate


  def __call__(self,xitp,diff=None):
    '''Returns the interpolant evaluated at xitp

    Parameters 
    ---------- 
      xitp: ((N,) or (N,D) array) points where the interpolant is to 
        be evaluated

      diff: ((D,) tuple, default=(0,)*dim) a tuple whos length is
        equal to the number of spatial dimensions.  Each value in the
        tfuple must be an integer indicating the order of the
        derivative in that spatial dimension.  For example, if the the
        spatial dimensions of the problem are 3 then diff=(2,0,1)
        would compute the second derivative in the first dimension and
        the first derivative in the third dimension.

    '''
    max_chunk = 100000
    n = 0
    xitp = np.asarray(xitp,dtype=float)
    Nitp = xitp.shape[0]

    if diff is None:
      diff = (0,)*self.Ndim

    out = np.zeros((len(xitp),)+self.coeff.shape[1:])
    while n < len(xitp):
      xitp_i = xitp[n:n+max_chunk]
      A = np.zeros((len(xitp_i),self.Ns+self.Np))
      A[:,:self.Ns] = self.basis(xitp_i,
                                 self.x,
                                 eps=self.eps,
                                 diff=diff)
      A[:,self.Ns:] = rbf.poly.mvmonos(xitp_i,
                                       self.powers,
                                       np.array(diff,dtype=int))
      out[n:n+max_chunk] = np.einsum('ij,j...->i...',A,self.coeff)
      n += max_chunk

    # return zero for points outside of the convex hull if 
    # extrapolation is not allowed
    if not self.extrapolate:
      out[~in_hull(xitp,self.x)] = self.fill

    return out

