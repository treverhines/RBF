#!/usr/bin/env python
import numpy as np
from scipy.linalg import solve
import scipy.spatial
import rbf.basis
import rbf.poly
import rbf.geometry

class RBFInterpolant(object):
  '''
  A callable RBF interpolant
  '''
  def __init__(self,
               x,
               value, 
               eps=None, 
               basis=rbf.basis.phs2,
               order=0,
               extrapolate=True,
               fill=np.nan):
    '''
    Initiates the RBF interpolant

    Parameters 
    ---------- 
      x: (N,D) array of interpolation points which make up the rbf
        centers

      value: (N,...) function values at the interpolation points. This
        may have any shape as long as the first axis has length N.

      eps: (N,) array shape parameters for each RBF

      rbf: (default:phs2) type of rbf to use

      order: (default:0) order of added polynomial terms
 
      extrapolate: (default:True) whether to allows points to be
        extrapolated outside of a convex hull formed by x

      fill: (default:np.nan) if extrapolate is False then points
        outside of the convex hull will be assigned this value

    '''
    x = np.asarray(x)

    # number of RBFs and dimensions                                            
    Ns,Ndim = x.shape

    # number of monomial terms                                          
    Np = rbf.poly.monomial_count(order,Ndim)

    value = np.asarray(value)

    # extend value to include a zero for each added polynomial term
    added_zeros = np.repeat(np.zeros((1,)+value.shape[1:]),Np,axis=0)

    value = np.concatenate((value,added_zeros))

    if eps is None:
      eps = np.ones(Ns)

    eps = np.asarray(eps)

    A = np.zeros((Ns+Np,Ns+Np))

    A[:Ns,:Ns] = basis(x,x,eps)

    powers = rbf.poly.monomial_powers(order,Ndim)
    diff = np.zeros(Ns,dtype=int)
    Ap = rbf.poly.mvmonos(x,powers,diff)
    A[Ns:,:Ns] = Ap.T
    A[:Ns,Ns:] = Ap

    coeff = solve(A,value)

    # if extrapolation is not allowed then form a bounding convex hull
    if not extrapolate:
      if Ndim == 1:
        self.simplices = np.array([np.argmin(x,axis=0),np.argmax(x,axis=0)])

      else:
        chull = scipy.spatial.ConvexHull(x)
        self.vertices = np.array(chull.points,dtype=float)
        self.simplices = np.array(chull.simplices,dtype=int)

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
    '''
    Returns the interpolant evaluated at xitp

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
    xitp = np.asarray(xitp,dtype=float)
    Nitp = xitp.shape[0]

    if diff is None:
      diff = (0,)*self.Ndim

    A = np.zeros((Nitp,self.Ns+self.Np))
    A[:,:self.Ns] = self.basis(xitp,
                               self.x,
                               eps=self.eps,
                               diff=diff)
    A[:,self.Ns:] = rbf.poly.mvmonos(xitp,
                                     self.powers,
                                     np.array(diff,dtype=int))

    out = np.einsum('ij,j...->i...',A,self.coeff)

    # return zero for points outside of the convex hull if 
    # extrapolation is not allowed
    if not self.extrapolate:
      is_inside = rbf.geometry.contains(xitp,
                                        self.vertices,
                                        self.simplices)
      out[~is_inside] = self.fill

    return out

