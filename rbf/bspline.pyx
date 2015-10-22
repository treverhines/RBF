# distutils: extra_compile_args = -fopenmp  
# distutils: extra_link_args = -fopenmp

from __future__ import division
import numpy as np
cimport numpy as np
from cython cimport boundscheck,wraparound
from cython.parallel cimport prange

cpdef bint is_sorted(double[:] x):
  cdef:
    int N = x.shape[0] - 1
    int i    
    
  for i in range(N):
    if x[i+1] < x[i]:
      return False

  return True

def augmented_knots(knots,p,side='both'):
  assert len(knots) >= 2,(
    'at least two knots must be given')

  if (side == 'left') | (side == 'both'):
    left = np.repeat(knots[0],p)
    knots = np.concatenate((left,knots))

  if (side == 'right') | (side == 'both'):
    right = np.repeat(knots[-1],p)
    knots = np.concatenate((knots,right))

  return knots


def natural_knots(nmax,p,side='both'):
  if side == 'both':
    k = np.linspace(0,1,nmax - p + 1)
    return augmented_knots(k,p,side)

  if (side == 'left') | (side == 'right'):
    k = np.linspace(0,1,nmax + 1)
    return augmented_knots(k,p,side)

  if (side == 'none'):
    k = np.linspace(0,1,nmax + p + 1)
    return k


def basis_number(k,p):
  return len(k) - p - 1

@wraparound(False)
@boundscheck(False)
cpdef np.ndarray bsp1d(double[:] x,
                       double[:] k,
                       unsigned int n,
                       unsigned int p,
                       unsigned int diff=0):
  cdef:
    unsigned int N = x.shape[0]
    unsigned int i
    double tol
    double[:] out = np.empty(N,dtype=np.float64,order='C')

  assert diff <= p,(
    'derivative order must be less than or equal to the spline order')
  
  assert k.shape[0] >= (n+p+2),(
    'there are not enough knots for the given spline order and index')
  
  assert is_sorted(k),(
    'knots must be in ascending order')

  tol = (k[k.shape[0]-1] - k[0])*1e-6
  a = tol
  with nogil:
    #for i in prange(N,schedule='static',chunksize=CHUNKSIZE):
    for i in prange(N):
      out[i] = bsp1d_k(x[i],k,n,p,diff,tol)
  
  return np.asarray(out)


@wraparound(False)
@boundscheck(False)
cdef double bsp1d_k(double x,
                    double[:] k,
                    unsigned int n,
                    unsigned int p,
                    unsigned int diff,
                    double tol) nogil:
  cdef:
    double out = 0.0

  if diff > 0:
    if (k[n+p] - k[n]) > tol:
      out = p/(k[n+p] - k[n])*bsp1d_k(x,k,n,p-1,diff-1,tol)

    if (k[n+p+1] - k[n+1]) > tol:
      out -= p/(k[n+p+1] - k[n+1])*bsp1d_k(x,k,n+1,p-1,diff-1,tol)

  elif p == 0:
    if k[n+1] == k[k.shape[0]-1]:
      if ((x >= k[n]) & (x <= k[n+1])):
        out = 1.0

    else:
      if ((x >= k[n]) & (x < k[n+1])):
        out = 1.0

  else:
    if (k[n+p] - k[n]) > tol:
      out = (x - k[n])/(k[n+p] - k[n])*bsp1d_k(x,k,n,p-1,0,tol)

    if (k[n+p+1] - k[n+1]) > tol:
      out -= (x - k[n+p+1])/(k[n+p+1] - k[n+1])*bsp1d_k(x,k,n+1,p-1,0,tol)

  return out


  
  
  
