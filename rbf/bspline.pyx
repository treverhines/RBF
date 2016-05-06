from __future__ import division
import numpy as np
cimport numpy as np
from cython cimport boundscheck,wraparound

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
                       double[:] knots,
                       int index,
                       int order,
                       int diff=0):
  '''returns a 1D B-spline evaluated at x
  '''
  cdef:
    int N = x.shape[0]
    int M = knots.shape[0]
    int i
    double tol = (knots[M-1] - knots[0])*1e-10
    double[:] out = np.empty(N)

  assert diff <= order,(
    'derivative order must be less than or equal to the spline order')
  
  assert M >= (index+order+2),(
    'there are not enough knots for the given spline order and index')
  
  assert is_sorted(knots),(
    'knots must be in ascending order')

  with nogil:
    # can be parallelized with prange
    for i in range(N):
      out[i] = bsp1d_k(x[i],knots,index,order,diff,tol)
  
  return np.asarray(out)


@wraparound(False)
@boundscheck(False)
cdef double bsp1d_k(double x,
                    double[:] k,
                    int n,
                    int p,
                    int diff,
                    double tol) nogil:
  '''returns a bspline evaluated at x

  Parameters
  ----------
    x: position where the B-spline is evaluated
    k: B-spline knots
    n: B-spline index. 0 returns the left-most B-spline evaluated at x
    p: B-spline order. 0 is a boxcar function
    diff: derivative order
    tol: tolerance used to determine whether two knots are identical.
    
  '''
  cdef:
    double out = 0.0

  if diff > 0:
    # check knot spacing to prevent zero division
    if (k[n+p] - k[n]) > tol:
      out = p/(k[n+p] - k[n])*bsp1d_k(x,k,n,p-1,diff-1,tol)

    if (k[n+p+1] - k[n+1]) > tol:
      out -= p/(k[n+p+1] - k[n+1])*bsp1d_k(x,k,n+1,p-1,diff-1,tol)

  elif p == 0:
    # If the order is zero and the right-most B-spline is being 
    # evaluated then return 1 on the closed interval between the
    # two knots. Otherwise return 1 on the left-closed interval
    # between the two knots
    if k[n+1] == k[k.shape[0]-1]:
      if ((x >= k[n]) & (x <= k[n+1])):
        out = 1.0

    else:
      if ((x >= k[n]) & (x < k[n+1])):
        out = 1.0

  # if the derivative is zero and order is not zero
  else:
    if (k[n+p] - k[n]) > tol:
      out = (x - k[n])/(k[n+p] - k[n])*bsp1d_k(x,k,n,p-1,0,tol)

    if (k[n+p+1] - k[n+1]) > tol:
      out -= (x - k[n+p+1])/(k[n+p+1] - k[n+1])*bsp1d_k(x,k,n+1,p-1,0,tol)

  return out

def bspnd(x,k,n,p,diff=None):
  '''returns an N-D B-spline which is the tensor product of 1-D B-splines   
  The arguments for this function should all be length N sequences and       
  each element will be passed to bspline_1d             
                                                    
  Parameters                                       
  ----------                                  
                                              
    x: points where the b spline will be evaluated          
                                   
    k: knots for each dimension
                                                       
    n: B-spline index                 
                                                                                   
    p: order of the B-spline (0 is a step function)                
                                                        
  '''
  x = np.transpose(x)

  d = len(n)
  if diff is None:
    diff = (0,)*d
  assert ((len(x) == len(k)) &
          (len(x) == len(n)) &
          (len(x) == len(p)) &
          (len(x) == len(diff)))

  val = [bsp1d(a,b,c,d,e) for a,b,c,d,e in zip(x,k,n,p,diff)]
  return np.prod(val,0)

  
  
  
