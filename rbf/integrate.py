#!/usr/bin/env python
from __future__ import division
import numpy as np
import rbf.halton as hlt
import rbf.geometry as gm
import logging 
logger = logging.getLogger(__name__)

def mcint(f,vert,smp,samples=None,lower_bounds=None,
          upper_bounds=None,rng=None):
  '''
  Description
  -----------
    Monte Carlo integration algorithm over an arbitrary 1, 2, or 3  
    dimensional domain. This algorithm treats the integration domain
    as a bounding box and converts all function values for points outside 
    of the simplicial complex to zero. 

  Parameters
  ----------
    f: Scalar value function being integrated.  This function should
      take an (N,D) array as input and return an (N,) array

    vert: vertices of integration domain boundary
   
    smp: simplices describing how the vertices are connected to form
      the domain boundary

    samples (default=20**D): number of samples to use 

    lower_bounds (default=None): If given, then the lower bounds for the 
      integration domain are truncated to these value. Used in rmcint

    upper_bounds (default=None): If given, then the upper bounds for the 
      integration domain are truncated to these value. Used in rmcint

    rng (default=Halton(D)): random number generator. Must take an 
      integer input, N, and return an (N,D) array of random points 

  Returns
  -------
    answer,error,maximum,minimum

    answer: integral over the domain

    error: uncertainty of the solution. Note that this tends to be
      overestimated when using a quasi-random number generator such as
      a Halton sequence

    maximum: maximum function value within the domain

    minimum: minimum function value within the domain

  Note 
  ---- 
    When integrating constant or nearly constant functions over a
    complicated domain, it is far more efficient to use mcint2

  '''
  vert = np.asarray(vert,dtype=float)
  smp = np.asarray(smp,dtype=int)

  dim = smp.shape[1]

  if lower_bounds is None:
    lower_bounds = np.min(vert,0)
  else:
    lower_bounds = np.asarray(lower_bounds)

  if upper_bounds is None:
    upper_bounds = np.max(vert,0)
  else:
    bounded = True
    upper_bounds = np.asarray(upper_bounds)

  if rng is None:
    rng = hlt.Halton(dim)

  if samples is None:
    samples = 20**dim

  if np.any(lower_bounds > upper_bounds):
    raise ValueError(
      'lowers bounds found to be larger than upper bounds')

  if samples < 2:
    raise ValueError(
      'sample size must be at least 2')

  pnts = rng(samples)*(upper_bounds-lower_bounds) + lower_bounds
  val = f(pnts)
  is_inside = gm.contains(pnts,vert,smp)
  # If there are any points within the domain then return
  # the max and min value found within the domain
  if np.any(is_inside):
    minval = np.min(val[is_inside])
    maxval = np.max(val[is_inside])
  else:
    minval = np.inf
    maxval = -np.inf

  # copy val because its contents are going to be changed
  val = np.copy(val)
  val[~is_inside] = 0.0
  volume = np.prod(upper_bounds-lower_bounds)

  soln = np.sum(val)*volume/len(val)
  err = volume*np.std(val,ddof=1)/np.sqrt(len(val))

  return soln,err,minval,maxval

def mcint2(f,vert,smp,samples=None,
           check_simplices=True,rng=None):
  '''Description
  -----------
    Monte Carlo integration algorithm over an arbitrary 1, 2, or 3
    dimensional domain. This algorithm uses the simplicial complex
    itself as the integration domain. Doing so requires the ability to
    compute the domain area/volume exactly, which can cause
    significant overhead for very large simplicial complexes if the
    simplices are not properly oriented.

  Parameters
  ----------
    f: Scalar value function being integrated.  This function should
      take an (N,D) array as input and return an (N,) array

    vert: vertices of integration domain boundary
   
    smp: simplices describing how the vertices are connected to form
      the domain boundary

    samples (default=20**D): number of samples to use 

    check_simplices (default=False): Whether to check that the
      simplices define a closed surface and oriented such that their
      normals point outward

    rng (default=Halton(D)): random number generator. Must take an 
      integer input, N, and return an (N,D) array of random points 

  Returns
  -------
    answer,error,maximum,minimum

    answer: integral over the domain

    error: uncertainty of the solution. Note that this tends to be
      overestimated when using a quasi-random number generator such as
      a Halton sequence

    maximum: maximum function value within the domain

    minimum: minimum function value within the domain


  Note 
  ---- 
    Volume calculations require simplices to be oriented such that
    their normal vectors, by the right-hand rule, point outside the
    domain. If check_simplices is True, then the simplices are check
    and reordered to ensure such an orientation. Checking the
    simplices is an O(N^2) process and should be set to False if the 
    simplices are known to be properly oriented.

  '''
  vert = np.asarray(vert,dtype=float)
  smp = np.asarray(smp,dtype=int)
  if check_simplices:
    smp = gm.oriented_simplices(vert,smp)

  dim = smp.shape[1]

  lower_bounds = np.min(vert,0)
  upper_bounds = np.max(vert,0)

  if rng is None:
    rng = hlt.Halton(dim)

  if samples is None:
    samples = 20**dim

  pnts = rng(samples)*(upper_bounds-lower_bounds) + lower_bounds
  val = f(pnts)
  is_inside = gm.contains(pnts,vert,smp)
  # If there are any points within the domain then return
  # the max and min value found within the domain
  if np.any(is_inside):
    minval = np.min(val[is_inside])
    maxval = np.max(val[is_inside])
  else:
    minval = np.inf
    maxval = -np.inf

  val = val[is_inside]
  volume = gm.complex_volume(vert,smp,orient=False)
  if (volume < 0.0):
    raise ValueError(
      'Simplicial complex found to have a negative volume. Check the '
      'orientation of simplices and ensure closedness')

  if (volume > 0.0) & (len(val) < 2):
    raise ValueError(
      'Number of values used to estimate the integral is less than 2.'
      'Ensure the simplicial complex is closed and then increase the '
      'sample size')

  if volume == 0.0:
     soln = 0.0
     err = 0.0 
   
  else:
    soln = np.sum(val)*volume/len(val)
    err = volume*np.std(val,ddof=1)/np.sqrt(len(val))

  return soln,err,minval,maxval


def _divide_bbox(lb,ub,depth=0):
  '''
  divides the bounding box in half along an axis determined by the 
  recursion depth
  '''
  lb = np.asarray(lb,dtype=float)
  ub = np.asarray(ub,dtype=float)
  mp = (lb + ub)/2.0
  dim = lb.shape[0]
  out = [(np.copy(lb),np.copy(ub)),
         (np.copy(lb),np.copy(ub))]

  # change upper bound for first box to the midpoint
  out[0][1][depth%dim] = mp[depth%dim]
  # change lower bound for second box to the midpoint
  out[1][0][depth%dim] = mp[depth%dim]
  return out


def rmcint(f,vert,smp,tol=None,max_depth=50,samples=None,
           lower_bounds=None,upper_bounds=None,_depth=0,
           rng=None):
  '''
  Description
  -----------
    Recursive Monte Carlo integration algorithm over an arbitrary 1,
    2, or 3 dimensional domain.  For each iteration, the integration
    domain is divided in half and then integrated with mcint. If the
    estimated integration error for either region exceeds
    tol/(2**(R/2)), where R is the recursion depth, then the region is
    further subdivided and integrated. The axis along which the domain
    is divided depends only on the recursion depth.  

  Parameters
  ----------
    f: Scalar value function being integrated.  This function should
      take an (N,D) array as input and return an (N,) array

    vert: vertices of the integration domain boundary
   
    smp: simplices describing how the vertices are connected to form
      the domain boundary

    tol(default=None): Recursion proceeds until this error tolerance
      for the final solution is guaranteed to be satisfied. If no
      tolerance is specified then a crude estimate of the integral is 
      made and then tol is set to 0.001 times that estimate.
 
    samples (default=20**D): number of samples to use in each  
      iteration

    check_simplices (default=True): Whether to ensure that the
      simplices have outwardly oriented normal vectors

    rng (default=Halton(D)): random number generator. Must take an 
      integer input, N, and return an (N,D) array of random points 

    max_depth (default=50): maximum recursion depth allowed. If this
      limit is reached then a solution is still returned but it is not
      guaranteed to be more accurate than the specified tolerance 

    lower_bounds (default=None): the lower bounds for the integration
      domain. This truncates the integration domain specified by the
      vertices and simplices. If given then the integration domain is
      considered to be a bounding box and function values outside the
      simplicial complex are set to zero. When not specified, the
      integration domain for the first iteration is the simplicial
      complex and any sampled points which are outside of the domain
      are thrown out.

    upper_bounds (default=None): the upper bounds for the integration 
      domain

  Returns
  -------
    answer,error,maximum,minimum

    answer: integral over the domain

    error: uncertainty of the solution. Note that this tends to be
      overestimated when using a quasi-random number generator such as
      a Halton sequence. Note also that this error is often 
      significantly less than the specified tolerance.

    maximum: maximum function value within the domain

    minimum: minimum function value within the domain

  '''
  vert = np.asarray(vert,dtype=float)
  smp = np.asarray(smp,dtype=int)
  dim = smp.shape[1]

  if rng is None:
    rng = hlt.Halton(dim)

  if lower_bounds is None:
    lower_bounds = np.min(vert,0)
  else:
    lower_bounds = np.asarray(lower_bounds)

  if upper_bounds is None:
    upper_bounds = np.max(vert,0)
  else:
    upper_bounds = np.asarray(upper_bounds)

  # Make an estimate of the integral
  out = mcint(f,vert,smp,samples=samples,
              lower_bounds=lower_bounds,
              upper_bounds=upper_bounds,
              rng=rng)

  soln = out[0]
  err = out[1]
  minval = out[2]
  maxval = out[3]

  if tol is None:
    # if no tolerance is specified then use the 1e-2 time the estimate
    # of the integral. If the estimate is less than 1e-2 then the
    # tolerance is set to 1e-4.
    if abs(soln) > 1e-2:
      tol = abs(soln)*1e-2
    else:
      tol = 1e-4
    
  # if the error from the estimate is below the tolerance or if the maximum
  # recursion depth has been reached then return the solution
  if (err < tol) | (_depth == max_depth):
    return out

  # if the error is above the tolerance then divide the domain
  new_soln = 0
  new_err = 0
  for lbi,ubi in _divide_bbox(lower_bounds,upper_bounds,depth=_depth):
    outi = rmcint(f,vert,smp,
                  tol=tol/np.sqrt(2),
                  samples=samples,
                  lower_bounds=lbi,
                  upper_bounds=ubi,
                  _depth=_depth+1,
                  max_depth=max_depth,
                  rng=rng)
    solni,erri,mini,maxi = outi
    if mini < minval:
      minval = mini

    if maxi > maxval:
      maxval = maxi                

    new_soln += solni
    new_err = np.sqrt(new_err**2 + erri**2)
      
  # combine the previous solution with the new one  
  # if the new solution has no error then do not include
  # the previous estimate
  if new_err == 0.0:
    soln = new_soln
    err = new_err
  else: 
    numer = soln/(err**2) + new_soln/(new_err**2)
    denom = 1.0/(err**2) + 1.0/(new_err**2)
    soln = numer/denom
    err = 1.0/np.sqrt(denom)

  return soln,err,minval,maxval


def _normalizer(fin,vert,smp,kind='integral',N=None,tol=None):
  '''
  normalize a scalar values fucntion in 1,2 or 3 dimensional space.
  The function should takes an (N,D) array of points as its only
  argument and return an (N,) array.  The kind of normalization is
  specified with "kind", which can either be "density" to normalize
  the function so that it integrates to N, or "max" so that the
  maximum value is 1.
  '''
  out = rmcint(fin,vert,smp,tol=tol)
  integral,err,minval,maxval = out 
  if kind == 'max':
    denom = maxval

  if kind == 'density':
    denom = integral/N

  if denom == 0.0:
    raise ValueError(
      'normalized function by 0, this may be due to to an '
      'insufficiently large MC integration sample size')

  def fout(p):
    return fin(p)/denom

  return fout

def density_normalizer(vert,smp,N,tol=None):
  '''
  Description 
  ----------- 
    returns a decorator which normalizes a function such that it
    integrates to the specified value

  Parameters
  ----------
    vert: vertices of the integration domain boundary
   
    smp: simplices describing how the vertices are connected to form
      the domain boundary
    
    N: Value which the normalized function integrates to

  Returns 
  -------
    decorator

  '''
  def dout(fin):
    fout = _normalizer(fin,vert,smp,kind='density',N=N,tol=tol)
    return fout
  return dout


def max_normalizer(vert,smp,tol=None):
  '''
  Description 
  ----------- 
    returns a decorator which normalizes a function such that its
    maximum value within the integration domain is 1.

  Parameters
  ----------
    vert: vertices of the integration domain boundary
   
    smp: simplices describing how the vertices are connected to form
      the domain boundary

  Returns 
  -------
    decorator

  '''
  def dout(fin):
    fout = _normalizer(fin,vert,smp,kind='max',tol=tol)
    return fout
  return dout
