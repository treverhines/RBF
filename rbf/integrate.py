#!/usr/bin/env python
from __future__ import division
import numpy as np
from rbf.halton import Halton
from rbf.geometry import complex_contains
from rbf.geometry import is_valid
import logging 
logger = logging.getLogger(__name__)

def mcint(f,vert,smp,samples=None,lower_bounds=None,
          upper_bounds=None,check_valid=False,rng=None):
  '''
  Description
  -----------
    Monte Carlo integration algorithm over an arbitrary 1, 2, or 3  
    dimensional domain

  Parameters
  ----------
    f: Scalar value function being integrated.  This function should
      take an (N,D) array as input and return an (N,) array

    vert: vertices of integration domain boundary
   
    smp: simplices describing how the vertices are connected to form
      the domain boundary

    samples (default=50**D): number of samples to use 

    check_valid (default=False): Whether to check if the simplices 
      define a closed boundary 

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

  '''
  vert = np.asarray(vert,dtype=float)
  smp = np.asarray(smp,dtype=int)
  if check_valid:
    assert is_valid(smp), (
      'invalid simplices, see documentation for rbf.geometry.is_valid')

  dim = vert.shape[1]
  
  if lower_bounds is None:
    lb = np.min(vert,0)
  else:
    lb = np.asarray(lower_bounds)
    assert lb.shape[0] == dim

  if upper_bounds is None:
    ub = np.max(vert,0)
  else:
    ub = np.asarray(upper_bounds)
    assert ub.shape[0] == dim

  if rng is None:
    rng = Halton(dim)

  if samples is None:
    samples = 20**dim

  pnts = rng(samples)*(ub-lb) + lb
  val = f(pnts)
  is_inside = complex_contains(pnts,vert,smp)
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

  soln = np.sum(val)*np.prod(ub-lb)/samples
  err = np.prod(ub-lb)*np.std(val)/np.sqrt(samples)

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
           rng=None,check_valid=False):
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
 
    samples (default=50**D): number of samples to use in each  
      iteration

    check_valid (default=False): Whether to check if the simplices 
      define a closed boundary 

    rng (default=Halton(D)): random number generator. Must take an 
      integer input, N, and return an (N,D) array of random points 

    max_depth (default=50): maximum recursion depth allowed. If this
      limit is reached then a solution is still returned but it is not
      guaranteed to be more accurate than the specified tolerance 

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
  dim = vert.shape[1]

  if _depth == 0:
    logger.debug('integrating function with domain defined by %s '
                 'vertices and %s simplices' % (len(vert),len(smp)))

    if check_valid:
      assert is_valid(smp), (
        'invalid simplices, see documentation for rbf.geometry.is_valid')

  if lower_bounds is None:
    lb = np.min(vert,0)
  else:
    lb = np.asarray(lower_bounds)
    assert lb.shape[0] == dim

  if upper_bounds is None:
    ub = np.max(vert,0)
  else:
    ub = np.asarray(upper_bounds)
    assert ub.shape[0] == dim

  if rng is None:
    rng = Halton(dim)

  if tol is None:
    # if no tolerance is specified then an rough initial estimate for
    # the integral is made and then the tolerance is set to 1e-3 times
    # that estimate. If the initial estimate is less than 1e-3 then
    # the tolerance is set to 1e-6
    init_est = mcint(f,vert,smp,samples=samples,
                     lower_bounds=lower_bounds,
                     upper_bounds=upper_bounds,
                     check_valid=False,rng=rng)
    init_integral = init_est[0]
    if abs(init_integral) > 1e-1:
      tol = abs(init_integral*1e-3)
    else:
      tol = 1e-4

  # The tolerance decreases by a factor of 1/sqrt(2) for each
  # recursion depth. This ensures that combined uncertainties
  # is less than the specified tolerance. 
  tol = tol/np.sqrt(2)
  soln = 0.0
  var = 0.0
  minval = np.inf
  maxval = -np.inf
  for lbi,ubi in _divide_bbox(lb,ub,depth=_depth):
    out = mcint(f,vert,smp,samples=samples,
                lower_bounds=lbi,upper_bounds=ubi,
                check_valid=False,rng=rng)
    solni,erri,mini,maxi = out

    if mini < minval:
      minval = mini

    if maxi > maxval:
      maxval = maxi

    if _depth == max_depth:
      print('WARNING: reached soft recursion depth limit of %s' % max_depth) 
      logger.warning('reached soft recursion depth limit of %s' % max_depth)

    if (erri > tol) & (_depth < max_depth):
      out = rmcint(f,vert,smp,tol=tol,samples=samples,
                   lower_bounds=lbi,upper_bounds=ubi,
                   _depth=_depth+1,max_depth=max_depth,
                   rng=rng)
      new_solni,new_erri,mini,maxi = out

      if mini < minval:
        minval = mini

      if maxi > maxval:
        maxval = maxi                

      # combine the previous solution with the new one  
      # if the new solution has no error then do not include
      # the previous estimate
      if new_erri == 0.0:
        solni = new_solni 
        erri = new_erri
      else: 
        numer = solni/(erri**2) + new_solni/(new_erri**2)
        denom = 1.0/(erri**2) + 1.0/(new_erri**2)
        solni = numer/denom
        erri = 1.0/np.sqrt(denom)

    soln += solni
    var += erri**2

  err = np.sqrt(var)

  if _depth == 0:
    logger.debug('finished integration')

  return soln,err,minval,maxval


def _normalizer(fin,vert,smp,kind='integral',N=None):
  '''
  normalize a scalar values fucntion in 1,2 or 3 dimensional space.
  The function should takes an (N,D) array of points as its only
  argument and return an (N,) array.  The kind of normalization is
  specified with "kind", which can either be "density" to normalize
  the function so that it integrates to N, or "max" so that the
  maximum value is 1.
  '''
  out = rmcint(fin,vert,smp)
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

def density_normalizer(vert,smp,N):
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
    fout = _normalizer(fin,vert,smp,kind='density',N=N)
    return fout
  return dout


def max_normalizer(vert,smp):
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
    fout = _normalizer(fin,vert,smp,kind='max')
    return fout
  return dout
