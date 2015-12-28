#!/usr/bin/env python
from __future__ import division
import numpy as np
from rbf.halton import Halton
from rbf.geometry import boundary_contains
from rbf.geometry import is_valid
import logging 
from modest import funtime
import matplotlib.pyplot as plt
import mayavi.mlab
import modest
logger = logging.getLogger(__name__)


@funtime
def mcint(f,vert,smp,samples=None,lower_bounds=None,upper_bounds=None):
  '''
  Monte Carlo integration of a function that takes a (M,2) or (M,3)
  array of points and returns an (M,) vector. vert and smp are the
  vertices and simplices which define the bounds of integration. N
  is the number of samples.
  '''
  dim = vert.shape[1]
  assert is_valid(smp), (
    'invalid simplices, see documentation for rbf.geometry.is_valid')

  batch_size = 1000000

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

  H = Halton(dim)

  if samples is None:
    samples = 50**dim

  soln = 0
  count = 0
  minval = np.inf
  maxval = -np.inf
  while count < samples:
    if (count + batch_size) > samples:
      batch_size = samples - count

    pnts = H(batch_size)*(ub-lb) + lb
    #pnts = np.random.random((batch_size,dim))*(ub-lb) + lb
    val = f(pnts)
    is_inside = boundary_contains(pnts,vert,smp)
    if np.any(is_inside):
      if minval > np.min(val[is_inside]):
        minval = np.min(val[is_inside])

      if maxval < np.max(val[is_inside]):
        maxval = np.max(val[is_inside])

    val[~is_inside] = 0.0
    soln += np.sum(val)*np.prod(ub-lb)
    count += batch_size

  soln /= samples
  err = np.prod(ub-lb)*np.std(val)/np.sqrt(samples)

  return soln,err,minval,maxval


def divide_bbox(lb,ub,depth=0):
  '''
  divides the bounding box in half along an axis determined by the 
  recursion depth
  '''
  lb = np.asarray(lb,dtype=float)
  ub = np.asarray(ub,dtype=float)
  mp = (lb + ub)/2.0
  dim = lb.shape[0]
  out = []

  # there is surely a more elegent way to do this, but it is faster
  # to code up the brute force method than to think of a better way
  # to do this               
  if dim == 1:
    out += [(np.array([lb[0]]),
             np.array([mp[0]]))]
    out += [(np.array([mp[0]]),
             np.array([ub[0]]))]

  if dim == 2:
    if depth%2 == 0:
      out += [(np.array([lb[0],lb[1]]),
               np.array([mp[0],ub[1]]))]
      out += [(np.array([mp[0],lb[1]]),
               np.array([ub[0],ub[1]]))]
    if depth%2 == 1:
      out += [(np.array([lb[0],lb[1]]),
               np.array([ub[0],mp[1]]))]
      out += [(np.array([lb[0],mp[1]]),
               np.array([ub[0],ub[1]]))]

  if dim == 3:
    if depth%3 == 0:
      out += [(np.array([lb[0],lb[1],lb[2]]),
               np.array([mp[0],ub[1],ub[2]]))]
      out += [(np.array([mp[0],lb[1],lb[2]]),
               np.array([ub[0],ub[1],ub[2]]))]
    if depth%3 == 1:
      out += [(np.array([lb[0],lb[1],lb[2]]),
               np.array([ub[0],mp[1],ub[2]]))]
      out += [(np.array([lb[0],mp[1],lb[2]]),
               np.array([ub[0],ub[1],ub[2]]))]
    if depth%3 == 2:
      out += [(np.array([lb[0],lb[1],lb[2]]),
               np.array([ub[0],ub[1],mp[2]]))]
      out += [(np.array([lb[0],lb[1],mp[2]]),
               np.array([ub[0],ub[1],ub[2]]))]

  return out


def rmcint(f,vert,smp,tol=None,max_depth=20,samples=None,
           lower_bounds=None,upper_bounds=None,_depth=0):
  '''
  recursive Monte Carlo integration
  '''
  vert = np.asarray(vert,dtype=float)
  smp = np.asarray(smp,dtype=int)
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

  if tol is None:
    # if no tolerance is specified then an rough initial guess for the
    # integral is made and then the tolerance is set to 0.001 times
    # that value
    init_guess = mcint(f,vert,smp,samples=samples,
                       lower_bounds=lower_bounds,
                       upper_bounds=upper_bounds)
    tol = np.abs(init_guess[0]/1000.0)

  soln = 0.0
  err = 0.0
  minval = np.inf
  maxval = -np.inf
  for lbi,ubi in divide_bbox(lb,ub,depth=_depth):
    out = mcint(f,vert,smp,samples=samples,
                lower_bounds=lbi,upper_bounds=ubi)
    solni,erri,mini,maxi = out

    if mini < minval:
      minval = mini

    if maxi > maxval:
      maxval = maxi

    if (erri > tol) & (_depth < max_depth):
      out = rmcint(f,vert,smp,tol=tol,samples=samples,
                   lower_bounds=lbi,upper_bounds=ubi,
                   _depth=_depth+1,max_depth=max_depth)
      new_solni,new_erri,mini,maxi = out

      if mini < minval:
        minval = mini

      if maxi > maxval:
        maxval = maxi                

      # combine the previous solution with the new one  
      numer = solni/(erri**2) + new_solni/(new_erri**2)
      denom = 1.0/(erri**2) + 1.0/(new_erri**2)
      solni = numer/denom
      erri = 1.0/np.sqrt(denom)

    soln += solni
    err += erri**2

  err = np.sqrt(err)

  return soln,err,minval,maxval


def normalize(fin,vert,smp,kind='integral',tol=None,nodes=None):
  '''
  normalize a function that takes a (N,1) array and returns an (N,)
  array. The kind of normalization is specified with "kind", which can
  either be "integral" to normalize so that the function integrates to
  1.0, "max" so that the maximum value is 1.0, or "density" so that
  the function returns a node density with "nodes" being the total
  number of nodes in the domain
  '''
  out = rmcint(fin,vert,smp,tol=tol)
  integral,err,minval,maxval = out 
  if kind == 'integral':
    denom = integral

  if kind == 'max':
    denom = maxval

  if kind == 'density':
    if nodes is None:
      raise ValueError(
        'must specify number of nodes with "nodes" key word argument '
        'if normalizing by density')

    denom = integral/nodes

  if denom == 0.0:
    raise ValueError(
      'normalized function by 0, this may be due to to an '
      'insufficiently large MC integration sample size')

  def fout(p):
    return fin(p)/denom

  return fout


def normalize_decorator(*args,**kwargs):
  def dout(fin):
    fout = normalize(fin,*args,**kwargs)
    return fout
  return dout
