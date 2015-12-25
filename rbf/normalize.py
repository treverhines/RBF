#!/usr/bin/env python
from __future__ import division
import numpy as np
import rbf.halton
import rbf.geometry
import logging 
import modest
logger = logging.getLogger(__name__)

@modest.funtime
def mcint_1d(f,lb,ub,N=None):
  '''
  Monte Carlo integration of a function that takes a (M,1) array
  of points and returns an (M,) vector. lb and ub are the bounds
  of integration and N is the number of samples
  '''
  if N is None:
    N = 200

  pnts = rbf.halton.halton(N,1)*(ub-lb) + lb
  val = f(pnts)
  soln = np.mean(val)*abs(ub-lb)
  return soln


@modest.funtime
def mcmax_1d(f,lb,ub,N=None):
  '''
  Monte Carlo estimation of the maximum value of a function that
  takes a (M,1) array of points and returns an (M,) vector.
  '''
  if N is None:
    N = 200

  pnts = rbf.halton.halton(N,1)*(ub-lb) + lb
  val = f(pnts)
  soln = np.max(val)
  return soln


@modest.funtime
def mcint(f,vert,smp,N=None):
  '''
  Monte Carlo integration of a function that takes a (M,2) or (M,3)
  array of points and returns an (M,) vector. vert and smp are the
  vertices and simplices which define the bounds of integration. N
  is the number of samples.
  '''
  sample_size = 1000000  
  lb = np.min(vert,0)
  ub = np.max(vert,0)
  dim = lb.shape[0]
  soln = 0 
  count = 0
  H = rbf.halton.Halton(dim)
  if N is None:
    N = 200**dim

  while count < N:
    if (count + sample_size) > N:
      sample_size = N - count

    pnts = H(sample_size)*(ub-lb) + lb
    val = f(pnts)
    if dim == 2:
      val = val[rbf.geometry.contains_2d(pnts,vert,smp)]
    if dim == 3:
      val = val[rbf.geometry.contains_3d(pnts,vert,smp)]

    soln += np.sum(val)*np.prod(ub-lb)
    count += sample_size
    
  soln /= N
  return soln


@modest.funtime
def mcmax(f,vert,smp,N=None):
  '''
  Monte Carlo estimation of the maximum of a function that takes a
  (M,2) or (M,3) array of points and returns an (M,) vector. vert and
  smp are the vertices and simplices which define the bounds over which
  a maximum will be estimated. N is the number of samples.
  '''
  sample_size = 1000000  
  lb = np.min(vert,0)
  ub = np.max(vert,0)
  dim = lb.shape[0]
  soln = -np.inf
  count = 0
  H = rbf.halton.Halton(dim)
  if N is None:
    N = 200**dim

  while count < N:
    if (count + sample_size) > N:
      sample_size = N - count

    pnts = H(sample_size)*(ub-lb) + lb
    val = f(pnts)
    if dim == 2:
      val = val[rbf.geometry.contains_2d(pnts,vert,smp)]
    if dim == 3:
      val = val[rbf.geometry.contains_3d(pnts,vert,smp)]

    maxval = np.max(val)
    if maxval > soln:
      soln = maxval
    
    count += sample_size

  return soln


def normalize_1d(fin,lb,ub,kind='integral',N=None,nodes=None):
  '''
  normalize a function that takes a (N,1) array and returns an (N,)
  array. The kind of normalization is specified with "kind", which can
  either be "integral" to normalize so that the function integrates to
  1.0, "max" so that the maximum value is 1.0, or "density" so that
  the function returns a node density with "nodes" being the total
  number of nodes in the domain
  '''
  if kind == 'integral':
    denom = mcint_1d(fin,lb,ub,N=N)

  if kind == 'max':
    denom = mcmax_1d(fin,lb,ub,N=N)

  if kind == 'density':
    if nodes is None:
      raise ValueError(
        'must specify number of nodes with 'nodes' key word argument '
        'if normalizing by density')

    denom = mcint_1d(fin,lb,ub,N=N)/nodes

  if denom == 0.0:
    raise ValueError(
      'normalized function by 0, this may be due to to an '
      'insufficiently large MC integration sample size')

  def fout(p):
    return fin(p)/denom

  return fout


def normalize_decorator_1d(*args,**kwargs):
  def dout(fin):
    fout = normalize_1d(fin,*args,**kwargs)
    return fout
  return dout


def normalize(fin,vert,smp,by='integral',N=None,nodes=None):
  '''
  normalize a function that takes a (N,1) array and returns an (N,)
  array. The kind of normalization is specified with "kind", which can
  either be "integral" to normalize so that the function integrates to
  1.0, "max" so that the maximum value is 1.0, or "density" so that
  the function returns a node density with "nodes" being the total
  number of nodes in the domain
  '''
  if by == 'integral':
    denom = mcint(fin,vert,smp,N=N)
  if by == 'max':
    denom = mcmax(fin,vert,smp,N=N)

  if by == 'density':
    if nodes is None:
      raise ValueError(
        'must specify number of nodes with "nodes" key word argument '
        'if normalizing by density')

    denom = mcint_1d(fin,lb,ub,N=N)/nodes

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
