#!/usr/bin/env python
from __future__ import division
import numpy as np
import rbf.halton
import rbf.geometry
import logging 
logger = logging.getLogger(__name__)

def mcint_1d(f,lb,ub,N=10000):
  '''
  Monte Carlo integration of a function that takes a (M,1) array
  of points and returns an (M,) vector. lb and ub are the bounds
  of integration and N is the number of samples
  '''
  pnts = rbf.halton.halton(N,1)*(ub-lb) + lb
  val = f(pnts)
  soln = np.mean(val)*(ub-lb)
  return soln


def mcmax_1d(f,lb,ub,N=10000):
  '''
  Monte Carlo estimation of the maximum value of a function that
  takes a (M,1) array of points and returns an (M,) vector.
  '''
  pnts = rbf.halton.halton(N,1)*(ub-lb) + lb
  val = f(pnts)
  soln = np.max(val)
  return soln


def mcint(f,vert,smp,N=10000):
  '''
  Monte Carlo integration of a function that takes a (M,2) or (M,3)
  array of points and returns an (M,) vector. vert and smp are the
  vertices and simplices which define the bounds of integration. N
  is the number of samples.
  '''
  lb = np.min(vert,0)
  ub = np.max(vert,0)
  dim = lb.shape[0]
  pnts = rbf.halton.halton(N,dim)*(ub-lb) + lb
  val = f(pnts)
  if dim == 2:
    val = val[rbf.geometry.contains_2d(pnts,vert,smp)]
  if dim == 3:
    val = val[rbf.geometry.contains_3d(pnts,vert,smp)]

  soln = np.sum(val)*np.prod(ub-lb)/N
  return soln


def mcmax(f,vert,smp,N=10000):
  '''
  Monte Carlo estimation of the maximum of a function that takes a
  (M,2) or (M,3) array of points and returns an (M,) vector. vert and
  smp are the vertices and simplices which define the bounds over which
  a maximum will be estimated. N is the number of samples.
  '''
  lb = np.min(vert,0)
  ub = np.max(vert,0)
  dim = lb.shape[0]
  pnts = rbf.halton.halton(N,dim)*(ub-lb) + lb
  val = f(pnts)
  if dim == 2:
    val = val[rbf.geometry.contains_2d(pnts,vert,smp)]
  if dim == 3:
    val = val[rbf.geometry.contains_3d(pnts,vert,smp)]

  soln = np.max(val)
  return soln


def normalize_1d(fin,lb,ub,by='integral',N=10000):
  '''
  normalize a function that takes a (N,1) array and returns an (N,) 
  array by its integral or max value over the specified interval
  '''
  if by == 'integral':
    denom = mcint_1d(fin,lb,ub,N=N)
  if by == 'max':
    denom = mcmax_1d(fin,lb,ub,N=N)

  if denom == 0.0:
    print('WARNING: normalizing function by 0, this may be due to to '
          'an insufficiently large MC integration sample size')
    logger.warning('normalizing function by 0, this may be due to to '
                   'an insufficiently large MC integration sample size')
  def fout(p):
    return fin(p)/denom

  return fout


def normalize_decorator_1d(*args,**kwargs):
  def dout(fin):
    fout = normalize_1d(fin,*args,**kwargs)
    return fout
  return dout


def normalize(fin,vert,smp,by='integral',N=10000):
  '''
  normalize a function that takes a (N,2) or (N,3) array and returns
  an (N,) array by its integral or max value over the specified
  interval
  '''
  if by == 'integral':
    denom = mcint(fin,vert,smp,N=N)
  if by == 'max':
    denom = mcmax(fin,vert,smp,N=N)

  if denom == 0.0:
    print('WARNING: normalizing function by 0, this may be due to to '
          'an insufficiently large MC integration sample size')
    logger.warning('normalizing function by 0, this may be due to to '
                   'an insufficiently large MC integration sample size')
  def fout(p):
    return fin(p)/denom

  return fout

def normalize_decorator(*args,**kwargs):
  def dout(fin):
    fout = normalize(fin,*args,**kwargs)
    return fout
  return dout
