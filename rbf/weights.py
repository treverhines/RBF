#!/usr/bin/env python
from __future__ import division
import numpy as np
import rbf.basis

def poly(p,order,diff=0):
  p = np.asarray(p,dtype=float)
  if diff == 0:
    if order == 0:
      out = 0.0*p + 1.0
    else:
      out = p**order

  else:
    if order == 0:
      out = 0.0*p
    else:
      out = order*poly(p,order-1,diff-1)

  return np.asarray(out,dtype=float)


def vpoly(c,order=None):
  c = np.asarray(c,dtype=float)
  if order is None:
    order = range(c.shape[0])

  order = np.asarray(order,dtype=int)
  out = np.zeros((order.shape[0],c.shape[0]))
  for i,r in enumerate(order):
    out[i,:] = poly(c,r)

  return out


def vrbf(c,basis=rbf.basis.mq,eps=1.0,Np=0):
  '''
  returns the matrix:

  A =   | Ar Ap |
        | Ap 0  |
     
  where Ar is consists of RBFs with specified cnts evaluated
  at those cnts. Ap consists of additional polynomial terms
  '''
  Ns,Ndim = c.shape
  eps = eps*np.ones(Ns)
  Ar = basis(c,c,eps).T
  Ap = np.zeros((0,Ns))
  for i in range(Ndim):
    if i == 0:
      Api = vpoly(c[:,i],range(Np))  
      Ap = np.vstack((Ap,Api))
    else:
      Api = vpoly(c[:,i],range(1,Np))  
      Ap = np.vstack((Ap,Api))

  Z = np.zeros((Ap.shape[0],Ap.shape[0]))
  left = np.vstack((Ar,Ap))
  right = np.vstack((Ap.T,Z))
  A = np.hstack((left,right))
  return A


def drbf(x,c,diff,basis=rbf.basis.mq,eps=1.0,Np=0):
  x = np.asarray(x,dtype=float)
  c = np.asarray(c,dtype=float)
  x = x[None,:]
  Ns,Ndim = c.shape
  eps = eps*np.ones(Ns)  
  dr = basis(x,c,eps,diff=diff)[0,:]
  dp = np.zeros(0)
  for i in range(Ndim):
    if i == 0:
      dpi = [poly(x[0,i],j,diff=diff[i]) for j in range(Np)]
      dp = np.concatenate((dp,dpi))
    else:
      dpi = [poly(x[0,i],j,diff=diff[i]) for j in range(1,Np)]
      dp = np.concatenate((dp,dpi))

  d = np.concatenate((dr,dp))
  return d    


def rbf_weight(x,c,diff,basis=rbf.basis.mq,eps=1.0,Np=0):
  '''
  finds the weights, w, such that

  f_i(c_j) = f(||c_j - c_i||) 

  | f_0(c_0) ... f_0(c_N) |     | L[f_0(y)]y=x  |
  |    :             :    | w = |     :         |
  | f_N(c_0) ... f_N(c_N) |     | L[f_N(y)]y=x  |
  '''
  x = np.asarray(x)
  c = np.asarray(c)
  A = vrbf(c,basis=basis,eps=eps,Np=Np)
  d = drbf(x,c,basis=basis,eps=eps,Np=Np,diff=diff)
  w = np.linalg.solve(A,d)[:c.shape[0]]
  return w 


def poly_weight(x,c,diff):
  '''
  finds the weights, w, such that

  f_i(c_j) = c_j**i

  | f_0(c_0) ... f_0(c_N) |     | L[f_0(y)]y=x  |
  |    :             :    | w = |     :         |
  | f_N(c_0) ... f_N(c_N) |     | L[f_N(y)]y=x  |
  '''
  x = np.asarray(x)
  c = np.asarray(c)
  A =  vpoly(c)
  d =  [poly(x,j,diff=diff) for j in range(c.shape[0])]
  w = np.linalg.solve(A,d)
  return w 

