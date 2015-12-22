#!/usr/bin/env python
from __future__ import division
import numpy as np
import rbf.basis
import modest
import scipy


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


def vrbf(n,eps,Np,basis):
  '''
  returns the matrix:

  A =   | Ar Ap |
        | Ap 0  |
     
  where Ar is consists of RBFs with specified cnts evaluated
  at those cnts. Ap consists of additional polynomial terms
  '''
  n = np.asarray(n)
  Ns,Ndim = n.shape
  eps = eps*np.ones(Ns)
  Ar = basis(n,n,eps)
  Ap = np.zeros((0,Ns))
  if Np > 0:
      Api = np.ones(len(n))
      Ap = np.vstack((Ap,Api))
    
  for i in range(Ndim):
    Api = vpoly(n[:,i],range(1,Np))  
    Ap = np.vstack((Ap,Api))

  Z = np.zeros((Ap.shape[0],Ap.shape[0]))
  left = np.vstack((Ar,Ap))
  right = np.vstack((Ap.T,Z))
  A = np.hstack((left,right))
  return A


def drbf(x,n,eps,Np,diff,basis):
  n = np.asarray(n)
  x = np.asarray(x)
  x = x[None,:]
  Ns,Ndim = n.shape
  eps = eps*np.ones(Ns)
  dr = basis(x,n,eps,diff=diff)[0,:]
  dp = np.zeros(0)
  if Np > 0:
    dpi = [float(sum(diff) == 0)]
    dp = np.concatenate((dp,dpi))       

  for i in range(Ndim):
    dpi = [poly(x[0,i],j,diff=diff[i]) for j in range(1,Np)]
    dp = np.concatenate((dp,dpi))

  d = np.concatenate((dr,dp))
  return d    

@modest.funtime
def optimal_eps(n,basis,cond):
  n = np.asarray(n)
  def system(eps):
    A = basis(n,n,eps[0]*np.ones(len(n)))
    cond = np.linalg.cond(A)
    return np.array([np.log10(cond)])

  T = scipy.spatial.cKDTree(n)
  dist,idx = T.query(n,2)
  # average shortest distance between nodes
  dx = np.mean(dist[:,1])
  eps = [0.1/dx]
  eps = modest.nonlin_lstsq(system,[cond],
                            eps,
                            solver=modest.nnls,
                            atol=1e-2,rtol=1e-4,
                            LM_param=1e-2,
                            maxitr=100)
  return eps[0]

@modest.funtime
def rbf_weight(x,n,diff,basis=rbf.basis.mq,Np=1,eps=None,cond=10):
  '''
  finds the weights, w, such that

  f_i(c_j) = f(||c_j - c_i||) 

  | f_0(c_0) ... f_0(c_N) |     | L[f_0(y)]y=x  |
  |    :             :    | w = |     :         |
  | f_N(c_0) ... f_N(c_N) |     | L[f_N(y)]y=x  |
  '''
  if eps is None:
    eps = optimal_eps(n,basis,cond)  

  x = np.array(x,copy=True)
  n = np.array(n,copy=True)
  # center about x
  n -= x
  x -= x
  A = vrbf(n,eps,Np,basis)
  d = drbf(x,n,eps,Np,diff,basis)
  w = np.linalg.solve(A,d)[:n.shape[0]]
  return w 


def poly_weight(x,n,diff):
  '''
  finds the weights, w, such that

  f_i(c_j) = c_j**i

  | f_0(c_0) ... f_0(c_N) |     | L[f_0(y)]y=x  |
  |    :             :    | w = |     :         |
  | f_N(c_0) ... f_N(c_N) |     | L[f_N(y)]y=x  |
  '''
  x = np.array(x,copy=True)
  n = np.array(n,copy=True)
  n -= x
  x -= x
  A =  vpoly(n)
  d =  [poly(x,j,diff=diff) for j in range(n.shape[0])]
  w = np.linalg.solve(A,d)
  return w 

