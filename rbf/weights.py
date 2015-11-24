#!/usr/bin/env python
from __future__ import division
import numpy as np

def poly(p,order,diff=0):
  if diff == 0:
    if order == 0:
      return 0.0*p + 1.0
    else:
      return p**order

  else:
    if order == 0:
      return 0.0*p
    else:
      return order*poly(p,order-1,diff-1)

def vpoly(p,order,diff=0):
  p = np.asarray(p)
  order = np.asarray(order,dtype=int)
  out = np.zeros((order.shape[0],p.shape[0]))
  for i,r in enumerate(order):
    out[i,:] = poly(p,r,diff)

  return out


def interp_matrix(basis,cnts,eps,Np=0):
  '''
  returns the matrix:

  A =   | Ar Ap |
        | Ap 0  |
     
  where Ar is consists of RBFs with specified cnts evaluated
  at those cnts. Ap consists of additional polynomial terms
  '''
  Ns,Ndim = cnts.shape
  eps = eps*np.ones(Ns)
  Ar = basis(cnts,cnts,eps).T
  Ap = np.zeros((0,Ns))
  for i in range(Ndim):
    if i == 0:
      Api = vpoly(cnts[:,i],range(Np))  
      Ap = np.vstack((Ap,Api))
    else:
      Api = vpoly(cnts[:,i],range(1,Np))  
      Ap = np.vstack((Ap,Api))

  Z = np.zeros((Ap.shape[0],Ap.shape[0]))
  left = np.vstack((Ar,Ap))
  right = np.vstack((Ap.T,Z))
  A = np.hstack((left,right))
  return A

def data_term(basis,pnt,cnts,eps,diff,Np=0):
  pnt = pnt[None,:]
  Ns,Ndim = cnts.shape
  eps = eps*np.ones(Ns)  
  dr = basis(pnt,cnts,eps,diff=diff).T
  dp = np.zeros((0,1))
  for i in range(Ndim):
    if i == 0:
      dpi = vpoly(pnt[:,i],range(Np),diff=diff[i])
      dp = np.vstack((dp,dpi))
    else:
      dpi = vpoly(pnt[:,i],range(1,Np),diff=diff[i])
      dp = np.vstack((dp,dpi))

  d = np.vstack((dr,dp))
  d = d[:,0]
  return d    
  



