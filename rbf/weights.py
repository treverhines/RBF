#!/usr/bin/env python
from __future__ import division
import numpy as np
import rbf.basis
import modest
import scipy
import scipy.spatial
import random
import logging
logger = logging.getLogger(__name__)


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
  if Np == 0:
    out = np.zeros((Ns,Ns))
  else:
    out = np.zeros((Ns + 1 + (Np-1)*Ndim,
                    Ns + 1 + (Np-1)*Ndim))

  out[:Ns,:Ns] = basis(n,n,eps)
  if Np > 0:
    out[Ns,:Ns] = 1
    out[:Ns,Ns] = 1

  if Np > 1:
    poly_mat = np.vstack([vpoly(n[:,i],range(1,Np)) 
                          for i in range(Ndim)])
    out[Ns+1:,:Ns] = poly_mat
    out[:Ns,Ns+1:] = poly_mat.T

  return out


def drbf(x,n,eps,Np,diff,basis):
  n = np.asarray(n)
  x = np.asarray(x)
  x = x[None,:]
  Ns,Ndim = n.shape
  eps = eps*np.ones(Ns)
  if Np == 0:
    out = np.zeros(Ns)
  else:
    out = np.zeros(Ns + 1 + (Np-1)*Ndim)

  out[:Ns] = basis(x,n,eps,diff=diff)[0,:]
  if Np > 0:
    out[Ns] = float(sum(diff) == 0)

  if Np > 1:
    poly_terms = [[poly(x[0,i],j,diff=diff[i]) 
                   for j in range(1,Np)] 
                   for i in range(Ndim)]
    poly_terms = np.array(poly_terms).flatten()
    out[Ns+1:] = poly_terms

  return out


def shape_factor(nodes,s,basis,cond=10,samples=100):
  '''
  The shape factor for stencil i, eps_i, is chosen by
 
    eps_i = alpha/mu_i                  
 
  where mu_i is the mean shortest path between nodes in stencil i. and 
  alpha is a proportionality constant chosen to obtain the desired  
  condition number for each stencils Vandermonde matrix.  This    
  function assumes that the optimal alpha for each stencil is equal. 
  Alpha is then estimated from the specified number of stencil samples
  and then eps_i is returned for each stencil
  '''
  alpha_list = np.zeros(samples)
  for i,si in enumerate(random.sample(s,samples)):
    eps = optimal_shape_factor(nodes[si,:],basis,cond)
    T = scipy.spatial.cKDTree(nodes[si,:])
    dx,idx = T.query(nodes[si,:],2)
    mu = np.mean(dx[:,1])
    alpha_list[i] = eps*mu

  alpha = np.mean(alpha_list)
  eps_list = np.zeros(s.shape[0])
  for i,si in enumerate(s):
    T = scipy.spatial.cKDTree(nodes[si,:])
    dx,idx = T.query(nodes[si,:],2)
    mu = np.mean(dx[:,1])
    eps_list[i] = alpha/mu

  return eps_list


def optimal_shape_factor(n,basis,cond):
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
                            atol=1e-2,rtol=1e-8,
                            LM_param=1e-2,
                            maxitr=500)
  return eps[0]


def is_operator(diff):
  '''
  diff can either be a tuple describing the differentiation order in
  each direction, e.g. (0,0,1), or it can describe a differential
  operator, e.g. [(1,(0,0,1)),(2,(0,1,0))], where each element
  contains a coefficient and differentiation tuple for each term in
  the operator. This function returns true if diff is describing an
  operator.
  '''
  # if each element in diff is iterable then return True. This returns
  # True if diff is an empty list 
  if all(hasattr(d,'__iter__') for d in diff):
    return True
  else:
    return False



def rbf_weight(x,n,diff,basis=rbf.basis.mq,Np=1,eps=None,cond=10.0):
  '''
  finds the weights, w, such that

  f_i(c_j) = f(||c_j - c_i||) 

  | f_0(c_0) ... f_0(c_N) |     | L[f_0(y)]y=x  |
  |    :             :    | w = |     :         |
  | f_N(c_0) ... f_N(c_N) |     | L[f_N(y)]y=x  |
  '''
  if eps is None:
    eps = optimal_shape_factor(n,basis,cond)  

  x = np.array(x,copy=True)
  n = np.array(n,copy=True)
  # center about x
  n -= x
  x -= x
  A = vrbf(n,eps,Np,basis)
  if is_operator(diff):
    if Np == 0:
      d = np.zeros(n.shape[0])
    else:
      d = np.zeros(n.shape[0] + 1 + (Np-1)*n.shape[1])

    for coeff,diff_tuple in diff:
      d += coeff*drbf(x,n,eps,Np,diff_tuple,basis)

  else:
    d = drbf(x,n,eps,Np,diff,basis)

  try:
    w = np.linalg.solve(A,d)[:n.shape[0]]

  except np.linalg.linalg.LinAlgError:
    print(
      'WARNING: encountered singular matrix. Now trying to compute weights '
      'with condition number of 10^%s' % (cond-0.5))
    logger.warning(
      'encountered singular matrix. Now trying to compute weights '
      'with condition number of 10^%s' % (cond-0.5))
    return rbf_weight(x,n,diff,basis=basis,
                      Np=Np,eps=None,cond=cond-0.5)       
    
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

