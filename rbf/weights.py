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


def vrbf(nodes,centers,eps,Np,basis):
  '''
  returns the matrix:

  A =   | Ar Ap |
        | Ap 0  |
     
  where Ar is consists of RBFs with specified cnts evaluated
  at those cnts. Ap consists of additional polynomial terms
  '''
  nodes = np.asarray(nodes)
  centers = np.asarray(centers)
  Ns,Ndim = nodes.shape
  eps = eps*np.ones(Ns)
  if Np == 0:
    out = np.zeros((Ns,Ns))
  else:
    out = np.zeros((Ns + 1 + (Np-1)*Ndim,
                    Ns + 1 + (Np-1)*Ndim))

  out[:Ns,:Ns] = basis(nodes,centers,eps).T
  if Np > 0:
    out[Ns,:Ns] = 1
    out[:Ns,Ns] = 1

  if Np > 1:
    poly_mat = np.vstack([vpoly(nodes[:,i],range(1,Np)) 
                          for i in range(Ndim)])
    out[Ns+1:,:Ns] = poly_mat
    out[:Ns,Ns+1:] = poly_mat.T

  return out


def drbf(x,centers,eps,Np,diff,basis):
  centers = np.asarray(centers)
  x = np.asarray(x)
  x = x[None,:]
  Ns,Ndim = centers.shape
  eps = eps*np.ones(Ns)
  if Np == 0:
    out = np.zeros(Ns)
  else:
    out = np.zeros(Ns + 1 + (Np-1)*Ndim)

  out[:Ns] = basis(x,centers,eps,diff=diff)[0,:]
  if Np > 0:
    out[Ns] = float(sum(diff) == 0)

  if Np > 1:
    poly_terms = [[poly(x[0,i],j,diff=diff[i]) 
                   for j in range(1,Np)] 
                   for i in range(Ndim)]
    poly_terms = np.array(poly_terms).flatten()
    out[Ns+1:] = poly_terms

  return out


def shape_factor(nodes,s,basis,centers=None,alpha=None,cond=10,samples=100):
  '''
  The shape factor for stencil i, eps_i, is chosen by
 
    eps_i = alpha/mu_i                  
 
  where mu_i is the mean shortest path between nodes in stencil i. and 
  alpha is a proportionality constant.  This function assumes the same 
  alpha for each stencil.  If alpha is not given then an alpha is estimated
  which produces the desired condition number for the Vandermonde matrix 
  of each stencil. if alpha is given then cond does nothing.  This funtion
  returns eps_i for each stencil
  '''
  if centers is None:
    centers = nodes

  nodes = np.asarray(nodes,dtype=float)
  s = np.asarray(s,dtype=int)
  centers = np.asarray(centers,dtype=float)

  if alpha is None:
    alpha_list = []
    for si in random.sample(s,samples):
      eps = condition_based_shape_factor(nodes[si,:],centers[si,:],basis,cond)
      if eps is not None:
        T = scipy.spatial.cKDTree(centers[si,:])
        dx,idx = T.query(centers[si,:],2)
        mu = np.mean(dx[:,1])
        alpha_list += [eps*mu]

    if len(alpha_list) == 0:
      raise ValueError(
        'did not find a shape parameters which produces the desired '
        'condition number for any stencils')
  
    alpha = np.mean(alpha_list)
    logger.info('using shape parameter %s' % alpha)

  eps_list = np.zeros(s.shape[0])
  for i,si in enumerate(s):
    T = scipy.spatial.cKDTree(centers[si,:])
    dx,idx = T.query(centers[si,:],2)
    mu = np.mean(dx[:,1])
    eps_list[i] = alpha/mu

  return eps_list


def condition_based_shape_factor(nodes,centers,basis,cond):
  nodes = np.asarray(nodes)
  centers = np.asarray(centers)
  def system(eps):
    A = basis(nodes,centers,eps[0]*np.ones(len(nodes)))
    cond = np.linalg.cond(A)
    return np.array([np.log10(cond)])

  T = scipy.spatial.cKDTree(centers)
  dist,idx = T.query(centers,2)
  # average shortest distance between nodes
  dx = np.mean(dist[:,1])
  eps = [0.1/dx]
  eps,pred_cond = modest.nonlin_lstsq(system,[cond],
                                      eps,
                                      solver=modest.nnls,
                                      atol=1e-2,rtol=1e-8,
                                      LM_param=1e-2,
                                      maxitr=500,
                                      output=['solution','predicted'])
  if np.abs(pred_cond - cond) > 1e-2:
    logger.warning(
      'did not find a shape parameter which produces the desired '
      'condition number')  
    return None

  else:
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



def rbf_weight(x,nodes,diff,centers=None,basis=rbf.basis.mq,Np=1,eps=None,cond=10.0):
  '''
  finds the weights, w, such that

  f_i(n_j) = f(||n_j - c_i||) 

  | f_0(n_0) ... f_0(n_N) |     | L[f_0(y)]y=x  |
  |    :             :    | w = |     :         |
  | f_N(n_0) ... f_N(n_N) |     | L[f_N(y)]y=x  |
  '''
  if centers is None:
    centers = nodes

  if eps is None:
    eps = condition_based_shape_factor(nodes,centers,basis,cond)  
    if eps is None:
      raise ValueError(
        'cannot find shape parameter that produces a condition number '
        'of %s' % cond)

  x = np.array(x,copy=True)
  nodes = np.array(nodes,copy=True)
  centers = np.array(centers,copy=True)
  # center about x
  nodes -= x
  x -= x
  centers -= x

  A = vrbf(nodes,centers,eps,Np,basis)
  if is_operator(diff):
    if Np == 0:
      d = np.zeros(nodes.shape[0])
    else:
      d = np.zeros(nodes.shape[0] + 1 + (Np-1)*nodes.shape[1])

    for coeff,diff_tuple in diff:
      d += coeff*drbf(x,centers,eps,Np,diff_tuple,basis)

  else:
    d = drbf(x,centers,eps,Np,diff,basis)

  try:
    w = np.linalg.solve(A,d)[:nodes.shape[0]]

  except np.linalg.linalg.LinAlgError:
    print(
      'WARNING: encountered singular matrix. Now trying to compute weights '
      'with condition number of 10^%s' % (cond-0.5))
    logger.warning(
      'encountered singular matrix. Now trying to compute weights '
      'with condition number of 10^%s' % (cond-0.5))
    return rbf_weight(x,nodes,diff,centers=centers,basis=basis,
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

