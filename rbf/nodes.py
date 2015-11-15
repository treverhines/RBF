#!/usr/bin/env python
from __future__ import division
import numpy as np
import scipy.spatial
import rbf.halton
import modest

def _boundary_filter(nodes,bnd=None):
  if bnd == None:
    return nodes
  else:
    return nodes[bnd(nodes)]

def _density_filter(nodes,seq,rho=None):
  if rho == None:
    return nodes
  else:
    return nodes[rho(nodes) > seq]

def _proximity_filter(nodes,seq,rho=None):
  if rho == None:
    return nodes
  else:
    return nodes[rho(nodes) > seq]

def _intersection(inside,outside,bnd):
  tol = 1e-10
  inside = np.copy(inside)
  outside = np.copy(outside)
  err = np.linalg.norm(inside-outside,axis=1)
  while np.any(err > tol):
    mid = (inside + outside)/2
    result = bnd(mid)
    inside[result] = mid[result]
    outside[~result] = mid[~result]
    err = np.linalg.norm(inside-outside,axis=1)

  return inside

def _repel_k(free_nodes,fix_nodes,n,eps):
  tol = 1e-10
  nodes = np.vstack((free_nodes,fix_nodes))
  T = scipy.spatial.cKDTree(nodes)
  d,i = T.query(free_nodes,n)
  i = i[:,1:]
  d = d[:,1:]
  force = np.sum((free_nodes[:,None,:] - nodes[i,:])/d[:,:,None]**3,1)
  mag = np.linalg.norm(force,axis=1)
  idx = mag > tol
  force[idx] /= mag[idx,None]
  force[~idx] *= 0.0
  step = eps*d[:,0,None]*force
  free_nodes = free_nodes + step
  return free_nodes

def _repel(free_nodes,fix_nodes=None,itr=10,n=10,eps=0.1,bnd=None):
  if fix_nodes is None:
    fix_nodes = np.zeros((0,free_nodes.shape[1]))

  fix_nodes = np.asarray(fix_nodes)
  if n > (free_nodes.shape[0]+fix_nodes.shape[0]):
    n = free_nodes.shape[0]+fix_nodes.shape[0]

  for k in range(itr):
    old_free_nodes = free_nodes
    free_nodes = _repel_k(free_nodes,fix_nodes,n,eps)
    if bnd is not None:
      is_outside = ~bnd(free_nodes)
      free_nodes[is_outside] = _intersection(
                                 old_free_nodes[is_outside],     
                                 free_nodes[is_outside],
                                 bnd)
  return free_nodes        

def pick_nodes(N,lb,ub,bnd_nodes=None,bnd=None,rho=None,
               sample_size=None,
               repel_itr=10,repel_n=10,repel_eps=0.1):
  lb = np.asarray(lb)
  ub = np.asarray(ub)
  assert len(lb) == len(ub)
  assert np.all(ub > lb)
  if sample_size is None: 
    sample_size = N//2

  ndim = len(lb)
  H = rbf.halton.Halton(ndim+1)
  nodes = np.zeros((0,ndim))
  while nodes.shape[0] < N:
    size_start = nodes.shape[0]
    seqNd = H(sample_size)
    new_nodes = seqNd[:,:ndim]
    seq1d = seqNd[:,-1]
    new_nodes *= (ub - lb)
    new_nodes += lb
    new_nodes = _density_filter(new_nodes,seq1d,rho)
    nodes = np.vstack((nodes,new_nodes))
    nodes = _boundary_filter(nodes,bnd)    
    nodes = _repel(nodes,bnd_nodes,
                   itr=repel_itr,
                   n=repel_n,
                   eps=repel_eps,
                   bnd=bnd)

    size_end = nodes.shape[0]
    acceptance = (size_end - size_start)/sample_size
    if acceptance <= 0:
      print('Warning: no samples are being retained')

  return nodes[:N]



    

  
