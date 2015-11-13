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

def _repel(int_nodes,fix_nodes=None,itr=10,n=10,eps=0.1):
  tol = 1e-10
  int_nodes = np.copy(int_nodes)
  if fix_nodes is None:
    fix_nodes = np.zeros((0,int_nodes.shape[1]))

  fix_nodes = np.asarray(fix_nodes)
  nodes = np.vstack((int_nodes,fix_nodes))
  if n > nodes.shape[0]:
    n = nodes.shape[0]

  for k in range(itr):
    T = scipy.spatial.cKDTree(nodes)
    d,i = T.query(int_nodes,n)

    i = i[:,1:]
    d = d[:,1:]
    force = np.sum((int_nodes[:,None,:] - nodes[i,:])/d[:,:,None]**3,1)
    mag = np.linalg.norm(force,axis=1)
    idx = mag > tol
    force[idx] /= mag[idx,None]
    force[~idx] *= 0.0
    step = eps*d[:,0,None]*force
    int_nodes += step
    nodes = np.vstack((int_nodes,fix_nodes))

  return int_nodes        

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
    nodes = _repel(nodes,bnd_nodes,
                   itr=repel_itr,
                   n=repel_n,
                   eps=repel_eps)
    nodes = _boundary_filter(nodes,bnd)    
    size_end = nodes.shape[0]
    acceptance = (size_end - size_start)/sample_size
    if acceptance <= 0:
      print('Warning: no samples are being retained')

  return nodes[:N]



    

  
