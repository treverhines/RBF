''' 
This module creates stencils used for the Radial Basis Function Finite 
Difference (RBF-FD) method

'''
from __future__ import division
import numpy as np
from scipy.spatial import cKDTree
from rbf.geometry import intersection_count
from itertools import combinations
import networkx

class StencilError(Exception):
  ''' 
  raised when a stencil cannot be made for topological purposes
  '''
  pass

def stencils_to_edges(stencils):
  ''' 
  returns an array of edges defined by the *stencils*
  
  Parameters
  ----------
  stencil : (N,D) int aray

  Returns
  -------
  edges : (K,2) int array

  '''
  stencils = np.asarray(stencils)
  N,S = stencils.shape
  node1 = np.arange(N)[:,None].repeat(S,axis=1)
  node2 = np.array(stencils,copy=True)
  edges = zip(node1.flatten(),node2.flatten())
  edges = np.array(edges,dtype=int)
  return edges


def is_connected(stencils):
  ''' 
  returns True if *stencils* forms a connected graph (i.e. connectivity 
  greater than 0)

  Parameters
  ----------
  stencil : (N,D) int aray

  Returns
  -------
  out : bool

  '''
  edges = stencils_to_edges(stencils)
  # edges needs to be a list of tuples
  edges = [tuple(e) for e in edges] 
  graph = networkx.Graph(edges)
  return networkx.is_connected(graph)


def connectivity(stencils):
  ''' 
  returns the minimum number of edges that must be removed in order to 
  break the connectivity of the graph defined by the *stencils*

  Parameters
  ----------
  stencil : (N,D) int aray

  Returns
  -------
  out : int

  '''
  edges = stencils_to_edges(stencils)
  # edges needs to be a list of tuples
  edges = [tuple(e) for e in edges] 
  graph = networkx.Graph(edges)
  return networkx.node_connectivity(graph)


def _argsort_closest(c,x):
  ''' 
  Returns the indices of nodes in *x* sorted in order of distance to *c*
  '''
  dist = np.sum((x - c[None,:])**2,axis=1)
  idx = np.argsort(dist)
  return idx


def _has_edge_intersections(c,x,vert,smp,check_all_edges):
  ''' 
  Check if any of the edges (*c*,*x[i]*) intersect the boundary 
  defined by *vert* and *smp*. If *check_all_edges* is True then the 
  edges (*x[i]*,*x[j]*) are also tested.
  '''
  N = len(x)
  cext = np.repeat(c[None,:],N,axis=0)
  # number of times each edge intersects the boundary
  count = intersection_count(x,cext,vert,smp)
  # return True if there are no intersections
  intersects = np.any(count > 0)
  if ~intersects & check_all_edges:
    a,b = [],[]
    for i,j in combinations(range(N),2):
      a += [i]
      b += [j]

    # number of times each edge intersects the boundary
    count = intersection_count(x[a],x[b],vert,smp)
    # return True if there are no intersections
    intersects = np.any(count > 0)

  return intersects


def _stencil(c,x,n,vert,smp,check_all_edges):
  ''' 
  Forms a stencil about *c* made up of *n* nearby nodes in *x*. The 
  stencil is constrained so that it does not reach across the boundary 
  defined by *vert* and *smp*.
  '''
  sorted_idx = _argsort_closest(c,x)
  stencil_idx = []    
  for si in sorted_idx:
    if len(stencil_idx) == n:
      break

    stencil_idx += [si]
    if _has_edge_intersections(c,x[stencil_idx],vert,smp,check_all_edges):
      stencil_idx.pop()

  if len(stencil_idx) == n:
    return np.array(stencil_idx,dtype=int)
  else: 
    raise StencilError('cannot not form a stencil with size %s' % n)


def _stencil_network_no_boundary(x,p,n):
  ''' 
  Returns the *n* nearest points in *p* for each point in *x*
  '''
  if n == 0:
    out = np.zeros((x.shape[0],0),dtype=int)
  else:
    T = cKDTree(p)
    dummy,out = T.query(x,n)
    if n == 1:
      out = out[:,None]

  return out 


def stencil_network(x,p,n,vert=None,smp=None,check_all_edges=False):
  ''' 
  Forms a stencil for each point in *x*. Each stencil is made up of 
  *n* nearby points from *p*. Stencils can be constrained to not 
  intersect a boundary defined by *vert* and *smp*.
  
  Parameters
  ----------
  x : (N,D) array
    Estimation points. A stencil will be made for each point in *x*

  p : (M,D) array
    observation points. The stencils will be made up of points from *p*    

  n : int
    Stencil size

  vert : (P,D) array, optional
    Vertices making up the boundary which stencils cannot intersect

  smp : (Q,D) array, optional
    Connectivity of vertices in *vert*

  check_all_edges : bool, optional
    If False then a stencil centered on *x[i]* will not contain any 
    points in *p* which form an edge with *x[i]* that intersects the 
    boundary.  If True then, additionally, no stencils will contain 
    two points from *p* which form an edge that intersects the 
    boundary. 
    
  Returns
  -------
  sn : (N,D) array
    indices of points in *p* which form a stencil for each point in *x*
    
  '''
  Nx = len(x)
  Np = len(p)
  x = np.asarray(x)
  p = np.asarray(p)
  if n > Np:
    raise StencilError('cannot form a stencil with size %s from %s nodes' % (n,Np))
    
  if (vert is None) | (smp is None):
    vert = np.zeros((0,x.shape[1]),dtype=float)
    smp = np.zeros((0,x.shape[1]),dtype=int)
  
  sn = _stencil_network_no_boundary(x,p,n)
  if smp.shape[0] == 0:
    return sn

  for i in range(Nx):
    if _has_edge_intersections(x[i],p[sn[i]],vert,smp,check_all_edges):
      sn[i,:] = _stencil(x[i],p,n,vert,smp,check_all_edges)

  return sn


def nearest(x,p,n,vert=None,smp=None):
  ''' 
  Returns the *n* nearest points in *p* for each point in *x*. Nearest 
  neighbors cannot extend across the boundary defined by *vert* and 
  *smp*
  
  Parameters
  ----------
  x : (N,D) array
    Query points.

  p : (M,D) array
    Population points.

  vert : (P,D) array, optional     
    Vertices making up a boundary 

  smp : (Q,D) array, optional  
    Connectivity of vertices in *vert*
    
  Returns
  -------
  idx : (N,D) array
    Indices of nearest points in *p*

  dist : (N,D) array
    Distance to the nearest points in *p*  

  '''
  idx = stencil_network(x,p,n,vert=vert,smp=smp,check_all_edges=False)
  dist = np.sqrt(np.sum((x[:,None,:] - p[idx])**2,axis=2))
  return idx,dist
  
