''' 
This module creates RBF-FD stencils
'''
from __future__ import division
import numpy as np
import scipy.spatial
import rbf.geometry 
import networkx
import logging
logger = logging.getLogger(__name__)


def _distance_matrix(pnts1,pnts2,vert,smp):
  ''' 
  returns a distance matrix where the distance is inf if the segment 
  connecting two points intersects the boundary 
  '''
  distance_matrix = np.zeros((pnts1.shape[0],pnts2.shape[0]))
  for i,p in enumerate(pnts1):
    pi = p[None,:].repeat(pnts2.shape[0],axis=0)
    di = np.sqrt(np.sum((p - pnts2)**2,axis=1))
    cc = np.zeros(pnts2.shape[0],dtype=int)
    cc[di!=0.0] = rbf.geometry.intersection_count(
                      pi[di!=0.0],
                      pnts2[di!=0.0],
                      vert,smp)

    distance_matrix[i,:] = di
    distance_matrix[i,cc>0] = np.inf

  return distance_matrix


def _naive_nearest(query,population,N,vert,smp):
  ''' 
  should have the same functionality as nearest except this is slower 
  and does not check the input for proper sizes and types
  '''
  dm = _distance_matrix(query,population,vert,smp)
  neighbors = np.argsort(dm,axis=1)[:,:N]
  dist = np.sort(dm,axis=1)[:,:N]
  return neighbors,dist


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


def nearest(query,population,N,vert=None,smp=None):
  ''' 
  Identifies the *N* points among the population that are closest to 
  each of the query points. If two points form a line segment which 
  intersects any part of the boundary defined by *vert* and *smp* then 
  they are considered infinitely far away.

  Parameters
  ----------
    query: (Q,D) array 
      query points 
  
    population: (P,D) array
      population points 

    N : int
      number of neighbors to find for each query point
 
    vert : (N,D) array, optional

    smp : (M,D) int array, optional
      
  Returns 
  -------
    neighbors, dist

    neighbors : (Q,N) int array
    
    dist : (Q,N) array

  Note
  ----
    If a query point lies on the boundary then this function will
    fail because the query point will be infinitely far from every 
    other point
  '''
  query = np.asarray(query,dtype=float)
  population = np.asarray(population,dtype=float)
  if smp is None:
    smp = np.zeros((0,query.shape[1]),dtype=int)
  else:
    smp = np.asarray(smp,dtype=int)

  if vert is None:
    vert = np.zeros((0,population.shape[1]),dtype=float)
  else:
    vert = np.asarray(vert,dtype=float)

  if query.ndim != 2:
    raise ValueError(
      'query points must be a two-dimensional array')

  if population.ndim != 2:
    raise ValueError(
      'population points must be a two-dimensional array')

  if N > population.shape[0]:
    raise ValueError(
      'cannot find %s nearest neighbors with %s points' % (N,population.shape[0]))

  if N < 0:
    raise ValueError(
      'must specify a non-negative number of nearest neighbors')

  # querying the KDTree returns a segmentation fault if N is zero and 
  # so this needs to be handles separately 
  if N == 0:
    dist = np.zeros((query.shape[0],0),dtype=float)
    neighbors = np.zeros((query.shape[0],0),dtype=int)
  else:
    T = scipy.spatial.cKDTree(population)
    dist,neighbors= T.query(query,N)
    if N == 1:
      dist = dist[:,None]
      neighbors = neighbors[:,None]

  if (smp.shape[0] == 0):
    # if there are no boundaries then return the output of the KDTree
    return neighbors,dist

  for i in range(query.shape[0]):
    subpop_size = N
    di = _distance_matrix(query[[i]],population[neighbors[i]],vert,smp)[0,:]
    while np.any(np.isinf(di)):
      # search over an incrementally larger set of nearest neighbors 
      # until we find N neighbors which do not cross a boundary
      if subpop_size == population.shape[0]:
        raise ValueError('cannot find %s nearest neighbors for point '
                         '%s without crossing a boundary' % (N,query[i]))
      subpop_size = min(subpop_size+N,population.shape[0])
      dummy,subpop_idx = T.query(query[i],subpop_size)
      ni,di = _naive_nearest(query[[i]],population[subpop_idx],N,vert,smp)
      ni,di = ni[0],di[0]
      neighbors[i] = subpop_idx[ni]
      dist[i] = di

  return neighbors,dist

def stencil_network(nodes,N,vert=None,smp=None):
  ''' 
  Returns the indices of *N* nearest neighbors for each node in 
  *nodes*.

  Parameters
  ----------
    nodes : (N,D) array 
    
    N : int
      stencil size
      
    vert : (P,D) array, optional
      vertices of the boundary that edges cannot cross

    smp : (Q,D) array, optional
      connectivity of the boundary vertices

  '''
  s,dx = nearest(nodes,nodes,N,vert=vert,smp=smp)
  return s
    
    
def _slice_list(lst,cuts):
  ''' 
  segments lst by cuts 
  '''
  lst = np.asarray(lst)
  cuts = np.asarray(cuts)
  cuts = np.sort(cuts)

  lbs = np.concatenate(([-np.inf],cuts))
  ubs = np.concatenate((cuts,[np.inf]))
  intervals = zip(lbs,ubs)
  out = []
  for lb,ub in intervals:
    idx_in_segment = np.nonzero((lst >= lb) & (lst < ub))[0]
    if len(idx_in_segment) > 0:
      out += [idx_in_segment]

  return out


def _stencil_network_1d(P,N):
  ''' 
  returns stencils for sequential 1d nodes
  '''
  P = int(P)
  N = int(N)
  if P < N:
    raise ValueError('cannot form a size %s stencil with %s nodes' % (N,P))

  if N == 0:
    return np.zeros((P,N),dtype=int)

  # number of repeated stencils for the right side
  right = N//2
  # number of repeated stencils for the left side
  left = N - right - 1
  center = P - N + 1

  stencils_c = np.arange(center,dtype=int)[:,None].repeat(N,axis=1)
  stencils_c += np.arange(N,dtype=int)
  stencils_r = stencils_c[[-1],:].repeat(right,axis=0)
  stencils_l = stencils_c[[0],:].repeat(left,axis=0)
  stencils = np.vstack((stencils_l,stencils_c,stencils_r))
  return stencils


def stencil_network_1d(nodes,N,vert=None,smp=None):
  ''' 
  returns a stencil network for 1d nodes where each stencil is 
  determined by adjacency and not distance.  For each node, its 
  stencil is comprised of the N//2 nodes to its right and the (N - 
  N//2 - 1) nodes to its left. This ensures better connectivity than 
  what rbf.stencil.stencil_network provides
  
  Parameters
  ----------
    nodes : (M,1) array 

    N : int
      stencil size

    vert : (P,1) array, optional
      vertices of the boundary that edges cannot cross

    smp : (Q,1) array, optional
      connectivity of the boundary vertices

  Returns
  -------
    out : (M,N) int array
    
  Example
  -------
    # create a first-order forward finite difference stencil
    >>> x = np.arange(4.0)[:,None]
    >>> stencil_network_1d(x,2)

    array([[0, 1],
           [1, 2],
           [2, 3],
           [2, 3]])
                         
  '''
  nodes = np.asarray(nodes)

  if nodes.ndim != 2:
    raise ValueError('nodes must be a two-dimensional array')

  if nodes.shape[1] != 1:
    raise ValueError('nodes must only have one spatial dimension')

  nodes = nodes[:,0]
  P = len(nodes)
  
  if vert is None:
    vert = np.zeros((0,1),dtype=float)
  if smp is None:
    smp = np.zeros((0,1),dtype=int)

  vert = np.asarray(vert,dtype=float)
  smp = np.asarray(smp,dtype=int)
  cuts = vert[smp[:,0],0]

  segments = _slice_list(nodes,cuts)
  stencil = np.zeros((P,N),dtype=int)
  for idx in segments:
    count = len(idx)
    stencil_i = _stencil_network_1d(count,N)
    sorted_idx = idx[np.argsort(nodes[idx])]
    stencil_i = sorted_idx[stencil_i]
    stencil[sorted_idx,:] = stencil_i

  return stencil                      
