#!/usr/bin/env python
from __future__ import division
import numpy as np
import scipy.spatial
import rbf.geometry 
import networkx
import logging
logger = logging.getLogger(__name__)

class StencilError(Exception):
  pass


def stencils_to_edges(stencils):
  ''' 
  returns an array of edges defined by the stencils
  '''
  N,S = stencils.shape
  node1 = np.arange(N)[:,None].repeat(S,axis=1)
  node2 = np.array(stencils,copy=True)
  edges = zip(node1.flatten(),node2.flatten())
  edges = np.array(edges,dtype=int)
  return edges


def is_connected(stencils):
  ''' 
  returns True if stencils forms a connected graph (i.e. connectivity
  greater than 0)
  '''
  edges = stencils_to_edges(stencils)
  # edges needs to be a list of tuples
  edges = [tuple(e) for e in edges] 
  graph = networkx.Graph(edges)
  return networkx.is_connected(graph)


def connectivity(stencils):
  ''' 
  returns the minimum number of edges that must be removed in order to 
  break the connectivity of the graph defined by the stencils
  '''
  edges = stencils_to_edges(stencils)
  # edges needs to be a list of tuples
  edges = [tuple(e) for e in edges] 
  graph = networkx.Graph(edges)
  return networkx.node_connectivity(graph)


def distance(test,pnts,vert=None,smp=None):
  ''' 
  returns euclidean distance between test and pnts. If the line
  segment between test and pnts crosses a boundary then the distance
  is inf
  '''
  if smp is None:
    smp = np.zeros((0,len(test)),dtype=int)

  if vert is None:
    vert = np.zeros((0,len(test)),dtype=float)

  test = np.asarray(test,dtype=float)
  pnts = np.asarray(pnts,dtype=float)
  vert = np.asarray(vert,dtype=float)
  smp = np.asarray(smp,dtype=int)

  test = np.repeat(test[None,:],pnts.shape[0],axis=0)
  dist = np.sqrt(np.sum((pnts-test)**2,1))
  cc = np.zeros(pnts.shape[0],dtype=int)
  cc[dist!=0.0] = rbf.geometry.intersection_count(
                    test[dist!=0.0],
                    pnts[dist!=0.0],
                    vert,smp)
  dist[cc>0] = np.inf
  return dist


def nearest(query,population,N,vert=None,smp=None,excluding=None):
  ''' 
  Description 
  -----------
    Identifies the N points among the population that are closest 
    to each of the query points. If two points form a line segment 
    which intersects any part of the boundary defined by vert and 
    smp then they are considered infinitely far away.  

  Parameters
  ----------
    query: (Q,D) array of query points 
  
    population: (P,D) array of population points 

    N: number of neighbors within the population to find for each 
      query point
 
    vert (default=None): float array of vertices for the boundary 

    smp (default=None): integer array of connectivity for the vertices
      
    excluding (default=None): indices of points in the population 
      which cannot be identified as a nearest neighbor

  Note
  ----
    If a query point lies on the boundary then this function will
    fail because the query point will be infinitely far from every 
    other point
  '''
  query = np.asarray(query,dtype=float)
  population = np.asarray(population,dtype=float)
  
  if excluding is None:
    # dont exclude any points
    excluding_bool = np.zeros(population.shape[0],dtype=bool)

  else:
    # exclude indicated points
    excluding_bool = np.zeros(population.shape[0],dtype=bool)
    excluding_bool[excluding] = True

  if len(query.shape) != 2:
    raise StencilError(
      'query points must be a 2-D array')

  if len(population.shape) != 2:
    raise StencilError(
      'population points must be a 2-D array')

  if N > population.shape[0]: 
    raise StencilError(
      'cannot find %s nearest neighbors with %s points' % (N,population.shape[0]))

  if N < 0:
    raise StencilError(
      'must specify a non-negative number of nearest neighbors')
 
  # querying the KDTree returns a segmentation fault if N is zero and 
  # so this needs to be handles seperately 
  if N == 0:
    dist = np.zeros((query.shape[0],0),dtype=float)
    neighbors = np.zeros((query.shape[0],0),dtype=int)
  else:
    T = scipy.spatial.cKDTree(population)
    dist,neighbors= T.query(query,N)
    if N == 1:
      dist = dist[:,None]
      neighbors = neighbors[:,None]

  if (vert is None) & (excluding is None):
    return neighbors,dist

  for i in range(query.shape[0]):
    # distance from point i to nearest neighbors, crossing
    # a boundary gives infinite distance. If the neighbor 
    # is in the excluding list then it also has infinite 
    # distance
    dist_i = distance(query[i],population[neighbors[i]],vert=vert,smp=smp)
    dist_i[excluding_bool[neighbors[i]]] = np.inf
    
    query_size = N
    while np.any(np.isinf(dist_i)):
      # if some neighbors cross a boundary then query a larger
      # set of nearest neighbors from the KDTree
      query_size += N
      if query_size > population.shape[0]:
        query_size = population.shape[0]
         
      dist_i,neighbors_i = T.query(query[i],query_size)
      # recompute distance to larger set of neighbors
      dist_i = distance(query[i],population[neighbors_i],vert=vert,smp=smp)
      dist_i[excluding_bool[neighbors_i]] = np.inf
      # assign the closest N neighbors to the neighbors array
      neighbors[i] = neighbors_i[np.argsort(dist_i)[:N]]
      dist_i = dist_i[np.argsort(dist_i)[:N]]
      dist[i] = dist_i
      if (query_size == population.shape[0]) & (np.any(np.isinf(dist_i))):
        raise StencilError('cannot find %s nearest neighbors for point '
                           '%s without crossing a boundary' % (N,query[i]))

  return neighbors,dist


def stencil_network(nodes,N=None,C=None,vert=None,smp=None):
  ''' 
  returns a stencil of nearest neighbors for each node. The number of 
  nodes in each stencil can be explicitly specified with N or the 
  N can be chosen such that the connectivity is at least C.

  Parameters
  ----------
    nodes: (N,D) array of nodes

    C (default=None): desired connectivity of the resulting stencils. The 
      stencil size is then chosen so that the connectivity is at least 
      this large. Overrides N if specified
    
    N (default=None): stencil size. Defaults to 10 or the number of 
      nodes, whichever is smaller

    vert (default=None): vertices of the boundary that edges cannot 
      cross

    smp (default=None): connectivity of the vertices

  Note
  ----
    computing connectivity can be expensive when the number of nodes 
    is greater than about 100. Specify N when dealing with a large
    number of nodes  
  '''
  nodes = np.asarray(nodes,dtype=float)

  if C is not None:
    N = 2
    s,dx = nearest(nodes,nodes,N,vert=vert,smp=smp)
    while connectivity(s) < C:
      N += 1
      if N > nodes.shape[0]:
        raise StencilError('cannot create a stencil with the desired '
                           'connectivity')
      s,dx = nearest(nodes,nodes,N,vert=vert,smp=smp)

    return s

  else:
    if N is None:
      N = min(nodes.shape[0],10)

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
    raise StencilError('cannot form a size %s stencil with %s nodes' % (N,P))

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


def stencil_network_1d(nodes,N=None,C=None,vert=None,smp=None):
  ''' 
  returns a stencil network for 1d nodes where each stencil is 
  determined by adjacency and not distance.  For each node, its 
  stencil is comprised of the N//2 nodes to its right and the (N - 
  N//2 - 1) nodes to its left. This ensures better connectivity than 
  what rbf.stencil.stencil_network provides
  
  Parameters
  ----------
    nodes: (N,1) array of nodes

    N: stencil size

    vert: vertices of boundaries

    smp: simplices of boundaries  

  '''
  nodes = np.asarray(nodes)

  if N is None:
    N = min(nodes.shape[0],3)

  if C is not None:
    raise NotImplementedError('specifying connectivity is not yet supported')
  
  if len(nodes.shape) != 2:
    raise ValueError('nodes must be 2-D array')

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

  # for the sake of consistency, sort each stencil in order of 
  # distance from center node. Note that this takes up the most time
  #for i in range(P):
  #  nodes_i = nodes[stencil[i]]
  #  dist_i = np.abs(nodes_i - nodes[i])
  #  sorted_idx = np.argsort(dist_i)
  #  stencil[i,:] = stencil[i,sorted_idx]

  return stencil                      
