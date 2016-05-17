#!/usr/bin/env python
from __future__ import division
import numpy as np
import scipy.spatial
from rbf.geometry import intersection_count
import networkx
import logging
logger = logging.getLogger(__name__)

def stencil_to_edges(stencil):
  ''' 
  returns an array of edges defined by the stencil
  '''
  N,S = stencil.shape
  node1 = np.arange(N)[:,None].repeat(S,axis=1)
  node2 = np.array(stencil,copy=True)
  edges = zip(node1.flatten(),node2.flatten())
  return edges


def is_connected(stencil):
  ''' 
  returns True if stencil forms a connected graph (i.e. connectivity
  greater than 0)
  '''
  edges = stencil_to_edges(stencil)
  graph = networkx.Graph(edges)
  return networkx.is_connected(graph)


def connectivity(stencil):
  ''' 
  returns the minimum number of edges that must be removed in order to 
  break the connectivity of the graph defined by the stencil
  '''
  edges = stencil_to_edges(stencil)
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
  cc[dist!=0.0] = intersection_count(test[dist!=0.0],
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

  assert N <= population.shape[0], (
    'cannot find %s nearest neighbors with %s points' % (N,population.shape[0]))

  assert N >= 0, (
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
        print('WARNING: could not find %s nearest neighbors for point '
              '%s without crossing a boundary' % (N,population[i]))
        break

  return neighbors,dist


def stencil(nodes,N=None,C=None,vert=None,smp=None):
  ''' 
  returns a stencil of nearest neighbors for each node. The number of 
  nodes in each stencil can be explicitly specified with N or the 
  N can be chosen such that connectivity is at least C.

  Note
  ----
    computing connectivity can be expensive when the number of nodes 
    is greater than about 100. Specify N when dealing with a large
    number of nodes  
  '''
  if (N is not None) & (C is not None):
    raise ValueError('N and C cannot simultaneously be input arguments')
  
  if N is not None:
    s,dx = nearest(nodes,nodes,N,vert=vert,smp=smp)
    return s

  if C is not None:
    N = 2
    s,dx = nearest(nodes,nodes,N,vert=vert,smp=smp)
    while connectivity(s) < C:
      N += 1
      if N > nodes.shape[0]:
        print('WARNING: cannot create a stencil with the desired '
              'connectivity')
        break 
      s,dx = nearest(nodes,nodes,N,vert=vert,smp=smp)

    return s

