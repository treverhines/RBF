#!/usr/bin/env python
from __future__ import division
import numpy as np
import scipy.spatial
from rbf.geometry import boundary_cross_count
import logging
logger = logging.getLogger(__name__)


def distance(test,pnts,vert,smp):
  '''
  returns euclidean distance between test and pnts. If the line
  segment between test and pnts crosses a boundary then the distance
  is inf

  '''  
  test = np.asarray(test,dtype=float)
  pnts = np.asarray(pnts,dtype=float)
  vert = np.asarray(vert,dtype=float)
  smp = np.asarray(smp,dtype=int)
  test = np.repeat(test[None,:],pnts.shape[0],axis=0)
  dist = np.sqrt(np.sum((pnts-test)**2,1))
  cc = np.zeros(pnts.shape[0],dtype=int)
  cc[dist!=0.0] = boundary_cross_count(test[dist!=0.0],
                                       pnts[dist!=0.0],
                                       vert,smp)
  dist[cc>0] = np.inf
  return dist


def nearest(test,pnts,N,vert=None,smp=None):
  '''
  returns the N points within pnts that are closest to the given
  test points.  If vert and smp are specified then two points whos
  line segment intersects any of the simplexes are considered to
  be infinitely far away
  '''
  test = np.asarray(test,dtype=float)
  pnts = np.asarray(pnts,dtype=float)

  assert N <= pnts.shape[0], (
    'cannot find %s nearest neighbors with %s points' % (N,pnts.shape[0]))

  assert N >= 1, (
    'must specify a non-negative non-zero number of nearest neighbors')
 
  T = scipy.spatial.cKDTree(pnts)
  dist,neighbors= T.query(test,N)
  if N == 1:
    dist = dist[:,None]
    neighbors = neighbors[:,None]

  if vert is None:
    return neighbors,dist

  vert = np.asarray(vert,dtype=float)
  smp = np.asarray(smp,dtype=int)
  for i in range(test.shape[0]):
    # distance from point i to nearest neighbors, crossing
    # a boundary gives infinite distance
    dist_i = distance(test[i],pnts[neighbors[i]],vert,smp)
    query_size = N
    while np.any(np.isinf(dist_i)):
      # if some neighbors cross a boundary then query a larger
      # set of nearest neighbors from the KDTree
      query_size += N
      if query_size > pnts.shape[0]:
        query_size = pnts.shape[0]
         
      dist_i,neighbors_i = T.query(test[i],query_size)
      # recompute distance to larger set of neighbors
      dist_i = distance(test[i],pnts[neighbors_i],vert,smp)
      # assign the closest N neighbors to the neighbors array
      neighbors[i] = neighbors_i[np.argsort(dist_i)[:N]]
      dist_i = dist_i[np.argsort(dist_i)[:N]]
      dist[i] = dist_i
      if (query_size == pnts.shape[0]) & (np.any(np.isinf(dist_i))):
        print('WARNING: could not find %s nearest neighbors for point '
              '%s without crossing a boundary' % (N,pnts[i]))
        logger.warning('could not find %s nearest neighbors for point '
                       '%s without crossing a boundary' % (N,pnts[i]))
        break

  return neighbors,dist
