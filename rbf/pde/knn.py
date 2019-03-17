''' 
This module is used for k nearest neighbor searches with the
constraint that the segment between neighbors cannot intersect a
boundary. 
'''
from __future__ import division
import numpy as np
from scipy.spatial import cKDTree
from rbf.utils import assert_shape
from rbf.pde.geometry import intersection_count


class KNearestNeighborError(Exception):
  ''' 
  raised when k nearest neighbors cannot be found for topological
  purposes
  '''
  pass


def _closest_argsort(c, x):
  ''' 
  Returns the indices of nodes in `x` sorted in order of distance to
  `c`
  '''
  dist = np.sum((x - c[None, :])**2, axis=1)
  idx = np.argsort(dist)
  return idx


def _intersects_boundary(c, x, vert, smp):
  ''' 
  Check if any of the segments (`c`, `x[i]`) intersect the boundary
  defined by `vert` and `smp`.
  '''
  cext = np.repeat(c[None, :], x.shape[0], axis=0)
  # number of times each segment intersects the boundary
  count = intersection_count(x, cext, vert, smp)
  # return True if there are intersections
  out = np.any(count > 0)
  return out


def _knn_with_boundary(x, p, k, vert, smp):
  ''' 
  Finds the `k` nearest points in `p` for the single point `x`.
  Nearest points are restricted to not cross the boundary defined by
  `vert` and `smp`.
  '''
  sorted_idx = _closest_argsort(x, p)
  indices = []    
  for si in sorted_idx:
    if not _intersects_boundary(x, p[[si]], vert, smp):
      indices += [si]
      if len(indices) == k:
        break

  if len(indices) == k:
    indices = np.array(indices, dtype=int)
    dist = np.linalg.norm(x[None,:] - p[indices], axis=1)
    return indices, dist

  else: 
    raise KNearestNeighborError(
      'Cannot find %s neighbors for the point %s that do not '
      'intersect the boundary' % (k, x))


def _knn_without_boundary(x, p, k):
  ''' 
  Returns the `k` nearest points in `p` for each point in `x`
  '''
  if k == 0:
    dist = np.zeros((x.shape[0], 0), dtype=float)
    indices = np.zeros((x.shape[0], 0), dtype=int)

  else:
    dist, indices = cKDTree(p).query(x, k)
    if k == 1:
      # cKDTree returns a flattened array when k=1. Expand it to be
      # consistent with k>1
      dist = dist[:, None]
      indices = indices[:, None]

  return indices, dist


def k_nearest_neighbors(x, p, k, vert=None, smp=None):
  ''' 
  Finds the `k` nearest points in `p` for each point in `x`. If a
  boundary is specified, with `vert` and `smp`, then points that are
  separated by this boundary will not be identified as neighbors.
  
  Parameters
  ----------
  x : (N, D) array
    Query points

  p : (M, D) array
    Source points 

  k : int
    Number of nearest neighbors to find

  vert : (P, D) array, optional
    Vertices of the boundary

  smp : (Q, D) array, optional
    Connectivity of the vertices to form the boundary

  Returns
  -------
  (N, k) int array
    The indices of the nearest points in `p` 

  (N, k) float array        
    This distance to the nearest points in `p`
    
  '''
  x = np.asarray(x, dtype=float)
  p = np.asarray(p, dtype=float)
  assert_shape(x, (None, None), 'x')
  dim = x.shape[1]
  assert_shape(p, (None, dim), 'p')
  
  if k > p.shape[0]:
    raise KNearestNeighborError(
      'Cannot find %s neighbors from %s points' % 
      (k, p.shape[0]))
    
  indices, dist = _knn_without_boundary(x, p, k)
  if (vert is not None) & (smp is not None):
    # Check if neighbors intersect the boundary. If they do, then fix
    # it
    vert = np.asarray(vert, dtype=float)
    smp = np.asarray(smp, dtype=int)
    assert_shape(vert, (None, dim), 'vert')
    assert_shape(smp, (None, dim), 'smp')
    for i, xi in enumerate(x):
      if _intersects_boundary(xi, p[indices[i]], vert, smp):
        indices_i, dist_i = _knn_with_boundary(xi, p, k, vert, smp)
        indices[i, :] = indices_i
        dist[i, :] = dist_i

  return indices, dist
