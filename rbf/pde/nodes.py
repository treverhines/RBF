'''
This module provides functions for generating nodes used for solving
PDEs with the RBF and RBF-FD method.
'''
from __future__ import division
import logging

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from rbf.utils import assert_shape
from rbf.pde.knn import k_nearest_neighbors
from rbf.pde.sampling import rejection_sampling, poisson_discs
from rbf.pde.geometry import (intersection,
                              intersection_count,
                              simplex_outward_normals,
                              simplex_normals,
                              nearest_point)

logger = logging.getLogger(__name__)


def _disperse(nodes,
              rho=None,
              fixed_nodes=None,
              neighbors=None,
              delta=0.1,
              vert=None,
              smp=None):
  '''
  Returns the new position of the free nodes after a dispersal step.
  Nodes on opposite sides of the boundary defined by `vert` and `smp`
  cannot repel eachother. This does not handle node intersections with
  the boundary
  '''
  if rho is None:
    def rho(x): 
        return np.ones(x.shape[0])

  if fixed_nodes is None:
    fixed_nodes = np.zeros((0, nodes.shape[1]), dtype=float)
  else:
    fixed_nodes = np.asarray(fixed_nodes)
    assert_shape(fixed_nodes, (None, nodes.shape[1]), 'fixed_nodes')     
    
  if neighbors is None:
    # the default number of neighboring nodes to use when computing
    # the repulsion force is 7 for 2D and 13 for 3D
    if nodes.shape[1] == 2:
      neighbors = 7

    elif nodes.shape[1] == 3:
      neighbors = 13

  # ensure that the number of nodes used to determine repulsion force
  # is less than or equal to the total number of nodes
  neighbors = min(neighbors, nodes.shape[0] + fixed_nodes.shape[0])
  # if m is 0 or 1 then the nodes remain stationary
  if neighbors <= 1:
    return np.array(nodes, copy=True)

  # form collection of all nodes
  all_nodes = np.vstack((nodes, fixed_nodes))
  # find index and distance to nearest nodes
  i, d = k_nearest_neighbors(nodes, all_nodes, neighbors, 
                             vert=vert, smp=smp)
  # dont consider a node to be one of its own nearest neighbors
  i, d = i[:, 1:], d[:, 1:]
  # compute the force proportionality constant between each node
  # based on their charges
  c = 1.0/(rho(all_nodes)[i, None]*rho(nodes)[:, None, None])
  # calculate forces on each node resulting from the `m` nearest
  # nodes.
  forces = c*(nodes[:, None, :] - all_nodes[i, :])/d[:, :, None]**3
  # sum up all the forces for each node
  direction = np.sum(forces, axis=1)
  # normalize the net forces to one
  direction /= np.linalg.norm(direction, axis=1)[:, None]
  # in the case of a zero vector replace nans with zeros
  direction = np.nan_to_num(direction)
  # move in the direction of the force by an amount proportional to
  # the distance to the nearest neighbor
  step = delta*d[:, 0, None]*direction
  # new node positions
  out = nodes + step
  return out


def disperse(nodes,
             vert,
             smp,
             rho=None,
             fixed_nodes=None,
             neighbors=None,
             delta=0.1,
             bound_force=False):
  '''
  Slightly disperses the nodes within the domain defined by `vert` and
  `smp`. The disperson is analogous to electrostatic repulsion, where
  neighboring node exert a repulsive force on eachother. If a node is
  repelled into a boundary then it bounces back in. 

  Parameters
  ----------
  nodes : (n, d) float array
    Initial node positions

  vert : (p, d) float array
    Domain vertices
  
  smp : (q, d) int array
    Connectivity of the vertices to form the boundary
  
  rho : callable, optional
    Takes an (n, d) array as input and returns the repulsion force for
    a node at those position. 

  fixed_nodes : (k, d) float array, optional
    Nodes which do not move and only provide a repulsion force

  neighbors : int, optional
    The number of adjacent nodes used to determine repulsion forces
    for each node
                 
  delta : float, optional
    The step size. Each node moves in the direction of the repulsion
    force by a distance `delta` times the distance to the nearest
    neighbor.

  bound_force : bool, optional
    If True then nodes cannot repel eachother across the domain
    boundaries.

  Returns
  -------
  (n, d) float array
  
  '''
  nodes = np.asarray(nodes, dtype=float)
  vert = np.asarray(vert, dtype=float)
  smp = np.asarray(smp, dtype=int) 
  assert_shape(nodes, (None, None), 'nodes')
  dim = nodes.shape[1]
  assert_shape(vert, (None, dim), 'vert')
  assert_shape(smp, (None, dim), 'smp')
  
  if bound_force:
    bound_vert, bound_smp = vert, smp
  else:
    bound_vert, bound_smp = None, None

  # node positions after repulsion
  out = _disperse(nodes, rho=rho, fixed_nodes=fixed_nodes, 
                  neighbors=neighbors, delta=delta, vert=bound_vert, 
                  smp=bound_smp)
  # boolean array of nodes which are now outside the domain
  crossed = intersection_count(nodes, out, vert, smp) > 0
  # point where nodes intersected the boundary and the simplex they
  # intersected at
  intr_pnt, intr_idx = intersection(nodes[crossed], out[crossed], 
                                    vert, smp)
  # normal vector to intersection point
  intr_norms = simplex_normals(vert, smp[intr_idx])
  # distance that the node wanted to travel beyond the boundary
  res = out[crossed] - intr_pnt
  # bouce node off the boundary
  out[crossed] -= 2*intr_norms*np.sum(res*intr_norms, 1)[:, None]
  # check to see if the bounced nodes still intersect the boundary. If
  # not then set the bounced nodes back to their original position
  crossed = intersection_count(nodes, out, vert, smp) > 0
  out[crossed] = nodes[crossed]
  return out


def snap_to_boundary(nodes, vert, smp, delta=0.5):
  '''
  Snaps `nodes` to the boundary defined by `vert` and `smp`. If a node
  is sufficiently close to the boundary, then it will be snapped to
  the closest point on the boundary. A node is sufficiently close if
  the distance to the boundary is `delta` times the distance to its
  nearest neighbor.

  Parameters
  ----------
  nodes : (n, d) float array
    Node positions
  
  vert : (p, d) float array
    Domain vertices
  
  smp : (q, d) int array
    Connectivity of the vertices to form the domain boundary
  
  delta : float, optional
    Snapping distance factor. The snapping distance is `delta` times
    the distance to the nearest neighbor.
      
  Returns
  -------
  (n, d) float array
    Node poistion    

  (n, d) int array
    Index of the simplex that each node snapped to. If a node did not
    snap to the boundary then its value will be -1.

  '''
  nodes = np.asarray(nodes, dtype=float)
  vert = np.asarray(vert, dtype=float)
  smp = np.asarray(smp, dtype=int) 
  assert_shape(nodes, (None, None), 'nodes')
  n, dim = nodes.shape
  assert_shape(vert, (None, dim), 'vert')
  assert_shape(smp, (None, dim), 'smp')

  # find the distance to the nearest node
  dist = k_nearest_neighbors(nodes, nodes, 2)[1][:, 1]
  nrst_pnt, nrst_smpid = nearest_point(nodes, vert, smp)
  snap = np.linalg.norm(nrst_pnt - nodes, axis=1) < dist*delta
  out_smpid = np.full(n, -1, dtype=int)
  out_nodes = np.array(nodes, copy=True)
  out_nodes[snap] = nrst_pnt[snap]
  out_smpid[snap] = nrst_smpid[snap]
  return out_nodes, out_smpid


def neighbor_argsort(nodes, m=None):
  '''
  Returns a permutation array that sorts `nodes` so that each node and
  its `m` nearest neighbors are close together in memory. This is done
  through the use of a KD Tree and the Reverse Cuthill-McKee
  algorithm.

  Parameters
  ----------
  nodes : (n, d) float array
  
  m : int, optional
         
  Returns
  -------
  (N,) int array

  Examples
  --------
  >>> nodes = np.array([[0.0, 1.0],
                        [2.0, 1.0],
                        [1.0, 1.0]])
  >>> idx = neighbor_argsort(nodes, 2)
  >>> nodes[idx]
  array([[ 2.,  1.],
         [ 1.,  1.],
         [ 0.,  1.]])

  '''
  nodes = np.asarray(nodes, dtype=float)
  assert_shape(nodes, (None, None), 'nodes')
  
  if m is None:
    # this should be roughly equal to the stencil size for the RBF-FD
    # problem
    m = 5**nodes.shape[1]

  m = min(m, nodes.shape[0])
  # find the indices of the nearest m nodes for each node
  idx = k_nearest_neighbors(nodes, nodes, m)[0]
  # efficiently form adjacency matrix
  col = idx.ravel()
  row = np.repeat(np.arange(nodes.shape[0]), m)
  data = np.ones(nodes.shape[0]*m, dtype=bool)
  mat = csc_matrix((data, (row, col)), dtype=bool)
  permutation = reverse_cuthill_mckee(mat)
  return permutation


def _check_spacing(nodes, rho=None):
  '''
  Check if any nodes are unusually close to eachother. If so, a
  warning will be printed.
  '''
  n, dim = nodes.shape

  if rho is None:
    def rho(x):
        return np.ones(x.shape[0])

  # distance to nearest neighbor
  dist = k_nearest_neighbors(nodes, nodes, 2)[1][:, 1]
  if np.any(dist == 0.0):
    is_zero = (dist == 0.0)
    indices, = is_zero.nonzero()
    for idx in indices:
      logger.warning(
        'Node %s (%s) is in the same location as another node.' 
        % (idx, nodes[idx]))
    
  density = 1.0/dist**dim
  normalized_density = np.log10(density / rho(nodes))
  percs = np.percentile(normalized_density, [10, 50, 90])
  med = percs[1]
  idr = percs[2] - percs[0]
  is_too_close = normalized_density < (med - 2*idr)
  if np.any(is_too_close):
    indices, = is_too_close.nonzero()
    for idx in indices:
      logger.warning(
        'Node %s (%s) is unusually close to a neighboring '
        'node.' % (idx, nodes[idx]))


def prepare_nodes(nodes, vert, smp,
                  rho=None,
                  iterations=100,
                  neighbors=None,
                  dispersion_delta=0.05,
                  bound_force=False,
                  pinned_nodes=None,
                  snap_delta=0.5,
                  boundary_groups=None,
                  boundary_groups_with_ghosts=None,
                  include_vertices=False):
  '''
  Prepares a set of nodes for solving PDEs with the RBF and RBF-FD
  method. This includes: dispersing the nodes away from eachother to
  ensure a more even spacing, snapping nodes to the boundary,
  determining the normal vectors for each node, determining the group
  that each node belongs to, creating ghost nodes, sorting the nodes
  so that adjacent nodes are close in memory, and verifying that no
  two nodes are anomalously close to eachother.

  The function returns a set of nodes, the normal vectors for each
  node, and a dictionary identifying which group each node belongs to.

  Parameters
  ----------
  nodes : (n, d) float arrary
    An initial sampling of nodes within the domain

  vert : (p, d) float array
    Vertices making up the domain boundary

  smp : (q, d) array
    Describes how the vertices are connected to form the boundary

  rho : function, optional 
    Node density function. Takes a (n, d) array of coordinates and
    returns an (n,) array of desired node densities at those
    coordinates. This is used during the node dispersion step.

  iterations : int, optional
    Number of dispersion iterations.

  neighbors : int, optional
    Number of neighboring nodes to use when calculating the repulsion
    force. This defaults to 7 for 2D nodes and 13 for 3D nodes.
    Deviating from these default values may yield a node distribution
    that is not consistent with the node density function `rho`.

  dispersion_delta : float, optional
    Scaling factor for the node step size in each iteration. The step
    size is equal to `dispersion_delta` times the distance to the
    nearest neighbor.

  bound_force : bool, optional
    If `True`, then nodes cannot repel other nodes through the domain
    boundary. Set this to `True` if the domain has edges that nearly
    touch eachother. Setting this to `True` may significantly increase
    computation time.

  pinned_nodes : (k, d) array, optional
    Nodes which do not move and only provide a repulsion force. These
    nodes are included in the set of nodes returned by this function
    and they are in the group named "pinned".

  snap_delta : float, optional
    Controls the maximum snapping distance. The maximum snapping
    distance for each node is `snap_delta` times the distance to the
    nearest neighbor. This defaults to 0.5.

  boundary_groups: dict, optional 
    Dictionary defining the boundary groups. The keys are the names of
    the groups and the values are lists of simplex indices making up
    each group. This function will return a dictionary identifying
    which nodes belong to each boundary group. By default, there is a
    single group named 'all' for the entire boundary. Specifically,
    The default value is `{'all':range(len(smp))}`.

  boundary_groups_with_ghosts: list of strs, optional
    List of boundary groups that will be given ghost nodes. By
    default, no boundary groups are given ghost nodes. The groups
    specified here must exist in `boundary_groups`.

  include_vertices : bool, optional
    If `True`, then the vertices will be included in the output nodes.
    Each vertex will be assigned to the boundary group that its
    adjoining simplices are part of. If the simplices are in multiple
    groups, then the vertex will be assigned to the group containing
    the simplex that comes first in `smp`.

  Returns
  -------
  (m, d) float array
    Nodes positions

  dict 
    The indices of nodes belonging to each group. There will always be
    a group called 'interior' containing the nodes that are not on the
    boundary. By default there is a group containing all the boundary
    nodes called 'boundary:all'. If `boundary_groups` was specified,
    then those groups will be included in this dictionary and their
    names will be given a 'boundary:' prefix. If
    `boundary_groups_with_ghosts` was specified then those groups of
    ghost nodes will be included in this dictionary and their names
    will be given a 'ghosts:' prefix.
    
  (n, d) float array
    Outward normal vectors for each node. If a node is not on the
    boundary then its corresponding row will contain NaNs.

  '''
  nodes = np.asarray(nodes, dtype=float)
  vert = np.asarray(vert, dtype=float)
  smp = np.asarray(smp, dtype=int)
  assert_shape(nodes, (None, None), 'nodes')
  dim = nodes.shape[1]
  assert_shape(vert, (None, dim), 'vert')
  assert_shape(smp, (None, dim), 'smp')    

  # the `fixed_nodes` are used to provide a repulsion force during
  # dispersion, but they do not move. TODO There is chance that one of
  # the points in `fixed_nodes` is equal to a point in `nodes`. This
  # situation should be handled
  fixed_nodes = np.zeros((0, dim), dtype=float)     
  if pinned_nodes is not None:
    pinned_nodes = np.asarray(pinned_nodes, dtype=float)
    assert_shape(pinned_nodes, (None, dim), 'pinned_nodes')
    fixed_nodes = np.vstack((fixed_nodes, pinned_nodes))

  if include_vertices:
    fixed_nodes = np.vstack((fixed_nodes, vert))

  for i in range(iterations):
    logger.debug('starting node dispersion iterations %s of %s' 
                 % (i + 1, iterations))
    nodes = disperse(nodes, vert, smp, 
                     rho=rho, 
                     fixed_nodes=fixed_nodes, 
                     neighbors=neighbors, 
                     delta=dispersion_delta, 
                     bound_force=bound_force)

  # append the domain vertices to the collection of nodes if requested
  if include_vertices:
    nodes = np.vstack((nodes, vert))
    
  # snap nodes to the boundary, identifying which simplex each node
  # was snapped to
  nodes, smpid = snap_to_boundary(nodes, vert, smp, delta=snap_delta)

  # find the normal vectors for each node that snapped to the boundary
  smp_normals = simplex_outward_normals(vert, smp)
  normals = np.full_like(nodes, np.nan)
  normals[smpid >= 0] = smp_normals[smpid[smpid >= 0]]
  
  # create a dictionary identifying which nodes belong to which group
  groups = {}
  groups['interior'], = (smpid == -1).nonzero()

  # append the user specified pinned nodes
  if pinned_nodes is not None:
    pinned_idx = np.arange(pinned_nodes.shape[0]) + nodes.shape[0]
    pinned_normals = np.full_like(pinned_nodes, np.nan)
    nodes = np.vstack((nodes, pinned_nodes))
    normals = np.vstack((normals, pinned_normals))
    groups['pinned'] = pinned_idx
    
  if boundary_groups is None:
    boundary_groups = {'all': range(smp.shape[0])}

  if boundary_groups_with_ghosts is None:
    boundary_groups_with_ghosts = []    

  # create groups for the boundary nodes
  for k, v in boundary_groups.items():
    bnd_idx = np.array([i for i, j in enumerate(smpid) if j in v])
    groups['boundary:' + k] = bnd_idx
    if k in boundary_groups_with_ghosts:
      # append ghost nodes if requested
      dist = k_nearest_neighbors(nodes[bnd_idx], nodes, 2)[1][:, [1]]
      ghost_idx = np.arange(bnd_idx.shape[0]) + nodes.shape[0]         
      ghost_nodes = nodes[bnd_idx] + 0.5*dist*normals[bnd_idx]
      ghost_normals = np.full_like(ghost_nodes, np.nan)
      nodes = np.vstack((nodes, ghost_nodes))
      normals = np.vstack((normals, ghost_normals))
      groups['ghosts:' + k] = ghost_idx
  
  # sort `nodes` so that spatially adjacent nodes are close together
  sort_idx = neighbor_argsort(nodes)
  nodes = nodes[sort_idx]
  normals = normals[sort_idx]
  reverse_sort_idx = np.argsort(sort_idx)
  groups = {k: reverse_sort_idx[v] for k, v in groups.items()}

  _check_spacing(nodes, rho)

  return nodes, groups, normals

  
def min_energy_nodes(n, vert, smp, rho=None, **kwargs):
  '''
  Generates nodes within a two or three dimensional. This first
  generates nodes with a rejection sampling algorithm, and then the
  nodes are dispersed to ensure a more even distribution.

  Parameters
  ----------
  n : int
    The number of nodes generated during rejection sampling. This is
    not necessarily equal to the number of nodes returned.

  vert : (p, d) array
    Vertices making up the boundary

  smp : (q, d) array
    Describes how the vertices are connected to form the boundary

  rho : function, optional
    Node density function. Takes a (n, d) array of coordinates and
    returns an (n,) array of desired node densities at those
    coordinates. This function should be normalized to be between 0
    and 1.

  **kwargs
    Additional arguments passed to `prepare_nodes`    

  Returns
  -------
  (n, d) float array
    Nodes positions

  dict 
    The indices of nodes belonging to each group. There will always be
    a group called 'interior' containing the nodes that are not on the
    boundary. By default there is a group containing all the boundary
    nodes called 'boundary:all'. If `boundary_groups` was specified,
    then those groups will be included in this dictionary and their
    names will be given a 'boundary:' prefix. If
    `boundary_groups_with_ghosts` was specified then those groups of
    ghost nodes will be included in this dictionary and their names
    will be given a 'ghosts:' prefix.
    
  (n, d) float array
    Outward normal vectors for each node. If a node is not on the
    boundary then its corresponding row will contain NaNs.

  Notes
  -----
  It is assumed that `vert` and `smp` define a closed domain. If this
  is not the case, then it is likely that an error message will be
  raised which says "ValueError: No intersection found for segment

  Examples
  --------
  make 9 nodes within the unit square   

  >>> vert = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  
  >>> smp = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
  >>> out = min_energy_nodes(9, vert, smp)  

  view the nodes

  >>> out[0]  
  array([[ 0.50325675,  0.        ],
         [ 0.00605261,  1.        ],
         [ 1.        ,  0.51585247],
         [ 0.        ,  0.00956821],
         [ 1.        ,  0.99597894],
         [ 0.        ,  0.5026365 ],
         [ 1.        ,  0.00951112],
         [ 0.48867638,  1.        ],
         [ 0.54063894,  0.47960892]])

  view the indices of nodes making each group

  >>> out[1] 
  {'boundary:all': array([7, 6, 5, 4, 3, 2, 1, 0]),
   'interior': array([8])}

  view the outward normal vectors for each node, note that the normal
  vector for the interior node is `nan`

  >>> out[2] 
  array([[  0.,  -1.],
         [  0.,   1.],
         [  1.,  -0.],
         [ -1.,  -0.],
         [  1.,  -0.],
         [ -1.,  -0.],
         [  1.,  -0.],
         [  0.,   1.],
         [ nan,  nan]])
    
  '''
  vert = np.asarray(vert, dtype=float)
  assert_shape(vert, (None, None), 'vert')
  dim = vert.shape[1]
  smp = np.asarray(smp, dtype=int)
  assert_shape(smp, (None, dim), 'smp')
  
  if rho is None:
    def rho(x): 
        return np.ones(x.shape[0])

  nodes = rejection_sampling(n, rho, vert, smp)
  out = prepare_nodes(nodes, vert, smp, rho=rho, **kwargs)
  return out                      


def poisson_disc_nodes(radius, vert, smp, **kwargs):
  '''
  Generates nodes within a two or three dimensional domain. This first
  generate nodes with Poisson disc sampling, and then the nodes are
  dispersed to ensure a more even distribution. This function is
  considerably slower than `min_energy_nodes` but it has the advantage
  of directly specifying the node spacing.

  Parameters
  ----------
  radius : float or callable
    The radius for each disc. This is the minimum allowable distance
    between the nodes generated by Poisson disc sampling. This can be
    a float or a function that takes a (n, d) array of locations and
    returns an (n,) array of disc radii.

  vert : (p, d) array
    Vertices making up the boundary

  smp : (q, d) array
    Describes how the vertices are connected to form the boundary

  **kwargs
    Additional arguments passed to `prepare_nodes`    

  Returns
  -------
  (n, d) float array
    Nodes positions

  dict 
    The indices of nodes belonging to each group. There will always be
    a group called 'interior' containing the nodes that are not on the
    boundary. By default there is a group containing all the boundary
    nodes called 'boundary:all'. If `boundary_groups` was specified,
    then those groups will be included in this dictionary and their
    names will be given a 'boundary:' prefix. If
    `boundary_groups_with_ghosts` was specified then those groups of
    ghost nodes will be included in this dictionary and their names
    will be given a 'ghosts:' prefix.
    
  (n, d) float array
    Outward normal vectors for each node. If a node is not on the
    boundary then its corresponding row will contain NaNs.

  Notes
  -----
  It is assumed that `vert` and `smp` define a closed domain. If this
  is not the case, then it is likely that an error message will be
  raised which says "ValueError: No intersection found for segment
    
  '''
  vert = np.asarray(vert, dtype=float)
  assert_shape(vert, (None, None), 'vert')
  dim = vert.shape[1]
  smp = np.asarray(smp, dtype=int)
  assert_shape(smp, (None, dim), 'smp')
  
  if np.isscalar(radius):
    scalar_radius = radius
    def radius(x): 
        return np.full(x.shape[0], scalar_radius)

  def rho(x):
    # the density function corresponding to the radius function
    return 1.0/(radius(x)**dim)
        
  nodes = poisson_discs(radius, vert, smp)
  out = prepare_nodes(nodes, vert, smp, rho=rho, **kwargs)
  return out
