'''
This module provides functions for generating nodes used for solving
PDEs with the RBF and RBF-FD method.
'''
from __future__ import division
import logging

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from rbf.utils import assert_shape, KDTree
from rbf.pde.domain import as_domain
from rbf.pde.sampling import rejection_sampling, poisson_discs


logger = logging.getLogger(__name__)


def _disperse(nodes, rho, fixed_nodes, neighbors, delta):
  '''
  Returns the new position of the free nodes after a dispersal step.
  This does not handle node intersections with the boundary.
  '''
  # form collection of all nodes
  all_nodes = np.vstack((nodes, fixed_nodes))
  # find index and distance to nearest nodes
  d, i = KDTree(all_nodes).query(nodes, neighbors)
  # dont consider a node to be one of its own nearest neighbors
  d, i = d[:, 1:], i[:, 1:]
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
             domain,
             rho=None,
             fixed_nodes=None,
             neighbors=None,
             delta=0.1):
  '''
  Slightly disperses the nodes within the domain. The disperson is
  analogous to electrostatic repulsion, where neighboring node exert a
  repulsive force on eachother. If a node is repelled into a boundary
  then it bounces back in.

  Parameters
  ----------
  nodes : (n, d) float array
    Initial node positions

  domain : (p, d) float array and (q, d) int array
    Vertices of the domain and connectivity of the vertices.
  
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

  Returns
  -------
  (n, d) float array
  
  '''
  domain = as_domain(domain)
  nodes = np.asarray(nodes, dtype=float)
  assert_shape(nodes, (None, domain.dim), 'nodes')

  if rho is None:
    def rho(x): 
      return np.ones(x.shape[0])

  if fixed_nodes is None:
    fixed_nodes = np.zeros((0, domain.dim), dtype=float)
  else:
    fixed_nodes = np.asarray(fixed_nodes)
    assert_shape(fixed_nodes, (None, domain.dim), 'fixed_nodes')

  if neighbors is None:
    # the default number of neighboring nodes to use when computing
    # the repulsion force is 4 for 2D and 5 for 3D
    if domain.dim == 2:
      neighbors = 4

    elif domain.dim == 3:
      neighbors = 5

  # ensure that the number of neighboring nodes used for the repulsion
  # force is less than or equal to the total number of nodes
  neighbors = min(neighbors, nodes.shape[0] + fixed_nodes.shape[0])
  # if m is 0 or 1 then the nodes remain stationary
  if neighbors <= 1:
    return np.array(nodes, copy=True)

  # node positions after repulsion
  out = _disperse(nodes, rho, fixed_nodes, neighbors, delta)
  # indices of nodes which are now outside the domain
  crossed = domain.intersection_count(nodes, out) > 0
  crossed, = crossed.nonzero()
  # points where nodes intersected the boundary and the simplex they
  # intersected at
  intr_pnt, intr_idx = domain.intersection_point(nodes[crossed], 
                                                 out[crossed])
  # normal vector to intersection points
  intr_norms = domain.normals[intr_idx]
  # distance that the node wanted to travel beyond the boundary
  res = out[crossed] - intr_pnt
  # bounce node off the boundary
  out[crossed] -= 2*intr_norms*np.sum(res*intr_norms, 1)[:, None]
  # check to see if the bounced nodes still intersect the boundary. If
  # they do, then set them back to their original position
  still_crossed = domain.intersection_count(nodes[crossed], 
                                            out[crossed]) > 0
  out[crossed[still_crossed]] = nodes[crossed[still_crossed]]
  return out


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
  _, idx = KDTree(nodes).query(nodes, m)
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
  dist = KDTree(nodes).query(nodes, 2)[0][:, 1]
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


def prepare_nodes(nodes, domain,
                  rho=None,
                  iterations=20,
                  neighbors=None,
                  dispersion_delta=0.1,
                  pinned_nodes=None,
                  snap_delta=0.5,
                  boundary_groups=None,
                  boundary_groups_with_ghosts=None,
                  include_vertices=False,
                  orient_simplices=True):
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

  domain : (p, d) float array and (q, d) int array
    Vertices of the domain and connectivity of the vertices

  rho : function, optional 
    Node density function. Takes a (n, d) array of coordinates and
    returns an (n,) array of desired node densities at those
    coordinates. This is used during the node dispersion step.

  iterations : int, optional
    Number of dispersion iterations.

  neighbors : int, optional
    Number of neighboring nodes to use when calculating the repulsion
    force. This defaults to 4 for 2D nodes and 5 for 3D nodes.

  dispersion_delta : float, optional
    Scaling factor for the node step size in each iteration. The step
    size is equal to `dispersion_delta` times the distance to the
    nearest neighbor.

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

  orient_simplices : bool, optional
    If `False` then it is assumed that the simplices are already
    oriented such that their normal vectors point outward.
        
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
  domain = as_domain(domain)
  nodes = np.asarray(nodes, dtype=float)
  assert_shape(nodes, (None, domain.dim), 'nodes')

  # the `fixed_nodes` are used to provide a repulsion force during
  # dispersion, but they do not move. TODO There is chance that one of
  # the points in `fixed_nodes` is equal to a point in `nodes`. This
  # situation should be handled
  fixed_nodes = np.zeros((0, domain.dim), dtype=float)     
  if pinned_nodes is not None:
    pinned_nodes = np.asarray(pinned_nodes, dtype=float)
    assert_shape(pinned_nodes, (None, domain.dim), 'pinned_nodes')
    fixed_nodes = np.vstack((fixed_nodes, pinned_nodes))

  if include_vertices:
    fixed_nodes = np.vstack((fixed_nodes, domain.vertices))

  for i in range(iterations):
    logger.debug('starting node dispersion iterations %s of %s' 
                 % (i + 1, iterations))
    nodes = disperse(nodes, domain,
                     rho=rho, 
                     fixed_nodes=fixed_nodes, 
                     neighbors=neighbors, 
                     delta=dispersion_delta)

  # append the domain vertices to the collection of nodes if requested
  if include_vertices:
    nodes = np.vstack((nodes, domain.vertices))
    
  # snap nodes to the boundary, identifying which simplex each node
  # was snapped to
  logger.debug('snapping nodes to boundary ...')
  nodes, smpid = domain.snap(nodes, delta=snap_delta)
  logger.debug('done')

  # get the normal vectors for the boundary nodes    
  if orient_simplices:
    logger.debug('orienting simplices ...')
    domain.orient_simplices()
    logger.debug('done')
    
  normals = np.full_like(nodes, np.nan)
  normals[smpid >= 0] = domain.normals[smpid[smpid >= 0]]
  
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
    boundary_groups = {'all': range(len(domain.simplices))}

  if boundary_groups_with_ghosts is None:
    boundary_groups_with_ghosts = []    

  # create groups for the boundary nodes
  logger.debug('grouping boundary nodes and generating ghosts ...')
  for k, v in boundary_groups.items():
    # convert the list of simplices in the boundary group to a set,
    # because it is much faster to determine membership of a set
    v = set(v)
    bnd_idx = np.array([i for i, j in enumerate(smpid) if j in v])
    groups['boundary:' + k] = bnd_idx
    if k in boundary_groups_with_ghosts:
      # append ghost nodes if requested
      dist = KDTree(nodes).query(nodes[bnd_idx], 2)[0][:, [1]]
      ghost_idx = np.arange(bnd_idx.shape[0]) + nodes.shape[0]         
      ghost_nodes = nodes[bnd_idx] + 0.5*dist*normals[bnd_idx]
      ghost_normals = np.full_like(ghost_nodes, np.nan)
      nodes = np.vstack((nodes, ghost_nodes))
      normals = np.vstack((normals, ghost_normals))
      groups['ghosts:' + k] = ghost_idx
  
  logger.debug('done')
  
  # sort `nodes` so that spatially adjacent nodes are close together
  logger.debug('sorting nodes ...')
  sort_idx = neighbor_argsort(nodes)
  logger.debug('done')
  nodes = nodes[sort_idx]
  normals = normals[sort_idx]
  reverse_sort_idx = np.argsort(sort_idx)
  groups = {k: reverse_sort_idx[v] for k, v in groups.items()}

  logger.debug('checking the quality of the generated nodes ...')
  _check_spacing(nodes, rho)
  logger.debug('done')

  return nodes, groups, normals

  
def min_energy_nodes(n, domain, rho=None, build_rtree=False, 
                     **kwargs):
  '''
  Generates nodes within a two or three dimensional. This first
  generates nodes with a rejection sampling algorithm, and then the
  nodes are dispersed to ensure a more even distribution.

  Parameters
  ----------
  n : int
    The number of nodes generated during rejection sampling. This is
    not necessarily equal to the number of nodes returned.

  domain : (p, d) float array and (q, d) int array
    Vertices of the domain and connectivity of the vertices

  rho : function, optional
    Node density function. Takes a (n, d) array of coordinates and
    returns an (n,) array of desired node densities at those
    coordinates. This function should be normalized to be between 0
    and 1.

  build_rtree : bool, optional
    If `True`, then an R-tree will be built to speed up computational
    geometry operations. This should be set to `True` if there are
    many (>10,000) simplices making up the domain.

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
  domain = as_domain(domain)
  if build_rtree:
    logger.debug('building R-tree ...')
    domain.build_rtree()
    logger.debug('done')
    
  if rho is None:
    def rho(x): 
        return np.ones(x.shape[0])

  nodes = rejection_sampling(n, rho, domain)
  out = prepare_nodes(nodes, domain, rho=rho, **kwargs)
  return out                      


def poisson_disc_nodes(radius, domain, ntests=50, rmax_factor=1.5, 
                       build_rtree=False, **kwargs):
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

  domain : (p, d) float array and (q, d) int array
    Vertices of the domain and connectivity of the vertices

  build_rtree : bool, optional
    If `True`, then an R-tree will be built to speed up computational
    geometry operations. This should be set to `True` if there are
    many (>10,000) simplices making up the domain

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
  domain = as_domain(domain)
  if build_rtree:
    logger.debug('building R-tree ...')
    domain.build_rtree()
    logger.debug('done')

  if np.isscalar(radius):
    scalar_radius = radius
    def radius(x): 
      return np.full(x.shape[0], scalar_radius)

  def rho(x):
    # the density function corresponding to the radius function
    return 1.0/(radius(x)**x.shape[1])
        
  nodes = poisson_discs(radius, domain, ntests=ntests, 
                        rmax_factor=rmax_factor)
  out = prepare_nodes(nodes, domain, rho=rho, **kwargs)
  return out
