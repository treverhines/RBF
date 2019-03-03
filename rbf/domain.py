''' 
This module contains functions that generate simplices defining 
commonly used domains.
'''
import logging

import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from rbf.geometry import oriented_simplices

LOG = logging.getLogger(__name__)

def save_as_polygon_file(filename, vert, smp):
    '''
    Write the three-dimensional domain to a polygon file format. This
    file format can be read in by Paraview.

    http://paulbourke.net/dataformats/ply/

    Parameters
    -----------
    filename : str

    vert : (n, 3) float array

    smp : (m, 3) int array
  
    '''
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %s\n' % len(vert))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('element face %s\n' % len(smp))
        f.write('property list uchar int vertex_index\n')
        f.write('end_header\n')
        for v in vert:
            f.write('%.4f %.4f %.4f\n' % (v[0], v[1], v[2]))

        for s in smp:
            f.write('3 %s %s %s\n' % (s[0], s[1], s[2]))


def _circle_refine(vert, smp):
  V = vert.shape[0]
  S = smp.shape[0]
  new_vert = np.zeros((V+S, 2), dtype=float)
  new_vert[:V, :] = vert
  new_smp = np.zeros((2*S, 2), dtype=int)
  for si, s in enumerate(smp):
    a, b = vert[s]
    i = V + si
    new_vert[i] = a+b
    new_smp[2*si]   = [s[0],    i]
    new_smp[2*si+1] = [   i, s[1]]

  new_vert = new_vert / np.linalg.norm(new_vert, axis=1)[:, None]
  return new_vert, new_smp


def circle(r=5):
  ''' 
  Returns the outwardly oriented simplices of a circle
  
  Parameters
  ----------
  r : int, optional
    refinement order
      
  Returns
  -------
  vert : (N, 2) float array
  smp : (M, 2) int array
    
  '''
  vert = np.array([[ 1.0, 0.0],
                   [ 0.0, 1.0],
                   [-1.0, 0.0],
                   [0.0, -1.0]])
  smp = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 0]])
  for i in range(r):
    vert, smp = _circle_refine(vert, smp)

  return vert, smp  


def _sphere_refine(vert, smp):
  V = vert.shape[0]
  S = smp.shape[0]
  new_vert = np.zeros((V+3*S, 3), dtype=float)
  new_vert[:V, :] = vert
  new_smp = np.zeros((4*S, 3), dtype=int)
  for si, s in enumerate(smp):
    a, b, c = vert[s]
    i = V + 3*si
    j = i + 1
    k = i + 2
    new_vert[i] = a+b
    new_vert[j] = b+c
    new_vert[k] = a+c
    new_smp[4*si]   = [   i,    j,    k]
    new_smp[4*si+1] = [s[0],    i,    k]
    new_smp[4*si+2] = [   i, s[1],    j]
    new_smp[4*si+3] = [   k,    j, s[2]]

  new_vert = new_vert / np.linalg.norm(new_vert, axis=1)[:, None]
  return new_vert, new_smp


def sphere(r=5):
  ''' 
  returns the outwardly oriented simplices of a sphere

  Parameters
  ----------
  r : int, optional
    refinement order
      
  Returns
  -------
  vert : (N,2) float array

  smp : (M,2) int array
  
  '''
  f = np.sqrt(2.0)/2.0
  vert = np.array([[ 0.0, -1.0, 0.0],
                   [  -f,  0.0,   f],
                   [   f,  0.0,   f],
                   [   f,  0.0,  -f],
                   [  -f,  0.0,  -f],
                   [ 0.0,  1.0, 0.0]])
  smp = np.array([[0, 2, 1],
                  [0, 3, 2],
                  [0, 4, 3],
                  [0, 1, 4],
                  [5, 1, 2],
                  [5, 2, 3],
                  [5, 3, 4],
                  [5, 4, 1]])

  for i in range(r):
    vert, smp = _sphere_refine(vert, smp)

  return vert,smp


def _topography_surface(xbounds, 
                        ybounds,
                        topo_func,
                        maxitr=10000,
                        tol=0.01,
                        fine_grid=1000,
                        coarse_grid=10):
  '''
  Creates the surface vertices for the topography function
  '''
  # create a coarse and fine grid. The output vertices will contain
  # the coarse grid plus whichever vertices are needed from the fine
  # grid
  xfine = np.linspace(*xbounds, fine_grid + 1)
  yfine = np.linspace(*ybounds, fine_grid + 1)
  xfine, yfine = np.meshgrid(xfine, yfine)
  xfine, yfine = xfine.flatten(), yfine.flatten()
  xyfine = np.array([xfine, yfine]).T

  xcoarse = np.linspace(*xbounds, coarse_grid + 1)
  ycoarse = np.linspace(*ybounds, coarse_grid + 1)
  xcoarse, ycoarse = np.meshgrid(xcoarse, ycoarse)
  xcoarse, ycoarse = xcoarse.flatten(), ycoarse.flatten()
  xycoarse = np.array([xcoarse, ycoarse]).T

  zfine = topo_func(xyfine)
  zcoarse = topo_func(xycoarse)

  # make sure the topography function is zero at the edges. This
  # ensures that the domain will be closed
  idx = ((xyfine[:, 0] == xbounds[0]) | 
         (xyfine[:, 0] == xbounds[1]) |
         (xyfine[:, 1] == ybounds[0]) | 
         (xyfine[:, 1] == ybounds[1]))
  zfine[idx] = 0.0    

  idx = ((xycoarse[:, 0] == xbounds[0]) | 
         (xycoarse[:, 0] == xbounds[1]) |
         (xycoarse[:, 1] == ybounds[0]) | 
         (xycoarse[:, 1] == ybounds[1]))
  zcoarse[idx] = 0.0    
  
  xyout = np.copy(xycoarse)
  zout = np.copy(zcoarse)
  LOG.info('Generating the surface facets ...')
  for itr in range(maxitr):
    # find where the linear interpolant (created with a delaunay
    # triangulation) has the greatest misfit with the true topography
    # function and then add a vertex at that point
    I = LinearNDInterpolator(xyout, zout)
    zitp = I(xyfine)
    err = np.abs(zitp - zfine)
    if err.max() <= tol*zfine.ptp():
      LOG.info(
        'Finished generating the surface facets. The maximum '
        'interpolation error is %s' % err.max())
      break

    idx = np.argmax(err)
    zout = np.hstack((zout, zfine[idx]))
    xyout = np.vstack((xyout, xyfine[idx]))
    
  if itr == (maxitr - 1):
      LOG.warning(
        'The maximum number of iterations was reached while '
        'generating the surface facets. The maximum interpolation '
        'error is %s'
        % err.max())
  
  vert = np.hstack((xyout, zout[:, None]))
  smp = Delaunay(xyout).simplices     
  return vert, smp


def topography(topo_func,
               xbounds,
               ybounds,
               depth,
               tol=0.01,
               maxitr=10000,
               fine_grid=1000,
               coarse_grid=10):

  '''
  Creates a three-dimensional cylindrical domain where the elevation
  of the top of the cylinder is determined by `topo_func`.

  Parameters
  ----------
  topo_func : function
    This takes an (n, 2) array of (x, y) coordinates and returns the
    elevation at that point. The elevation can be positive or negative
    but it should taper to zero at the edges of the domain and it
    should not go lower than -`depth`.

  xbounds, ybounds : 2-tuple
    Domain x and y bounds
      
  depth : float
    Depth of the cylinder        

  tol : float
    The maximum error allowed when approximating the topography
    function with triangular facets. This is a fraction of the
    peak-to-peak for `topo_func`.

  maxitr : int
    Max number of vertices to add to the top of the domain

  Returns
  -------
  vert : (P, 3) float array
    vertices of the domain

  smp : (Q, 3) int array
    Indices of the vertices that make up each facet of the domain

  boundary_groups : dict
    Dictionary identifying which facets belong to which part of the
    domain

  '''
  vert = np.array([[xbounds[0], ybounds[0], -depth],
                   [xbounds[0], ybounds[0],    0.0],
                   [xbounds[0], ybounds[1], -depth],
                   [xbounds[0], ybounds[1],    0.0],
                   [xbounds[1], ybounds[0], -depth],
                   [xbounds[1], ybounds[0],    0.0],
                   [xbounds[1], ybounds[1], -depth],
                   [xbounds[1], ybounds[1],    0.0]])
  smp = np.array([[0, 2, 6],
                  [0, 4, 6],
                  [0, 1, 3],
                  [0, 2, 3],
                  [0, 1, 4],
                  [1, 5, 4],
                  [4, 5, 7],
                  [4, 6, 7],
                  [2, 3, 7],
                  [2, 6, 7]])
  # build the top vertices
  vert_surf, smp_surf = _topography_surface(xbounds, 
                                            ybounds, 
                                            topo_func, 
                                            tol=tol, 
                                            maxitr=maxitr,
                                            fine_grid=fine_grid,
                                            coarse_grid=coarse_grid)
  vert = np.vstack((vert, vert_surf))
  smp = np.vstack((smp, smp_surf + 8))    
  smp = oriented_simplices(vert, smp)
  boundary_groups = {
    'bottom': np.array([0, 1]),
    'sides': np.array([2, 3, 4, 5, 6, 7, 8, 9]),
    'top': 10 + np.arange(smp_surf.shape[0])}

  return vert, smp, boundary_groups
