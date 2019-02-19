''' 
This module contains functions that generate simplices defining 
commonly used domains.
'''
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator


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


def _topography_refine(x, y,
                       topo_func,
                       maxitr=1000,
                       tol=0.005):
  '''
  Adds vertices inside the two-dimensional complex hull defined by
  `x` and `y` so that the facets from a Delaunay triangulation can
  better approximate the topography function.
  '''
  xy = np.array([x, y]).T
  xtest = np.linspace(x.min(), x.max(), 1000)
  ytest = np.linspace(y.min(), y.max(), 1000)
  xtest, ytest = np.meshgrid(xtest, ytest)
  xtest, ytest = xtest.flatten(), ytest.flatten()
  xytest = np.array([xtest, ytest]).T
  # throw away points that are not in the convex hull
  xytest = xytest[Delaunay(xy).find_simplex(xytest) != -1]
  # evaluate the true topography function, which we want to
  # approximate with triangular facets.
  ftrue = topo_func(xytest)
  xyout = np.array(xy, copy=True)
  fout = np.array(topo_func(xyout))
  for _ in range(maxitr):
    I = LinearNDInterpolator(xyout, fout)
    ftest = I(xytest)
    # find where the linear interpolant (created with a delaunay
    # triangulation) has the greatest misfit with the true
    # topography function and then add a vertex at that point
    err = np.abs(ftest - ftrue)
    if err.max() <= tol*ftrue.ptp():
      break

    idx = np.argmax(err)
    xyout = np.vstack((xyout, xytest[[idx]]))
    fout = np.hstack((fout, ftrue[[idx]]))

  return xyout[:, 0], xyout[:, 1]


def topography(topo_func,
               radius,
               depth,
               tol=0.005,
               maxitr=1000):
  '''
  Creates a three-dimensional cylindrical domain where the elevation
  of the top of the cylinder is determined by `topo_func`.

  Parameters
  ----------
  topo_func : function
    This takes an (n, 2) array of (x, y) coordinates and returns the
    elevation at that point. The elevation can be positive or
    negative but it should not go lower than -`depth` because that
    will cause the domain to intersect itself. The length-scale of
    topographic features should be greater than `radius/1000`,
    otherwise they may not be well approximated by the output
    domain.

  radius : float
    Radius of the cylinder

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
  # the cylinder is approximated as a 60 sided polygon
  theta = np.linspace(0, 2*np.pi, 101)[:-1]
  nt = len(theta)
  # build the bottom vertices
  x1 = radius*np.cos(theta)
  y1 = radius*np.sin(theta)
  # add a point at the bottom center so that we have nicer shaped
  # facets
  x1 = np.hstack((x1, [0.0]))
  y1 = np.hstack((y1, [0.0]))
  z1 = -np.ones_like(x1)*depth
  n1 = len(x1)
  vert1 = np.array([x1, y1, z1]).T
  smp1 = Delaunay(np.array([x1, y1]).T).simplices
  # build the top vertices
  x2 = radius*np.cos(theta)
  y2 = radius*np.sin(theta)
  x2, y2 = _topography_refine(x2, y2, topo_func, tol=tol, maxitr=maxitr)
  z2 = topo_func(np.array([x2, y2]).T)
  vert2 = np.array([x2, y2, z2]).T
  smp2 = Delaunay(np.array([x2, y2]).T).simplices + n1
  # create the facets for the side of the domain
  smp_side = []
  for i in range(nt - 1):
    smp_side += [[i,    i+1,   n1+i],
                 [i+1, n1+i, n1+i+1]]

  smp_side += [[nt-1,       0, n1+nt-1],
               [   0, n1+nt-1,      n1]]
  # combine all the simplices together
  smp = np.vstack((smp1, smp2, smp_side))
  # join the vertices
  vert = np.vstack((vert1, vert2))
  # create a dictionary identifying which simplex belongs to which
  # group
  boundary_groups = {
    'bottom': np.arange(len(smp1)),
    'top': np.arange(len(smp2)) + len(smp1),
    'sides': np.arange(len(smp_side)) + len(smp1) + len(smp2)}

  return vert, smp, boundary_groups
