''' 
Module of cythonized functions for basic computational geometry in 2
and 3 dimensions. This modules requires all geometric objects (e.g.
volumes, polygons, surfaces, segments, etc.) to be described as
simplicial complexes. A simplicial complex is a collection of
simplices (e.g. segments, triangles, tetrahedra, etc.).  In this
module, simplicial complexes in D-dimenional space are described with
an (N, D) array of vertices and and (M, D) array describing the
indices of vertices making up each simplex. As an example, the unit
square in two dimensions can be described as collection of line
segments:

>>> vertices = [[0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0]]
>>> simplices = [[0, 1],
                 [1, 2],
                 [2, 3],
                 [3, 0]]

A three dimensional cube can similarly be described as a collection
of triangles:

>>> vertices = [[0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0]]
>>> simplices = [[0, 1, 4],
                 [1, 5, 4],
                 [1, 7, 5],
                 [1, 3, 7],
                 [0, 1, 3],
                 [0, 2, 3],
                 [0, 2, 6],
                 [0, 4, 6],
                 [4, 5, 7],
                 [4, 6, 7],
                 [2, 3, 7],
                 [2, 6, 7]]
 
This module is primarily use to find whether and where line segments
intersect a simplicial complex and whether points are contained within
a closed simplicial complex.  For example, one can determine whether a
collection of points, saved as `points`, are contained within a
simplicial complex, defined by `vertices` and `simplices` with the
command

>>> contains(points, vertices, simplices)

which returns a boolean array.

One can find the number of times a collection of line segments, 
defined by `start_points` and `end_points`, intersect a simplicial 
complex with the command

>> intersection_count(start_points, end_points, vertices, simplices)

which returns an array of the number of simplexes intersections for
each segment. If it is known that a collection of line segments
intersect a simplicial complex then the intersection point can be
found with the command

>> intersection(start_points, end_points, vertices, simplices)
 
This returns an (N, D) float array of intersection points, and an (N,)
int array identifying which simplex the intersection occurred at. If a
line segment does not intersect the simplicial complex then the above
command returns a ValueError. If there are multiple intersections for
a single segment then only the first detected intersection will be
returned.

There are numerous other packages which can perform the same tasks as
this module.  For example geos (http://trac.osgeo.org/geos/) and gts
(http://gts.sourceforge.net/).  However, the python bindings for these
packages are too slow for RBF purposes.
'''
# python imports
from __future__ import division
from itertools import combinations

import numpy as np
from scipy.special import factorial

from rbf.utils import assert_shape

# cython imports
cimport numpy as np
from cython cimport boundscheck, wraparound, cdivision
from libc.math cimport fabs, fmin, sqrt, INFINITY


cdef struct vector1d:
  double x


cdef struct vector2d:
  double x
  double y


cdef struct vector3d:
  double x
  double y
  double z


cdef struct segment1d:
  vector1d a
  vector1d b


cdef struct segment2d:
  vector2d a
  vector2d b


cdef struct segment3d:
  vector3d a
  vector3d b


cdef struct triangle2d:
  vector2d a
  vector2d b
  vector2d c


cdef struct triangle3d:
  vector3d a
  vector3d b
  vector3d c


@cdivision(True)
cdef bint point_in_segment(vector1d vec, segment1d seg) nogil:  
  ''' 
  Identifies whether a point in 1D space is within a 1D segment. For 
  the sake of consistency, this is done by projecting the point into a 
  barycentric coordinate system defined by the segment. The point is 
  then inside the segment if both of the barycentric coordinates are 
  positive
  '''
  cdef: 
    double l1, l2
  
  # find barycentric coordinates  
  l1 = (seg.a.x - vec.x)/(seg.a.x - seg.b.x)
  l2 = (vec.x - seg.b.x)/(seg.a.x - seg.b.x)
  if (l1 >= 0.0) & (l2 >= 0.0):
    return True
  else: 
    return False


@cdivision(True)
cdef bint point_in_triangle(vector2d vec, triangle2d tri) nogil:  
  ''' 
  Identifies whether a point in 2D space is within a 2D triangle. This 
  is done by projecting the point into a barycentric coordinate system 
  defined by the triangle. The point is then inside the segment if all 
  three of the barycentric coordinates are positive
  '''
  cdef: 
    double det, l1, l2, l3

  # find barycentric coordinates  
  det = ((tri.b.y - tri.c.y)*(tri.a.x - tri.c.x) + 
         (tri.c.x - tri.b.x)*(tri.a.y - tri.c.y))
  l1 = ((tri.b.y - tri.c.y)*(vec.x - tri.c.x) + 
        (tri.c.x - tri.b.x)*(vec.y - tri.c.y))/det
  l2 = ((tri.c.y - tri.a.y)*(vec.x - tri.c.x) + 
        (tri.a.x - tri.c.x)*(vec.y - tri.c.y))/det
  l3 = 1 - l1 - l2
  if (l1 >= 0.0) & (l2 >= 0.0) & (l3 >= 0.0):
    return True
  else:
    return False


cdef vector2d segment_normal_2d(segment2d seg) nogil:
  ''' 
  Returns the vector normal to a 2d line segment
  '''
  cdef:
    vector2d out

  out.x = seg.b.y - seg.a.y
  out.y = -(seg.b.x - seg.a.x)
  return out
  

cdef vector3d triangle_normal_3d(triangle3d tri) nogil:
  ''' 
  Returns the vector normal to a 3d triangle
  '''
  cdef:
    vector3d out

  out.x = ((tri.b.y - tri.a.y)*(tri.c.z - tri.a.z) - 
           (tri.b.z - tri.a.z)*(tri.c.y - tri.a.y))
  out.y = -((tri.b.x - tri.a.x)*(tri.c.z - tri.a.z) - 
            (tri.b.z - tri.a.z)*(tri.c.x - tri.a.x)) 
  out.z = ((tri.b.x - tri.a.x)*(tri.c.y - tri.a.y) - 
           (tri.b.y - tri.a.y)*(tri.c.x - tri.a.x))
  return out


@cdivision(True)
cdef bint is_intersecting_2d(segment2d seg1,
                             segment2d seg2) nogil:
  ''' 
  Identifies whether two 2D segments intersect. An intersection is
  detected if both segments are not colinear and if any part of the
  two segments touch
  '''
  cdef:
    double proj1, proj2
    vector2d pnt, norm
    vector1d pnt_proj
    segment1d seg_proj

  # find the normal vector components for segment 2
  norm = segment_normal_2d(seg2)

  # project both points in segment 1 onto the normal vector
  proj1 = ((seg1.a.x - seg2.a.x)*norm.x +
           (seg1.a.y - seg2.a.y)*norm.y)
  proj2 = ((seg1.b.x - seg2.a.x)*norm.x +
           (seg1.b.y - seg2.a.y)*norm.y)

  if proj1*proj2 > 0:
    return False

  # return false if the segments are collinear
  if (proj1 == 0) & (proj2 == 0):
    return False

  # find the point where segment 1 intersects the line overlapping 
  # segment 2 
  pnt.x = seg1.a.x + (proj1/(proj1 - proj2))*(seg1.b.x - seg1.a.x)
  pnt.y = seg1.a.y + (proj1/(proj1 - proj2))*(seg1.b.y - seg1.a.y)

  # if the normal x component is larger then compare y values
  if fabs(norm.x) >= fabs(norm.y):
    pnt_proj.x = pnt.y    
    seg_proj.a.x = seg2.a.y
    seg_proj.b.x = seg2.b.y
    return point_in_segment(pnt_proj, seg_proj)

  else:
    pnt_proj.x = pnt.x
    seg_proj.a.x = seg2.a.x
    seg_proj.b.x = seg2.b.x
    return point_in_segment(pnt_proj, seg_proj)


@cdivision(True)
cdef bint is_intersecting_3d(segment3d seg,
                             triangle3d tri) nogil:
  ''' 
  Identifies whether a 3D segment intersects a 3D triangle. An
  intersection is detected if the segment and triangle are not
  coplanar and if any part of the segment touches the triangle at an
  edge or in the interior.
  '''
  cdef:
    double proj1, proj2
    vector3d pnt, norm
    vector2d pnt_proj
    triangle2d tri_proj

  # find triangle normal vector components
  norm = triangle_normal_3d(tri)

  proj1 = ((seg.a.x - tri.a.x)*norm.x + 
           (seg.a.y - tri.a.y)*norm.y +
           (seg.a.z - tri.a.z)*norm.z)
  proj2 = ((seg.b.x - tri.a.x)*norm.x + 
           (seg.b.y - tri.a.y)*norm.y +
           (seg.b.z - tri.a.z)*norm.z)

  if proj1*proj2 > 0:
    return False

  # coplanar segments will always return false There is a possibility
  # that the segment touches one point on the triangle
  if (proj1 == 0) & (proj2 == 0):
    return False

  # intersection point
  pnt.x = seg.a.x + (proj1/(proj1 - proj2))*(seg.b.x - seg.a.x)
  pnt.y = seg.a.y + (proj1/(proj1 - proj2))*(seg.b.y - seg.a.y)
  pnt.z = seg.a.z + (proj1/(proj1 - proj2))*(seg.b.z - seg.a.z)

  if (fabs(norm.x) >= fabs(norm.y)) & (fabs(norm.x) >= fabs(norm.z)):
    pnt_proj.x = pnt.y
    pnt_proj.y = pnt.z
    tri_proj.a.x = tri.a.y
    tri_proj.a.y = tri.a.z
    tri_proj.b.x = tri.b.y
    tri_proj.b.y = tri.b.z
    tri_proj.c.x = tri.c.y
    tri_proj.c.y = tri.c.z
    return point_in_triangle(pnt_proj, tri_proj)

  elif (fabs(norm.y) >= fabs(norm.x)) & (fabs(norm.y) >= fabs(norm.z)):
    pnt_proj.x = pnt.x
    pnt_proj.y = pnt.z
    tri_proj.a.x = tri.a.x
    tri_proj.a.y = tri.a.z
    tri_proj.b.x = tri.b.x
    tri_proj.b.y = tri.b.z
    tri_proj.c.x = tri.c.x
    tri_proj.c.y = tri.c.z
    return point_in_triangle(pnt_proj, tri_proj)

  elif (fabs(norm.z) >= fabs(norm.x)) & (fabs(norm.z) >= fabs(norm.y)):
    pnt_proj.x = pnt.x
    pnt_proj.y = pnt.y
    tri_proj.a.x = tri.a.x
    tri_proj.a.y = tri.a.y
    tri_proj.b.x = tri.b.x
    tri_proj.b.y = tri.b.y
    tri_proj.c.x = tri.c.x
    tri_proj.c.y = tri.c.y
    return point_in_triangle(pnt_proj, tri_proj)


@boundscheck(False)
@wraparound(False)
def intersection_count_2d(double[:, :] start_pnts,
                          double[:, :] end_pnts,
                          double[:, :] vertices,
                          long[:, :] simplices):
  ''' 
  Returns an array containing the number of simplices intersected
  between start_pnts and end_pnts. This is parallelizable.
  '''
  cdef:
    unsigned int i, j
    unsigned int N = start_pnts.shape[0]
    unsigned int M = simplices.shape[0]
    long[:] out = np.zeros((N,), dtype=int, order='c')
    segment2d seg1, seg2
    
  for i in range(N):
    seg1.a.x = start_pnts[i, 0]
    seg1.a.y = start_pnts[i, 1]
    seg1.b.x = end_pnts[i, 0]
    seg1.b.y = end_pnts[i, 1]
    for j in range(M):
      seg2.a.x = vertices[simplices[j, 0], 0]
      seg2.a.y = vertices[simplices[j, 0], 1]
      seg2.b.x = vertices[simplices[j, 1], 0]
      seg2.b.y = vertices[simplices[j, 1], 1]
      if is_intersecting_2d(seg1, seg2):
        out[i] += 1

  return np.asarray(out)


@boundscheck(False)
@wraparound(False)
def intersection_count_3d(double[:, :] start_pnts,
                          double[:, :] end_pnts,                         
                          double[:, :] vertices,
                          long[:, :] simplices):
  ''' 
  Returns an array of the number of intersections between each line
  segment, described by start_pnts and end_pnts, and the simplices
  '''
  cdef:
    int i, j
    int N = start_pnts.shape[0]
    int M = simplices.shape[0]
    long[:] out = np.zeros((N,), dtype=int, order='c')
    segment3d seg
    triangle3d tri

  for i in range(N):
    seg.a.x = start_pnts[i, 0]
    seg.a.y = start_pnts[i, 1]
    seg.a.z = start_pnts[i, 2]
    seg.b.x = end_pnts[i, 0]
    seg.b.y = end_pnts[i, 1]
    seg.b.z = end_pnts[i, 2]
    for j in range(M):
      tri.a.x = vertices[simplices[j, 0], 0]
      tri.a.y = vertices[simplices[j, 0], 1]
      tri.a.z = vertices[simplices[j, 0], 2]
      tri.b.x = vertices[simplices[j, 1], 0]
      tri.b.y = vertices[simplices[j, 1], 1]
      tri.b.z = vertices[simplices[j, 1], 2]
      tri.c.x = vertices[simplices[j, 2], 0]
      tri.c.y = vertices[simplices[j, 2], 1]
      tri.c.z = vertices[simplices[j, 2], 2]
      if is_intersecting_3d(seg, tri):
        out[i] += 1

  return np.asarray(out)  


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def intersection_2d(double[:, :] start_pnts,
                    double[:, :] end_pnts,
                    double[:, :] vertices,
                    long[:, :] simplices):
  ''' 
  Returns the intersection point and the simplex being intersected by
  the segment defined by `start_pnts` and `end_pnts`.

  Notes
  -----
  if there is no intersection then a ValueError is returned. If there
  are multiple intersections, then the intersection closest to
  `start_pnts` will be returned.
  '''
  cdef:
    unsigned int i, j
    unsigned int N = start_pnts.shape[0]
    unsigned int M = simplices.shape[0]
    long[:] out_idx = np.empty((N,), dtype=int, order='c')
    double[:, :] out_pnt = np.empty((N, 2), dtype=float, order='c')
    double proj1, proj2, t, tmin
    bint found_intersection
    segment2d seg1, seg2 
    vector2d norm
    
  for i in range(N):
    seg1.a.x = start_pnts[i, 0]
    seg1.a.y = start_pnts[i, 1]
    seg1.b.x = end_pnts[i, 0]
    seg1.b.y = end_pnts[i, 1]
    found_intersection = False
    tmin = INFINITY
    for j in range(M):
      seg2.a.x = vertices[simplices[j, 0], 0]
      seg2.a.y = vertices[simplices[j, 0], 1]
      seg2.b.x = vertices[simplices[j, 1], 0]
      seg2.b.y = vertices[simplices[j, 1], 1]
      if is_intersecting_2d(seg1, seg2):
        found_intersection = True
        # the intersecting segment should be the first segment
        # intersected when going from seg1.a to seg1.b
        norm = segment_normal_2d(seg2) 
        proj1 = ((seg1.a.x - seg2.a.x)*norm.x +
                 (seg1.a.y - seg2.a.y)*norm.y)
        proj2 = ((seg1.b.x - seg2.a.x)*norm.x +
                 (seg1.b.y - seg2.a.y)*norm.y)
        # t is a scalar between 0 and 1. If t=0 then the intersection
        # is at seg1.a and if t=1 then the intersection is at seg1.b
        t = proj1/(proj1 - proj2)
        if t < tmin:
          tmin = t
          out_idx[i] = j
          out_pnt[i, 0] = seg1.a.x + t*(seg1.b.x - seg1.a.x)
          out_pnt[i, 1] = seg1.a.y + t*(seg1.b.y - seg1.a.y)

    if not found_intersection:
      raise ValueError(
        'No intersection was found for segment [[%s, %s], [%s, %s]]' % 
        (seg1.a.x, seg1.a.y, seg1.b.x, seg1.b.y))

  return np.asarray(out_pnt), np.asarray(out_idx)  


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def intersection_3d(double[:, :] start_pnts,
                    double[:, :] end_pnts,                         
                    double[:, :] vertices,
                    long[:, :] simplices):
  ''' 
  Returns the intersection point and the simplex being intersected by
  the segment defined by `start_pnts` and `end_pnts`.

  Notes
  -----
  if there is no intersection then a ValueError is returned. If there
  are multiple intersections, then the intersection closest to
  `start_pnts` will be returned.
  '''
  cdef:
    int i, j
    int N = start_pnts.shape[0]
    int M = simplices.shape[0]
    double proj1, proj2, t, tmin
    bint found_intersection
    long[:] out_idx = np.empty((N,), dtype=int, order='c')
    double[:, :] out_pnt = np.empty((N, 3), dtype=float, order='c')
    segment3d seg
    triangle3d tri
    vector3d norm

  for i in range(N):
    seg.a.x = start_pnts[i, 0]
    seg.a.y = start_pnts[i, 1]
    seg.a.z = start_pnts[i, 2]
    seg.b.x = end_pnts[i, 0]
    seg.b.y = end_pnts[i, 1]
    seg.b.z = end_pnts[i, 2]
    tmin = INFINITY        
    found_intersection = False
    for j in range(M):
      tri.a.x = vertices[simplices[j, 0], 0]
      tri.a.y = vertices[simplices[j, 0], 1]
      tri.a.z = vertices[simplices[j, 0], 2]
      tri.b.x = vertices[simplices[j, 1], 0]
      tri.b.y = vertices[simplices[j, 1], 1]
      tri.b.z = vertices[simplices[j, 1], 2]
      tri.c.x = vertices[simplices[j, 2], 0]
      tri.c.y = vertices[simplices[j, 2], 1]
      tri.c.z = vertices[simplices[j, 2], 2]
      if is_intersecting_3d(seg, tri):
        found_intersection = True
        norm = triangle_normal_3d(tri)
        proj1 = ((seg.a.x - tri.a.x)*norm.x + 
                 (seg.a.y - tri.a.y)*norm.y +
                 (seg.a.z - tri.a.z)*norm.z)
        proj2 = ((seg.b.x - tri.a.x)*norm.x + 
                 (seg.b.y - tri.a.y)*norm.y +
                 (seg.b.z - tri.a.z)*norm.z)
        # t is a scalar between 0 and 1. If t=0 then the intersection is 
        # at seg1.a and if t=1 then the intersection is at seg1.b
        t = proj1/(proj1 - proj2)
        if t < tmin:
          tmin = t
          out_idx[i] = j
          out_pnt[i, 0] = seg.a.x + t*(seg.b.x - seg.a.x)
          out_pnt[i, 1] = seg.a.y + t*(seg.b.y - seg.a.y)
          out_pnt[i, 2] = seg.a.z + t*(seg.b.z - seg.a.z)

    if not found_intersection:
      raise ValueError(
        'No intersection was found for segment '
        '[[%s, %s, %s], [%s, %s, %s]]' % 
        (seg.a.x, seg.a.y, seg.a.z, seg.b.x, seg.b.y, seg.b.z))

  return np.asarray(out_pnt), np.asarray(out_idx)  


# end-user functions
####################################################################
def intersection(start_points, end_points, vertices, simplices):
  ''' 
  Returns the intersection between line segments and a simplicial
  complex.  The line segments are described by `start_points` and
  `end_points`, and the simplicial complex is described by `vertices`
  and `simplices`. This function works for 2 and 3 spatial dimensions.

  Parameters
  ----------
  start_points : (N, D) float array
    Vertices describing one end of the line segments. `N` is the
    number of line segments and `D` is the number of dimensions

  end_points : (N, D) float array 
    Vertices describing the other end of the line segments. 

  vertices : (M, D) float array 
    Vertices within the simplicial complex. M is the number of 
    vertices.

  simplices : (P, D) int array
    Connectivity of the vertices. Each row contains the vertex 
    indices which form one simplex of the simplicial complex

  Returns
  -------
  out : (N, D) float array
    The points where the line segments intersect the simplicial
    complex

  out : (N,) int array
    The index of the simplex that the line segments intersect

  Notes
  -----
  This function fails when a intersection is not found for a line
  segment. If there are multiple intersections then the intersection
  closest to start_point is used.

  '''
  start_points = np.asarray(start_points, dtype=float)
  end_points = np.asarray(end_points, dtype=float)
  vertices = np.asarray(vertices, dtype=float)
  simplices = np.asarray(simplices, dtype=int)

  assert_shape(start_points, (None, None), 'start_points')
  assert_shape(end_points, start_points.shape, 'end_points') 
  dim = start_points.shape[1]
  assert_shape(vertices, (None, dim), 'vertices')
  assert_shape(simplices, (None, dim), 'simplices')    

  if dim == 2:
    out = intersection_2d(start_points, 
                          end_points, 
                          vertices, 
                          simplices)
  elif dim == 3:
    out = intersection_3d(start_points, 
                          end_points, 
                          vertices, 
                          simplices)
  else:
    raise ValueError(
      'The number of spatial dimensions must be 2 or 3')
      
  return out


def intersection_count(start_points, end_points, vertices, simplices):
  ''' 
  Returns the number of simplices crossed by the line segments. The
  line segments are described by `start_points` and `end_points`. This
  function works for 2 and 3 spatial dimensions.

  Parameters
  ----------
  start_points : (N, D) array
    Vertices describing one end of the line segments. `N` is the
    number of line segments and `D` is the number of dimensions

  end_points : (N, D) array 
    Vertices describing the other end of the line segments

  vertices : (M, D) array 
    Vertices within the simplicial complex. `M` is the number of 
    vertices

  simplices : (P, D) array
    Connectivity of the vertices. Each row contains the vertex 
    indices which form one simplex of the simplicial complex

  Returns
  -------
  out : (N,) int array
    intersection counts

  '''
  start_points = np.asarray(start_points, dtype=float)
  end_points = np.asarray(end_points, dtype=float)
  vertices = np.asarray(vertices, dtype=float)
  simplices = np.asarray(simplices, dtype=int)
  
  assert_shape(start_points, (None, None), 'start_points')
  assert_shape(end_points, start_points.shape, 'end_points') 
  dim = start_points.shape[1]
  assert_shape(vertices, (None, dim), 'vertices')
  assert_shape(simplices, (None, dim), 'simplices')    

  if dim == 2:
    out = intersection_count_2d(start_points,
                                end_points, 
                                vertices, 
                                simplices)
  elif dim == 3:
    out = intersection_count_3d(start_points, 
                                end_points, 
                                vertices, 
                                simplices)
  else:
    raise ValueError(
      'The number of spatial dimensions must be 2 or 3')

  return out


def contains(points, vertices, simplices):
  ''' 
  Returns a boolean array identifying whether the points are contained
  within a closed simplicial complex. The simplicial complex is
  described by `vertices` and `simplices`. This function works for 2
  and 3 spatial dimensions.

  Parameters
  ----------
  points : (N,D) array
    Test points

  vertices : (M,D) array
    Vertices of the simplicial complex

  simplices : (P,D) int array 
    Connectivity of the vertices. Each row contains the vertex 
    indices which form one simplex of the simplicial complex

  Returns
  -------
  out : (N,) bool array 
    Indicates which test points are in the simplicial complex

  Notes
  -----
  This function does not ensure that the simplicial complex is
  closed.  If it is not then bogus results will be returned. 
    
  This function determines whether a point is contained within the
  simplicial complex by finding the number of intersections between
  each point and an arbitrary outside point.  It is possible,
  although rare, that this function will fail if the line segment
  intersects a simplex at an edge.

  This function does not require any particular orientation for the 
  simplices

  '''
  points = np.asarray(points, dtype=float)
  vertices = np.asarray(vertices, dtype=float)
  simplices = np.asarray(simplices, dtype=int)

  assert_shape(points, (None, None), 'points')
  dim = points.shape[1]
  assert_shape(vertices, (None, dim), 'vertices')
  assert_shape(simplices, (None, dim), 'simplices')    

  rnd = np.random.uniform(0.5, 2.0, (points.shape[1],))    
  outside_point = vertices.min(axis=0) - rnd*vertices.ptp(axis=0)
  outside_point = np.repeat([outside_point], points.shape[0], axis=0)
  count = intersection_count(points, 
                             outside_point, 
                             vertices, 
                             simplices)
  out = np.array(count % 2, dtype=bool)
  return out


def oriented_simplices(vert, smp):
  ''' 
  Returns simplex indices that are ordered such that each simplex
  normal vector, as defined by the right hand rule, points outward
                                    
  Parameters
  ----------
  vertices : (M, D) array
    Vertices within the simplicial complex

  simplices : (P, D) int array 
    Connectivity of the vertices. Each row contains the vertex 
    indices which form one simplex of the simplicial complex

  Returns
  -------
  out : (P,D) int array
    oriented simplices

  Notes                                
  -----                        
  If one dimensional simplices are given, then the simplices are
  returned unaltered.

  This function does not ensure that the simplicial complex is
  closed.  If it is not then bogus results will be returned. 
  '''
  vert = np.asarray(vert, dtype=float)
  smp = np.array(smp, dtype=int, copy=True)
  assert_shape(vert, (None, None), 'vert')
  dim = vert.shape[1]
  assert_shape(smp, (None, dim), 'smp')

  # length scale of the domain
  scale = vert.ptp(axis=0).max()
  dx = 1e-10*scale
  # find the normal for each simplex    
  norms = simplex_normals(vert, smp)
  # find the centroid for each simplex      
  points = np.mean(vert[smp], axis=1)
  # push points in the direction of the normals  
  points += dx*norms
  # find which simplices are oriented such that their normals point  
  # inside                           
  faces_inside = contains(points, vert, smp)
  flip_smp = smp[faces_inside]
  flip_smp[:, [0, 1]] = flip_smp[:, [1, 0]]
  smp[faces_inside] = flip_smp
  return smp


def simplex_normals(vert, smp):
  ''' 
  Returns the normal vectors for each simplex. Orientation is 
  determined by the right hand rule

  Parameters
  ----------
  vertices : (M, D) array
    Vertices within the simplicial complex

  simplices : (P, D) int array 
    Connectivity of the vertices. Each row contains the vertex 
    indices which form one simplex of the simplicial complex

  Returns
  -------
  out : (P, D) array
    normals vectors
      
  Notes
  -----
  This is only defined for two and three dimensional simplices

  '''
  vert = np.asarray(vert, dtype=float)
  smp = np.asarray(smp, dtype=int)
  assert_shape(vert, (None, None), 'vert')
  dim = vert.shape[1]
  assert_shape(smp, (None, dim), 'smp')

  M = vert[smp[:, 1:]] - vert[smp[:, [0]]]
  Msubs = [np.delete(M, i, -1) for i in range(dim)]
  out = np.linalg.det(Msubs)
  out[1::2] *= -1
  out = np.rollaxis(out, -1)
  out /= np.linalg.norm(out, axis=-1)[..., None]
  return out


def simplex_outward_normals(vert, smp):
  ''' 
  Returns the outward normal vectors for each simplex. The sign of the 
  returned vectors are only meaningful if the simplices enclose an 
  area in two-dimensional space or a volume in three-dimensional space

  Parameters
  ----------
  vertices : (M, D) array
    Vertices within the simplicial complex

  simplices : (P, D) int array 
    Connectivity of the vertices. Each row contains the vertex indices
    which form one simplex of the simplicial complex

  Returns
  -------
  out : (P, D) array
    normals vectors
      
  Notes
  -----
  This is only defined for two and three dimensional simplices

  '''
  smp = oriented_simplices(vert, smp)
  return simplex_normals(vert, smp)


def simplex_upward_normals(vert, smp):
  ''' 
  Returns the upward pointing normal vectors for each simplex. The up
  direction is assumed to be the last coordinate axis.

  Parameters
  ----------
  vertices : (M,D) array
    Vertices within the simplicial complex

  simplices : (P,D) int array 
    Connectivity of the vertices. Each row contains the vertex 
    indices which form one simplex of the simplicial complex

  Returns
  -------
  out : (P,D) array
    normals vectors

  '''
  out = simplex_normals(vert, smp)
  out[out[:, -1] < 0] *= -1
  return out


def enclosure(vert, smp, orient=True):
  ''' 
  Returns the volume of a polyhedra, area of a polygon, or length of a
  segment enclosed by the simplicial complex

  Parameters
  ----------
  vertices : (M, D) array
    Vertices within the simplicial complex

  simplices : (P, D) int array 
    Connectivity of the vertices. Each row contains the vertex 
    indices which form one simplex of the simplicial complex

  orient : bool, optional
    If true, the simplices are reordered with oriented_simplices. 
    The time for this function increase quadratically with the 
    number of simplices. Set to false if you are confident that the 
    simplices are properly oriented. This does nothing for 
    one-dimensional simplices

  Returns
  -------
  out : float
     
  Notes
  -----
  This function does not ensure that the simplicial complex is 
  closed and does not intersect itself. If it is not then bogus 
  results will be returned.
  '''
  vert = np.array(vert, dtype=float, copy=True)
  smp = np.asarray(smp, dtype=int)
  assert_shape(vert, (None, None), 'vert')
  dim = vert.shape[1]
  assert_shape(smp, (None, dim), 'smp')
  if orient:
    smp = oriented_simplices(vert, smp)

  # center the vertices for the purpose of numerical stability
  vert -= np.mean(vert, axis=0)
  signed_volumes = (1.0/factorial(dim))*np.linalg.det(vert[smp])
  volume = np.sum(signed_volumes)
  return volume
