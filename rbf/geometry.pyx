''' 
Module of cythonized functions for basic computational geometry in 1, 
2, and 3 dimensions. This modules requires all geometric objects (e.g. 
volumes, polygons, surfaces, segments, etc.) to be described as 
simplicial complexes. A simplicial complex is a collection of 
simplices (e.g. segments, triangles, tetrahedra, etc.).  In this 
module, simplicial complexes in D-dimenional space are described with 
an (N,D) array of vertices and and (M,D) array describing the indices 
of vertices making up each simplex. As an example, the unit square in 
two dimensions can be described as collection of line segments:

>>> vertices = [[0.0,0.0],
                [1.0,0.0],
                [1.0,1.0],
                [0.0,1.0]]
>>> simplices = [[0,1],
                 [1,2],
                 [2,3],
                 [3,0]]

A three dimensional cube can similarly be described as a collection
of triangles:

>>> vertices = [[0.0,0.0,0.0],
                [0.0,0.0,1.0],
                [0.0,1.0,0.0],
                [0.0,1.0,1.0],
                [1.0,0.0,0.0],
                [1.0,0.0,1.0],
                [1.0,1.0,0.0],
                [1.0,1.0,1.0]]
>>> simplices = [[0,1,4],
                 [1,5,4],
                 [1,7,5],
                 [1,3,7],
                 [0,1,3],
                 [0,2,3],
                 [0,2,6],
                 [0,4,6],
                 [4,5,7],
                 [4,6,7],
                 [2,3,7],
                 [2,6,7]]
 
Although the notation is clumsy, a 1D domains can be described as a 
collection of vertices in a manner that is consistent with the above 
two examples:
   
>>> vertices = [[0.0],[1.0]]
>>> simplices = [[0],[1]]

This module is primarily use to find whether and where line segments 
intersect a simplicial complex and whether points are contained within 
a closed simplicial complex.  For example, one can determine whether a 
collection of points, saved as *points*, are contained within a 
simplicial complex, defined by *vertices* and *simplices* with the 
command

>>> contains(points,vertices,simplices)

which returns a boolean array.

One can find the number of times a collection of line segments, 
defined by *start_points* and *end_points*, intersect a simplicial 
complex with the command

>> intersection_count(start_points,end_points,vertices,simplices)

which returns an array of the number of simplexes intersections for
each segment. If it is known that a collection of line segments
intersect a simplicial complex then the intersection point can be
found with the command

>> intersection_point(start_points,end_points,vertices,simplices)
 
This returns an (N,D) array of intersection points where N is the 
number of line segments.  If a line segment does not intersect the 
simplicial complex then the above command returns a ValueError. If 
there are multiple intersections for a single segment then only the 
first detected intersection will be returned.

There are numerous other packages which can perform the same tasks 
as this module.  For example geos (http://trac.osgeo.org/geos/) and 
gts (http://gts.sourceforge.net/).  However, the python bindings for 
these packages are too slow for RBF purposes.
'''
# python imports
from __future__ import division
import numpy as np
from itertools import combinations
from scipy.special import factorial
# cython imports
cimport numpy as np
from cython cimport boundscheck,wraparound,cdivision
from cython.parallel import prange
from libc.stdlib cimport malloc,free
from libc.stdlib cimport rand


# NOTE: fabs is not the same as abs in C!!! 
cdef extern from "math.h":
  cdef float fabs(float x) nogil

cdef extern from "math.h":
  cdef float sqrt(float x) nogil

cdef extern from "limits.h":
    int RAND_MAX

## geometric data types
#####################################################################

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

cdef segment2d* allocate_segment2d_array(long n):
  ''' 
  This is used in *contains_2d* and *intersection_count_2d*
  '''
  cdef:
    segment2d* out = <segment2d*>malloc(n*sizeof(segment2d))

  if not out:
    raise MemoryError()

  return out  

cdef segment3d* allocate_segment3d_array(long n):
  ''' 
  This is used in *contains_3d* and *intersection_count_3d*
  '''
  cdef:
    segment3d* out = <segment3d*>malloc(n*sizeof(segment3d))
    
  if not out:
    raise MemoryError()

  return out  

## point in simplex functions
#####################################################################
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
    double l1,l2
  
  # find barycentric coordinates  
  l1 = (seg.a.x - vec.x)/(seg.a.x-seg.b.x)
  l2 = (vec.x - seg.b.x)/(seg.a.x-seg.b.x)
  if (l1>=0.0) & (l2>=0.0):
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
    double det,l1,l2,l3

  # find barycentric coordinates  
  det =  (tri.b.y - tri.c.y)*(tri.a.x-tri.c.x) + (tri.c.x-tri.b.x)*(tri.a.y-tri.c.y)
  l1 =  ((tri.b.y - tri.c.y)*(vec.x-tri.c.x) + (tri.c.x-tri.b.x)*(vec.y-tri.c.y))/det
  l2 =  ((tri.c.y - tri.a.y)*(vec.x-tri.c.x) + (tri.a.x-tri.c.x)*(vec.y-tri.c.y))/det
  l3 = 1 - l1 - l2
  if (l1>=0.0) & (l2>=0.0) & (l3>=0.0):
    return True
  else:
    return False


## simplex normals functions
#####################################################################
cdef vector2d segment_normal_2d(segment2d seg) nogil:
  ''' 
  Returns the vector normal to a 2d line segment
  '''
  cdef:
    vector2d out

  out.x = seg.b.y-seg.a.y
  out.y = -(seg.b.x-seg.a.x)
  return out
  

cdef vector3d triangle_normal_3d(triangle3d tri) nogil:
  ''' 
  Returns the vector normal to a 3d triangle
  '''
  cdef:
    vector3d out

  out.x =  ((tri.b.y-tri.a.y)*(tri.c.z-tri.a.z) - 
            (tri.b.z-tri.a.z)*(tri.c.y-tri.a.y))
  out.y = -((tri.b.x-tri.a.x)*(tri.c.z-tri.a.z) - 
            (tri.b.z-tri.a.z)*(tri.c.x-tri.a.x)) 
  out.z =  ((tri.b.x-tri.a.x)*(tri.c.y-tri.a.y) - 
            (tri.b.y-tri.a.y)*(tri.c.x-tri.a.x))
  return out


## find point outside simplicial complex
#####################################################################
@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef vector2d find_outside_2d(double[:,:] v) nogil:
  ''' 
  Finds a arbitrary point that is outside of a polygon defined by
  the given vertices
  '''
  cdef:
    unsigned int i
    vector2d out
    
  out.x = v[0,0]
  out.y = v[0,1]
  for i in range(1,v.shape[0]):
    if v[i,0] < out.x:
      out.x = v[i,0] 

    if v[i,1] < out.y:
      out.y = v[i,1] 

  out.x -= 1.23456789# + rand()*1.0/RAND_MAX
  out.y -= 2.34567891# + rand()*1.0/RAND_MAX

  return out


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef vector3d find_outside_3d(double[:,:] v) nogil:
  ''' 
  Finds a arbitrary point that is outside of a polyhedron defined by
  the given vertices
  '''
  cdef:
    unsigned int i
    vector3d out
    
  out.x = v[0,0]
  out.y = v[0,1]
  out.z = v[0,2]
  for i in range(1,v.shape[0]):
    if v[i,0] < out.x:
      out.x = v[i,0] 

    if v[i,1] < out.y:
      out.y = v[i,1] 

    if v[i,2] < out.z:
      out.z = v[i,2] 

  out.x -= 1.23456789# + rand()*1.0/RAND_MAX
  out.y -= 2.34567891# + rand()*1.0/RAND_MAX
  out.z -= 3.45678912# + rand()*1.0/RAND_MAX
  return out


## 2D point in polygon functions
#####################################################################
@cdivision(True)
cdef bint is_intersecting_2d(segment2d seg1,
                             segment2d seg2) nogil:
  ''' 
  Identifies whether two 2D segments intersect. An intersection is
  detected if both segments are not colinear and if any part of the
  two segments touch
  '''
  cdef:
    double proj1,proj2
    vector2d pnt
    vector2d n
    vector1d pnt_proj
    segment1d seg_proj

  # find the normal vector components for segment 2
  n = segment_normal_2d(seg2)

  # project both points in segment 1 onto the normal vector
  proj1 = ((seg1.a.x-seg2.a.x)*n.x +
           (seg1.a.y-seg2.a.y)*n.y)
  proj2 = ((seg1.b.x-seg2.a.x)*n.x +
           (seg1.b.y-seg2.a.y)*n.y)

  if proj1*proj2 > 0:
    return False

  # return false if the segments are collinear
  if (proj1 == 0) & (proj2 == 0):
    return False

  # find the point where segment 1 intersects the line overlapping 
  # segment 2 
  pnt.x = seg1.a.x + (proj1/(proj1-proj2))*(
          (seg1.b.x-seg1.a.x))
  pnt.y = seg1.a.y + (proj1/(proj1-proj2))*(
          (seg1.b.y-seg1.a.y))

  # if the normal x component is larger then compare y values
  if fabs(n.x) >= fabs(n.y):
    pnt_proj.x = pnt.y    
    seg_proj.a.x = seg2.a.y
    seg_proj.b.x = seg2.b.y
    return point_in_segment(pnt_proj,seg_proj)

  else:
    pnt_proj.x = pnt.x
    seg_proj.a.x = seg2.a.x
    seg_proj.b.x = seg2.b.x
    return point_in_segment(pnt_proj,seg_proj)


@boundscheck(False)
@wraparound(False)
cdef np.ndarray intersection_count_2d(double[:,:] start_pnts,
                                      double[:,:] end_pnts,
                                      double[:,:] vertices,
                                      long[:,:] simplices):
  ''' 
  Returns an array containing the number of simplices intersected 
  between start_pnts and end_pnts. This is parallelizable.
  '''
  cdef:
    int i
    int N = start_pnts.shape[0]
    long[:] out = np.empty((N,),dtype=int,order='c')
    segment2d* segs = allocate_segment2d_array(N)
    
  try:
    with nogil:
      for i in prange(N):
        segs[i].a.x = start_pnts[i,0]
        segs[i].a.y = start_pnts[i,1]
        segs[i].b.x = end_pnts[i,0]
        segs[i].b.y = end_pnts[i,1]
        out[i] = _intersection_count_2d(segs[i],vertices,simplices)

  finally:
    free(segs)
    
  return np.asarray(out,dtype=int)


@boundscheck(False)
@wraparound(False)
cdef int _intersection_count_2d(segment2d seg1,
                                double[:,:] vertices,
                                long[:,:] simplices) nogil:
  cdef:
    unsigned int i
    unsigned int count = 0
    segment2d seg2

  for i in range(simplices.shape[0]):
    seg2.a.x = vertices[simplices[i,0],0]
    seg2.a.y = vertices[simplices[i,0],1]
    seg2.b.x = vertices[simplices[i,1],0]
    seg2.b.y = vertices[simplices[i,1],1]
    if is_intersecting_2d(seg1,seg2):
      count += 1

  return count


@boundscheck(False)
@wraparound(False)
cdef np.ndarray intersection_index_2d(double[:,:] start_pnts,
                                      double[:,:] end_pnts,
                                      double[:,:] vertices,
                                      long[:,:] simplices):
  ''' 
  Returns an array identifying which simplex is intersected by
  start_pnts and end_pnts. 

  Notes
  -----
  if there is no intersection then a ValueError is returned. Since an 
  error could potentially be returned, this is not parallelizable.

  '''
  cdef:
    int i
    int N = start_pnts.shape[0]
    long[:] out = np.empty((N,),dtype=int,order='c')
    segment2d seg 
    
  for i in range(N):
    seg.a.x = start_pnts[i,0]
    seg.a.y = start_pnts[i,1]
    seg.b.x = end_pnts[i,0]
    seg.b.y = end_pnts[i,1]
    out[i] = _intersection_index_2d(seg,vertices,simplices)

  return np.asarray(out)


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef int _intersection_index_2d(segment2d seg1,
                                double[:,:] vertices,
                                long[:,:] simplices) except *:
  cdef:
    int i
    int out = -1
    double proj1,proj2,t
    double closest = 999.9
    segment2d seg2
    vector2d n
  
  for i in range(simplices.shape[0]):
    seg2.a.x = vertices[simplices[i,0],0]
    seg2.a.y = vertices[simplices[i,0],1]
    seg2.b.x = vertices[simplices[i,1],0]
    seg2.b.y = vertices[simplices[i,1],1]
    if is_intersecting_2d(seg1,seg2):
      # the intersecting segment should be the first segment 
      # intersected when going from seg1.a to seg1.b
      n = segment_normal_2d(seg2) 
      proj1 = ((seg1.a.x-seg2.a.x)*n.x +
               (seg1.a.y-seg2.a.y)*n.y)
      proj2 = ((seg1.b.x-seg2.a.x)*n.x +
               (seg1.b.y-seg2.a.y)*n.y)
      # t is a scalar between 0 and 1. If t=0 then the intersection is 
      # at seg1.a and if t=1 then the intersection is at seg1.b
      t = proj1/(proj1-proj2)
      if t < closest:
        closest = t
        out = i

  if out == -1:
    # out is -1 iff no intersection was found
    raise ValueError('No intersection found for segment [[%s,%s],[%s,%s]]' % 
                     (seg1.a.x,seg1.a.y,seg1.b.x,seg1.b.y))

  return out


@boundscheck(False)
@wraparound(False)
cdef np.ndarray intersection_point_2d(double[:,:] start_pnts,
                                      double[:,:] end_pnts,
                                      double[:,:] vertices,
                                      long[:,:] simplices):         
  ''' 
  Returns an array of intersection points between the line segments, 
  defined in terms of start_pnts and end_pnts, and the simplices.

  Notes
  -----
  if there is no intersection then a ValueError is returned.
  '''
  cdef:
    int i
    int N = start_pnts.shape[0]
    double[:,:] out = np.empty((N,2),dtype=float,order='c')
    segment2d seg
    vector2d vec

  for i in range(N):
    seg.a.x = start_pnts[i,0]
    seg.a.y = start_pnts[i,1]
    seg.b.x = end_pnts[i,0]
    seg.b.y = end_pnts[i,1]
    vec = _intersection_point_2d(seg,vertices,simplices)
    out[i,0] = vec.x
    out[i,1] = vec.y

  return np.asarray(out)


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef vector2d _intersection_point_2d(segment2d seg1,
                                     double[:,:] vertices,
                                     long[:,:] simplices) except *:
  cdef:
    int idx
    double proj1,proj2
    segment2d seg2
    vector2d out,n
    

  idx = _intersection_index_2d(seg1,vertices,simplices)
  seg2.a.x = vertices[simplices[idx,0],0]
  seg2.a.y = vertices[simplices[idx,0],1]
  seg2.b.x = vertices[simplices[idx,1],0]
  seg2.b.y = vertices[simplices[idx,1],1]

  n = segment_normal_2d(seg2) 

  proj1 = ((seg1.a.x-seg2.a.x)*n.x +
           (seg1.a.y-seg2.a.y)*n.y)
  proj2 = ((seg1.b.x-seg2.a.x)*n.x +
           (seg1.b.y-seg2.a.y)*n.y)

  out.x = seg1.a.x + (proj1/(proj1-proj2))*(seg1.b.x-seg1.a.x)
  out.y = seg1.a.y + (proj1/(proj1-proj2))*(seg1.b.y-seg1.a.y)

  return out


@boundscheck(False)
@wraparound(False)
cdef np.ndarray cross_normals_2d(double[:,:] start_pnts,
                                 double[:,:] end_pnts,
                                 double[:,:] vertices,
                                 long[:,:] simplices):         
  ''' 
  Returns an array of normal vectors to the simplices intersected by
  the line segments 

  Notes
  -----
  if there is not intersection then a ValueError is returned

  '''
  cdef:
    int i
    int N = start_pnts.shape[0]
    double[:,:] out = np.empty((N,2),dtype=float,order='c')
    segment2d seg
    vector2d vec

  for i in range(N):
    seg.a.x = start_pnts[i,0]
    seg.a.y = start_pnts[i,1]
    seg.b.x = end_pnts[i,0]
    seg.b.y = end_pnts[i,1]
    vec = _cross_normals_2d(seg,vertices,simplices)
    out[i,0] = vec.x
    out[i,1] = vec.y
    
  return np.asarray(out)  


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef vector2d _cross_normals_2d(segment2d seg1,
                                double[:,:] vertices,
                                long[:,:] simplices) except *:      
  cdef:
    double proj,mag
    int idx
    segment2d seg2
    vector2d n

  idx = _intersection_index_2d(seg1,vertices,simplices)
  seg2.a.x = vertices[simplices[idx,0],0]
  seg2.a.y = vertices[simplices[idx,0],1]
  seg2.b.x = vertices[simplices[idx,1],0]
  seg2.b.y = vertices[simplices[idx,1],1]

  n = segment_normal_2d(seg2)

  proj = ((seg1.b.x-seg2.a.x)*n.x +
          (seg1.b.y-seg2.a.y)*n.y)

  # This ensures that the normal vector points in the direction of seg1
  if proj < 0:
    n.x *= -1
    n.y *= -1

  # normalize the normal vector to 1
  mag = sqrt(n.x**2 + n.y**2)
  n.x /= mag
  n.y /= mag

  return n


@boundscheck(False)
@wraparound(False)
cdef np.ndarray contains_2d(double[:,:] pnt,
                            double[:,:] vertices,
                            long[:,:] simplices):
  ''' 
  Returns a boolean array identifying which points are contained in 
  the closed simplicial complex described by vertices and simplices
  '''
  cdef:
    int count,i
    int N = pnt.shape[0]
    long[:] out = np.empty((N,),dtype=int,order='c') 
    segment2d* segs = allocate_segment2d_array(N)
    vector2d outside_pnt

  try:
    with nogil:
      outside_pnt = find_outside_2d(vertices)
      for i in prange(N):
        segs[i].a.x = outside_pnt.x
        segs[i].a.y = outside_pnt.y
        segs[i].b.x = pnt[i,0]
        segs[i].b.y = pnt[i,1]
        count = _intersection_count_2d(segs[i],vertices,simplices)
        out[i] = count%2

  finally:
    free(segs)
    
  return np.asarray(out,dtype=bool)


## 3D point in polygon functions
#####################################################################
@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef bint is_intersecting_3d(segment3d seg,
                             triangle3d tri) nogil:
  ''' 
  Returns True if the 3D segment intersects the 3D triangle. An 
  intersection is detected if the segment and triangle are not 
  coplanar and if any part of the segment touches the triangle at an 
  edge or in the interior.

  Notes
  -----
  This function determines where the segment intersects the plane
  containing the triangle and then projects the intersection point
  and triangle into a 2D plane where the point is then tested if it
  is within the triangle.

  '''
  cdef:
    vector3d pnt,n
    vector2d pnt_proj
    triangle2d tri_proj
    double proj1,proj2

  # find triangle normal vector components
  n = triangle_normal_3d(tri)

  proj1 = ((seg.a.x-tri.a.x)*n.x + 
           (seg.a.y-tri.a.y)*n.y +
           (seg.a.z-tri.a.z)*n.z)
  proj2 = ((seg.b.x-tri.a.x)*n.x + 
           (seg.b.y-tri.a.y)*n.y +
           (seg.b.z-tri.a.z)*n.z)

  if proj1*proj2 > 0:
    return False

  # coplanar segments will always return false
  # There is a possibility that the segment touches
  # one point on the triangle
  if (proj1 == 0) & (proj2 == 0):
    return False

  # intersection point
  pnt.x = seg.a.x + (proj1/(proj1-proj2))*(seg.b.x-seg.a.x)
  pnt.y = seg.a.y + (proj1/(proj1-proj2))*(seg.b.y-seg.a.y)
  pnt.z = seg.a.z + (proj1/(proj1-proj2))*(seg.b.z-seg.a.z)

  if (fabs(n.x) >= fabs(n.y)) & (fabs(n.x) >= fabs(n.z)):
    pnt_proj.x = pnt.y
    pnt_proj.y = pnt.z
    tri_proj.a.x = tri.a.y
    tri_proj.a.y = tri.a.z
    tri_proj.b.x = tri.b.y
    tri_proj.b.y = tri.b.z
    tri_proj.c.x = tri.c.y
    tri_proj.c.y = tri.c.z
    return point_in_triangle(pnt_proj,tri_proj)

  elif (fabs(n.y) >= fabs(n.x)) & (fabs(n.y) >= fabs(n.z)):
    pnt_proj.x = pnt.x
    pnt_proj.y = pnt.z
    tri_proj.a.x = tri.a.x
    tri_proj.a.y = tri.a.z
    tri_proj.b.x = tri.b.x
    tri_proj.b.y = tri.b.z
    tri_proj.c.x = tri.c.x
    tri_proj.c.y = tri.c.z
    return point_in_triangle(pnt_proj,tri_proj)

  elif (fabs(n.z) >= fabs(n.x)) & (fabs(n.z) >= fabs(n.y)):
    pnt_proj.x = pnt.x
    pnt_proj.y = pnt.y
    tri_proj.a.x = tri.a.x
    tri_proj.a.y = tri.a.y
    tri_proj.b.x = tri.b.x
    tri_proj.b.y = tri.b.y
    tri_proj.c.x = tri.c.x
    tri_proj.c.y = tri.c.y
    return point_in_triangle(pnt_proj,tri_proj)


@boundscheck(False)
@wraparound(False)
cdef np.ndarray intersection_count_3d(double[:,:] start_pnts,
                                      double[:,:] end_pnts,                         
                                      double[:,:] vertices,
                                      long[:,:] simplices):
  ''' 
  Returns an array of the number of intersections between each line
  segment, described by start_pnts and end_pnts, and the simplices
  '''
  cdef:
    int i
    int N = start_pnts.shape[0]
    long[:] out = np.empty((N,),dtype=int,order='c')
    segment3d* segs = allocate_segment3d_array(N)

  try:
    with nogil:
      for i in prange(N):
        segs[i].a.x = start_pnts[i,0]
        segs[i].a.y = start_pnts[i,1]
        segs[i].a.z = start_pnts[i,2]
        segs[i].b.x = end_pnts[i,0]
        segs[i].b.y = end_pnts[i,1]
        segs[i].b.z = end_pnts[i,2]
        out[i] = _intersection_count_3d(segs[i],vertices,simplices)

  finally:
    free(segs)
    
  return np.asarray(out)  


@boundscheck(False)
@wraparound(False)
cdef int _intersection_count_3d(segment3d seg,
                                double[:,:] vertices,
                                long[:,:] simplices) nogil:
  cdef:
    unsigned int i
    unsigned int count = 0
    triangle3d tri

  for i in range(simplices.shape[0]):
    tri.a.x = vertices[simplices[i,0],0]
    tri.a.y = vertices[simplices[i,0],1]
    tri.a.z = vertices[simplices[i,0],2]
    tri.b.x = vertices[simplices[i,1],0]
    tri.b.y = vertices[simplices[i,1],1]
    tri.b.z = vertices[simplices[i,1],2]
    tri.c.x = vertices[simplices[i,2],0]
    tri.c.y = vertices[simplices[i,2],1]
    tri.c.z = vertices[simplices[i,2],2]
    if is_intersecting_3d(seg,tri):
      count += 1

  return count


@boundscheck(False)
@wraparound(False)
cdef np.ndarray intersection_index_3d(double[:,:] start_pnts,
                                      double[:,:] end_pnts,                         
                                      double[:,:] vertices,
                                      long[:,:] simplices):
  ''' 
  Returns an array identifying which simplex is intersected by
  start_pnts and end_pnts. 

  Notes
  -----
  if there is no intersection then a ValueError is returned.

  '''

  cdef:
    int i
    int N = start_pnts.shape[0]
    long[:] out = np.empty((N,),dtype=int,order='c')
    segment3d seg

  for i in range(N):
    seg.a.x = start_pnts[i,0]
    seg.a.y = start_pnts[i,1]
    seg.a.z = start_pnts[i,2]
    seg.b.x = end_pnts[i,0]
    seg.b.y = end_pnts[i,1]
    seg.b.z = end_pnts[i,2]
    out[i] = _intersection_index_3d(seg,vertices,simplices)
    
  return np.asarray(out)  


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef int _intersection_index_3d(segment3d seg,
                                double[:,:] vertices,
                                long[:,:] simplices) except *:         
  cdef:
    int i
    int out = -1
    int N = simplices.shape[0]
    triangle3d tri
    double proj1,proj2,t
    double closest = 999.9
    vector3d n
    
  for i in range(N):
    tri.a.x = vertices[simplices[i,0],0]
    tri.a.y = vertices[simplices[i,0],1]
    tri.a.z = vertices[simplices[i,0],2]
    tri.b.x = vertices[simplices[i,1],0]
    tri.b.y = vertices[simplices[i,1],1]
    tri.b.z = vertices[simplices[i,1],2]
    tri.c.x = vertices[simplices[i,2],0]
    tri.c.y = vertices[simplices[i,2],1]
    tri.c.z = vertices[simplices[i,2],2]
    if is_intersecting_3d(seg,tri):
      n = triangle_normal_3d(tri)
      proj1 = ((seg.a.x-tri.a.x)*n.x + 
               (seg.a.y-tri.a.y)*n.y +
               (seg.a.z-tri.a.z)*n.z)
      proj2 = ((seg.b.x-tri.a.x)*n.x + 
               (seg.b.y-tri.a.y)*n.y +
               (seg.b.z-tri.a.z)*n.z)
      # t is a scalar between 0 and 1. If t=0 then the intersection is 
      # at seg1.a and if t=1 then the intersection is at seg1.b
      t = proj1/(proj1-proj2)
      if t < closest:
        closest = t
        out = i

  if out == -1:
    # out is -1 iff it has never been changed and no intersection was found
    raise ValueError('No intersection found for segment [[%s,%s,%s],[%s,%s,%s]]' % 
                     (seg.a.x,seg.a.y,seg.a.z,seg.b.x,seg.b.y,seg.b.z))

  return out


@boundscheck(False)
@wraparound(False)
cdef np.ndarray intersection_point_3d(double[:,:] start_pnts,
                                      double[:,:] end_pnts,
                                      double[:,:] vertices,
                                      long[:,:] simplices):         
  ''' 
  Returns the intersection points between the line segments,
  described by start_pnts and end_pnts, and the simplices

  Notes
  -----
  if there is no intersection then a ValueError is returned.

  '''
  cdef:
    int i
    int N = start_pnts.shape[0]
    double[:,:] out = np.empty((N,3),dtype=float,order='c')
    vector3d vec
    segment3d seg

  for i in range(N):
    seg.a.x = start_pnts[i,0]
    seg.a.y = start_pnts[i,1]
    seg.a.z = start_pnts[i,2]
    seg.b.x = end_pnts[i,0]
    seg.b.y = end_pnts[i,1]
    seg.b.z = end_pnts[i,2]
    vec = _intersection_point_3d(seg,vertices,simplices)
    out[i,0] = vec.x
    out[i,1] = vec.y
    out[i,2] = vec.z

  return np.asarray(out)


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef vector3d _intersection_point_3d(segment3d seg,
                                     double[:,:] vertices,
                                     long[:,:] simplices) except *:         
  cdef:
    double proj1,proj2
    int idx
    vector3d n
    triangle3d tri
    vector3d out 

  idx = _intersection_index_3d(seg,vertices,simplices)
  tri.a.x = vertices[simplices[idx,0],0]
  tri.a.y = vertices[simplices[idx,0],1]
  tri.a.z = vertices[simplices[idx,0],2]
  tri.b.x = vertices[simplices[idx,1],0]
  tri.b.y = vertices[simplices[idx,1],1]
  tri.b.z = vertices[simplices[idx,1],2]
  tri.c.x = vertices[simplices[idx,2],0]
  tri.c.y = vertices[simplices[idx,2],1]
  tri.c.z = vertices[simplices[idx,2],2]

  n = triangle_normal_3d(tri)

  proj1 = ((seg.a.x-tri.a.x)*n.x +
           (seg.a.y-tri.a.y)*n.y +
           (seg.a.z-tri.a.z)*n.z)
  proj2 = ((seg.b.x-tri.a.x)*n.x +
           (seg.b.y-tri.a.y)*n.y +
           (seg.b.z-tri.a.z)*n.z)

  out.x = seg.a.x + (proj1/(proj1-proj2))*(seg.b.x-seg.a.x)
  out.y = seg.a.y + (proj1/(proj1-proj2))*(seg.b.y-seg.a.y)
  out.z = seg.a.z + (proj1/(proj1-proj2))*(seg.b.z-seg.a.z)

  return out


@boundscheck(False)
@wraparound(False)
cdef np.ndarray cross_normals_3d(double[:,:] start_pnts,
                                 double[:,:] end_pnts,
                                 double[:,:] vertices,
                                 long[:,:] simplices):
  ''' 
  Returns the normal vectors to the simplices intersected by 
  start_pnts and end_pnts

  Notes
  -----
  if there is no intersection then a ValueError is returned.

  '''

  cdef:
    int i
    int N = start_pnts.shape[0]
    double[:,:] out = np.empty((N,3),dtype=float,order='c')
    segment3d seg
    vector3d vec

  for i in range(N):
    seg.a.x = start_pnts[i,0]
    seg.a.y = start_pnts[i,1]
    seg.a.z = start_pnts[i,2]
    seg.b.x = end_pnts[i,0]
    seg.b.y = end_pnts[i,1]
    seg.b.z = end_pnts[i,2]
    vec = _cross_normals_3d(seg,vertices,simplices)
    out[i,0] = vec.x
    out[i,1] = vec.y
    out[i,2] = vec.z

  return np.asarray(out)


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef vector3d _cross_normals_3d(segment3d seg,
                                double[:,:] vertices,
                                long[:,:] simplices) except *:         
  cdef:
    double proj,mag
    int idx
    triangle3d tri
    vector3d n

  idx = _intersection_index_3d(seg,vertices,simplices)
  tri.a.x = vertices[simplices[idx,0],0]
  tri.a.y = vertices[simplices[idx,0],1]
  tri.a.z = vertices[simplices[idx,0],2]
  tri.b.x = vertices[simplices[idx,1],0]
  tri.b.y = vertices[simplices[idx,1],1]
  tri.b.z = vertices[simplices[idx,1],2]
  tri.c.x = vertices[simplices[idx,2],0]
  tri.c.y = vertices[simplices[idx,2],1]
  tri.c.z = vertices[simplices[idx,2],2]

  n = triangle_normal_3d(tri)

  proj = ((seg.b.x-tri.a.x)*n.x +
          (seg.b.y-tri.a.y)*n.y +
          (seg.b.z-tri.a.z)*n.z)

  # ensures that normal points in the direction of the segment
  if proj < 0:
    n.x *= -1
    n.y *= -1
    n.z *= -1

  mag = sqrt(n.x**2 + n.y**2 + n.z**2)
  n.x /= mag
  n.y /= mag
  n.z /= mag

  return n


@boundscheck(False)
@wraparound(False)
cdef np.ndarray contains_3d(double[:,:] pnt,
                            double[:,:] vertices,
                            long[:,:] simplices):
  ''' 
  Returns a boolean array identifying whether the points are contained 
  within the closed simplicial complex described by vertices and 
  simplices
  '''
  cdef:
    int count,i
    int N = pnt.shape[0]
    long[:] out = np.empty((N,),dtype=int,order='c') 
    segment3d* segs = allocate_segment3d_array(N)
    vector3d outside_pnt

  try:
    with nogil:
      outside_pnt = find_outside_3d(vertices)
      for i in prange(N):
        segs[i].a.x = outside_pnt.x
        segs[i].a.y = outside_pnt.y
        segs[i].a.z = outside_pnt.z
        segs[i].b.x = pnt[i,0]
        segs[i].b.y = pnt[i,1]
        segs[i].b.z = pnt[i,2]
        count = _intersection_count_3d(segs[i],vertices,simplices)
        out[i] = count%2
  
  finally:
    free(segs)
    
  return np.asarray(out,dtype=bool)

## end-user functions
#####################################################################
def intersection_point(start_points,end_points,vertices,simplices):
  ''' 
  Returns the intersection points between line segments and a 
  simplicial complex.  The line segments are described by 
  *start_points* and *end_points*, and the simplicial complex is 
  described by *vertices* and *simplices*. This function works for 1, 
  2, and 3 spatial dimensions.

  Parameters
  ----------
  start_points : (N,D) array
    Vertices describing one end of the line segments. *N* is the 
    number of line segments and *D* is the number of dimensions

  end_points : (N,D) array 
    Vertices describing the other end of the line segments. 

  vertices : (M,D) array 
    Vertices within the simplicial complex. M is the number of 
    vertices.

  simplices : (P,D) array
    Connectivity of the vertices. Each row contains the vertex 
    indices which form one simplex of the simplicial complex

  Returns
  -------
  out : (N,D) array
    intersection points    

  Notes
  -----
  This function fails when a intersection is not found for a line 
  segment. If there are multiple intersections then the intersection 
  closest to start_point is used.

  '''
  start_points = np.asarray(start_points,dtype=float)
  end_points = np.asarray(end_points,dtype=float)
  vertices = np.asarray(vertices,dtype=float)
  simplices = np.asarray(simplices,dtype=int)
  if not (start_points.shape[1] == end_points.shape[1]):
    raise ValueError('inconsistent spatial dimensions')

  if not (start_points.shape[1] == vertices.shape[1]):
    raise ValueError('inconsistent spatial dimensions')

  if not (start_points.shape[0] == end_points.shape[0]):
    raise ValueError('start points and end points must have the same length')

  dim = start_points.shape[1]
  if dim == 1:
    crossed_idx = intersection_index(start_points,end_points,vertices,simplices)
    vert = vertices[simplices[:,0]]
    out = vert[crossed_idx]

  elif dim == 2:
    out = intersection_point_2d(start_points,end_points,vertices,simplices)

  elif dim == 3:
    out = intersection_point_3d(start_points,end_points,vertices,simplices)

  else:
    raise ValueError(
      'intersections can only be found for a 1, 2, or 3 dimensional '
      'simplicial complex')
      
  return out


def intersection_normal(start_points,end_points,vertices,simplices):
  ''' 
  Returns the normal vectors to the simplices intersected by the line 
  segments. The line segments are described by *start_points* and 
  *end_points*. This function works for 1, 2, and 3 spatial 
  dimensions. The normal vector is in the direction that points 
  towards end_points

  Parameters
  ----------
  start_points : (N,D) array
    Vertices describing one end of the line segments. *N* is the 
    number of line segments and *D* is the number of dimensions.

  end_points : (N,D) array 
    Vertices describing the other end of the line segments.

  vertices : (M,D) array 
    Vertices within the simplicial complex. *M* is the number of 
    vertices

  simplices : (P,D) array
    Connectivity of the vertices. Each row contains the vertex 
    indices which form one simplex of the simplicial complex

  Returns
  -------
  out : (N,D) array
    normal vectors

  Notes
  -----
  This function fails when a intersection is not found for a line 
  segment. If there are multiple intersections then the intersection 
  closest to start_point is used.

  '''
  start_points = np.asarray(start_points,dtype=float)
  end_points = np.asarray(end_points,dtype=float)
  vertices = np.asarray(vertices,dtype=float)
  simplices = np.asarray(simplices,dtype=int)
  if not (start_points.shape[1] == end_points.shape[1]):
    raise ValueError('inconsistent spatial dimensions')

  if not (start_points.shape[1] == vertices.shape[1]): 
    raise ValueError('inconsistent spatial dimensions')

  if not (start_points.shape[0] == end_points.shape[0]):
    raise ValueError('start points and end points must have the same length')

  dim = start_points.shape[1]
  if dim == 1:
    out = np.ones(start_points.shape,dtype=float)
    crossed_idx = intersection_index(start_points,end_points,vertices,simplices)
    vert = vertices[simplices[:,0]]
    crossed_vert = vert[crossed_idx]
    out[crossed_vert < start_points] = -1.0

  elif dim == 2:
    out = cross_normals_2d(start_points,end_points,vertices,simplices)

  elif dim == 3:
    out = cross_normals_3d(start_points,end_points,vertices,simplices)

  else:
    raise ValueError(
      'intersections can only be found for a 1, 2, or 3 dimensional '
      'simplicial complex')

  return out


def intersection_index(start_points,end_points,vertices,simplices):
  ''' 
  Returns the indices of the simplices intersected by the line 
  segments. The line segments are described by *start_points* and 
  *end_points*. This function works for 1, 2, and 3 spatial 
  dimensions. 

  Parameters
  ----------
  start_points : (N,D) array
    Vertices describing one end of the line segments. *N* is the 
    number of line segments and *D* is the number of dimensions

  end_points : (N,D) array 
    Vertices describing the other end of the line segments

  vertices : (M,D) array 
    Vertices within the simplicial complex. *M* is the number of 
    vertices

  simplices : (P,D) array
    Connectivity of the vertices. Each row contains the vertex 
    indices which form one simplex of the simplicial complex

  Returns
  -------
  out : (N,) int array
    simplex indices 

  Notes
  -----
  This function fails when a intersection is not found for a line
  segment. If there are multiple intersections then the intersection 
  closest to *start_point* is used.

  '''
  start_points = np.asarray(start_points,dtype=float)
  end_points = np.asarray(end_points,dtype=float)
  vertices = np.asarray(vertices,dtype=float)
  simplices = np.asarray(simplices,dtype=int)
  if not (start_points.shape[1] == end_points.shape[1]): 
    raise ValueError('inconsistent spatial dimesions')

  if not (start_points.shape[1] == vertices.shape[1]): 
    raise ValueError('inconsistent spatial dimensions')

  if not (start_points.shape[0] == end_points.shape[0]): 
    raise ValueError('start points and end points must have the same length')

  dim = start_points.shape[1]
  if dim == 1:
    out = np.zeros(start_points.shape[0],dtype=int)
    vert = vertices[simplices[:,0]]
    proj1 = (start_points-vert.T) 
    proj2 = (end_points-vert.T) 
    # identify intersections in a manner that is consistent with the 
    # 2d and 3d method
    crossed_bool = (proj1*proj2 <= 0.0) & ~((proj1 == 0.0) & (proj2 == 0.0))
    for i in range(start_points.shape[0]):
      # indices of all simplices crossed for segment i
      crossed_idx, = np.nonzero(crossed_bool[i])
      proj1i = proj1[i,crossed_idx]
      proj2i = proj2[i,crossed_idx]
      # find the intersection closest to start_point i
      idx = np.argmin(proj1i/(proj1i-proj2i))
      out[i] = crossed_idx[idx]

  elif dim == 2:
    out = intersection_index_2d(start_points,end_points,vertices,simplices)

  elif dim == 3:
    out = intersection_index_3d(start_points,end_points,vertices,simplices)

  else:
    raise ValueError(
      'intersections can only be found for a 1, 2, or 3 dimensional '
      'simplicial complex')

  return out


def intersection_count(start_points,end_points,vertices,simplices):
  ''' 
  Returns the number of simplices crossed by the line segments. The 
  line segments are described by *start_points* and *end_points*. This 
  function works for 1, 2, and 3 spatial dimensions.

  Parameters
  ----------
  start_points : (N,D) array
    Vertices describing one end of the line segments. *N* is the 
    number of line segments and *D* is the number of dimensions

  end_points : (N,D) array 
    Vertices describing the other end of the line segments

  vertices : (M,D) array 
    Vertices within the simplicial complex. *M* is the number of 
    vertices

  simplices : (P,D) array
    Connectivity of the vertices. Each row contains the vertex 
    indices which form one simplex of the simplicial complex

  Returns
  -------
  out : (N,) int array
    intersection counts

  '''
  start_points = np.asarray(start_points,dtype=float)
  end_points = np.asarray(end_points,dtype=float)
  vertices = np.asarray(vertices,dtype=float)
  simplices = np.asarray(simplices,dtype=int)
  if not (start_points.shape[1] == end_points.shape[1]): 
    raise ValueError('inconsistent spatial dimensions')

  if not (start_points.shape[1] == vertices.shape[1]): 
    raise ValueError('inconsistent spatial dimensions')

  if not (start_points.shape[0] == end_points.shape[0]):
    raise ValueError('start points and end points must have the same length')

  dim = start_points.shape[1]
  if dim == 1:
    vert = vertices[simplices[:,0]]
    proj1 = (start_points-vert.T) 
    proj2 = (end_points-vert.T) 
    crossed_bool = (proj1*proj2 <= 0.0) & ~((proj1 == 0.0) & (proj2 == 0.0))
    out = np.sum(crossed_bool,axis=1)

  elif dim == 2:
    out = intersection_count_2d(start_points,end_points,vertices,simplices)

  elif dim == 3:
    out = intersection_count_3d(start_points,end_points,vertices,simplices)

  else:
    raise ValueError(
      'intersections can only be found for a 1, 2, or 3 dimensional '
      'simplicial complex')

  return out


def contains(points,vertices,simplices):
  ''' 
  Returns a boolean array identifying whether the points are contained 
  within a closed simplicial complex. The simplicial complex is 
  described by *vertices* and *simplices*. This function works for 1, 
  2, and 3 spatial dimensions.

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
  points = np.asarray(points)
  vertices = np.asarray(vertices)
  simplices = np.asarray(simplices)
  dim = points.shape[1]
  if not (points.shape[1] == vertices.shape[1]):
    raise ValueError('inconsistent spatial dimensions')

  if dim == 1:
    vert = vertices[simplices[:,0]]
    outside_point = np.min(vert,axis=0) - 1.0
    end_points = outside_point[:,None].repeat(points.shape[0],axis=0)
    proj1 = (points-vert.T) 
    proj2 = (end_points-vert.T) 
    crossed_bool = (proj1*proj2 <= 0.0) & ~((proj1 == 0.0) & (proj2 == 0.0))
    crossed_count = np.sum(crossed_bool,axis=1)
    out = np.array(crossed_count%2,dtype=bool)

  elif dim == 2:
    out = contains_2d(points,vertices,simplices)

  elif dim == 3:
    out = contains_3d(points,vertices,simplices)

  else:
    raise ValueError(
      'point in polygon tests can only be done for a 1, 2, or 3 '
      'dimensional simplicial complex')

  return out


def simplex_normals(vert,smp):
  ''' 
  Returns the normal vectors for each simplex. Orientation is 
  determined by the right hand rule

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
      
  Notes
  -----
  This is only defined for two and three dimensional simplices

  '''
  vert = np.asarray(vert,dtype=float)
  smp = np.asarray(smp,dtype=int)

  # spatial dimensions        
  dim = vert.shape[1]
  
  if (dim != 2) & (dim != 3):
    raise ValueError('simplicial complex must be 2 or 3 dimensional')

  # Create a N by D-1 by D matrix    
  M = vert[smp[:,1:]] - vert[smp[:,[0]]]

  Msubs = [np.delete(M,i,-1) for i in range(dim)]
  out = np.linalg.det(Msubs)
  out[1::2] *= -1
  out = np.rollaxis(out,-1)
  out /= np.linalg.norm(out,axis=-1)[...,None]
  return out


def simplex_outward_normals(vert,smp):
  ''' 
  Returns the outward normal vectors for each simplex. The sign of the 
  returned vectors are only meaningful if the simplices enclose an 
  area in two-dimensional space or a volume in three-dimensional space

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
      
  Notes
  -----
  This is only defined for two and three dimensional simplices

  '''
  vert = np.asarray(vert,dtype=float)
  smp = np.asarray(smp,dtype=int)
  
  # spatial dimensions        
  dim = vert.shape[1]
  
  if (dim != 2) & (dim != 3):
    raise ValueError('simplicial complex must be 2 or 3 dimensional')

  smp = oriented_simplices(vert,smp)
  return simplex_normals(vert,smp)


def simplex_upward_normals(vert,smp):
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
      
  Notes
  -----
  This is only defined for two and three dimensional simplices

  '''
  vert = np.asarray(vert,dtype=float)
  smp = np.asarray(smp,dtype=int)
  
  # spatial dimensions        
  dim = vert.shape[1]
  
  if (dim != 2) & (dim != 3):
    raise ValueError('simplicial complex must be 2 or 3 dimensional')

  out = simplex_normals(vert,smp)
  out[out[:,-1]<0] *= -1
  return out


def oriented_simplices(vert,smp):
  ''' 
  Returns simplex indices that are ordered such that each simplex 
  normal vector, as defined by the right hand rule, points outward
                                    
  Parameters
  ----------
  vertices : (M,D) array
    Vertices within the simplicial complex

  simplices : (P,D) int array 
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
  vert = np.asarray(vert,dtype=float)
  smp = np.array(smp,dtype=int,copy=True)

  dim = vert.shape[1]

  # if 1D then return smp
  if dim == 1:
    return np.copy(smp)
  
  # length scale of the domain
  scale = np.max(vert)-np.min(vert)
  dx = 1e-10*scale

  # find the normal for each simplex    
  norms = simplex_normals(vert,smp)

  # find the centroid for each simplex      
  points = np.mean(vert[smp],axis=1)

  # push points in the direction of the normals  
  points += dx*norms

  # find which simplices are oriented such that their normals point  
  # inside                           
  faces_inside = contains(points,vert,smp)

  flip_smp = smp[faces_inside]
  flip_smp[:,[0,1]] = flip_smp[:,[1,0]]

  smp[faces_inside] = flip_smp
  return smp


def enclosure(vert,smp,orient=True):
  ''' 
  Returns the volume of a polyhedra, area of a polygon, or length of
  a segment enclosed by the simplicial complex

  Parameters
  ----------
  vertices : (M,D) array
    Vertices within the simplicial complex

  simplices : (P,D) int array 
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
  vert = np.array(vert,dtype=float,copy=True)
  smp = np.asarray(smp,dtype=int)
  dim = smp.shape[1]
  if orient:
    smp = oriented_simplices(vert,smp)

  # If 1D vertices and simplices are given then sort the vertices
  # and sum the length between alternating pairs of vertices
  if dim == 1:
    vert = np.sort(vert[smp].flatten())
    vert[::2] *= -1
    return abs(np.sum(vert))

  # center the vertices for the purpose of numerical stability
  vert -= np.mean(vert,axis=0)
  signed_volumes = (1.0/factorial(dim))*np.linalg.det(vert[smp])
  volume = np.sum(signed_volumes)
  return volume


