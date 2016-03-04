# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
'''
Description 
----------- 
  Defines functions for basic computational geometry in 1, 2, and 3
  dimensions. This modules requires all volumes, surfaces and segments
  to be described as simplicial complexes, that is, as a collection of
  simplexes defined by their vertices.  Most end user functions in
  this module have a vertices and simplices argument, the former is a
  (N,D) collection of all D dimensional vertices in the simplicial
  complex and the latter is an (M,D) array of vertex indices making up
  each simplex. For example the unit square in two dimensions can be
  described as collection of line segments:

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
  intersect a simplicial complex and whether points are contained
  within a closed simplicial complex.  For example, one can determine
  whether a collection of points, saved as 'points', are contained
  within a simplicial complex, defined by 'vertices' and 'simplices'
  with the command

  >>> contains(points,vertices,simplices)

  which returns a boolean array.

  One can find the number of times a collection of line segments,
  defined by 'start_points' and 'end_points', intersect a simplicial
  complex with the command

  >> cross_count(start_points,end_points,vertices,simplices)

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

  Note
  ----
    There are numerous other packages which can perform the same tasks
    as this module.  For example geos (http://trac.osgeo.org/geos/)
    and gts (http://gts.sourceforge.net/).  However, the python
    bindings for these packages are too slow for RBF purposes.

'''
import numpy as np
cimport numpy as np
from cython cimport boundscheck,wraparound,cdivision
from cython.parallel import prange
from libc.stdlib cimport rand
from libc.stdlib cimport malloc,free
from itertools import combinations
from scipy.special import factorial

# NOTE: fabs is not the same as abs in C!!! 
cdef extern from "math.h":
  cdef float fabs(float x) nogil

cdef extern from "math.h":
  cdef float sqrt(float x) nogil

cdef extern from "limits.h":
    int RAND_MAX

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


cdef double min2(double a, double b) nogil:
  if a <= b:
    return a
  else:
    return b

cdef double max2(double a, double b) nogil:
  if a >= b:
    return a
  else:
    return b

cdef double min3(double a, double b, double c) nogil:
  if (a <= b) & (a <= c):
    return a

  if (b <= a) & (b <= c):
    return b

  if (c <= a) & (c <= b):
    return c

@cdivision(True)
cdef bint point_in_segment(vector1d vec, segment1d seg) nogil:  
  '''
  identifies whether a point in 1d space is within a segment
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
  identifies whether a point in 2d space is within a triangle by
  converting the point to barycentric coordinates
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


cdef vector2d segment_normal_2d(segment2d seg) nogil:
  '''
  returns the vector normal to a 2d line segment
  '''
  cdef:
    vector2d out

  out.x = seg.b.y-seg.a.y
  out.y = -(seg.b.x-seg.a.x)
  return out
  

cdef vector3d triangle_normal_3d(triangle3d tri) nogil:
  '''
  returns the vector normal to a 3d triangle
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

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef vector2d find_outside_2d(double[:,:] v) nogil:
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


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef bint is_intersecting_2d(segment2d seg1,
                             segment2d seg2) nogil:
  '''
  Description
  -----------
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
cdef np.ndarray cross_count_2d(double[:,:] start_pnts,
                               double[:,:] end_pnts,
                               double[:,:] vertices,
                               long[:,:] simplices):
  '''
  Description
  -----------
    returns an array containing the number of simplexes intersected
    between start_pnts and end_pnts.

  '''
  cdef:
    int i
    int N = start_pnts.shape[0]
    long[:] out = np.empty((N,),dtype=int,order='c')
    segment2d seg
    
  # This can be parallelized with prange
  #for i in prange(N):
  for i in range(N):
    seg.a.x = start_pnts[i,0]
    seg.a.y = start_pnts[i,1]
    seg.b.x = end_pnts[i,0]
    seg.b.y = end_pnts[i,1]
    out[i] = _cross_count_2d(seg,vertices,simplices)


  return np.asarray(out,dtype=int)


@boundscheck(False)
@wraparound(False)
cdef int _cross_count_2d(segment2d seg1,
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
  Description
  -----------
    returns an array identifying which simplex is intersected by
    start_pnts and end_pnts. 

  Note
  ----
    if there is no intersection then a ValueError is returned.

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
cdef int _intersection_index_2d(segment2d seg1,
                                double[:,:] vertices,
                                long[:,:] simplices) except *:
  cdef:
    int i
    segment2d seg2
  
  for i in range(simplices.shape[0]):
    seg2.a.x = vertices[simplices[i,0],0]
    seg2.a.y = vertices[simplices[i,0],1]
    seg2.b.x = vertices[simplices[i,1],0]
    seg2.b.y = vertices[simplices[i,1],1]
    if is_intersecting_2d(seg1,seg2):
      return i

  raise ValueError('No intersection found for segment [[%s,%s],[%s,%s]]' % 
                   (seg1.a.x,seg1.a.y,seg1.b.x,seg1.b.y))


@boundscheck(False)
@wraparound(False)
cdef np.ndarray intersection_point_2d(double[:,:] start_pnts,
                                      double[:,:] end_pnts,
                                      double[:,:] vertices,
                                      long[:,:] simplices):         
  '''
  Description
  -----------
    returns an array of intersection points between line segments,
    defined in terms of start_pnts and end_pnts, and simplices. 

  Note
  ----
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
  Description
  -----------
    returns an array of normal vectors to the simplices intersected by
    the line segments described in terms of start_pnts and end_pnts

p  Note
  ----
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
  Description
  -----------
    returns a boolean array identifying which points are contained in
    the closed simplicial complex described by vertices and simplices

  '''
  cdef:
    int count,i
    int N = pnt.shape[0]
    long[:] out = np.empty((N,),dtype=int,order='c') 
    segment2d seg
    vector2d outside_pnt

  outside_pnt = find_outside_2d(vertices)
  for i in range(N):
    seg.a.x = outside_pnt.x
    seg.a.y = outside_pnt.y
    seg.b.x = pnt[i,0]
    seg.b.y = pnt[i,1]
    count = _cross_count_2d(seg,vertices,simplices)
    out[i] = count%2

  return np.asarray(out,dtype=bool)


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef bint is_intersecting_3d(segment3d seg,
                             triangle3d tri) nogil:
  '''
  Description
  ----------- 
    returns True if the 3D segment intersects the 3D triangle. An
    intersection is detected if the segment and triangle are not
    coplanar and if any part of the segment touches the triangle at an
    edge or in the interior. Intersections at corners are not detected

  Note
  ----
    This function determines where the segment intersects the plane
    containing the triangle and then projects the intersection point
    and triangle into 2D where a point in polygon test is
    performed. Although rare, 2D point in polygon tests can fail if
    the randomly determined outside point and the test point cross a
    vertex of the polygon. 
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
cdef np.ndarray cross_count_3d(double[:,:] start_pnts,
                               double[:,:] end_pnts,                         
                               double[:,:] vertices,
                               long[:,:] simplices):
  '''
  Description
  -----------
    returns an array of the number of intersections between each line
    segment, described by start_pnts and end_pnts, and the simplices

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
    out[i] = _cross_count_3d(seg,vertices,simplices)

  return np.asarray(out)  


@boundscheck(False)
@wraparound(False)
cdef int _cross_count_3d(segment3d seg,
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
  Description
  -----------
    returns an array identifying which simplex is intersected by
    start_pnts and end_pnts. 

  Note
  ----
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
cdef int _intersection_index_3d(segment3d seg,
                                double[:,:] vertices,
                                long[:,:] simplices) except *:         
  cdef:
    int i
    int N = simplices.shape[0]
    triangle3d tri
    
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
      return i
 
  raise ValueError('No intersection found for segment [[%s,%s,%s],[%s,%s,%s]]' % 
                   (seg.a.x,seg.a.y,seg.a.z,seg.b.x,seg.b.y,seg.b.z))


@boundscheck(False)
@wraparound(False)
cdef np.ndarray intersection_point_3d(double[:,:] start_pnts,
                                      double[:,:] end_pnts,
                                      double[:,:] vertices,
                                      long[:,:] simplices):         
  '''
  Description
  -----------
    returns the intersection points between the line segments,
    described by start_pnts and end_pnts, and the simplices

  Note
  ----
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
  Description
  -----------
    returns the normal vectors to the simplices intersected start_pnts
    and end_pnts

  Note
  ----
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
  Description
  -----------
    returns a boolean array identifying whether the points are
    contained within the closed simplicial complex described by
    vertices and simplices

  '''
  cdef:
    int count,i
    int N = pnt.shape[0]
    long[:] out = np.empty((N,),dtype=int,order='c') 
    segment3d seg
    vector3d outside_pnt

  outside_pnt = find_outside_3d(vertices)
  for i in range(N):
    seg.a.x = outside_pnt.x
    seg.a.y = outside_pnt.y
    seg.a.z = outside_pnt.z
    seg.b.x = pnt[i,0]
    seg.b.y = pnt[i,1]
    seg.b.z = pnt[i,2]
    count = _cross_count_3d(seg,vertices,simplices)
    out[i] = count%2

  return np.asarray(out,dtype=bool)


def intersection_point(start_points,end_points,vertices,simplices):
  '''
  Description
  -----------
    returns the intersection points between the line segments,
    described by start_points and end_points, and the simplicial
    complex, described by vertices and simplices. This function works
    for 1, 2, and 3 spatial dimensions.

  Parameters
  ----------
    start_points: (N,D) array of vertices describing one end of each
      line segment. N is the number of line segments

    end_points: (N,D) array of vertices describing the other end of
      each line segment. N is the number of line segments.

    vertices: (M,D) array of vertices within the simplicial complex. M
      is the number of vertices

    simplices: (P,D) array of vertex indices for each simplex. P is
      the number of simplices

  Returns
  -------
    out: (N,D) array of intersection points    

  Note
  ----
    This function fails when a intersection is not found for a line
    segment

  '''
  start_points = np.asarray(start_points)
  end_points = np.asarray(end_points)
  vertices = np.asarray(vertices)
  simplices = np.asarray(simplices)
  assert start_points.shape[1] == end_points.shape[1]
  assert start_points.shape[1] == vertices.shape[1]
  assert start_points.shape[0] == end_points.shape[0]
  dim = start_points.shape[1]
  if dim == 1:
    vert = vertices[simplices[:,0]]
    crossed_bool = (start_points-vert.T)*(end_points-vert.T) <= 0.0
    crossed_idx = np.array([np.nonzero(i)[0][0] for i in crossed_bool],dtype=int)
    out = vert[crossed_idx]

  if dim == 2:
    out = intersection_point_2d(start_points,end_points,vertices,simplices)

  if dim == 3:
    out = intersection_point_3d(start_points,end_points,vertices,simplices)

  return out


def intersection_normal(start_points,end_points,vertices,simplices):
  '''
  Description
  -----------
    returns the normal vectors to the simplexes intersected by the
    line segments, described by start_points and end_points. This
    function works for 1, 2, and 3 spatial dimensions.

  Parameters
  ----------
    start_points: (N,D) array of vertices describing one end of each
      line segment. N is the number of line segments

    end_points: (N,D) array of vertices describing the other end of
      each line segment. N is the number of line segments.

    vertices: (M,D) array of vertices within the simplicial complex. M
      is the number of vertices

    simplices: (P,D) array of vertex indices for each simplex. P is
      the number of simplices

  Returns
  -------
    out: (N,D) array of normal vectors

  Note
  ----
    This function fails when a intersection is not found for a line
    segment

  '''
  start_points = np.asarray(start_points)
  end_points = np.asarray(end_points)
  vertices = np.asarray(vertices)
  simplices = np.asarray(simplices)
  assert start_points.shape[1] == end_points.shape[1]
  assert start_points.shape[1] == vertices.shape[1]
  assert start_points.shape[0] == end_points.shape[0]
  dim = start_points.shape[1]
  if dim == 1:
    out = np.ones(start_points.shape,dtype=float)
    vert = vertices[simplices[:,0]]
    crossed_bool = (start_points-vert.T)*(end_points-vert.T) <= 0.0
    crossed_idx = np.array([np.nonzero(i)[0][0] for i in crossed_bool],dtype=int)
    crossed_vert = vert[crossed_idx]
    out[crossed_vert < start_points] = -1.0

  if dim == 2:
    out = cross_normals_2d(start_points,end_points,vertices,simplices)

  if dim == 3:
    out = cross_normals_3d(start_points,end_points,vertices,simplices)

  return out


def intersection_index(start_points,end_points,vertices,simplices):
  '''
  Description
  -----------
    returns the indices of the simplices intersected by the line
    segments. This function works for 1, 2, and 3 spatial dimensions.

  Parameters
  ----------
    start_points: (N,D) array of vertices describing one end of each
      line segment. N is the number of line segments

    end_points: (N,D) array of vertices describing the other end of
      each line segment. N is the number of line segments.

    vertices: (M,D) array of vertices within the simplicial complex. M
      is the number of vertices

    simplices: (P,D) array of vertex indices for each simplex. P is
      the number of simplices

  Returns
  -------
    out: (N,) array of simplex indices 

  Note
  ----
    This function fails when a intersection is not found for a line
    segment

  '''
  start_points = np.asarray(start_points,dtype=float)
  end_points = np.asarray(end_points,dtype=float)
  vertices = np.asarray(vertices,dtype=float)
  simplices = np.asarray(simplices,dtype=int)
  assert start_points.shape[1] == end_points.shape[1]
  assert start_points.shape[1] == vertices.shape[1]
  assert start_points.shape[0] == end_points.shape[0]
  dim = start_points.shape[1]
  if dim == 1:
    out = np.ones(start_points.shape,dtype=float)
    vert = vertices[simplices[:,0]]
    crossed_bool = (start_points-vert.T)*(end_points-vert.T) <= 0.0
    out = np.array([np.nonzero(i)[0][0] for i in crossed_bool],dtype=int)

  if dim == 2:
    out = intersection_index_2d(start_points,end_points,vertices,simplices)

  if dim == 3:
    out = intersection_index_3d(start_points,end_points,vertices,simplices)

  return out


def cross_count(start_points,end_points,vertices,simplices):
  '''
  Description
  -----------
    returns the number of simplexes crossed by the line segments
    described by start_points and end_points. This function works for
    1, 2, and 3 spatial dimensions.

  Parameters
  ----------
    start_points: (N,D) array of vertices describing one end of each
      line segment. N is the number of line segments

    end_points: (N,D) array of vertices describing the other end of
      each line segment. N is the number of line segments.

    vertices: (M,D) array of vertices within the simplicial complex. M
      is the number of vertices

    simplices: (P,D) array of vertex indices for each simplex. P is
      the number of simplices

  Returns
  -------
    out: (N,) array of intersection counts

  '''
  start_points = np.asarray(start_points,dtype=float)
  end_points = np.asarray(end_points,dtype=float)
  vertices = np.asarray(vertices,dtype=float)
  simplices = np.asarray(simplices,dtype=int)
  assert start_points.shape[1] == end_points.shape[1]
  assert start_points.shape[1] == vertices.shape[1]
  assert start_points.shape[0] == end_points.shape[0]
  dim = start_points.shape[1]
  if dim == 1:
    vert = vertices[simplices[:,0]]
    crossed_bool = (start_points-vert.T)*(end_points-vert.T) <= 0.0
    out = np.sum(crossed_bool,axis=1)

  if dim == 2:
    out = cross_count_2d(start_points,end_points,vertices,simplices)

  if dim == 3:
    out = cross_count_3d(start_points,end_points,vertices,simplices)

  return out


def contains(points,vertices,simplices):
  '''
  Description
  -----------
    returns a boolean array identifying whether the points are
    contained within a closed simplicial complex described by vertices
    and simplices.  This function works for 1, 2, and 3 spatial
    dimensions.

  Parameters
  ----------
    points: (N,D) array of points

    vertices: (M,D) array of vertices within the simplicial complex. M
      is the number of vertices

    simplices: (P,D) array of vertex indices for each simplex. P is
      the number of simplices

  Returns
  -------
    out: (N,) boolean array identifying whether each point is in the
      simplicial complex

  Note
  ----
    This function does not ensure that the simplicial complex is
    closed.  If it is not then bogus results will be returned.  The
    closedness can be checked using the is_valid function.  

    
    This function determines whether a point is contained within the
    simplicial complex by finding the number of intersections between
    each point and an arbitrary outside point.  It is possible,
    although rare, that this function will fail if the line segment
    intersects a simplex at an edge.

    This function does not require any orientation for the simplices

  '''
  points = np.asarray(points)
  vertices = np.asarray(vertices)
  simplices = np.asarray(simplices)
  dim = points.shape[1]
  assert points.shape[1] == vertices.shape[1]
  if dim == 1:
    vert = vertices[simplices[:,0]]
    end_points = np.ones(np.shape(points))*np.min(vertices) - 1.0
    crossed_bool = (points-vert.T)*(end_points-vert.T) <= 0.0
    crossed_count = np.sum(crossed_bool,axis=1)
    out = np.array(crossed_count%2,dtype=bool)

  if dim == 2:
    out = contains_2d(points,vertices,simplices)

  if dim == 3:
    out = contains_3d(points,vertices,simplices)

  return out


def simplex_normals(vert,smp):
  '''                       
  Description           
  -----------                         
    returns the normal vectors for each simplex. Orientation is 
    determined by the right hand rule 

  Note
  ----
    This is only defined for 2 and 3 dimensional simplices

  '''
  vert = np.asarray(vert,dtype=float)
  smp = np.asarray(smp,dtype=int)

  # spatial dimensions        
  dim = vert.shape[1]
  
  if (dim != 2) & (dim != 3):
    raise ValueError('simplices must be 2 or 3 dimensional')

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
  Description
  -----------
    returns the outward normal vectors for each simplex. The sign
    of the returned vectors are only meaningful if the simplices 
    enclose an area in 2D or a volume in 3D

  Note
  ----
    This is only defined for 2 and 3 dimensional simplices

  '''
  vert = np.asarray(vert,dtype=float)
  smp = np.asarray(smp,dtype=int)
  
  # spatial dimensions        
  dim = vert.shape[1]
  
  if (dim != 2) & (dim != 3):
    raise ValueError('simplices must be 2 or 3 dimensional')

  smp = oriented_simplices(vert,smp)
  return simplex_normals(vert,smp)


def simplex_upward_normals(vert,smp):
  '''
  Description
  -----------
    returns the normal vectors for each simplex whose sign for the
    last spatial dimension is positive.

  Note
  ----
    This is only defined for 2 and 3 dimensional simplices

  '''
  vert = np.asarray(vert,dtype=float)
  smp = np.asarray(smp,dtype=int)
  
  # spatial dimensions        
  dim = vert.shape[1]
  
  if (dim != 2) & (dim != 3):
    raise ValueError('simplices must be 2 or 3 dimensional')

  out = simplex_normals(vert,smp)
  out[out[:,-1]<0] *= -1
  return out


def oriented_simplices(vert,smp):
  '''
  Description                       
  -----------                   
    Returns simplex indices that are ordered such that each simplex
    normal vector, as defined by the right hand rule, points outward
                                    
  Note                                
  ----                         
    If one dimensional simplices are given, then the simplices are
    returned unaltered.

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


def complex_volume(vert,smp,orient=True):
  '''
  Description
  -----------
    returns the volume of a polyhedra, area of a polygon, or length of
    a segment enclosed by the simplices

  Parameters
  ----------
    vert: vertices of the domain

    smp: vertex indices making of each simplex 

    orient (default=True): If true, the simplices are oriented such
      that their normals from the right hand rule point outward. The 
      time for this operation increase quadratically with the number
      of simplexes. This does nothing for 1D simplices

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


def contains_N_duplicates(iterable,N=1):
  '''            
  returns True if every element in iterable is repeated N times
  '''
  for s in iterable:
    count = 0
    for t in iterable:
      # s == t can either return a boolean, or a boolean array in the
      # event of s and t being numpy arrays.  np.all returns the 
      # appropriate value in either case
      if np.all(s == t):
        count += 1

    if count != N:
      return False

  return True


def is_valid(smp):
  '''             
  Description
  -----------
    Returns True if the following conditions are met:

      every simplex is unique

      every simplex contains unique vertices

      (for 2D and 3D) every simplex shares an edge with exactly one 
      other simplex. (for 1D) exactly 2 simplexes are given                     

  Parameters
  ----------
    smp: simplices defining the domain

  Note
  ----
    This function can take a while for a large (>1000) number of 
    simplexes 

  '''
  smp = np.asarray(smp)
  smp = np.array([np.sort(i) for i in smp])
  dim = smp.shape[1]
  sub_smp = []
  # check for repeated simplexes 
  if not contains_N_duplicates(smp,1):
    return False

  for s in smp:
    # check for repeated vertices in a simplex
    if not contains_N_duplicates(s,1):
      return False

    for c in combinations(s,dim-1):
      c_list = list(c)
      c_list.sort()
      sub_smp.append(c_list)

  out = contains_N_duplicates(sub_smp,2)
  return out





