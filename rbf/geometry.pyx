# distutils: extra_compile_args = -fopenmp 
# distutils: extra_link_args = -fopenmp
import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from cython cimport boundscheck,wraparound
from libc.stdlib cimport rand
from libc.stdlib cimport malloc,free

cdef extern from "math.h":
  cdef float abs(float x) nogil

cdef extern from "math.h":
  cdef float sqrt(float x) nogil

cdef extern from "limits.h":
    int RAND_MAX

cdef struct vector2d:
  double x
  double y

cdef struct segment2d:
  vector2d a
  vector2d b

cdef struct vector3d:
  double x
  double y
  double z

cdef struct segment3d:
  vector3d a
  vector3d b

cdef struct triangle3d:
  vector3d a
  vector3d b
  vector3d c


cdef double min3(double a,double b, double c) nogil:
  if (a <= b) & (a <= c):
    return a

  if (b <= a) & (b <= c):
    return b

  if (c <= a) & (c <= b):
    return c


@boundscheck(False)
@wraparound(False)
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

  out.x -= 1.123456789 + rand()*1.0/RAND_MAX
  out.y -= 1.123456789 + rand()*1.0/RAND_MAX

  return out

@boundscheck(False)
@wraparound(False)
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

  out.x -= 1.123456789 + rand()*1.0/RAND_MAX
  out.y -= 1.123456789 + rand()*1.0/RAND_MAX
  out.z -= 1.123456789 + rand()*1.0/RAND_MAX
  return out

@boundscheck(False)
@wraparound(False)
cdef bint is_collinear_2d(segment2d s1,
                          segment2d s2) nogil:
  cdef:
    double a,b
  
  a = ((s2.a.x-s1.a.x)*(s1.b.y-s1.a.y) - 
       (s1.b.x-s1.a.x)*(s2.a.y-s1.a.y))
  b = ((s2.b.x-s2.a.x)*(s1.b.y-s1.a.y) - 
       (s1.b.x-s1.a.x)*(s2.b.y-s2.a.y))

  if not (a == 0.0):
    return False

  if not (b == 0.0):
    return False

  else:
    return True


@boundscheck(False)
@wraparound(False)
cdef bint is_overlapping_2d(segment2d s1,
                            segment2d s2) nogil:
  cdef:
    double a,b,c,t0,t1

  if not is_collinear_2d(s1,s2):
    return False

  a = ((s2.a.x-s1.a.x)*(s1.b.x-s1.a.x) + 
       (s2.a.y-s1.a.y)*(s1.b.y-s1.a.y))
  b = ((s2.b.x-s2.a.x)*(s1.b.x-s1.a.x) + 
       (s2.b.y-s2.a.y)*(s1.b.y-s1.a.y))
  c = ((s1.b.x-s1.a.x)*(s1.b.x-s1.a.x) + 
       (s1.b.y-s1.a.y)*(s1.b.y-s1.a.y))
  t0 = a/c
  t1 = t0 + b/c
  if ((t0 <= 0.0) & (t1 <= 0.0)) | ((t0 >= 1.0) & (t1 >= 1.0)):
    return False

  else:
    return True


@boundscheck(False)
@wraparound(False)
cdef bint is_intersecting_2d(segment2d s1,
                             segment2d s2,
                             bint anchor=True,
                             bint tip=True) nogil:
  cdef:
    double d,s,t

  d = ((s2.b.x-s2.a.x)*(s1.b.y-s1.a.y) - 
       (s1.b.x-s1.a.x)*(s2.b.y-s2.a.y))
  if d == 0.0:
    if is_overlapping_2d(s1,s2):
      return False

    elif ((s1.a.x == s2.a.x) & 
          (s1.a.y == s2.a.y)):
      s = 0.0
      t = 0.0

    elif ((s1.a.x == s2.b.x) & 
          (s1.a.y == s2.b.y)):
      s = 1.0
      t = 0.0

    elif ((s1.b.x == s2.b.x) & 
          (s1.b.y == s2.b.y)):
      s = 1.0
      t = 1.0

    elif ((s1.b.x == s2.a.x) & 
          (s1.b.y == s2.a.y)):
      s = 0.0
      t = 1.0

    else:
      return False

  else:
    s = (1/d)*((s1.a.x-s2.a.x)*(s1.b.y-s1.a.y) - 
               (s1.a.y-s2.a.y)*(s1.b.x-s1.a.x))
    t = (1/d)*((s1.a.x-s2.a.x)*(s2.b.y-s2.a.y) - 
               (s1.a.y-s2.a.y)*(s2.b.x-s2.a.x))

  if anchor & tip:
    if (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0):
      return True

  if (not anchor) & tip:
    if (s > 0.0) & (s <= 1.0) & (t > 0.0) & (t <= 1.0):
      return True

  if anchor & (not tip):
    if (s >= 0.0) & (s < 1.0) & (t >= 0.0) & (t < 1.0):
      return True

  if (not anchor) & (not tip):
    if (s > 0.0) & (s < 1.0) & (t > 0.0) & (t < 1.0):
      return True

  return False


@boundscheck(False)
@wraparound(False)
cpdef np.ndarray cross_count_2d(double[:,:] start_pnts,
                                double[:,:] end_pnts,
                                double[:,:] vertices,
                                long[:,:] simplices):
  '''
  returns an array containing the number of boundary intersections
  between start_pnts[i] and end_pnts[i].  The boundary is defined 
  in terms of vertices and simplices.
  '''
  cdef:
    int i
    int N = start_pnts.shape[0]
    long[:] out = np.empty((N,),dtype=int,order='c')
    segment2d *seg_array = <segment2d *>malloc(N*sizeof(segment2d))
    
  if not seg_array:
    raise MemoryError()

  try:
    with nogil:
      for i in prange(N):
        seg_array[i].a.x = start_pnts[i,0]
        seg_array[i].a.y = start_pnts[i,1]
        seg_array[i].b.x = end_pnts[i,0]
        seg_array[i].b.y = end_pnts[i,1]
        out[i] = _cross_count_2d(seg_array[i],vertices,simplices)

  finally:
    free(seg_array)

  return np.asarray(out,dtype=int)


@boundscheck(False)
@wraparound(False)
cdef int _cross_count_2d(segment2d seg,
                         double[:,:] vertices,
                         long[:,:] simplices) nogil:
  cdef:
    unsigned int i
    unsigned int count = 0
    segment2d dummy_seg

  for i in range(simplices.shape[0]):
    dummy_seg.a.x = vertices[simplices[i,0],0]
    dummy_seg.a.y = vertices[simplices[i,0],1]
    dummy_seg.b.x = vertices[simplices[i,1],0]
    dummy_seg.b.y = vertices[simplices[i,1],1]
    if is_intersecting_2d(seg,dummy_seg):
      count += 1

  return count


@boundscheck(False)
@wraparound(False)
cpdef np.ndarray cross_which_2d(double[:,:] start_pnts,
                                double[:,:] end_pnts,
                                double[:,:] vertices,
                                long[:,:] simplices):
  '''
  returns an array identifying the index of the facet (i.e. which
  doublet of simplices) intersected by start_pnts[i] and
  end_pnts[i]. Note: if there is no intersection then an arbitrary
  integer is returned.
  '''
  cdef:
    int i
    int N = start_pnts.shape[0]
    long[:] out = np.empty((N,),dtype=int,order='c')
    segment2d *seg_array = <segment2d *>malloc(N*sizeof(segment2d))
    
  if not seg_array:
    raise MemoryError()
  
  try:
    with nogil:
      for i in prange(N):
        seg_array[i].a.x = start_pnts[i,0]
        seg_array[i].a.y = start_pnts[i,1]
        seg_array[i].b.x = end_pnts[i,0]
        seg_array[i].b.y = end_pnts[i,1]
        out[i] = _cross_which_2d(seg_array[i],vertices,simplices)

  finally:
    free(seg_array)

  return np.asarray(out)


@boundscheck(False)
@wraparound(False)
cdef int _cross_which_2d(segment2d seg,
                         double[:,:] vertices,
                         long[:,:] simplices) nogil:
  cdef:
    int i
    segment2d dummy_seg
  
  for i in range(simplices.shape[0]):
    dummy_seg.a.x = vertices[simplices[i,0],0]
    dummy_seg.a.y = vertices[simplices[i,0],1]
    dummy_seg.b.x = vertices[simplices[i,1],0]
    dummy_seg.b.y = vertices[simplices[i,1],1]
    if is_intersecting_2d(seg,dummy_seg):
      return i


@boundscheck(False)
@wraparound(False)
cpdef np.ndarray cross_where_2d(double[:,:] start_pnts,
                                double[:,:] end_pnts,
                                double[:,:] vertices,
                                long[:,:] simplices):         
  '''
  returns an array identifying the position where the boundary is
  intersected by start_pnts[i] and end_pnts[i]. Note: if there is no
  intersection then an arbitrary vector is returned.
  '''
  cdef:
    int i
    int N = start_pnts.shape[0]
    double[:,:] out = np.empty((N,2),dtype=float,order='c')
    vector2d vec
    segment2d *seg_array = <segment2d *>malloc(N*sizeof(segment2d))

  if not seg_array:
    raise MemoryError()

  try:
    with nogil:
      for i in prange(N):
        seg_array[i].a.x = start_pnts[i,0]
        seg_array[i].a.y = start_pnts[i,1]
        seg_array[i].b.x = end_pnts[i,0]
        seg_array[i].b.y = end_pnts[i,1]
        vec = _cross_where_2d(seg_array[i],vertices,simplices)
        out[i,0] = vec.x
        out[i,1] = vec.y

  finally:
    free(seg_array)

  return np.asarray(out)


@boundscheck(False)
@wraparound(False)
cdef vector2d _cross_where_2d(segment2d seg,
                              double[:,:] vertices,
                              long[:,:] simplices) nogil:
  cdef:
    int idx
    double proj1,proj2,n1,n2
    segment2d dummy_seg
    vector2d out

  idx = _cross_which_2d(seg,vertices,simplices)
  dummy_seg.a.x = vertices[simplices[idx,0],0]
  dummy_seg.a.y = vertices[simplices[idx,0],1]
  dummy_seg.b.x = vertices[simplices[idx,1],0]
  dummy_seg.b.y = vertices[simplices[idx,1],1]

  n1 =  (dummy_seg.b.y-dummy_seg.a.y)
  n2 = -(dummy_seg.b.x-dummy_seg.a.x)

  proj1 = ((seg.a.x-dummy_seg.a.x)*n1 +
           (seg.a.y-dummy_seg.a.y)*n2)
  proj2 = ((seg.b.x-dummy_seg.a.x)*n1 +
           (seg.b.y-dummy_seg.a.y)*n2)

  out.x = seg.a.x + (proj1/(proj1-proj2))*(
           (seg.b.x-seg.a.x))
  out.y = seg.a.y + (proj1/(proj1-proj2))*(
           (seg.b.y-seg.a.y))

  return out


@boundscheck(False)
@wraparound(False)
cpdef np.ndarray cross_normals_2d(double[:,:] start_pnts,
                                  double[:,:] end_pnts,
                                  double[:,:] vertices,
                                  long[:,:] simplices):         
  '''
  returns an array of normal vectors to the facets intersected by
  start_pnts[i] and end_pnts[i]. Note: if there is no intersection
  then an arbitrary vector is returned.
  '''
  cdef:
    int i
    int N = start_pnts.shape[0]
    double[:,:] out = np.empty((N,2),dtype=float,order='c')
    segment2d *seg_array = <segment2d *>malloc(N*sizeof(segment2d))
    vector2d vec

  if not seg_array:
    raise MemoryError()

  try:
    with nogil:
      for i in prange(N):
        seg_array[i].a.x = start_pnts[i,0]
        seg_array[i].a.y = start_pnts[i,1]
        seg_array[i].b.x = end_pnts[i,0]
        seg_array[i].b.y = end_pnts[i,1]
        vec = _cross_normals_2d(seg_array[i],vertices,simplices)
        out[i,0] = vec.x
        out[i,1] = vec.y
    
  finally:
    free(seg_array)

  return np.asarray(out)  


@boundscheck(False)
@wraparound(False)
cdef vector2d _cross_normals_2d(segment2d seg,
                                double[:,:] vertices,
                                long[:,:] simplices) nogil:         
  cdef:
    double proj,n
    int idx
    segment2d dummy_seg
    vector2d vec

  idx = _cross_which_2d(seg,vertices,simplices)
  dummy_seg.a.x = vertices[simplices[idx,0],0]
  dummy_seg.a.y = vertices[simplices[idx,0],1]
  dummy_seg.b.x = vertices[simplices[idx,1],0]
  dummy_seg.b.y = vertices[simplices[idx,1],1]

  vec.x =  (dummy_seg.b.y-dummy_seg.a.y)
  vec.y = -(dummy_seg.b.x-dummy_seg.a.x)
  proj = ((seg.b.x-dummy_seg.a.x)*vec.x +
          (seg.b.y-dummy_seg.a.y)*vec.y)
  if proj <= 0:
    vec.x *= -1
    vec.y *= -1

  n = sqrt(vec.x**2 + vec.y**2)
  vec.x /= n
  vec.y /= n

  return vec


@boundscheck(False)
@wraparound(False)
cpdef np.ndarray contains_2d(double[:,:] pnt,
                             double[:,:] vertices,
                             long[:,:] simplices):
  '''
  returns a boolean array indentifying whether the pnts are 
  contained within the domain specified by the vertices and simplices
  '''
  cdef:
    int count,i
    int N = pnt.shape[0]
    long[:] out = np.empty((N,),dtype=int,order='c') 
    segment2d *seg_array = <segment2d *>malloc(N*sizeof(segment2d))
    vector2d vec

  if not seg_array:
    raise MemoryError()

  try:
    vec = find_outside_2d(vertices)
    with nogil:
      for i in prange(N):
        seg_array[i].a.x = vec.x
        seg_array[i].a.y = vec.y
        seg_array[i].b.x = pnt[i,0]
        seg_array[i].b.y = pnt[i,1]
        count = _cross_count_2d(seg_array[i],vertices,simplices)
        out[i] = count%2

  finally:
    free(seg_array)

  return np.asarray(out,dtype=bool)


@boundscheck(False)
@wraparound(False)
cdef bint is_intersecting_3d(segment3d seg,
                             triangle3d tri) nogil:
  cdef:
    vector3d dummy_pnt1,dummy_pnt2
    segment2d dummy_seg1,dummy_seg2,dummy_seg3,dummy_seg4
    double proj1,proj2,n1,n2,n3
    unsigned int i,idx1,idx2
    unsigned int count = 0

  dummy_pnt2.x = min3(tri.a.x,tri.b.x,tri.c.x) - 1.234567890
  dummy_pnt2.y = min3(tri.a.y,tri.b.y,tri.c.y) - 2.345678901
  dummy_pnt2.z = min3(tri.a.z,tri.b.z,tri.c.z) - 3.456789012

  n1 =  ((tri.b.y-tri.a.y)*(tri.c.z-tri.a.z) - 
         (tri.b.z-tri.a.z)*(tri.c.y-tri.a.y))
  n2 = -((tri.b.x-tri.a.x)*(tri.c.z-tri.a.z) - 
         (tri.b.z-tri.a.z)*(tri.c.x-tri.a.x)) 
  n3 =  ((tri.b.x-tri.a.x)*(tri.c.y-tri.a.y) - 
         (tri.b.y-tri.a.y)*(tri.c.x-tri.a.x))

  proj1 = ((seg.a.x-tri.a.x)*n1 + 
           (seg.a.y-tri.a.y)*n2 +
           (seg.a.z-tri.a.z)*n3)
  proj2 = ((seg.b.x-tri.a.x)*n1 + 
           (seg.b.y-tri.a.y)*n2 +
           (seg.b.z-tri.a.z)*n3)

  if proj1*proj2 > 0:
    return False

  # coplanar segments will always return false
  # There is a possibility that the segment touches
  # one point on the triangle
  if (proj1 == 0) & (proj2 == 0):
    return False

  # intersection point
  dummy_pnt1.x = seg.a.x + (proj1/(proj1-proj2))*(
                  (seg.b.x-seg.a.x))
  dummy_pnt1.y = seg.a.y + (proj1/(proj1-proj2))*(
                  (seg.b.y-seg.a.y))
  dummy_pnt1.z = seg.a.z + (proj1/(proj1-proj2))*(
                  (seg.b.z-seg.a.z))

  if (abs(n1) >= abs(n2)) & (abs(n1) >= abs(n3)):
    dummy_seg1.a.x = dummy_pnt1.y
    dummy_seg1.a.y = dummy_pnt1.z
    dummy_seg1.b.x = dummy_pnt2.y
    dummy_seg1.b.y = dummy_pnt2.z

    dummy_seg2.a.x = tri.a.y
    dummy_seg2.a.y = tri.a.z
    dummy_seg2.b.x = tri.b.y
    dummy_seg2.b.y = tri.b.z

    dummy_seg3.a.x = tri.b.y
    dummy_seg3.a.y = tri.b.z
    dummy_seg3.b.x = tri.c.y
    dummy_seg3.b.y = tri.c.z

    dummy_seg4.a.x = tri.c.y
    dummy_seg4.a.y = tri.c.z
    dummy_seg4.b.x = tri.a.y
    dummy_seg4.b.y = tri.a.z

  if (abs(n2) >= abs(n1)) & (abs(n2) >= abs(n3)):
    dummy_seg1.a.x = dummy_pnt1.x
    dummy_seg1.a.y = dummy_pnt1.z
    dummy_seg1.b.x = dummy_pnt2.x
    dummy_seg1.b.y = dummy_pnt2.z

    dummy_seg2.a.x = tri.a.x
    dummy_seg2.a.y = tri.a.z
    dummy_seg2.b.x = tri.b.x
    dummy_seg2.b.y = tri.b.z

    dummy_seg3.a.x = tri.b.x
    dummy_seg3.a.y = tri.b.z
    dummy_seg3.b.x = tri.c.x
    dummy_seg3.b.y = tri.c.z

    dummy_seg4.a.x = tri.c.x
    dummy_seg4.a.y = tri.c.z
    dummy_seg4.b.x = tri.a.x
    dummy_seg4.b.y = tri.a.z

  if (abs(n3) >= abs(n1)) & (abs(n3) >= abs(n2)):
    dummy_seg1.a.x = dummy_pnt1.x
    dummy_seg1.a.y = dummy_pnt1.y
    dummy_seg1.b.x = dummy_pnt2.x
    dummy_seg1.b.y = dummy_pnt2.y

    dummy_seg2.a.x = tri.a.x
    dummy_seg2.a.y = tri.a.y
    dummy_seg2.b.x = tri.b.x
    dummy_seg2.b.y = tri.b.y

    dummy_seg3.a.x = tri.b.x
    dummy_seg3.a.y = tri.b.y
    dummy_seg3.b.x = tri.c.x
    dummy_seg3.b.y = tri.c.y

    dummy_seg4.a.x = tri.c.x
    dummy_seg4.a.y = tri.c.y
    dummy_seg4.b.x = tri.a.x
    dummy_seg4.b.y = tri.a.y


  if is_intersecting_2d(dummy_seg1,dummy_seg2,True,False):
    count += 1


  if is_intersecting_2d(dummy_seg1,dummy_seg3,True,False):
    count += 1


  if is_intersecting_2d(dummy_seg1,dummy_seg4,True,False):
    count += 1

  return count%2 == 1


@boundscheck(False)
@wraparound(False)
cpdef np.ndarray cross_count_3d(double[:,:] start_pnts,
                                double[:,:] end_pnts,                         
                                double[:,:] vertices,
                                long[:,:] simplices):
  cdef:
    int i
    int N = start_pnts.shape[0]
    long[:] out = np.empty((N,),dtype=int,order='c')
    segment3d *seg_array = <segment3d *>malloc(N*sizeof(segment3d))

  if not seg_array:
    raise MemoryError()

  try:
    with nogil:
      for i in prange(N):
        seg_array[i].a.x = start_pnts[i,0]
        seg_array[i].a.y = start_pnts[i,1]
        seg_array[i].a.z = start_pnts[i,2]
        seg_array[i].b.x = end_pnts[i,0]
        seg_array[i].b.y = end_pnts[i,1]
        seg_array[i].b.z = end_pnts[i,2]
        out[i] = _cross_count_3d(seg_array[i],vertices,simplices)
    
  finally:
    free(seg_array)

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
cpdef np.ndarray cross_which_3d(double[:,:] start_pnts,
                                double[:,:] end_pnts,                         
                                double[:,:] vertices,
                                long[:,:] simplices):
  cdef:
    int i
    int N = start_pnts.shape[0]
    long[:] out = np.empty((N,),dtype=int,order='c')
    segment3d *seg_array = <segment3d *>malloc(N*sizeof(segment3d))

  if not seg_array:
    raise MemoryError()

  try:
    with nogil:
      for i in prange(N):
        seg_array[i].a.x = start_pnts[i,0]
        seg_array[i].a.y = start_pnts[i,1]
        seg_array[i].a.z = start_pnts[i,2]
        seg_array[i].b.x = end_pnts[i,0]
        seg_array[i].b.y = end_pnts[i,1]
        seg_array[i].b.z = end_pnts[i,2]
        out[i] = _cross_which_3d(seg_array[i],vertices,simplices)
    
  finally:
    free(seg_array)

  return np.asarray(out)  


@boundscheck(False)
@wraparound(False)
cdef int _cross_which_3d(segment3d seg,
                         double[:,:] vertices,
                         long[:,:] simplices) nogil:         
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
 

@boundscheck(False)
@wraparound(False)
cpdef np.ndarray cross_where_3d(double[:,:] start_pnts,
                                double[:,:] end_pnts,
                                double[:,:] vertices,
                                long[:,:] simplices):         
  cdef:
    int i
    int N = start_pnts.shape[0]
    double[:,:] out = np.empty((N,3),dtype=float,order='c')
    vector3d vec
    segment3d *seg_array = <segment3d *>malloc(N*sizeof(segment3d))

  if not seg_array:
    raise MemoryError()

  try:
    with nogil:
      for i in prange(N):
        seg_array[i].a.x = start_pnts[i,0]
        seg_array[i].a.y = start_pnts[i,1]
        seg_array[i].a.z = start_pnts[i,2]
        seg_array[i].b.x = end_pnts[i,0]
        seg_array[i].b.y = end_pnts[i,1]
        seg_array[i].b.z = end_pnts[i,2]
        vec = _cross_where_3d(seg_array[i],vertices,simplices)
        out[i,0] = vec.x
        out[i,1] = vec.y
        out[i,2] = vec.z

  finally:
    free(seg_array)

  return np.asarray(out)


@boundscheck(False)
@wraparound(False)
cdef vector3d _cross_where_3d(segment3d seg,
                              double[:,:] vertices,
                              long[:,:] simplices) nogil:         
  cdef:
    double proj1,proj2
    int idx
    vector3d norm
    triangle3d tri
    vector3d out 

  idx = _cross_which_3d(seg,vertices,simplices)
  tri.a.x = vertices[simplices[idx,0],0]
  tri.a.y = vertices[simplices[idx,0],1]
  tri.a.z = vertices[simplices[idx,0],2]
  tri.b.x = vertices[simplices[idx,1],0]
  tri.b.y = vertices[simplices[idx,1],1]
  tri.b.z = vertices[simplices[idx,1],2]
  tri.c.x = vertices[simplices[idx,2],0]
  tri.c.y = vertices[simplices[idx,2],1]
  tri.c.z = vertices[simplices[idx,2],2]

  norm.x =  ((tri.b.y-tri.a.y)*(tri.c.z-tri.a.z) -
             (tri.b.z-tri.a.z)*(tri.c.y-tri.a.y))
  norm.y = -((tri.b.x-tri.a.x)*(tri.c.z-tri.a.z) -
             (tri.b.z-tri.a.z)*(tri.c.x-tri.a.x))
  norm.z =  ((tri.b.x-tri.a.x)*(tri.c.y-tri.a.y) -
             (tri.b.y-tri.a.y)*(tri.c.x-tri.a.x))
  proj1 = ((seg.a.x-tri.a.x)*norm.x +
           (seg.a.y-tri.a.y)*norm.y +
           (seg.a.z-tri.a.z)*norm.z)
  proj2 = ((seg.b.x-tri.a.x)*norm.x +
           (seg.b.y-tri.a.y)*norm.y +
           (seg.b.z-tri.a.z)*norm.z)
  out.x = seg.a.x + (proj1/(proj1-proj2))*(
          (seg.b.x-seg.a.x))
  out.y = seg.a.y + (proj1/(proj1-proj2))*(
          (seg.b.y-seg.a.y))
  out.z = seg.a.z + (proj1/(proj1-proj2))*(
          (seg.b.z-seg.a.z))

  return out


@boundscheck(False)
@wraparound(False)
cpdef np.ndarray cross_normals_3d(double[:,:] start_pnts,
                                  double[:,:] end_pnts,
                                  double[:,:] vertices,
                                  long[:,:] simplices):
  cdef:
    int i
    int N = start_pnts.shape[0]
    double[:,:] out = np.empty((N,3),dtype=float,order='c')
    segment3d *seg_array = <segment3d *>malloc(N*sizeof(segment3d))
    vector3d vec

  if not seg_array:
    raise MemoryError()

  try:
    with nogil:
      for i in prange(N):
        seg_array[i].a.x = start_pnts[i,0]
        seg_array[i].a.y = start_pnts[i,1]
        seg_array[i].a.z = start_pnts[i,2]
        seg_array[i].b.x = end_pnts[i,0]
        seg_array[i].b.y = end_pnts[i,1]
        seg_array[i].b.z = end_pnts[i,2]
        vec = _cross_normals_3d(seg_array[i],vertices,simplices)
        out[i,0] = vec.x
        out[i,1] = vec.y
        out[i,2] = vec.z

  finally:
    free(seg_array)

  return np.asarray(out)


@boundscheck(False)
@wraparound(False)
cdef vector3d _cross_normals_3d(segment3d seg,
                                double[:,:] vertices,
                                long[:,:] simplices) nogil:         

  cdef:
    double proj,n
    int idx
    triangle3d tri
    vector3d out

  idx = _cross_which_3d(seg,vertices,simplices)
  tri.a.x = vertices[simplices[idx,0],0]
  tri.a.y = vertices[simplices[idx,0],1]
  tri.a.z = vertices[simplices[idx,0],2]
  tri.b.x = vertices[simplices[idx,1],0]
  tri.b.y = vertices[simplices[idx,1],1]
  tri.b.z = vertices[simplices[idx,1],2]
  tri.c.x = vertices[simplices[idx,2],0]
  tri.c.y = vertices[simplices[idx,2],1]
  tri.c.z = vertices[simplices[idx,2],2]

  out.x =  ((tri.b.y-tri.a.y)*(tri.c.z-tri.a.z) -
            (tri.b.z-tri.a.z)*(tri.c.y-tri.a.y))
  out.y = -((tri.b.x-tri.a.x)*(tri.c.z-tri.a.z) -
            (tri.b.z-tri.a.z)*(tri.c.x-tri.a.x))
  out.z =  ((tri.b.x-tri.a.x)*(tri.c.y-tri.a.y) -
            (tri.b.y-tri.a.y)*(tri.c.x-tri.a.x))
  proj = ((seg.b.x-tri.a.x)*out.x +
          (seg.b.y-tri.a.y)*out.y +
          (seg.b.z-tri.a.z)*out.z)

  if proj <= 0:
    out.x *= -1
    out.y *= -1
    out.z *= -1

  n = sqrt(out.x**2 + out.y**2 + out.z**2)
  out.x /= n
  out.y /= n
  out.z /= n

  return out


@boundscheck(False)
@wraparound(False)
cpdef np.ndarray contains_3d(double[:,:] pnt,
                             double[:,:] vertices,
                             long[:,:] simplices):
  cdef:
    int count,i
    int N = pnt.shape[0]
    long[:] out = np.empty((N,),dtype=int,order='c') 
    segment3d *seg_array = <segment3d *>malloc(N*sizeof(segment3d))
    vector3d vec

  if not seg_array:
    raise MemoryError()

  try:
    vec = find_outside_3d(vertices)
    with nogil:
      for i in prange(N):
        seg_array[i].a.x = vec.x
        seg_array[i].a.y = vec.y
        seg_array[i].a.z = vec.z
        seg_array[i].b.x = pnt[i,0]
        seg_array[i].b.y = pnt[i,1]
        seg_array[i].b.z = pnt[i,2]
        count = _cross_count_3d(seg_array[i],vertices,simplices)
        out[i] = count%2

  finally:
    free(seg_array)

  return np.asarray(out,dtype=bool)

