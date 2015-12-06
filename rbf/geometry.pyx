# distutils: extra_compile_args = -fopenmp 
# distutils: extra_link_args = -fopenmp
import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from cython cimport boundscheck,wraparound
from libc.stdlib cimport rand

cdef extern from "math.h":
  cdef float abs(float x) nogil

cdef extern from "math.h":
  cdef float sqrt(float x) nogil

cdef extern from "limits.h":
    int RAND_MAX


@boundscheck(False)
@wraparound(False)
cdef void set_outside(double[:,:] v,
                      double[:] out) nogil:
  cdef:
    unsigned int i,j
    
  for j in range(v.shape[1]):
    out[j] = v[0,j]

  for i in range(1,v.shape[0]):
    for j in range(v.shape[1]):
      if v[i,j] < out[j]:
        out[j] = v[i,j] 

  for j in range(v.shape[1]):
    out[j] -= 1.123456789 + rand()*1.0/RAND_MAX

  return 
  

@boundscheck(False)
@wraparound(False)
cdef bint is_collinear_2d(double[:,:] s1,
                          double[:,:] s2) nogil:
  cdef:
    double a,b
  
  a = ((s2[0,0]-s1[0,0])*(s1[1,1]-s1[0,1]) - 
       (s1[1,0]-s1[0,0])*(s2[0,1]-s1[0,1]))
  b = ((s2[1,0]-s2[0,0])*(s1[1,1]-s1[0,1]) - 
       (s1[1,0]-s1[0,0])*(s2[1,1]-s2[0,1]))

  if not (a == 0.0):
    return False

  if not (b == 0.0):
    return False

  else:
    return True


@boundscheck(False)
@wraparound(False)
cdef bint is_overlapping_2d(double[:,:] s1,
                            double[:,:] s2) nogil:
  cdef:
    double a,b,c,t0,t1

  if not is_collinear_2d(s1,s2):
    return False

  a = ((s2[0,0]-s1[0,0])*(s1[1,0]-s1[0,0]) + 
       (s2[0,1]-s1[0,1])*(s1[1,1]-s1[0,1]))
  b = ((s2[1,0]-s2[0,0])*(s1[1,0]-s1[0,0]) + 
       (s2[1,1]-s2[0,1])*(s1[1,1]-s1[0,1]))
  c = ((s1[1,0]-s1[0,0])*(s1[1,0]-s1[0,0]) + 
       (s1[1,1]-s1[0,1])*(s1[1,1]-s1[0,1]))
  t0 = a/c
  t1 = t0 + b/c
  if ((t0 <= 0.0) & (t1 <= 0.0)) | ((t0 >= 1.0) & (t1 >= 1.0)):
    return False

  else:
    return True


@boundscheck(False)
@wraparound(False)
cdef bint is_intersecting_2d(double[:,:] s1,
                             double[:,:] s2,
                             bint anchor=True,
                             bint tip=True) nogil:
  cdef:
    double d,s,t

  d = ((s2[1,0]-s2[0,0])*(s1[1,1]-s1[0,1]) - 
       (s1[1,0]-s1[0,0])*(s2[1,1]-s2[0,1]))
  if d == 0.0:
    if is_overlapping_2d(s1,s2):
      return False

    elif ((s1[0,0] == s2[0,0]) & 
          (s1[0,1] == s2[0,1])):
      s = 0.0
      t = 0.0

    elif ((s1[0,0] == s2[1,0]) & 
          (s1[0,1] == s2[1,1])):
      s = 1.0
      t = 0.0

    elif ((s1[1,0] == s2[1,0]) & 
          (s1[1,1] == s2[1,1])):
      s = 1.0
      t = 1.0

    elif ((s1[1,0] == s2[0,0]) & 
          (s1[1,1] == s2[0,1])):
      s = 0.0
      t = 1.0

    else:
      return False

  else:
    s = (1/d)*((s1[0,0]-s2[0,0])*(s1[1,1]-s1[0,1]) - 
               (s1[0,1]-s2[0,1])*(s1[1,0]-s1[0,0]))
    t = (1/d)*((s1[0,0]-s2[0,0])*(s2[1,1]-s2[0,1]) - 
               (s1[0,1]-s2[0,1])*(s2[1,0]-s2[0,0]))

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
  N = start_pnts.shape[0]
  cdef:
    int i
    double[:,:] seg = np.empty((2,2),dtype=float,order='c')
    double[:,:] dummy_seg = np.empty((2,2),dtype=float,order='c')
    long[:] out = np.empty((N,),dtype=int,order='c')

  for i in range(N):
    seg[0,0] = start_pnts[i,0]
    seg[0,1] = start_pnts[i,1]
    seg[1,0] = end_pnts[i,0]
    seg[1,1] = end_pnts[i,1]
    out[i] = _cross_count_2d(seg,vertices,simplices,dummy_seg)

  return np.asarray(out,dtype=int)


@boundscheck(False)
@wraparound(False)
cdef int _cross_count_2d(double[:,:] seg,
                         double[:,:] vertices,
                         long[:,:] simplices,
                         double[:,:] dummy_seg) nogil:
  cdef:
    unsigned int i
    unsigned int count = 0

  for i in range(simplices.shape[0]):
    dummy_seg[0,0] = vertices[simplices[i,0],0]
    dummy_seg[0,1] = vertices[simplices[i,0],1]
    dummy_seg[1,0] = vertices[simplices[i,1],0]
    dummy_seg[1,1] = vertices[simplices[i,1],1]
    if is_intersecting_2d(seg,dummy_seg):
      count += 1

  return count


@boundscheck(False)
@wraparound(False)
cpdef np.ndarray cross_which_2d(double[:,:] start_pnts,
                                double[:,:] end_pnts,
                                double[:,:] vertices,
                                long[:,:] simplices):
  N = start_pnts.shape[0]
  cdef:
    int i
    double[:,:] seg = np.empty((2,2),dtype=float,order='c')
    double[:,:] dummy_seg = np.empty((2,2),dtype=float,order='c')
    long[:] out = np.empty((N,),dtype=int,order='c')

  for i in range(N):
    seg[0,0] = start_pnts[i,0]
    seg[0,1] = start_pnts[i,1]
    seg[1,0] = end_pnts[i,0]
    seg[1,1] = end_pnts[i,1]
    out[i] = _cross_which_2d(seg,vertices,simplices,dummy_seg)

  return np.asarray(out,dtype=int)


@boundscheck(False)
@wraparound(False)
cdef int _cross_which_2d(double[:,:] seg,
                         double[:,:] vertices,
                         long[:,:] simplices,
                         double[:,:] dummy_seg) except -1:
  cdef:
    int i

  for i in range(simplices.shape[0]):
    dummy_seg[0,0] = vertices[simplices[i,0],0]
    dummy_seg[0,1] = vertices[simplices[i,0],1]
    dummy_seg[1,0] = vertices[simplices[i,1],0]
    dummy_seg[1,1] = vertices[simplices[i,1],1]
    if is_intersecting_2d(seg,dummy_seg):
      return i

  raise ValueError('No intersection found for segment [%s,%s]' % 
                   (np.asarray(seg[0]),np.asarray(seg[1])))

@boundscheck(False)
@wraparound(False)
cpdef np.ndarray cross_where_2d(double[:,:] start_pnts,
                                double[:,:] end_pnts,
                                double[:,:] vertices,
                                long[:,:] simplices):         
  N = start_pnts.shape[0]
  cdef:
    int i
    double[:,:] seg = np.empty((2,2),dtype=float,order='c')
    double[:,:] out = np.empty((N,2),dtype=float,order='c')
    double[:,:] dummy_seg = np.empty((2,2),dtype=float,order='c')
    double[:] dummy_pnt = np.empty((2,),dtype=float,order='c')

  for i in range(N):
    seg[0,0] = start_pnts[i,0]
    seg[0,1] = start_pnts[i,1]
    seg[1,0] = end_pnts[i,0]
    seg[1,1] = end_pnts[i,1]
    _cross_where_2d(seg,vertices,simplices,dummy_pnt,dummy_seg)
    out[i,0] = dummy_pnt[0]
    out[i,1] = dummy_pnt[1]

  return np.asarray(out)


@boundscheck(False)
@wraparound(False)
cdef int _cross_where_2d(double[:,:] seg,
                         double[:,:] vertices,
                         long[:,:] simplices,
                         double[:] out,
                         double[:,:] dummy_seg) except -1:
  cdef:
    int idx
    double proj1,proj2,n1,n2

  idx = _cross_which_2d(seg,vertices,simplices,dummy_seg)
  dummy_seg[0,0] = vertices[simplices[idx,0],0]
  dummy_seg[0,1] = vertices[simplices[idx,0],1]
  dummy_seg[1,0] = vertices[simplices[idx,1],0]
  dummy_seg[1,1] = vertices[simplices[idx,1],1]

  n1 =  (dummy_seg[1,1]-dummy_seg[0,1])
  n2 = -(dummy_seg[1,0]-dummy_seg[0,0])

  proj1 = ((seg[0,0]-dummy_seg[0,0])*n1 +
           (seg[0,1]-dummy_seg[0,1])*n2)
  proj2 = ((seg[1,0]-dummy_seg[0,0])*n1 +
           (seg[1,1]-dummy_seg[0,1])*n2)

  out[0] = seg[0,0] + (proj1/(proj1-proj2))*(
           (seg[1,0]-seg[0,0]))
  out[1] = seg[0,1] + (proj1/(proj1-proj2))*(
           (seg[1,1]-seg[0,1]))

  return 0


@boundscheck(False)
@wraparound(False)
cpdef np.ndarray cross_normals_2d(double[:,:] start_pnts,
                                  double[:,:] end_pnts,
                                  double[:,:] vertices,
                                  long[:,:] simplices):         
  N = start_pnts.shape[0]
  cdef:
    int i
    double[:,:] seg = np.empty((2,2),dtype=float,order='c')
    double[:,:] out = np.empty((N,2),dtype=float,order='c')
    double[:,:] dummy_seg = np.empty((2,2),dtype=float,order='c')
    double[:] dummy_pnt = np.empty((2,),dtype=float,order='c')

  for i in range(N):
    seg[0,0] = start_pnts[i,0]
    seg[0,1] = start_pnts[i,1]
    seg[1,0] = end_pnts[i,0]
    seg[1,1] = end_pnts[i,1]
    _cross_normals_2d(seg,vertices,simplices,dummy_pnt,dummy_seg)
    out[i,0] = dummy_pnt[0]
    out[i,1] = dummy_pnt[1]

  return np.asarray(out)  


@boundscheck(False)
@wraparound(False)
cdef int _cross_normals_2d(double[:,:] seg,
                           double[:,:] vertices,
                           long[:,:] simplices,
                           double[:] out,
                           double[:,:] dummy_seg) except -1:         
  cdef:
    double proj,n
    int idx

  idx = _cross_which_2d(seg,vertices,simplices,dummy_seg)
  dummy_seg[0,0] = vertices[simplices[idx,0],0]
  dummy_seg[0,1] = vertices[simplices[idx,0],1]
  dummy_seg[1,0] = vertices[simplices[idx,1],0]
  dummy_seg[1,1] = vertices[simplices[idx,1],1]

  out[0] =  (dummy_seg[1,1]-dummy_seg[0,1])
  out[1] = -(dummy_seg[1,0]-dummy_seg[0,0])
  proj = ((seg[1,0]-dummy_seg[0,0])*out[0] +
          (seg[1,1]-dummy_seg[0,1])*out[1])
  if proj <= 0:
    out[0] *= -1
    out[1] *= -1

  n = sqrt(out[0]**2 + out[1]**2)
  out[0] /= n
  out[1] /= n

  return 0


@boundscheck(False)
@wraparound(False)
cpdef np.ndarray contains_2d(double[:,:] pnt,
                             double[:,:] vertices,
                             long[:,:] simplices):
  N = pnt.shape[0]
  cdef:
    int count,i
    double[:] dummy_pnt = np.empty((2,),dtype=float,order='c')
    double[:,:] dummy_seg1 = np.empty((2,2),dtype=float,order='c')
    double[:,:] dummy_seg2 = np.empty((2,2),dtype=float,order='c')
    long[:] out = np.empty((N,),dtype=int,order='c') 

  set_outside(vertices,dummy_pnt)
  dummy_seg1[0,0] = dummy_pnt[0]
  dummy_seg1[0,1] = dummy_pnt[1]
  for i in range(N):
    dummy_seg1[1,0] = pnt[i,0]
    dummy_seg1[1,1] = pnt[i,1]
    count = _cross_count_2d(dummy_seg1,vertices,simplices,dummy_seg2)
    out[i] = count%2

  return np.asarray(out,dtype=bool)


@boundscheck(False)
@wraparound(False)
cdef bint is_intersecting_3d(double[:,:] seg,
                             double[:,:] tri):
  cdef:
    double[:] dummy_pnt1 = np.empty((3,),dtype=float,order='c')
    double[:] dummy_pnt2 = np.empty((3,),dtype=float,order='c')
    double[:,:] dummy_seg1 = np.empty((2,2),dtype=float,order='c')
    double[:,:] dummy_seg2 = np.empty((2,2),dtype=float,order='c')

  return _is_intersecting_3d(seg,tri,
                             dummy_pnt1,dummy_pnt2,
                             dummy_seg1,dummy_seg2)


@boundscheck(False)
@wraparound(False)
cdef bint _is_intersecting_3d(double[:,:] seg,
                              double[:,:] tri,
                              double[:] dummy_pnt1,
                              double[:] dummy_pnt2,
                              double[:,:] dummy_seg1,
                              double[:,:] dummy_seg2) nogil:
  cdef:
    double proj1,proj2,n1,n2,n3
    unsigned int i,idx1,idx2
    unsigned int count = 0

  set_outside(tri,dummy_pnt2)

  n1 =  ((tri[1,1]-tri[0,1])*(tri[2,2]-tri[0,2]) - 
         (tri[1,2]-tri[0,2])*(tri[2,1]-tri[0,1]))
  n2 = -((tri[1,0]-tri[0,0])*(tri[2,2]-tri[0,2]) - 
         (tri[1,2]-tri[0,2])*(tri[2,0]-tri[0,0])) 
  n3 =  ((tri[1,0]-tri[0,0])*(tri[2,1]-tri[0,1]) - 
         (tri[1,1]-tri[0,1])*(tri[2,0]-tri[0,0]))

  proj1 = ((seg[0,0]-tri[0,0])*n1 + 
           (seg[0,1]-tri[0,1])*n2 +
           (seg[0,2]-tri[0,2])*n3)
  proj2 = ((seg[1,0]-tri[0,0])*n1 + 
           (seg[1,1]-tri[0,1])*n2 +
           (seg[1,2]-tri[0,2])*n3)

  if proj1*proj2 > 0:
    return False

  # coplanar segments will always return false
  # There is a possibility that the segment touches
  # one point on the triangle
  if (proj1 == 0) & (proj2 == 0):
    return False

  # intersection point
  dummy_pnt1[0] = seg[0,0] + (proj1/(proj1-proj2))*(
                  (seg[1,0]-seg[0,0]))
  dummy_pnt1[1] = seg[0,1] + (proj1/(proj1-proj2))*(
                  (seg[1,1]-seg[0,1]))
  dummy_pnt1[2] = seg[0,2] + (proj1/(proj1-proj2))*(
                  (seg[1,2]-seg[0,2]))

  if (abs(n1) >= abs(n2)) & (abs(n1) >= abs(n3)):
    idx1 = 1 
    idx2 = 2

  if (abs(n2) >= abs(n1)) & (abs(n2) >= abs(n3)):
    idx1 = 0 
    idx2 = 2

  if (abs(n3) >= abs(n1)) & (abs(n3) >= abs(n2)):
    idx1 = 0 
    idx2 = 1

  dummy_seg1[0,0] = dummy_pnt1[idx1]
  dummy_seg1[0,1] = dummy_pnt1[idx2]
  dummy_seg1[1,0] = dummy_pnt2[idx1]
  dummy_seg1[1,1] = dummy_pnt2[idx2]
  
  dummy_seg2[0,0] = tri[0,idx1]
  dummy_seg2[0,1] = tri[0,idx2]
  dummy_seg2[1,0] = tri[1,idx1]
  dummy_seg2[1,1] = tri[1,idx2]
  if is_intersecting_2d(dummy_seg1,dummy_seg2,True,False):
    count += 1

  dummy_seg2[0,0] = tri[1,idx1]
  dummy_seg2[0,1] = tri[1,idx2]
  dummy_seg2[1,0] = tri[2,idx1]
  dummy_seg2[1,1] = tri[2,idx2]
  if is_intersecting_2d(dummy_seg1,dummy_seg2,True,False):
    count += 1

  dummy_seg2[0,0] = tri[2,idx1]
  dummy_seg2[0,1] = tri[2,idx2]
  dummy_seg2[1,0] = tri[0,idx1]
  dummy_seg2[1,1] = tri[0,idx2]
  if is_intersecting_2d(dummy_seg1,dummy_seg2,True,False):
    count += 1

  return count%2 == 1


@boundscheck(False)
@wraparound(False)
cpdef int cross_count_3d(double[:,:] seg,
                         double[:,:] vertices,
                         long[:,:] simplices):
  cdef:
    double[:] dummy_pnt1 = np.empty((3,),dtype=float,order='c')
    double[:] dummy_pnt2 = np.empty((3,),dtype=float,order='c')
    double[:,:] dummy_tri = np.empty((3,3),dtype=float,order='c')
    double[:,:] dummy_seg1 = np.empty((2,2),dtype=float,order='c')
    double[:,:] dummy_seg2 = np.empty((2,2),dtype=float,order='c')

  return _cross_count_3d(seg,vertices,simplices,
                         dummy_pnt1,dummy_pnt2,dummy_tri,
                         dummy_seg1,dummy_seg2)


@boundscheck(False)
@wraparound(False)
cdef int _cross_count_3d(double[:,:] seg,
                         double[:,:] vertices,
                         long[:,:] simplices,
                         double[:] dummy_pnt1,
                         double[:] dummy_pnt2,
                         double[:,:] dummy_tri,
                         double[:,:] dummy_seg1,
                         double[:,:] dummy_seg2) nogil:
  cdef:
    unsigned int i
    unsigned int count = 0

  for i in range(simplices.shape[0]):
    dummy_tri[0] = vertices[simplices[i,0]]
    dummy_tri[1] = vertices[simplices[i,1]]
    dummy_tri[2] = vertices[simplices[i,2]]
    if _is_intersecting_3d(seg,dummy_tri,
                           dummy_pnt1,dummy_pnt2,
                           dummy_seg1,dummy_seg2):
      count += 1

  return count


@boundscheck(False)
@wraparound(False)
cpdef np.ndarray cross_which_3d(double[:,:] seg,
                                double[:,:] vertices,
                                long[:,:] simplices):         
  cdef:
    unsigned int i
    list out = []
    double[:,:] tri = np.empty((3,3),dtype=float,order='c')
    double[:] dummy_pnt1 = np.empty((3,),dtype=float,order='c')
    double[:] dummy_pnt2 = np.empty((3,),dtype=float,order='c')
    double[:,:] dummy_seg1 = np.empty((2,2),dtype=float,order='c')
    double[:,:] dummy_seg2 = np.empty((2,2),dtype=float,order='c')

    
  for i in range(simplices.shape[0]):
    tri[0] = vertices[simplices[i,0]]
    tri[1] = vertices[simplices[i,1]]
    tri[2] = vertices[simplices[i,2]]
    if _is_intersecting_3d(seg,tri,
                           dummy_pnt1,dummy_pnt2,
                           dummy_seg1,dummy_seg2):
      out.append(i)
 
  return np.asarray(out)


@boundscheck(False)
@wraparound(False)
cpdef np.ndarray cross_where_3d(double[:,:] seg,
                                double[:,:] vertices,
                                long[:,:] simplices):         
  indices = cross_which_3d(seg,vertices,simplices)
  cdef:
    double proj1,proj2
    unsigned int i,idx
    double[:] norm = np.empty((3,),dtype=float,order='c')
    double[:,:] tri = np.empty((3,3),dtype=float,order='c')
    double[:,:] out = np.empty((indices.shape[0],3),dtype=float,order='c')

  for i in range(indices.shape[0]):
    idx = indices[i]  
    tri[0] = vertices[simplices[idx,0]]
    tri[1] = vertices[simplices[idx,1]]
    tri[2] = vertices[simplices[idx,2]]
    norm[0] =  ((tri[1,1]-tri[0,1])*(tri[2,2]-tri[0,2]) -
                (tri[1,2]-tri[0,2])*(tri[2,1]-tri[0,1]))
    norm[1] = -((tri[1,0]-tri[0,0])*(tri[2,2]-tri[0,2]) -
                (tri[1,2]-tri[0,2])*(tri[2,0]-tri[0,0]))
    norm[2] =  ((tri[1,0]-tri[0,0])*(tri[2,1]-tri[0,1]) -
                (tri[1,1]-tri[0,1])*(tri[2,0]-tri[0,0]))
    proj1 = ((seg[0,0]-tri[0,0])*norm[0] +
             (seg[0,1]-tri[0,1])*norm[1] +
             (seg[0,2]-tri[0,2])*norm[2])
    proj2 = ((seg[1,0]-tri[0,0])*norm[0] +
             (seg[1,1]-tri[0,1])*norm[1] +
             (seg[1,2]-tri[0,2])*norm[2])
    out[i,0] = seg[0,0] + (proj1/(proj1-proj2))*(
               (seg[1,0]-seg[0,0]))
    out[i,1] = seg[0,1] + (proj1/(proj1-proj2))*(
               (seg[1,1]-seg[0,1]))
    out[i,2] = seg[0,2] + (proj1/(proj1-proj2))*(
               (seg[1,2]-seg[0,2]))

  return np.asarray(out)


@boundscheck(False)
@wraparound(False)
cpdef np.ndarray cross_normals_3d(double[:,:] seg,
                                  double[:,:] vertices,
                                  long[:,:] simplices):         
  indices = cross_which_3d(seg,vertices,simplices)
  cdef:
    double proj
    double n
    unsigned int i,idx
    double[:,:] out = np.empty((indices.shape[0],3),dtype=float,order='c')
    double[:,:] tri = np.empty((3,3),dtype=float,order='c')

  for i in range(indices.shape[0]):
    idx = indices[i]  
    tri[0] = vertices[simplices[idx,0]]
    tri[1] = vertices[simplices[idx,1]]
    tri[2] = vertices[simplices[idx,2]]
    out[i,0] =  ((tri[1,1]-tri[0,1])*(tri[2,2]-tri[0,2]) -
                (tri[1,2]-tri[0,2])*(tri[2,1]-tri[0,1]))
    out[i,1] = -((tri[1,0]-tri[0,0])*(tri[2,2]-tri[0,2]) -
                (tri[1,2]-tri[0,2])*(tri[2,0]-tri[0,0]))
    out[i,2] =  ((tri[1,0]-tri[0,0])*(tri[2,1]-tri[0,1]) -
                (tri[1,1]-tri[0,1])*(tri[2,0]-tri[0,0]))
    proj = ((seg[1,0]-tri[0,0])*out[i,0] +
            (seg[1,1]-tri[0,1])*out[i,1] +
            (seg[1,2]-tri[0,2])*out[i,2])
    if proj <= 0:
      out[i,0] *= -1
      out[i,1] *= -1
      out[i,2] *= -1

    n = sqrt(out[i,0]**2 + out[i,1]**2 + out[i,2]**2)
    out[i,0] /= n
    out[i,1] /= n
    out[i,2] /= n

  return np.asarray(out)


@boundscheck(False)
@wraparound(False)
cpdef np.ndarray contains_3d(double[:,:] pnt,
                             double[:,:] vertices,
                             long[:,:] simplices):
  cdef:
    unsigned int count,i
    double[:] dummy_pnt1 = np.empty((3,),dtype=float,order='c')
    double[:] dummy_pnt2 = np.empty((3,),dtype=float,order='c')
    double[:,:] dummy_tri = np.empty((3,3),dtype=float,order='c')
    double[:,:] dummy_seg1 = np.empty((2,3),dtype=float,order='c')
    double[:,:] dummy_seg2 = np.empty((2,3),dtype=float,order='c')
    double[:,:] dummy_seg3 = np.empty((2,2),dtype=float,order='c')
    double[:,:] dummy_seg4 = np.empty((2,2),dtype=float,order='c')
    long[:] out = np.empty((pnt.shape[0],),dtype=int,order='c') 

  set_outside(vertices,dummy_pnt1)
  dummy_seg1[0,0] = dummy_pnt1[0]
  dummy_seg1[0,1] = dummy_pnt1[1]
  dummy_seg1[0,2] = dummy_pnt1[2]
  for i in range(pnt.shape[0]):
    dummy_seg1[1,0] = pnt[i,0]
    dummy_seg1[1,1] = pnt[i,1]
    dummy_seg1[1,2] = pnt[i,2]
    count = _cross_count_3d(dummy_seg1,vertices,simplices,
                            dummy_pnt1,dummy_pnt2,dummy_tri,
                            dummy_seg3,dummy_seg4)
    out[i] = count%2

  return np.asarray(out,dtype=bool)



  



