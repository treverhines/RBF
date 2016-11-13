''' 
Contains functions which generate simple domains, such as a unit 
circle, or cube
'''
import numpy as np

def _sphere_refine(vert,smp):
  V = vert.shape[0]
  S = smp.shape[0]
  new_vert = np.zeros((V+3*S,3),dtype=float)
  new_vert[:V,:] = vert
  new_smp = np.zeros((4*S,3),dtype=int)
  for si,s in enumerate(smp):
    a,b,c = vert[s]
    i = V + 3*si
    j = i + 1
    k = i + 2
    new_vert[i] = a+b
    new_vert[j] = b+c
    new_vert[k] = a+c
    new_smp[4*si]   = [   i,   j,   k]
    new_smp[4*si+1] = [s[0],   i,   k]
    new_smp[4*si+2] = [   i,s[1],   j]
    new_smp[4*si+3] = [   k,   j,s[2]]

  new_vert = new_vert / np.linalg.norm(new_vert,axis=1)[:,None]
  return new_vert,new_smp


def _circle_refine(vert,smp):
  V = vert.shape[0]
  S = smp.shape[0]
  new_vert = np.zeros((V+S,2),dtype=float)
  new_vert[:V,:] = vert
  new_smp = np.zeros((2*S,2),dtype=int)
  for si,s in enumerate(smp):
    a,b = vert[s]
    i = V + si
    new_vert[i] = a+b
    new_smp[2*si]   = [s[0],   i]
    new_smp[2*si+1] = [   i,s[1]]

  new_vert = new_vert / np.linalg.norm(new_vert,axis=1)[:,None]
  return new_vert,new_smp


def circle(r=5):
  ''' 
  returns the outwardly oriented simplices of a circle
  '''
  vert = np.array([[1.0,0.0],
                   [0.0,1.0],
                   [-1.0,0.0],
                   [0.0,-1.0]])
  smp = np.array([[0,1],
                  [1,2],
                  [2,3],
                  [3,0]])
  for i in range(r):
    vert,smp = _circle_refine(vert,smp)

  return vert,smp  


def sphere(r=5):
  ''' 
  returns the outwardly oriented simplices of a sphere
  '''
  f = np.sqrt(2.0)/2.0
  vert = np.array([[ 0.0,-1.0, 0.0],
                   [  -f, 0.0,   f],
                   [   f, 0.0,   f],
                   [   f, 0.0,  -f],
                   [  -f, 0.0,  -f],
                   [ 0.0, 1.0, 0.0]])
  smp = np.array([[0,2,1],
                  [0,3,2],
                  [0,4,3],
                  [0,1,4],
                  [5,1,2],
                  [5,2,3],
                  [5,3,4],
                  [5,4,1]])

  for i in range(r):
    vert,smp = _sphere_refine(vert,smp)

  return vert,smp

