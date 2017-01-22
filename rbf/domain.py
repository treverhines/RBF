''' 
This module contains functions that generate simplices defining 
commonly used domains.
'''
import numpy as np
from scipy.spatial import Delaunay

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
  Returns the outwardly oriented simplices of a circle
  
  Parameters
  ----------
  r : int, optional
    refinement order
      
  Returns
  -------
  vert : (N,2) float array
  smp : (M,2) int array
    
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

def logo():
  ''' 
  Returns a domain resembling the Python logo
  
  Returns
  -------
  vert : (76,2) float array

  smp : (76,2) int array
    Rows 0-12 define the simplices for the edges of the logo. Rows 
    12-44 define the simplices for the top eye, and the remaining rows 
    make up the simplices for the bottom eye.
  
  '''  
  edge_vert = np.array([[-0.5,0.0],[-0.5,1.0],[0.0,1.0],[0.0,1.5],
                        [1.0,1.5],[1.0,1.0],[1.5,1.0],[1.5,0.0],
                        [1.0,0.0],[1.0,-0.5],[0.0,-0.5],[0.0,0.0]])
  edge_smp = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],
                       [6,7],[7,8],[8,9],[9,10],[10,11],[11,0]])
  eye1_vert,eye1_smp = circle(3)
  eye2_vert,eye2_smp = circle(3)
  eye1_vert *= 0.15
  eye2_vert *= 0.15
  eye1_vert += np.array([0.25,1.25])
  eye2_vert += np.array([0.75,-0.25])
  eye1_smp += 12
  eye2_smp += 44
  vert = np.vstack((edge_vert,eye1_vert,eye2_vert))
  smp = np.vstack((edge_smp,eye1_smp,eye2_smp))
  return vert,smp


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


def topography(zfunc,xlim,ylim,depth,n=10):
  ''' 
  Returns a collection of simplices that form a box-like domain. The 
  bottom and sides of the box are flat and to top of the box is shaped 
  according to a user-specified topography function. This is only for 
  three-dimensional domains.
  
  Parameters
  ----------
  zfunc : callable
    Function that takes a (N,2) array of surface coordinates and 
    returns an (N,) array of elevations at those coordinates. This 
    function should taper to zero at the edges of the domain.

  xlim : (2,) array
    x bounds of the domain

  ylim : (2,) array
    y bounds of the domain
  
  depth : float
    Depth of the domain. This should be positive.
  
  n : int, optional
    Number of simplices along the x and y axis. Increasing this 
    number results in a better resolved domain. 

  Returns
  -------
  vert : (P,3) float array
    Vertices of the domain.
    
  smp : (Q,3) int array
    Indices of the vertices that make up each simplex. Rows 0-2 make 
    up the simplices for the bottom of the domain. Rows 2-10 make up 
    the sides of the domain. The remaining rows make up the simplices 
    for the surface.

  '''
  base_vert = np.array([[0.0,0.0,-1.0],[0.0,0.0,0.0],[0.0,1.0,-1.0],
                        [0.0,1.0,0.0],[1.0,0.0,-1.0],[1.0,0.0,0.0],
                        [1.0,1.0,-1.0],[1.0,1.0,0.0]])
  # scale vertices to user specs
  base_vert[:,0] *= (xlim[1] - xlim[0])
  base_vert[:,0] += xlim[0]
  base_vert[:,1] *= (ylim[1] - ylim[0])
  base_vert[:,1] += ylim[0]
  base_vert[:,2] *= depth
  base_smp = np.array([[0,2,6],[0,4,6],
                       [0,1,3],[0,2,3],[0,1,4],[1,5,4],
                       [4,5,7],[4,6,7],[2,3,7],[2,6,7]])
  # make grid of top vertices
  x = np.linspace(xlim[0],xlim[1],n)
  y = np.linspace(ylim[0],ylim[1],n)
  xg,yg = np.meshgrid(x,y)
  xf,yf,zf = xg.flatten(),yg.flatten(),np.zeros(n**2)
  # find the interior top vertices and change their z value according 
  # to zfunc
  interior = ((xf != xlim[0]) & (xf != xlim[1]) &
              (yf != ylim[0]) & (yf != ylim[1]))
  interior_xy = np.array([xf[interior],yf[interior]]).T
  zf[interior] = zfunc(interior_xy)
  # make the top simplices
  top_vert = np.array([xf,yf,zf]).T
  top_smp = Delaunay(top_vert[:,:2]).simplices
  # combine the base with the top
  vert = np.vstack((base_vert,top_vert))
  smp = np.vstack((base_smp,8 + top_smp))
  return vert,smp

