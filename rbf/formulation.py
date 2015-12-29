#!/usr/bin/env python
import sympy as sp
import numpy as np
sp.init_printing()

def lamb(x,diff=(0,0)):
  return 

def mu(x,diff=(0,0)):
  return 

def norm1(x):
  return 

def norm2(x):
  return 

def norm3(x):
  return 

dim = 3
x = sp.symbols('x0:%s' % dim)
n = sp.symbols('n0:%s' % dim)
L = sp.Function('L')(*x)
M = sp.Function('M')(*x)
u = (sp.Function('u0')(*x),
     sp.Function('u1')(*x),
     sp.Function('u2')(*x))

#sym2num = {L:lamb,M:mu,
#           sp.Integer(1):1.0,
#           sp.Integer(2):2.0,
#           x[0]:0,x[1]:1,x[2]:2}


half = sp.Rational(1,2)
dirs = range(dim)

strain = [[half*(u[j].diff(x[i]) + u[i].diff(x[j])) for i in dirs] for j in dirs]
strain = sp.Matrix(strain)
stress = L*sp.eye(dim)*sp.trace(strain) + 2*M*strain
PDEs = [sum(stress[i,j].diff(x[j]) for j in dirs) for i in dirs]
BCs = [sum(stress[i,j]*n[j] for j in dirs) for i in dirs]

def derivative_order(expr):
  '''
  checks to see if expr is a Derivative, if it is then returns the 
  base expression and the variables its derivatives are with respect
  to.  If expr is is Not a Derivative then returns (expr,())
  '''
  if expr.is_Derivative:
    return expr.expr,expr.variables
  else:
    return expr,()


def symbolic_coeffs_and_diffs(expr,u):
  '''
  returns the coefficients for each term containing u or a derivative 
  of u. Also returns the variables that derivatives of u are with 
  respect to
  '''
  # convert expr to a list of terms
  expr = expr.expand()
  expr = expr.as_ordered_terms()
  # throw out terms not containing f
  expr = [i for i in expr if i.has(u)]
  coeffs = []
  diffs = []
  for e in expr:
    # if the expression is a product then expand it into multipliers
    if e.is_Mul:
      e = sp.flatten(e.as_coeff_mul())
    else:
      e = [sp.Integer(1),e]  

    # find multipliers without the queried term
    without_u = [i for i in e if not i.has(u)] 
    coeffs += [without_u]

    # find multipliers with the queried term
    with_u = [i for i in e if i.has(u)]
    assert len(with_u) == 1, (
      'term contains multiple multipliers with %s' % u)

    base,diff = derivative_order(with_u[0])
    assert base == u, (
      'cannot find the derivatives with respect to %s for the '
      'expression %s' % (u,base))
      
    diffs += diff,

  return zip(coeffs,diffs)
  

def rmap(val,mappings):
  '''
  recursively map the values in val using the mapping dictionary
  '''
  if not hasattr(val,'__iter__'):
    try:
      return mappings[val]
    except KeyError:
      print('WARNING: cannot map %s, attempting to coerce it to a '
            'float' % val)
      return float(val)

  else: 
    out = []
    for i in val:
      out += [rmap(i,mappings)]

    return out


def reformat_diff(diff,dirs):
  '''
  converts diff from a collection of differentiation directions to
  the count for each differentiation direction
  '''
  diff = list(diff)
  assert all([i in dirs for i in diff]), (
    'not all differentiation directions provided')

  out = [0]*len(dirs)
  for i,xi in enumerate(dirs):
    out[i] = diff.count(xi)

  return tuple(out)


def function_product(*args):
  '''
  takes scalar valued functions and makes a function which returns a
  product of the input functions.  The resulting function passes the
  same positional and key word arguments to each input function
  '''
  def fprod(*fprod_args,**fprod_kwargs):  
    out = 1.0
    for i in args:
      if not hasattr(i,'__call__'):
        out *= i
      else:
        out *= i(*fprod_args,**fprod_kwargs)
    return out

  return fprod


def function_sum(*args):
  '''
  takes a scalar valued functions of one variable and returns 
  a function which returns a sum of the input functions.    
  '''
  def fsum(*fsum_args,**fsum_kwargs):  
    out = 0.0
    for i in args:
      if not hasattr(i,'__call__'):
        out += i
      else:
        out += i(*fsum_args,**fsum_kwargs)
    return out

  return fsum
    

def numerical_coeffs_and_diffs(expr,u,dirs=None,mapping=None):
  if dirs is None:
    dirs = ()
  if mapping is None:
    mapping = {}

  out = []
  a = symbolic_coeffs_and_diffs(expr,u)
  for coeff,diff in a:
    coeff = rmap(coeff,mapping)   
    diff = reformat_diff(diff,dirs) 
    out += (function_product(*coeff),diff),

  # I need to combine like terms in the out list to cut
  # down the computation time
  return out    

mapping = {sp.Integer(1):1.0,sp.Integer(2):2.0, 
           L:lamb,
           n[0]:norm1,n[1]:norm2,n[2]:norm3,  
           L.diff(x[0]):lambda x:lamb(x,diff=(1,0,0)),
           L.diff(x[1]):lambda x:lamb(x,diff=(0,1,0)),
           L.diff(x[2]):lambda x:lamb(x,diff=(0,0,1)),
           M:mu,
           M.diff(x[0]):lambda x:mu(x,diff=(1,0,0)),
           M.diff(x[1]):lambda x:mu(x,diff=(0,1,0)),
           M.diff(x[2]):lambda x:mu(x,diff=(0,0,1))}

import modest

modest.tic()
for i in dirs:
  for j in dirs:
    #out = numerical_coeffs_and_diffs(PDEs[i],u[j],dirs=x,mapping=mapping)
    out = numerical_coeffs_and_diffs(BCs[i],u[j],dirs=x,mapping=mapping)
    print(out)

print(modest.toc())
#sp.pprint(out)
#sp.pprint(rmap(out,sym2num))

