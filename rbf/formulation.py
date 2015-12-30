#!/usr/bin/env python
import sympy as sp
import numpy as np
import logging
import modest
logger = logging.getLogger(__name__)


class FormulationError(Exception):
  pass


def unique(x):
  '''
  returns unique values of x
  '''
  out = []
  for v in x:
    if out.count(v) == 0:
      out.append(v)

  return out 


def indices(x,val):
  '''
  return indices of x which equal val
  '''
  out = []
  for i,v in enumerate(x):
    if v == val:
      out.append(i)

  return out    
 

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
  # throw out terms not containing u
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
    if not (len(with_u) == 1):
      raise FormulationError(
        'the term %s has multiple occurrences of %s' % (sp.prod(e),u))

    base,diff = derivative_order(with_u[0])
    if not (base == u):
      raise FormulationError( 
        'cannot express %s as a differential operation of %s' % (base,u))
      
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
      logger.warning('cannot map %s, attempting to coerce it to a '
            'float' % val)
      print('WARNING: cannot map %s, attempting to coerce it to a '
            'float' % val)
      return float(val)

  else: 
    out = []
    for i in val:
      out += [rmap(i,mappings)]

    return out


def reformat_diff(diff,ivars):
  '''
  converts diff from a collection of differentiation directions to
  the count for each differentiation direction
  '''
  diff = list(diff)
  if not all([i in ivars for i in diff]):
    raise FormulationError(
      'not all differentiation directions provided')

  out = [0]*len(ivars)
  for i,xi in enumerate(ivars):
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
    

def coeffs_and_diffs(expr,u,ivar,mapping=None):
  if len(ivar) == 0:
    raise FormulationError(
      'at least one independent variable must be provides')  

  if mapping is None:
    mapping = {}

  coeff_list = []
  diff_list = []
  for coeff,diff in symbolic_coeffs_and_diffs(expr,u):
    coeff = function_product(*rmap(coeff,mapping))
    diff = reformat_diff(diff,ivar) 
    coeff_list += [coeff]
    diff_list += [diff]

  compressed_diff_list = unique(diff_list)
  compressed_coeff_list = []
  for d in compressed_diff_list:
    # find the indices of matching diff tuples
    idx = indices(diff_list,d)
    # find the coefficients for associated with the matching diff 
    # tuples
    coeffs_with_same_diff = [coeff_list[i] for i in idx]
    # sum up coefficients    
    compressed_coeff_list += [function_sum(*coeffs_with_same_diff)]
    
  out = zip(compressed_coeff_list,compressed_diff_list)
  return out    


def evaluate_coeffs_and_diffs(cd,*args,**kwargs):
  out = []
  for c,d in cd:
    c_evaluated = c(*args,**kwargs)
    # if the coefficient evaluates to zeros then ignore the term
    if c_evaluated != 0.0:
      out += (c(*args,**kwargs),d),

  return out

