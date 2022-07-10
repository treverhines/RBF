/******************************************************************************
 *                       Code generated with sympy 1.9                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                      This file is part of 'ufuncify'                       *
 ******************************************************************************/
#include "wrapped_code_30.h"
#include <math.h>

double autofunc0(double x0, double c0, double eps) {

   double autofunc0_result;
   double d0 = x0 - c0;
   double r2 = (d0*d0);
   double r = sqrt(r2);
   autofunc0_result = 1.0/((eps*eps)*r2 + 1);
   return autofunc0_result;

}
