/******************************************************************************
 *                       Code generated with sympy 1.9                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                      This file is part of 'ufuncify'                       *
 ******************************************************************************/
#include "wrapped_code_16.h"
#include <math.h>

double autofunc0(double x0, double x1, double c0, double c1, double eps) {

   double autofunc0_result;
   double d0 = x0 - c0;
   double d1 = x1 - c1;
   double r2 = (d0*d0) + (d1*d1);
   double r = sqrt(r2);
   if (r <= 0.0) {
      autofunc0_result = 0;
   }
   else {
      autofunc0_result = (eps*eps*eps)*(r*r*r);
   }
   return autofunc0_result;

}
