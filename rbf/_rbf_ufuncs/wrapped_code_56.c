/******************************************************************************
 *                       Code generated with sympy 1.9                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                      This file is part of 'ufuncify'                       *
 ******************************************************************************/
#include "wrapped_code_56.h"
#include <math.h>

double autofunc0(double x0, double x1, double x2, double c0, double c1, double c2, double eps) {

   double autofunc0_result;
   double d0 = x0 - c0;
   double d1 = x1 - c1;
   double d2 = x2 - c2;
   double r2 = (d0*d0) + (d1*d1) + (d2*d2);
   double r = sqrt(r2);
   if (eps >= r) {
      if (1.0e-8*eps >= r) {
         autofunc0_result = 1;
      }
      else {
         autofunc0_result = pow(1 - r/eps, 5)*(1 + 5*r/eps + 8*(r2)/(eps*eps));
      }
   }
   else {
      autofunc0_result = 0;
   }
   return autofunc0_result;

}
