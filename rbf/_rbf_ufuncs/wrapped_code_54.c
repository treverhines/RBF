/******************************************************************************
 *                    Code generated with SymPy 1.13.0rc2                     *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                      This file is part of 'ufuncify'                       *
 ******************************************************************************/
#include "wrapped_code_54.h"
#include <math.h>

double autofunc0(double x0, double c0, double eps) {

   double autofunc0_result;
   double d0 = x0 - c0;
   double r2 = (d0*d0);
   double r = sqrt(r2);
   if (eps >= r) {
      if (1.0e-8*eps >= r) {
         autofunc0_result = 1;
      }
      else {
         autofunc0_result = pow(1 - r/eps, 5)*(1 + 5*r/eps + 8*r2/(eps*eps));
      }
   }
   else {
      autofunc0_result = 0;
   }
   return autofunc0_result;

}
