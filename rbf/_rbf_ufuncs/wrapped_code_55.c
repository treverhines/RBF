/******************************************************************************
 *                       Code generated with sympy 1.9                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                      This file is part of 'ufuncify'                       *
 ******************************************************************************/
#include "wrapped_code_55.h"
#include <math.h>

double autofunc0(double x0, double x1, double c0, double c1, double eps) {

   double autofunc0_result;
   if (eps >= sqrt(pow(-c0 + x0, 2) + pow(-c1 + x1, 2))) {
      if (1.0e-8*eps >= sqrt(pow(-c0 + x0, 2) + pow(-c1 + x1, 2))) {
         autofunc0_result = 1;
      }
      else {
         autofunc0_result = pow(1 - sqrt(pow(-c0 + x0, 2) + pow(-c1 + x1, 2))/eps, 5)*(1 + 5*sqrt(pow(-c0 + x0, 2) + pow(-c1 + x1, 2))/eps + 8*(pow(-c0 + x0, 2) + pow(-c1 + x1, 2))/pow(eps, 2));
      }
   }
   else {
      autofunc0_result = 0;
   }
   return autofunc0_result;

}
