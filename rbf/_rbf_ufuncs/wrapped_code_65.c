/******************************************************************************
 *                       Code generated with sympy 1.9                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                      This file is part of 'ufuncify'                       *
 ******************************************************************************/
#include "wrapped_code_65.h"
#include <math.h>

double autofunc0(double x0, double x1, double x2, double c0, double c1, double c2, double eps) {

   double autofunc0_result;
   if (eps >= sqrt(pow(-c0 + x0, 2) + pow(-c1 + x1, 2) + pow(-c2 + x2, 2))) {
      if (1.0e-8*eps >= sqrt(pow(-c0 + x0, 2) + pow(-c1 + x1, 2) + pow(-c2 + x2, 2))) {
         autofunc0_result = 1;
      }
      else {
         autofunc0_result = (1.0/3.0)*pow(1 - sqrt(pow(-c0 + x0, 2) + pow(-c1 + x1, 2) + pow(-c2 + x2, 2))/eps, 6)*(3 + 18*sqrt(pow(-c0 + x0, 2) + pow(-c1 + x1, 2) + pow(-c2 + x2, 2))/eps + 35*(pow(-c0 + x0, 2) + pow(-c1 + x1, 2) + pow(-c2 + x2, 2))/pow(eps, 2));
      }
   }
   else {
      autofunc0_result = 0;
   }
   return autofunc0_result;

}
