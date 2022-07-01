/******************************************************************************
 *                       Code generated with sympy 1.9                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                      This file is part of 'ufuncify'                       *
 ******************************************************************************/
#include "wrapped_code_14.h"
#include <math.h>

double autofunc0(double x0, double x1, double x2, double c0, double c1, double c2, double eps) {

   double autofunc0_result;
   if (sqrt(pow(-c0 + x0, 2) + pow(-c1 + x1, 2) + pow(-c2 + x2, 2)) <= 0.0) {
      autofunc0_result = 0;
   }
   else {
      autofunc0_result = -pow(eps, 4)*pow(pow(-c0 + x0, 2) + pow(-c1 + x1, 2) + pow(-c2 + x2, 2), 2)*log(eps*sqrt(pow(-c0 + x0, 2) + pow(-c1 + x1, 2) + pow(-c2 + x2, 2)));
   }
   return autofunc0_result;

}
