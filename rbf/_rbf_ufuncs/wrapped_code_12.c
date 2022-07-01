/******************************************************************************
 *                       Code generated with sympy 1.9                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                      This file is part of 'ufuncify'                       *
 ******************************************************************************/
#include "wrapped_code_12.h"
#include <math.h>

double autofunc0(double x0, double c0, double eps) {

   double autofunc0_result;
   if (sqrt(pow(-c0 + x0, 2)) <= 0.0) {
      autofunc0_result = 0;
   }
   else {
      autofunc0_result = -pow(eps, 4)*pow(-c0 + x0, 4)*log(eps*sqrt(pow(-c0 + x0, 2)));
   }
   return autofunc0_result;

}
