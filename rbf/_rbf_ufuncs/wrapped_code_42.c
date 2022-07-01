/******************************************************************************
 *                       Code generated with sympy 1.9                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                      This file is part of 'ufuncify'                       *
 ******************************************************************************/
#include "wrapped_code_42.h"
#include <math.h>

double autofunc0(double x0, double c0, double eps) {

   double autofunc0_result;
   if (1.0e-8*eps >= sqrt(pow(-c0 + x0, 2))) {
      autofunc0_result = 1;
   }
   else {
      autofunc0_result = (1 + sqrt(3)*sqrt(pow(-c0 + x0, 2))/eps)*exp(-sqrt(3)*sqrt(pow(-c0 + x0, 2))/eps);
   }
   return autofunc0_result;

}
