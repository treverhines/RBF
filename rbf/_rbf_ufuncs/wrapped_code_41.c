/******************************************************************************
 *                       Code generated with sympy 1.9                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                      This file is part of 'ufuncify'                       *
 ******************************************************************************/
#include "wrapped_code_41.h"
#include <math.h>

double autofunc0(double x0, double x1, double x2, double c0, double c1, double c2, double eps) {

   double autofunc0_result;
   autofunc0_result = exp(-1.0/2.0*(pow(-c0 + x0, 2) + pow(-c1 + x1, 2) + pow(-c2 + x2, 2))/pow(eps, 2));
   return autofunc0_result;

}
