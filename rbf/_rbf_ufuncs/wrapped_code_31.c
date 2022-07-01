/******************************************************************************
 *                       Code generated with sympy 1.9                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                      This file is part of 'ufuncify'                       *
 ******************************************************************************/
#include "wrapped_code_31.h"
#include <math.h>

double autofunc0(double x0, double x1, double c0, double c1, double eps) {

   double autofunc0_result;
   autofunc0_result = 1.0/(pow(eps, 2)*(pow(-c0 + x0, 2) + pow(-c1 + x1, 2)) + 1);
   return autofunc0_result;

}
