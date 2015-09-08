/* It implements Regression Algorithms */

#ifndef LOGISTIC_H
#define LOGISTIC_H

#include <math.h>
#include "deep.h"

/* Logistic Regression */

double Logistic_Regression(gsl_matrix *X, gsl_vector *w, gsl_vector *Y); /* It executes the logistic regression */
double Logistic_RegressionPartialDerivative(gsl_matrix *X, gsl_vector *w, gsl_vector *Y, int j); /*It executes the partial derivative of variable j concerning logistic regression with Equation 21 as the cost function */
double h_logistic(gsl_vector *x, gsl_vector *w); /* It executes the hypothesis function */
gsl_vector *Logistic_Regression4Classification(gsl_matrix *X, gsl_vector *w); /* It executes the Logistic Regression for classification purposes */
#endif