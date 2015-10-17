/* It implements Regression Algorithms */

#ifndef REGRESSION_H
#define REGRESSION_H

#include "deep.h"

/* Linear Regression */

double Linear_Regression(Subgraph *g, gsl_vector *w); /* It executes the linear regression */
double Linear_RegressionPartialDerivative(Subgraph *g, gsl_vector *w, int j); /*It executes the partial derivative of variable j concerning linear regression with MSE as the cost function */
double h_linear_regression(gsl_vector *x, gsl_vector *w); /* It executes the hypothesis function */

#endif