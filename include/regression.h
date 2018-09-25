/* It implements Linear Regression Algorithms */

#ifndef REGRESSION_H
#define REGRESSION_H

#include "deep.h"

/* Linear Regression */

double Linear_Regression(Subgraph *g, double *w); /* It executes the linear regression */
double Linear_RegressionPartialDerivative(Subgraph *g, double *w, int j); /* It executes the partial derivative of variable j concerning linear regression with MSE as the cost function */
double h_linear_regression(double *x, double *w, int n); /* It executes the hypothesis function */
double LinearRegression_Fitting(double **X, double *Y, int m, int n, double alpha, double *w); /* It fits a linear regression model using Equation 5 as the error function optimized by Gradient Descent */
double GradientDescentLinear(Subgraph *g, double alpha, double *w); /* It executes optimization through Batch Gradient Descent */

#endif