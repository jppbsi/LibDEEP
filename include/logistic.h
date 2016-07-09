/* It implements Logistic Regression Algorithms */

#ifndef LOGISTIC_H
#define LOGISTIC_H

#include <math.h>
#include "deep.h"

/* Logistic Regression */

double Logistic_Regression(Subgraph *g, double *w); /* It executes the logistic regression */
double Logistic_RegressionPartialDerivative(Subgraph *g, double *w, int j); /* It executes the partial derivative of variable j concerning logistic regression with Equation 21 as the cost function */
double h_logistic(double *x, double *w, int n); /* It executes the hypothesis function */
double GradientDescentLogistic(Subgraph *g, double alpha, double *w); /* It executes optimization through Batch Gradient Descent */
double LogisticRegression_Fitting(Subgraph *g, double alpha, double *w); /* It fits a logistic regression model using Equation 21 as the error function optimized by Gradient Descent */
void Logistic_Regression4Classification(Subgraph *Test, double *w); /* It executes the Logistic Regression for classification purposes */

#endif