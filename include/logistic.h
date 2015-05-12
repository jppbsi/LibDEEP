/* It implements Regression Algorithms */

#ifndef LOGISTIC_H
#define LOGISTIC_H

#include "deep.h"

/* Logistic Regression */

gsl_vector *LogisticRegression_Fitting(gsl_matrix *X, gsl_vector *Y, mac_prtFun Function, int FUNCTION_ID, ...); /* It fits a linear regression model using the Minimum Square Error as the error function */
double Logistic_Regression(gsl_matrix *X, gsl_vector *w, gsl_vector *Y); /* It executes the logistic regression */
double Logistic_RegressionPartialDerivative(gsl_matrix *X, gsl_vector *w, gsl_vector *Y, int j); /*It executes the partial derivative of variable j concerning logistic regression with Equation 21 as the cost function */
double h_logistic(gsl_vector *x, gsl_vector *w); /* It executes the hypothesis function */
#endif