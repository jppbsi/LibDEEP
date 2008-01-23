/* It implements Regression Algorithms */

#ifndef REGRESSION_H
#define REGRESSION_H

#include "deep.h"

/* Linear Regression */

/* It fits a linear regression model using the Minimum Square Error as the error function */
void LinearRegression_Fitting(Subgraph *Train, gsl_vector *w, mac_prtFun Optimization_Func, ...);

#endif