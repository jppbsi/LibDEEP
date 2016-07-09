#include <math.h> 
#include <gsl/gsl_matrix.h>

/* Mathematical functions */

double SigmoidLogistic(double x); /* It computes the Sigmoid Logistic function */
double SoftPlus(double x); /* It computes the Soft Plus function */
gsl_matrix *CovarianceMatrix(gsl_matrix *M); /* It computes the covariance matrix */