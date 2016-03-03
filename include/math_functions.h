#include <math.h> 
#include <gsl/gsl_matrix.h>

double SigmoidLogistic(double x);
double SoftPlus(double x);

gsl_matrix *CovarianceMatrix(gsl_matrix *M); /*It computes the covariance matrix */