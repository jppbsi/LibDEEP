#include <math.h> 
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

#define PI 3.1415926536

/* Mathematical functions */

double SigmoidLogistic(double x); /* It computes the Sigmoid Logistic function */
double SoftPlus(double x); /* It computes the Soft Plus function */
double Euclidean_Distance(gsl_vector *x, gsl_vector *y); /* It computes the Euclidean Distance between two vectors */

gsl_matrix *CovarianceMatrix(gsl_matrix *M); /* It computes the covariance matrix */
void ComputeVariances(int size, gsl_matrix *mu, gsl_matrix **cov); /* It computes the variance as sigma = 2*d_{avg}, in which this last one stands for the average distance between centers */
double Determinant(gsl_matrix *m); /* It computes the matrix determinant */
double GaussianDensity(gsl_matrix **cov, gsl_matrix *mu, gsl_vector *x, int j); /* It computes the a multivariate gaussian density of a sample x */
gsl_matrix *PseudoInverse(gsl_matrix *A); /* It computes the Moore–Penrose Pseudoinverse A+ = VS^+U^T */