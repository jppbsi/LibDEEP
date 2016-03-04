#include "math_functions.h"
#include <gsl/gsl_statistics_double.h>

double SigmoidLogistic(double x){
    double y;
    
    y = 1.0/(1+exp(-x));
    
    return y;
}

double SoftPlus(double x){
    double y;

    y = log(1+exp(x));
    
    return y;
}

/*It computes the covariance matrix
M: input matrix each row is an observation and each column is a variable */
gsl_matrix *CovarianceMatrix(gsl_matrix *M){
    if(!M){
        fprintf(stderr,"\nThere is not matrix allocated @CovarianceMatrix\n");
        return NULL;
    }
    
    gsl_matrix *cov = NULL;
    gsl_vector_view a, b;
    int i, j;
    double v;

    cov = gsl_matrix_calloc(M->size2, M->size2);
    for (i = 0; i < M->size2; i++){
        for (j = i; j < M->size2; j++){
            a = gsl_matrix_column (M, i);
            b = gsl_matrix_column (M, j);
            v = gsl_stats_covariance (a.vector.data, a.vector.stride, b.vector.data, b.vector.stride, a.vector.size);
            gsl_matrix_set(cov, i, j, v); gsl_matrix_set(cov, j, i, v); /* covariance matrices are symmetric */
        }
    }
    
    return cov;
}