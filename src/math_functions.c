#include "math_functions.h"
#include <gsl/gsl_statistics_double.h>

/* Mathematical functions */

/* It computes the Sigmoid Logistic function
Parameters: [x]
x: double value */
double SigmoidLogistic(double x){
    double y;
    
    y = 1.0/(1+exp(-x));
    
    return y;
}

/* It computes the Soft Plus function
Parameters: [x]
x: double value */
double SoftPlus(double x){
    double y;

    y = log(1+exp(x));
    
    return y;
}

/* It computes the Euclidean Distance between two vectors
Parameters: [x, y]
x: first vector
y: second vector */
double Euclidean_Distance(gsl_vector *x, gsl_vector *y){
    double sum = 0.0;
    int i;
    
    for(i = 0; i < x->size; i++)
        sum += pow(gsl_vector_get(x,i)-gsl_vector_get(y,i),2);
    
    return sqrt(sum);
}

/* It computes the covariance matrix
Parameters: [M]
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
            gsl_matrix_set(cov, i, j, v); gsl_matrix_set(cov, j, i, v); /* Covariance matrices are symmetric */
        }
    }
    
    return cov;
}

/* It computes the variance as sigma = 2*d_{avg}, in which this last one stands for the average distance between centers
Parameters: [size, mu, cov]
size: size of vector array - number of features
mu: mean matrix
cov: covariance matrix */
void ComputeVariances(int size, gsl_matrix *mu, gsl_matrix **cov){
    int i, j;
    double max_distance = FLT_MIN, avg_distance, distance;
    gsl_vector *x = NULL, *y= NULL;
    
    x = gsl_vector_calloc(size);
    y = gsl_vector_calloc(size);
    
    for(i = 0; i < mu->size1; i++){
        gsl_matrix_get_row(x, mu, i);
        for(j = 0; j < mu->size2; j++){
            if(i != j){
                gsl_matrix_get_row(y, mu, j);
                distance = Euclidean_Distance(x, y);
                if(distance > max_distance)
                    max_distance = distance;
            }
        }
    }
    gsl_vector_free(x);
    gsl_vector_free(y);
    
    avg_distance = max_distance / mu->size1;
    for(i = 0; i < mu->size1; i++)
        for(j = 0; j < cov[0]->size1; j++)
            gsl_matrix_set(cov[i], j, j, 2*avg_distance); 
}

/* It computes the matrix determinant
Parameters: [m]
m: input matrix */
double Determinant(gsl_matrix *m){
    double det;
    gsl_permutation *p = NULL;
    int s;
    
    p = gsl_permutation_alloc(m->size2);
    gsl_linalg_LU_decomp(m, p, &s);
    det  = (double)gsl_linalg_LU_det(m, s);
    
    gsl_permutation_free(p);
    
    return det;
}

/* It computes the a multivariate gaussian density of a sample x
Parameters: [cov, mu, x, j]
cov: covariance matrix
mu: mean values matrix
x: sample array
j: position */
double GaussianDensity(gsl_matrix **cov, gsl_matrix *mu, gsl_vector *x, int j){
    gsl_matrix *inv = NULL;
    gsl_vector *v = NULL, *f = NULL;
    double g = 0.0, lside, rside, aux;
    int i, z, k;
    
    /* Allocating variables */
    inv = gsl_matrix_calloc(cov[j]->size1, cov[j]->size2);
    v = gsl_vector_calloc(cov[j]->size1);
    f = gsl_vector_calloc(cov[j]->size1);
    
    gsl_matrix_memcpy(inv, cov[j]);
    gsl_linalg_cholesky_decomp(inv);
    gsl_linalg_cholesky_invert(inv);
    
    lside = 1.0/(pow(2*PI,x->size/2.0)*sqrt(Determinant(cov[j])));
    
    /* It computes (x-mu) */
    for(i = 0; i < v->size; i++)
        gsl_vector_set(v, i, gsl_vector_get(x, i)-gsl_matrix_get(mu, j, i));
    
    /* It computes (x-mu)*sigma^{-1} */
    for(z = 0; z < inv->size2; z++){
        aux = 0.0; k = 0;
        for(i = 0; i < inv->size1; i++)
            aux+=(gsl_vector_get(v, k++)*gsl_matrix_get(inv, i, z));
        gsl_vector_set(f, z, aux);
    }
    
    /* It computes (x-mu)^{-1}*(x-mu) */
    rside = 0.0;
    for(i = 0; i < v->size; i++)
        rside+=(gsl_vector_get(f, i)*gsl_vector_get(v, i));
    
    g = lside*exp(-0.5*rside);
    
    gsl_matrix_free(inv);
    gsl_vector_free(v);
    gsl_vector_free(f);
    
    return g;
}

/* It computes the MooreÐPenrose Pseudoinverse A+ = VS^+U^T
Parameters: [A]
A: input matrix */
gsl_matrix *PseudoInverse(gsl_matrix *A){
    gsl_matrix *inv = NULL, *V = NULL, *Ut = NULL, *B = NULL, *S = NULL, *U = NULL;
    gsl_vector *S_aux = NULL, *work = NULL;
    double aux;
    int i;
    
    inv = gsl_matrix_calloc(A->size2, A->size1);
    U = gsl_matrix_calloc(A->size1, A->size2);
    Ut = gsl_matrix_calloc(U->size2, U->size1); /* Transpose of U */
    V = gsl_matrix_calloc(A->size2, A->size2);
    S = gsl_matrix_calloc(A->size2, A->size2);
    S_aux = gsl_vector_calloc(A->size2);
    work = gsl_vector_calloc(A->size2);

    gsl_matrix_memcpy(U, A);
    gsl_linalg_SV_decomp (U, V, S_aux, work); /* A = U.S.V^T */
    gsl_matrix_transpose_memcpy (Ut, U); /* Computing U^T */
    
    /* S^+ can be obtained by S[i] = 1/S[i], since the iverse of a diagonal matrix is computed iverting the elements of its diagonal */
    gsl_matrix_set_zero(S);
    for(i = 0; i < S_aux->size; i++){
        aux = gsl_vector_get(S_aux, i);
        if(aux > 0.0000000001) gsl_matrix_set(S, i, i, 1.0/gsl_vector_get(S_aux, i));
    }
    B = gsl_matrix_calloc(A->size2, A->size1);
    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,	1.0, S, Ut, 0.0, B); /* Calculating  B = S^+U^T */
        
    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,	1.0, V, B, 0.0, inv); /* Calculating  A^+ = VB = VS^+U^T */
    
    gsl_matrix_free(V);
    gsl_matrix_free(B);
    gsl_matrix_free(U);
    gsl_matrix_free(S);
    gsl_matrix_free(Ut);
    gsl_vector_free(S_aux);
    gsl_vector_free(work);
    
    return inv;
}