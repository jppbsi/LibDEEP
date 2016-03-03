#include "OPF.h"
#include "deep.h"

int main(int argc, char **argv){
    if(argc != 4){
        fprintf(stderr,"\nusage PCA <input file> <output file> <percentage of the final number of dimensions>\n");
        exit(-1);
    }
    
    Subgraph *in = NULL, *out = NULL;
    double p = atof(argv[3]);
    
    //out = PCA(in, p);
    gsl_matrix *M = gsl_matrix_alloc(2,2);
    gsl_matrix_set(M, 0, 0, 1);gsl_matrix_set(M, 0, 1, 2);gsl_matrix_set(M, 1, 0, 1); gsl_matrix_set(M, 1, 1, 2);
    gsl_matrix *cov = CovarianceMatrix(M);
    
    int i, j;
    for(i = 0; i < cov->size1; i++)
        for(j = 0; j < cov->size2; j++)
            fprintf(stderr,"\ncov[%d][%d]: %lf", i, j, gsl_matrix_get(cov, i, j));
    
    gsl_matrix_free(M);
    gsl_matrix_free(cov);
    DestroySubgraph(&in);
    DestroySubgraph(&out);
    
    return 0;
}