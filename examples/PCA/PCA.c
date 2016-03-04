#include "OPF.h"
#include "deep.h"

int main(int argc, char **argv){
    if(argc != 4){
        fprintf(stderr,"\nusage PCA <input file> <output file> <percentage of the final number of dimensions>\n");
        exit(-1);
    }
    
    Subgraph *in = NULL, *out = NULL;
    double p = atof(argv[3]);
    
    in = ReadSubgraph(argv[1]);
    
    //out = PCA(in, p);
    gsl_matrix *M = NULL;
    M = Subgraph2gsl_matrix(in);
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