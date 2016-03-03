#include "pca.h"

/* It performs the standard Principal Component Analisys algorithm
Parameters
in: input graph
p: percentage of the total number of dimensions to generate the new dataset ]0,1] */
Subgraph *PCA(Subgraph *in, int p){
    if(!in){
        fprintf(stderr,"\nNo input graph defined @PCA.\n");
        return NULL;
    }
    
    if(p > 1 || p <= 0){
        fprintf(stderr,"\nWrong percentage range @PCA\n");
        return NULL;
    }
    
    Subgraph *out = NULL, *tmp = NULL;
    int d = ceil(p*in->nfeats);
    gsl_matrix *cov = NULL;
    
    tmp = CopySubgraph(in);
    
    opf_NormalizeFeatures(tmp);
    cov = CovarianceMatrix(tmp);
    
    gsl_matrix_free(cov);
    DestroySubgraph(&tmp);
    
    return out;
}