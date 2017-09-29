#include "pca.h"
#include "math_functions.h"
#include "auxiliary.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

/* It performs the standard Principal Component Analisys algorithm
Parameters: [in, p]
in: input graph
p: percentage of the total number of dimensions to generate the new dataset ]0,1] */
Subgraph *PCA(Subgraph *in, double p)
{
    if (!in)
    {
        fprintf(stderr, "\nNo input graph defined @PCA.\n");
        return NULL;
    }

    if (p > 1 || p <= 0)
    {
        fprintf(stderr, "\nWrong percentage range @PCA\n");
        return NULL;
    }

    Subgraph *out = NULL, *tmp = NULL;
    int k = ceil(p * in->nfeats), i, j;
    gsl_matrix *cov = NULL, *tcov = NULL, *M = NULL, *X = gsl_matrix_alloc(in->nfeats, in->nfeats);
    gsl_matrix *V = gsl_matrix_alloc(in->nfeats, in->nfeats), *tmp_cov = NULL, *aux = NULL, *r = NULL;
    gsl_vector *work = gsl_vector_alloc(in->nfeats), *S = gsl_vector_alloc(in->nfeats), *v = NULL;

    if (k < 1)
        k = 1;
    tmp = CopySubgraph(in);
    opf_NormalizeFeatures(tmp);
    M = Subgraph2gsl_matrix(tmp);
    cov = CovarianceMatrix(M);

    /* It performs SVD */
    //result = gsl_linalg_SV_decomp_mod(cov, X, V, S, work);

    /* Picking the first k eigeinvectors */
    tmp_cov = gsl_matrix_alloc(cov->size1, k);
    for (i = 0; i < tmp_cov->size1; i++)
        for (j = 0; j < tmp_cov->size2; j++)
            gsl_matrix_set(tmp_cov, i, j, gsl_matrix_get(cov, i, j));

    tcov = gsl_matrix_calloc(tmp_cov->size2, tmp_cov->size1);
    gsl_matrix_transpose_memcpy(tcov, tmp_cov);
    aux = gsl_matrix_calloc(in->nfeats, 1);
    r = gsl_matrix_calloc(k, 1);

    out = CreateSubgraph(in->nnodes);
    out->nfeats = k;
    out->nlabels = in->nlabels;
    for (i = 0; i < out->nnodes; i++)
    {
        out->node[i].feat = AllocFloatArray(k);
        out->node[i].truelabel = in->node[i].truelabel;
        out->node[i].position = in->node[i].position;
        v = node2gsl_vector(in->node[i].feat, in->nfeats);
        gsl_matrix_set_col(aux, 0, v);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tcov, aux, 0.0, r); /* It performs the mapping to the k-dimensional space */
        for (j = 0; j < k; j++)
            out->node[i].feat[j] = (float)gsl_matrix_get(r, j, 0);
        gsl_vector_free(v);
    }

    gsl_matrix_free(cov);
    gsl_matrix_free(tcov);
    gsl_matrix_free(tmp_cov);
    gsl_matrix_free(aux);
    gsl_matrix_free(r);
    gsl_matrix_free(X);
    gsl_matrix_free(V);
    gsl_matrix_free(M);
    gsl_vector_free(work);
    gsl_vector_free(S);
    DestroySubgraph(&tmp);

    return out;
}
