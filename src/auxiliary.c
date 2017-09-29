#include "auxiliary.h"

/* Functions related to the Dataset struct */

/* It creates a dataset
Parameters: [size, nfeatures]
size: size of dataset
nfeatures: number of features */
Dataset *CreateDataset(int size, int nfeatures)
{
    Dataset *D = NULL;
    int i;

    D = (Dataset *)malloc(sizeof(Dataset));
    if (!D)
    {
        fprintf(stderr, "\nDataset not allocated @CreateDataset.\n");
        exit(-1);
    }

    D->size = size;
    D->nfeatures = nfeatures;

    D->sample = NULL;
    D->sample = (Sample *)malloc(D->size * sizeof(Sample));
    for (i = 0; i < D->size; i++)
        D->sample[i].feature = gsl_vector_alloc(D->nfeatures);

    return D;
}

/* It destroys a dataset
Parameters: [D]
D: dataset */
void DestroyDataset(Dataset **D)
{
    int i;
    if (*D)
    {
        if ((*D)->sample)
        {
            for (i = 0; i < (*D)->size; i++)
                gsl_vector_free((*D)->sample[i].feature);
            free((*D)->sample);
        }
        free(*D);
    }
}

/* It copies a given dataset
Parameters: [D]
D: dataset */
Dataset *CopyDataset(Dataset *d)
{
    Dataset *cpy = NULL;
    int i;

    if (d)
    {

        cpy = CreateDataset(d->size, d->nfeatures);
        cpy->nlabels = d->nlabels;

        for (i = 0; i < cpy->size; i++)
        {
            gsl_vector_memcpy(cpy->sample[i].feature, d->sample[i].feature);
            cpy->sample[i].label = d->sample[i].label;
        }
    }
    else
        fprintf(stderr, "\nThere is no dataset allocated @CopyDataset\n");

    return cpy;
}

/* It concatenates 2 subsets of a dataset
Parameters: [d1, d2]
d1: first dataset
d2: second dataset */
Dataset *ConcatenateDataset(Dataset *d1, Dataset *d2)
{
    Dataset *cpy = NULL;
    int i, j;

    if (d1 && d2)
    {

        cpy = CreateDataset(d1->size, (d1->nfeatures + d2->nfeatures));
        cpy->nlabels = d1->nlabels;

        for (i = 0; i < d1->size; i++)
        {
            for (j = 0; j < d1->nfeatures; j++)
            {
                gsl_vector_set(cpy->sample[i].feature, j, gsl_vector_get(d1->sample[i].feature, j));
                gsl_vector_set(cpy->sample[i].feature, (j + d1->nfeatures), gsl_vector_get(d2->sample[i].feature, j));
            }
            cpy->sample[i].label = d1->sample[i].label;
        }
    }
    else
        fprintf(stderr, "\nThere is no dataset allocated @CopyDataset\n");

    return cpy;
}

/* It undo concatenation of datasets
Parameters: [d1]
d1: dataset */
Dataset *UndoConcatenateDataset(Dataset *d1)
{
    Dataset *cpy = NULL;
    int i, j;

    if (d1)
    {
        cpy = CreateDataset(d1->size, (d1->nfeatures / 2));
        cpy->nlabels = d1->nlabels;

        for (i = 0; i < cpy->size; i++)
        {
            for (j = 0; j < cpy->nfeatures; j++)
                gsl_vector_set(cpy->sample[i].feature, j, gsl_vector_get(d1->sample[i].feature, j));
            cpy->sample[i].label = d1->sample[i].label;
        }
    }
    else
        fprintf(stderr, "\nThere is no dataset allocated @CopyDataset\n");

    return cpy;
}
/**********************************************/

/* Common auxiliary functions */

/* It waives a comment in a LibDEEP model file
Parameters: [fp]
fp: file */
void WaiveLibDEEPComment(FILE *fp)
{
    char c;

    fscanf(fp, "%c", &c);
    while ((c != '\n') && (!feof(fp)))
        fscanf(fp, "%c", &c);
}

/* It converts a Dataset to a Subgraph
Parameters: [D]
D: dataset */
Subgraph *Dataset2Subgraph(Dataset *D)
{
    Subgraph *g = NULL;
    int i, j;

    g = CreateSubgraph(D->size);
    g->nfeats = D->nfeatures;
    g->nlabels = D->nlabels;
    for (i = 0; i < g->nnodes; i++)
    {
        g->node[i].feat = AllocFloatArray(g->nfeats);
        g->node[i].truelabel = D->sample[i].label;
        g->node[i].label = D->sample[i].predict;
        g->node[i].position = i;
        for (j = 0; j < g->nfeats; j++)
            g->node[i].feat[j] = (float)gsl_vector_get(D->sample[i].feature, j);
    }

    return g;
}

/* It converts a Subgraph to a Dataset
Parameters: [g]
g: graph */
Dataset *Subgraph2Dataset(Subgraph *g)
{
    Dataset *D = NULL;
    int i, j;

    D = CreateDataset(g->nnodes, g->nfeats);
    D->nlabels = g->nlabels;

    for (i = 0; i < D->size; i++)
    {
        D->sample[i].label = g->node[i].truelabel;

        for (j = 0; j < D->nfeatures; j++)
            gsl_vector_set(D->sample[i].feature, j, g->node[i].feat[j]);
    }

    return D;
}

/* It converts an integer to a set of bits. Ex.: for a 3-bit representation, if label = 2, output = 010
Parameters: [l, n_bits]
l: label
n_bits: number of bits */
gsl_vector *label2binary_gsl_vector(int l, int n_bits)
{
    gsl_vector *y = NULL;

    y = gsl_vector_calloc(n_bits);
    gsl_vector_set_zero(y);
    gsl_vector_set(y, l - 1, 1.0);

    return y;
}

/* It converts a graph node to a gsl_vector
Parameters: [*x, n]
*x: float vector
n: size of vector */
gsl_vector *node2gsl_vector(float *x, int n)
{
    gsl_vector *v = NULL;
    int i;

    if (x)
    {
        v = gsl_vector_alloc(n);
        for (i = 0; i < v->size; i++)
            gsl_vector_set(v, i, x[i]);
        return v;
    }
    else
    {
        fprintf(stderr, "\nThere is no node allocated @node2gsl_vector\n");
        return NULL;
    }
}

/* It converts a graph node to a double vector
Paramaters: [*x, n]
*x: float vector
n: size of vector */
double *node2double_vector(float *x, int n)
{
    double *v = NULL;
    int i;

    if (x)
    {
        v = (double *)calloc(n, sizeof(double));
        for (i = 0; i < n; i++)
            v[i] = x[i];
        return v;
    }
    else
    {
        fprintf(stderr, "\nThere is no node allocated @node2double_vector\n");
        return NULL;
    }
}

/* It converts a graph to a gsl_matrix
Paramaters: [g]
g: graph */
gsl_matrix *Subgraph2gsl_matrix(Subgraph *g)
{
    if (!g)
    {
        fprintf(stderr, "\nThere is no graph allocated @Subgraph2gsl_matrix.\n");
        return NULL;
    }

    gsl_matrix *out = NULL;
    int i, j;

    out = gsl_matrix_calloc(g->nnodes, g->nfeats);

    for (i = 0; i < g->nnodes; i++)
        for (j = 0; j < g->nfeats; j++)
            gsl_matrix_set(out, i, j, (double)g->node[i].feat[j]);

    return out;
}

/* It generates a random seed */
unsigned long int random_seed_deep()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (tv.tv_sec + tv.tv_usec);
}
