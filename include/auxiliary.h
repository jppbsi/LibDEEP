#ifndef AUXILIARY_H
#define AUXILIARY_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dirent.h>

/* GSL libraries */
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <stdarg.h>

/* LibOPF library */
#include "OPF.h"

typedef struct _Sample
{
    gsl_vector *feature;
    int label, predict;
} Sample;

typedef struct _Dataset
{
    int size, nfeatures, nlabels;
    Sample *sample;
} Dataset;

/* Functions related to the Dataset struct */
Dataset *CreateDataset(int size, int nfeatures);       /* It creates a dataset */
void DestroyDataset(Dataset **D);                      /* It destroys a dataset */
Dataset *CopyDataset(Dataset *d);                      /* It copies a given dataset */
Dataset *ConcatenateDataset(Dataset *d1, Dataset *d2); /* It concatenates 2 subsets of a dataset */
Dataset *UndoConcatenateDataset(Dataset *d1);          /* It undo concatenation of datasets */

/* Common auxiliary functions */
void WaiveLibDEEPComment(FILE *fp);                     /* It waives a comment in a LibDEEP model file */
Subgraph *Dataset2Subgraph(Dataset *D);                 /* It converts a Dataset to a Subgraph */
Dataset *Subgraph2Dataset(Subgraph *g);                 /* It converts a Subgraph to a Dataset */
gsl_vector *label2binary_gsl_vector(int l, int n_bits); /* It converts an integer to a set of bits. Ex.: for a 3-bit representation, if label = 2, output = 010 */
gsl_vector *node2gsl_vector(float *x, int n);           /* It converts a graph node to a gsl_vector */
double *node2double_vector(float *x, int n);            /* It converts a graph node to a double vector */
gsl_matrix *Subgraph2gsl_matrix(Subgraph *g);           /* It converts a graph to a gsl_matrix */
unsigned long int random_seed_deep();                   /* It generates a random seed */

#endif