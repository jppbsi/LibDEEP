/* The Artificial Neural Network implemented here follows the technical reported available in papers/RBM/guideTR.pdf */

#ifndef ANN_H
#define ANN_H

/* GSL libraries */
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>

#include "auxiliary.h"
#include "math_functions.h"

void k_Means(Subgraph *g, gsl_matrix *mu, int k);                                        /* It computes the cluster centroids by k-means clustering */
gsl_matrix *ComputeHiddenLayerOutput(Subgraph *g, gsl_matrix *mu, gsl_matrix **cov);     /* It computes matrix Phi */
gsl_matrix *TrainANNbyOPF(Subgraph *g, gsl_matrix *mu, gsl_matrix **cov, int kmax);      /* It trains the neural network by OPF and outputs the matrix of weights */
gsl_matrix *TrainANNbyKMeans(Subgraph *g, gsl_matrix *mu, gsl_matrix **cov, int kvalue); /* It trains the neural network by K-Means and outputs the matrix of weights */
void ClassifyANN(Subgraph *g, gsl_matrix *mu, gsl_matrix **cov, gsl_matrix *w);          /* It classifies an ANN */

#endif