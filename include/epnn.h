#ifndef EPNN_H
#define EPNN_H

#include "OPF.h"

/* GSL libraries */
#include <gsl/gsl_matrix.h>

//Enhanced probabilistic neural network with local decision circles based on the Parzen-window estimation
void epnn(Subgraph *Train, Subgraph *Test, double sigma, gsl_vector *lNode, gsl_vector  *nsample4class, gsl_vector *alpha, gsl_vector *nGaussians);

//opfcluster4epnn
gsl_vector **opfcluster4epnn(Subgraph *Train, gsl_vector **gaussians, int kmax); // opfcluster4epnn

//Auxiliary functions
gsl_vector *hyperSphere(Subgraph *graph, double radius);// It calculates the hyper-sphere with radius r for each training node
gsl_vector *orderedListLabel(Subgraph *Train, gsl_vector *nGaussians, gsl_vector *root);// Ordered List Label
gsl_vector *countClasses(Subgraph *Train, gsl_vector *nGaussians, gsl_vector *root);// Count classes
gsl_vector *loadLabels(Subgraph *Train);// Loading labels in training set
double maxDistance(Subgraph *graph);// Maximum Euclidian Distance of training data pairs
double minDistance(Subgraph *graph);// Minimum Euclidian Distance of training data pairs

//grid-search
gsl_vector *gridSearch(Subgraph *Train, Subgraph *Eval, double radius, int kmax);// Grid-search for k, sigma and radius

#endif