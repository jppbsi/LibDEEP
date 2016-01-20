#ifndef EPNN_H
#define EPNN_H

#include "OPF.h"

/* GSL libraries */
#include <gsl/gsl_matrix.h>

void EPNN(Subgraph *Train, Subgraph *Test, double sigma, gsl_vector *lNode, gsl_vector  *nsample4class, gsl_vector *alpha, gsl_vector *nGaussians);// Enhanced probabilistic neural network with local decision circles based on the Parzen-window estimation
double MaxDistance(Subgraph *graph);// Maximum Euclidian Distance of training data pairs
double MinDistance(Subgraph *graph);// Minimum Euclidian Distance of training data pairs
gsl_vector *HyperSphere(Subgraph *graph, double radius);// It calculates the hyper-sphere with radius r for each training node
gsl_vector *OrderedListLabel(Subgraph *Train, gsl_vector *nGaussians, gsl_vector *root);// Ordered List Label
gsl_vector *CountClasses(Subgraph *Train, gsl_vector *nGaussians, gsl_vector *root);// Count classes
gsl_vector *LoadLabels(Subgraph *Train);// Loading labels in training set
gsl_vector *LearnBestParameters(Subgraph *Train, Subgraph *Eval, int step, gsl_vector *lNode, gsl_vector  *nsample4class, double maxRadius, double minRadius, gsl_vector *nGaussians); // Learn best parameters for sigma and radius using linear-search
gsl_vector *gridSearch(Subgraph *Train, Subgraph *Eval, gsl_vector *lNode, gsl_vector  *nsample4class, double maxRadius, double minRadius, gsl_vector *nGaussians);  // Learn best parameters for sigma and radius using grid-search

#endif