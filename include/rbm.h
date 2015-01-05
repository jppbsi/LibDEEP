/* The Restricted Bolztmann Machine implemented here follows the technical reported available in papers/RBM/guideTR.pdf */

#ifndef RBM_H
#define RBM_H

/* GSL libraries */
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>

#include "auxiliary.h"
#include "math_functions.h"

typedef struct _RBM{
    int n_visible_layer_neurons, n_hidden_layer_neurons, n_labels;
    double eta, lambda, alpha;
    gsl_vector *v; //visible layer neurons
    gsl_vector *h; //hidden layer neurons
    gsl_matrix *W; //weight matrix
    gsl_matrix *U; //weight matrix for labels
    gsl_vector *a;//visible neurons' bias
    gsl_vector *b;//hidden neurons' bias
    gsl_vector *c;//label neurons' bias
    gsl_vector *sigma; //variance associated to each visible neuron for Gaussian visible units
}RBM;

/* Allocation and deallocation */
RBM *CreateRBM(int n_visible_layers, int n_hidden_layers, int n_labels); // It allocates an RBM
RBM *CreateDRBM(int n_visible_units, int n_hidden_units, int n_labels, gsl_vector *sigma); // It allocates a DRBM
void DestroyRBM(RBM **m); // It deallocates an RBM
void DestroyDRBM(RBM **m); // It deallocates a DRBM

/* RBM initialization */
void InitializeBias4VisibleUnits(RBM *m, Dataset *D); // It initializes the bias of visible units according to Section 8.1
void InitializeBias4VisibleUnitsWithRandomValues(RBM *m); // It initializes the bias of visible units with small random values [0,1]
void InitializeBias4HiddenUnits(RBM *m); // It initializes the bias of hidden units according to Section 8.1
void InitializeBias4LabelUnits(RBM *m); // It initializes the bias of label units
void InitializeWeights(RBM *m); // It initializes the weight matrix according to Section 8.1
void InitializeLabelWeights(RBM *m); // It initializes the label weight matrix according to Section 8.1
void setVisibleLayer(RBM *m, gsl_vector *visible_layer); // It sets the visible layer of a Restricted Boltzmann Machine

/* RBM information */
void PrintWeights(RBM *m); // It prints the weights
void PrintLabelWeights(RBM *m); // It prints the label weights
void PrintVisibleUnitsBias(RBM *m); // It prints the visible units' bias
void PrintVisibleUnits(RBM *m); // It prints the visible units
void SaveWeights(RBM *m, char *path, int height, int width); // It writes the weight matrix as PGM images

/* Bernoulli-Bernoulli RBM training */
double BernoulliRBMTrainingbyContrastiveDivergence(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size); // It trains a Bernoulli RBM by Constrative Divergence for image reconstruction (binary images)
double DiscriminativeBernoulliRBMTrainingbyContrastiveDivergence(Dataset *D, RBM *m, int n_epochs, int batch_size); // It trains a Discriminative Bernoulli RBM by Constrative Divergence for pattern classification

/*Gaussian-Bernoulli RBM training */
double DiscriminativeGaussianBernoulliRBMTrainingbyContrastiveDivergence(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size); // It trains a Discriminative Gaussian-Bernoulli RBM by Constrative Divergence for pattern classification

/* Bernoulli RBM reconstruction/classification */
double BernoulliRBMReconstruction(Dataset *D, RBM *m); /* It reconstructs an input dataset given a trained RBM */
double DiscriminativeBernoulliRBMClassification(Dataset *D, RBM *m); /* It classifies an input dataset given a trained RBM and it outputs the classification error */

/* RMB image reconstruction */
//IplImage *ReconstructImage(RBM *m, IplImage *input); // It reconstructs an input image given a trained RBM

/* Auxiliary functions */
double FreeEnergy(RBM *m, int y, gsl_vector *x); /* It computes the free energy of a given label and a sample */
gsl_vector *getProbabilityTurningOnHiddenUnit(RBM *m, gsl_vector *v); // It computes the probability of turning on a hidden unit j, as described by Equation 10
gsl_vector *getProbabilityTurningOnVisibleUnit(RBM *m, gsl_vector *h); // It computes the probability of turning on a visible unit j, as described by Equation 11
gsl_vector *getDiscriminativeProbabilityTurningOnHiddenUnit(RBM *m, gsl_vector *y); //It computes the probability of turning on a hidden unit j considering Discriminative RBMs and Bernoulli visible units
gsl_vector *getDiscriminativeProbabilityTurningOnHiddenUnit4GaussianVisibleUnit(RBM *m, gsl_vector *y); //It computes the probability of turning on a hidden unit j considering Discriminative RBMs and Gaussian visible units
gsl_vector *getDiscriminativeProbabilityTurningOnVisibleUnit4GaussianVisibleUnit(RBM *m, gsl_vector *h); //It computes the probability of turning on a visible unit i considering Discriminative RBMs and Gaussian visible units
gsl_vector *getDiscriminativeProbabilityLabelUnit(RBM *m); /* It computes the probability of label unit (y) given the hidden (h) one, i.e., P(y|h) */
double getReconstructionError(gsl_vector *input, gsl_vector *output); // It computes the minimum square error among input and output

#endif