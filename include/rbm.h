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
    double eta, lambda, alpha,t;
    double eta_min, eta_max; //mininum and maximum values of the lerning rate
    gsl_vector *v; //visible layer neurons
    gsl_vector *h; //hidden layer neurons
    gsl_matrix *W; //weight matrix
    gsl_matrix *U; //weight matrix for labels
    gsl_vector *a;//visible neurons' bias
    gsl_vector *b;//hidden neurons' bias
    gsl_vector *c;//label neurons' bias
    gsl_vector *r;//hidden neurons' dropout bias
    gsl_vector *s;//visible neurons' dropout bias
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
void InitializeBias4DropoutHiddenUnits(RBM *m, double p); // It initializes the bias of hidden units dropout
void InitializeBias4DropoutVisibleUnits(RBM *m, double q); // It initializes the bias of visible units dropout
void InitializeBias4LabelUnits(RBM *m); // It initializes the bias of label units
void InitializeWeights(RBM *m); // It initializes the weight matrix according to Section 8.1
void InitializeLabelWeights(RBM *m); // It initializes the label weight matrix according to Section 8.1
void setVisibleLayer(RBM *m, gsl_vector *visible_layer); // It sets the visible layer of a Restricted Boltzmann Machine

/* RBM information */
void PrintWeights(RBM *m); // It prints the weights
void PrintLabelWeights(RBM *m); // It prints the label weights
void PrintVisibleUnitsBias(RBM *m); // It prints the visible units' bias
void PrintVisibleUnits(RBM *m); // It prints the visible units
void PrintHiddenUnits(RBM *m); // It prints the hidden units
void PrintVisibleDropoutUnits(RBM *m); // It prints the visible dropout units
void PrintHiddenDropoutUnits(RBM *m); // It prints the hidden dropout units
void SaveWeights(RBM *m, char *path, int height, int width); // It writes the weight matrix as PGM images
void SaveWeightsWithoutCV(RBM *m, char *name, int indexHiddenUnit, int width, int height);// It writes the weight matrix as PGM images without using CV

/* Bernoulli-Bernoulli RBM training */
double BernoulliRBMTrainingbyContrastiveDivergence(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size); // It trains a Bernoulli RBM by Constrative Divergence for image reconstruction (binary images)
double BernoulliRBMTrainingbyContrastiveDivergencewithDropout(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size, double p, double q); // It trains a Bernoulli RBM by Constrative Divergence for image reconstruction (binary images) with Dropout
double BernoulliRBMTrainingbyPersistentContrastiveDivergence(Dataset *D, RBM *m, int n_epochs, int n_PCD_iterations, int batch_size); // It trains a Bernoulli RBM by Persistent Constrative Divergence
double BernoulliRBMTrainingbyFastPersistentContrastiveDivergence(Dataset *D, RBM *m, int n_epochs, int n_gibbs_sampling, int batch_size); // It trains a Bernoulli RBM by Fast Persistent Constrative Divergence
double DiscriminativeBernoulliRBMTrainingbyContrastiveDivergence(Dataset *D, RBM *m, int n_epochs, int batch_size); // It trains a Discriminative Bernoulli RBM by Constrative Divergence for pattern classification
double Bernoulli_TrainingRBMbyCD4DBM_BottomLayer(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size); // It trains a Bernoulli RBM by Constrative Divergence for image reconstruction regarding DBMs at the bottom layer
double Bernoulli_TrainingRBMbyCD4DBM_TopLayer(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size); // It trains a Bernoulli RBM by Constrative Divergence for image reconstruction regarding DBMs at the top layer 
double Bernoulli_TrainingRBMbyCD4DBM_IntermediateLayers(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size); // It trains a Bernoulli RBM by Constrative Divergence for image reconstruction regarding DBMs at the intermediate layers
double Bernoulli_TrainingRBMbyPCD4DBM_BottomLayer(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size);/* It trains a Bernoulli RBM by Persistent Constrative Divergence for image reconstruction regarding DBMs at the bottom layer */
double Bernoulli_TrainingRBMbyPCD4DBM_TopLayer(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size);/* It trains a Bernoulli RBM by Persistent Constrative Divergence for image reconstruction regarding DBMs at the top layer */
double Bernoulli_TrainingRBMbyPCD4DBM_IntermediateLayers(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size);/* It trains a Bernoulli RBM by Constrative Divergence for image reconstruction regarding DBMs at the intermediate layers */

//double Bernoulli_TrainingRBMbyFPCD4DBM_BottomLayer(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size);/* It trains a Bernoulli RBM by Persistent Fast Constrative Divergence for //image reconstruction regarding DBMs at the bottom layer */
//double Bernoulli_TrainingRBMbyFPCD4DBM_TopLayer(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size);/* It trains a Bernoulli RBM by Persistent Fast Constrative Divergence for image reconstruction regarding DBMs at the top layer */
//double Bernoulli_TrainingRBMbyFPCD4DBM_IntermediateLayers(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size);/* It trains a Bernoulli RBM by Fast Constrative Divergence for image reconstruction regarding DBMs at the intermediate layers */

/*Gaussian-Bernoulli RBM training */
double DiscriminativeGaussianBernoulliRBMTrainingbyContrastiveDivergence(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size); // It trains a Discriminative Gaussian-Bernoulli RBM by Constrative Divergence for pattern classification

/* Bernoulli RBM reconstruction/classification */
double BernoulliRBMReconstruction(Dataset *D, RBM *m); /* It reconstructs an input dataset given a trained RBM */
double BernoulliRBMReconstructionwithDropout(Dataset *D, RBM *m, double p); /* It reconstructs an input dataset given a trained RBM with Dropout */
double DiscriminativeBernoulliRBMClassification(Dataset *D, RBM *m); /* It classifies an input dataset given a trained RBM and it outputs the classification error */

/* RMB image reconstruction */
//IplImage *ReconstructImage(RBM *m, IplImage *input); // It reconstructs an input image given a trained RBM

/* Auxiliary functions */
double FreeEnergy(RBM *m, gsl_vector *v); /* It computes the pseudo-likelihood of a sample x in an RBM, and it assumes x is a binary vector */
double FreeEnergy4DRBM(RBM *m, int y, gsl_vector *x); /* It computes the free energy of a given label and a sample */
gsl_vector *getProbabilityTurningOnHiddenUnit(RBM *m, gsl_vector *v); // It computes the probability of turning on a hidden unit j, as described by Equation 10
gsl_vector *getProbabilityTurningOnHiddenUnit4DBM(RBM *m, gsl_vector *v); // It computes the probability of turning on a hidden unit j considering a DBM at bottom layer
gsl_vector *getProbabilityTurningOnHiddenUnit4FPCD(RBM *m, gsl_vector *v, gsl_matrix *fast_W); // It computes the probability of turning on a hidden unit j for FPCD
gsl_vector *getProbabilityTurningOnVisibleUnit(RBM *m, gsl_vector *h); // It computes the probability of turning on a visible unit j, as described by Equation 11
gsl_vector *getProbabilityTurningOnVisibleUnit4DBM(RBM *m, gsl_vector *h);// It computes the probability of turning on a visible unit j considering a DBM at top layer
gsl_vector *getProbabilityTurningOnVisibleUnit4FPCD(RBM *m, gsl_vector *h, gsl_matrix *fast_W); // It computes the probability of turning on a visible unit j for FPCD
gsl_vector *getDiscriminativeProbabilityTurningOnHiddenUnit(RBM *m, gsl_vector *y); //It computes the probability of turning on a hidden unit j considering Discriminative RBMs and Bernoulli visible units
gsl_vector *getDiscriminativeProbabilityTurningOnHiddenUnit4GaussianVisibleUnit(RBM *m, gsl_vector *y); //It computes the probability of turning on a hidden unit j considering Discriminative RBMs and Gaussian visible units
gsl_vector *getDiscriminativeProbabilityTurningOnVisibleUnit4GaussianVisibleUnit(RBM *m, gsl_vector *h); //It computes the probability of turning on a visible unit i considering Discriminative RBMs and Gaussian visible units
gsl_vector *getDiscriminativeProbabilityLabelUnit(RBM *m); /* It computes the probability of label unit (y) given the hidden (h) one, i.e., P(y|h) */
gsl_vector *getProbabilityDroppingHiddenUnitOut4TurningOnVisibleUnit(RBM *m, gsl_vector *r, gsl_vector *h); // It computes the probability of dropping out hidden units and turning on a visible unit j, as described by Equation 11
gsl_vector *getProbabilityDroppingVisibleUnitOut4TurningOnHiddenUnit(RBM *m, gsl_vector *s, gsl_vector *v); // It computes the probability of dropping out visible units and turning on a hidden unit j, as described by Equation 11
double getReconstructionError(gsl_vector *input, gsl_vector *output); // It computes the minimum square error among input and output
double getPseudoLikelihood(RBM *m, gsl_vector *x); //it computes the pseudo-likelihood of a sample x in an RBM

void FASTgetProbabilityTurningOnHiddenUnit(RBM *m, gsl_vector *v, gsl_vector *prob_h);
void FASTgetProbabilityTurningOnHiddenUnit4FPCD(RBM *m, gsl_vector *v, gsl_matrix *fast_W, gsl_vector *fast_b, gsl_vector *prob_h);
void FASTgetProbabilityTurningOnVisibleUnit4FPCD(RBM *m, gsl_vector *h, gsl_matrix *fast_W, gsl_vector *fast_a, gsl_vector *prob_v);

#endif
