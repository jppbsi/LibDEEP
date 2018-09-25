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
    double eta, lambda, alpha, t;
    double eta_min, eta_max; /* mininum and maximum values of the lerning rate */
    gsl_vector *v; /* visible layer neurons */
    gsl_vector *h; /* hidden layer neurons */
    gsl_matrix *W; /* weight matrix */
    gsl_matrix *U; /* weight matrix for labels */
    gsl_vector *a; /* visible neurons' bias */
    gsl_vector *b; /* hidden neurons' bias */
    gsl_vector *c; /* label neurons' bias */
    gsl_vector *r; /* hidden neurons' dropout bias */
    gsl_matrix *M; /* weight matrix dropconnect bias */
    gsl_vector *sigma; /* variance associated to each visible neuron for Gaussian visible units */
}RBM;

/* Allocation and deallocation */
RBM *CreateRBM(int n_visible_layers, int n_hidden_layers, int n_labels); /* It allocates an RBM */
RBM *CreateDRBM(int n_visible_units, int n_hidden_units, int n_labels, gsl_vector *sigma); /* It allocates a DRBM */
RBM *CreateNewDRBM(int n_visible_units, int n_hidden_units, int n_labels, double *sigma); /* It allocates a new DRBM */
void DestroyRBM(RBM **m); /* It deallocates an RBM */
void DestroyDRBM(RBM **m); /* It deallocates a DRBM */

/* RBM initialization */
void InitializeBias4VisibleUnits(RBM *m, Dataset *D); /* It initializes the bias of visible units according to Section 8.1 */
void InitializeBias4VisibleUnitsWithRandomValues(RBM *m); /* It initializes the bias of visible units with small random values [0,1] */
void InitializeBias4HiddenUnits(RBM *m); /* It initializes the bias of hidden units according to Section 8.1 */
void InitializeBias4DropoutHiddenUnits(RBM *m, double p); /* It initializes the bias of hidden units dropout */
void InitializeBias4DropconnectWeight(RBM *m, double p); /* It initializes the bias of dropconnect */
void InitializeBias4LabelUnits(RBM *m); /* It initializes the bias of label units */
void InitializeWeights(RBM *m); /* It initializes the weight matrix according to Section 8.1 */
void InitializeLabelWeights(RBM *m); /* It initializes the label weight matrix according to Section 8.1 */
void setVisibleLayer(RBM *m, gsl_vector *visible_layer); /* It sets the visible layer of a Restricted Boltzmann Machine */

/* RBM information */
void PrintWeights(RBM *m); /* It prints the weights */
void PrintLabelWeights(RBM *m); /* It prints the label weights */
void PrintVisibleUnitsBias(RBM *m); /* It prints the visible units' bias */
void PrintVisibleUnits(RBM *m); /* It prints the visible units */
void PrintHiddenUnits(RBM *m); /* It prints the hidden units */
void PrintHiddenDropoutUnits(RBM *m); /* It prints the hidden dropout units */
void PrintDropconnectWeight(RBM *m); /* It prints the dropconnect matrix mask */
void SaveRBMFeatures(char *s, Dataset *D, RBM *m); /* It saves the learned features from the hidden vector neurons */
void SaveWeightsWithoutCV(RBM *m, char *name, int indexHiddenUnit, int width, int height); /* It writes the weight matrix as PGM images without using CV */

/* Bernoulli-Bernoulli RBM training */
double BernoulliRBMTrainingbyContrastiveDivergence(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size); /* It trains a Bernoulli RBM by Constrative Divergence for image reconstruction (binary images) */
double BernoulliRBMTrainingbyContrastiveDivergencewithDropout(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size, double p); /* It trains a Bernoulli RBM by Constrative Divergence for image reconstruction (binary images) with Dropout */
double BernoulliRBMTrainingbyContrastiveDivergencewithDropconnect(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size, double p); /* It trains a Bernoulli RBM with Dropconnect by Constrative Divergence for image reconstruction (binary images) */
double BernoulliRBMTrainingbyPersistentContrastiveDivergence(Dataset *D, RBM *m, int n_epochs, int n_PCD_iterations, int batch_size); /* It trains a Bernoulli RBM by Persistent Constrative Divergence */
double BernoulliRBMTrainingbyPersistentContrastiveDivergencewithDropout(Dataset *D, RBM *m, int n_epochs, int n_PCD_iterations, int batch_size, double p); /* It trains a Bernoulli RBM by Persistent Constrative Divergence with Dropout */
double BernoulliRBMTrainingbyPersistentContrastiveDivergencewithDropconnect(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size, double p); /* It trains a Bernoulli RBM with Dropconnect by Persistent Constrative Divergence */
double BernoulliRBMTrainingbyFastPersistentContrastiveDivergence(Dataset *D, RBM *m, int n_epochs, int n_gibbs_sampling, int batch_size); /* It trains a Bernoulli RBM by Fast Persistent Constrative Divergence */
double BernoulliRBMTrainingbyFastPersistentContrastiveDivergencewithDropout(Dataset *D, RBM *m, int n_epochs, int n_gibbs_sampling, int batch_size, double p); /* It trains a Bernoulli RBM by Fast Persistent Constrative Divergence with Dropout */
double BernoulliRBMTrainingbyFastPersistentContrastiveDivergencewithDropconnect(Dataset *D, RBM *m, int n_epochs, int n_gibbs_sampling, int batch_size, double p); /* It trains a Bernoulli RBM with Dropconnect by Fast Persistent Constrative Divergence */
double DiscriminativeBernoulliRBMTrainingbyContrastiveDivergence(Dataset *D, RBM *m, int n_epochs, int n_gibbs_sampling, int batch_size); /* It trains a Discriminative Bernoulli RBM by Constrative Divergence for pattern classification */
double DiscriminativeBernoulliRBMTrainingbyContrastiveDivergencewithDropout(Dataset *D, RBM *m, int n_epochs, int n_gibbs_sampling, int batch_size, double p); /* It trains a Discriminative Bernoulli RBM with Dropout by Constrative Divergence for pattern classification */
double Bernoulli_TrainingRBMbyCD4DBM_BottomLayer(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size); /* It trains a Bernoulli RBM by Constrative Divergence for image reconstruction regarding DBMs at the bottom layer */
double Bernoulli_TrainingRBMbyCD4DBM_BottomLayerwithDropout(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size, double p); /* It trains a Bernoulli RBM with Dropout by Constrative Divergence for image reconstruction regarding DBMs at the bottom layer */
double Bernoulli_TrainingRBMbyCD4DBM_BottomLayerwithDropconnect(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size, double p); /* It trains a Bernoulli RBM with Dropconnect by Constrative Divergence for image reconstruction regarding DBMs at the bottom layer */
double Bernoulli_TrainingRBMbyCD4DBM_TopLayer(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size); /* It trains a Bernoulli RBM by Constrative Divergence for image reconstruction regarding DBMs at the top layer */
double Bernoulli_TrainingRBMbyCD4DBM_TopLayerwithDropout(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size, double p); /* It trains a Bernoulli RBM with Dropout by Constrative Divergence for image reconstruction regarding DBMs at the top layer */
double Bernoulli_TrainingRBMbyCD4DBM_TopLayerwithDropconnect(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size, double p); /* It trains a Bernoulli RBM with Dropconnect by Constrative Divergence for image reconstruction regarding DBMs at the top layer */
double Bernoulli_TrainingRBMbyCD4DBM_IntermediateLayers(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size); /* It trains a Bernoulli RBM by Constrative Divergence for image reconstruction regarding DBMs at the intermediate layers */
double Bernoulli_TrainingRBMbyCD4DBM_IntermediateLayerswithDropout(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size, double p); /* It trains a Bernoulli RBM with Dropout by Constrative Divergence for image reconstruction regarding DBMs at the intermediate layers */
double Bernoulli_TrainingRBMbyCD4DBM_IntermediateLayerswithDropconnect(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size, double p); /* It trains a Bernoulli RBM with Dropconnect by Constrative Divergence for image reconstruction regarding DBMs at the intermediate layers */
double Bernoulli_TrainingRBMbyPCD4DBM_BottomLayer(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size); /* It trains a Bernoulli RBM by Persistent Constrative Divergence for image reconstruction regarding DBMs at the bottom layer */
double Bernoulli_TrainingRBMbyPCD4DBM_TopLayer(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size); /* It trains a Bernoulli RBM by Persistent Constrative Divergence for image reconstruction regarding DBMs at the top layer */
double Bernoulli_TrainingRBMbyPCD4DBM_IntermediateLayers(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size); /* It trains a Bernoulli RBM by Constrative Divergence for image reconstruction regarding DBMs at the intermediate layers */

/* Gaussian-Bernoulli RBM training */
double GaussianBernoulliRBMTrainingbyContrastiveDivergence(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size); /* It trains a Gaussian-Bernoulli RBM by Constrative Divergence for image reconstruction (binary images) */
double GaussianBernoulliRBMTrainingbyContrastiveDivergencewithDropout(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size, double p); /* It trains a Gaussian-Bernoulli RBM with Dropout by Constrative Divergence for image reconstruction (binary images) */
double DiscriminativeGaussianBernoulliRBMTrainingbyContrastiveDivergence(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size); /* It trains a Discriminative Gaussian-Bernoulli RBM by Constrative Divergence for pattern classification */
double DiscriminativeGaussianBernoulliRBMTrainingbyContrastiveDivergencewithDropout(Dataset *D, RBM *m, int n_epochs, int n_CD_iterations, int batch_size, double p); /* It trains a Discriminative Gaussian-Bernoulli RBM with Dropout by Constrative Divergence for pattern classification */

/* Bernoulli RBM reconstruction/classification */
double BernoulliRBMReconstruction(Dataset *D, RBM *m); /* It reconstructs an input dataset given a trained RBM */
double GaussianBernoulliRBMReconstruction(Dataset *D, RBM *m); /* It reconstructs an input dataset given a trained Gaussian RBM */
double DiscriminativeBernoulliRBMClassification(Dataset *D, RBM *m); /* It classifies an input dataset given a trained RBM and it outputs the classification error */

/* Auxiliary functions */
double FreeEnergy(RBM *m, gsl_vector *v); /* It computes the pseudo-likelihood of a sample x in an RBM, and it assumes x is a binary vector */
double FreeEnergy4DRBM(RBM *m, int y, gsl_vector *x); /* It computes the free energy of a given label and a sample */
gsl_vector *getProbabilityTurningOnHiddenUnit(RBM *m, gsl_vector *v); /* It computes the probability of turning on a hidden unit j, as described by Equation 10 */
gsl_vector *getProbabilityDroppingVisibleUnitOut4TurningOnHiddenUnit(RBM *m, gsl_vector *r, gsl_vector *v); /* It computes the probability of dropping out visible units and turning on a hidden unit j, as described by Equation 11 */
gsl_vector *getProbabilityTurningOnHiddenUnit4Dropconnect(RBM *m, gsl_matrix *M, gsl_vector *v); /* It computes the probability of turning on a hidden unit j using a dropconnect mask */
gsl_vector *getProbabilityTurningOnHiddenUnit4DBM(RBM *m, gsl_vector *v); /* It computes the probability of turning on a hidden unit j considering a DBM at bottom layer */
gsl_vector *getProbabilityTurningOnHiddenUnit4DBM4Dropconnect(RBM *m, gsl_matrix *M, gsl_vector *v); /* It computes the probability of turning on a hidden unit j using a dropconnect mask considering a DBM at bottom layer */
gsl_vector *getProbabilityDroppingVisibleUnitOut4TurningOnHiddenUnit4DBM(RBM *m, gsl_vector *r, gsl_vector *v); /* It computes the probability of dropping visible units for turning on a hidden unit j considering a DBM at bottom layer using Equation 22 */
gsl_vector *getProbabilityTurningOnHiddenUnit4FPCD(RBM *m, gsl_vector *v, gsl_matrix *fast_W); /* It computes the probability of turning on a hidden unit j for FPCD */
gsl_vector *getProbabilityDroppingVisibleUnitOut4TurningOnHiddenUnit4FPCD(RBM *m, gsl_vector *r, gsl_vector *v, gsl_matrix *fast_W); /* It computes the probability of dropping out visible units and turning on a hidden unit j for FPCD */
gsl_vector *getProbabilityTurningOnHiddenUnit4FPCD4Dropconnect(RBM *m, gsl_matrix *M, gsl_vector *v, gsl_matrix *fast_W); /* It computes the probability of turning on a hidden unit j for FPCD with a dropconnect mask */
gsl_vector *getProbabilityTurningOnVisibleUnit(RBM *m, gsl_vector *h); /* It computes the probability of turning on a visible unit j, as described by Equation 11 */
gsl_vector *getProbabilityDroppingHiddenUnitOut4TurningOnVisibleUnit(RBM *m, gsl_vector *r, gsl_vector *h); /* It computes the probability of dropping out hidden units and turning on a visible unit j, as described by Equation 11 */
gsl_vector *getProbabilityTurningOnVisibleUnit4Dropconnect(RBM *m, gsl_matrix *M, gsl_vector *h); /* It computes the probability of turning on a visible unit j using a dropconnect mask */
gsl_vector *getProbabilityTurningOnVisibleUnit4DBM(RBM *m, gsl_vector *h); /* It computes the probability of turning on a visible unit j considering a DBM at top layer */
gsl_vector *getProbabilityTurningOnVisibleUnit4DBM4Dropconnect(RBM *m, gsl_matrix *M, gsl_vector *h); /* It computes the probability of turning on a visible unit j using a dropconnect mask considering a DBM at top layer */
gsl_vector *getProbabilityDroppingHiddenUnitOut4TurningOnVisibleUnit4DBM(RBM *m, gsl_vector *r, gsl_vector *h); /* It computes the probability of dropping hidden units for turning on a visible unit j considering a DBM at top layer */
gsl_vector *getProbabilityTurningOnVisibleUnit4FPCD(RBM *m, gsl_vector *h, gsl_matrix *fast_W); /* It computes the probability of turning on a visible unit j for FPCD */
gsl_vector *getProbabilityDroppingHiddenUnitOut4TurningOnVisibleUnit4FPCD(RBM *m, gsl_vector *r, gsl_vector *h, gsl_matrix *fast_W); /* It computes the probability of dropping out hidden units and turning on a visible unit j for FPCD */
gsl_vector *getProbabilityTurningOnVisibleUnit4FPCD4Dropconnect(RBM *m, gsl_matrix *M, gsl_vector *h, gsl_matrix *fast_W); /* It computes the probability of turning on a visible unit j for FPCD using a dropconnect mask */
gsl_vector *getProbabilityTurningOnHiddenUnit4Gaussian(RBM *m, gsl_vector *v, gsl_vector *sigma); /* It computes the probability of turning on a hidden unit j considering Gaussian RBMs */
gsl_vector *getProbabilityTurningOnHiddenUnit4Gaussian4Dropout(RBM *m, gsl_vector *r, gsl_vector *v, gsl_vector *sigma); /* It computes the probability of turning on a hidden unit j considering Gaussian RBMs with Dropout */
gsl_vector *getProbabilityTurningOnVisibleUnit4Gaussian(RBM *m, gsl_vector *h, gsl_vector *sigma); /* It computes the probability of turning on a visible unit i considering Gaussian RBMs */
gsl_vector *getProbabilityTurningOnVisibleUnit4Gaussian4Dropout(RBM *m, gsl_vector *r, gsl_vector *h, gsl_vector *sigma); /* It computes the probability of turning on a visible unit i considering Gaussian RBMs with Dropout */
gsl_vector *getDiscriminativeProbabilityTurningOnHiddenUnit(RBM *m, gsl_vector *y); /* It computes the probability of turning on a hidden unit j considering Discriminative RBMs and Bernoulli visible units */
gsl_vector *getDiscriminativeProbabilityTurningOnHiddenUnit4Dropout(RBM *m, gsl_vector *r, gsl_vector *y); /* It computes the probability of turning on a hidden unit j with Dropout considering Discriminative RBMs with Bernoulli visible units, i..e, p(h|y,x) */
gsl_vector *getDiscriminativeProbabilityTurningOnHiddenUnit4GaussianVisibleUnit(RBM *m, gsl_vector *y); /* It computes the probability of turning on a hidden unit j considering Discriminative RBMs and Gaussian visible units */
gsl_vector *getDiscriminativeProbabilityTurningOnHiddenUnit4GaussianVisibleUnit4Dropout(RBM *m, gsl_vector *r, gsl_vector *y); /* It computes the probability of turning on a hidden unit j with Dropout considering Discriminative RBMs and Gaussian visible units */
gsl_vector *getDiscriminativeProbabilityTurningOnVisibleUnit4GaussianVisibleUnit(RBM *m, gsl_vector *h); /* It computes the probability of turning on a visible unit i considering Discriminative RBMs and Gaussian visible units */
gsl_vector *getDiscriminativeProbabilityTurningOnVisibleUnit4GaussianVisibleUnit4Dropout(RBM *m, gsl_vector *r, gsl_vector *h); /* It computes the probability of turning on a visible unit i with Dropout considering Discriminative RBMs and Gaussian visible units */
gsl_vector *getDiscriminativeProbabilityLabelUnit(RBM *m); /* It computes the probability of label unit (y) given the hidden (h) one, i.e., P(y|h) */
double getReconstructionError(gsl_vector *input, gsl_vector *output); /* It computes the minimum square error among input and output */
double getPseudoLikelihood(RBM *m, gsl_vector *x); /* It computes the pseudo-likelihood of a sample x in an RBM */
void FASTgetProbabilityTurningOnHiddenUnit(RBM *m, gsl_vector *v, gsl_vector *prob_h); /* It computes the probability of turning on a hidden unit - Fast version */

#endif
