/* It implements a Deep Belief Network */

#ifndef DBM_H
#define DBM_H

#include "rbm.h"


typedef struct _DBM{
    RBM **m;
    int n_layers;
}DBM;

/* Allocation and deallocation */
DBM *CreateDBM(int n_visible_layer_neurons,  int n_labels, int n_hidden_layers,  ...);
//DBM *CreateDBM(int n_visible_layers, int n_hidden_layers, int n_labels, int n_layers); /* It allocates an DBM */
void DestroyDBM(DBM **d); /* It deallocates an DBM */

/* DBM information */
void DBMSaveWeights(DBM *d, char *path); /* It writes the weight matrix as PGM images */

/* DBM initialization */
void InitializeDBM(DBM *d); /* It initializes an DBM */

/* RBM initialization */
void InitializeRBM(RBM *r);/* It initializes an RBM randonly */

/* Paste DBM Parameters */
void PasteDBMParameters(RBM *r, RBM *r2);/*  paste doubled paramns to the original size */

/* DBM pre-training */
double GreedyPreTrainingAlgorithmForADeepBoltzmannMachine(Dataset *D, DBM *d, int n_epochs, int n_CD_iterations, int batch_size); /* It sets the initial parameters of DBM */

double DBMReconstruction(Dataset *D, DBM *d);

/* Probability of turning on intermadiate layers in a DBM */
gsl_vector *getProbabilityTurningOnHiddenUnitDBM(RBM *rbm,RBM *next, gsl_vector *v, double beta);/* It computes the probability of turning on a hidden unit j, as described by Equation 38 - An Efficient Learning Procedure for Deep Boltzmann Machines */

/* Probability of turning on the Top layers in a DBM */
gsl_vector *getProbabilityTurningOnHiddenUnitDBMLastLayer(RBM *m, gsl_vector *v, double beta);/* It computes the probability of turning on a hidden unit j, as described by Equation 40 - An Efficient Learning Procedure for Deep Boltzmann Machines */

/* Probability of turning on the First layers in a DBM */
gsl_vector *getProbabilityTurningOnVisibleUnitDBMFirstLayer(RBM *m, gsl_vector *h, double beta);/* It computes the probability of turning on a visible unit j, as described by Equation 41 */
#endif
