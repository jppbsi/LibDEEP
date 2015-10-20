/* It implements a Deep Bolztmann Machine */

#ifndef DBM_H
#define DBM_H

#include "rbm.h"

typedef struct _DBM{
    RBM **m;
    int n_layers;
}DBM;

/* Allocation and deallocation */
DBM *CreateDBM(int n_visible_layer_neurons,  gsl_vector *n_hidden_units, int n_labels); /* It allocates a DBM */
void DestroyDBM(DBM **d); /* It deallocates a DBM */

/* DBM initialization */
void InitializeDBM(DBM *d); /* It initializes a DBM */

/* DBM pre-training */
double GreedyPreTrainingDBM(Dataset *D, DBM *d, int n_epochs, int n_samplings, int batch_size, int LearningType); /* It performs DBM greedy pre-training step */

/* DBM information */
void DBMSaveWeights(DBM *d, char *path); /* It writes the weight matrix as PGM images */

/* Paste DBM Parameters */
void PasteDBMParameters(RBM *r, RBM *r2);/*  paste doubled paramns to the original size */

double DBMReconstruction(Dataset *D, DBM *d);

/* Probability of turning on intermadiate layers in a DBM */
gsl_vector *getProbabilityTurningOnHiddenUnitDBM(RBM *rbm,RBM *next, gsl_vector *v, double beta);/* It computes the probability of turning on a hidden unit j, as described by Equation 38 - An Efficient Learning Procedure for Deep Boltzmann Machines */

/* Probability of turning on the Top layers in a DBM */
gsl_vector *getProbabilityTurningOnHiddenUnitDBMLastLayer(RBM *m, gsl_vector *v, double beta);/* It computes the probability of turning on a hidden unit j, as described by Equation 40 - An Efficient Learning Procedure for Deep Boltzmann Machines */

/* Probability of turning on the First layers in a DBM */
gsl_vector *getProbabilityTurningOnVisibleUnitDBMFirstLayer(RBM *m, gsl_vector *h, double beta);/* It computes the probability of turning on a visible unit j, as described by Equation 41 */
#endif
