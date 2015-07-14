/* It implements a Deep Belief Network */

#ifndef DBN_H
#define DBN_H

#include "rbm.h"

typedef struct _DBM{
    RBM **m;
    int n_layers;
}DBM;

/* Allocation and deallocation */
DBM *CreateDBM(int n_visible_layers, int n_hidden_layers, int n_labels, int n_layers); /* It allocates an DBM */
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

/* Bernoulli DBM reconstruction */
double BernoulliDBMReconstruction(Dataset *D, DBM *d); /* It reconstructs an input dataset given a trained DBM */

/* Backpropagation fine-tuning */
gsl_vector *ForwardPass(gsl_vector *s, DBM *d); /* It executes the forward pass for a given sample s, and outputs the net's response for that sample */

#endif
