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
DBM *CreateNewDBM(int n_visible_layer_neurons,  int *n_hidden_units, int n_labels, int n_layers); /* It allocates a new DBM */
void DestroyDBM(DBM **d); /* It deallocates a DBM */

/* Bernoulli DBM initialization */
void InitializeDBM(DBM *d); /* It initializes a DBM */

/* Bernoulli DBM pre-training */
double GreedyPreTrainingDBM(Dataset *D, DBM *d, int n_epochs, int n_samplings, int batch_size, int LearningType); /* It performs DBM greedy pre-training step */
double GreedyPreTrainingDBMwithDropout(Dataset *D, DBM *d, int n_epochs, int n_samplings, int batch_size, int LearningType, double *p); /* It performs DBM with Dropout greedy pre-training step */
double GreedyPreTrainingDBMwithDropconnect(Dataset *D, DBM *d, int n_epochs, int n_samplings, int batch_size, int LearningType, double *p); /* It performs DBM with Dropconnect greedy pre-training step */

/* Bernoulli DBM reconstruction */
double BernoulliDBMReconstruction(Dataset *D, DBM *d);/* It reconstructs an input dataset given a trained DBM */

/* Auxiliary functions */
gsl_vector *getProbabilityTurningOnDBMIntermediateLayersOnDownPass(RBM *m, gsl_vector *h, RBM *beneath_layer); /* It computes the probability of turning on an intermediate layer of a DBM, as show in Eq. 28 and 29 */
void saveDBMParameters(DBM *d, char *file); /* It saves DBM weight matrixes and bias vectors */
void loadDBMParametersFromFile(DBM *d, char *file); /* It loads DBM weight matrixes and bias vectors from file */

#endif
