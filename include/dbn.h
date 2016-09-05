/* It implements a Deep Belief Network */

#ifndef DBN_H
#define DBN_H

#include "rbm.h"

typedef struct _DBN{
    RBM **m;
    int n_layers;
}DBN;

/* Allocation and deallocation */
DBN *CreateDBN(int n_visible_units, gsl_vector *n_hidden_units, int n_labels, int n_layers); /* It allocates an DBN */
DBN *CreateNewDBN(int n_visible_units, int *n_hidden_units, int n_labels, int n_layers); /* It allocates an new DBN */
void DestroyDBN(DBN **d); /* It deallocates an DBN */

/* DBN information */
void DBNSaveWeights(DBN *d, char *path); /* It writes the weight matrix as PGM images */

/* DBN initialization */
void InitializeDBN(DBN *d); /* It initializes an DBN */

/* Bernoulli DBN training */
double BernoulliDBNTrainingbyContrastiveDivergence(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size); /* It trains a DBN for image reconstruction using Contrastive Divergence */
double BernoulliDBNTrainingbyContrastiveDivergenceWithDropout(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size, double *p); /* It trains a DBN with Dropout for image reconstruction using Contrastive Divergence */
double BernoulliDBNTrainingbyContrastiveDivergenceWithDropconnect(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size, double *p); /* It trains a DBN with Dropconnect for image reconstruction using Contrastive Divergence */
double BernoulliDBNTrainingbyPersistentContrastiveDivergence(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size); /* It trains a DBN for image reconstruction using Persistent Contrastive Divergence */
double BernoulliDBNTrainingbyPersistentContrastiveDivergenceWithDropout(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size, double *p); /* It trains a DBN with Dropout for image reconstruction using Persistent Contrastive Divergence */
double BernoulliDBNTrainingbyPersistentContrastiveDivergenceWithDropconnect(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size, double *p); /* It trains a DBN with Dropconnect for image reconstruction using Persistent Contrastive Divergence */
double BernoulliDBNTrainingbyFastPersistentContrastiveDivergence(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size); /* It trains a DBN for image reconstruction using Fast Persistent Contrastive Divergence */
double BernoulliDBNTrainingbyFastPersistentContrastiveDivergenceWithDropout(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size, double *p); /* It trains a DBN with Dropout for image reconstruction using Fast Persistent Contrastive Divergence */
double BernoulliDBNTrainingbyFastPersistentContrastiveDivergenceWithDropconnect(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size, double *p); /* It trains a DBN with Dropconnect for image reconstruction using Fast Persistent Contrastive Divergence */

/* Bernoulli DBN reconstruction */
double BernoulliDBNReconstruction(Dataset *D, DBN *d); /* It reconstructs an input dataset given a trained DBN */

/* Backpropagation fine-tuning (IN PROGRESS) */
gsl_vector *ForwardPass(gsl_vector *s, DBN *d); /* It executes the forward pass for a given sample s, and outputs the net's response for that sample */

/* Data conversion */
Subgraph *DBN2Subgraph(DBN *d, Dataset *D); /* It generates a subgraph using the learned features from the top layer of the DBN over the dataset */

/* Auxiliary functions */
void saveDBNParameters(DBN *d, char *file); /* It saves DBN weight matrixes and bias vectors */
void loadDBNParametersFromFile(DBN *d, char *file); /* It loads DBN weight matrixes and bias vectors from file */

#endif