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
void DestroyDBN(DBN **d); /* It deallocates an DBN */

/* DBN information */
void DBNSaveWeights(DBN *d, char *path); /* It writes the weight matrix as PGM images */

/* DBN initialization */
void InitializeDBN(DBN *d); /* It initializes an DBN */

/* Bernoulli DBN training */
double BernoulliDBNTrainingbyContrastiveDivergence(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size); /* It trains a DBN for image reconstruction using Contrastive Divergence */
double BernoulliDBNTrainingbyPersistentContrastiveDivergence(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size); /* It trains a DBN for image reconstruction using Persistent Contrastive Divergence */
double BernoulliDBNTrainingbyFastPersistentContrastiveDivergence(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size); /* It trains a DBN for image reconstruction using Fast Persistent Contrastive Divergence */

/* Bernoulli DBN reconstruction */
double BernoulliDBNReconstruction(Dataset *D, DBN *d); /* It reconstructs an input dataset given a trained DBN */

/* Backpropagation fine-tuning */
gsl_vector *ForwardPass(gsl_vector *s, DBN *d); /* It executes the forward pass for a given sample s, and outputs the net's response for that sample */

/* Image reconstruction */
//IplImage *DBNReconstructImage(DBN *d, IplImage *img); // It reconstructs an input image given a trained DBN

/* Data conversion */
Subgraph *DBN2Subgraph(DBN *d, Dataset *D); /* It generates a subgraph using the learned features from the top layer of the DBN over the dataset */

void saveDBNParameters(DBN *d, char *file); /* It saves DBN weight matrixes and bias vectors */

#endif
