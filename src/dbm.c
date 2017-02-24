#include "dbm.h"

/* Allocation and deallocation */

/* It allocates a DBM
Parameters: [n_visible_layer_neurons, *n_hidden_units, n_labels]
n_visible_layer_neurons: number of visible neurons
*n_hidden_units: array of hidden neurons number
n_labels: number of labels */
DBM *CreateDBM(int n_visible_layer_neurons, gsl_vector *n_hidden_units, int n_labels){
    DBM *d = NULL;
    int i;

    d = (DBM *)malloc(sizeof(DBM));
    d->n_layers = n_hidden_units->size;
    d->m = (RBM **)malloc(d->n_layers*sizeof(RBM *));

    /* Only the first layer has the number of visible inputs equals to the number of features */
    d->m[0] = CreateRBM(n_visible_layer_neurons, (int)gsl_vector_get(n_hidden_units, 0), n_labels);
    for(i = 1; i < d->n_layers; i++)
	d->m[i] = CreateRBM((int)gsl_vector_get(n_hidden_units, i-1), (int)gsl_vector_get(n_hidden_units, i), n_labels);

    return d;
}

/* It allocates a new DBM
Parameters: [n_visible_layer_neurons, *n_hidden_units, n_labels, n_layers]
n_visible_layer_neurons: number of visible neurons
*n_hidden_units: array of hidden neurons number
n_labels: number of labels
n_layers: number of layers */
DBM *CreateNewDBM(int n_visible_layer_neurons, int *n_hidden_units, int n_labels, int n_layers){
    DBM *d = NULL;
    int i;

    d = (DBM *)malloc(sizeof(DBM));
    d->n_layers = n_layers;
    d->m = (RBM **)malloc(d->n_layers*sizeof(RBM *));

    /* Only the first layer has the number of visible inputs equals to the number of features */
    d->m[0] = CreateRBM(n_visible_layer_neurons, (int)n_hidden_units[0], n_labels);
    for(i = 1; i < d->n_layers; i++)
	   d->m[i] = CreateRBM((int)n_hidden_units[i-1], (int)n_hidden_units[i], n_labels);

    return d;
}

/* It deallocates an DBM
Parameters: [d]
d: DBM */
void DestroyDBM(DBM **d){
    int i;

    if(*d){
	for(i = 0; i < (*d)->n_layers; i++)
	    if((*d)->m[i]) DestroyRBM(&(*d)->m[i]);
        free((*d)->m);
        free(*d);
    }
}
/**************************/

/* DBM initialization */

/* It initializes a DBM
Parameters: [d]
d: DBM */
void InitializeDBM(DBM *d){
    int i;
    srand(time(NULL));

    for(i = 0; i < d->n_layers; i++){
	InitializeBias4VisibleUnitsWithRandomValues(d->m[i]);
        InitializeBias4HiddenUnits(d->m[i]);
        InitializeBias4LabelUnits(d->m[i]);
        InitializeWeights(d->m[i]);
        InitializeLabelWeights(d->m[i]);
    }
}

/* It performs DBM greedy pre-training step
Parameters: [D, d, n_epochs, n_samplings, batch_size, LearningType]
D: dataset
d: DBM
n_epochs: number of epochs
n_samplings: number of samplings
batch_size: mini-batch size
LearningType: type of learning algorithm [1 - CD | 2 - PCD | 3 - FPCD] */
double GreedyPreTrainingDBM(Dataset *D, DBM *d, int n_epochs, int n_samplings, int batch_size, int LearningType){
    double error = 0.0,aux = 0.0;
    int i, j, k, l;
    Dataset *tmp1 = NULL, *tmp2 = NULL;

    error = 0;
    tmp1 = CopyDataset(D);

    for (i = 0; i < d->n_layers;i++){
	switch (LearningType){
	    case 1:
		if(i == 0){
		    fprintf(stderr,"\n Training bottom layer ... ");
		    error = Bernoulli_TrainingRBMbyCD4DBM_BottomLayer(tmp1, d->m[0], n_epochs, n_samplings, batch_size);
		}else if(i == d->n_layers - 1){
		    fprintf(stderr,"\n Training top layer ... ");
		    error += Bernoulli_TrainingRBMbyCD4DBM_TopLayer(tmp1, d->m[d->n_layers-1], n_epochs, n_samplings, batch_size);
		}else{
		fprintf(stderr,"\n Training layer %i ... ",i+1);
		    error += Bernoulli_TrainingRBMbyCD4DBM_IntermediateLayers(tmp1, d->m[i], n_epochs, n_samplings, batch_size);
		}
	    break;
	    case 2:
		if(i == 0){
		    fprintf(stderr,"\n Training bottom layer ... ");
		    error = Bernoulli_TrainingRBMbyPCD4DBM_BottomLayer(tmp1, d->m[0], n_epochs, n_samplings, batch_size);
		}else if(i == d->n_layers - 1){
		    fprintf(stderr,"\n Training top layer ... ");
		    error += Bernoulli_TrainingRBMbyPCD4DBM_TopLayer(tmp1, d->m[d->n_layers-1], n_epochs, n_samplings, batch_size);
		}else{
		fprintf(stderr,"\n Training layer %i ... ",i+1);
		    error += Bernoulli_TrainingRBMbyPCD4DBM_IntermediateLayers(tmp1, d->m[i], n_epochs, n_samplings, batch_size);
		}
	    break;
	   /* case 3:
		if(i == 0){
		    fprintf(stderr,"\n Training bottom layer ... ");
		    error = Bernoulli_TrainingRBMbyFPCD4DBM_BottomLayer(tmp1, d->m[0], n_epochs, n_samplings, batch_size);
		}else if(i == d->n_layers - 1){
		    fprintf(stderr,"\n Training top layer ... ");
		    error += Bernoulli_TrainingRBMbyFPCD4DBM_TopLayer(tmp1, d->m[d->n_layers-1], n_epochs, n_samplings, batch_size);
		}else{
		fprintf(stderr,"\n Training layer %i ... ",i+1);
		    error += Bernoulli_TrainingRBMbyFPCD4DBM_IntermediateLayers(tmp1, d->m[i], n_epochs, n_samplings, batch_size);
		}
	    break;*/
	}
        fprintf(stderr,"OK");

	/* Making the hidden layer of RBM i to be the visible layer of RBM i+1 */
	if(i < d->n_layers-1){
	    tmp2 = CopyDataset(tmp1);
	    DestroyDataset(&tmp1);
	    tmp1 = CreateDataset(D->size, d->m[i]->n_hidden_layer_neurons);
	    tmp1->nlabels = D->nlabels;
	    for(j = 0; j < tmp1->size; j++){
		for(k = 0; k < tmp1->nfeatures; k++){
		    aux = 0.0;
		    for(l = 0; l < tmp2->nfeatures; l++)
			aux+=(gsl_vector_get(tmp2->sample[j].feature, l)*gsl_matrix_get(d->m[i]->W, l, k)+gsl_vector_get(tmp2->sample[j].feature, l)*gsl_matrix_get(d->m[i]->W, l, k));
		    aux+=gsl_vector_get(d->m[i]->b, k);
		    gsl_vector_set(tmp1->sample[j].feature, k, SigmoidLogistic(aux));
		    }
	    }
	    DestroyDataset(&tmp2);
	}
    }
    DestroyDataset(&tmp1);

    return error;
}

/* It performs DBM with Dropout greedy pre-training step
Parameters: [D, d, n_epochs, n_samplings, batch_size, LearningType, *p]
D: dataset
d: DBM
n_epochs: number of epochs
n_samplings: number of samplings
batch_size: mini-batch size
LearningType: type of learning algorithm [1 - CD | 2 - PCD | 3 - FPCD]
*p: array of hidden neurons dropout rate */
double GreedyPreTrainingDBMwithDropout(Dataset *D, DBM *d, int n_epochs, int n_samplings, int batch_size, int LearningType, double *p){
    double error = 0.0,aux = 0.0;
    int i, j, k, l;
    Dataset *tmp1 = NULL, *tmp2 = NULL;

    error = 0;
    tmp1 = CopyDataset(D);

    for (i = 0; i < d->n_layers;i++){
	switch (LearningType){
	    case 1:
		if(i == 0){
		    fprintf(stderr,"\n Training bottom layer ... ");
		    error = Bernoulli_TrainingRBMbyCD4DBM_BottomLayerwithDropout(tmp1, d->m[0], n_epochs, n_samplings, batch_size, p[i]);
		}else if(i == d->n_layers - 1){
		    fprintf(stderr,"\n Training top layer ... ");
		    error += Bernoulli_TrainingRBMbyCD4DBM_TopLayerwithDropout(tmp1, d->m[d->n_layers-1], n_epochs, n_samplings, batch_size, p[i]);
		}else{
		fprintf(stderr,"\n Training layer %i ... ",i+1);
		    error += Bernoulli_TrainingRBMbyCD4DBM_IntermediateLayerswithDropout(tmp1, d->m[i], n_epochs, n_samplings, batch_size, p[i]);
		}
	    break;
	    /*case 2:
		if(i == 0){
		    fprintf(stderr,"\n Training bottom layer ... ");
		    error = Bernoulli_TrainingRBMbyPCD4DBM_BottomLayerwithDropout(tmp1, d->m[0], n_epochs, n_samplings, batch_size, p[i]);
		}else if(i == d->n_layers - 1){
		    fprintf(stderr,"\n Training top layer ... ");
		    error += Bernoulli_TrainingRBMbyPCD4DBM_TopLayerwithDropout(tmp1, d->m[d->n_layers-1], n_epochs, n_samplings, batch_size, p[i]);
		}else{
		fprintf(stderr,"\n Training layer %i ... ",i+1);
		    error += Bernoulli_TrainingRBMbyPCD4DBM_IntermediateLayerswithDropout(tmp1, d->m[i], n_epochs, n_samplings, batch_size, p[i]);
		}
	    break;*/
	   /* case 3:
		if(i == 0){
		    fprintf(stderr,"\n Training bottom layer ... ");
		    error = Bernoulli_TrainingRBMbyFPCD4DBM_BottomLayer(tmp1, d->m[0], n_epochs, n_samplings, batch_size);
		}else if(i == d->n_layers - 1){
		    fprintf(stderr,"\n Training top layer ... ");
		    error += Bernoulli_TrainingRBMbyFPCD4DBM_TopLayer(tmp1, d->m[d->n_layers-1], n_epochs, n_samplings, batch_size);
		}else{
		fprintf(stderr,"\n Training layer %i ... ",i+1);
		    error += Bernoulli_TrainingRBMbyFPCD4DBM_IntermediateLayers(tmp1, d->m[i], n_epochs, n_samplings, batch_size);
		}
	    break;*/
	}
        fprintf(stderr,"OK");

	/* Making the hidden layer of RBM i to be the visible layer of RBM i+1 */
	if(i < d->n_layers-1){
	    tmp2 = CopyDataset(tmp1);
	    DestroyDataset(&tmp1);
	    tmp1 = CreateDataset(D->size, d->m[i]->n_hidden_layer_neurons);
	    tmp1->nlabels = D->nlabels;
	    for(j = 0; j < tmp1->size; j++){
		for(k = 0; k < tmp1->nfeatures; k++){
		    aux = 0.0;
		    for(l = 0; l < tmp2->nfeatures; l++)
			aux+=(gsl_vector_get(tmp2->sample[j].feature, l)*gsl_matrix_get(d->m[i]->W, l, k)+gsl_vector_get(tmp2->sample[j].feature, l)*gsl_matrix_get(d->m[i]->W, l, k));
		    aux+=gsl_vector_get(d->m[i]->b, k);
		    gsl_vector_set(tmp1->sample[j].feature, k, SigmoidLogistic(aux));
		    }
	    }
	    DestroyDataset(&tmp2);
	}
    }
    DestroyDataset(&tmp1);

    return error;
}

/* It performs DBM with Dropconnect greedy pre-training step
Parameters: [D, d, n_epochs, n_samplings, batch_size, LearningType, *p]
D: dataset
d: DBM
n_epochs: number of epochs
n_samplings: number of samplings
batch_size: mini-batch size
LearningType: type of learning algorithm [1 - CD | 2 - PCD | 3 - FPCD]
*p: array of dropconnect masks rate */
double GreedyPreTrainingDBMwithDropconnect(Dataset *D, DBM *d, int n_epochs, int n_samplings, int batch_size, int LearningType, double *p){
    double error = 0.0,aux = 0.0;
    int i, j, k, l;
    Dataset *tmp1 = NULL, *tmp2 = NULL;

    error = 0;
    tmp1 = CopyDataset(D);

    for (i = 0; i < d->n_layers;i++){
	switch (LearningType){
	    case 1:
		if(i == 0){
		    fprintf(stderr,"\n Training bottom layer ... ");
		    error = Bernoulli_TrainingRBMbyCD4DBM_BottomLayerwithDropconnect(tmp1, d->m[0], n_epochs, n_samplings, batch_size, p[i]);
		}else if(i == d->n_layers - 1){
		    fprintf(stderr,"\n Training top layer ... ");
		    error += Bernoulli_TrainingRBMbyCD4DBM_TopLayerwithDropconnect(tmp1, d->m[d->n_layers-1], n_epochs, n_samplings, batch_size, p[i]);
		}else{
		fprintf(stderr,"\n Training layer %i ... ",i+1);
		    error += Bernoulli_TrainingRBMbyCD4DBM_IntermediateLayerswithDropconnect(tmp1, d->m[i], n_epochs, n_samplings, batch_size, p[i]);
		}
	    break;
	    /*case 2:
		if(i == 0){
		    fprintf(stderr,"\n Training bottom layer ... ");
		    error = Bernoulli_TrainingRBMbyPCD4DBM_BottomLayerwithDropconnect(tmp1, d->m[0], n_epochs, n_samplings, batch_size, p[i]);
		}else if(i == d->n_layers - 1){
		    fprintf(stderr,"\n Training top layer ... ");
		    error += Bernoulli_TrainingRBMbyPCD4DBM_TopLayerwithDropconnect(tmp1, d->m[d->n_layers-1], n_epochs, n_samplings, batch_size, p[i]);
		}else{
		fprintf(stderr,"\n Training layer %i ... ",i+1);
		    error += Bernoulli_TrainingRBMbyPCD4DBM_IntermediateLayerswithDropconnect(tmp1, d->m[i], n_epochs, n_samplings, batch_size, p[i]);
		}
	    break;*/
	   /* case 3:
		if(i == 0){
		    fprintf(stderr,"\n Training bottom layer ... ");
		    error = Bernoulli_TrainingRBMbyFPCD4DBM_BottomLayer(tmp1, d->m[0], n_epochs, n_samplings, batch_size);
		}else if(i == d->n_layers - 1){
		    fprintf(stderr,"\n Training top layer ... ");
		    error += Bernoulli_TrainingRBMbyFPCD4DBM_TopLayer(tmp1, d->m[d->n_layers-1], n_epochs, n_samplings, batch_size);
		}else{
		fprintf(stderr,"\n Training layer %i ... ",i+1);
		    error += Bernoulli_TrainingRBMbyFPCD4DBM_IntermediateLayers(tmp1, d->m[i], n_epochs, n_samplings, batch_size);
		}
	    break;*/
	}
        fprintf(stderr,"OK");

	/* Making the hidden layer of RBM i to be the visible layer of RBM i+1 */
	if(i < d->n_layers-1){
	    tmp2 = CopyDataset(tmp1);
	    DestroyDataset(&tmp1);
	    tmp1 = CreateDataset(D->size, d->m[i]->n_hidden_layer_neurons);
	    tmp1->nlabels = D->nlabels;
	    for(j = 0; j < tmp1->size; j++){
		for(k = 0; k < tmp1->nfeatures; k++){
		    aux = 0.0;
		    for(l = 0; l < tmp2->nfeatures; l++)
			aux+=(gsl_vector_get(tmp2->sample[j].feature, l)*gsl_matrix_get(d->m[i]->W, l, k)+gsl_vector_get(tmp2->sample[j].feature, l)*gsl_matrix_get(d->m[i]->W, l, k));
		    aux+=gsl_vector_get(d->m[i]->b, k);
		    gsl_vector_set(tmp1->sample[j].feature, k, SigmoidLogistic(aux));
		    }
	    }
	    DestroyDataset(&tmp2);
	}
    }
    DestroyDataset(&tmp1);

    return error;
}

/* Bernoulli DBM reconstruction */

/* It reconstructs an input dataset given a trained DBM
Parameters: [D, d]
D: dataset
d: DBM */
double BernoulliDBMReconstruction(Dataset *D, DBM *d){
    double error = 0.0;
    int l, i;

    for(i = 0; i < D->size; i++){
	/* Going up */
	gsl_vector_free(d->m[0]->v);
	d->m[0]->v = gsl_vector_alloc(d->m[0]->n_visible_layer_neurons);
	gsl_vector_memcpy(d->m[0]->v, D->sample[i].feature);
        for(l = 0; l < d->n_layers; l++){
	    gsl_vector_free(d->m[l]->h);
	    d->m[l]->h = getProbabilityTurningOnHiddenUnit(d->m[l], d->m[l]->v);
            if(l < d->n_layers-1){
		gsl_vector_free(d->m[l+1]->v);
		d->m[l+1]->v = gsl_vector_alloc(d->m[l+1]->n_visible_layer_neurons);
		gsl_vector_memcpy(d->m[l+1]->v,d->m[l]->h);
            }
        }
        /* Going down */
        for(l = d->n_layers-1; l > 0; l--){
	    gsl_vector_free(d->m[l]->v);
            d->m[l]->v = getProbabilityTurningOnDBMIntermediateLayersOnDownPass(d->m[l], d->m[l]->h, d->m[l-1]);
            if(l >0){
		gsl_vector_free(d->m[l-1]->h);
		d->m[l-1]->h = gsl_vector_alloc(d->m[l-1]->n_hidden_layer_neurons);
		gsl_vector_memcpy(d->m[l-1]->h,d->m[l]->v);
            }
        }
	/* Reconstruction of the visible layer */
	gsl_vector_free(d->m[0]->v);
        d->m[0]->v= getProbabilityTurningOnVisibleUnit(d->m[0],d->m[0]->h);
        error+=getReconstructionError(D->sample[i].feature, d->m[0]->v);
    }
    error/=D->size;
    fprintf(stderr,"Reconstruction error: %lf OK", error);

    return error;
}

/* Auxiliary functions */

/* It computes the probability of turning on an intermediate layer of a DBM, as show in Eq. 28 and 29
Parameters: [m, *h, beneath_layer]
m: RBM
*h: hidden neurons vector
beneath layer: RBM's beneath layer */
gsl_vector *getProbabilityTurningOnDBMIntermediateLayersOnDownPass(RBM *m, gsl_vector *h, RBM *beneath_layer){
    int i,j;
    gsl_vector *inter = NULL;
    double tmp;

    inter = gsl_vector_calloc(m->n_visible_layer_neurons);

    for(j = 0; j < m->n_visible_layer_neurons; j++){
	tmp = 0.0;
        for(i = 0; i < m->n_hidden_layer_neurons; i++)
            tmp+=(gsl_vector_get(h, i)*gsl_matrix_get(m->W, j, i));
        tmp+=gsl_vector_get(m->a, j);
        for(i = 0; i < beneath_layer->n_visible_layer_neurons; i++)
            tmp+=(gsl_vector_get(beneath_layer->v, i)*gsl_matrix_get(beneath_layer->W, i, j));
        tmp+=gsl_vector_get(m->a, j);
        tmp = SigmoidLogistic(tmp);
        gsl_vector_set(inter, j, tmp);
    }

    return inter;
}

/* It saves DBM weight matrixes and bias vectors
Parameters: [d, file]
d: DBM
file: file name */
void saveDBMParameters(DBM *d, char *file){
    FILE *fpout = NULL;
    int id;
    int i, j;

    for(id = 0; id < d->n_layers; id++){
    	fpout = fopen(file,"a");
	fprintf(fpout,"W%d ",id);
	for(i = 0; i < d->m[id]->n_visible_layer_neurons; i++){
	    for(j = 0; j < d->m[id]->n_hidden_layer_neurons; j++){
		fprintf(fpout,"%f ",gsl_matrix_get(d->m[id]->W, i, j));
	    }
	}
	fprintf(fpout,"\n");
	fprintf(fpout,"b%d ",id);
	for(i = 0; i < d->m[id]->n_hidden_layer_neurons; i++)
	    fprintf(fpout,"%f ",gsl_vector_get(d->m[id]->b, i));
	fprintf(fpout,"\n");
	fprintf(fpout,"a%d ",id);
	for(i = 0; i < d->m[id]->n_visible_layer_neurons; i++)
	    fprintf(fpout,"%f ",gsl_vector_get(d->m[id]->a, i));
	fprintf(fpout,"\n");
	fprintf(fpout,"\n");
	fclose(fpout);
    }
}

/* It loads DBM weight matrixes and bias vectors from file
Parameters: [d, file]
d: DBM
file: file name */
void loadDBMParametersFromFile(DBM *d, char *file){
    int i, j, w;
    float values;
    char aux[30];

    FILE *fpin = NULL;
    fpin = fopen(file,"rt");

    for(w = 0; w < d->n_layers; w++){ /* Loading w */
	if(fscanf(fpin,"%s",aux)==1){
	    for(i = 0; i < d->m[w]->n_visible_layer_neurons; i++){
		for(j = 0; j < d->m[w]->n_hidden_layer_neurons; j++){
		    if(fscanf(fpin,"%f",&values)==1){
			gsl_matrix_set(d->m[w]->W, i, j, values);
		    }else{
			printf("failed to read float.\n");
		    }
		}
	    }
	}else{
	    printf("failed to read string.\n");
	}
	if(fscanf(fpin,"%s",aux)==1){ /* Loading b */
	    for(j = 0; j < d->m[w]->n_hidden_layer_neurons; j++){
		if(fscanf(fpin,"%f",&values)==1){
		    gsl_vector_set(d->m[w]->b, j, values);
		}else{
		    printf("failed to read float.\n");
		}
	    }
	}else{
	    printf("failed to read string.\n");
	}
	if(fscanf(fpin,"%s",aux)==1){ /* Loading a */
	    for(i = 0; i < d->m[w]->n_visible_layer_neurons; i++){
		if(fscanf(fpin,"%f",&values)==1){
		    gsl_vector_set(d->m[w]->a, i, values);
		}else{
		    printf("failed to read float.\n");
		}
	    }
	}else{
	    printf("failed to read string.\n");
	}
    }
    fclose(fpin);
}
