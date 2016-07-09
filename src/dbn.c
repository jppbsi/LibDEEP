#include "dbn.h"

/* Allocation and deallocation */

/* It allocates an DBN
Parameters: [n_visible_units, *n_hidden_units, n_labels, n_layers]
n_visible_units: number of visible units of the bottom RBM
n_hidden_units: array with the number of hidden units of each RBM
n_labels: number of labels
n_layers: number of layers */
DBN *CreateDBN(int n_visible_units, gsl_vector *n_hidden_units, int n_labels, int n_layers){
    DBN *d = NULL;
    int i;
    
    if((n_hidden_units) && (n_hidden_units->size == n_layers)){
	d = (DBN *)malloc(sizeof(DBN));
        d->n_layers = n_layers;
        d->m = (RBM **)malloc(d->n_layers*sizeof(RBM *));
    
        /* Only the first layer has the number of visible inputs equals to the number of features */
        d->m[0] = CreateRBM(n_visible_units, (int)gsl_vector_get(n_hidden_units, 0), n_labels);
        for(i = 1; i < d->n_layers; i++)
	    d->m[i] = CreateRBM((int)gsl_vector_get(n_hidden_units, i-1), (int)gsl_vector_get(n_hidden_units, i), n_labels);
        return d;
    }else{
        fprintf(stderr,"\nArray of hidden units not allocated or with a different number of hidden layers (n_layers) @CreateDBN.\n");
        return NULL;
    }
}

/* It allocates an new DBN
Parameters: [n_visible_units, *n_hidden_units, n_labels, n_layers]
n_visible_units: number of visible units of the bottom RBM
n_hidden_units: array with the number of hidden units of each RBM
n_labels: number of labels
n_layers: number of layers */
DBN *CreateNewDBN(int n_visible_units, int *n_hidden_units, int n_labels, int n_layers){
    DBN *d = NULL;
    int i;
    
    if(n_hidden_units){
    
        d = (DBN *)malloc(sizeof(DBN));
        d->n_layers = n_layers;
        d->m = (RBM **)malloc(d->n_layers*sizeof(RBM *));
        
        /* Only the first layer has the number of visible inputs equals to the number of features */
        d->m[0] = CreateRBM(n_visible_units, (int)n_hidden_units[0], n_labels);
        for(i = 1; i < d->n_layers; i++)
            d->m[i] = CreateRBM((int)n_hidden_units[i-1], (int)n_hidden_units[i], n_labels);
        return d;
    }else{
        fprintf(stderr,"\nArray of hidden units not allocated or with a different number of hidden layers (n_layers) @CreateNewDBN.\n");
        return NULL;
    }
}

/* It deallocates an DBN
Parameters: [d]
d: DBN */
void DestroyDBN(DBN **d){
    int i;
    
    if(*d){
	for(i = 0; i < (*d)->n_layers; i++)
            if((*d)->m[i]) DestroyRBM(&(*d)->m[i]);
        free((*d)->m);
        free(*d);
    }
}
/**************************/

/* DBN initialization */

/* It initializes an DBN
Parameters: [d]
d: DBN */
void InitializeDBN(DBN *d){
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
/**************************/

/* Bernoulli DBN training */

/* It trains an DBN for image reconstruction using Contrastive Divergence
Parameters: [D, d, n_epochs, n_CD_iterations, batch_size]
D: dataset
d: DBN
n_epochs: number of training epochs
n_CD_iterations: number of CD iterations
batch size: size of batch data */
double BernoulliDBNTrainingbyContrastiveDivergence(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size){
    double error = 0.0, aux = 0.0;
    Dataset *tmp1 = NULL, *tmp2 = NULL;
    int i, j, z, id;
    
    tmp1 = CopyDataset(D);
    
    for(id = 0; id < d->n_layers; id++){
	fprintf(stderr,"\nTraining layer %i ... ", id+1);
	error = BernoulliRBMTrainingbyContrastiveDivergence(tmp1, d->m[id], n_epochs, n_CD_iterations, batch_size);
	
	/* It updates the last layer to be the input to the next RBM */
	tmp2 = CopyDataset(tmp1);
	DestroyDataset(&tmp1);
	tmp1 = CreateDataset(D->size, d->m[id]->n_hidden_layer_neurons);
	for(z = 0; z < tmp1->size; z++){
	    for(j = 0; j < tmp1->nfeatures; j++){
		aux = 0.0;
		for(i = 0; i < tmp2->nfeatures; i++)
		    aux+=(gsl_vector_get(tmp2->sample[z].feature, i)*gsl_matrix_get(d->m[id]->W, i, j));
		aux+=gsl_vector_get(d->m[id]->b, j);
		gsl_vector_set(tmp1->sample[z].feature, j, SigmoidLogistic(aux));
	    }
	}
	DestroyDataset(&tmp2);
	fprintf(stderr,"\nOK");
    }
    DestroyDataset(&tmp1);
    error = BernoulliDBNReconstruction(D, d);
    
    return error;
}

/* It trains an DBN with Dropout for image reconstruction using Contrastive Divergence
Parameters: [D, d, n_epochs, n_CD_iterations, batch_size, *p, *q]
D: dataset
d: DBN
n_epochs: number of training epochs
n_CD_iterations: number of CD iterations
batch size: size of batch data
*p: array of hidden neurons dropout rate
*q: array of visible neurons dropout rate */
double BernoulliDBNTrainingbyContrastiveDivergenceWithDropout(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size, double *p, double *q){
    double error = 0.0, aux = 0.0;
    Dataset *tmp1 = NULL, *tmp2 = NULL;
    int i, j, z, id;
    
    tmp1 = CopyDataset(D);
    
    for(id = 0; id < d->n_layers; id++){
	fprintf(stderr,"\nTraining layer %i ... ", id+1);
	error = BernoulliRBMTrainingbyContrastiveDivergencewithDropout(tmp1, d->m[id], n_epochs, n_CD_iterations, batch_size, p[id], q[id]);
	
	/* It updates the last layer to be the input to the next RBM */
	tmp2 = CopyDataset(tmp1);
	DestroyDataset(&tmp1);
	tmp1 = CreateDataset(D->size, d->m[id]->n_hidden_layer_neurons);
	for(z = 0; z < tmp1->size; z++){
	    for(j = 0; j < tmp1->nfeatures; j++){
		aux = 0.0;
		for(i = 0; i < tmp2->nfeatures; i++)
		    aux+=(gsl_vector_get(tmp2->sample[z].feature, i)*gsl_matrix_get(d->m[id]->W, i, j));
		aux+=gsl_vector_get(d->m[id]->b, j);
		gsl_vector_set(tmp1->sample[z].feature, j, SigmoidLogistic(aux));
	    }
	}
	DestroyDataset(&tmp2);
	fprintf(stderr,"\nOK");
    }
    DestroyDataset(&tmp1);
    error = BernoulliDBNReconstructionWithDropout(D, d, p, q);
    
    return error;
}

/* It trains a DBN for image reconstruction using Persistent Contrastive Divergence
Parameters: [D, d, n_epochs, n_CD_iterations, batch_size]
D: dataset
d: DBN
n_epochs: number of training epochs
n_CD_iterations: number of CD iterations
batch size: size of batch data */
double BernoulliDBNTrainingbyPersistentContrastiveDivergence(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size){
    double error, aux;
    Dataset *tmp1 = NULL, *tmp2 = NULL;
    int i, j, z, id;
    
    tmp1 = CopyDataset(D);
    
    for(id = 0; id < d->n_layers; id++){
        fprintf(stderr,"\nTraining layer %d ... ", id+1);
        error = BernoulliRBMTrainingbyPersistentContrastiveDivergence(tmp1, d->m[id], n_epochs, n_CD_iterations, batch_size);
        
        /* It updates the last layer to be the input to the next RBM */
        tmp2 = CopyDataset(tmp1);
        DestroyDataset(&tmp1);
        tmp1 = CreateDataset(D->size, d->m[id]->n_hidden_layer_neurons);
        for(z = 0; z < tmp1->size; z++){
            for(j = 0; j < tmp1->nfeatures; j++){
                aux = 0.0;
                for(i = 0; i < tmp2->nfeatures; i++)
                    aux+=(gsl_vector_get(tmp2->sample[z].feature, i)*gsl_matrix_get(d->m[id]->W, i, j));
                aux+=gsl_vector_get(d->m[id]->b, j);
                gsl_vector_set(tmp1->sample[z].feature, j, SigmoidLogistic(aux));
            }
        }
        DestroyDataset(&tmp2);
        fprintf(stderr,"\nOK");
    }
    DestroyDataset(&tmp1);
    error = BernoulliDBNReconstruction(D, d);
    
    return error;
}

/* It trains a DBN with Dropout for image reconstruction using Persistent Contrastive Divergence
Parameters: [D, d, n_epochs, n_CD_iterations, batch_size, *p, *q]
D: dataset
d: DBN
n_epochs: number of training epochs
n_CD_iterations: number of CD iterations
batch size: size of batch data
*p: array of hidden neurons dropout rate
*q: array of visible neurons dropout rate */
double BernoulliDBNTrainingbyPersistentContrastiveDivergenceWithDropout(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size, double *p, double *q){
    double error, aux;
    Dataset *tmp1 = NULL, *tmp2 = NULL;
    int i, j, z, id;
    
    tmp1 = CopyDataset(D);
    
    for(id = 0; id < d->n_layers; id++){
        fprintf(stderr,"\nTraining layer %d ... ", id+1);
        error = BernoulliRBMTrainingbyPersistentContrastiveDivergencewithDropout(tmp1, d->m[id], n_epochs, n_CD_iterations, batch_size, p[id], q[id]);
        
        /* It updates the last layer to be the input to the next RBM */
        tmp2 = CopyDataset(tmp1);
        DestroyDataset(&tmp1);
        tmp1 = CreateDataset(D->size, d->m[id]->n_hidden_layer_neurons);
        for(z = 0; z < tmp1->size; z++){
            for(j = 0; j < tmp1->nfeatures; j++){
                aux = 0.0;
                for(i = 0; i < tmp2->nfeatures; i++)
                    aux+=(gsl_vector_get(tmp2->sample[z].feature, i)*gsl_matrix_get(d->m[id]->W, i, j));
                aux+=gsl_vector_get(d->m[id]->b, j);
                gsl_vector_set(tmp1->sample[z].feature, j, SigmoidLogistic(aux));
            }
        }
        DestroyDataset(&tmp2);
        fprintf(stderr,"\nOK");
    }
    DestroyDataset(&tmp1);
    error = BernoulliDBNReconstructionWithDropout(D, d, p, q);
    
    return error;
}

/* It trains a DBN for image reconstruction using Fast Persistent Contrastive Divergence
Parameters: [D, d, n_epochs, n_CD_iterations, batch_size]
D: dataset
d: DBN
n_epochs: number of training epochs
n_CD_iterations: number of CD iterations
batch size: size of batch data */
double BernoulliDBNTrainingbyFastPersistentContrastiveDivergence(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size){
    double error, aux;
    Dataset *tmp1 = NULL, *tmp2 = NULL;
    int i, j, z, id;
    
    tmp1 = CopyDataset(D);
    
    for(id = 0; id < d->n_layers; id++){
        fprintf(stderr,"\nTraining layer %d ... ", id+1);
        error = BernoulliRBMTrainingbyFastPersistentContrastiveDivergence(tmp1, d->m[id], n_epochs, n_CD_iterations, batch_size);
        
        /* It updates the last layer to be the input to the next RBM */
        tmp2 = CopyDataset(tmp1);
        DestroyDataset(&tmp1);
        tmp1 = CreateDataset(D->size, d->m[id]->n_hidden_layer_neurons);
        for(z = 0; z < tmp1->size; z++){
            for(j = 0; j < tmp1->nfeatures; j++){
                aux = 0.0;
                for(i = 0; i < tmp2->nfeatures; i++)
                    aux+=(gsl_vector_get(tmp2->sample[z].feature, i)*gsl_matrix_get(d->m[id]->W, i, j));
                aux+=gsl_vector_get(d->m[id]->b, j);
                gsl_vector_set(tmp1->sample[z].feature, j, SigmoidLogistic(aux));
            }
        }
        DestroyDataset(&tmp2);
        fprintf(stderr,"\nOK");
    }
    DestroyDataset(&tmp1);
    error = BernoulliDBNReconstruction(D, d);
    
    return error;
}

/* It trains a DBN with Dropout for image reconstruction using Fast Persistent Contrastive Divergence
Parameters: [D, d, n_epochs, n_CD_iterations, batch_size, *p, *q]
D: dataset
d: DBN
n_epochs: number of training epochs
n_CD_iterations: number of CD iterations
batch size: size of batch data
*p: array of hidden neurons dropout rate
*q: array of visible neurons dropout rate */
double BernoulliDBNTrainingbyFastPersistentContrastiveDivergenceWithDropout(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size, double *p, double *q){
    double error, aux;
    Dataset *tmp1 = NULL, *tmp2 = NULL;
    int i, j, z, id;
    
    tmp1 = CopyDataset(D);
    
    for(id = 0; id < d->n_layers; id++){
        fprintf(stderr,"\nTraining layer %d ... ", id+1);
        error = BernoulliRBMTrainingbyFastPersistentContrastiveDivergencewithDropout(tmp1, d->m[id], n_epochs, n_CD_iterations, batch_size, p[id], q[id]);
        
        /* It updates the last layer to be the input to the next RBM */
        tmp2 = CopyDataset(tmp1);
        DestroyDataset(&tmp1);
        tmp1 = CreateDataset(D->size, d->m[id]->n_hidden_layer_neurons);
        for(z = 0; z < tmp1->size; z++){
            for(j = 0; j < tmp1->nfeatures; j++){
                aux = 0.0;
                for(i = 0; i < tmp2->nfeatures; i++)
                    aux+=(gsl_vector_get(tmp2->sample[z].feature, i)*gsl_matrix_get(d->m[id]->W, i, j));
                aux+=gsl_vector_get(d->m[id]->b, j);
                gsl_vector_set(tmp1->sample[z].feature, j, SigmoidLogistic(aux));
            }
        }
        DestroyDataset(&tmp2);
        fprintf(stderr,"\nOK");
    }
    DestroyDataset(&tmp1);
    error = BernoulliDBNReconstructionWithDropout(D, d, p, q);
    
    return error;
}
/**************************/

/* Bernoulli DBN reconstruction */

/* It reconstructs an input dataset given a trained DBN
Parameters: [D, d]
D: dataset
d: DBN */
double BernoulliDBNReconstruction(Dataset *D, DBN *d){
    gsl_vector *h_prime = NULL, *v_prime = NULL, *aux = NULL;
    double error = 0.0;
    int l, i;

    for(i = 0; i < D->size; i++){
        /* Going up */
        aux = gsl_vector_calloc(d->m[0]->n_visible_layer_neurons);
        gsl_vector_memcpy(aux, D->sample[i].feature);
        for(l = 0; l < d->n_layers; l++){
	    h_prime = getProbabilityTurningOnHiddenUnit(d->m[l], aux);
            gsl_vector_free(aux);
            if(l < d->n_layers-1){
                aux = gsl_vector_calloc(d->m[l+1]->n_visible_layer_neurons);
                gsl_vector_memcpy(aux, h_prime);
                gsl_vector_free(h_prime);
            }
        }
	
	/* Going down */
        aux = gsl_vector_calloc(d->m[l-1]->n_hidden_layer_neurons);
        gsl_vector_memcpy(aux, h_prime);
        for(l = d->n_layers-1; l >= 0; l--){
            v_prime = getProbabilityTurningOnVisibleUnit(d->m[l], aux);
            gsl_vector_free(aux);
            if(l > 0){
                aux = gsl_vector_calloc(d->m[l-1]->n_hidden_layer_neurons);
                gsl_vector_memcpy(aux, v_prime);
                gsl_vector_free(v_prime);
            }
        }
        error+=getReconstructionError(D->sample[i].feature, v_prime);
        gsl_vector_free(v_prime);
        gsl_vector_free(h_prime);
    }
    error/=D->size;

    return error;
}

/* It reconstructs an input dataset given a trained DBN
Parameters: [D, d, *p, *q]
D: dataset
d: DBN
*p: array of hidden neurons dropout rate
*q: array of visible neurons dropout rate */
double BernoulliDBNReconstructionWithDropout(Dataset *D, DBN *d, double *p, double *q){
    gsl_vector *h_prime = NULL, *v_prime = NULL, *aux = NULL;
    double error = 0.0;
    int l, i;
    
    for (l = 0; l < d->n_layers; l++)
	gsl_matrix_scale(d->m[l]->W, p[l]*q[l]);
	
    for(i = 0; i < D->size; i++){
        /* Going up */
        aux = gsl_vector_calloc(d->m[0]->n_visible_layer_neurons);
        gsl_vector_memcpy(aux, D->sample[i].feature);
        for(l = 0; l < d->n_layers; l++){
	    h_prime = getProbabilityTurningOnHiddenUnit(d->m[l], aux);
            gsl_vector_free(aux);
            if(l < d->n_layers-1){
                aux = gsl_vector_calloc(d->m[l+1]->n_visible_layer_neurons);
                gsl_vector_memcpy(aux, h_prime);
                gsl_vector_free(h_prime);
            }
        }
	
	/* Going down */
        aux = gsl_vector_calloc(d->m[l-1]->n_hidden_layer_neurons);
        gsl_vector_memcpy(aux, h_prime);
        for(l = d->n_layers-1; l >= 0; l--){
            v_prime = getProbabilityTurningOnVisibleUnit(d->m[l], aux);
            gsl_vector_free(aux);
            if(l > 0){
                aux = gsl_vector_calloc(d->m[l-1]->n_hidden_layer_neurons);
                gsl_vector_memcpy(aux, v_prime);
                gsl_vector_free(v_prime);
            }
        }
        error+=getReconstructionError(D->sample[i].feature, v_prime);
        gsl_vector_free(v_prime);
        gsl_vector_free(h_prime);
    }
    error/=D->size;

    return error;
}

/* Backpropagation fine-tuning (IN PROGRESS) */

/* It executes the forward pass for a given sample s, and outputs the net's response for that sample
Parameters: [s, d]
s: array of visible layer
d: DBN */
gsl_vector *ForwardPass(gsl_vector *s, DBN *d){
    int l;
    gsl_vector *h = NULL, *v = NULL;
    
    if(d){
        v = gsl_vector_calloc(d->m[0]->n_visible_layer_neurons);
        setVisibleLayer(d->m[0], s);
        
        /* For each layer */
        for(l = 0; l < d->n_layers;  l++){
            h = gsl_vector_calloc(d->m[l]->n_hidden_layer_neurons);
            h = getProbabilityTurningOnHiddenUnit(d->m[l], v);
            gsl_vector_free(v);
            v = gsl_vector_calloc(d->m[l]->n_hidden_layer_neurons);
            gsl_vector_memcpy(v, h);
            gsl_vector_free(h);
        }
        return v;
    }else{
        fprintf(stderr,"\nThere is no DBN allocated @ForwardPass.\n");
        return NULL;
    }
}
/**********************************************/

/* Data conversion */

/* It generates a subgraph using the learned features from the top layer of the DBN over the dataset
Parameters: [d, D]
d: trained DBN
D: input dataset */
Subgraph *DBN2Subgraph(DBN *d, Dataset *D){
    Subgraph *g = NULL;
    gsl_vector *aux = NULL, *h_prime = NULL;
    int i, l;
    
    if(d && D){
        g = CreateSubgraph(D->size);
        g->nfeats = d->m[d->n_layers-1]->n_hidden_layer_neurons;
        g->nlabels = D->nlabels;

        for(i = 0; i < D->size; i++){        
            /* Going up */
            aux = gsl_vector_calloc(d->m[0]->n_visible_layer_neurons);
            gsl_vector_memcpy(aux, D->sample[i].feature);
            for(l = 0; l < d->n_layers; l++){
                h_prime = getProbabilityTurningOnHiddenUnit(d->m[l], aux);
                gsl_vector_free(aux);
                if(l < d->n_layers-1){
                    aux = gsl_vector_calloc(d->m[l+1]->n_visible_layer_neurons);
                    gsl_vector_memcpy(aux, h_prime);
                    gsl_vector_free(h_prime);
                }
            }
            g->node[i].feat = AllocFloatArray(g->nfeats);
            g->node[i].truelabel = D->sample[i].label;
            g->node[i].label = D->sample[i].predict;
            g->node[i].position = i;
            for(l = 0; l < g->nfeats; l++)
                g->node[i].feat[l] = (float)gsl_vector_get(h_prime, l);
            gsl_vector_free(h_prime);
        }
        return g;
    }else{
        fprintf(stderr,"\nThere is no DBN and/or Dataset allocated @DBN2Subgraph.\n");
        return NULL;
    }
}
/**********************************************/

/* Auxiliary functions */

/* It saves DBN weight matrixes and bias vectors
Parameters: [d, file]
d: DBN
file: file name */
void saveDBNParameters(DBN *d, char *file){
    int id;
    FILE *fpout = NULL;
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

/* It loads DBN weight matrixes and bias vectors from file
Parameters: [d, file]
d: DBN
file: file name */
void loadDBNParametersFromFile(DBN *d, char *file){
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
			fprintf(stderr, "Failed to read float.\n");
		    }					
		}
	    }
	}else{
	    fprintf(stderr, "Failed to read string.\n");
	}	
	if(fscanf(fpin,"%s",aux)==1){ /* Loading b */
	    for(j = 0; j < d->m[w]->n_hidden_layer_neurons; j++){
		if(fscanf(fpin,"%f",&values)==1){
		    gsl_vector_set(d->m[w]->b, j, values);
		}else{
		    fprintf(stderr, "Failed to read float.\n");
		}	
	    }
	}else{
	    fprintf(stderr, "Failed to read string.\n");
	}	
	if(fscanf(fpin,"%s",aux)==1){ /* Loading a */
	    for(i = 0; i < d->m[w]->n_visible_layer_neurons; i++){
		if(fscanf(fpin,"%f",&values)==1){
		    gsl_vector_set(d->m[w]->a, i, values);
		}else{
		    fprintf(stderr, "Failed to read float.\n");
		}	
	    }
	}else{
	    fprintf(stderr, "Failed to read string.\n");
	}
    }
    fclose(fpin);
}