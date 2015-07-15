#include "dbm.h"

/* Allocation and deallocation */

/* It allocates an DBM */
DBM *CreateDBM(int n_visible_layer_neurons, int n_hidden_layers_neurons, int n_labels, int n_layers){
    DBM *d = NULL;
    int i;
    
    d = (DBM *)malloc(sizeof(DBM));
    d->n_layers = n_layers;
    d->m = (RBM **)malloc(d->n_layers*sizeof(RBM *));
    
    //only the first layer has the number of visible inputs equals to the number of features
    d->m[0] = CreateRBM(n_visible_layer_neurons , n_hidden_layers_neurons, n_labels);
    for(i = 1; i < d->n_layers; i++){
    	d->m[i] = CreateRBM(n_hidden_layers_neurons, n_hidden_layers_neurons, n_labels);
	}
    
    return d;
}

/* It deallocates an DBM */
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

/* It initializes an DBM */
void InitializeDBM(DBM *d){
    int i;
    srand(time(NULL));
    
    for(i = 0; i < d->n_layers; i++){
        InitializeRBM(d->m[i]);
    }
}

/* It initializes an RBM */
void InitializeRBM(RBM *r){
    int i;
    srand(time(NULL));    

    InitializeBias4VisibleUnitsWithRandomValues(r);
    InitializeBias4HiddenUnits(r);
    InitializeBias4LabelUnits(r);
    InitializeWeights(r);
    InitializeLabelWeights(r);
    
}
/**************************/

void PasteDBMParameters(RBM *r, RBM *r2){
	int i,j;
	for(i=0;i<r->n_visible_layer_neurons;i++){
		for(j= 0;j<r->n_hidden_layer_neurons;j++){
			//set W
			gsl_matrix_set(r->W,i,j,gsl_matrix_get(r2->W, i, j));					
		}
		//set a
		gsl_vector_set(r->a,i,gsl_vector_get(r2->a, i));	
		gsl_vector_set(r->v,i,gsl_vector_get(r2->v, i));	
	}
	for(j= 0;j<r->n_hidden_layer_neurons;j++){
		//set b
		gsl_vector_set(r->b,i,gsl_vector_get(r2->b, i));	
		gsl_vector_set(r->h,i,gsl_vector_get(r2->h, i));	
	}
}

/* DBM pre-training */
double GreedyPreTrainingAlgorithmForADeepBoltzmannMachine(Dataset *D, DBM *d, int n_epochs, int n_CD_iterations, int batch_size){
    double error, aux;
    Dataset *tmp1 = NULL, *tmp2 = NULL;
    int i, j, z, id;
    
    //Concatenate 2 folds of the dataset
	tmp1 = ConcatenateDataset(D,D);
	RBM *rbmTemp = NULL; 

    for(id = 0; id < d->n_layers; id++){
        fprintf(stderr,"\nTraining layer %d ... ", id+1);
		//For the first layer of the DBM
		if(id==0){
			//Creates a temp RBM with double visible layers			
			rbmTemp = CreateRBM(d->m[id]->n_visible_layer_neurons *2, d->m[id]->n_hidden_layer_neurons, d->m[id]->n_labels);
			InitializeRBM(rbmTemp);
			//Trains the new RBM with contrastive divergence
			error = BernoulliRBMTrainingbyContrastiveDivergence(tmp1, rbmTemp, n_epochs, n_CD_iterations, batch_size);
			//Paste the first half of the temp RBM to the first layer of the DBM
			PasteDBMParameters(d->m[id], rbmTemp);
			DestroyRBM(&rbmTemp);
		//For the last layer of the DBM
		}else if(id==d->n_layers-1){
			//Creates a temp RBM with double hiden layers
			rbmTemp = CreateRBM(d->m[id]->n_visible_layer_neurons, d->m[id]->n_hidden_layer_neurons*2, d->m[id]->n_labels );
			InitializeRBM(rbmTemp);
			error = BernoulliRBMTrainingbyContrastiveDivergence(tmp1, rbmTemp, n_epochs, n_CD_iterations, batch_size);
			PasteDBMParameters(d->m[id], rbmTemp);
			DestroyRBM(&rbmTemp);
		//Intermediate hiden layers
		}else{
        	error = BernoulliRBMTrainingbyContrastiveDivergence(tmp1, d->m[id], n_epochs, n_CD_iterations, batch_size);
			//halves W
			gsl_matrix_scale(d->m[id]->W, 0.5);
			//gsl_matrix_set(r->W,i,j,gsl_matrix_get(r->W, i, j)/2);				

		}
        
        /* it updates the last layer to be the input to the next RBM */
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
    
    error = BernoulliDBMReconstruction(D, d);
    
    return error;


	//1 - Make two copies of the visible vector and tie the visible-to-hidden weights W 1 . Fit W 1 of the 1 st layer RBM to data.
	//2 - Freeze W 1 that defines the 1 st layer of features, and use samples h l from P (h 1 |v, 2W 1 ) (Eq. 22) as the data for training the next layer RBM with weight vector 2W 2 .
	//3 - Freeze W 2 that defines the 2 nd layer of features and use the samples h 2 from P (h 2 |h 1 , 2W 2 ) as the data for training the 3 rd layer RBM with weight vector 2W 3 .
	//4 - Proceed recursively for the next layers L − 1.
	//5 - When learning the top-level RBM, double the number of hidden units and tie the visible-to-hidden weights W L .
	//6 - Use the weights {W 1 , W 2 , ...., W L } to compose a Deep Boltzmann Machine.
}



