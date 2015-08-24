#include "dbm.h"


/* Allocation and deallocation */


/* It allocates an DBM */
DBM *CreateDBM(int n_visible_layer_neurons,  int n_labels, int n_hidden_layers,  ...){
    DBM *d = NULL;
    int i, layer_old, layer;

    va_list ap;
    
    d = (DBM *)malloc(sizeof(DBM));
    d->n_layers = n_hidden_layers;
    d->m = (RBM **)malloc(d->n_layers*sizeof(RBM *));
    
    //only the first layer has the number of visible inputs equals to the number of features
	layer_old = n_visible_layer_neurons;
   
	va_start(ap, n_hidden_layers); 
    //for(i = 1; i < d->n_layers; i++){
	for (i = 0; i <  n_hidden_layers; i++) {
		layer = va_arg(ap, int); 
    	d->m[i] = CreateRBM(layer_old, layer, n_labels);
		layer_old = layer;
	}
	va_end(ap);
    
    return d;
}

/* It allocates an DBM */
//DBM *CreateDBM(int n_visible_layer_neurons, int n_hidden_layers_neurons, int n_labels, int n_layers){
//    DBM *d = NULL;
//    int i;
//    
//    d = (DBM *)malloc(sizeof(DBM));
//    d->n_layers = n_layers;
//    d->m = (RBM **)malloc(d->n_layers*sizeof(RBM *));
//    
//    //only the first layer has the number of visible inputs equals to the number of features
//    d->m[0] = CreateRBM(n_visible_layer_neurons , n_hidden_layers_neurons, n_labels);
//    for(i = 1; i < d->n_layers; i++){
//    	d->m[i] = CreateRBM(n_hidden_layers_neurons, n_hidden_layers_neurons, n_labels);
//	}
//    
//    return d;
//}

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
    int i,j;
    srand(time(NULL));
    
    for(i = 0; i < d->n_layers; i++){
		//fprintf(stdout,"\nLayer %i hiden before= %f",i);
		//for(j = 0;j< d->m[i]->n_hidden_layer_neurons;j++)
		//	fprintf(stdout,"\ unidade %i = %f",j, gsl_vector_get(d->m[i]->h,j));
		//PrintWeights(d->m[i]);
        InitializeRBM(d->m[i]);
		//fprintf(stdout,"\nLayer %i hiden after= %f",i);
		//PrintWeights(d->m[i]);
		//for(j = 0;j< d->m[i]->n_hidden_layer_neurons;j++)
		//	fprintf(stdout,"\ unidade %i = %f",j, gsl_vector_get(d->m[i]->h,j));
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
		gsl_vector_set(r->b,j,gsl_vector_get(r2->b, j));	
		gsl_vector_set(r->h,j,gsl_vector_get(r2->h, j));	
	}	
}

/* DBM pre-training */
double GreedyPreTrainingAlgorithmForADeepBoltzmannMachine(Dataset *D, DBM *d, int n_epochs, int n_CD_iterations, int batch_size){
    double error, aux;
    Dataset *tmp1 = NULL, *tmp2 = NULL;
    int i, j, z, id;
   // fprintf(stderr,"\nChegou antes de concatenar base");
    //Concatenate 2 folds of the dataset
	tmp1 = ConcatenateDataset(D,D);

	
	RBM *r = NULL; 
	//fprintf(stdout,"\nCconcatenou base");

    for(id = 0; id < d->n_layers; id++){
        
		//For the first layer of the DBM
		if(id==0){
			r = CreateRBM(tmp1->nfeatures, d->m[0]->n_hidden_layer_neurons, d->m[0]->n_labels);
			error = BernoulliRBMTrainingbyContrastiveDivergence(tmp1, r, n_epochs, n_CD_iterations, batch_size);
			//Paste the first half of the temp RBM to the first layer of the DBM	
			PasteDBMParameters(d->m[id], r);
			DestroyRBM(&r);
		//For the last layer of the DBM
		}else if(id==d->n_layers-1){
			//Creates a temp RBM with double hiden layers
			r = CreateRBM(d->m[id]->n_visible_layer_neurons, d->m[id]->n_hidden_layer_neurons*2, d->m[id]->n_labels );
			error = BernoulliRBMTrainingbyContrastiveDivergence(tmp1, r, n_epochs, n_CD_iterations, batch_size);
			PasteDBMParameters(d->m[id], r);
			DestroyRBM(&r);
		//Intermediate hiden layers
		}else{
        	error = BernoulliRBMTrainingbyContrastiveDivergence(tmp1, d->m[id], n_epochs, n_CD_iterations, batch_size);
			//halves W
			gsl_matrix_scale(d->m[id]->W, 0.5);	

		}

        
		if(id==0){
			tmp2 = UndoConcatenateDataset(tmp1);
		}else{
        	tmp2 = CopyDataset(tmp1);
		}
        DestroyDataset(&tmp1);

		/* it updates the last layer to be the input to the next RBM */
        tmp1 = CreateDataset(D->size, d->m[id]->n_hidden_layer_neurons);

        for(z = 0; z < tmp1->size; z++){
            for(j = 0; j < tmp1->nfeatures; j++){
                aux = 0.0;
                for(i = 0; i < tmp2->nfeatures; i++){
                    aux+=(gsl_vector_get(tmp2->sample[z].feature, i)*gsl_matrix_get(d->m[id]->W, i, j));
				}
                aux+=gsl_vector_get(d->m[id]->b, j);
                gsl_vector_set(tmp1->sample[z].feature, j, SigmoidLogistic(aux));
            }
        }
        DestroyDataset(&tmp2);
    }
    DestroyDataset(&tmp1);
    error = DBMReconstruction(D, d);	
    return error;
}

double DBMReconstruction(Dataset *D, DBM *d){
    gsl_vector *h_prime = NULL,*h_prime0 = NULL, *v_prime = NULL, *aux = NULL;
    double error = 0.0, beta = 1.0;
    int id, i;


    for(i = 0; i < D->size; i++){      
		beta = (D->size-(float)i)/D->size;

        aux = gsl_vector_calloc(d->m[0]->n_visible_layer_neurons);
        gsl_vector_memcpy(aux, D->sample[i].feature);
		for(id = 0; id < d->n_layers-1; id++){
			h_prime = getProbabilityTurningOnHiddenUnitDBM(d->m[id],d->m[id+1], aux, beta);
			if (id==0){
				h_prime0 = gsl_vector_calloc(d->m[id+1]->n_visible_layer_neurons);
                gsl_vector_memcpy(h_prime0, h_prime);
			}
            gsl_vector_free(aux);
            
            aux = gsl_vector_calloc(d->m[id+1]->n_visible_layer_neurons);
            gsl_vector_memcpy(aux, h_prime);
           
            gsl_vector_free(h_prime);
		}
		//last layer	
		h_prime = getProbabilityTurningOnHiddenUnitDBMLastLayer(d->m[d->n_layers-1], aux, beta);
		gsl_vector_free(aux);
		gsl_vector_free(h_prime);

		//first layer
        //aux = gsl_vector_calloc(d->m[0]->n_hidden_layer_neurons);
        //gsl_vector_memcpy(aux, d->m[0]->h);
        v_prime = getProbabilityTurningOnVisibleUnitDBMFirstLayer(d->m[0], h_prime0,beta);
        gsl_vector_free(h_prime0);


        error+=getReconstructionError(D->sample[i].feature, v_prime);
        gsl_vector_free(v_prime);
    }

    error/=D->size;
    fprintf(stdout,"\nError= %f",error);
    return error;
}

/* It computes the probability of turning on a hidden unit j, as described by Equation 38 - An Efficient Learning Procedure for Deep
Boltzmann Machines */
gsl_vector *getProbabilityTurningOnHiddenUnitDBM(RBM *rbm,RBM *next, gsl_vector *v, double beta){
    int i, j, m;
    gsl_vector *h = NULL;
    double tmp, tmpPre, tmpPos;
    
	//ex: h1
    h = gsl_vector_calloc(rbm->n_hidden_layer_neurons);
    for(j = 0; j < rbm->n_hidden_layer_neurons; j++){
        tmp = 0.0;
		tmpPre = 0.0;
		tmpPos = 0.0;
		//ex: v
        for(i = 0; i < rbm->n_visible_layer_neurons; i++)
            tmpPre+=(gsl_vector_get(v, i)*gsl_matrix_get(rbm->W, i, j));
		tmpPre+=gsl_vector_get(rbm->b, j);

		//ex: h2
        for(m = 0; m < next->n_hidden_layer_neurons; m++)
            tmpPos+=(gsl_vector_get(rbm->h, j)*gsl_matrix_get(next->W, j, m));
		//should i consider the bias twice??
		tmpPos+=gsl_vector_get(rbm->b, j);

		tmp = beta*(tmpPre+tmpPos);		
        
		tmp = SigmoidLogistic(tmp);
        gsl_vector_set(h, j, tmp);
    }
    
    return h;
}

/* It computes the probability of turning on a hidden unit j, as described by Equation 40 - An Efficient Learning Procedure for Deep
Boltzmann Machines */
gsl_vector *getProbabilityTurningOnHiddenUnitDBMLastLayer(RBM *m, gsl_vector *v, double beta){
    int i, j;
    gsl_vector *h = NULL;
    double tmp;
    
    h = gsl_vector_calloc(m->n_hidden_layer_neurons);
    for(j = 0; j < m->n_hidden_layer_neurons; j++){
        tmp = 0.0;
        for(i = 0; i < m->n_visible_layer_neurons; i++)
            tmp+=(beta*(gsl_vector_get(v, i)*gsl_matrix_get(m->W, i, j)));
        tmp+=gsl_vector_get(m->b, j);
		tmp = SigmoidLogistic(tmp);
        gsl_vector_set(h, j, tmp);
    }
    
    return h;
}

/* It computes the probability of turning on a visible unit j, as described by Equation 41 */
gsl_vector *getProbabilityTurningOnVisibleUnitDBMFirstLayer(RBM *m, gsl_vector *h, double beta){
    int i,j;
    gsl_vector *v = NULL;
    double tmp;
    
    v = gsl_vector_calloc(m->n_visible_layer_neurons);
    
    for(j = 0; j < m->n_visible_layer_neurons; j++){
        tmp = 0.0;
        for(i = 0; i < m->n_hidden_layer_neurons; i++)
            tmp+=(beta*(gsl_vector_get(h, i)*gsl_matrix_get(m->W, j, i)));
        tmp+=gsl_vector_get(m->a, j);
	
        tmp = SigmoidLogistic(tmp);
        gsl_vector_set(v, j, tmp);
    }
    
    return v;
}




