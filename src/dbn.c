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
        
        //only the first layer has the number of visible inputs equals to the number of features
        d->m[0] = CreateRBM(n_visible_units, (int)gsl_vector_get(n_hidden_units, 0), n_labels);
        for(i = 1; i < d->n_layers; i++)
            d->m[i] = CreateRBM((int)gsl_vector_get(n_hidden_units, i-1), (int)gsl_vector_get(n_hidden_units, i), n_labels);
        
        return d;
    }else{
        fprintf(stderr,"\nArray of hidden units not allocated or with a different number of hidden layers (n_layers) @CreateDBN\n");
        return NULL;
    }
}

/* It deallocates an DBN */
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

/* It initializes an DBN */
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

/* DBN information */

/* It writes the weight matrix as PGM images */
/* Notice: to visualize the weights, we need squared images and a number of hidden layers that has an integer square root (n=16, for instance) */
/*void DBNSaveWeights(DBN *d, char *path){
    int i, j, z, w, l, width, height;
    double min, max;
    IplImage *img = NULL;
    CvScalar s;
    gsl_vector_view v;
    char filename[256];
    
    for(l = 0; l < d->n_layers; l++){
        fprintf(stderr,"\nd->m[%d]->n_visible_layer_neurons: %d", l, (int)sqrt(d->m[l]->n_visible_layer_neurons));
    
        width = (int)sqrt(d->m[l]->n_visible_layer_neurons);
        height = (int)sqrt(d->m[l]->n_visible_layer_neurons);
        for(z = 0; z < d->m[l]->W->size2; z++){
            img = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1);
        
            // It normalizes weights within the interval [0,1]
            v = gsl_matrix_column(d->m[l]->W, z);
            min = gsl_vector_min(&v.vector);
            gsl_vector_add_constant(&v.vector, -min);
            max = gsl_vector_max(&v.vector);
            gsl_vector_scale(&v.vector, 1.0/max);
        
            // It transforms the weights in a 8bits/pixel image
            gsl_vector_scale(&v.vector, 255);
        
            w = 0;
            for(i = 0; i < img->height; i++){
                for(j = 0; j < img->width; j++){
                    s.val[0] = (double)round(gsl_vector_get(&v.vector, w++));
                    cvSet2D(img,i,j,s);
                }
            }
            sprintf(filename, "%s/layer_%d_weight_image_%d.pgm", path, l+1, z);
            if(!cvSaveImage(filename,img, 0)) fprintf(stderr,"\nCould not save %s", filename);
            cvReleaseImage(&img);
        }

    }
}*/
/**************************/

/* Bernoulli DBN training */

/* It trains an DBN for image reconstruction using Contrastive Divergence */
double BernoulliDBNTrainingbyContrastiveDivergence(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size){
    double error = 0.0, aux = 0.0;
    Dataset *tmp1 = NULL, *tmp2 = NULL;
    int i, j, z, id;
    
    tmp1 = CopyDataset(D);
    
    for(id = 0; id < d->n_layers; id++){
	    fprintf(stderr,"\nTraining layer %i ... ", id+1);
	    error = BernoulliRBMTrainingbyContrastiveDivergence(tmp1, d->m[id], n_epochs, n_CD_iterations, batch_size);
	    
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
    
    error = BernoulliDBNReconstruction(D, d);
    
    return error;
}

/* It trains a DBN for image reconstruction using Persistent Contrastive Divergence */
double BernoulliDBNTrainingbyPersistentContrastiveDivergence(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size){
    double error, aux;
    Dataset *tmp1 = NULL, *tmp2 = NULL;
    int i, j, z, id;
    
    tmp1 = CopyDataset(D);
    
    for(id = 0; id < d->n_layers; id++){

        fprintf(stderr,"\nTraining layer %d ... ", id+1);
        error = BernoulliRBMTrainingbyPersistentContrastiveDivergence(tmp1, d->m[id], n_epochs, n_CD_iterations, batch_size);
        
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
    
    error = BernoulliDBNReconstruction(D, d);
    
    return error;
}

/* It trains a DBN for image reconstruction using Fast Persistent Contrastive Divergence */
double BernoulliDBNTrainingbyFastPersistentContrastiveDivergence(Dataset *D, DBN *d, int n_epochs, int n_CD_iterations, int batch_size){
    double error, aux;
    Dataset *tmp1 = NULL, *tmp2 = NULL;
    int i, j, z, id;
    
    tmp1 = CopyDataset(D);
    
    for(id = 0; id < d->n_layers; id++){
        fprintf(stderr,"\nTraining layer %d ... ", id+1);
        error = BernoulliRBMTrainingbyFastPersistentContrastiveDivergence(tmp1, d->m[id], n_epochs, n_CD_iterations, batch_size);
        
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
    
    error = BernoulliDBNReconstruction(D, d);
    
    return error;
}
/**************************/

/* Bernoulli DBN reconstruction */

/* It reconstructs an input dataset given a trained DBN */
double BernoulliDBNReconstruction(Dataset *D, DBN *d){
    gsl_vector *h_prime = NULL, *v_prime = NULL, *aux = NULL;
    double error = 0.0;
    int l, i;

    for(i = 0; i < D->size; i++){
        //going up
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

/* Backpropagation fine-tuning ****************/

/* It executes the forward pass for a given sample s, and outputs the net's response for that sample */
gsl_vector *ForwardPass(gsl_vector *s, DBN *d){
    int l;
    gsl_vector *h = NULL, *v = NULL;
    
    if(d){
        v = gsl_vector_calloc(d->m[0]->n_visible_layer_neurons);
        setVisibleLayer(d->m[0], s);
        
        /* for each layer */
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

void ComputeBackPropagateError(gsl_vector *s, gsl_vector *output, DBN *d){
    
}

/**********************************************/


/* Image reconstruction */

// It reconstructs an input image given a trained DBN
/*IplImage *DBNReconstructImage(DBN *d, IplImage *img){
    int i, j, w;
    CvScalar s;
    gsl_vector *input = NULL, *h_prime = NULL, *v_prime = NULL, *aux = NULL;
    IplImage *output = NULL;
    
    output = cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,1);
    input = gsl_vector_calloc(d->m[0]->n_visible_layer_neurons);
    
    w = 0;
    for(i = 0; i < img->height; i++){
        for(j = 0; j < img->width; j++){
            s = cvGet2D(img,i,j);
            gsl_vector_set(input, w, s.val[0]);
            if(gsl_vector_get(input, w) == 255) gsl_vector_set(input, w, 1.0); 
            w++;
        }
    }
    
    // going up
    aux = gsl_vector_calloc(d->m[0]->n_visible_layer_neurons);
    gsl_vector_memcpy(aux, input);
    gsl_vector_free(input);
    
    for(j = 0; j < d->n_layers; j++){
        h_prime = gsl_vector_calloc(d->m[j]->n_hidden_layer_neurons);
        
        h_prime = getProbabilityTurningOnHiddenUnit(d->m[j], aux);
        gsl_vector_free(aux);
        
        if(j < d->n_layers-1){
            aux = gsl_vector_calloc(d->m[j+1]->n_visible_layer_neurons);
            gsl_vector_memcpy(aux, h_prime);
        }
        
        gsl_vector_free(h_prime);
    }
    
    //going down
    aux = gsl_vector_calloc(d->m[j-1]->n_hidden_layer_neurons);
    gsl_vector_memcpy(aux, d->m[j-1]->h);
    for(j = d->n_layers-1; j >= 0; j--){
        v_prime = gsl_vector_calloc(d->m[j]->n_visible_layer_neurons);
        
        v_prime = getProbabilityTurningOnVisibleUnit(d->m[j], aux);
        gsl_vector_free(aux);
        
        if(j > 0){
            aux = gsl_vector_calloc(d->m[j-1]->n_hidden_layer_neurons);
            gsl_vector_memcpy(aux, v_prime);
            gsl_vector_free(v_prime);
        }
    }
    
    w = 0;
    for(i = 0; i < output->height; i++){
        for(j = 0; j < output->width; j++){
            s.val[0] = (double)round(gsl_vector_get(v_prime, w++));
            if(s.val[0]) s.val[0] = 255;
            cvSet2D(output,i,j,s);
        }
    }
    
    gsl_vector_free(v_prime);
    
    return output;
}*/

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
            
            // going up
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
        fprintf(stderr,"\nThere is no DBN and/or Dataset allocated @DBN2Subgraph\n");
        return NULL;
    }
}

/* It saves DBN weight matrixes and bias vectors */
void saveDBNParameters(DBN *d, char *file){
	int id;
    FILE *fpout = NULL;
    
    int i, j;

    for(id = 0; id < d->n_layers; id++){
		fpout = fopen(file,"a");
		fprintf(fpout,"W%d ",id);
		for(i = 0; i < d->m[id]->n_visible_layer_neurons; i++){
		    for(j = 0; j < d->m[id]->n_hidden_layer_neurons; j++){
		        //printf("\n[%d] [%d] = %f ", i, j, gsl_matrix_get(d->m[id]->W, i, j));
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

/* It loads DBN weight matrixes and bias vectors from file*/
void loadDBNParametersFromFile(DBN *d, char *file){
    int i, j, w;
    float values;
    char aux[30];

    FILE *fpin = NULL;
    fpin = fopen(file,"rt");

    for(w = 0; w < d->n_layers; w++){ //load w
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

	    if(fscanf(fpin,"%s",aux)==1){ //load b
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

	    if(fscanf(fpin,"%s",aux)==1){  //load a
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
