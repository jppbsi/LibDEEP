#include "OPF.h"
#include "deep.h"
#include <math.h>
#include <time.h>


int main(int argc, char **argv){

    if(argc != 8){
        fprintf(stderr,"\nusage DBN <training set> <input parameters file> <model file> <number of DBN layers> <output file name> <number of images> <layer #>\n");
        exit(-1);
    }
    int i, j;
    gsl_vector *n_hidden_units = NULL, *eta = NULL, *lambda = NULL, *alpha = NULL, *eta_min = NULL, *eta_max = NULL;
	int n_layers = atoi(argv[4]), n_images = atoi(argv[6]), layer_number = atoi(argv[7]) - 1; 
	int side, indexHiddenUnit;
    double temp_eta, temp_lambda, temp_alpha, temp_eta_min, temp_eta_max, temp_hidden_units;
    double p, q;
    char *fileName = argv[3];
	char str[80];
    FILE *fp = NULL;
    Dataset *DatasetTrain = NULL;
    DBN *d = NULL;
    
    Subgraph *Train = NULL, *Test = NULL;
    Train = ReadSubgraph(argv[1]);
    
    DatasetTrain = Subgraph2Dataset(Train);    

    n_hidden_units = gsl_vector_alloc(n_layers);
	eta = gsl_vector_alloc(n_layers);	
	lambda = gsl_vector_alloc(n_layers);
	alpha = gsl_vector_alloc(n_layers);
	eta_min = gsl_vector_alloc(n_layers);
	eta_max = gsl_vector_alloc(n_layers);

    fp = fopen(fileName, "r");
    if(!fp){
            fprintf(stderr,"\nUnable to open file %s.\n", fileName);
            exit(1);
    }

    j = 0;

    for(i = 0; i < n_layers; i++){
		fscanf(fp, "%lf %lf %lf %lf", &temp_hidden_units, &temp_eta, &temp_lambda, &temp_alpha);
		WaiveComment(fp);
        gsl_vector_set(n_hidden_units, i, temp_hidden_units);
		gsl_vector_set(eta, i, temp_eta);
		gsl_vector_set(lambda, i, temp_lambda);
		gsl_vector_set(alpha, i, temp_alpha);
		fscanf(fp, "%lf %lf", &temp_eta_min, &temp_eta_max);
        gsl_vector_set(eta_min, i, temp_eta_min);
		gsl_vector_set(eta_max, i, temp_eta_max);
		WaiveComment(fp);

	}
    fclose(fp);

    
    fprintf(stderr,"\nCreating and initializing DBN ... ");

    d = CreateDBN(Train->nfeats, n_hidden_units, Train->nlabels, n_layers);    
    InitializeDBN(d);
    for(i = 0; i < d->n_layers; i++){
        d->m[i]->eta = gsl_vector_get(eta,i); 
        d->m[i]->lambda = gsl_vector_get(lambda,i); 
        d->m[i]->alpha = gsl_vector_get(alpha,i); 
        d->m[i]->eta_min = gsl_vector_get(eta_min,i); 
        d->m[i]->eta_max = gsl_vector_get(eta_max,i); 
		d->m[i]->t = 1.0;
    }   

	loadDBNParametersFromFile(d,argv[2]);
	side = (int)floor(sqrt(d->m[layer_number]->n_visible_layer_neurons));

	srand(time(NULL));
	for(i=0;i<n_images;i++){
		sprintf(str, "%s%d.pgm", argv[5],i);
		indexHiddenUnit = rand()%d->m[layer_number]->n_hidden_layer_neurons;
		SaveWeightsWithoutCV(d->m[layer_number], str,indexHiddenUnit , side, side);
	}


    
    DestroyDataset(&DatasetTrain);
    DestroySubgraph(&Train);
    DestroyDBN(&d);
    gsl_vector_free(n_hidden_units);
    gsl_vector_free(lambda);
    gsl_vector_free(eta);
    gsl_vector_free(alpha);
    gsl_vector_free(eta_min);
    gsl_vector_free(eta_max);
    
    return 0;
}
