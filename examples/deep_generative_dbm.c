#include "deep.h"

int main(int argc, char **argv){
  	int i,n, n_epochs,n_samplings, batch_size;
	Dataset *Train = NULL, *Test = NULL;
	Subgraph *gTrain = NULL, *gTest = NULL;
	gsl_vector *num_hidden_layers = NULL;
	DBM *d = NULL;
  	float value;
	double error;
  	char fileName[256];
  	FILE *f = NULL;

	fprintf(stdout, "\nProgram that computes generative learning by Deep Boltzmann Machine\n");
	fprintf(stdout, "\nIf you have any problem, please contact: ");
	fprintf(stdout, "\n- leandropassosjr@gmail.com");
	fprintf(stdout, "\n- papa.joaopaulo@gmail.com\n");
	fprintf(stdout, "\nLibDEEP version 1.0 (2015)\n");
	fprintf(stdout, "\n");

	if((argc <7)||(argc>16)){
		fprintf(stderr, "\nusage generative_dbm <P1> <P2> <P3> <P4> <P5> <P6> <P7...16>");
		fprintf(stderr, "\nP1: training set in the OPF file format (Train)");
		fprintf(stderr, "\nP2: test set in the OPF file format (Test)");
		fprintf(stderr, "\nP3: number of Epochs");
		fprintf(stderr, "\nP4: number of samplings");
		fprintf(stderr, "\nP5: batch size");
		fprintf(stderr, "\nP6: number of hidden layers");
		fprintf(stderr, "\nP7 , ..., P16: number of units for each hidden layer from [1,10]\n");
		exit(-1);
	}

	fprintf(stderr, "\nReading data file ... ");
	gTrain = ReadSubgraph(argv[1]); gTest = ReadSubgraph(argv[2]);
	Train = Subgraph2Dataset(gTrain); Test = Subgraph2Dataset(gTest);
	n_epochs = atoi(argv[3]);
	n_samplings = atoi(argv[4]);
	batch_size = atoi(argv[5]);
	
	num_hidden_layers = gsl_vector_alloc(atoi(argv[6]));
	for(i = 0; i < num_hidden_layers->size; i++)
		gsl_vector_set(num_hidden_layers, i, atoi(argv[7+i]));
	fprintf(stderr,"OK");
	
	d = CreateDBM(gTrain->nfeats, num_hidden_layers, gTrain->nlabels);
	InitializeDBM(d);
	GreedyPreTrainingDBM(Train, d, n_epochs, n_samplings, batch_size, 0);
	error = DBMReconstruction(Train, d);
	
	
	
	//error = DBMReconstruction(Train, d);
	//fprintf(stdout,"\nErro reconstrucao no treinamento= %f\n",erro);

/*	//erro pre treino
	error = GreedyPreTrainingAlgorithmForADeepBoltzmannMachine(training_ds, d, n_epochs, n_CD_iterations, batch_size);
	fprintf(stdout,"\nErro pre treino = %lf",erro);
	//erro validacao
	error = DBMReconstruction(D, d);	
	fprintf(stdout,"\nErro reconstrucao = %lf",erro);
	//error = GreedyPreTrainingAlgorithmForADeepBoltzmannMachine(training_ds, d, 45, 1, 20);




	//Dataset *testing_ds = Subgraph2Dataset(ReadSubgraph(argv[2]));

	//op = atoi(argv[3]);

	//opf_BestkMinCut(g,1,atoi(argv[2])); //default kmin = 1

	//value = atof(argv[4]);
	

	fprintf(stdout, "\n\nClustering by OPF ");
	opf_OPFClustering(g);
	printf("num of clusters %d\n",g->nlabels);

	fprintf(stdout, "\nWriting classifier's model file ..."); fflush(stdout);
	opf_WriteModelFile(g, "classifier.opf");
	fprintf(stdout, " OK"); fflush(stdout);

	fprintf(stdout, "\nWriting output file ..."); fflush(stdout);
	sprintf(fileName,"%s.out",argv[1]);
	f = fopen(fileName,"w");
	for (i = 0; i < g->nnodes; i++)
		fprintf(f,"%d\n",g->node[i].label);
	fclose(f);
	fprintf(stdout, " OK"); fflush(stdout);

	fprintf(stdout, "\n\nDeallocating memory ...\n");*/
	DestroyDBM(&d);
	DestroyDataset(&Train); DestroyDataset(&Test);
	DestroySubgraph(&gTrain); DestroySubgraph(&gTest);
	//DestroyDataset(&testing_ds);
	gsl_vector_free(num_hidden_layers);

	return 0;
}
