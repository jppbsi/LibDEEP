#include "deep.h"

/*DBM * CreateDBMLManyLayers(int nfeats, int nlabels, int nHidden, char **argv){
	switch (nHidden){
		case 1:	
			return  CreateDBM(nfeats, nlabels, nHidden,atoi(argv[7]));
			break;
		case 2:	
			return CreateDBM(nfeats, nlabels, nHidden,atoi(argv[7]),atoi(argv[8]));
			break;
		case 3:	
			return CreateDBM(nfeats, nlabels, nHidden,atoi(argv[7]),atoi(argv[8]),atoi(argv[9]));
			break;
		case 4:	
			return CreateDBM(nfeats, nlabels, nHidden,atoi(argv[7]),atoi(argv[8]),atoi(argv[9]),atoi(argv[10]));
			break;
		case 5:	
			return CreateDBM(nfeats, nlabels, nHidden,atoi(argv[7]),atoi(argv[8]),atoi(argv[9]),atoi(argv[10]),atoi(argv[11]));
			break;
		case 6:	
			return CreateDBM(nfeats, nlabels, nHidden,atoi(argv[7]),atoi(argv[8]),atoi(argv[9]),atoi(argv[10]),atoi(argv[11]),atoi(argv[12]));
			break;
		case 7:	
			return CreateDBM(nfeats, nlabels, nHidden,atoi(argv[7]),atoi(argv[8]),atoi(argv[9]),atoi(argv[10]),atoi(argv[11]),atoi(argv[12]),atoi(argv[13]));
			break;
		case 8:	
			return CreateDBM(nfeats, nlabels, nHidden,atoi(argv[7]),atoi(argv[8]),atoi(argv[9]),atoi(argv[10]),atoi(argv[11]),atoi(argv[12]),atoi(argv[13]),atoi(argv[14]));
			break;
		case 9:	
			return CreateDBM(nfeats, nlabels, nHidden,atoi(argv[7]),atoi(argv[8]),atoi(argv[9]),atoi(argv[10]),atoi(argv[11]),atoi(argv[12]),atoi(argv[13]),atoi(argv[14]),atoi(argv[15]));
			break;
		case 10:	
			return CreateDBM(nfeats, nlabels, nHidden,atoi(argv[7]),atoi(argv[8]),atoi(argv[9]),atoi(argv[10]),atoi(argv[11]),atoi(argv[12]),atoi(argv[13]),atoi(argv[14]),atoi(argv[15]),atoi(argv[16]));
			break;
	}
}*/

int main(int argc, char **argv){
  	int i,n, n_epochs,n_CD_iterations, batch_size;
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
		fprintf(stderr, "\nP4: number of contrastive divergence iterations");
		fprintf(stderr, "\nP5: batch size");
		fprintf(stderr, "\nP6: number of hidden layers");
		fprintf(stderr, "\nP7 , ..., P16: number of units for each hidden layer from [1,10]\n");
		exit(-1);
	}

	fprintf(stdout, "\nReading data file ...");
	Dataset *training_ds = Subgraph2Dataset(ReadSubgraph(argv[1]));
	n_epochs = atoi(argv[3]);
	n_CD_iterations = atoi(argv[4]);
	batch_size = atoi(argv[5]);
	
	num_hidden_layers = gsl_vector_alloc(atoi(argv[6]));
	for(i = 0; i < num_hidden_layers->size; i++)
		gsl_vector_set(num_hidden_layers, i, atoi(argv[7+i]));

	/*DBM *d  = CreateDBMLManyLayers(training_ds->nfeatures, training_ds->nlabels, num_hidden_layers,argv);
	InitializeDBM(d);
	//erro pre treino
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
	
/*
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
	DestroyDataset(&training_ds);
	//DestroyDataset(&testing_ds);
	gsl_vector_free(num_hidden_layers);

	return 0;
}
