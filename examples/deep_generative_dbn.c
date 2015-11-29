#include "deep.h"

int main(int argc, char **argv){
  	int i,n, n_epochs,n_samplings, batch_size;
	Dataset *Train = NULL, *Test = NULL;
	Subgraph *gTrain = NULL, *gTest = NULL;
	gsl_vector *num_hidden_layers = NULL;
	DBN *d = NULL;
  	float value;
	double error = 0.0;
  	char fileName[256];
  	FILE *f = NULL;

	fprintf(stdout, "\nProgram that computes generative learning by Deep Belief Networks\n");
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
	
	d = CreateDBN(gTrain->nfeats, num_hidden_layers, gTrain->nlabels, num_hidden_layers->size);
	InitializeDBN(d);
	BernoulliDBNTrainingbyContrastiveDivergence(Train, d, n_epochs, n_samplings, batch_size);
	error = BernoulliDBNReconstruction(Test, d);
	fprintf(stderr,"    -> Total DBN Reconstruction error for Test Set: %lf", error);
	
	DestroyDBN(&d);
	DestroyDataset(&Train); DestroyDataset(&Test);
	DestroySubgraph(&gTrain); DestroySubgraph(&gTest);
	gsl_vector_free(num_hidden_layers);

	return 0;
}
