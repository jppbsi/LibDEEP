#include "deep.h"

int main(int argc, char **argv){
    
    if(argc != 7){
	fprintf(stderr,"\nusage EPNN <training set> <evaluation set> <test set> <output results file name> <input parameters file> \
	    <best parameters output file>\n");
	exit(-1);
    }
    
    int i;
    char *outputName = argv[4], *fileName = argv[5];
    int learningPhase = 0, kmax = 0;
    double radius = 0.0, sigma = 1.0;
    float accuracy;
    FILE *f = NULL;
    gsl_vector *alpha = NULL;
    gsl_vector *lNode = NULL;
    gsl_vector *nGaussians = NULL;
    gsl_vector *nsample4class = NULL;
    gsl_vector *root = NULL;
    Subgraph *Train = NULL, *Eval = NULL, *Test = NULL;
	
    /* Auxiliary vector to manipulate individual parameters */
    gsl_vector **gaussians = (gsl_vector **)malloc(2 * sizeof(gsl_vector *));

    /* Loading input parameters */
    f = fopen(fileName, "r");
    if(!f){
        fprintf(stderr,"\nUnable to open file %s.\n", fileName);
        exit(1);
    }
    fscanf(f, "%d %lf %lf %d", &kmax, &sigma, &radius, &learningPhase);
    WaiveLibDEEPComment(f);
    fclose(f);
    
    /* Loading datasets */
    fprintf(stderr, "Reading training set ...");
    Train = ReadSubgraph(argv[1]);
    fprintf(stderr, "\nOK\n");
	
    /* Learning best parameters (grid-search) */
    if(learningPhase){
	gsl_vector *BestParameters = NULL;
	fprintf(stdout, "\nReading evaluation set ...");
	Eval = ReadSubgraph(argv[2]);
	fprintf(stdout, "\nOK\n");
	
	fprintf(stderr,"\nGrid-search on evaluating set ... ");	
	BestParameters = gridSearch(Train, Eval, radius, kmax);
	
	if(kmax){
	    kmax = (int)gsl_vector_get(BestParameters, 0);
	} 
	sigma = gsl_vector_get(BestParameters, 1);
	radius = gsl_vector_get(BestParameters, 2);
	
	if(learningPhase == 2){
	    fprintf(stderr, "\n\nMerging training set and evaluation set to training phase ...");
	    Subgraph *Merge = opf_MergeSubgraph(Train, Eval);
	    DestroySubgraph(&Train);
	    Train = CopySubgraph(Merge);
	    DestroySubgraph(&Merge);
	    fprintf(stderr, "\nOk");
	}
	gsl_vector_free(BestParameters);
	DestroySubgraph(&Eval);

	fprintf(stderr,"\n\nBest sigma = %lf", sigma);
	fprintf(stderr,"\nBest radius = %lf\n", radius);
    }
	
    /* Training phase */
    if(kmax){
	fprintf(stderr, "\nClustering with kmax ... ");
	gaussians = opfcluster4epnn(Train, gaussians, kmax);
	nGaussians = gaussians[0];
	root = gaussians[1];
	fprintf(stderr, "Ok\n");	
    }
    else{
	/* Set gaussians = nlabels if not */
	nGaussians = loadLabels(Train);
    }
    
    fprintf(stdout, "\nInitializing training...");
    if(radius)
	fprintf(stdout, "\nComputing Hyper-Sphere with radius: %lf ...", radius);
    lNode = orderedListLabel(Train, nGaussians, root);
    nsample4class = countClasses(Train, nGaussians, root);
    alpha = hyperSphere(Train, radius);
    fprintf(stdout, "\nOK\n");
	
    /* Writing parameters files */
    fprintf(stderr, "\nWriting parameters file ...");
    f = fopen(argv[6], "w");
    if(kmax > 0)
	fprintf(f,"%i", kmax);
    fprintf(f," %lf",sigma);
    fprintf(f," %lf\n",radius);
    fclose(f);
    fprintf(stderr,"\nOK\n");
	
    /* Testing phase */
    fprintf(stderr, "\nReading testing set ...");
    Test = ReadSubgraph(argv[3]);
    fprintf(stderr, "\nOK\n");

    fprintf(stderr, "\nInitializing EPNN ...");
    epnn(Train, Test, sigma, lNode, nsample4class, alpha, nGaussians);
    fprintf(stdout,"\nOK\n");
    
    /* Writing output files */
    fprintf(stderr, "\nWriting output file ...");
    f = fopen(argv[4], "w");
    for (i = 0; i < Test->nnodes; i++)
    	fprintf(f,"%d\n",Test->node[i].label);
    fclose(f);
    fprintf(stderr,"\nOK\n");

    /* Accuracy section */
    fprintf(stderr, "\nComputing accuracy ...");
    accuracy = opf_Accuracy(Test);

    fprintf(stderr, "\nAccuracy: %.2f%%\n", accuracy*100);

    fprintf(stdout, "\nWriting accuracy in output file ..."); fflush(stdout);
    sprintf(outputName,"%s.acc",outputName);
    f = fopen(outputName,"a");
    fprintf(f,"%f\n",accuracy*100);
    fclose(f);
    fprintf(stderr, "\nOK\n");
    
    /* Deallocating memory */
    fflush(stderr);
    fprintf(stdout,"\nDeallocating memory ...");
    DestroySubgraph(&Train);
    DestroySubgraph(&Test);
    gsl_vector_free(alpha);
    gsl_vector_free(lNode);
    gsl_vector_free(nsample4class);
    gsl_vector_free(nGaussians);
    gsl_vector_free(root);
    free(gaussians);
	
    fprintf(stdout,"\nOK\n");

    return 0;
}