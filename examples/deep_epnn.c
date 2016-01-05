#include "deep.h"

// Info and help usage
// ===================================================================================================
void info(){
    fflush(stdout);
	fprintf(stdout, "\n\nProgram that executes Enhanced Probabilistic Neural Network classifier\n");
	fprintf(stdout, "\nIf you have any problem, please contact: ");
	fprintf(stdout, "\n- papa.joaopaulo@gmail.com\n");
	fprintf(stdout, "\n- silasevandro@gmail.com\n");
	fprintf(stdout, "\n"); fflush(stdout);
}

void help_usage()
{

fprintf(stdout,
"\nusage epnn [options] training_file evaluation_file test_file\n\
Options:\n\
	-s (sigma): Where sigma is the spread of the Gaussian function (take a value between 0 and 1).\n\
		> Too small deviations cause a very spiky approximation which can not generalize well;\n\
		> Too large deviations smooth out details.\n\
		> Default: 0.3.\n\
	\n\
	-r (radius): Set 1 to use Hyper-Sphere in Enhanced Probabilistic Neural Network.\n\
			     For only Probabilistic Neural Network set radius = 0. Default: 0.\n\
	\n\
	-l (learning best parameters): For learning phase, set 1.\n\
		> Learning computes best sigma and best radius based on the evaluation_file.\n\
		> Default: 0.\n\
		-p (pitch): Step for optimization in learning phase. Default: 10.\n\
	\n\n"
);
exit(1);
}


// Main program 
// ==================================================================================================
int main(int argc, char **argv){
    char fileName[256];
    int i, j, step = 10, learningPhase = 0;
    double radius = 0.0, sigma = 0.3;
	float Acc, time;
	timer tic, toc;
    FILE *f = NULL;
	
	gsl_vector *alpha = NULL;
	gsl_vector *lNode = NULL;
    gsl_vector *nsample4class = NULL;

    // parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
						
			case 's':
				fprintf(stdout,"\nSet sigma: %f .\n", atof(argv[i]));
				sigma = atof(argv[i]);
				break;

			case 'r':
				fprintf(stdout,"\nSet radius: %f .\n", atof(argv[i]));
				radius = atof(argv[i]);
				break;

			case 'l':
				learningPhase = 1;
				break;
							
			case 'p':
				fprintf(stdout,"\nSet pitch: %f .\n", atof(argv[i]));
				step = atof(argv[i]);
				break;
		

			default:
				fprintf(stderr,"\n*** Unknown option: -%c ***\n", argv[i-1][1]);
				info();
				help_usage();
		}
	}


    if((i>=argc-1) || (argc < 2)){
        info();
		help_usage();
    }

    //verify input files
    j = i;
	for(; i<argc; i++){
		sprintf(fileName,"%s",argv[i]);
		f = fopen(fileName,"r");
		if(!f){
			fprintf(stderr,"\n*** Unable to open file %s ***\n", argv[i]);
			info();
			help_usage();
			exit(-1);
		}else{
			fclose(f);
		}
    }



    //ID argv in input files
    int train_set = 0, eval_set = 0, test_set = 0;

    if((j == argc - 2) && (learningPhase == 1)){
        fprintf(stderr,"\n*** Required training, evaluation and testing files! ***\n");
        help_usage();
    }
	else if((j == argc - 2) && (learningPhase == 0)) train_set = j, test_set = j+1;
	else if(j == argc - 3) train_set = j, eval_set = j+1, test_set = j+2;


	//LOADING DATASETS
	fflush(stderr);
    fprintf(stdout, "\nReading training set [%s] ...", argv[train_set]); fflush(stdout);
    Subgraph *Train = ReadSubgraph(argv[train_set]);
    fprintf(stdout, " OK"); fflush(stdout);
	
	//LEARNING PHASE
	if(learningPhase){
		fprintf(stdout, "\nReading evaluating set [%s] ...", argv[eval_set]); fflush(stdout);
		Subgraph *Eval = ReadSubgraph(argv[eval_set]);
		fprintf(stdout, " OK"); fflush(stdout);
		
		fprintf(stderr,"\n\nLearning Best Parameters on evaluating set ... ");
		gettimeofday(&tic,NULL);	
		lNode = OrderedListLabel(Train);
		nsample4class = CountClasses(Train);
		double maxRadius = MaxDistance(Train);
		gsl_vector *BestParameters = LearnBestParameters(Train, Eval, step, lNode, nsample4class, maxRadius);
		gettimeofday(&toc,NULL);

		time = (((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0);

		fprintf(stdout, "\nLearning time: %f seconds\n", time); fflush(stdout);

		sprintf(fileName,"%s.time",argv[eval_set]);
		f = fopen(fileName,"a");
		fprintf(f,"%f\n",time);
		fclose(f);

		sigma = gsl_vector_get(BestParameters, 0);
		radius = gsl_vector_get(BestParameters, 1);
		gsl_vector_free(BestParameters);
		DestroySubgraph(&Eval);
		
	}
	
	
	//TRAINING PHASE	
	if(radius){
		fprintf(stdout, "\nComputing Hyper-Sphere with radius: %lf ...", radius); fflush(stdout);
	}
	else{
		fprintf(stdout, "\nAllocating training set ..."); fflush(stdout);
	}
	gettimeofday(&tic,NULL);
	alpha = HyperSphere(Train, radius);
	if(!lNode) lNode = OrderedListLabel(Train);
	if(!nsample4class) nsample4class = CountClasses(Train);
	gettimeofday(&toc,NULL);
	fprintf(stdout, " OK\n"); fflush(stdout);
	
	time = (((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0);

	fprintf(stdout, "\nTraining time: %f seconds\n", time); fflush(stdout);

	sprintf(fileName,"%s.time",argv[train_set]);
	f = fopen(fileName,"a");
	fprintf(f,"%f\n",time);
	fclose(f);
	
	
    //WRITING PARAMETERS FILES
    fprintf(stdout, "\nWriting parameters file ... ");
    fflush(stdout);
	sprintf(fileName,"%s.parameters.out",argv[train_set]);
    f = fopen(fileName, "w");
	fprintf(f,"%lf\n",sigma);
	fprintf(f,"%lf\n",radius);
    fclose(f);
	fprintf(stderr,"OK\n");
	
	
	//TESTING PHASE
	fprintf(stdout, "\nReading testing set [%s] ...", argv[test_set]); fflush(stdout);
	Subgraph *Test = ReadSubgraph(argv[test_set]);
	fprintf(stdout, " OK\n"); fflush(stdout);
	
    fprintf(stdout, "\nInitializing EPNN ... ");
  	gettimeofday(&tic,NULL); EPNN(Train, Test, sigma, lNode, nsample4class, alpha); gettimeofday(&toc,NULL);
	fprintf(stdout,"OK\n");

	time = (((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0);

	fprintf(stdout, "\nTesting time: %f seconds\n", time); fflush(stdout);

	sprintf(fileName,"%s.time",argv[test_set]);
	f = fopen(fileName,"a");
	fprintf(f,"%f\n",time);
	fclose(f);
	

    //WRITING OUTPUT FILES
    fprintf(stdout, "\nWriting output file ... ");
    fflush(stdout);
	sprintf(fileName,"%s.out",argv[test_set]);
    f = fopen(fileName, "w");
    for (i = 0; i < Test->nnodes; i++)
    	fprintf(f,"%d\n",Test->node[i].label);
    fclose(f);
    fprintf(stdout,"OK\n");

	
	//ACCURACY SECTION
    fprintf(stdout, "\nComputing accuracy ..."); fflush(stdout);
    Acc = opf_Accuracy(Test);

	fprintf(stdout, "\nAccuracy: %.2f%%", Acc*100); fflush(stdout);

	fprintf(stdout, "\nWriting accuracy in output file ..."); fflush(stdout);
	sprintf(fileName,"%s.acc",argv[test_set]);
	f = fopen(fileName,"a");
	fprintf(f,"%f\n",Acc*100);
	fclose(f);
	fprintf(stdout, " OK"); fflush(stdout);


    //DEALLOCATING MEMORY
    fflush(stderr);
    fprintf(stdout,"\nDeallocating memory ... ");
    DestroySubgraph(&Train);
	DestroySubgraph(&Test);
	gsl_vector_free(alpha);
	gsl_vector_free(lNode);
	gsl_vector_free(nsample4class);
	
    fprintf(stdout,"OK\n");

    return 0;
}
