#include "deep.h"

// Info and help usage
// ===================================================================================================
void info(){
    fflush(stdout);
	fprintf(stdout, "\n\nProgram that executes Enhanced Probabilistic Neural Network classifier\n");
	fprintf(stdout, "\nIf you have any problem, please contact: ");
	fprintf(stdout, "\n- silasevandro@fc.unesp.br");
	fprintf(stdout, "\n- papa.joaopaulo@gmail.com\n");
	fprintf(stdout, "\n"); fflush(stdout);
}

void help_usage()
{

fprintf(stdout,
"\nusage epnn [options] training_file evaluation_file(*) test_file\n\
Options:\n\
	-s (sigma): Where sigma is the spread of the Gaussian function (take a value between 0 and 1).\n\
	Too small deviations cause a very spiky approximation which can not generalize well;\n\
	Too large deviations smooth out details.\n\
	Default: 1.0.\n\
	\n\
	-r (radius): Set r > 0 to use Hyper-Sphere in Enhanced Probabilistic Neural Network.\n\
	For only Probabilistic Neural Network set r = 0. Default: 0.\n\
 	\n\
 	-k (k maximum degree for the knn graph): Set k >= 1 for learning of gaussians from training file.\n\
	By default is given as the number of labels of the training set, or 'k = 0' as parameter. \n\
	\n\
	-l (learning best parameters):\n\
	For linear-search, set 1.\n\
	For grid-search, set 2.\n\
	Learning computes best sigma and best radius based on the evaluation_file.\n\
	Default: 0.\n\
	\n\
	-p (pitch): Step for optimization in learning phase (linear-search only). Default: 10.\n\
	\n\
	(*) - evaluation_file required only in learning phase.\n\
	\n\n"
);
exit(1);
}


// Main program 
// ==================================================================================================
int main(int argc, char **argv){
    char fileName[256];
    int i, j, step = 10, learningPhase = 0, learnGaussians = 0, kmax = -1;
    double radius = 0.0, sigma = 1.0;
	float Acc, time1 = 0.0, time2 = 0.0;
	timer tic, toc;
    FILE *f = NULL;
	
	gsl_vector *alpha = NULL;
	gsl_vector *lNode = NULL;
	gsl_vector *nGaussians = NULL;
    gsl_vector *nsample4class = NULL;
	gsl_vector *root = NULL;

    // parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
						
			case 's':
				fprintf(stdout,"\nSet sigma: %f\n", atof(argv[i]));
				sigma = atof(argv[i]);
				break;

			case 'r':
				fprintf(stdout,"\nSet radius: %f\n", atof(argv[i]));
				radius = atof(argv[i]);
				break;

			case 'k':
				kmax = atoi(argv[i]);
				if(kmax > 0){
					learnGaussians = atoi(argv[i]);
				}
				if(kmax == 0 ) fprintf(stdout,"\nUsing labels in training set as gaussians.\n");
				else if(kmax < 0){
					fprintf(stderr,"\n*** Unknown parameter for gaussians. ***\n");
					info();
					help_usage();
				}
				break;
																
			case 'l':
				learningPhase = atoi(argv[i]); // 1 for linear search, 2 for grid search
				if((learningPhase < 0 ) || (learningPhase > 2 )){
					fprintf(stderr,"\n*** Unknown parameter for learning. ***\n");
					info();
					help_usage();
				}
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
	
	
	
	//LEARNING GAUSSIANS
	if(learnGaussians){
		fprintf(stdout, "\n\nLearning gaussians from [%s]... ", argv[train_set]); fflush(stdout);
		
		gettimeofday(&tic,NULL);
		opf_BestkMinCut(Train,1,kmax); //default kmin = 1
		gettimeofday(&toc,NULL);

		time1 = (((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0);

		fprintf(stdout, "\n\nClustering by OPF ");
		gettimeofday(&tic,NULL); opf_OPFClustering(Train); gettimeofday(&toc,NULL);
		printf("num of clusters %d\n",Train->nlabels);
		
		time1 += (((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0);
		
		//SET N-CLUSTER IN N-GAUSSIANS
		nGaussians = LoadLabels(Train);
		root = gsl_vector_calloc(Train->nlabels); //Allocate space root
		
		/* If the training set has true labels, then create a
		   classifier by propagating the true label of each root to
		   the nodes of its tree (cluster). This classifier can be
		   evaluated by running opf_knn_classify on the training set
		   or on unseen testing set. Otherwise, copy the cluster
		   labels to the true label of the training set and write a
		   classifier, which essentially can propagate the cluster
		   labels to new nodes in a testing set. */

		if (Train->node[0].truelabel!=0){ // labeled training set
			Train->nlabels = 0;
			j = 1;
			for (i = 0; i < Train->nnodes; i++){//propagating root labels
				if (Train->node[i].root==i){
					Train->node[i].label = Train->node[i].truelabel;
					gsl_vector_set(root,j-1,i);// Assign corresponding root ID
					gsl_vector_set(nGaussians,j,Train->node[i].label);// Assign corresponding label for each Gaussian
					j++;
				}
			else
				Train->node[i].label = Train->node[Train->node[i].root].truelabel;
			}
			for (i = 0; i < Train->nnodes; i++){ // retrieve the original number of true labels
				if (Train->node[i].label > Train->nlabels) Train->nlabels = Train->node[i].label;
			}
		}
		else{ // unlabeled training set
			for (i = 0; i < Train->nnodes; i++) Train->node[i].truelabel = Train->node[i].label+1;
		}
		
		fprintf(stdout, "\nLearning gaussians time : %f seconds\n", time1); fflush(stdout);	
	}
	
	
	//SET N-LABELS IN N-GAUSSIANS
	if(!nGaussians) nGaussians = LoadLabels(Train);
	
	//LEARNING BEST PARAMETERS FOR SIGMA AND RADIUS(GRID-SEARCH)
	if(learningPhase){
		gsl_vector *BestParameters = NULL;
		fprintf(stdout, "\nReading evaluating set [%s] ...", argv[eval_set]); fflush(stdout);
		Subgraph *Eval = ReadSubgraph(argv[eval_set]);
		fprintf(stdout, " OK"); fflush(stdout);
		
		if(learningPhase == 1) fprintf(stderr,"\n\nLearning Best Parameters on evaluating set ... ");
		else fprintf(stderr,"\n\nGrid-search on evaluating set ... ");
		
		gettimeofday(&tic,NULL);	
		lNode = OrderedListLabel(Train, nGaussians, root);
		nsample4class = CountClasses(Train, nGaussians, root);
		double maxRadius = MaxDistance(Train);
		double minRadius = MinDistance(Train);
		if(learningPhase == 1)
			 BestParameters = LearnBestParameters(Train, Eval, step, lNode, nsample4class, maxRadius, minRadius, nGaussians);
		else
			BestParameters = gridSearch(Train, Eval, lNode, nsample4class, maxRadius, minRadius, nGaussians);
		
		gettimeofday(&toc,NULL);

		time2 = (((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0);

		fprintf(stdout, "\nLearning parameters time: %f seconds\n", time2); fflush(stdout);

		sigma = gsl_vector_get(BestParameters, 0);
		radius = gsl_vector_get(BestParameters, 1);
		gsl_vector_free(BestParameters);
		DestroySubgraph(&Eval);
		
		//WRITING EVALUATING TIME
		sprintf(fileName,"%s.time",argv[eval_set]);
		f = fopen(fileName,"a");
		fprintf(f,"%f\n",time2);
		fclose(f);
		
		
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
	if(!lNode) lNode = OrderedListLabel(Train, nGaussians, root);
	if(!nsample4class) nsample4class = CountClasses(Train, nGaussians, root);
	gettimeofday(&toc,NULL);
	fprintf(stdout, " OK\n"); fflush(stdout);
	
	time1 += (((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0);

	fprintf(stdout, "\nAllocating training set time: %f seconds\n", (((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0)); fflush(stdout);

	sprintf(fileName,"%s.time",argv[train_set]);
	f = fopen(fileName,"a");
	fprintf(f,"%f\n",time1);
	fclose(f);
	
	
    //WRITING PARAMETERS FILES
    fprintf(stdout, "\nWriting parameters file ... ");
    fflush(stdout);
	sprintf(fileName,"%s.parameters.out",argv[train_set]);
    f = fopen(fileName, "w");
	if(kmax > 0) fprintf(f,"%i\n", kmax);
	fprintf(f,"%lf\n",sigma);
	fprintf(f,"%lf\n",radius);
    fclose(f);
	fprintf(stderr,"OK\n");
	
	

	//TESTING PHASE
	fprintf(stdout, "\nReading testing set [%s] ...", argv[test_set]); fflush(stdout);
	Subgraph *Test = ReadSubgraph(argv[test_set]);
	fprintf(stdout, " OK\n"); fflush(stdout);
	
    fprintf(stdout, "\nInitializing EPNN ... ");
  	gettimeofday(&tic,NULL); EPNN(Train, Test, sigma, lNode, nsample4class, alpha, nGaussians); gettimeofday(&toc,NULL);
	fprintf(stdout,"OK\n");

	time1 = (((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0);

	fprintf(stdout, "\nTesting time: %f seconds\n", time1); fflush(stdout);

	sprintf(fileName,"%s.time",argv[test_set]);
	f = fopen(fileName,"a");
	fprintf(f,"%f\n",time1);
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
	gsl_vector_free(nGaussians);
	gsl_vector_free(root);
	
    fprintf(stdout,"OK\n\n");

    return 0;
}
