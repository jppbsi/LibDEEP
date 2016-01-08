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
	Default: 0.3.\n\
	\n\
	-r (radius): Set 1 to use Hyper-Sphere in Enhanced Probabilistic Neural Network.\n\
	For only Probabilistic Neural Network set radius = 0. Default: 0.\n\
 	\n\
 	-g (gaussians): Set 1 for learning of gaussians from training file.\n\
	By default is given as the number of labels of the training set, or 'g 0' as parameter. \n\
	For learning gaussians is required parameters of OPF_CLUSTER.\n\
	Parameters for OPF_CLUSTER:\n\
	-k (kmax): maximum degree for the knn graph\n\
	-t (type): 0 for height, 1 for area and 2 for volume\n\
	-v (value of type): value of parameter type in (0-1)\n\
	\n\
	-l (learning best parameters): For learning phase, set 1.\n\
	Learning computes best sigma and best radius based on the evaluation_file.\n\
	Default: 0.\n\
	\n\
	-p (pitch): Step for optimization in learning phase. Default: 10.\n\
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
    int i, j, step = 10, learningPhase = 0, learnGaussians = 0, kmax = -1, type = -1;
    double radius = 0.0, sigma = 0.3;
	float Acc, value = -1.0, time = 0.0;
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

			case 'g':
				learnGaussians = atoi(argv[i]);
				if(learnGaussians == 0 ) fprintf(stdout,"\nUsing labels in training set as gaussians.\n");
				else if(learnGaussians != 1){
					fprintf(stdout,"\n*** Unknown parameter for gaussians. ***\n");
					info();
					help_usage();
				}
				break;

			case 'k':
				kmax = atoi(argv[i]);
				break;
				
			case 't':
				type = atoi(argv[i]);
				if(type > 2 || type < 0){
					fprintf(stdout,"\n*** Unknown parameter for type. ***\n");
					info();
					help_usage();
				}
				break;

			case 'v':
				value = atof(argv[i]);
				if(value > 1 || value < 0){
					fprintf(stdout,"\n*** Unknown parameter for value. ***\n");
					info();
					help_usage();
				}
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


	if(learnGaussians == 1 && (kmax == -1 || type == -1 || value == -1)){
		fprintf(stderr,"\n*** Missing parameter in learning optimum gaussians from training file using OPF_CLUSTER ***\n");
		info();
		help_usage();
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

		time = (((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0);
		
		if ((value < 1)&&(value > 0)){
		  fprintf(stdout, "\n\n Filtering clusters ... ");
		  switch(type){
			  case 0:
					fprintf(stdout, "\n by dome height ... ");
					float Hmax=0;
					for (i=0; i < Train->nnodes; i++) if (Train->node[i].dens > Hmax) Hmax = Train->node[i].dens;
					opf_ElimMaxBelowH(Train, value*Hmax);
					break;
			  case 1:
				    fprintf(stdout, "\n by area ... ");
				    opf_ElimMaxBelowArea(Train, (int)(value*Train->nnodes));
				    break;
			  case 2:
				    fprintf(stdout, "\n by volume ... ");
				    double Vmax=0;
				    for (i=0; i < Train->nnodes; i++) Vmax += Train->node[i].dens;
				    opf_ElimMaxBelowVolume(Train, (int)(value*Vmax/Train->nnodes));
				    break;
			  }
		}
		
		fprintf(stdout, "\n\nClustering by OPF ");
		gettimeofday(&tic,NULL); opf_OPFClustering(Train); gettimeofday(&toc,NULL);
		printf("num of clusters %d\n",Train->nlabels);
		
		time += (((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0);
		
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
			for (i = 0; i < Train->nnodes; i++){
				// retrieve the original number of true labels
				if (Train->node[i].label > Train->nlabels) Train->nlabels = Train->node[i].label;
			}
		}
		else{ // unlabeled training set
			for (i = 0; i < Train->nnodes; i++) Train->node[i].truelabel = Train->node[i].label+1;
		}
		
		fprintf(stdout, "\nLearning gaussians time : %f seconds\n", time); fflush(stdout);	
	}
	
	
	//SET N-LABELS IN N-GAUSSIANS
	if(!nGaussians) nGaussians = LoadLabels(Train);
	
	//LEARNING BEST PARAMETERS FOR SIGMA AND RADIUS(GRID-SEARCH)
	if(learningPhase){
		fprintf(stdout, "\nReading evaluating set [%s] ...", argv[eval_set]); fflush(stdout);
		Subgraph *Eval = ReadSubgraph(argv[eval_set]);
		fprintf(stdout, " OK"); fflush(stdout);
		
		fprintf(stderr,"\n\nLearning Best Parameters on evaluating set ... ");
		gettimeofday(&tic,NULL);	
		lNode = OrderedListLabel(Train, nGaussians, root);
		nsample4class = CountClasses(Train, nGaussians, root);
		double maxRadius = MaxDistance(Train);
		double minRadius = MinDistance(Train);
		gsl_vector *BestParameters = LearnBestParameters(Train, Eval, step, lNode, nsample4class, maxRadius, minRadius, nGaussians);
		gettimeofday(&toc,NULL);

		time += (((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0);

		fprintf(stdout, "\nLearning parameters time: %f seconds\n", (((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0)); fflush(stdout);

		sigma = gsl_vector_get(BestParameters, 0);
		radius = gsl_vector_get(BestParameters, 1);
		gsl_vector_free(BestParameters);
		DestroySubgraph(&Eval);
		
	}
	
	//WRITING EVALUATING TIME
	if(time > 0.0){
		sprintf(fileName,"%s.time",argv[eval_set]);
		f = fopen(fileName,"a");
		fprintf(f,"%f\n",time);
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
  	gettimeofday(&tic,NULL); EPNN(Train, Test, sigma, lNode, nsample4class, alpha, nGaussians); gettimeofday(&toc,NULL);
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
	gsl_vector_free(nGaussians);
	gsl_vector_free(root);
	
    fprintf(stdout,"OK\n");

    return 0;
}
