#include "epnn.h"


// ENHANCED PROBABILISTIC NEURAL NETWORK WITH LOCAL DECISION CIRCLES BASED ON THE PARZEN-WINDOW ESTIMATION
void epnn(Subgraph *Train, Subgraph *Test, double sigma, gsl_vector *lNode, gsl_vector  *nsample4class, gsl_vector *alpha, gsl_vector *nGaussians)
{	
	double sum[(int)gsl_vector_get(nGaussians,0)+1], prob, norm; 
	int i=0, j, k, l, aux;
	
	// The INPUT layer
	for(k = 0; k < Test->nnodes; k++){
		// The OUTPUT layer which computes the PDF for each class
		prob = 0.0; 
		aux = 0;
		
		for (l=1; l<=(int)gsl_vector_get(nGaussians,0); l++ ) 
		{
			sum[l] = 0.0;
			// The SUMMATION layer which accumulates the pdf
			// for each example from the particular class k 
			for (i=0; i < (int)gsl_vector_get(nsample4class,l); i++ ) 
			{
				norm = 0.0;
				// The PATTERN layer that multiplies the test example
				for (j=0; j<Train->nfeats; j++ ){
					norm += pow(Test->node[k].feat[j] - Train->node[(int)gsl_vector_get(lNode,aux+i)].feat[j],2);
					
				}
				/* Optimization: Since the term 1/sqrt(2*PI) is a common factor in both y1(x) and y2(x),
				 it can be dropped out without changing the classification result.*/
				sum[l] += exp(-norm/(double)(2.0*(pow(gsl_vector_get(alpha,(int)gsl_vector_get(lNode,aux+i))*sigma,2)))); 
			}
			aux += (int)gsl_vector_get(nsample4class,l);
			sum[l] /=(double)gsl_vector_get(nsample4class,l);
		} 

		// The DECISION layer
		for (l=1; l <= (int)gsl_vector_get(nGaussians,0); l++ ){
			if ( sum[l] > prob ) 
			{
				prob = sum[l]; 
				Test->node[k].label = (int)gsl_vector_get(nGaussians,l);
			}
			
		}
	}
}




// OPF-CLUSTER
gsl_vector **opfcluster4epnn(Subgraph *Train, gsl_vector **gaussians, int kmax)
{

	int i, j;
	
	opf_BestkMinCut(Train,1,kmax); //default kmin = 1

	fprintf(stdout, "\n\nClustering by OPF...  ");
	opf_OPFClustering(Train);
	printf("num of clusters: %d\n",Train->nlabels);

	//set n-cluster in n-gaussians
	gaussians[0] = loadLabels(Train); //nGaussians
	gaussians[1] = gsl_vector_calloc(Train->nlabels); //Allocate space root

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
				gsl_vector_set(gaussians[1],j-1,i);// Assign corresponding root ID
				gsl_vector_set(gaussians[0],j,Train->node[i].label);// Assign corresponding label for each Gaussian
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

	return gaussians;
}



// MAXIMUM EUCLIDIAN DISTANCE OF TRAINING DATA PAIRS
double maxDistance(Subgraph *graph){
    
    double maxRadius = -1.0, dist;
    int i, j;
	
    for(i = 0; i < graph->nnodes; i++){
        for(j = 0; j < graph->nnodes; j++){
            if(j != i){
                dist = opf_EuclDist(graph->node[i].feat, graph->node[j].feat, graph->nfeats);
                if(dist > maxRadius) maxRadius = dist;
            }
        }
    }
    return  maxRadius;
}


// MINIMUM EUCLIDIAN DISTANCE OF TRAINING DATA PAIRS
double minDistance(Subgraph *graph){
    
    double minRadius = FLT_MAX, dist;
    int i, j;
	
    for(i = 0; i < graph->nnodes; i++){
        for(j = 0; j < graph->nnodes; j++){
            if(j != i){
                dist = opf_EuclDist(graph->node[i].feat, graph->node[j].feat, graph->nfeats);
                if(dist < minRadius) minRadius = dist;
            }
        }
    }
    return  minRadius;
}


// IT CALCULATES THE HYPER-SPHERE WITH RADIUS R FOR EACH TRAINING NODE
gsl_vector *hyperSphere(Subgraph *graph, double radius){
	
    int i, j, inSphere, sameLabel;
    double dist = 0.0;
	
	gsl_vector *alpha = gsl_vector_calloc(graph->nnodes); 
	
	if(radius > 0.0){
		for(i = 0; i < graph->nnodes; i++){
		    inSphere = sameLabel = 1;
		    for(j = 0; j < graph->nnodes; j++){
		        if(j != i){           
		            dist = opf_EuclDist(graph->node[j].feat, graph->node[i].feat, graph->nfeats);                              
		            if(dist <= radius){
		                inSphere++;
		                if(graph->node[j].truelabel == graph->node[i].truelabel) sameLabel++;
		            }
		        }       
		    }
			gsl_vector_set(alpha,i,(((double)sameLabel)/((double)inSphere)));
	    }
	}
	else{		
		for(i = 0; i < graph->nnodes; i++) gsl_vector_set(alpha,i,1);
	}
		
	return alpha;   
}

// ORDERED LIST LABEL
gsl_vector *orderedListLabel(Subgraph *Train, gsl_vector *nGaussians, gsl_vector *root)
{
	int i = 0, j, l;

    //Allocating lNode matrix
	//Ordered list by label
	gsl_vector *lNode = gsl_vector_calloc(Train->nnodes); 
	
	//Unlabeled training set
	if(Train->nlabels != (int)gsl_vector_get(nGaussians,0)){
		for(l = 0; l < (int)gsl_vector_get(nGaussians,0); l++ ){		
			for(j = 0; j < Train->nnodes; j++ ){
				if(Train->node[j].root ==  (int)gsl_vector_get(root,l)){
					gsl_vector_set(lNode,i,j);
					i++;		
				}
			}
		}
	}
	//Labeled training set
	else{
		for(l = 1; l <= Train->nlabels; l++ ){
			for(j = 0; j < Train->nnodes; j++ ){
				if(Train->node[j].truelabel == l){
						gsl_vector_set(lNode,i,j);
						i++;
				} 
			}
		}
	}
	
	return lNode;
}	


// COUNT CLASSES
gsl_vector *countClasses(Subgraph *Train, gsl_vector *nGaussians, gsl_vector *root)
{
	int i, l;
	gsl_vector *nsample4class = gsl_vector_calloc((int)gsl_vector_get(nGaussians,0)+1); 
	
	if(Train->nlabels != (int)gsl_vector_get(nGaussians,0)){
		
		for(i = 0; i < Train->nnodes; i++){
			for(l = 0; l < (int)gsl_vector_get(nGaussians,0); l++ ){
				if(Train->node[i].root ==  (int)gsl_vector_get(root,l)){
					gsl_vector_set(nsample4class,l+1, gsl_vector_get(nsample4class,l+1)+1);
					break;
				}
			}
			
		}
	}
	else{
		for(i = 0; i < Train->nnodes; i++) gsl_vector_set(nsample4class, Train->node[i].truelabel, gsl_vector_get(nsample4class,Train->node[i].truelabel)+1);
	}
	return nsample4class;
}



// LOADING LABELS IN TRAINING SET
gsl_vector *loadLabels(Subgraph *Train)
{
	int i;
	gsl_vector *nGaussians = gsl_vector_calloc(Train->nlabels+1); 
	
	gsl_vector_set(nGaussians, 0, Train->nlabels); //Allocating number of labels
	
	for(i = 1; i <= Train->nlabels; i++) gsl_vector_set(nGaussians, i, i);
	
	return nGaussians;
}




// GRID-SEARCH (KMAX, SIGMA AND RADIUS) IN EVALUTAING SET
gsl_vector *gridSearch(Subgraph *Train, Subgraph *Eval, gsl_vector *lNode, gsl_vector  *nsample4class, double maxRadius, double minRadius, gsl_vector *nGaussians, int kmax){  
	
    double sigma=-0.05, radius;
	float acc, bestAcc = -1.0;
	int i,k;
	Subgraph *g = NULL;
	g = CopySubgraph(Train);
	gsl_vector *root = NULL;
	gsl_vector *alpha = hyperSphere(Train, 0);
	gsl_vector *BestParameters = gsl_vector_calloc(3);
	
	gsl_vector **gaussians = (gsl_vector **)malloc(2 * sizeof(gsl_vector *));


	//GRID-SEARCH FOR PARAMTER KAMX FOR OPF_CLUSTER	
	if(kmax > 0){
		for(k = 1; k <= kmax; k++){
			fprintf(stdout,"\nTrying kmax: %i ", k); fflush(stdout);
			
			opf_BestkMinCut(g,1,k);
			opf_OPFClustering(g);
		
			if (g->node[0].truelabel!=0){ // labeled training set
				g->nlabels = 0;
				for (i = 0; i < g->nnodes; i++){//propagating root labels
					if (g->node[i].root==i) g->node[i].label = g->node[i].truelabel;
				else
					g->node[i].label = g->node[g->node[i].root].truelabel;}
				for (i = 0; i < g->nnodes; i++){ // retrieve the original number of true labels
					if (g->node[i].label > g->nlabels) g->nlabels = g->node[i].label;}
			}
			else{ // unlabeled training set
				for (i = 0; i < g->nnodes; i++) g->node[i].truelabel = g->node[i].label+1;
			}
			
			opf_OPFClassifying(g, Eval);
			// opf_OPFknnClassify(g, Eval);
			acc = opf_Accuracy(Eval);
			if(acc > bestAcc){
				gsl_vector_set(BestParameters,0,k);
	            bestAcc = acc;
	        }
			printf("  acc: %f", acc);
			opf_ResetSubgraph(g);
		}
		
		kmax = (int)gsl_vector_get(BestParameters,0);
		fprintf(stdout,"\nBest kmax = %i\n", kmax); fflush(stdout);

		//free variables
		opf_ResetSubgraph(g);
		gaussians[0] = nGaussians;
		gaussians[1] = root;
		
		//using best kmax for opf_cluster
		opf_BestkMinCut(g,1,kmax);
		opf_OPFClustering(g);
		
		gaussians = opfcluster4epnn(g, gaussians, kmax);
	
		nGaussians = gaussians[0];
		root = gaussians[1];
		
		//updating epnn
		lNode = orderedListLabel(g, nGaussians, root);
		nsample4class = countClasses(g, nGaussians, root);
	}


	
	//GRID-SEARCH FOR SIGMA AND RADIUS
	bestAcc = -1.0;
	for( ; sigma <= 1.0;  ){
		sigma+=0.05;
		for(radius = 0.0000001; radius <= (maxRadius+minRadius)/2; radius+=(minRadius+(radius*2))){
			fprintf(stdout,"\nTrying sigma: %lf and radius: %lf", sigma, radius); fflush(stdout);

			gsl_vector_free(alpha);
			alpha = hyperSphere(g, radius);

			epnn(g, Eval, sigma, lNode, nsample4class, alpha, nGaussians);

			acc = opf_Accuracy(Eval);
			fprintf(stdout,", Accuracy: %f", acc*100); fflush(stdout);
	        if(acc >= bestAcc){
				gsl_vector_set(BestParameters,1,sigma);
				gsl_vector_set(BestParameters,2,radius);
	            bestAcc = acc;
	        }
			
		}
	}
		
	gsl_vector_free(alpha);
	DestroySubgraph(&g);
    free(gaussians);
	
    return BestParameters; 
}
