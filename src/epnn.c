#include "epnn.h"

/* Enhanced probabilistic neural network with local decision circles based on the Parzen-window estimation
Parameters: [Train, Test, sigma, *lNode, *nsample4class, *alpha, *nGaussians]
Train: training graph
Test: testing graph
sigma: variance value
*lNode: array of layer nodes
*nsample4class: number of samples for labels
*alpha: array of alpha values
*nGaussians: array of number of Gaussians */
void epnn(Subgraph *Train, Subgraph *Test, double sigma, gsl_vector *lNode, gsl_vector  *nsample4class, gsl_vector *alpha, gsl_vector *nGaussians){	
	double sum[(int)gsl_vector_get(nGaussians,0)+1], prob, norm; 
	int i = 0, j, k, l, aux;
	
	/* The INPUT layer */
	for(k = 0; k < Test->nnodes; k++){
		/* The OUTPUT layer which computes the PDF for each class */
		prob = 0.0; 
		aux = 0;
		for(l = 1; l <= (int)gsl_vector_get(nGaussians, 0); l++){
			sum[l] = 0.0;
			/* The SUMMATION layer which accumulates the pdf
			For each example from the particular class k */
			for(i = 0; i < (int)gsl_vector_get(nsample4class, l); i++){
				norm = 0.0;
				/* The PATTERN layer that multiplies the test example */
				for(j = 0; j < Train->nfeats; j++)
					norm += pow(Test->node[k].feat[j] - Train->node[(int)gsl_vector_get(lNode,aux+i)].feat[j],2);
				/* Optimization: Since the term 1/sqrt(2*PI) is a common factor in both y1(x) and y2(x),
				it can be dropped out without changing the classification result */
				sum[l] += exp(-norm/(double)(2.0*(pow(gsl_vector_get(alpha,(int)gsl_vector_get(lNode,aux+i))*sigma,2)))); 
			}
			aux += (int)gsl_vector_get(nsample4class,l);
			sum[l] /=(double)gsl_vector_get(nsample4class,l);
		} 
		/* The DECISION layer */
		for (l=1; l <= (int)gsl_vector_get(nGaussians, 0); l++ ){
			if ( sum[l] > prob ){
				prob = sum[l]; 
				Test->node[k].label = (int)gsl_vector_get(nGaussians, l);
			}
		}
	}
}
/**********************************************/

/* OPF clustering function for EPNN
Parameters: [Train, **gaussians, kmax]
Train: training graph
**gaussians: pointer to array of gaussians
kmax: maximum value of k */
gsl_vector **opfcluster4epnn(Subgraph *Train, gsl_vector **gaussians, int kmax){
	int i, j;
	
	opf_BestkMinCut(Train, 1, kmax); /* Default kmin = 1 */
	fprintf(stdout, "\n\nClustering by OPF...  ");
	opf_OPFClustering(Train);
	printf("num of clusters: %d\n",Train->nlabels);

	/* Set n-cluster in n-gaussians */
	gaussians[0] = loadLabels(Train); /* nGaussians */
	gaussians[1] = gsl_vector_calloc(Train->nlabels); /* Allocate space root */

	/* If the training set has true labels, then create a
	classifier by propagating the true label of each root to
	the nodes of its tree (cluster). This classifier can be
	evaluated by running opf_knn_classify on the training set
	or on unseen testing set. Otherwise, copy the cluster
	labels to the true label of the training set and write a
	classifier, which essentially can propagate the cluster
	labels to new nodes in a testing set */
	if (Train->node[0].truelabel!=0){ /* Labeled training set */
		Train->nlabels = 0;
		j = 1;
		for (i = 0; i < Train->nnodes; i++){ /* Propagating root labels */
			if (Train->node[i].root==i){
				Train->node[i].label = Train->node[i].truelabel;
				gsl_vector_set(gaussians[1],j-1,i); /* Assign corresponding root ID */
				gsl_vector_set(gaussians[0],j,Train->node[i].label); /* Assign corresponding label for each Gaussian */
				j++;
			}
			else
			Train->node[i].label = Train->node[Train->node[i].root].truelabel;
		}
		for (i = 0; i < Train->nnodes; i++){ /* Retrieves the original number of true labels */
			if (Train->node[i].label > Train->nlabels) Train->nlabels = Train->node[i].label;
		}
	}
	else{ /* Unlabeled training set */
		for (i = 0; i < Train->nnodes; i++)
			Train->node[i].truelabel = Train->node[i].label+1;
	}
	return gaussians;
}
/**********************************************/

/* Auxiliary functions */

/* It calculates the hyper-sphere with radius r for each training node
Parameters: [graph, raudius]
graph: graph
radius: value of sphere's radius */
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
		for(i = 0; i < graph->nnodes; i++)
			gsl_vector_set(alpha,i,1);
	}	
	return alpha;   
}

/* It orders a list label
Parameters: [Train, *nGaussians, *root]
Train: training graph
*nGaussians: array of number of Gaussians
*root: array of root nodes */
gsl_vector *orderedListLabel(Subgraph *Train, gsl_vector *nGaussians, gsl_vector *root){
	int i = 0, j, l;

	/* Allocating lNode matrix */
	gsl_vector *lNode = gsl_vector_calloc(Train->nnodes); 
	
	/* Unlabeled training set */
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
	/* Labeled training set */
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


/* It counts the number of classes
Parameters: [Train, *nGaussians, *root]
Train: training graph
*nGaussians: array of number of Gaussians
*root: array of root nodes */
gsl_vector *countClasses(Subgraph *Train, gsl_vector *nGaussians, gsl_vector *root){
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
		for(i = 0; i < Train->nnodes; i++)
			gsl_vector_set(nsample4class, Train->node[i].truelabel, gsl_vector_get(nsample4class,Train->node[i].truelabel)+1);
	}
	return nsample4class;
}

/* It loads labels in training set
Parameters: [Train]
Train: training graph */
gsl_vector *loadLabels(Subgraph *Train){
	int i;
	gsl_vector *nGaussians = gsl_vector_calloc(Train->nlabels+1); 
	
	gsl_vector_set(nGaussians, 0, Train->nlabels); /* Allocating number of labels */
	for(i = 1; i <= Train->nlabels; i++)
		gsl_vector_set(nGaussians, i, i);
	return nGaussians;
}

/* Maximum Euclidian Distance of training data pairs
Parameters: [graph]
graph: graph */
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
    return maxRadius;
}

/* Minimum Euclidian Distance of training data pairs
Parameters: [graph]
graph: graph */
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
    return minRadius;
}
/**********************************************/

/* Grid-Search */

/* Grid-search for k, sigma and radius
Parameters: [Train, Eval, radius, kmax]
Train: training graph
Eval : evaluating graph
radius: radius value
kmax: maximum value of k */
gsl_vector *gridSearch(Subgraph *Train, Subgraph *Eval, double radius, int kmax){  	
	double sigma=-0.05;
	float acc, bestAcc = -1.0;
	int i,k;
	Subgraph *g = NULL;
	
	g = CopySubgraph(Train);
	gsl_vector *alpha = hyperSphere(Train, 0);
	gsl_vector *BestParameters = gsl_vector_calloc(3);	
	
	double maxRadius = maxDistance(Train);
	double minRadius = minDistance(Train);
	
	gsl_vector **gaussians = (gsl_vector **)malloc(4 * sizeof(gsl_vector *));
	gaussians[0] = NULL; /* nGaussians */
	gaussians[1] = NULL; /* root */
	gaussians[2] = NULL; /* lNode */
	gaussians[3] = NULL; /* nsample4class */

	/* Grid-search for paramter kamx for opf_cluster */	
	if(kmax > 0){
		for(k = 1; k <= kmax; k++){
			fprintf(stdout,"\nTrying kmax: %i ", k); fflush(stdout);
			opf_BestkMinCut(g,1,k);
			opf_OPFClustering(g);
			if (g->node[0].truelabel!=0){ /* Labeled training set */
				g->nlabels = 0;
				for (i = 0; i < g->nnodes; i++){ /* Propagating root labels */
					if (g->node[i].root==i) g->node[i].label = g->node[i].truelabel;
				else
					g->node[i].label = g->node[g->node[i].root].truelabel;
				}
				for (i = 0; i < g->nnodes; i++){ /* Retrieves the original number of true labels */
					if (g->node[i].label > g->nlabels) g->nlabels = g->node[i].label;
				}
			}
			else{ /* Unlabeled training set */
				for (i = 0; i < g->nnodes; i++)
					g->node[i].truelabel = g->node[i].label+1;
			}	
			opf_OPFClassifying(g, Eval);			
			acc = opf_Accuracy(Eval);
			if(acc > bestAcc){
				gsl_vector_set(BestParameters,0,k);
			bestAcc = acc;
	        }
			fprintf(stderr, "  acc: %f", acc);
			opf_ResetSubgraph(g);
		}
		kmax = (int)gsl_vector_get(BestParameters,0);
		fprintf(stdout,"\nBest kmax = %i\n", kmax);
		fflush(stdout);

		/* Free variables */
		opf_ResetSubgraph(g);
				
		/* Using best kmax for opf_cluster */
		opf_BestkMinCut(g,1,kmax);
		opf_OPFClustering(g);
		
		gaussians = opfcluster4epnn(g, gaussians, kmax);
		
		/* Updating epnn */
		gaussians[2] = orderedListLabel(g, gaussians[0], gaussians[1]);
		gaussians[3] = countClasses(g, gaussians[0], gaussians[1]);
	}
	else{
		gaussians[0] = loadLabels(Train); /* nGaussians <- Start with nLabels */
		gaussians[1] = NULL; /* root */
		gaussians[2] = orderedListLabel(Train, gaussians[0], gaussians[1]); /* lNode global */
		gaussians[3] = countClasses(Train, gaussians[0], gaussians[1]); /* nsample4class global */
	}

	/* Grid-search for PNN */
	if(radius == -1) maxRadius = 0;
	
	/* Grid-search for sigma and radius */
	bestAcc = -1.0;
	for(sigma = 0.000000001; sigma <= 1.0;){
		sigma+=0.05;
		for(radius = 0; radius <= maxRadius; radius+=(minRadius+radius/7+0.000001)){
			fprintf(stdout,"\nTrying sigma: %lf and radius: %lf", sigma, radius); fflush(stdout);
			gsl_vector_free(alpha);
			alpha = hyperSphere(g, radius);
			epnn(g, Eval, sigma, gaussians[2], gaussians[3], alpha, gaussians[0]);
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
	gsl_vector_free(gaussians[0]);
	gsl_vector_free(gaussians[1]);
	gsl_vector_free(gaussians[2]);
	gsl_vector_free(gaussians[3]);
	free(gaussians);
	
	return BestParameters; 
}
/**********************************************/