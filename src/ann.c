#include "ann.h"

/* Artificial Neural Networks */

/* It computes matrix Phi
Parameters: [g, mu, cov]
g: subgraph (training or testing)
mu: matrix with m centers (means)
cov: covariance matrix */
gsl_matrix *ComputeHiddenLayerOutput(Subgraph *g, gsl_matrix *mu, gsl_matrix **cov){
	gsl_matrix *Phi = NULL;
	gsl_vector *x = NULL;
	int i, j;
	
	Phi = gsl_matrix_calloc(g->nnodes, mu->size1);
	for(i = 0; i < Phi->size1; i++){
		for(j = 0; j < Phi->size2; j++){
			x = node2gsl_vector(g->node[i].feat, mu->size1);
			gsl_matrix_set(Phi, i, j, GaussianDensity(cov, mu, x, j));
			gsl_vector_free(x);
		}
	}	
	return Phi;    
}

/* It trains the neural network by OPF and outputs the matrix of weights
Parameters: [g, mu, cov, kmax]
g: training subgraph
mu: mean matrix
cov: covariance matrix
kmax: k max value for OPF clustering */
gsl_matrix *TrainANNbyOPF(Subgraph *g, gsl_matrix *mu, gsl_matrix **cov, int kmax){
	gsl_matrix *Phi = NULL, *invPhi = NULL, *w = NULL, *y = NULL;
	int i, j, z = 0, p, nlabels = g->nlabels, *proto = NULL;
	Set *prototypes = NULL;
	
	/* Unsupervised phase */
	/* Initializing the centers, cutting and clustering dataset */
	opf_BestkMinCut(g, 1, kmax);
	prototypes = opf_OPFClustering4ANN(g); /* It initializes the centers */
	proto = AllocIntArray(GetSetSize(prototypes));
	fprintf(stderr,"\n# of clusters: %d\n", g->nlabels);
	
	for(i = 0; i < mu->size1; i++){
		p = RemoveSet(&prototypes);
		proto[z++] = p;
		for(j = 0; j < mu->size2; j++){
			gsl_matrix_set(mu, i, j, g->node[p].feat[j]);
		}
	}
	ComputeVariances(g->nfeats, mu, cov);
	
	/* Supervised phase */
	Phi = ComputeHiddenLayerOutput(g, mu, cov); /* It computes the hidden layer outputs */
	invPhi = PseudoInverse(Phi);
	
	/* It creates the desired outputs: if we have an output layer with p nodes,
	the desired output for a sample from class 2 is: 0 1 ... 0 (p bits) */
	y = gsl_matrix_calloc(Phi->size1, nlabels);
	gsl_matrix_set_zero(y);
	for(i = 0; i < y->size1; i++)
		gsl_matrix_set(y, i, g->node[i].truelabel-1, 1);
	w = gsl_matrix_calloc(Phi->size2, nlabels);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, invPhi, y, 0.0, w); /* Calculating  w = phi^+y */
	
	gsl_matrix_free(Phi);
	gsl_matrix_free(invPhi);
	gsl_matrix_free(y);

	free(proto);
	DestroySet(&prototypes);
	
	return w;
}

/* It trains the neural network by K-Means and outputs the matrix of weights
g: training subgraph
mu: mean matrix
cov: covariance matrix
k: number of cluster for k-means */
gsl_matrix *TrainANNbyKMeans(Subgraph *g, gsl_matrix *mu, gsl_matrix **cov, int kvalue){
	gsl_matrix *Phi = NULL, *invPhi = NULL, *w = NULL, *y = NULL;
	int i;
	
	/* Unsupervised phase */
	//kMeans(g, mu, kvalue); /* It initializes the centers */
	ComputeVariances(g->nfeats, mu, cov); /* It initializes the variances */
	
	/* Supervised phase */
	Phi = ComputeHiddenLayerOutput(g, mu, cov); /* It computes the hidden layer outputs */
	invPhi = PseudoInverse(Phi);
	
	/* It creates the desired outputs: if we have an output layer with p nodes, 
	 the desired output for a sample from class 2 is: 0 1 ... 0 (p bits) */
	y = gsl_matrix_calloc(Phi->size1, Phi->size2);
	gsl_matrix_set_zero(y);
	for(i = 0; i < y->size1; i++)
		gsl_matrix_set(y, i, g->node[i].truelabel-1, 1);
	
	w = gsl_matrix_calloc(Phi->size2, Phi->size2);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, invPhi, y, 0.0, w); /* Calculating  w = phi^+y */
	
	gsl_matrix_free(Phi);
	gsl_matrix_free(invPhi);
	gsl_matrix_free(y);
	
	return w;
}

/* It classifies an ANN
Parameters: [g, mu, cov, w]
g: testing subgraph
mu: mean matrix
cov: covariance matrix
w: weights matrix */
void ClassifyANN(Subgraph *g, gsl_matrix *mu, gsl_matrix **cov, gsl_matrix *w){
	gsl_matrix *output = NULL;
	double sum, oldsum;
	int i, j, z;
	
	/* It computes the output of hidden layer for all test set */
	output = ComputeHiddenLayerOutput(g, mu, cov);
	
	/* For each test node */
	for(z = 0; z < g->nnodes; z++){
		/* Computing the outputs of output layer */
		oldsum = -999999999999999;
		for(j = 0; j < w->size2; j++){
			sum = 0.0;
			for(i = 0; i < w->size1; i++)
			sum += gsl_matrix_get(w, i, j)*gsl_matrix_get(output, z, i); /* It computes wij*hi */
			if(sum > oldsum){
				oldsum = sum;
				g->node[z].label = j+1;
			}
		}
	}
	gsl_matrix_free(output);
}
/**********************************************/