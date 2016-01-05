#include "epnn.h"

// Enhanced probabilistic neural network with local decision circles based on the Parzen-window estimation
// ==================================================================================================
void EPNN(Subgraph *Train, Subgraph *Test, double sigma, gsl_vector *lNode, gsl_vector  *nsample4class, gsl_vector *alpha)
{	
	double sum[Train->nlabels+1], prob, norm; 
	int i=0, j, k, l, aux;
	
	// The INPUT layer
	for(k = 0; k < Test->nnodes; k++){
		// The OUTPUT layer which computes the PDF for each class
		prob = 0.0; 
		aux = 0;
		
		for (l=1; l<=Train->nlabels; l++ ) 
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
		for (l=1; l<=Train->nlabels; l++ ){
			if ( sum[l] > prob ) 
			{
				prob = sum[l]; 
				Test->node[k].label = l;
			}
			
		}
	}
}


// Maximum Euclidian Distance of training data pairs
// ==================================================================================================
double MaxDistance(Subgraph *graph){
    
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

// It calculates the hyper-sphere with radius r for each training node
// ==================================================================================================
gsl_vector *HyperSphere(Subgraph *graph, double radius){
	
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

// Ordered List Label
// ==================================================================================================
gsl_vector *OrderedListLabel(Subgraph *Train)
{
	int i=0, j, l;

    //Allocating lNode matrix
	//Ordered list by label
	gsl_vector *lNode = gsl_vector_calloc(Train->nnodes); 
	
	for(l = 1; l <= Train->nlabels; l++ ){
		for(j = 0; j < Train->nnodes; j++ ){
			if(Train->node[j].truelabel == l){
					gsl_vector_set(lNode,i,j);
					i++;
			} 
		}
	}
	
	return lNode;
}	


// Count classes
// ==================================================================================================
gsl_vector *CountClasses(Subgraph *Train)
{
	int i;
	gsl_vector *nsample4class = gsl_vector_calloc(Train->nlabels+1); 
	
	for(i = 0; i < Train->nnodes; i++) gsl_vector_set(nsample4class, Train->node[i].truelabel, gsl_vector_get(nsample4class,Train->node[i].truelabel)+1);
	
	return nsample4class;
}


// Learn best parameters (sigma and radius) in evalutaing set
// ==================================================================================================
gsl_vector *LearnBestParameters(Subgraph *Train, Subgraph *Eval, int step, gsl_vector *lNode, gsl_vector  *nsample4class, double maxRadius){  
    double rstep = maxRadius/step, learnVar = 0.0;
	float acc, bestAcc = -1.0;
	
	gsl_vector *alpha = HyperSphere(Train, 0);
	
	gsl_vector *BestParameters = gsl_vector_calloc(2);
	
    while(learnVar <= 1){
        fprintf(stdout,"\nTrying sigma %lf ", learnVar);	
		EPNN(Train, Eval, learnVar, lNode, nsample4class, alpha);
		acc = opf_Accuracy(Eval);
		fprintf(stdout,"Accuracy: %f", acc*100);
        if(acc > bestAcc){
			gsl_vector_set(BestParameters,0,learnVar);
            bestAcc = acc;
        }
		learnVar += 0.1; 		
    }
		
	fprintf(stdout,"\nBest sigma = %lf\n", gsl_vector_get(BestParameters,0));
	
	bestAcc = -1;
	learnVar = 0.0;
    while(learnVar <= maxRadius){   
        fprintf(stdout,"\nTrying Radius %lf ", learnVar);
		gsl_vector_free(alpha);
		alpha = HyperSphere(Train, learnVar);	
		EPNN(Train, Eval, gsl_vector_get(BestParameters,0), lNode, nsample4class, alpha);
        acc = opf_Accuracy(Eval);
		fprintf(stdout,"Accuracy: %f", acc*100);
        if(acc > bestAcc){
			gsl_vector_set(BestParameters,1,learnVar);
            bestAcc = acc;
        }
        learnVar += rstep;  
    }
	fprintf(stdout,"\nBest radius = %lf\n", gsl_vector_get(BestParameters,1)); fflush(stdout);
	
	gsl_vector_free(alpha);
	
    return BestParameters; 
}

