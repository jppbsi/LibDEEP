#include "logistic.h"

/* Logistic Regression */

/* It executes the logistic regression
Parameters: [X, w, Y]
X: input data
w: learned parameters of the model
Y: target values
---
Output: error */
double Logistic_Regression(Subgraph *g, gsl_vector *w){
    double error, h_w;
    gsl_vector *x = NULL;
    int i;
    
    if(g && w){
        
        error = 0.0;
        for(i = 0; i < g->nnodes; i++){ // it runs over all data samples
            x = node2gsl_vector(g->node[i].feat, g->nfeats); // it picks sample x_i  
	    h_w = h_logistic(x, w);
            error+=(g->node[i].truelabel*log(h_w+0.000001)+((1-g->node[i].truelabel)*log(1-h_w+0.000001))); //Equation 23
	    gsl_vector_free(x);
        }
        return -error/g->nnodes;
        
    }else{
        fprintf(stderr,"\nThere is no data and/or w allocated @Logistic_Regression.\n");
        return -1.0;
    }
}

/* It executes the partial derivative of variable j concerning linear regression with MSE as the cost function
Parameters: [X, w, Y, j]
X: input data
w: learned parameters of the model
Y: target values
j: ID of the feature
---
Output: cost */
double Logistic_RegressionPartialDerivative(Subgraph *g, gsl_vector *w, int j){
    double partial_derivative_value;
    gsl_vector *x = NULL;
    int i;
    
    if(g && w){
        
        partial_derivative_value = 0.0;
        for(i = 0; i < g->nnodes; i++){ // it runs over all data samples
            x = node2gsl_vector(g->node[i].feat, g->nfeats); // it picks sample x_i            
            partial_derivative_value+=((h_logistic(x, w)-g->node[i].truelabel)*gsl_vector_get(x, j)); //tmp = sum(h(x_i)-y_i)x_i^j
	    gsl_vector_free(x);
        }
        return partial_derivative_value;
        
    }else{
        fprintf(stderr,"\nThere is no data and/or w allocated @Linear_Regression.\n");
        return DBL_MAX;
    }
}

/* It executes the hypothesis function
Parameters: [x, w]
x: input sample
w: parameters of the model
---
Output: value of h function */
double h_logistic(gsl_vector *x, gsl_vector *w){
    double tmp = 0.0;
    int i;
    
    if(x && w){
        for(i = 0; i < x->size; i++)
            tmp+=(gsl_vector_get(w, i)*gsl_vector_get(x, i)); // tmp = wTx
        return 1.0/(1+exp(-tmp)); //Equation 19
    }
    else{
        fprintf(stderr,"\nThere is no x and/or w allocated @h.\n");
        return -1;
    }
}

/* It executes the Logistic Regression for classification purposes
Parameters: [X, w]
X: test set
w: parameters of the model
---
Output: predicted labels */
void Logistic_Regression4Classification(Subgraph *Test, gsl_vector *w){
	int i;
	gsl_vector *x = NULL;
	
	for(i = 0; i < Test->nnodes; i++){ //for each test sample
		x = node2gsl_vector(Test->node[i].feat, Test->nfeats); // it picks sample x_i
		Test->node[i].label = round(h_logistic(x, w));
		gsl_vector_free(x);
	}
}