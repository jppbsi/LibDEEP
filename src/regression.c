#include "regression.h"

/* Linear Regression */

/* It executes the linear regression
Parameters: [g, w]
g: input data
w: learned parameters of the model
---
Output: mean squared error */
double Linear_Regression(Subgraph *g, gsl_vector *w){
    double MSE, tmp, h_w, y;
    gsl_vector *x = NULL;
    int i, m = g->nnodes, n = g->nfeats-1;
    
    if(g && w){
        
        MSE = 0.0;
        for(i = 0; i < g->nnodes; i++){ // it runs over all data samples
            x = node2gsl_vector(g->node[i].feat, n); // it picks sample x_i  
            h_w = h_linear_regression(x, w);
            y = (double)g->node[i].feat[g->nfeats-1];
            MSE+=pow(h_w-y,2); //tmp = sum(h(x_i)-y_i)^2
            gsl_vector_free(x);
        }
        return MSE/(2*m);
        
    }else{
        fprintf(stderr,"\nThere is no X and/or w allocated @Linear_Regression.\n");
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
double Linear_RegressionPartialDerivative(Subgraph *g, gsl_vector *w, int j){
    double partial_derivative_value, tmp, h_w, y, x_j;
    gsl_vector *x = NULL;
    int i, n = g->nfeats-1;
    
    if(g && w){
        
        partial_derivative_value = 0.0;
        for(i = 0; i < g->nnodes; i++){ // it runs over all data samples
            x = node2gsl_vector(g->node[i].feat, n); // it picks sample x_i
            h_w = h_linear_regression(x, w);
            y = (double)g->node[i].feat[g->nfeats-1];
            x_j = g->node[i].feat[j];
            partial_derivative_value+=((h_w-y)*x_j); //tmp = sum(h(x_i)-y_i)x_i^j
            gsl_vector_free(x);
        }
        return partial_derivative_value;
        
    }else{
        fprintf(stderr,"\nThere is no X and/or w allocated @Linear_Regression.\n");
        return DBL_MAX;
    }
}

/* It executes the hypothesis function
Parameters: [x, w]
x: input sample
w: parameters of the model
---
Output: value of h function */
double h_linear_regression(gsl_vector *x, gsl_vector *w){
    double tmp = 0.0;
    int i;
    
    if(x && w){
        for(i = 0; i < x->size; i++)
            tmp+=(gsl_vector_get(w, i)*gsl_vector_get(x, i)); // tmp = wTx
        return tmp;
    }
    else{
        fprintf(stderr,"\nThere is no x and/or w allocated @h.\n");
        return -1;
    }
}