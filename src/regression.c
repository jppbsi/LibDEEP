#include "regression.h"

/* Linear Regression */

/* It executes the linear regression
Parameters: [g, w]
g: input data
w: learned parameters of the model */
double Linear_Regression(Subgraph *g, double *w){
    double MSE, h_w, y;
    double *x = NULL;
    int i, m = g->nnodes, n = g->nfeats-1;
    
    if(g && w){
        
        MSE = 0.0;
        for(i = 0; i < g->nnodes; i++){ /* It runs over all data samples */
            x = node2double_vector(g->node[i].feat, n); /* It picks sample x_i */ 
            h_w = h_linear_regression(x, w, n);
            y = (double)g->node[i].feat[g->nfeats-1];
            MSE+=pow(h_w-y,2); /* tmp = sum(h(x_i)-y_i)^2 */
            free(x);
        }
        return MSE/(2*m);
        
    }else{
        fprintf(stderr,"\nThere is no X and/or w allocated @Linear_Regression.\n");
        return -1.0;
    }
}

/* It executes the partial derivative of variable j concerning linear regression with MSE as the cost function
Parameters: [g, w, j]
g: training graph
w: learned parameters of the model
j: ID of the feature */
double Linear_RegressionPartialDerivative(Subgraph *g, double *w, int j){
    double partial_derivative_value, h_w, y, x_j;
    double *x = NULL;
    int i, n = g->nfeats-1;
    
    if(g && w){
        
        partial_derivative_value = 0.0;
        for(i = 0; i < g->nnodes; i++){ /* It runs over all data samples */
            x = node2double_vector(g->node[i].feat, n); /* It picks sample x_i */ 
            h_w = h_linear_regression(x, w, n);
            y = (double)g->node[i].feat[g->nfeats-1];
            x_j = g->node[i].feat[j];
            partial_derivative_value+=((h_w-y)*x_j); /* tmp = sum(h(x_i)-y_i)x_i^j */
            free(x);
        }
        return partial_derivative_value;
        
    }else{
        fprintf(stderr,"\nThere is no X and/or w allocated @Linear_RegressionPartialDerivative.\n");
        return DBL_MAX;
    }
}

/* It executes the hypothesis function
Parameters: [x, w]
x: input sample
w: parameters of the model */
double h_linear_regression(double *x, double *w, int n){
    double tmp = 0.0;
    int i;
    
    if(x && w){
        for(i = 0; i < n; i++)
            tmp+=(w[i]*x[i]); /* tmp = wTx */
        return tmp;
    }
    else{
        fprintf(stderr,"\nThere is no X and/or w allocated @h_linear_regression.\n");
        return -1;
    }
}

/* It executes optimization through Batch Gradient Descent
Parameters: [g, alpha, w]
g: training graph
alpha: learning rate
w: parameters of the model */
double GradientDescentLinear(Subgraph *g, double alpha, double *w){
    double *w_tmp = NULL;
    double error = 0.0, old_error = DBL_MAX, tmp;
    int i, j, m = g->nnodes, n = g->nfeats-1, max_iteration = 10000;
		
    if(g){
        w_tmp = (double *)calloc(n, sizeof(double));
        memcpy(w_tmp, w, n*sizeof(double));
        i = 1;
        while((fabs(error-old_error) > 0.000001) && (i <= max_iteration)){
            old_error = error;
            for(j = 0; j < n; j++){
                tmp = w_tmp[j] - (alpha/m)*Linear_RegressionPartialDerivative(g, w_tmp, j); /* tmp = w_j - alpha*1/m*sum(h(x_i)-y_i)x_i^j */
		w[j] = tmp;
            }
            memcpy(w_tmp, w, n*sizeof(double));
            error = Linear_Regression(g, w);
            fprintf(stderr,"\nIteration: %d -> cost function value: %lf", i, error);
            i++;
        }
        fprintf(stderr,"\nMSE over the training set %.7lf", error);
    }else
        fprintf(stderr,"\n.There is no data allocated @GradientDescent\n");
    
    free(w_tmp);
    
    return error;
}

/* It fits a linear regression model using Equation 5 as the error function optimized by Gradient Descent
Parameters: [X, Y, m, n, alpha, w]
X: input data
Y: target values
m: matrix first dimension
n: matrix second dimension
alpha: learning rate
w: parameters of the linear function */
double LinearRegression_Fitting(double **X, double *Y, int m, int n, double alpha, double *w){
    int i, j;
    double error;
    Subgraph *g = NULL;
	                   
    /* Mapping data to another format */
    g = CreateSubgraph(m);
    g->nfeats = (n+1)+1; g->nlabels = 1;
    for(i = 0; i < m; i++){
	g->node[i].feat = AllocFloatArray((n+1)+1);
	for(j = 0; j < n+1; j++)
            g->node[i].feat[j] = X[i][j];
	g->node[i].feat[j] = Y[i]; /* Last position stores the target */
    }

    for(i = 0; i < n+1; i++) /* It initalizes w with a uniform distribution [0,1] -> small values */
	w[i] = RandomFloat(0, 1);
    
    error = GradientDescentLinear(g, alpha, w); /* Gradient descent optimization */
    
    DestroySubgraph(&g);
    
    return error;
}