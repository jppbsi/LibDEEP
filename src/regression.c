#include "regression.h"

/* Linear Regression */

/* It fits a linear regression model using the Minimum Square Error as the error function
Parameters: [X, Y, w, Optimization_Func, ...]
X: training set
Y: target values
w: parameters of the linear function
Optimization_Func: function used to find the parameters that best fits the linear model
remaining parameters of each specific optimization function
---
Output: learned set of parameters w */
gsl_vector *LinearRegression_Fitting(gsl_matrix *X, gsl_vector *Y, mac_prtFun Function, int FUNCTION_ID, ...){
    gsl_vector *w = NULL;
    va_list arg;
    const gsl_rng_type *T = NULL;
    gsl_rng *r = NULL;
    int i;
    double alpha;
	               
    srand(time(NULL));
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    gsl_rng_set(r, rand());
    
    w = gsl_vector_calloc(X->size2); // w has size 1x(n+1)
    for(i = 0; i < w->size; i++) // it initalizes w with a uniform distribution [0,1] -> small values{
        gsl_vector_set(w, i, gsl_rng_uniform(r));
    
    va_start(arg, FUNCTION_ID);

    switch (FUNCTION_ID){
        case 5: // Gradient Descent
            alpha = va_arg(arg, double);
            Function(X, Y, alpha, 7, w); // 7 is the Linear Regression ID at LibOPT
        break;
    }
    
    va_end(arg);
    gsl_rng_free(r);
    
    return w;
}

/* It executes the linear regression
Parameters: [X, w, Y]
X: input data
w: learned parameters of the model
Y: target values
---
Output: mean squared error */
double Linear_Regression(gsl_matrix *X, gsl_vector *w, gsl_vector *Y){
    double MSE, tmp;
    gsl_vector_view row;
    int i;
    
    if(X && w){
        
        MSE = 0.0;
        for(i = 0; i < X->size1; i++){ // it runs over all data samples
            row = gsl_matrix_row(X, i);
            MSE+=pow((h(&row.vector, w)-gsl_vector_get(Y, i)),2); //tmp = sum(h(x_i)-y_i)^2
        }
        return MSE/(2*X->size1);
        
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
double Linear_RegressionPartialDerivative(gsl_matrix *X, gsl_vector *w, gsl_vector *Y, int j){
    double partial_derivative_value, tmp;
    gsl_vector_view x;
    int i;
    
    if(X && w){
        
        partial_derivative_value = 0.0;
        for(i = 0; i < X->size1; i++){ // it runs over all data samples
            x = gsl_matrix_row(X, i); // it picks sample x_i            
            partial_derivative_value+=((h(&x.vector, w)-gsl_vector_get(Y, i))*gsl_vector_get(&x.vector, j)); //tmp = sum(h(x_i)-y_i)x_i^j
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
double h(gsl_vector *x, gsl_vector *w){
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