#include "logistic.h"

/* Logistic Regression */

/* It executes the logistic regression
Parameters: [X, w, Y]
X: input data
w: learned parameters of the model
Y: target values
---
Output: error */
double Logistic_Regression(gsl_matrix *X, gsl_vector *w, gsl_vector *Y){
    double error, tmp, h_w;
    gsl_vector_view row;
    int i;
    
    if(X && w){
        
        error = 0.0;
        for(i = 0; i < X->size1; i++){ // it runs over all data samples
            row = gsl_matrix_row(X, i);
			h_w = h_logistic(&row.vector, w);
			//fprintf(stderr,"\nh_w: %lf with class %lf -> %lf", h_w, gsl_vector_get(Y, i), gsl_vector_get(Y, i)*log(h_w)+((1-gsl_vector_get(Y, i))*log(1-h_w)));
            error+=(gsl_vector_get(Y, i)*log(h_w)+((1-gsl_vector_get(Y, i))*log(1-h_w))); //Equation 23
        }
		fprintf(stderr, "\nerror: %lf", -error/X->size1);
        return -error/X->size1;
        
    }else{
        fprintf(stderr,"\nThere is no X and/or w allocated @Logistic_Regression.\n");
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
double Logistic_RegressionPartialDerivative(gsl_matrix *X, gsl_vector *w, gsl_vector *Y, int j){
    double partial_derivative_value, tmp;
    gsl_vector_view x;
    int i;
    
    if(X && w){
        
        partial_derivative_value = 0.0;
        for(i = 0; i < X->size1; i++){ // it runs over all data samples
            x = gsl_matrix_row(X, i); // it picks sample x_i            
            partial_derivative_value+=((h_logistic(&x.vector, w)-gsl_vector_get(Y, i))*gsl_vector_get(&x.vector, j)); //tmp = sum(h(x_i)-y_i)x_i^j
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