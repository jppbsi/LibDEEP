#include "regression.h"

/* Linear Regression */

/* It fits a linear regression model using the Minimum Square Error as the error function
Parameters: [Train, w, Optimization_Func, ...]
Train: training set
w: parameters of the linear function
Optimization_Func: function used to find the parameters that best fits the linear model
remaining parameters of each specific optimization function */
void LinearRegression_Fitting(Subgraph *Train, gsl_vector *w, mac_prtFun Optimization_Func, ...){
    double alpha;
    va_list arg;
    
    va_start(arg, alpha);
    
    while(abs(error-old_error) > 0.00001){
        Optimization_Func(Train, w, alpha);
    }
    
    va_end(arg);
}