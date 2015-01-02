#include "math_functions.h"

double SigmoidLogistic(double x){
    double y;
    
    y = 1.0/(1+exp(-x));
    
    return y;
}

double SoftPlus(double x){
    double y;

    y = log(1+exp(x));
    
    return y;
}