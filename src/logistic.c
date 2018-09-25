#include "logistic.h"

/* Logistic Regression */

/* It executes the logistic regression
Parameters: [X, w, Y]
X: input data
w: learned parameters of the model
Y: target values */
double Logistic_Regression(Subgraph *g, double *w)
{
    int i;
    double error, h_w;
    double *x = NULL;

    if (g && w)
    {
        error = 0.0;
        for (i = 0; i < g->nnodes; i++)
        {                                                       /* It runs over all data samples */
            x = node2double_vector(g->node[i].feat, g->nfeats); /* It picks sample x_i */
            h_w = h_logistic(x, w, g->nfeats);
            error += (g->node[i].truelabel * log(h_w + 0.000001) + ((1 - g->node[i].truelabel) * log(1 - h_w + 0.000001))); /* Equation 23 */
            free(x);
        }
        return -error / g->nnodes;
    }
    else
    {
        fprintf(stderr, "\nThere is no data and/or w allocated @Logistic_Regression.\n");
        return -1.0;
    }
}

/* It executes the partial derivative of variable j concerning linear regression with MSE as the cost function
Parameters: [X, w, Y, j]
X: input data
w: learned parameters of the model
Y: target values
j: ID of the feature */
double Logistic_RegressionPartialDerivative(Subgraph *g, double *w, int j)
{
    int i;
    double partial_derivative_value;
    double *x = NULL;

    if (g && w)
    {
        partial_derivative_value = 0.0;
        for (i = 0; i < g->nnodes; i++)
        {                                                                                              /* It runs over all data samples */
            x = node2double_vector(g->node[i].feat, g->nfeats);                                        /* It picks sample x_i */
            partial_derivative_value += ((h_logistic(x, w, g->nfeats) - g->node[i].truelabel) * x[j]); /* tmp = sum(h(x_i)-y_i)x_i^j */
            free(x);
        }
        return partial_derivative_value;
    }
    else
    {
        fprintf(stderr, "\nThere is no data and/or w allocated @Logistic_RegressionPartialDerivative.\n");
        return DBL_MAX;
    }
}

/* It executes the hypothesis function
Parameters: [x, w]
x: input sample
w: parameters of the model */
double h_logistic(double *x, double *w, int n)
{
    int i;
    double tmp = 0.0;

    if (x && w)
    {
        for (i = 0; i < n; i++)
            tmp += (w[i] * x[i]);     /* tmp = wTx */
        return 1.0 / (1 + exp(-tmp)); /* Equation 19 */
    }
    else
    {
        fprintf(stderr, "\nThere is no x and/or w allocated @h_logistic.\n");
        return -1;
    }
}

/* It executes optimization through Batch Gradient Descent
Parameters: [g, alpha, w]
g: training graph
alpha: learning rate
w: parameters of the model */
double GradientDescentLogistic(Subgraph *g, double alpha, double *w)
{
    double *w_tmp = NULL;
    double error = 0.0, old_error = DBL_MAX, tmp;
    int i, j, m = g->nnodes, n = g->nfeats, max_iteration = 10000;

    if (g)
    {
        w_tmp = (double *)calloc(n, sizeof(double));
        memcpy(w_tmp, w, n * sizeof(double));
        i = 1;
        while ((fabs(error - old_error) > 0.000001) && (i <= max_iteration))
        {
            old_error = error;
            for (j = 0; j < n; j++)
            {
                tmp = w_tmp[j] - (alpha / m) * Logistic_RegressionPartialDerivative(g, w_tmp, j); /* tmp = w_j - alpha*1/m*sum(h(x_i)-y_i)x_i^j */
                w[j] = tmp;
            }
            memcpy(w_tmp, w, n * sizeof(double));
            error = Logistic_Regression(g, w);
            fprintf(stderr, "\nIteration: %d -> cost function value: %lf", i, error);
            i++;
        }
        fprintf(stderr, "\nError over the training set %.7lf", error);
    }
    else
        fprintf(stderr, "\n.There is no data allocated @GradientDescentLogistic\n");

    free(w_tmp);

    return error;
}

/* It fits a logistic regression model using Equation 21 as the error function optimized by Gradient Descent
Parameters: [X, Y, m, n, alpha, w]
X: input data
Y: target values
m: matrix first dimension
n: matrix second dimension
alpha: learning rate
w: parameters of the linear function */
double LogisticRegression_Fitting(Subgraph *g, double alpha, double *w)
{
    int i;
    double error;

    for (i = 0; i < g->nfeats; i++) /* It initalizes w with a uniform distribution [0,1] -> small values */
        w[i] = RandomFloat(0, 1);

    error = GradientDescentLogistic(g, alpha, w); /* Gradient descent optimization */

    return error;
}

/* It executes the Logistic Regression for classification purposes
Parameters: [X, w]
X: test set
w: parameters of the model */
void Logistic_Regression4Classification(Subgraph *Test, double *w)
{
    int i;
    double *x = NULL;

    for (i = 0; i < Test->nnodes; i++)
    {                                                             /* For each test sample */
        x = node2double_vector(Test->node[i].feat, Test->nfeats); /* It picks sample x_i */
        Test->node[i].label = round(h_logistic(x, w, Test->nfeats));
        free(x);
    }
}