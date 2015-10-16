#include "OPF.h"
#include "deep.h"
#include "opt.h"

int main(int argc, char **argv){

    if(argc != 4){
        fprintf(stderr,"\nusage Logistic_Regression <training set> <test set> <learning rate>\n");
        exit(-1);
    }

    int i,j;
    Subgraph *Train = NULL, *Test = NULL;
    double alpha = atof(argv[3]);
    gsl_matrix *X = NULL, *XTest = NULL;
    gsl_vector *Y = NULL, *YTest = NULL, *w = NULL, *predict = NULL;
    
    Train = ReadSubgraph(argv[1]);
    Test = ReadSubgraph(argv[2]);
    
    X = gsl_matrix_calloc(Train->nnodes, Train->nfeats);
    Y = gsl_vector_calloc(Train->nnodes);
    
    /* Reading training data */
    for(i = 0; i < X->size1; i++){
        for(j = 0; j < X->size2; j++)
            gsl_matrix_set(X, i, j, Train->node[i].feat[j]);
        gsl_vector_set(Y, i, Train->node[i].truelabel);
    }
    
    XTest = gsl_matrix_calloc(Test->nnodes, Test->nfeats);
    YTest = gsl_vector_calloc(Test->nnodes);
    
    /* Reading test data */
    for(i = 0; i < XTest->size1; i++){
        for(j = 0; j < XTest->size2; j++)
            gsl_matrix_set(XTest, i, j, Test->node[i].feat[j]);
        gsl_vector_set(YTest, i, Test->node[i].truelabel);
    }
    
    w = LogisticRegression_Fitting(X, Y, GRADIENT_DESCENT, alpha);
    predict = Logistic_Regression4Classification(XTest, w);
    
    for(i = 0; i < X->size1; i++)
        fprintf(stderr,"\n Test sample %d -> True label: %d -> Predicted label: %d", i, (int)gsl_vector_get(YTest, i), (int)gsl_vector_get(predict, i));

    
    DestroySubgraph(&Train);
    DestroySubgraph(&Test);
    gsl_matrix_free(X);
    gsl_matrix_free(XTest);
    gsl_vector_free(Y);
    gsl_vector_free(YTest);
    gsl_vector_free(w);
    gsl_vector_free(predict);
    
    return 0;
}
