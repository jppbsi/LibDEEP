#include "OPF.h"
#include "deep.h"
#include "opt.h"

void LoadData(char *fileName, gsl_matrix **X, gsl_vector **Y){
        FILE *fp = NULL;
        int m,n,i,j;
        double value;
        gsl_matrix *X_tmp = *X;
        gsl_vector *Y_tmp = *Y;
        
        fp = fopen(fileName, "r");
        if(!fp){
            fprintf(stderr,"\nunable to open file %s\n", fileName);
            exit(-1);
        }
        
        fscanf(fp,"%d %d", &m, &n);
        X_tmp = gsl_matrix_calloc(m, n+1); //adding extra dimension for x0
        Y_tmp = gsl_vector_calloc(m);
        
        for(i = 0; i < m; i++){
            fscanf(fp,"%lf",&value); //reading the target first
            gsl_vector_set(Y_tmp, i, value);
            gsl_matrix_set(X_tmp, i, 0, 1.0); //setting up x0 value
            for(j = 1; j < n+1; j++){
                fscanf(fp,"%lf",&value);
                gsl_matrix_set(X_tmp, i, j, value); //reading input feature
            }
        }
            
        fclose(fp);
        *X = X_tmp;
        *Y = Y_tmp;
}

int main(int argc, char **argv){

    if(argc != 4){
        fprintf(stderr,"\nusage Logistic_Regression <training set> <testing set> <learning rate>\n");
        exit(-1);
    }
    
    int i,j;
    double alpha = atof(argv[3]), errorTRAIN, errorTEST;
    gsl_matrix *XTrain = NULL, *XTest = NULL;
    gsl_vector *YTrain = NULL, *YTest = NULL, *w = NULL;
    FILE *fp = NULL;
    Subgraph *gTrain = NULL, *gTest = NULL;
    
    LoadData(argv[1], &XTrain, &YTrain);
    LoadData(argv[1], &XTest, &YTest);
    w = gsl_vector_alloc(XTrain->size2);
    
    /* mapping training data to LibOPF format */
    gTrain = CreateSubgraph(XTrain->size1);
    gTrain->nfeats = XTrain->size2; gTrain->nlabels = 2;
    for(i = 0; i < XTrain->size1; i++){
	gTrain->node[i].feat = AllocFloatArray(XTrain->size2);
	for(j = 0; j < XTrain->size2; j++)
	    gTrain->node[i].feat[j] = gsl_matrix_get(XTrain, i, j);
	gTrain->node[i].truelabel = gsl_vector_get(YTrain, i); 
    }
    
    /* mapping testing data to LibOPF format */
    gTest = CreateSubgraph(XTest->size1);
    gTest->nfeats = XTest->size2; gTest->nlabels = 2;
    for(i = 0; i < XTest->size1; i++){
	gTest->node[i].feat = AllocFloatArray(XTest->size2);
	for(j = 0; j < XTest->size2; j++)
	    gTest->node[i].feat[j] = gsl_matrix_get(XTest, i, j);
	gTest->node[i].truelabel = gsl_vector_get(YTest, i); 
    }
    
    errorTRAIN = LogisticRegression_Fitting(gTrain, GRADIENT_DESCENT, alpha, w);
    Logistic_Regression4Classification(gTest, w);

    fp = fopen("w_coefficients.txt", "w");
    for(i = 0; i < w->size; i++)
        fprintf(fp,"%lf ", gsl_vector_get(w, i));
    fclose(fp);
    
    errorTEST = 0;
    for(i = 0; i < XTest->size1; i++){
        if(gTest->node[i].truelabel != gTest->node[i].label) //misclassification occurs
            errorTEST++;
    }
    fprintf(stderr,"\nClassification accuracy: %.2f%%\n", (1-errorTEST/XTest->size1)*100);
    
    gsl_matrix_free(XTrain);
    gsl_matrix_free(XTest);
    gsl_vector_free(YTrain);
    gsl_vector_free(YTest);
    gsl_vector_free(w);
    DestroySubgraph(&gTrain);
    DestroySubgraph(&gTest);
    
    return 0;
}
