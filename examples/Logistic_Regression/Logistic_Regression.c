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
    double alpha = atof(argv[3]), errorTRAIN;
    gsl_matrix *X = NULL;
    gsl_vector *Y = NULL, *w = NULL;
    FILE *fp = NULL;
    Subgraph *g = NULL;
    
    LoadData(argv[1], &X, &Y);
    w = gsl_vector_alloc(X->size2);
    
    /* mapping data to another format */
    g = CreateSubgraph(X->size1);
    g->nfeats = X->size2; g->nlabels = 2;
    for(i = 0; i < X->size1; i++){
	g->node[i].feat = AllocFloatArray(X->size2);
	for(j = 0; j < X->size2; j++)
	    g->node[i].feat[j] = gsl_matrix_get(X, i, j);
	g->node[i].truelabel = gsl_vector_get(Y, i); 
    }
    
    errorTRAIN = LogisticRegression_Fitting(g, GRADIENT_DESCENT, alpha, w);

    fp = fopen("w_coefficients.txt", "w");
    for(i = 0; i < w->size; i++)
        fprintf(fp,"%lf ", gsl_vector_get(w, i));
    fclose(fp);
    
    gsl_matrix_free(X);
    gsl_vector_free(Y);
    gsl_vector_free(w);
    DestroySubgraph(&g);
    
    return 0;
}
