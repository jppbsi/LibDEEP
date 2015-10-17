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
        X_tmp = gsl_matrix_calloc(m, n);
        Y_tmp = gsl_vector_calloc(m);
        
        for(i = 0; i < m; i++){
            fscanf(fp,"%lf",&value); //reading the target first
            gsl_vector_set(Y_tmp, i, value);
            for(j = 0; j < n; j++){
                fscanf(fp,"%lf",&value);
                gsl_matrix_set(X_tmp, i, j, value); //reading input feature
            }
        }
            
        fclose(fp);
        *X = X_tmp;
        *Y = Y_tmp;
}

int main(int argc, char **argv){

    if(argc != 3){
        fprintf(stderr,"\nusage Linear_Regression <training set> <learning rate>\n");
        exit(-1);
    }

    int i,j;
    double alpha = atof(argv[2]), errorTRAIN;
    gsl_matrix *X = NULL;
    gsl_vector *Y = NULL, *w = NULL;
    FILE *fp = NULL;
    
    LoadData(argv[1], &X, &Y);
    w = gsl_vector_alloc(X->size2);
    
    errorTRAIN = LinearRegression_Fitting(X, Y, GRADIENT_DESCENT, alpha, w);

    fp = fopen("w_coefficients.txt", "w");
    for(i = 0; i < w->size; i++)
        fprintf(fp,"%lf ", gsl_vector_get(w, i));
    fclose(fp);
    
    gsl_matrix_free(X);
    gsl_vector_free(Y);
    gsl_vector_free(w);
    
    return 0;
}
