#include "deep.h"

void LoadData(char *fileName, double ***X, double **Y, int *m, int *n)
{
        FILE *fp = NULL;
        int i, j, m_tmp, n_tmp;
        double value;
        double **X_tmp = *X;
        double *Y_tmp = *Y;

        fp = fopen(fileName, "r");
        if (!fp)
        {
                fprintf(stderr, "\nunable to open file %s\n", fileName);
                exit(-1);
        }

        fscanf(fp, "%d %d", &m_tmp, &n_tmp);
        X_tmp = (double **)calloc(m_tmp, sizeof(double *));
        for (i = 0; i < m_tmp; i++)
                X_tmp[i] = (double *)calloc(n_tmp + 1, sizeof(double)); /* Adding extra dimension for x0 */
        Y_tmp = (double *)calloc(m_tmp, sizeof(double));

        for (i = 0; i < m_tmp; i++)
        {
                fscanf(fp, "%lf", &value); /* Reading the target first */
                Y_tmp[i] = value;
                X_tmp[i][0] = 1; /* Setting up x0 value */
                for (j = 1; j < n_tmp + 1; j++)
                {
                        fscanf(fp, "%lf", &value);
                        X_tmp[i][j] = value; /* Reading input features */
                }
        }
        fclose(fp);
        *X = X_tmp;
        *Y = Y_tmp;
        *m = m_tmp;
        *n = n_tmp;
}

int main(int argc, char **argv)
{

        if (argc != 3)
        {
                fprintf(stderr, "\nusage Linear_Regression <training set> <learning rate>\n");
                exit(-1);
        }

        int i, j, m, n;
        double alpha = atof(argv[2]), errorTrain;
        double **X = NULL;
        double *Y = NULL, *w = NULL;
        FILE *fp = NULL;

        LoadData(argv[1], &X, &Y, &m, &n);
        w = (double *)calloc(n + 1, sizeof(double));

        errorTrain = LinearRegression_Fitting(X, Y, m, n, alpha, w);

        fp = fopen("w_coefficients.txt", "w");
        for (i = 0; i < n + 1; i++)
                fprintf(fp, "%lf ", w[i]);
        fclose(fp);

        for (i = 0; i < m; i++)
                free(X[i]);
        free(X);
        free(Y);
        free(w);

        return 0;
}