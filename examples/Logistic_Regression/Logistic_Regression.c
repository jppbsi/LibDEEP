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

	if (argc != 4)
	{
		fprintf(stderr, "\nusage Logistic_Regression <training set> <testing set> <learning rate>\n");
		exit(-1);
	}

	int i, j, m_train, n_train, m_test, n_test;
	double alpha = atof(argv[3]), errorTrain, errorTest;
	double **XTrain = NULL, **XTest = NULL;
	double *YTrain = NULL, *YTest = NULL, *w = NULL;
	FILE *fp = NULL;
	Subgraph *gTrain = NULL, *gTest = NULL;

	LoadData(argv[1], &XTrain, &YTrain, &m_train, &n_train);
	LoadData(argv[1], &XTest, &YTest, &m_test, &n_test);
	w = (double *)calloc(n_train, sizeof(double));

	/* mapping training data to LibOPF format */
	gTrain = CreateSubgraph(m_train);
	gTrain->nfeats = n_train;
	gTrain->nlabels = 2;
	for (i = 0; i < m_train; i++)
	{
		gTrain->node[i].feat = AllocFloatArray(n_train);
		for (j = 0; j < n_train; j++)
			gTrain->node[i].feat[j] = XTrain[i][j];
		gTrain->node[i].truelabel = YTrain[i];
	}

	/* mapping testing data to LibOPF format */
	gTest = CreateSubgraph(m_test);
	gTest->nfeats = n_test;
	gTest->nlabels = 2;
	for (i = 0; i < m_test; i++)
	{
		gTest->node[i].feat = AllocFloatArray(n_test);
		for (j = 0; j < n_test; j++)
			gTest->node[i].feat[j] = XTest[i][j];
		gTest->node[i].truelabel = YTest[i];
	}

	errorTrain = LogisticRegression_Fitting(gTrain, alpha, w);
	Logistic_Regression4Classification(gTest, w);

	fp = fopen("w_coefficients.txt", "w");
	for (i = 0; i < n_train; i++)
		fprintf(fp, "%lf ", w[i]);
	fclose(fp);

	errorTest = 0;
	for (i = 0; i < m_test; i++)
	{
		if (gTest->node[i].truelabel != gTest->node[i].label) /* Misclassification occurs */
			errorTest++;
	}
	fprintf(stderr, "\nClassification accuracy: %.2f%%\n", (1 - errorTest / m_test) * 100);

	for (i = 0; i < m_train; i++)
		free(XTrain[i]);
	free(XTrain);
	free(YTrain);
	for (i = 0; i < m_test; i++)
		free(XTest[i]);
	free(XTest);
	free(YTest);
	free(w);
	DestroySubgraph(&gTrain);
	DestroySubgraph(&gTest);

	return 0;
}