#include "deep.h"

int main(int argc, char **argv)
{

    if (argc != 5)
    {
        fprintf(stderr, "\nusage ANN-RBF <training set> <test set> <1: OPF or 2: K-Means> <kmax (OPF) or k value (number of gaussians for k-means)>\n");
        exit(-1);
    }

    int i, type_function = atoi(argv[3]), kvalue = atoi(argv[4]);
    timer tic, toc;
    float accuracy, time;
    char fileName[30];
    FILE *f = NULL;
    gsl_matrix **cov = NULL, *mu = NULL, *w = NULL;
    Subgraph *Train = NULL, *Test = NULL;

    fprintf(stderr, "\nReading datasets ... ");
    Train = ReadSubgraph(argv[1]);
    Test = ReadSubgraph(argv[2]);
    fprintf(stderr, "OK\n");

    if (type_function == 1)
    {
        /* Allocating cov and mu with the actual number of clusters (Train -> nlabels) */
        mu = gsl_matrix_calloc(Train->nlabels, Train->nfeats);
        cov = (gsl_matrix **)calloc(Train->nlabels, sizeof(gsl_matrix **));
        for (i = 0; i < Train->nlabels; i++)
            cov[i] = gsl_matrix_calloc(Train->nfeats, Train->nfeats);

        fprintf(stderr, "\nTraining by OPF ... \n");
        gettimeofday(&tic, NULL);
        w = TrainANNbyOPF(Train, mu, cov, kvalue);
        gettimeofday(&toc, NULL);
        fprintf(stderr, "OK\n");
    }
    else if (type_function == 2)
    {
        /* Allocating cov and mu with the actual number of clusters (kvalue) */
        mu = gsl_matrix_calloc(kvalue, Train->nfeats);
        cov = (gsl_matrix **)calloc(kvalue, sizeof(gsl_matrix **));
        for (i = 0; i < kvalue; i++)
            cov[i] = gsl_matrix_calloc(Train->nfeats, Train->nfeats);

        fprintf(stderr, "\nTraining by K-Means ... ");
        gettimeofday(&tic, NULL);
        w = TrainANNbyKMeans(Train, mu, cov, kvalue);
        gettimeofday(&toc, NULL);
        fprintf(stderr, "OK\n");
    }

    fprintf(stderr, "\nTesting ... ");
    ClassifyANN(Test, mu, cov, w);
    fprintf(stderr, "OK\n");

    accuracy = opf_Accuracy(Test);
    fprintf(stderr, "\nAccuracy: %.2f%%\n", accuracy * 100);

    fprintf(stderr, "\nWriting output file ...");
    sprintf(fileName, "%s.out", argv[2]);
    f = fopen(fileName, "w");
    for (i = 0; i < Test->nnodes; i++)
        fprintf(f, "%d\n", Test->node[i].label);
    fclose(f);
    fprintf(stderr, " OK");

    fprintf(stderr, "\nDeallocating memory ... ");
    gsl_matrix_free(mu);
    if (type_function == 1)
    {
        for (i = 0; i < Test->nlabels; i++)
            gsl_matrix_free(cov[i]);
    }
    else if (type_function == 2)
    {
        for (i = 0; i < kvalue; i++)
            gsl_matrix_free(cov[i]);
    }
    free(cov);
    DestroySubgraph(&Train);
    DestroySubgraph(&Test);
    fprintf(stderr, "OK\n");

    fflush(stdout);
    time = ((toc.tv_sec - tic.tv_sec) * 1000.0 + (toc.tv_usec - tic.tv_usec) * 0.001) / 1000.0;
    fprintf(stdout, "\nTraining time: %f seconds\n", time);
    fflush(stdout);

    return 0;
}