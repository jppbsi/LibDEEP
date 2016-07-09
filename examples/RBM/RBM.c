#include "deep.h"

int main(int argc, char **argv){

    if(argc != 10){
        fprintf(stderr,"\nusage RBM <training set> <test set> <output results file name> <cross-validation iteration number> \
                <input parameters file> <n_epochs> <batch_size> <number of iterations for Constrastive Divergence> \
                <1 - CD | 2 - PCD | 3 - FPCD>\n");
        exit(-1);
    }
    int iteration = atoi(argv[4]), i, n_epochs = atoi(argv[6]), batch_size = atoi(argv[7]), n_gibbs_sampling = atoi(argv[8]), op = atoi(argv[9]);
    int n_hidden_units;
    double eta, lambda, alpha, eta_min, eta_max;
    double p, q;
    double errorTRAIN, errorTEST;
    char *fileName = argv[5];
    FILE *fp = NULL;
    Dataset *DatasetTrain = NULL, *DatasetTest = NULL;
    RBM *m = NULL;
    
    Subgraph *Train = NULL, *Test = NULL;
    Train = ReadSubgraph(argv[1]);
    Test = ReadSubgraph(argv[2]);
    
    DatasetTrain = Subgraph2Dataset(Train);
    DatasetTest = Subgraph2Dataset(Test);
    
    fp = fopen(fileName, "r");
    if(!fp){
        fprintf(stderr,"\nUnable to open file %s.\n", fileName);
        exit(1);
    }
    fscanf(fp, "%d %lf %lf %lf", &n_hidden_units, &eta, &lambda, &alpha);
    WaiveLibDEEPComment(fp);
    fscanf(fp, "%lf %lf", &eta_min, &eta_max);
    WaiveLibDEEPComment(fp);
    fscanf(fp, "%lf %lf", &p, &q);
    WaiveLibDEEPComment(fp);
    fclose(fp);
    
    fprintf(stderr,"\nCreating and initializing RBM ... ");
    m = CreateRBM(Train->nfeats, n_hidden_units, 1);
    m->eta = eta;
    m->lambda = lambda;
    m->alpha = alpha;
    m->eta_min = eta_min;
    m->eta_max = eta_max;
    InitializeWeights(m);    
    InitializeBias4HiddenUnits(m);
    InitializeBias4VisibleUnitsWithRandomValues(m);
    fprintf(stderr,"\nOk\n");
    
    fprintf(stderr,"\nTraining RBM ...\n");
    switch (op){
        case 1:
            errorTRAIN = BernoulliRBMTrainingbyContrastiveDivergencewithDropout(DatasetTrain, m, n_epochs, 1, batch_size, p, q);
        break;
        case 2:
            errorTRAIN = BernoulliRBMTrainingbyPersistentContrastiveDivergencewithDropout(DatasetTrain, m, n_epochs, n_gibbs_sampling, batch_size, p, q);
        break;
        case 3:
            errorTRAIN = BernoulliRBMTrainingbyFastPersistentContrastiveDivergencewithDropout(DatasetTrain, m, n_epochs, n_gibbs_sampling, batch_size, p, q);
        break;
    }
    fprintf(stderr,"\nOK\n");
    
    fprintf(stderr,"\nRunning RBM for reconstruction ... ");
    errorTEST = BernoulliRBMReconstructionwithDropout(DatasetTest, m, p, q);
    fprintf(stderr,"\nOK\n");
        
    fp = fopen(argv[3], "a");
    fprintf(fp,"\n%d %lf %lf", iteration, errorTRAIN, errorTEST);
    fclose(fp);
    
    fprintf(stderr,"\nTraining Error: %lf \nTesting Error: %lf\n\n", errorTRAIN, errorTEST);
    
    DestroyDataset(&DatasetTrain);
    DestroyDataset(&DatasetTest);
    DestroySubgraph(&Train);
    DestroySubgraph(&Test);
    DestroyRBM(&m);
    
    return 0;
}