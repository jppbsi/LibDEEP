#include "OPF.h"
#include "deep.h"

int main(int argc, char **argv){
    if(argc != 4){
        fprintf(stderr,"\nusage PCA <input file> <output file> <percentage of the final number of dimensions>\n");
        exit(-1);
    }
    
    Subgraph *in = NULL, *out = NULL;
    double p = atof(argv[3]);
    
    in = ReadSubgraph(argv[1]);
    out = PCA(in, p);
    
    WriteSubgraph(out, argv[2]);
    DestroySubgraph(&in);
    DestroySubgraph(&out);
    
    return 0;
}