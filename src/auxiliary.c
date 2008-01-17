#include "auxiliary.h"

/* Functions related to the Dataset struct ***/

/* It creates a dataset */
Dataset *CreateDataset(int size, int nfeatures){
    Dataset *D = NULL;
    int i;
    
    D = (Dataset *)malloc(sizeof(Dataset));
    if(!D){
        fprintf(stderr,"\nDataset not allocated @CreateDataset\n");
        exit(-1);
    }
    
    D->size = size;
    D->nfeatures = nfeatures;
    
    D->sample = NULL; D->sample = (Sample *)malloc(D->size*sizeof(Sample));
    for(i = 0; i < D->size; i++)
        D->sample[i].feature = gsl_vector_alloc(D->nfeatures);

    return D;
}

/* It destroys a dataset */
void DestroyDataset(Dataset **D){
    int i;
    if(*D){
        if((*D)->sample){
            for(i = 0; i < (*D)->size; i++)
                gsl_vector_free((*D)->sample[i].feature);
            free((*D)->sample);
        }
        free(*D);
    }
}

/* It copies a given dataset */
Dataset *CopyDataset(Dataset *d){
    Dataset *cpy = NULL;
    int i;
    
    if(d){
    
        cpy = CreateDataset(d->size, d->nfeatures);
        cpy->nlabels = d->nlabels;
    
        for(i = 0; i < cpy->size; i++){
            gsl_vector_memcpy(cpy->sample[i].feature, d->sample[i].feature);
            cpy->sample[i].label = d->sample[i].label;
        }
    }else fprintf(stderr, "\nThere is no dataset allocated @CopyDataset\n");
    
    return cpy;
}
/**********************************************/

/* Image classification functions */

/* It verifies if a given string ends with a given suffix */
/*int Endswith (char *string, char *suffix){
    int i = 0;
    int len_suffix = strlen(suffix);
    int len_string = strlen(string);
    for (i = 0; i < len_suffix ; i++){
        if (suffix[i] != string[len_string - len_suffix + i])
            return 0;
    }
    return 1;
}*/

/* It gets the following information about the images in the input dataset: numer of images, 
 width and height (it assumes all images have the same dimensions). It outputs the following information:
 - output[0] = number of images
 - output[1] = image's height
 - output[2] = image's width */
/*int *getImagesInformation(char *directory_path, char *file_extension){
    int *output = NULL, file_count = 0;
    DIR *directory_pointer = NULL;
    struct dirent *entry = NULL;
    IplImage *img = NULL;
    char filename[256], FLAG = 1;
    
    directory_pointer = opendir(directory_path);
    if (!directory_path){
        fprintf(stderr, "Error opening directory path @CountImages");
        exit(-1);
    }
    
    output = (int *)malloc(3*sizeof(int));
    while ((entry = readdir(directory_pointer)) != NULL){
        if (Endswith(entry->d_name, file_extension)){
            file_count++;
            if(FLAG){
                sprintf(filename,"%s%s",directory_path,entry->d_name);
                FLAG = 0;
            }
        }
    }
    closedir(directory_pointer);
    output[0] = file_count;
    
    img = cvLoadImage(filename, -1);
    output[1] = img->height;
    output[2] = img->width;
    cvReleaseImage(&img);
    
    return output;
}*/

/* Position means the segment position in the strings. The first segment is 0. */
char *SplitString(char *string, char * separator, int position){
    char *pch;
    char *copy = strdup(string);
    if (!copy){
        fprintf(stderr,"\nError allocation char array @splitString\n");
        exit(-1);
    }
    int i = 0;
    
    pch = strtok (copy, separator);
    while (pch != NULL && i < position){
        pch = strtok (NULL, separator);
        i++;
    }
    if (i < position){
        fprintf(stderr,"\nPosition bigger then segments on the string @splitString\n");
        exit(-1);
    }
    
    return pch;
}

//It loads a dataset from set of images
/*void LoadDatasetFromImages(Dataset *D, char *directory_path, char *file_extension){
    if(D){
        int z = 0, i, j, w, max_label = 0;
        DIR *directory_pointer = NULL;
        struct dirent *entry = NULL;
        char *class = NULL, filename[256];
        IplImage *img = NULL;
        CvScalar s;
        
        directory_pointer = opendir(directory_path);
        if(!directory_pointer){
            fprintf(stderr,"\nError opening directory path @LoadDatasetFromImages");
            exit(-1);
        }
        while ((entry = readdir(directory_pointer)) != NULL){
            if (Endswith(entry->d_name, file_extension)){
                class = SplitString(entry->d_name, "_", 0);
                D->sample[z].label = atoi(class); free(class);
                if(D->sample[z].label > max_label) max_label = D->sample[z].label;
                
                sprintf(filename, "%s%s", directory_path, entry->d_name);
                img = cvLoadImage(filename, -1); w = 0;
                for(i = 0; i < img->height; i++){
                    for(j = 0; j < img->width; j++){
                        s = cvGet2D(img,i,j);
                        gsl_vector_set(D->sample[z].feature, w, s.val[0]);
                        if(gsl_vector_get(D->sample[z].feature, w) == 255) gsl_vector_set(D->sample[z].feature, w, 1.0); 
                        w++;
                    }
                }
                
                cvReleaseImage(&img);
                z++;
            }
        }
        D->nlabels = max_label;
        
    }else{
        fprintf(stderr,"\nThere is not any dataset allocated @LoadDatasetFromImages\n");
        exit(-1);
    }
    
}*/

/**********************************************/

// It converts a Dataset to a Subgraph
Subgraph *Dataset2Subgraph(Dataset *D){
    Subgraph *g = NULL;
    int i, j;
    
    g = CreateSubgraph(D->size);
    g->nfeats = D->nfeatures;
    g->nlabels = D->nlabels;
    for(i = 0; i < g->nnodes; i++){
        g->node[i].feat = AllocFloatArray(g->nfeats);
        g->node[i].truelabel = D->sample[i].label;
        g->node[i].label = D->sample[i].predict;
        g->node[i].position = i;
        for(j = 0; j < g->nfeats; j++)
            g->node[i].feat[j] = (float)gsl_vector_get(D->sample[i].feature, j);
    }
    
    return g;
}

// It converts a Subgraph to a Dataset
Dataset *Subgraph2Dataset(Subgraph *g){
    Dataset *D = NULL;
    int i, j;
    
    D = CreateDataset(g->nnodes, g->nfeats);
    D->nlabels = g->nlabels;
    
    for(i = 0; i< D->size; i++){
        D->sample[i].label = g->node[i].truelabel;
        
        for(j = 0; j < D->nfeatures; j++)
            gsl_vector_set(D->sample[i].feature, j, g->node[i].feat[j]);
    }
    
    return D;
}

/* It converts an integer to a set of bits Ex: for a 3-bit representation, if label = 2, output = 010 */
gsl_vector *label2binary_gsl_vector(int l, int n_bits){
    gsl_vector *y = NULL;
    int i;
    
    y = gsl_vector_calloc(n_bits);
    gsl_vector_set_zero(y);
    gsl_vector_set(y, l-1, 1.0);
    
    return y;
    
}

/* It generates a random seed */
unsigned long int random_seed_deep(){
    struct timeval tv;
    gettimeofday(&tv,0);
    return (tv.tv_sec + tv.tv_usec);
}
