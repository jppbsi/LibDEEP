#ifndef _DEEP_H_
#define _DEEP_H_

#ifdef __cplusplus
extern "C" {
#endif

/* system libraries */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include <ctype.h>

/* GSL libraries */
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

/* LibOPF library */
#include "OPF.h"

/* libDeep libraries */
#include "auxiliary.h"
#include "rbm.h"
#include "math_functions.h"
#include "dbn.h"
#include "regression.h"
#include "logistic.h"
#include "dbm.h"
#include "epnn.h"
#include "pca.h"

#ifdef __cplusplus
}
#endif

#endif