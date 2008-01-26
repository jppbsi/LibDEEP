#ifndef _DEEP_H_
#define _DEEP_H_

#ifdef __cplusplus
extern "C" {
#endif

/* system libraries */
#include <stdarg.h>

/* GSL libraries */
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

/* LibOPF library */
#include "OPF.h"

typedef void (*mac_prtFun)(gsl_matrix *, gsl_vector *, ...);
    
/* libDeep libraries */
#include "auxiliary.h"
#include "rbm.h"
#include "math_functions.h"
#include "dbn.h"
#include "regression.h"

#ifdef __cplusplus
}
#endif

#endif
