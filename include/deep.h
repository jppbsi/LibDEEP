#ifndef _DEEP_H_
#define _DEEP_H_

#ifdef __cplusplus
extern "C" {
#endif

/* LibOPF library */
#include "OPF.h"

typedef void (*mac_prtFun)(Subgraph *, ...);
    
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
