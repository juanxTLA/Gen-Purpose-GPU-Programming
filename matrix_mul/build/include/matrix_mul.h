#ifndef __matrix_mul_h__
#define __matrix_mul_h__

/**
    When DEBUG not commented the program has fixed 
    sized matrices instead of user input and prints naïve, tiled, and reference result matrices

    When CUBLAS commented, the reference will be the BLAS library, with no evaluation in performance 

    When FILEDATA is not commented, the performance results will not be outputted to a file for python graphing
**/

//#define DEBUG

//#define COARSE

#define FILEDATA

#define CUBLAS

#ifdef CUBLAS
    #include "cublas_v2.h"
#endif

#ifdef DEBUG
    #define TILEDIM 16
#else
    #define TILEDIM 4
    //#define COARSE_FACTOR 2
#endif

#ifdef __cplusplus
extern "C"{
#endif

void matrix_mul (
    int     N, 
    float * aMatrixDevice,
    float * bMatrixDevice,
    float * cMatrixDevice
);

void matrix_mul_tile (
    int     N, 
    float * aMatrixDevice,
    float * bMatrixDevice,
    float * cMatrixDeviceTile
);

void matrix_mul_tile_coarse(
    int     N,
    float * aMatrixDevice,
    float * bMatrixDevice,
    float * cMatrixDeviceTile
);

#ifdef __cplusplus
}
#endif

#endif