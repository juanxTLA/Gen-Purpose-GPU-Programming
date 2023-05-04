#ifndef __convolution_h__
#define __convolution_h__

/**
    When DEBUG not commented the program has fixed 
    sized matrices instead of user input and prints naïve, tiled, and reference result matrices

    When CUBLAS commented, the reference will be the BLAS library, with no evaluation in performance 

    When FILEDATA is not commented, the performance results will not be outputted to a file for python graphing
**/

//#define DEBUG

//#define COARSE


#define PI 3.14159
#define FILTDIM 5
#define FILTRAD 2
#define INTILEDIM 16
#define OUTDIM ((INTILEDIM) - 2 * (FILTRAD))

#ifdef __cplusplus
extern "C"{
#endif

void convolutionNaive (
    int     width,
    int     height,
    int     filtDim,
    float * grayImageDevice,
    float * filtImageDevice,
    float * filtGaussDevice
);

void convolutionShared(
    int     width,
    int     height,
    int     filtRad,
    float * grayImageDevice,
    float * filtImageDevice,
    float * filtGaussDevice
);

void convolutionConst(
    int     width,
    int     height,
    int     filtRad,
    float * grayImageDevice,
    float * filtImageDevice
);

void convolutionSharedTile(
    int     width,
    int     height,
    int     filtRad,
    float * grayImageDevice,
    float * filtImageDevice,
    float * filtGaussDevice
);

void convolutionSharedTileConst(
    int     width,
    int     height,
    int     filtRad,
    float * grayImageDevice,
    float * filtImageDevice
);

void convolutionSameTileConst(
    int     width,
    int     height,
    int     filtRad,
    float * grayImageDevice,
    float * filtImageDevice
);

#ifdef __cplusplus
}
#endif

#endif