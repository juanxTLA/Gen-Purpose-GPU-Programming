#ifndef __reduction_h__
#define __reduction_h__

#define BLOCK_DIM 1024
#define COARSE_FACTOR 2

using namespace std;


#ifdef __cplusplus
extern "C"{
#endif

    void singleCudaThread(
        int    *    vector,
        int    *    res,
        int         n
    );

    
    void atomicVar(
        int   *   vector,
        int   *   res,
        int       n
    );

    
    void segmented(
        int *   vector,
        int *   res,
        int     n
    );

    void coalesced(
        int * vector,
        int * res,
        int   n
    );

    void sharedSegmented(
        int *   vector,
        int *   res,
        int     n
    );

    void sharedSegmentedCoarsened(
        int *   vector,
        int *   res,
        int     n
    );

    void verify(
        char v, 
        int *vector,
        int *res,
        int n
    );
    
#ifdef __cplusplus
}
#endif
#endif