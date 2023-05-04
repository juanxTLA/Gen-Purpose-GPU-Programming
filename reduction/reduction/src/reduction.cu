#include <reduction.h>
#include <helper_cuda.h>

#include <thrust/device_vector.h>
#include <cub/cub.cuh>

#define CUB_IGNORE_DEPRECATED_CPP_DIALECT  CUB_COMPILER_DEPRECATION_SOFT(C++14, C++11)

//SINGLE CUDA THREAD
__global__ void singleCudaThreadKernel(
    int * vector,
    int * res,
    int   n
){

    if(threadIdx.x == 0){
        for(int i = 0; i < n; i++){
            *res += vector[i];
        }
    }
}

void singleCudaThread(
    int   *   vector,
    int   *   res,
    int       n
) {

    singleCudaThreadKernel <<< 1, 1>>>(vector, res, n);

    checkCudaErrors(cudaGetLastError());

}

// ATOMIC VARIABLES
__global__ void atomicVarKernel(
    int * vector,
    int * res,
    int n
){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n) atomicAdd(res, vector[i]);
}

void atomicVar(
    int   *   vector,
    int   *   res,
    int       n
) {
    int blockSize = BLOCK_DIM;
    int gridSize = ceil((n + blockSize - 1)/blockSize);

    atomicVarKernel <<< gridSize, blockSize >>>(vector, res, n);

    checkCudaErrors(cudaGetLastError());
}

//SEGMENTED
__global__ void segmentedKernel(
    int *   vector,
    int *   res
) {
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + 2 * threadIdx.x;

    for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2){
        if(threadIdx.x % stride == 0)
            vector[i] += vector[i + stride]; 
        __syncthreads();
    }


    if(threadIdx.x == 0)
        atomicAdd(res, vector[i]);
}

void segmented(
    int *   vector,
    int *   res,
    int     n
) {
    int blockSize, min;
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
        &min, 
        &blockSize, 
        segmentedKernel, 0, 0)
    );

    int gridSize = (n/2 + blockSize-1)/blockSize;

    segmentedKernel <<< gridSize, blockSize >>> (vector, res);

    checkCudaErrors(cudaGetLastError());

}

//COALESCING
__global__ void coalescedKernel(
    int *   vector,
    int *   res
) {
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;

    for(unsigned int stride = blockDim.x; stride > 0; stride /= 2){
        if(threadIdx.x < stride)
            vector[i] += vector[i + stride];
        __syncthreads();
    }

  if(threadIdx.x == 0)
    atomicAdd(res, vector[i]);
}

void coalesced(
    int * vector,
    int * res,
    int   n
) {
    int blockSize, min;
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
        &min, 
        &blockSize, 
        coalescedKernel, 0, 0)
    );

    int gridSize = (n/2 + blockSize-1)/blockSize;

    coalescedKernel <<< gridSize, blockSize >>> (vector, res);

    checkCudaErrors(cudaGetLastError());
}

//SHARED SEGMENTED
__global__ void sharedSegmentedKernel(
    int *   vector,
    int *   res
) {
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;

    __shared__ int vectorShared[BLOCK_DIM];
    vectorShared[threadIdx.x] = vector[i] + vector[i + BLOCK_DIM];
    __syncthreads();

    for(unsigned int stride = BLOCK_DIM/2; stride > 0; stride /= 2) {
        if(threadIdx.x < stride)
            vectorShared[threadIdx.x] += vectorShared[threadIdx.x + stride];

        __syncthreads();
    }

    if(threadIdx.x == 0)
        atomicAdd(res, vectorShared[0]);
}

void sharedSegmented(
    int *   vector,
    int *   res,
    int     n
) {
    int blockSize = BLOCK_DIM;
    int gridSize = (n/2 + blockSize-1)/blockSize;

    sharedSegmentedKernel <<< gridSize, blockSize >>> (vector, res);

    checkCudaErrors(cudaGetLastError());
}

//COARSENED SHARED SEGMENTED
__global__ void sharedSegmentedCoarsenedKernel(
    int *   vector,
    int *   res
) {
    unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;

    __shared__ int vectorShared[BLOCK_DIM];
    int threadSum = 0;

    for(unsigned int c = 0; c < COARSE_FACTOR * 2; ++c){
        threadSum += vector[i + c*BLOCK_DIM];
    }

    vectorShared[threadIdx.x] = threadSum;
    __syncthreads();

    for(unsigned int stride = BLOCK_DIM/2; stride > 0; stride /= 2) {
        if(threadIdx.x < stride)
            vectorShared[threadIdx.x] += vectorShared[threadIdx.x + stride];

        __syncthreads();
    }
    

    atomicAdd(res, vectorShared[i]);

}

void sharedSegmentedCoarsened(
    int *   vector,
    int *   res,
    int     n
) {
    int blockSize = BLOCK_DIM;
    int gridSize = (n + blockSize - 1)/blockSize + 1;

    sharedSegmentedCoarsenedKernel <<< gridSize, blockSize >>> (vector, res);

    checkCudaErrors(cudaGetLastError());
}

//TESTING AGAINST THRUST AND CUB
void verify(
    char v, 
    int *vector,
    int *res, 
    int n
) {
    
    if(v == 't'){
        thrust::device_vector<int> thrustVec(vector, vector + n);
        *res =  thrust::reduce(thrustVec.begin(), thrustVec.end());
    }
    else if(v == 'c'){
        
        void *tempStore = NULL;
        size_t storeBytes = 0;

        cub::DeviceReduce::Sum(tempStore, storeBytes, vector, res, n);
        
        cudaMalloc(&tempStore, storeBytes);

        cub::DeviceReduce::Sum(tempStore, storeBytes, vector, res, n);

    }
}