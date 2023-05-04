#include <matrix_mul.h>
#include <helper_cuda.h>


__global__ void matrix_mul_kernel(
    int     N,
    float * aMatrixDevice,
    float * bMatrixDevice,
    float * cMatrixDevice
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int off = row * N + col; //pos we are computing
    
    float sum = 0;

    if(row < N && col < N){
        for(int i = 0; i < N; i++){
                sum += aMatrixDevice[row * N + i] * bMatrixDevice[col + i*N];
        }

        cMatrixDevice[off] = sum;
    }

}

__global__ void matrix_mul_tile_kernel(
    int     N,
    float * aMatrixDevice,
    float * bMatrixDevice,
    float * cMatrixDevice
) { 
    __shared__ float aTile[TILEDIM][TILEDIM];
    __shared__ float bTile[TILEDIM][TILEDIM];

    unsigned int row = blockIdx.y * TILEDIM + threadIdx.y;
    unsigned int col = blockIdx.x * TILEDIM + threadIdx.x;
    unsigned int off = row * N + col; //pos we are computing

    float sum = 0.0f;
    if(row < N && col < N){
        for(unsigned int tile = 0; tile < N/TILEDIM; ++tile){
            if((row < N) && (tile*TILEDIM + threadIdx.x) < N)
                aTile[threadIdx.y][threadIdx.x] = aMatrixDevice[row * N + tile*TILEDIM + threadIdx.x];
            
            if((tile*TILEDIM + threadIdx.y) < N && col < N)
                bTile[threadIdx.y][threadIdx.x] = bMatrixDevice[(tile*TILEDIM + threadIdx.y) * N + col];
            
            __syncthreads();

            for(unsigned int i = 0; i < TILEDIM; ++i){  
                if(i + (tile * TILEDIM) < N)    
                    sum += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];
            }
            
            __syncthreads();
        }

    
        cMatrixDevice[off] = sum;
    }
    
}

__global__ void matrix_mul_tile_coarse_kernel(
    float   *   aMatrixDevice,
    float   *   bMatrixDevice,
    float   *   cMatrixDevice,
    int         N,
    int         COARSE_FACTOR
) {
    __shared__ float aTile[TILEDIM][TILEDIM];
    __shared__ float bTile[TILEDIM][TILEDIM];

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y; 
    unsigned int colStart = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x; 

    float *sum = (float*)malloc(sizeof(float) * COARSE_FACTOR); //dynamically allocated

    for(unsigned int c = 0; c < COARSE_FACTOR; ++c){
        sum[c] = 0.0f;
    }

    for(unsigned int tile = 0; tile < N/TILEDIM; ++tile){
        //if((row < N) && (tile*TILEDIM + threadIdx.x) < N)
            aTile[threadIdx.y][threadIdx.x] = aMatrixDevice[row*N + tile*TILEDIM + threadIdx.x];
        for(unsigned int c = 0; c < COARSE_FACTOR; ++c){
            unsigned int col = colStart + c * TILEDIM;
            //Load B tile
            //if((tile*TILEDIM + threadIdx.y) < N && col < N)
                bTile[threadIdx.y][threadIdx.x] = bMatrixDevice[(tile*TILEDIM + threadIdx.y) * N + col];
            __syncthreads();
            //compute with tile
            for(unsigned int i = 0; i < TILEDIM; ++i){
                sum[c] += aTile[threadIdx.y][i] * bTile[i][threadIdx.x]; 
            }
            __syncthreads();
        }
    }

    for(unsigned int c = 0; c < COARSE_FACTOR; ++c){
        unsigned int col = colStart + c * TILEDIM;
        cMatrixDevice[row*N + col] = sum[c];
    }

 
    delete sum;
}

void matrix_mul(
    int     N,
    float * aMatrixDevice,
    float * bMatrixDevice,
    float * cMatrixDevice
) {

    dim3 numBlocks(16,16);
    dim3 gridSize((N + numBlocks.x - 1)/numBlocks.x,
                  (N + numBlocks.y - 1)/numBlocks.y);

    matrix_mul_kernel <<< gridSize, numBlocks >>>(
        N,
        aMatrixDevice,
        bMatrixDevice,
        cMatrixDevice
    );

    checkCudaErrors(cudaGetLastError());
}

void matrix_mul_tile(
    int     N,
    float * aMatrixDevice,
    float * bMatrixDevice,
    float * cMatrixDeviceTile
) {

    dim3 numBlocks(TILEDIM,TILEDIM);
    dim3 gridSize((N + numBlocks.x - 1)/numBlocks.x,
                  (N + numBlocks.y - 1)/numBlocks.y);

    matrix_mul_tile_kernel <<< gridSize, numBlocks >>>(
        N,
        aMatrixDevice,
        bMatrixDevice,
        cMatrixDeviceTile
    );

    checkCudaErrors(cudaGetLastError());
}

void matrix_mul_tile_coarse(
    int     N,
    float * aMatrixDevice,
    float * bMatrixDevice,
    float * cMatrixDeviceTile
) {

    unsigned int COARSE_FACTOR = 2;

    //while(((N + TILEDIM - 1)/TILEDIM/COARSE_FACTOR * (N+TILEDIM-1)/TILEDIM) > 1024) COARSE_FACTOR *= 2;

    dim3 numBlocks((N + TILEDIM - 1)/TILEDIM/COARSE_FACTOR, (N+TILEDIM-1)/TILEDIM);
   // dim3 numBlocks(32,16);
    dim3 gridSize((N + numBlocks.x - 1)/numBlocks.x,
                  (N + numBlocks.y - 1)/numBlocks.y);

    matrix_mul_tile_coarse_kernel <<< gridSize, numBlocks >>>(
        aMatrixDevice,
        bMatrixDevice,
        cMatrixDeviceTile,
        N,
        COARSE_FACTOR
    );

    checkCudaErrors(cudaGetLastError());
}