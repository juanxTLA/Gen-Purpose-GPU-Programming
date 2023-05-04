#include <convolution.h>
#include <helper_cuda.h>

__constant__ float constFilt[FILTDIM*FILTDIM];

//EXERCISE 1
__global__ void convolution_naive_kernel(
    int         width,
    int         height,
    int         filtRad,
    float   *   grayImageDevice,
    float   *   filtImageDevice,
    float   *   filterDevice
) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    float pVal = 0.0f;
    if(outCol < width && outRow < height){
        for(int r = 0; r < 2 * filtRad + 1; r++){
            for(int c = 0; c < 2 * filtRad + 1; c++){
                int inRow = outRow - filtRad + r; 
                int inCol = outCol - filtRad + c;

                if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width){
                    pVal += filterDevice[r * filtRad + c] * grayImageDevice[inRow * width + inCol];
                }
            }
        }

        filtImageDevice[outRow * width + outCol] = pVal;
    }
}

void convolutionNaive(
    int     width,
    int     height,
    int     filtRad,
    float * grayImageDevice,
    float * filtImageDevice,
    float * filtGaussDevice
) {

    dim3 numBlocks(32,32);
    dim3 gridSize((width + numBlocks.x - 1)/numBlocks.x,
                  (height + numBlocks.y - 1)/numBlocks.y);

    convolution_naive_kernel <<< gridSize, numBlocks >>>(
        width,
        height,
        filtRad,
        grayImageDevice,
        filtImageDevice,
        filtGaussDevice
    );

    checkCudaErrors(cudaGetLastError());
}

//EXERCISE 2

__global__ void convolution_shared_kernel(
    int     width,
    int     height,
    int     filtRad,
    float * grayImageDevice,
    float * filtImageDevice,
    float * filterDevice
) {
    
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float filter[FILTDIM*FILTDIM];

    float pVal = 0.0f;

    if(outCol < width && outRow < height){
        //load filter
        for(int r = 0; r < 2*filtRad + 1; r++){
            for(int c = 0; c < 2*filtRad + 1; c++){
                filter[r * filtRad*filtRad + c] = filterDevice[r * filtRad*filtRad + c];
            }
        }

        __syncthreads();

        for(int r = 0; r < 2 * filtRad + 1; r++){
            for(int c = 0; c < 2 * filtRad + 1; c++){
                int inRow = outRow - filtRad + r; 
                int inCol = outCol - filtRad + c;

                if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width){
                    pVal += filter[r * filtRad + c] * grayImageDevice[inRow * width + inCol];
                }
            }
        }

        filtImageDevice[outRow * width + outCol] = pVal;
    }
}


void convolutionShared(
    int     width,
    int     height,
    int     filtRad,
    float * grayImageDevice,
    float * filtImageDevice,
    float * filtGaussDevice
) {

    dim3 numBlocks(32,32);
    dim3 gridSize((width + numBlocks.x - 1)/numBlocks.x,
                  (height + numBlocks.y - 1)/numBlocks.y);

    convolution_shared_kernel <<< gridSize, numBlocks >>>(
        width,
        height,
        filtRad,
        grayImageDevice,
        filtImageDevice,
        filtGaussDevice
    );

    checkCudaErrors(cudaGetLastError());
}

//EXERCISE 3
__global__ void convolution_const_kernel(
    int         width,
    int         height,
    int         filtRad,
    float   *   grayImageDevice,
    float   *   filtImageDevice
) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    float pVal = 0.0f;
    if(outCol < width && outRow < height){
        for(int r = 0; r < 2 * filtRad + 1; r++){
            for(int c = 0; c < 2 * filtRad + 1; c++){
                int inRow = outRow - filtRad + r; 
                int inCol = outCol - filtRad + c;

                if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width){
                    pVal += constFilt[r * filtRad + c] * grayImageDevice[inRow * width + inCol];
                }
            }
        }

        filtImageDevice[outRow * width + outCol] = pVal;
    }
}

void convolutionConst(
    int     width,
    int     height,
    int     filtRad,
    float * grayImageDevice,
    float * filtImageDevice
) {

    dim3 numBlocks(32,32);
    dim3 gridSize((width + numBlocks.x - 1)/numBlocks.x,
                  (height + numBlocks.y - 1)/numBlocks.y);

    convolution_const_kernel <<< gridSize, numBlocks >>>(
        width,
        height,
        filtRad,
        grayImageDevice,
        filtImageDevice
    );

    checkCudaErrors(cudaGetLastError());
}

//EXERCISE 4
__global__ void convolution_shared_tile_kernel(
    int     width,
    int     height,
    int     filtRad,
    float * grayImageDevice,
    float * filtImageDevice,
    float * filterDevice
) {
    
    int outCol = blockIdx.x * OUTDIM + threadIdx.x - FILTRAD;
    int outRow = blockIdx.y * OUTDIM + threadIdx.y - FILTRAD;

    __shared__ float filter[FILTDIM*FILTDIM];
    __shared__ float imTile[INTILEDIM][INTILEDIM];


    if(outCol < width && outCol >= 0 && outRow >= 0 && outRow < height){
        imTile[threadIdx.y][threadIdx.x] = grayImageDevice[outRow * width + outCol];
    }
    else imTile[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    if(outCol < width && outCol >= 0 && outRow >= 0 && outRow < height){
        //load filter
        for(int r = 0; r < 2*filtRad + 1; r++){
            for(int c = 0; c < 2*filtRad + 1; c++){
                filter[r * filtRad*filtRad + c] = filterDevice[r * filtRad*filtRad + c];
            }
        }
        __syncthreads();

        int tileCol = threadIdx.x - FILTRAD;
        int tileRow = threadIdx.y - FILTRAD;

        float pVal = 0.0f;
        if(tileCol >= 0 && tileCol < OUTDIM && tileRow >= 0 && tileRow < OUTDIM){

            for(int r = 0; r < 2 * filtRad + 1; r++){
                for(int c = 0; c < 2 * filtRad + 1; c++){
                    
                    pVal += filter[r * filtRad + c] * imTile[tileRow + r][tileCol + c];
                    
                }
            }
            filtImageDevice[outRow * width + outCol] = pVal;
        }
    }
}


void convolutionSharedTile(
    int     width,
    int     height,
    int     filtRad,
    float * grayImageDevice,
    float * filtImageDevice,
    float * filtGaussDevice
) {

    dim3 numBlocks(16,16);
    dim3 gridSize((width + numBlocks.x - 1)/numBlocks.x,
                  (height + numBlocks.y - 1)/numBlocks.y);

    convolution_shared_tile_kernel <<< gridSize, numBlocks >>>(
        width,
        height,
        filtRad,
        grayImageDevice,
        filtImageDevice,
        filtGaussDevice
    );

    checkCudaErrors(cudaGetLastError());
}

//EXERCISE 5
__global__ void convolution_shared_tile_const_kernel(
    int     width,
    int     height,
    int     filtRad,
    float * grayImageDevice,
    float * filtImageDevice
) {
    
    int outCol = blockIdx.x * OUTDIM + threadIdx.x - FILTRAD;
    int outRow = blockIdx.y * OUTDIM + threadIdx.y - FILTRAD;

    __shared__ float imTile[INTILEDIM][INTILEDIM];

    if(outCol < width && outCol >= 0 && outRow >= 0 && outRow < height){
        imTile[threadIdx.y][threadIdx.x] = grayImageDevice[outRow * width + outCol];
    }
    else imTile[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    if(outCol < width && outCol >= 0 && outRow >= 0 && outRow < height){

        int tileCol = threadIdx.x - FILTRAD;
        int tileRow = threadIdx.y - FILTRAD;

        float pVal = 0.0f;
        if(tileCol >= 0 && tileCol < OUTDIM && tileRow >= 0 && tileRow < OUTDIM){

            for(int r = 0; r < 2 * filtRad + 1; r++){
                for(int c = 0; c < 2 * filtRad + 1; c++){
                    
                    pVal += constFilt[r * filtRad + c] * imTile[tileRow + r][tileCol + c];
                    
                }
            }
            filtImageDevice[outRow * width + outCol] = pVal;
        }
    }
}


void convolutionSharedTileConst(
    int     width,
    int     height,
    int     filtRad,
    float * grayImageDevice,
    float * filtImageDevice
) {

    dim3 numBlocks(16,16);
    dim3 gridSize((width + numBlocks.x - 1)/numBlocks.x,
                  (height + numBlocks.y - 1)/numBlocks.y);

    convolution_shared_tile_const_kernel <<< gridSize, numBlocks >>>(
        width,
        height,
        filtRad,
        grayImageDevice,
        filtImageDevice
    );

    checkCudaErrors(cudaGetLastError());
}


//EXERCISE 6
__global__ void convolution_same_tile_const_kernel(
    int     width,
    int     height,
    int     filtRad,
    float * grayImageDevice,
    float * filtImageDevice
) {
    
    int outCol = blockIdx.x * INTILEDIM + threadIdx.x;
    int outRow = blockIdx.y * INTILEDIM + threadIdx.y;

    __shared__ float imTile[INTILEDIM][INTILEDIM];

    if(outCol < width && outCol >= 0 && outRow >= 0 && outRow < height){
        imTile[threadIdx.y][threadIdx.x] = grayImageDevice[outRow * width + outCol];
    }
    else imTile[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    if(outCol < width && outRow < height){
        float pVal = 0.0f;

        int tileCol = threadIdx.x - FILTRAD;
        int tileRow = threadIdx.y - FILTRAD;

        for(int r = 0; r < 2 * filtRad + 1; r++){
            for(int c = 0; c < 2 * filtRad + 1; c++){
                if(tileCol + c >= 0 && tileCol + c < INTILEDIM &&
                    tileRow + r >= 0 && tileRow + r < INTILEDIM){

                    pVal += constFilt[r * filtRad + c] * imTile[tileRow + r][tileCol + c];
                }

                else{
                    if(outRow - FILTRAD + r >= 0 &&
                        outRow + FILTRAD + r < height &&
                        outCol - FILTRAD + c >= 0 &&
                        outCol + FILTRAD + c < width){
                            pVal += constFilt[r * filtRad + c] * grayImageDevice[(outRow - FILTRAD + r)*width + outCol - FILTRAD + c];
                        }
                    
                }
                
            }
        }

        filtImageDevice[outRow * width + outCol] = pVal;
        
    }
}


void convolutionSameTileConst(
    int     width,
    int     height,
    int     filtRad,
    float * grayImageDevice,
    float * filtImageDevice
) {

    dim3 numBlocks(16,16);
    dim3 gridSize((width + numBlocks.x - 1)/numBlocks.x,
                  (height + numBlocks.y - 1)/numBlocks.y);

    convolution_same_tile_const_kernel <<< gridSize, numBlocks >>>(
        width,
        height,
        filtRad,
        grayImageDevice,
        filtImageDevice
    );

    checkCudaErrors(cudaGetLastError());
}
