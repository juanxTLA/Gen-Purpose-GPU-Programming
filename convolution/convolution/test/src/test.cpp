#include <Timer.hpp>
#include <convolution.h>

#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>

#include <cuda_runtime.h>
#include <helper_cuda.h>

__constant__ float constFilt[FILTRAD*FILTRAD];

using namespace std;

void fileOut(string filename, float* filtImageHost, int width, int height);

int main(int argc, char** argv){

    ifstream inFile;
    inFile.open("grayscale.txt");
    
    int height, width;
    inFile >> width >> height;

    float   *grayImageHost  = (float*)malloc(width * height * sizeof(float));
    float   *filtImageHost  = (float*)malloc(width * height * sizeof(float));

    //get image values into arrays
    for(int r = 0; r < height; r++){
        for(int c = 0; c < width; c++){
            inFile >> grayImageHost[r*width + c];
        }
    }

    //gaussian filter
    int k = 2; //radius
    int filtDim = 2 * k + 1;
    float std = 1.0f;
    float *filtGaussHost = (float*)malloc(filtDim * filtDim * sizeof(float));
    float gaussSum = 0.0f;

    for(int r = 0; r < filtDim; r++){
        for(int c = 0; c < filtDim; c++){
            filtGaussHost[r * filtDim + c] = exp(-(pow(r-k,2) + pow(c-k,2))/(2.0*std*std))/(2.0*PI*std*std);
            gaussSum += filtGaussHost[r * filtDim + c];
            #ifdef DEBUG
                cout << filtGaussHost[r * filtDim + c] << " ";
            #endif
        }   

        #ifdef DEBUG
            cout << endl;
        #endif
    }

    //normalize
    for(int i = 0; i < filtDim * filtDim; i++){
        filtGaussHost[i] /= gaussSum;
    }

    //DEVICE RESOURCE ALLOCATION AND DATA COPY    

    size_t imageSize = width * height * sizeof(float);
    size_t filtSize = filtDim * filtDim * sizeof(float);

    float *grayImageDevice = nullptr;
    float *filtImageDevice = nullptr;
    float *filtGaussDevice = nullptr;

    checkCudaErrors(
        cudaMalloc((void**)&grayImageDevice, imageSize)
    );

    checkCudaErrors(
        cudaMalloc((void**)&filtImageDevice, imageSize)
    );

    checkCudaErrors(
        cudaMalloc((void**)&filtGaussDevice, filtSize)
    );

    checkCudaErrors(
        cudaMemcpy(
            grayImageDevice, 
            grayImageHost, 
            imageSize,
            cudaMemcpyHostToDevice
        )
    );  

    checkCudaErrors(
        cudaMemcpy(
            filtGaussDevice, 
            filtGaussHost, 
            filtSize,
            cudaMemcpyHostToDevice
        )
    );

    //performance analysis
    Timer timer;
    double flops, flopRate;
    double accesses, bandwidthBps;


    //EXERCISE 1
    timer.start();
    convolutionNaive(
        width,
        height,
        k,
        grayImageDevice,
        filtImageDevice,
        filtGaussDevice
    );
    timer.stop();

    flops = filtDim * filtDim * 2;
    accesses = filtDim * filtDim * sizeof(float);

    flopRate = flops/(timer.elapsedTime_ms() / 1.0e3);
    bandwidthBps = accesses/(timer.elapsedTime_ms() / 1.0e3);

    checkCudaErrors(
        cudaMemcpy(
            filtImageHost,
            filtImageDevice,
            imageSize,
            cudaMemcpyDeviceToHost
        )
    );

    fileOut("Exercise1.txt", filtImageHost, width, height);
    cout << "1 " << timer.elapsedTime_ms() << " " << flopRate << " " << bandwidthBps << endl;

    //EXERCISE 2
    timer.start();
    convolutionShared(
        width,
        height,
        k,
        grayImageDevice,
        filtImageDevice,
        filtGaussDevice
    );
    timer.stop();

    flops = filtDim * filtDim * 2;
    accesses = filtDim * filtDim * sizeof(float);

    flopRate = flops/(timer.elapsedTime_ms() / 1.0e3);
    bandwidthBps = accesses/(timer.elapsedTime_ms() / 1.0e3);

    checkCudaErrors(
        cudaMemcpy(
            filtImageHost,
            filtImageDevice,
            imageSize,
            cudaMemcpyDeviceToHost
        )
    );
    

    fileOut("Exercise2.txt", filtImageHost, width, height);
    cout << "2 " << timer.elapsedTime_ms() << " " << flopRate << " " << bandwidthBps << endl;

    //EXERCISE 4
    timer.start();
    convolutionSharedTile(
        width,
        height,
        k,
        grayImageDevice,
        filtImageDevice,
        filtGaussDevice
    );
    timer.stop();

    flops = OUTDIM * OUTDIM * filtDim * filtDim * 2;
    accesses = INTILEDIM * INTILEDIM * sizeof(float);

    flopRate = flops/(timer.elapsedTime_ms() / 1.0e3);
    bandwidthBps = accesses/(timer.elapsedTime_ms() / 1.0e3);

    checkCudaErrors(
        cudaMemcpy(
            filtImageHost,
            filtImageDevice,
            imageSize,
            cudaMemcpyDeviceToHost
        )
    );

    fileOut("Exercise4.txt", filtImageHost, width, height);
    cout << "4 " << timer.elapsedTime_ms() << " " << flopRate << " " << bandwidthBps << endl;

    //EXERCISE 3
    checkCudaErrors(
        cudaMemcpyToSymbol(constFilt, filtGaussHost, filtSize)
    );

    timer.start();
    convolutionConst(
        width,
        height,
        k,
        grayImageDevice,
        filtImageDevice
    );
    timer.stop();

    flops = filtDim * filtDim * 2;
    accesses = filtDim * filtDim * sizeof(float);

    flopRate = flops/(timer.elapsedTime_ms() / 1.0e3);
    bandwidthBps = accesses/(timer.elapsedTime_ms() / 1.0e3);

    flopRate = flops/(timer.elapsedTime_ms() / 1.0e3);
    bandwidthBps = accesses/(timer.elapsedTime_ms() / 1.0e3);

    checkCudaErrors(
        cudaMemcpy(
            filtImageHost,
            filtImageDevice,
            imageSize,
            cudaMemcpyDeviceToHost
        )
    );

    fileOut("Exercise3.txt", filtImageHost, width, height);
    cout << "3 " << timer.elapsedTime_ms() << " " << flopRate << " " << bandwidthBps << endl;

    //EXERCISE 5
    timer.start();
    convolutionSharedTileConst(
        width,
        height,
        k,
        grayImageDevice,
        filtImageDevice
    );
    timer.stop();

    flops = OUTDIM * OUTDIM * filtDim * filtDim * 2;
    accesses = INTILEDIM * INTILEDIM * sizeof(float);

    flopRate = flops/(timer.elapsedTime_ms() / 1.0e3);
    bandwidthBps = accesses/(timer.elapsedTime_ms() / 1.0e3);

    checkCudaErrors(
        cudaMemcpy(
            filtImageHost,
            filtImageDevice,
            imageSize,
            cudaMemcpyDeviceToHost
        )
    );

    fileOut("Exercise5.txt", filtImageHost, width, height);
    cout << "5 " << timer.elapsedTime_ms() << " " << flopRate << " " << bandwidthBps << endl;

    //EXERCISE 6
    timer.start();
    convolutionSameTileConst(
        width,
        height,
        k,
        grayImageDevice,
        filtImageDevice
    );
    timer.stop();

    flops = OUTDIM * OUTDIM * filtDim * filtDim * 2;
    accesses = INTILEDIM * INTILEDIM * sizeof(float);

    flopRate = flops/(timer.elapsedTime_ms() / 1.0e3);
    bandwidthBps = accesses/(timer.elapsedTime_ms() / 1.0e3);

    checkCudaErrors(
        cudaMemcpy(
            filtImageHost,
            filtImageDevice,
            imageSize,
            cudaMemcpyDeviceToHost
        )
    );

    fileOut("Exercise6.txt", filtImageHost, width, height);
    cout << "6 " << timer.elapsedTime_ms() << " " << flopRate << " " << bandwidthBps << endl;

    
    //FREE MEMORY
    checkCudaErrors(cudaFree(grayImageDevice));
    checkCudaErrors(cudaFree(filtImageDevice));
    checkCudaErrors(cudaFree(filtGaussDevice));

    return 0;
}

void fileOut(string filename, float* filtImageHost, int width, int height){

    int *filtImageInt = (int*)malloc(sizeof(int) * width * height);
    ofstream outFile;
        
    for(int i = 0; i < height * width; i++){
        filtImageInt[i] = (int)ceil(filtImageHost[i]);
    }

    outFile.open(filename);

    for(int r = 0; r < height; r++){
        for(int c = 0; c < width; c++){
            outFile << filtImageInt[r * width + c] << " ";
        }
        outFile << endl;
    }

    delete filtImageInt;

}