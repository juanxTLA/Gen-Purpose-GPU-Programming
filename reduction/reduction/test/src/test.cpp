#include <Timer.hpp>
#include <reduction.h>

#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <time.h>
#include <vector>


#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace std;

template <typename T>
void reduce(T *vector, T *res, int exercise, int n, Timer &t);

template <typename T>
int verifyHost(char v, T* vector, int n, Timer &t);

int main(int argc, char** argv){

    int n = atoi(argv[1]); //get number of elements in the vector
    
    int exercise = atoi(argv[2]);

    srand(1000); //seed for the random number

    ofstream outFile;
    outFile.open("data.txt", ios::app);

    int * vector = (int*)malloc(n * sizeof(int)); //vector
    int * res    = new int(0); 

    Timer t;

    for(int i = 0; i < n; i++){
        //vector[i] = rand() % 10;
        vector[i] = 1;
    }

    if(exercise < 6){

        reduce(vector, res, exercise, n, t);
    
        cout << "Ex " << exercise << "  : " << *res  << " | Time (ms): "<< t.elapsedTime_ms() << endl;
    }
    //run against thrust and cub
    else if(exercise == 7){
        t.start();
        int thrustSum = verifyHost('t', vector, n, t);
        t.stop();
        cout << "Thrust: " << thrustSum << " | Time (ms): "<< t.elapsedTime_ms() << endl;
    }

    else{    
        
        int cubSum = verifyHost('c', vector, n, t);
        
        cout << "CUB   : " << cubSum << " | Time (ms): " << t.elapsedTime_ms() << endl;
    }

    outFile << exercise << "," << n << "," << t.elapsedTime_ms() << endl;
    
    free(vector);
    free(res);

    outFile.close();

    return 0;
}

template <typename T>
void reduce(T *vector, T *res, int exercise, int n, Timer &t){

    if(exercise == 0){
        for(int i = 0; i < n; i++){
            *res += vector[i];
        }
    }

    else{

        size_t vectSize = n * sizeof(T);
        size_t resSize = sizeof(T);

        T* vectDevice = nullptr;
        T* resDevice = nullptr;

        checkCudaErrors(
            cudaMalloc(&vectDevice, vectSize)
        );

        checkCudaErrors(
            cudaMalloc(&resDevice, resSize)
        );


        checkCudaErrors(
            cudaMemcpy(
                vectDevice,
                vector,
                vectSize,
                cudaMemcpyHostToDevice
            )
        );

        switch(exercise){

            case 1:

                t.start();
                singleCudaThread(vectDevice, resDevice, n);
                t.stop();

                checkCudaErrors(
                    cudaMemcpy(
                        res,
                        resDevice,
                        resSize,
                        cudaMemcpyDeviceToHost
                    )
                );

                break;
            
            case 2:
                t.start();
                atomicVar(vectDevice, resDevice, n);
                t.stop();

                checkCudaErrors(
                    cudaMemcpy(
                        res,
                        resDevice,
                        resSize,
                        cudaMemcpyDeviceToHost
                    )
                );

                break;

            case 3:
                t.start();
                segmented(vectDevice, resDevice, n);
                t.stop();

                checkCudaErrors(
                    cudaMemcpy(
                        res,
                        resDevice,
                        resSize,
                        cudaMemcpyDeviceToHost
                    )
                );

                break;

            case 4:
                t.start();
                coalesced(vectDevice, resDevice, n);
                t.stop();

                checkCudaErrors(
                    cudaMemcpy(
                        res,
                        resDevice,
                        resSize,
                        cudaMemcpyDeviceToHost
                    )
                );

                break;

            case 5:
                t.start();
                sharedSegmented(vectDevice, resDevice, n);
                t.stop();

                checkCudaErrors(
                    cudaMemcpy(
                        res,
                        resDevice,
                        resSize,
                        cudaMemcpyDeviceToHost
                    )
                );

                break;

            case 6:
                t.start();
                sharedSegmentedCoarsened(vectDevice, resDevice, n);
                t.stop();

                checkCudaErrors(
                    cudaMemcpy(
                        res,
                        resDevice,
                        resSize,
                        cudaMemcpyDeviceToHost
                    )
                );

                break;

            checkCudaErrors(cudaFree(resDevice));
            checkCudaErrors(cudaFree(vectDevice));
        }
    }
}

template <typename T>
int verifyHost(char v, T* vector, int n, Timer &t){
    if(v == 't'){

        int *res = new int();
        t.start();
        verify(v, vector, res, n);
        t.stop();
        return *res; 
    }

    else if(v == 'c'){

        size_t vectSize = n * sizeof(T);
        size_t resSize = sizeof(T);

        T* vectDevice = nullptr;
        T* resDevice = nullptr;

        checkCudaErrors(
            cudaMalloc(&vectDevice, vectSize)
        );

        checkCudaErrors(
            cudaMalloc(&resDevice, resSize)
        );

        checkCudaErrors(
            cudaMemcpy(
                vectDevice,
                vector,
                vectSize,
                cudaMemcpyHostToDevice
            )
        );

        t.start();
        verify(v, vectDevice, resDevice, n);
        t.stop();

        T *res = (int*)malloc(sizeof(int));

        checkCudaErrors(
            cudaMemcpy(
                res,
                resDevice,
                resSize,
                cudaMemcpyDeviceToHost
            )
        );

        return *res;
        
        checkCudaErrors(cudaFree(resDevice));
        checkCudaErrors(cudaFree(vectDevice));
    }

    return -1;
}