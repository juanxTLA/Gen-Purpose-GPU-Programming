#include <Timer.hpp>
#include <matrix_mul.h>

#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>

#include <cuda_runtime.h>
#include <helper_cuda.h>


using namespace std;

extern "C" void sgemm_(
    unsigned char   *   transA,
    unsigned char   *   transB,
    int             *   m,
    int             *   n,
    int             *   k,
    float           *   alpha,
    float           *   a,
    int             *   lda,
    float           *   b,
    int             *   ldb,
    float           *   beta,
    float           *   c,
    int             *   ldc
);

float cublasRef(
    float* a, 
    float* b,
    float* c,
    int N
);

double relative_error_l2 (
  int n,
  float  * y_reference,
  float  * y_computed
);

int main(int argc, char** argv){

    //since I am testing manually, I am not going to check for correct input
    int N = atoi(argv[1]);

    #ifdef DEBUG
        N = 64;
    #endif

    #ifdef FILEDATA
        ofstream outFile;
        outFile.open("data_out.txt", ios::app);
    #endif

    float *aMatrixHost;
    float *bMatrixHost;
    float *cMatrixHost;
    float *cMatrixHostTile;
    float *cMatrixHostCrse;
    float *cMatrixRef;
    float *cMatrixRefCublas;

    srand(time(NULL));

    aMatrixHost     = (float*)malloc(sizeof(float) * N*N);
    bMatrixHost     = (float*)malloc(sizeof(float) * N*N);
    cMatrixHost     = (float*)malloc(sizeof(float) * N*N);
    cMatrixHostTile = (float*)malloc(sizeof(float) * N*N);
    cMatrixHostCrse = (float*)malloc(sizeof(float) * N*N);
    cMatrixRef      = (float*)malloc(sizeof(float) * N*N);
    cMatrixRefCublas= (float*)malloc(sizeof(float) * N*N);
    

    //initialize matrices with random values between 0 and 1
    for(int r = 0; r < N; r++){
        for(int c = 0; c < N; c++){
            aMatrixHost[r*N + c] = (float)rand()/RAND_MAX;
            bMatrixHost[r*N + c] = (float)rand()/RAND_MAX;
            
            // #ifdef DEBUG
                // aMatrixHost[r*N + c] = 1.0f + r*N + c;
                // bMatrixHost[r*N + c] = r*N + c - 1.0f;
            // #endif
        }
    }

    //get size of stream we want to store in device
    size_t matrixByteSize = N*N * sizeof(float);

    float *aMatrixDevice = nullptr;
    float *bMatrixDevice = nullptr;
    float *cMatrixDevice = nullptr;
    float *cMatrixDeviceTile = nullptr;
    float *cMatrixDeviceCrse = nullptr;
    float *cMatrixDeviceCublas = nullptr;

    // TODO: allocate resources in device and call kernel
    checkCudaErrors(
        cudaMalloc((void**)&aMatrixDevice, matrixByteSize)
    );    
    
    checkCudaErrors(
        cudaMalloc((void**)&bMatrixDevice, matrixByteSize)
    );    
    
    checkCudaErrors(
        cudaMalloc((void**)&cMatrixDevice, matrixByteSize)
    );    

    checkCudaErrors(
        cudaMalloc((void**)&cMatrixDeviceTile, matrixByteSize)
    );
    
    checkCudaErrors(
        cudaMalloc((void**)&cMatrixDeviceCublas, matrixByteSize)
    );   

    checkCudaErrors(
        cudaMalloc((void**)&cMatrixDeviceCrse, matrixByteSize)
    );
    
    checkCudaErrors(
        cudaMemcpy(
            aMatrixDevice,
            aMatrixHost,
            matrixByteSize,
            cudaMemcpyHostToDevice
        )
    );

    checkCudaErrors(
        cudaMemcpy(
            bMatrixDevice,
            bMatrixHost,
            matrixByteSize,
            cudaMemcpyHostToDevice
        )
    );

    float timeMul, timeTile, timeCoarse;
    
    // TILE 

    Timer *timer = new Timer();
    
    timer->start();
    matrix_mul_tile(
        N,
        aMatrixDevice,
        bMatrixDevice,
        cMatrixDeviceTile
    );
    timer->stop();

    timeTile = timer->elapsedTime_ms();
    delete timer;

    // NAIVE

    timer = new Timer();

    timer->start();
    matrix_mul(
        N,
        aMatrixDevice,
        bMatrixDevice,
        cMatrixDevice
    );
    timer->stop();

    timeMul = timer->elapsedTime_ms();

    delete timer;

    //COARSE
    #ifdef COARSE
        timer = new Timer();

        timer->start();
        matrix_mul_tile_coarse(
            N,
            aMatrixDevice,
            bMatrixDevice,
            cMatrixDeviceCrse
        );
        timer->stop();

        timeCoarse = timer->elapsedTime_ms();
        delete timer;
    #endif
    //Retrieve results in host

    checkCudaErrors(
        cudaMemcpy(
            cMatrixHostTile,
            cMatrixDeviceTile,
            matrixByteSize,
            cudaMemcpyDeviceToHost
        )
    );

    checkCudaErrors(
        cudaMemcpy(
            cMatrixHost,
            cMatrixDevice,
            matrixByteSize,
            cudaMemcpyDeviceToHost
        )
    );

    checkCudaErrors(
        cudaMemcpy(
            cMatrixHostCrse,
            cMatrixDeviceCrse,
            matrixByteSize,
            cudaMemcpyDeviceToHost
        )
    );

    //BLAS SECTION

    unsigned char transA = 'n'; // we will not use A**T or B**T
    unsigned char transB = 'n'; // we will not use A**T or B**T

    float alpha = 1.0f;
    float beta = 0.0f; //no need for scalar in C
    
    //CUBLAS SECTION
    
    #ifdef CUBLAS
        float timeCublas = cublasRef(
            bMatrixDevice,
            aMatrixDevice,
            cMatrixDeviceCublas,
            N    
        );

        cublasHandle_t handle;
        cublasCreate(&handle);

        checkCudaErrors(
            cudaMemcpy(
                cMatrixRefCublas,
                cMatrixDeviceCublas,
                matrixByteSize,
                cudaMemcpyDeviceToHost
            )
        );

        double cuBlasFlopRate = N * N / 1.0e9 * N * 2 / (timeCublas / 1.0e3);
        double numberOfAccessesCublas = 3 * N * N;
        double effectiveBandwidthCublas {
            (numberOfAccessesCublas) * sizeof(float) * 8 / 
            (timeCublas / 1.0e3)
        };
        double arithmeticIntCublas = cuBlasFlopRate/effectiveBandwidthCublas/8*1.0e09;

        double relerr = relative_error_l2 (
            N*N,
            cMatrixRefCublas,
            cMatrixHost
        );

        double relerr_tile = relative_error_l2 (
            N*N,
            cMatrixRefCublas,
            cMatrixHostTile
        );

        #ifdef FILEDATA
            outFile << N << " " << "R "<< cuBlasFlopRate << " " << effectiveBandwidthCublas/1.0e9 
                << " " << timeCublas << " " << arithmeticIntCublas  << endl;
        #endif

        printf("\n==== CUBLAS ====\n");
        printf (
            "\t- Computational Rate:         %20.16e Gflops\n",
            cuBlasFlopRate 
        );
        printf (
            "\t- Effective Bandwidth:        %20.16e Gbps\n",
            effectiveBandwidthCublas / 1e9 
        );

    #else
        sgemm_(
            &transA,
            &transB,
            &N,
            &N,
            &N,
            &alpha,
            bMatrixHost,
            &N,
            aMatrixHost,
            &N,
            &beta,
            cMatrixRef,
            &N
        );

        double relerr = relative_error_l2 (
            N*N,
            cMatrixCublasRef,
            cMatrixHost
        );

        double relerr_tile = relative_error_l2 (
            N*N,
            cMatrixCublasRef,
            cMatrixHostTile
        );

    #endif


    
    #ifdef DEBUG
        cout << "NOT TILED" << endl;
        for(int r = 0; r < N; ++r){
            for(int c = 0; c < N; ++c){
                cout << cMatrixHost[r * N + c] << " ";
            }
            cout << endl;
        }
        cout << "TILED" << endl;
        for(int r = 0; r < N; ++r){
            for(int c = 0; c < N; ++c){
                cout << cMatrixHostTile[r * N + c] << " ";
            }
            cout << endl;
        }

        cout << "REF" << endl;
        for(int r = 0; r < N; ++r){
            for(int c = 0; c < N; ++c){
                cout << cMatrixRefCublas[r * N + c] << " ";
            }
            cout << endl;
        }
    #else

        // output relative error
        printf (
            "\t- Relative Error (l2):        %20.16e\n",
            relerr
        );

        printf (
            "\t- Relative Error (tiled):        %20.16e\n",
            relerr_tile
        );

        if (relerr < 1.0e-5) {

            printf("\t- NAÏVE PASSED\n");
        } 
        else {

            printf("\t-NAÏVE FAILED\n");
        }

        if (relerr_tile < 1.0e-5) {

            printf("\t- TILED PASSED\n");
        } 
        else {

            printf("\t- TILED FAILED\n");
        }

        // get elapsed time, estimated flops per second, and effective bandwidth
        double numberOfFlops = N * N / 1.0e9 * N * 2;
        double flopRate = numberOfFlops / (timeMul / 1.0e3);
        double numberOfAccesses = 3 * N * N;
        double effectiveBandwidth_bitspersec {
            (numberOfAccesses) * sizeof(float) * 8 / 
            (timeMul / 1.0e3)
        }; 

        double arithmeticIntesity = flopRate/effectiveBandwidth_bitspersec/8*1.0e09;
        
        printf("\n==== NAÏVE ====\n");
        printf (
            "\t- Computational Rate:         %20.16e Gflops\n",
            flopRate 
        );
        printf (
            "\t- Effective Bandwidth:        %20.16e Gbps\n",
            effectiveBandwidth_bitspersec / 1e9 
        );

        #ifdef FILEDATA
            outFile << N << " " << "N "<< flopRate << " " << effectiveBandwidth_bitspersec/1.0e9 << " " << timeMul << " " << arithmeticIntesity << endl;
        #endif

        numberOfFlops = N * N / 1.0e9 * N * 2;
        flopRate = numberOfFlops / (timeTile / 1.0e3);
        numberOfAccesses = 3 * N * N / TILEDIM;
        effectiveBandwidth_bitspersec =
            (numberOfAccesses) * sizeof(float) * 8 / 
            (timeTile / 1.0e3); 
        arithmeticIntesity = flopRate/effectiveBandwidth_bitspersec/8*1.0e09;

        printf("\n==== TILED ====\n");
        printf (
            "\t- Computational Rate:         %20.16e Gflops\n",
            flopRate
        );
        printf (
            "\t- Effective Bandwidth:        %20.16e Gbps\n",
            effectiveBandwidth_bitspersec / 1e9 
        );
        

        #ifdef FILEDATA
            outFile << N << " " << "T "<< flopRate << " " << effectiveBandwidth_bitspersec/1.0e9 << " " << timeTile << " " << arithmeticIntesity << endl;
        #endif

        #ifdef COARSE
            numberOfFlops = N * N / 1.0e9 * N * 2;
            flopRate = numberOfFlops / (timeCoarse / 1.0e3);
            numberOfAccesses = 3 * N * N / TILEDIM;/// COARSE_FACTOR;
            effectiveBandwidth_bitspersec =
                (numberOfAccesses) * sizeof(float) * 8 / 
                (timeTile / 1.0e3); 
            arithmeticIntesity = flopRate/effectiveBandwidth_bitspersec/8*1.0e09;

            #ifdef FILEDATA
                outFile << N << " " << "C "<< flopRate << " " << effectiveBandwidth_bitspersec/1.0e9 << " " << timeCoarse << " " << arithmeticIntesity << endl;
            #endif
        #endif

    #endif

    checkCudaErrors(
        cudaFree(aMatrixDevice)
    );

    checkCudaErrors(
        cudaFree(bMatrixDevice)
    );

    checkCudaErrors(
        cudaFree(cMatrixDevice)
    );

    checkCudaErrors(
        cudaFree(cMatrixDeviceTile)
    );

    checkCudaErrors(
        cudaFree(cMatrixDeviceCublas)
    );    

    checkCudaErrors(
        cudaFree(cMatrixDeviceCrse)
    );

    delete aMatrixHost;
    delete bMatrixHost;
    delete cMatrixHost;
    delete cMatrixHostTile;
    delete cMatrixRef;
    delete cMatrixRefCublas;
    delete cMatrixHostCrse;

    return 0;
}

double relative_error_l2 (
  int n,
  float  * y_reference,
  float * y_computed
) {

  double difference_norm_squared = 0.0;
  double reference_norm_squared = 0.0;
  for (int idx = 0; idx < n; ++idx) {
    auto & reference_value = y_reference[idx];
    double difference {
      y_reference[idx] - 
      y_computed[idx]
    };
    difference_norm_squared += difference * difference;
    reference_norm_squared += reference_value * reference_value;
  }

  return sqrt (
    difference_norm_squared / reference_norm_squared
  );
}

float cublasRef(
    float* a, 
    float* b,
    float* c,
    int N
) {
    float alpha = 1.0f;
    float beta = 0.0f; //no need for scalar in C

    cublasHandle_t handle;
    cublasCreate(&handle);

    Timer *timer = new Timer();
    timer->start();
    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        N,
        N,
        &alpha,
        a,
        N,
        b,
        N,
        &beta,
        c,
        N
    );
    timer->stop();

    float timeCublas = timer->elapsedTime_ms();
    
    cublasDestroy(handle);
    delete timer;
    return timeCublas;
}

string isPass(double relErr){
    if(relErr < 1e-07){
        return "PASS";
    }

    return "FAILED";
}