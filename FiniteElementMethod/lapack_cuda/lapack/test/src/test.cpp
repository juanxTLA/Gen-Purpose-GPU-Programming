#include <Timer.hpp>
#include <lapacke.h>

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cublas_v2.h>

#include <Matrix.hpp>

using namespace std;

#define PI 3.141592

void fem1d(Matrix<float> &x);
void fem1dHost(Matrix<float> &x);

float intHat1(float x1, float x2);
float intHat2(float x1, float x2);

float hat1(float x, float x1, float x2);
float hat2(float x, float x1, float x2);

float f(float x);

void cublasGradient(int m, float* aHost, float* bHost, int lda);

int main(int argc, char** argv){

    int n = atoi(argv[1]);
    float step = 1.0f/(n-1);
    // Compute values
    Matrix<float> x(n, 1);

    for(int i = 0; i < n; ++i){
        x(i, 0) = i * step;
    }

    Timer t1;

    //COMMENT/UNCOMMENT FOR HOST
    // t1.start();
    // fem1dHost(x);
    // t1.stop();


    //COMMENT/UNCOMMENT FOR GPU
    t1.start();
    fem1d(x);
    t1.stop();


    cout << n << " " << t1.elapsedTime_ms() << endl;

    return 0;
}

void fem1d(Matrix<float> &x){
    int m = x.numberOfRows();

    Matrix<float> h(m-1, 1);

    for(int i = 0; i < m-1; ++i){
        h(i, 0) = x(i+1, 0) - x(i, 0);
    }

    Matrix<float> a(m, m);
    Matrix<float> f(m, 1);

    a(0, 0) = 1.0f;
    a(1, 1) = 1/h(0,0);
    a(m-1, m-1) = 1.0f;

    f(m-1, 0) = 0.0f;
    f(1, 0) = intHat1(x(0,0), x(1,0));

    for(int i = 1; i < m - 2; ++i){
        a(i,i) = a(i,i) + 1/h(i,0);
        a(i,i+1) = a(i,i+1) - 1/h(i,0);
        a(i+1,i) = a(i+1,i) - 1/h(i,0);
        a(i+1,i+1) = a(i+1,i+1) + 1/h(i,0);

        f(i, 0) = f(i,0) + intHat2(x(i,0), x(i+1,0));
        f(i+1, 0) = intHat1(x(i,0), x(i+1,0));
    }

    a(m-2,m-2) = a(m-2,m-2) + 1/h(m-2,0);
    f(m-2, 0) = f(m-2,0) + intHat2(x(m-2,0), x(m-1,0));  

    // a.print();
    // f.print();      

    cublasGradient(m, a.data(), f.data(), a.leadingDimension());

    // COMMENT/UNCOMMENT for validation
    // f.print();

}

void fem1dHost(Matrix<float> &x){
  int m = x.numberOfRows();

  Matrix<float> h(m-1, 1);

  for(int i = 0; i < m-1; ++i){
    h(i, 0) = x(i+1, 0) - x(i, 0);
  }

  Matrix<float> a(m, m);
  Matrix<float> f(m, 1);
  Matrix<int> ipiv(m, 1);

  a(0, 0) = 1.0f;
  a(1, 1) = 1/h(0,0);
  a(m-1, m-1) = 1.0f;

  f(m-1, 0) = 0.0f;
  f(1, 0) = intHat1(x(0,0), x(1,0));

  for(int i = 1; i < m - 2; ++i){
    a(i,i) = a(i,i) + 1/h(i,0);
    a(i,i+1) = a(i,i+1) - 1/h(i,0);
    a(i+1,i) = a(i+1,i) - 1/h(i,0);
    a(i+1,i+1) = a(i+1,i+1) + 1/h(i,0);

    f(i, 0) = f(i,0) + intHat2(x(i,0), x(i+1,0));
    f(i+1, 0) = intHat1(x(i,0), x(i+1,0));
  }

  a(m-2,m-2) = a(m-2,m-2) + 1/h(m-2,0);
  f(m-2, 0) = f(m-2,0) + intHat2(x(m-2,0), x(m-1,0));

  //we need to solve for A \ F
  //Gradient method

  LAPACKE_sgesv(
    LAPACK_ROW_MAJOR,
    a.numberOfRows(),
    f.leadingDimension(),
    a.data(),
    a.leadingDimension(),
    ipiv.data(),
    f.data(),
    f.leadingDimension()
  );
  
    //COMMENT/UNCOMMENT for validation
    //   f.print();
}

float intHat1(float x1, float x2){
    float xm = (x1 + x2)*0.5;
    float y = (x2 - x1) * (f(x1) * hat1(x1, x1, x2) + 4*f(xm) * hat1(xm, x1, x2) +
                    f(x2) * hat1(x2,x1,x2)) / 6;

    return y;
}

float intHat2(float x1, float x2){
    float xm = (x1 + x2)*0.5;
    float y = (x2 - x1) * (f(x1) * hat2(x1, x1, x2) + 4*f(xm) * hat2(xm, x1, x2) +
                    f(x2) * hat2(x2,x1,x2)) / 6;

    return y;
}

float hat1(float x, float x1, float x2){
    return (x-x1)/(x2-x1);
}

float hat2(float x, float x1, float x2){
    return (x2-x)/(x2-x1);
}

float f(float x){
    return PI*PI*sin(PI*x);
}

void cublasGradient(int m, float* aHost, float* bHost, int lda){
    // we first get our guess
    Matrix<float> xkHost(m,1);
    for(int i = 0; i < m; ++i){
        xkHost(i, 0) = 0.0f;
    }

    //First operation we need to do is cublasSgemv
    // y = a*A*x + By

    size_t vSize = m * sizeof(float); //vector size
    size_t mSize = m * m * sizeof(float); //matrixSize

    float *xkDevice = nullptr;
    float *rkDevice = nullptr;
    float *pkDevice = nullptr;
    float *tempDevice = nullptr;
    
    float *aDevice = nullptr;
    float *bDevice = nullptr;

    checkCudaErrors(cudaMalloc(&xkDevice, vSize));
    checkCudaErrors(cudaMalloc(&rkDevice, vSize));
    checkCudaErrors(cudaMalloc(&pkDevice, vSize));

    checkCudaErrors(cudaMalloc(&bDevice, vSize));
    checkCudaErrors(cudaMalloc(&aDevice, mSize));

    checkCudaErrors(cudaMemcpy(aDevice, aHost, mSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(bDevice, bHost, vSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(xkDevice, xkHost.data(), vSize, cudaMemcpyHostToDevice));

    //Start operating in the device
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasOperation_t trans = CUBLAS_OP_N;

    float *norm = (float*)malloc(sizeof(float));
    float *resdot = (float*)malloc(sizeof(float));
    float *alphaK = (float*)malloc(sizeof(float));
    float *betaK = (float*)malloc(sizeof(float));

    float alpha = -1.0f;
    float beta = 1.0f;

    cublasSgemv(handle, trans,
                m, m,
                &alpha,
                aDevice, lda,
                xkDevice, 1,
                &beta,
                bDevice, 1);

    cublasSnrm2(handle, m, bDevice, 1, norm);

    int iterationCount = 0;

    checkCudaErrors(cudaMemcpy(rkDevice, bDevice, vSize, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(pkDevice, bDevice, vSize, cudaMemcpyDeviceToDevice));

    //bDevice will now be used as temp variable
    // cout << "norm: " << *norm << endl;
    while(*norm > 1.0e-5f){
        // resdot = (r_k' * r_k) % dot product
        cublasSdot (handle, m, pkDevice, 1, pkDevice, 1, resdot);
        // cout << "\nresdot: " << *resdot << endl;

        // alpha_k = resdot / (p_k' * (A * p_k)); % denominator is gemv and a dot
        beta = 0.0f;
        alpha = 1.0f;
        cublasSgemv(handle, trans,
            m, m,
            &alpha,
            aDevice, lda,
            pkDevice, 1,
            &beta,
            bDevice, 1);
        
        float *temp = NULL;
        cublasSdot (handle, m, bDevice, 1, pkDevice, 1, alphaK);
        *alphaK = (*resdot) / (*alphaK);
        // cout << "\nalphaK: " << *alphaK << endl;

        // x_k = x_k + alpha_k * p_k; % this is a saxpy
        cublasSaxpy(handle, m, alphaK, pkDevice, 1, xkDevice, 1);

        // r_k = r_k - alpha_k * (A * p_k); % another gemv and a saxpy
        cublasSgemv(handle, trans,
            m, m,
            &alpha,
            aDevice, lda,
            pkDevice, 1,
            &beta,
            bDevice, 1);
        
        // checkCudaErrors(cudaMemcpy(rkDevice, bDevice, vSize, cudaMemcpyDeviceToDevice));
        alpha = -(*alphaK); 
        cublasSaxpy(handle, m, &alpha, bDevice, 1, rkDevice, 1);

        cublasSdot (handle, m, rkDevice, 1, rkDevice, 1, betaK);

        *betaK = (*betaK) / (*resdot);
        // cout << "\nbetaK: " << *betaK << endl;

        // p_k = r_k + beta_k * p_k; % another saxpy
        //save value of rk in bDevice to not lose it
        checkCudaErrors(cudaMemcpy(bDevice, rkDevice, vSize, cudaMemcpyDeviceToDevice));
        
        cublasSaxpy(handle, m, betaK, pkDevice, 1, rkDevice, 1);
        //store value of bDevice into pkDevice
        checkCudaErrors(cudaMemcpy(pkDevice, rkDevice, vSize, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(rkDevice, bDevice, vSize, cudaMemcpyDeviceToDevice));
        
        cublasSnrm2(handle, m, rkDevice, 1, norm);
        iterationCount++;
    }

    //retrieve results to host
    checkCudaErrors(cudaMemcpy(bHost, xkDevice, vSize, cudaMemcpyDeviceToHost));

    /*
        cublasStatus_t  cublasSnrm2(cublasHandle_t handle, int n,
                            const float           *x, int incx, float  *result)

        cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *x, int incx,
                           const float           *beta,
                           float           *y, int incy)

        cublasStatus_t cublasSdot (cublasHandle_t handle, int n,
                           const float           *x, int incx,
                           const float           *y, int incy,
                           float                   *result)
        
        y = alpha*x + y
        cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n,
                            const float           *alpha,
                            const float           *x, int incx,
                            float                 *y, int incy)
    */



    cublasDestroy(handle);
   
    cudaFree(xkDevice);
    cudaFree(rkDevice);
    cudaFree(pkDevice);
    cudaFree(aDevice);
    cudaFree(bDevice);

}
