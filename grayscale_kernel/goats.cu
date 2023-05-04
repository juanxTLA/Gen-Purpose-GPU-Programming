#include <iostream>
#include <fstream>

using namespace std;

__global__ void rg2gray_kernel(unsigned char* r, unsigned char* g, unsigned char* b, 
                                unsigned char* gray, int height, int width){
    
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;

    if(row < height && col < width){
        gray[row*width + col] = r[row*width + col] *3/10 + g[row*width + col]*6/10
                                + b[row*width + col]*1/10;
    }

}


int main(){

    int width, height;

    ofstream outFile; //txt file with the grayscale data
    ifstream inFile; //txt file with the rgb data

    inFile.open("rgb_data.txt");

    inFile >> width >> height;
    
    const int N = width * height; 

    unsigned char redMatrix[N];
    unsigned char greenMatrix[N];
    unsigned char blueMatrix[N];
    unsigned char grayMatrix[N];
    
    int temp;

    for(int r = 0; r < N; r++){
        inFile >> temp;
        redMatrix[r] = temp;        
        inFile >> temp;
        greenMatrix[r] = temp;        
        inFile >> temp;
        blueMatrix[r] = temp;        
    }
    
    //GPU Part
    //Memory management
    unsigned char  *red, *green, *blue, *gray;
    cudaMalloc((void**)&red, N*sizeof(unsigned char));
    cudaMalloc((void**)&green, N*sizeof(unsigned char));
    cudaMalloc((void**)&blue, N*sizeof(unsigned char));
    cudaMalloc((void**)&gray, N*sizeof(unsigned char));

    cudaMemcpy(red, &redMatrix, N * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(green, &greenMatrix, N * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(blue, &blueMatrix, N * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    dim3 numThreadsPerBlock(1024,1024);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x,
                    (height + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);

    rg2gray_kernel <<< numThreadsPerBlock, numBlocks >>> (red, green, blue, gray, height, width);

    cudaMemcpy(grayMatrix, gray, N*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(red);
    cudaFree(green);
    cudaFree(blue);
    cudaFree(gray);

    outFile.open("grayscale.txt");

    for(int r = 0; r < height; r++){
        for(int c = 0; c < width; c++){
                  
            outFile << +grayMatrix[r*width + c] << " ";

        }

        outFile << endl;
    }

    outFile.close();
    inFile.close();

    return 0;
}