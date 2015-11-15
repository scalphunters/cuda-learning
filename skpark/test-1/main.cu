#include <thrust/version.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <ctime>
#include <iostream>

#include <cuda.h>
#include <curand.h>

using namespace std;

__global__ void vectoradd(float* a,float* b, float* c){
    unsigned int thread_id=threadIdx.x + blockIdx.x*blockDim.x;
    c[thread_id]=a[thread_id]+b[thread_id];
}

int main(){
    clock_t timestamp=clock();
    cudaFree(0);

    int N=1000;
    size_t size=N*sizeof(float);

    float* h_a=(float*)malloc(size);    
    float* h_b=(float*)malloc(size);
    float* h_c=(float*)malloc(size);

    for(int i=0;i<N;i++)
    {
        h_a[i]=1.0f;
        h_b[i]=3.0f;
        h_c[i]=0.0f;
    }

    float* d_a;
    float* d_b;
    float* d_c;
    cudaMalloc(&d_a,size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);

    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);    
    vectoradd<<<4,256>>>(d_a,d_b,d_c);
    cudaMemcpy(h_c,d_c,size,cudaMemcpyDeviceToHost);
    for(int i=0;i<N;i++)
        cout << h_c[i] << " ";
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);
    cout << "hekll"<<endl << timestamp <<endl;
    return 0;
}