#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "impl/minimal.cuh"

int main()
{
    uint64_t const N = 1000;
    std::size_t const N_SIZE = sizeof( double ) * N;
    
    std::vector< double > A( N, 1.0 );
    std::vector< double > B( N, 2.0 );
    std::vector< double > C( N, 0.0 );
    
    double* cuda_a = nullptr;
    double* cuda_b = nullptr;
    double* cuda_c = nullptr;
    
    cudaMalloc( &cuda_a, N_SIZE );
    cudaMalloc( &cuda_b, N_SIZE );
    cudaMalloc( &cuda_c, N_SIZE );
    
    cudaMemcpy( cuda_a, A.data(), N_SIZE, cudaMemcpyHostToDevice );
    cudaMemcpy( cuda_b, B.data(), N_SIZE, cudaMemcpyHostToDevice );
    cudaMemcpy( cuda_c, C.data(), N_SIZE, cudaMemcpyHostToDevice );
    
    int device = 0;
    cudaGetDevice( &device );
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties( &deviceProp, device);
    std::cout << deviceProp.major << "." << deviceProp.minor << std::endl;
            
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    assert( err == cudaSuccess );
    
    calculateSum<<< 64, 64 >>>( N, cuda_a, cuda_b, cuda_c );
    err = cudaPeekAtLastError();
    assert( err == cudaSuccess );
    
    cudaMemcpy( C.data(), cuda_c, N_SIZE, cudaMemcpyDeviceToHost );
    
    cudaFree( cuda_a );
    cudaFree( cuda_b );
    cudaFree( cuda_c );
    
    return 0;
}

