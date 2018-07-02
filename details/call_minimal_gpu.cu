#include "call_minimal_gpu.h"

#include <algorithm>
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

#include "minimal.cuh"

extern __host__ void call_minimal_gpu( 
    double const* __restrict__ a_begin, 
    double const* __restrict__ a_end, 
    double const* __restrict__ b_begin, 
    double* __restrict__ c_begin, 
    int const num_of_blocks, int const threads_per_block );


void __host__ call_minimal_gpu( 
    double const* __restrict__ a_begin, 
    double const* __restrict__ a_end, 
    double const* __restrict__ b_begin, 
    double* __restrict__ c_begin, 
    int const num_of_blocks, 
    int const threads_per_block )
{
    if( ( a_begin != nullptr ) && ( a_end   != nullptr ) &&
        ( b_begin != nullptr ) && ( c_begin != nullptr ) )
    {
        std::ptrdiff_t const temp = std::distance( a_begin, a_end );
        
        if( temp > 0 )
        {
            uint64_t const N = temp;
            std::size_t const N_SIZE = sizeof( double ) * N;
            
            double* cuda_a = nullptr;
            double* cuda_b = nullptr;
            double* cuda_c = nullptr;
            
            ::cudaMalloc( &cuda_a, N_SIZE );
            ::cudaMalloc( &cuda_b, N_SIZE );
            ::cudaMalloc( &cuda_c, N_SIZE );
            
            ::cudaMemcpy( cuda_a, a_begin, N_SIZE, cudaMemcpyHostToDevice );
            ::cudaMemcpy( cuda_b, b_begin, N_SIZE, cudaMemcpyHostToDevice );
            ::cudaMemcpy( cuda_c, c_begin, N_SIZE, cudaMemcpyHostToDevice );
            
            int device = 0;
            ::cudaGetDevice( &device );
            
            ::cudaDeviceProp deviceProp;
            ::cudaGetDeviceProperties( &deviceProp, device);
            std::cout << deviceProp.major << "." 
                      << deviceProp.minor << std::endl;
                    
            ::cudaDeviceSynchronize();
            cudaError_t err = ::cudaGetLastError();
            assert( err == cudaSuccess );
            
            calculateSum<<< num_of_blocks, threads_per_block >>>( 
                N, cuda_a, cuda_b, cuda_c );
            
            err = ::cudaPeekAtLastError();
            assert( err == ::cudaSuccess );
            
            ::cudaMemcpy( c_begin, cuda_c, N_SIZE, cudaMemcpyDeviceToHost );
            
            ::cudaFree( cuda_a );
            ::cudaFree( cuda_b );
            ::cudaFree( cuda_c );
        }        
    }
    
    return;
}
