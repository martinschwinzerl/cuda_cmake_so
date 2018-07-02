#include "minimal.cuh"

extern __global__ void calculateSum( uint64_t const n,
                     double const* __restrict__ a,
                     double const* __restrict__ b,
                     double* __restrict__ c );

__global__ void calculateSum( uint64_t const n,
                     double const* __restrict__ a,
                     double const* __restrict__ b,
                     double* __restrict__ c )
{
    int const global_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    #if defined( __CUDA_ARCH__ )
    if( global_id == 0 ) printf( "__CUDA_ARCH__ :%d\r\n", __CUDA_ARCH__ );
    #endif /* defined( __CUDA_ARCH__ ) */
    
    if( global_id < n )
    {
        c[ global_id ] = a[ global_id ] + b[ global_id ];
    }
    
    
    return;
}

