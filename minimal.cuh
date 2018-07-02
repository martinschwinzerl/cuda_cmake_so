#ifndef CUDA_TEST_MINIMAL_CUH__
#define CUDA_TEST_MINIMAL_CUH__

#include <stdint.h>

__global__ void calculateSum( uint64_t const n,
                     double const* __restrict__ a,
                     double const* __restrict__ b,
                     double* __restrict__ c );

#endif /* CUDA_TEST_MINIMAL_CUH__ */
