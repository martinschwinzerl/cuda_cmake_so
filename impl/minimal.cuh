#ifndef SIXTRACKLIB_CUDA_IMPL_MINIMAL_CUH__
#define SIXTRACKLIB_CUDA_IMPL_MINIMAL_CUH__

__global__ void calculateSum( uint64_t const n,
                     double const* __restrict__ a,
                     double const* __restrict__ b,
                     double* __restrict__ c );

#endif /* SIXTRACKLIB_CUDA_IMPL_MINIMAL_CUH__ */

