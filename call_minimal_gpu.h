#ifndef CUDA_TEST_IMPL_CALL_MINIMAL_H__
#define CUDA_TEST_IMPL_CALL_MINIMAL_H__

#include <cuda_runtime_api.h>
#include <cuda.h>

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

__host__ void call_minimal_gpu( 
    double const* __restrict__ a_begin, 
    double const* __restrict__ a_end, 
    double const* __restrict__ b_begin, 
    double* __restrict__ c_begin, 
    int const num_of_blocks, int const threads_per_block );

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */
    
#endif /* CUDA_TEST_IMPL_CALL_MINIMAL_H__ */

