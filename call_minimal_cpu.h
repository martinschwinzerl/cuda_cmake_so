#ifndef CUDA_TEST_IMPL_CALL_MINIMAL_CPU_H__
#define CUDA_TEST_IMPL_CALL_MINIMAL_CPU_H__

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

void call_minimal_cpu( 
    double const* __restrict__ a_begin, 
    double const* __restrict__ a_end, 
    double const* __restrict__ b_begin, 
    double* __restrict__ c_begin );

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */
    
#endif /* CUDA_TEST_IMPL_CALL_MINIMAL_CPU_H__ */

