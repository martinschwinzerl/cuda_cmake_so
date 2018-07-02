#include "call_minimal_cpu.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

extern void call_minimal_cpu( 
    double const* __restrict__ a_begin, 
    double const* __restrict__ a_end, 
    double const* __restrict__ b_begin, 
    double* __restrict__ c_begin );

void call_minimal_cpu( 
    double const* __restrict__ a_it, 
    double const* __restrict__ a_end, 
    double const* __restrict__ b_it, double* __restrict__ c_it )
{
    if( ( a_it != nullptr ) && ( a_end != nullptr ) &&
        ( b_it != nullptr ) && ( c_it  != nullptr ) &&
        ( std::distance( a_it, a_end ) > 0 ) )
    {
        
        for( ; a_it != a_end ; ++a_it, ++b_it, ++c_it )
        {
            *c_it = *a_it + *b_it;
        }        
    }
    
    return;
}
