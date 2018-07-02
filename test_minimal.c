#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "call_minimal_gpu.h"
#include "call_minimal_cpu.h"

int main()
{
    size_t ii = 0u;
    size_t const N = 100u;
    size_t const N_SIZE = N * sizeof( double );
    double const tolerance = DBL_EPSILON;
    
    double* a     = ( double* )malloc( N_SIZE );
    double* b     = ( double* )malloc( N_SIZE );
    
    double* c_cpu = ( double* )malloc( N_SIZE );
    double* c_gpu = ( double* )malloc( N_SIZE );
    bool  success = true;
        
    assert( ( a != 0 ) && ( b != 0 ) && ( c_cpu != 0 ) && ( c_gpu != 0 ) );
    
    for( ; ii < N ; ++ii )
    {
        a[ ii ] = ( double )2.0l;
        b[ ii ] = ( double )1.0l;
        
        c_cpu[ ii ] = c_gpu[ ii ] = ( double )0.0L;
    }
    
    call_minimal_gpu( a, a + N, b, c_gpu, 64, 64 );
    call_minimal_cpu( a, a + N, b, c_cpu );
    
    for( ii = 0u ; ii < N ; ++ii )
    {
        double const diff = fabs( c_cpu[ ii ] - c_gpu[ ii ] );
        
        if( diff > tolerance )
        {
            printf( "difference @ index %6lu too large : \r\n"
                    " -> cpu           = %.16f \r\n"
                    " -> gpu           = %.16f \r\n"
                    " -> | cpu - gpu | = %.16f \r\n"
                    " -> tolerance     = %.16f \r\n"
                    " => Stop comparison with error! \r\n\r\n",
                    ii, c_cpu[ ii ], c_gpu[ ii ], diff, tolerance );
          
            success = false;
            break;
        }
    }
    
    if( success )
    {
        printf( "SUCCESS!!!\r\n\r\n" );
    }
    
    fflush( stdout );
    
    free( a );
    a = 0;
    
    free( b );
    b = 0;
    
    free( c_cpu );
    c_cpu = 0;
    
    free( c_gpu );
    c_gpu = 0;
    
    return 0;
}

