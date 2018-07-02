#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "impl/call_minimal.h"
#include "impl/call_minimal_cpu.h"

int main()
{
    std::size_t const N = 100;
    
    std::vector< double > const a( N, +1.0 );
    std::vector< double > const b( N, +2.0 );
    
    std::vector< double > c_cpu( N, 0.0 );
    std::vector< double > c_gpu( N, 0.0 );
    
    call_minimal( a.data(), a.data() + N, b.data(), c_gpu.data(), 64, 64 );
    call_minimal_cpu( a.data(), a.data() + N, b.data(), c_cpu.data() );
    
    double const tolerance = std::numeric_limits< double >::epsilon();
    
    bool success = true;
    std::size_t index = 0u;
    
    for( auto cpu_it = c_cpu.cbegin(), gpu_it = c_gpu.cbegin() ; 
            cpu_it != c_cpu.cend() ; ++cpu_it, ++gpu_it, ++index )
    {
        double const diff = std::fabs( *cpu_it - *gpu_it );
        
        if( diff > tolerance )
        {
            std::cout << "difference @ index " << std::setw( 6 ) << index 
                      << " too large : \r\n"
                      << " -> cpu           = " << *cpu_it << "\r\n"
                      << " -> gpu           = " << *gpu_it << "\r\n"
                      << " -> | cpu - gpu | = " << diff    << "\r\n"
                      << " -> tolerance     = " << tolerance << "\r\n"
                      << " => Stop comparison with error! \r\n"
                      << "\r\n";
          
            success = false;
            break;
        }
    }
    
    if( success )
    {
        std::cout << "SUCCESS!!!\r\n";
    }
    
    std::cout << std::endl;
    
    return 0;
}

