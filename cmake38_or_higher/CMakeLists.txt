
cmake_minimum_required( VERSION 3.8 FATAL_ERROR )

project( sixtracklib LANGUAGES C CXX )

# ==============================================================================
# ===== libcuda_test.so :

# ----- CPU only part -> OBJECT library libcuda_test_cpu_part

set( CUDA_TEST_CPU_PART_HEADERS
     ../call_minimal_cpu.h
)

set( CUDA_TEST_CPU_PART_SOURCES
     ../details/call_minimal_cpu.cpp
)

add_library( cuda_test_cpu_part OBJECT 
             ${CUDA_TEST_CPU_PART_HEADERS}
             ${CUDA_TEST_CPU_PART_SOURCES}
)

set_target_properties( cuda_test_cpu_part PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties( cuda_test_cpu_part PROPERTIES LINKER_LANGUAGE C )
set_target_properties( cuda_test_cpu_part PROPERTIES CXX_STANDARD 11 )
set_target_properties( cuda_test_cpu_part PROPERTIES CXX_STANDARD_REQUIRED ON )
target_include_directories( cuda_test_cpu_part PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../" )

# ----- GPU part with cuda:

enable_language( CUDA )

string( APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_30,code=compute_30" )
string( APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_50,code=compute_50" )
string( APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_60,code=compute_60" )
string( APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=compute_61" )
string( APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_62,code=compute_62" )

set( CUDA_TEST_GPU_PART_LIB_HEADERS 
    ../minimal.cuh
    ../call_minimal_gpu.h     
)

set( CUDA_TEST_GPU_PART_LIB_SOURCES
     ../details/minimal.cu
     ../details/call_minimal_gpu.cu
)

add_library( cuda_test_gpu_part OBJECT 
             ${CUDA_TEST_GPU_PART_LIB_HEADERS}
             ${CUDA_TEST_GPU_PART_LIB_SOURCES} 
)

set_target_properties( cuda_test_gpu_part PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties( cuda_test_gpu_part PROPERTIES LINKER_LANGUAGE C )
set_target_properties( cuda_test_gpu_part PROPERTIES CXX_STANDARD 11 )
set_target_properties( cuda_test_gpu_part PROPERTIES CXX_STANDARD_REQUIRED ON )
set_target_properties( cuda_test_gpu_part PROPERTIES CUDA_SEPARABLE_COMPILATION  ON  )
set_target_properties( cuda_test_gpu_part PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF )
set_target_properties( cuda_test_gpu_part PROPERTIES CUDA_STANDARD 11 )
set_target_properties( cuda_test_gpu_part PROPERTIES CUDA_STANDARD_REQUIRED 11 )
target_include_directories( cuda_test_gpu_part PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../" )

# ----  bring everything together into libcuda_test.so
    
add_library( cuda_test SHARED
             $<TARGET_OBJECTS:cuda_test_cpu_part>
             $<TARGET_OBJECTS:cuda_test_gpu_part>
)

set_target_properties( cuda_test PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties( cuda_test PROPERTIES LINKER_LANGUAGE C )

# ==============================================================================
# ===== use / test libcuda_test.so -> build executable programs for C and C++

add_executable( test_minimal_cxx ../test_minimal.cpp )
target_link_libraries( test_minimal_cxx cuda_test )
set_target_properties( test_minimal_cxx PROPERTIES CXX_STANDARD 11 )
set_target_properties( test_minimal_cxx PROPERTIES CXX_STANDARD_REQUIRED ON )

add_executable( test_minimal_c ../test_minimal.c )
target_link_libraries( test_minimal_c cuda_test )
set_target_properties( test_minimal_c PROPERTIES C_STANDARD 99 )
set_target_properties( test_minimal_c PROPERTIES C_STANDARD_REQUIRED ON )
