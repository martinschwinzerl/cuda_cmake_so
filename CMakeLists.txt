
cmake_minimum_required( VERSION 3.8 FATAL_ERROR )

project( sixtracklib LANGUAGES C CXX )

find_package( CUDA REQUIRED )

set( CUDA_TEST_LIB_HEADERS 
    impl/minimal.cuh     
)

set( CUDA_TEST_LIB_SOURCES
     details/minimal.cu
)

set( CUDA_USE_STATIC_CUDA_RUNTIME OFF )
set( CUDA_SEPARABLE_COMPILATION ON )
list( APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_50,code=compute_50" )
list( APPEND CUDA_NVCC_FLAGS "-w" )
# list( APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC" )

cuda_add_library( cuda_test 
             ${CUDA_TEST_LIB_HEADERS}
             ${CUDA_TEST_LIB_SOURCES}
)

cuda_add_executable( test_minimal test_minimal.cu )

target_link_libraries( test_minimal cuda_test )
target_include_directories( test_minimal PUBLIC ${CMAKE_SOURCE_DIR} )
