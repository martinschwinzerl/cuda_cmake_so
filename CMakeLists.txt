
cmake_minimum_required( VERSION 3.8 FATAL_ERROR )

project( sixtracklib LANGUAGES C CXX )

# ==============================================================================
# ===== libcuda_test.so :

# ----- CPU only part -> OBJECT library libcuda_test_cpu_part

set( CUDA_TEST_CPU_PART_HEADERS
     call_minimal_cpu.h
)

set( CUDA_TEST_CPU_PART_SOURCES
     details/call_minimal_cpu.cpp
)

add_library( cuda_test_cpu_part OBJECT 
             ${CUDA_TEST_CPU_PART_HEADERS}
             ${CUDA_TEST_CPU_PART_SOURCES}
)

set_target_properties( cuda_test_cpu_part PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties( cuda_test_cpu_part PROPERTIES LINKER_LANGUAGE CXX )
set_target_properties( cuda_test_cpu_part PROPERTIES CXX_STANDARD 11 )
set_target_properties( cuda_test_cpu_part PROPERTIES CXX_STANDARD_REQUIRED ON )
target_include_directories( cuda_test_cpu_part PUBLIC ${CMAKE_SOURCE_DIR} )

# ----- GPU part with cuda:

# set( CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH} )
find_package( CUDA REQUIRED )

set( CUDA_TEST_GPU_PART_LIB_HEADERS 
    minimal.cuh
    call_minimal_gpu.h     
)

set( CUDA_TEST_GPU_PART_LIB_SOURCES
     details/minimal.cu
     details/call_minimal_gpu.cu
)

set( CUDA_USE_STATIC_CUDA_RUNTIME OFF )
set( CUDA_SEPARABLE_COMPILATION ON )
list( APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_30,code=compute_30" )
list( APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_50,code=compute_50" )
list( APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=compute_60" )
list( APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_61,code=compute_61" )
list( APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_62,code=compute_62" )
list( APPEND CUDA_NVCC_FLAGS "-w" )
list( APPEND CUDA_NVCC_FLAGS "-std=c++11" )
list( APPEND CUDA_NVCC_FLAGS --compiler-options -fPIC )

set(  CUDA_TEST_GPU_PART_LIB_INPUT 
    ${CUDA_TEST_GPU_PART_LIB_HEADERS} 
    ${CUDA_TEST_GPU_PART_LIB_SOURCES} )

cuda_wrap_srcs( cuda_test OBJ CUDA_TEST_GPU_PART_LIB_GEN_FILES 
                ${CUDA_TEST_GPU_PART_LIB_INPUT} SHARED )

cuda_compute_separable_compilation_object_file_name(
    CUDA_TEST_GPU_PART_LIB_LINK_FILE cuda_test 
    "${cuda_test_SEPARABLE_COMPILATION_OBJECTS}" )

# ----  bring everything together into libcuda_test.so
    
add_library( cuda_test SHARED
             $<TARGET_OBJECTS:cuda_test_cpu_part>
             ${CUDA_TEST_GPU_PART_LIB_INPUT}
             ${CUDA_TEST_GPU_PART_LIB_GEN_FILES}
             ${CUDA_TEST_GPU_PART_LIB_LINK_FILE}
)

cuda_link_separable_compilation_objects( 
    "${CUDA_TEST_GPU_PART_LIB_LINK_FILE}" cuda_test ""
    "${cuda_test_SEPARABLE_COMPILATION_OBJECTS}" )

target_link_libraries( cuda_test PUBLIC ${CUDA_LIBRARIES} )

if( CUDA_cudadevrt_LIBRARY )
    target_link_libraries( cuda_test PUBLIC ${CUDA_cudadevrt_LIBRARY} )
endif()

set_target_properties( cuda_test PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties( cuda_test PROPERTIES LINKER_LANGUAGE CXX )
set_target_properties( cuda_test PROPERTIES CXX_STANDARD 11 )
set_target_properties( cuda_test PROPERTIES CXX_STANDARD_REQUIRED ON )
target_include_directories( cuda_test PUBLIC ${CMAKE_SOURCE_DIR} )

# ==============================================================================
# ===== use / test libcuda_test.so -> build executable program test_minimal

cuda_add_executable( test_minimal test_minimal.cpp )
target_link_libraries( test_minimal cuda_test )


