# cuda_cmake_so
This repository summarizes and documents my current understanding about the way to create a shared-object library containing both regular CPU based and NVIDIA's(TM) Cuda(TM) based implementations of the same functionality, all managed by CMake 3.x . Both the "traditional" and the "modern" approach are documented. 

This document is a work-in-progress and will be expanded as my understanding increases. Any corrections, remarks or questions are very much appreciated.

## General Remarks
This example demonstrates the use of _separable_ _compilation_ , i.e. the CUDA source code (and the c++ non-Cuda host-only code) are distributed over several header and source files. Cuda version 5.x or higher is reqired for this to work.
Cuda can not resolve device-link symbols past library boundaries, i.e. it's not possible to directly launch kernels or start device functions. All calls have to be routed via "extern" functions. Please cf. the Cuda Toolkit documentation [here](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#examples).

## Modern way (CMake $$\geq$$ 3.8): First class language support
### Pre-Requisities: 
* CMake $$\geq$$ 3.8 (for MSVC, which was not tested, 3.9 or even higher may be required)

### Building:
Use the CMakeLists.txt file from the `cmake38_or_higher` folder for building:
```
git clone https://github.com/martinschwinzerl/cuda_cmake_so.git
cd cuda_cmake_so
mkdir build_modern
cd build_modern
cmake ../cmake38_or_higher -DCMAKE_BUILD_TYPE=Release
```
In the modern approach, an `OBJECT` library `cuda_test_gpu_part` is created for the Cuda part of the library while a
`cuda_test_cpu_part` one is created for the non-GPU supported portion. On my machine (CUDA 9.1.85, g++/gcc 7.x), the following output is created:
```
user@computer:~/git/cuda_cmake_so/build_modern$ cmake ../cmake38_or_higher -DCMAKE_BUILD_TYPE=Release
-- The C compiler identification is GNU 7.3.0
-- The CXX compiler identification is GNU 7.3.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- The CUDA compiler identification is NVIDIA 9.1.85
-- Check for working CUDA compiler: /usr/bin/nvcc
-- Check for working CUDA compiler: /usr/bin/nvcc -- works
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Configuring done
-- Generating done
-- Build files have been written to: ~/git/cuda_cmake_so/build_38
user@computer:~/git/cuda_cmake_so/build_modern$ make
Scanning dependencies of target cuda_test_gpu_part
[ 11%] Building CUDA object CMakeFiles/cuda_test_gpu_part.dir~/git/cuda_cmake_so/details/minimal.cu.o
[ 22%] Building CUDA object CMakeFiles/cuda_test_gpu_part.dir~/git/cuda_cmake_so/details/call_minimal_gpu.cu.o
[ 22%] Built target cuda_test_gpu_part
Scanning dependencies of target cuda_test_cpu_part
[ 33%] Building CXX object CMakeFiles/cuda_test_cpu_part.dir~/git/cuda_cmake_so/details/call_minimal_cpu.cpp.o
[ 33%] Built target cuda_test_cpu_part
Scanning dependencies of target cuda_test
[ 44%] Linking CUDA device code CMakeFiles/cuda_test.dir/cmake_device_link.o
[ 55%] Linking C shared library libcuda_test.so
[ 55%] Built target cuda_test
Scanning dependencies of target test_minimal_cxx
[ 66%] Building CXX object CMakeFiles/test_minimal_cxx.dir~/git/cuda_cmake_so/test_minimal.cpp.o
[ 77%] Linking CXX executable test_minimal_cxx
[ 77%] Built target test_minimal_cxx
Scanning dependencies of target test_minimal_c
[ 88%] Building C object CMakeFiles/test_minimal_c.dir~/git/cuda_cmake_so/test_minimal.c.o
[100%] Linking C executable test_minimal_c
[100%] Built target test_minimal_c
```
Some remarks:
* Note the call to `enable_language( CUDA )` which is required for getting the language support and the (on purpose) missing `find_package( CUDA REQUIRED)`. 
* Furthermore, please note that the link-libraries from Cuda are not set explicitly but are transitively passed on by CMake.
* Note that for this documented compilation, the error message with gcc/g++ >= 6.x documente below in the traditional approach did *not* occur. There seem to be also no other adversial effects of running the program compiled with Gcc 7.x but further tests beyond this simplistic example are probably needed to verify this.

## Traditional approach using find_package( CUDA )
### Pre-Requisities: 
* CMake $$\geq$$ 3.3
* A g++/gcc compiler $$\leq$$ 6.x; Otherwise on this machine, the following error message is displayed at compiling some generated intermediate files: 
```
[ 22%] Building NVCC (Device) object CMakeFiles/cuda_test.dir/__/details/cuda_test_generated_call_minimal_gpu.cu.o
In file included from /usr/include/host_config.h:50:0,
                 from /usr/include/cuda_runtime.h:78,
                 from <command-line>:0:
/usr/include/crt/host_config.h:121:2: error: #error -- unsupported GNU version! gcc versions later than 6 are not supported!
 #error -- unsupported GNU version! gcc versions later than 6 are not supported!
  ^~~~~
CMake Error at cuda_test_generated_call_minimal_gpu.cu.o.Release.cmake:219 (message):
  Error generating
  ~/git/cuda_cmake_so/build_legacy/CMakeFiles/cuda_test.dir/__/details/./cuda_test_generated_call_minimal_gpu.cu.
  ```
  ### Building:
 Use the CMakeLists.txt file from the `cmake3x` folder for building:
```
git clone https://github.com/martinschwinzerl/cuda_cmake_so.git
cd cuda_cmake_so
mkdir build_legacy
cd build_legacy
cmake ../cmake3x -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-6 -DCMAKE_C_COMPILER=gcc-6
```
In the modern approach, an `OBJECT` library `cuda_test_cpu_part` is created for the CPU only part and the CUDA portion of the library is created using the `cuda_wrap_srcs` macro. Using `cuda_add_library` directly would have been possible and would have worked but would be less flexible in the context of more complicated non-Cuda parts. I.e. this is the boilerplate for how to handle a library like sixtracklib with an (optional) Cuda part.

Using the `cmake` command-line arguments from above, compilation looks like this:
```
user@computer:~/git/cuda_cmake_so/build_legacy$ cmake ../cmake3x/ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-6 -DCMAKE_C_COMPILER=gcc-6
-- Configuring done
You have changed variables that require your cache to be deleted.
Configure will be re-run and you may have to reset some variables.
The following variables have changed:
CMAKE_C_COMPILER= gcc-6
CMAKE_CXX_COMPILER= g++-6

-- The C compiler identification is GNU 6.4.0
-- The CXX compiler identification is GNU 6.4.0
-- Check for working C compiler: /usr/bin/gcc-6
-- Check for working C compiler: /usr/bin/gcc-6 -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/g++-6
-- Check for working CXX compiler: /usr/bin/g++-6 -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Looking for pthread_create
-- Looking for pthread_create - not found
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE  
-- Found CUDA: /usr (found version "9.1") 
-- Configuring done
-- Generating done
-- Build files have been written to: ~/git/cuda_cmake_so/build_legacy
user@computer:~/git/cuda_cmake_so/build_legacy$ make
Scanning dependencies of target cuda_test_cpu_part
[ 11%] Building CXX object CMakeFiles/cuda_test_cpu_part.dir~/git/cuda_cmake_so/details/call_minimal_cpu.cpp.o
[ 11%] Built target cuda_test_cpu_part
[ 22%] Building NVCC (Device) object CMakeFiles/cuda_test.dir/__/details/cuda_test_generated_call_minimal_gpu.cu.o
[ 33%] Building NVCC (Device) object CMakeFiles/cuda_test.dir/__/details/cuda_test_generated_minimal.cu.o
[ 44%] Building NVCC intermediate link file CMakeFiles/cuda_test.dir/cuda_test_intermediate_link.o
Scanning dependencies of target cuda_test
[ 55%] Linking C shared library libcuda_test.so
[ 55%] Built target cuda_test
Scanning dependencies of target test_minimal_c
[ 66%] Building C object CMakeFiles/test_minimal_c.dir~/git/cuda_cmake_so/test_minimal.c.o
[ 77%] Linking C executable test_minimal_c
[ 77%] Built target test_minimal_c
Scanning dependencies of target test_minimal_cxx
[ 88%] Building CXX object CMakeFiles/test_minimal_cxx.dir~/git/cuda_cmake_so/test_minimal.cpp.o
[100%] Linking CXX executable test_minimal_cxx
[100%] Built target test_minimal_cxx
```
Some remarks:
* The overall complexity is much higher and the proper usage of helper functions is to the best of my knowledge not explicitly documented. Usage and calling conventions were reconstucted by reading the FindCUDA.cmake file, thus the approach documented here may or may not work with other versions of cmake in the same way.
* Compilation of an OBJECT library is an open issue, as linking of the final resulting shared library leads to unresolved symbols $$\longrightarrow$$ TODO!
