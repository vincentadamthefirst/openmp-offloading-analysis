cmake_minimum_required(VERSION 3.10)

project(OpenMP-Offloading-Analysis VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CFLAGS, "-fopenmp")

add_compile_options("$<$<CONFIG:RELEASE>:-O3;${CFLAGS}>")
add_compile_options("$<$<CONFIG:DEBUG>:${CFLAGS}>")

set(CMAKE_BINARY_DIR "bin")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/../)

add_executable(openmp src/openmp/benchmark/benchmark.cpp)
add_executable(loop_order src/openmp/loop_ordering/loop_order.cpp)
add_executable(openmp_compare_c src/openmp/language_comparison/tiled.c)
add_executable(openmp_compare_cpp src/openmp/language_comparison/tiled.cpp)
add_executable(cublas src/cublas/main_cublas.cpp)
add_executable(rocblas src/rocblas/main_rocblas.cpp)
#add_executable(cuda src/cuda/main_cuda.cu)
