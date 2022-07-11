cmake_minimum_required(VERSION 3.10)

project(OpenMP-Offloading-Analysis VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_options("$<$<CONFIG:RELEASE>:-O3;${CFLAGS}>")
add_compile_options("$<$<CONFIG:DEBUG>:${CFLAGS}>")

set(CMAKE_BINARY_DIR "bin")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/../)

add_executable(openmp src/openmp/main_openmp.cpp)
add_executable(openmp_compare_c src/openmp_compare/tiled.c)
add_executable(openmp_compare_cpp src/openmp_compare/tiled.cpp)
add_executable(cublas src/cublas/main_cublas.cpp)
