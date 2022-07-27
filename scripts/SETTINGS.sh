#
# Configuration file for all compilers
#

#
# Global compiler flags
#

# default compiler flags
CFLAGS="-std=c++11 -O3"
# compiler flags to use in case a debug compile is issued
CFLAGS_DEBUG="-std=c++11 -O0"

#
# CLANG compiler configuration
#

# actual compiler call
CLANG_COMPILER=clang++
# settings to enable OpenMP (offloading) in CLANG
CLANG_OPENMP_FLAGS="-fopenmp -fopenmp-targets=%TARGET_TRIPLE% -fopenmp-version=51 --offload-arch=%OFFLOAD_ARCH%"
# flags the generally are part of the call to CLANG
CLANG_DEFAULT_FLAGS="-Ofast"
# flags that should be used in case a debug compile is issued
CLANG_DEBUG_FLAGS="-g"
# benchmark compilation specific flags
CLANG_BENCHMARK_FLAGS="-DNO_LOOP_DIRECTIVES"

#
# GCC compiler configuration
# TODO fill out
#

# actual compiler call
GCC_COMPILER=g++
# settings to enable OpenMP (offloading) in GCC
GCC_OPENMP_FLAGS=""
# flags the generally are part of the call to GCC
GCC_DEFAULT_FLAGS=""
# flags that should be used in case a debug compile is issued
GCC_DEBUG_FLAGS=""
# benchmark compilation specific flags
GCC_BENCHMARK_FLAGS=""

#
# XLC compiler configuration
#

# actual compiler call
XLC_COMPILER=xlc++
# settings to enable OpenMP (offloading) in XLC
XLC_OPENMP_FLAGS="-qsmp -qoffload -qtgtarch=%OFFLOAD_ARCH%"
# flags the generally are part of the call to XLC
XLC_DEFAULT_FLAGS="-Ofast"
# flags that should be used in case a debug compile is issued
XLC_DEBUG_FLAGS="-g"
# benchmark compilation specific flags
XLC_BENCHMARK_FLAGS="-DNO_LOOP_DIRECTIVES"

#
# NVC compiler configuration
#

# actual compiler call
NVC_COMPILER=nvc++
# settings to enable OpenMP (offloading) in NVC
NVC_OPENMP_FLAGS="-mp=gpu -target=gpu -gpu=%OFFLOAD_ARCH%"
# flags the generally are part of the call to NVC
NVC_DEFAULT_FLAGS="-fast"
# flags that should be used in case a debug compile is issued
NVC_DEBUG_FLAGS="-g"
# benchmark compilation specific flags
NVC_BENCHMARK_FLAGS="-DNO_MEM_DIRECTIVES"

#
# ROCM CLANG compiler configuration
#

# actual compiler call
ROCM_COMPILER=/opt/rocm-5.2.0/llvm/bin/clang++
# settings to enable OpenMP (offloading) in ROCM
ROCM_OPENMP_FLAGS="-target %HOST_TRIPLE% -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=%OFFLOAD_ARCH% --offload-arch=%OFFLOAD_ARCH%"
# flags the generally are part of the call to ROCM
ROCM_DEFAULT_FLAGS="-Ofast"
# flags that should be used in case a debug compile is issued
ROCM_DEBUG_FLAGS="-g"
# benchmark compilation specific flags
ROCM_BENCHMARK_FLAGS="-DNO_LOOP_DIRECTIVES -DNO_MEM_DIRECTIVES"

#
# AOMP compiler configuration
#

# actual compiler call
AOMP_COMPILER=aompcc
# settings to enable OpenMP (offloading) in AOMP
AOMP_OPENMP_FLAGS=""
# flags the generally are part of the call to AOMP
AOMP_DEFAULT_FLAGS="-Ofast"
# flags that should be used in case a debug compile is issued
AOMP_DEBUG_FLAGS="-g"
# benchmark compilation specific flags
AOMP_BENCHMARK_FLAGS="-DNO_LOOP_DIRECTIVES"

#
# CUDA compiler configuration
#

# actual compiler call
CUDA_COMPILER=nvcc
# flags the generally are part of the call to CUDA
CUDA_DEFAULT_FLAGS="-fast -lcurand -arch=%OFFLOAD_ARCH%"
# flags that should be used in case a debug compile is issued
CUDA_DEBUG_FLAGS="-g -lcurand -arch=%OFFLOAD_ARCH%"
# benchmark compilation specific flags
CUDA_BENCHMARK_FLAGS=""

#
# HIP compiler configuration
#

# actual compiler call
HIP_COMPILER=hipcc
# flags the generally are part of the call to HIP
HIP_DEFAULT_FLAGS="-lrocblas -lhiprand"
# flags that should be used in case a debug compile is issued
HIP_DEBUG_FLAGS="-g"
# benchmark compilation specific flags
HIP_BENCHMARK_FLAGS=""
