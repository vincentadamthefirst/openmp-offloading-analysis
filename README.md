# OpenMP Offloading Analysis

Source code for OpenMP, CUDA, HIP and BLAS matrix multiplication benchmarks.

## Directory Elements

- **include**: header files used throughout the code
- **libs**: libraries used throughout the code
- **scripts**: set of scripts to help with compiling and creating benchmarking runs for slurm
- **singularity**: singularity definition files
- **src**: C/C++ source code
  - **cublas**: cuBLAS implementation (matrix multiplication)
  - **cuda**: CUDA blocked matrix multiplication (shared memory usage)
  - **hip**: HIP blocked matrix multiplication (shared memory usage)
  - **openmp**: OpenMP implementations
    - **benchmark**: actual benchmark (IJK & blocked)
    - **language_comparison**: blocked matrix multiplication to compare C and C++ code
    - **loop_ordering**: code to test different loop orders 
  - **rocblas**: rocBLAS implementation (matrix multiplication)

## Third-Party code

- CmdParser to parse CLI arguments: [https://github.com/FlorianRappl/CmdParser](https://github.com/FlorianRappl/CmdParser), MIT license
- `cuda_helper.h` for CUDA error checking
- `error_macros.h` for HIP error checking
