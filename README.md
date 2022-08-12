# OpenMP Offloading Analysis

Source code and results for OpenMP, CUDA, HIP and BLAS matrix multiplication benchmarks.

## Note

The method names in the result section (`results/**/*.csv`) are from older versions of the application.
The underlying code did not change.

Name mapping:
- `ijk` = `ijk`
- `ijk_loop` = `ijk_loop`
- `blocked_shmem` = `blocked_ab`
- `blocked_shmem_mem_directives` = `blocked_ab_mem_allocator`
- `blocked_k` = `blocked_a` (no thread limit, block size = 1024)

All other methods are no longer in the source code.

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
