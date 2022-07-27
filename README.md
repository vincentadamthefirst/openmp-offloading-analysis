# OpenMP Offloading Analysis

Set of different matrix multiplication methods with different levels of optimization and loop reordering in OpenMP. 
Used to evaluate different compiler implementations of target offloading.

## Directory Elements

- **include**: header files used throughout the code
- **libs**: libraries used throughout the code
- **scripts**: set of scripts to help with compiling and creating benchmarking runs on Taurus
- **singularity**: singularity definition files
- **src**: C/C++ source code
  - **cublas**: cuBLAS implementation (matrix multiplication)
  - **cuda**: CUDA blocked matrix multiplication (shared memory usage)
  - **openmp**: OpenMP implementations
    - **benchmark**: actual benchmark (IJK & blocked)
    - **language_comparison**: blocked matrix multiplication to compare C and C++ code
    - **loop_ordering**: code to test different loop orders 
  - **rocblas**: rocBLAS implementation (matrix multiplication)

## Compilers & Remarks

- OpenMP:
  - during compilation some attributes of the matrices have to be specified (due to missing dynamic memory allocation in 
  some compilers) 
  - a complete list of preprocessor variables is listed below

| variable           | explanation                                                                | default | example                |
|--------------------|----------------------------------------------------------------------------|---------|------------------------|
| MATRIX_SIZE        | the size of the matrices, must fit into RAM at least 3 times               | 8192    | `-DMATRIX_SIZE=4096`   |
| TILE_SIZE          | tile size for tiled matrix multiplication, limited by SM size              | 16      | `-DTILE_SIZE=32`       |
| DATA_TYPE          | data type to populate the matrices with                                    | double  | `-DDATA_TYPE=float`    |
| NO_LOOP_DIRECTIVES | disable methods using `#pragma omp loop` directives (e.g. for clang)       | false   | `-DNO_LOOP_DIRECTIVES` |
| NO_MEM_DIRECTIVES  | disable methods using `#pragma omp allocate()` directives (e.g. for nvc++) | false   | `-DNO_MEM_DIRECTIVES`  |

- CUBLAS
  - the preprocessor fields have been moved into command line arguments (see [Execution](#execution)/CUBLAS)

### NVC++ (NVHPC 22.5)

#### Compilation

```shell
nvc++ -std=c++11 -O3 -mp=gpu -target=gpu src/openmp/benchmark/benchmark.cpp -o matmul_nvcpp -DNO_MEM_DIRECTIVES
```

#### Remarks

- tile sizes > 32 lead to a compile time error
- the shared memory portion of the tiled methods does not seem to be correctly placed in shared memory
- no support for `#pragma omp allocate()`
- should have support for `omp_alloc()` but it does not work (at least on the GPU)

### clang (15)

#### Compilation

```shell
clang++ -std=c++11 -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-version=51 src/openmp/benchmark/benchmark.cpp -o matmul_clang -DNO_LOOP_DIRECTIVES -DTILE_SIZE=32
```

#### Remarks

- LLVM/clang is significantly better for the tiled matrix multiplication due to correctly moving some fields into shared memory. Passing the option `-Rpass=openmp-opt` during compilation reveals, that the fields are indeed moved to shared memory:

```console
openmp/../include/Target.h:172:23: remark: Replaced globalized variable with 4096 bytes of shared memory. [OMP111] [-Rpass=openmp-opt]
                    T b_shm[BLOCK_SIZE * BLOCK_SIZE];
                      ^
```

- While `omp_alloc()` for dynamic allocation is currently supported (see [LLVM OpenMP Support](https://clang.llvm.org/docs/OpenMPSupport.html#openmp-5-0-implementation-details)) it is not supported in the device runtime yet (see [this answer](https://github.com/llvm/llvm-project/issues/56453#issuecomment-1179608681)). A workaround could be `llvm_omp_target_dynamic_shared_alloc()` described [here](https://openmp.llvm.org//design/Runtimes.html#dynamic-shared-memory). For this to work it seems to be necessary to disable automatic OpenMP optimizations, otherwise the compiler will replace the invocation with an incorrect (too small) shared memory allocation.
- To influence the memory allocation on the target the directive `#pragma omp allocate(<var>) allocator(<allocator>)` can be used without problems. Variables with this pragma will not be automatically moved to shared memory.
- does not support `#pragma omp loop`, currently worked on (see [here](https://clang.llvm.org/docs/OpenMPSupport.html#openmp-5-0-implementation-details))

### ROCm

#### Compilation

```shell
rocm/llvm/bin/clang++ -std=c++11 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 src/openmp/openmp_main.cpp -o matmul_rocm
```

#### Remarks

- currently not fully tested

### CUBLAS

#### Compilation

```shell
nvcc -std=c++11 -O3 -arch=sm_80 src/cublas/main_cublas.cpp -lcublas -lcurand -o matmul_cublas
```

### ROCBLAS

```shell
nvcc -std=c++11 -O3 src/rocblas/main_rocblas.cpp -lrocblas -lhiprand -o matmul_rocblas
```

#### Compilation

## Execution

### OpenMP

| argument        | alias | type        | description                                                                                                                                                                                 | default      | required |
|-----------------|-------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|----------|
| `output`        | `o`   | string      | file to write the results to                                                                                                                                                                | generate new | no       |
| `file_type`     | `ft`  | string      | file type that should be written (`txt` or `csv`)                                                                                                                                           | `txt`        | no       |
| `repetitions`   | `r`   | int         | repetitions within a matrix size                                                                                                                                                            | 11           | no       |
| `verbose`       | `v`   | flag        | when enabled, print more intermediate results to console                                                                                                                                    | false        | no       |
| `methods`       | `m`   | string list | methods to be benchmarked                                                                                                                                                                   | `all`        | no       |
| `print_methods` | `p`   | flag        | print all available methods                                                                                                                                                                 | false        | no       |
| `comparison`    | `c`   | flag        | if set generates a check matrix on the host (CPU) and compares the first matrix of each GPU method against it, cancels the execution if it is incorrect (EXTREMELY slow for large matrices) | false        | no       |

Example:
```shell
# start 20 repetitions, generate a new output file (.csv)
./matmul -ft csv -v -r 20
```

### CUBLAS / ROCBLAS

| argument            | alias | type   | description                                              | default | required |
|---------------------|-------|--------|----------------------------------------------------------|---------|----------|
| `output`            | `o`   | string | file to write the results to                             | /       | no       |
| `matrix_size_start` | `s`   | int    | initial matrix size, will be doubled each iteration      | 4096    | no       |
| `matrix_size_end`   | `e`   | int    | maximum matrix size, size will be included               | 16384   | no       |
| `repetitions`       | `r`   | int    | repetitions within a matrix size                         | 11      | no       |
| `tensor`            | `t`   | flag   | when enabled, force tensor operations                    | false   | no       |
| `verbose`           | `v`   | flag   | when enabled, print more intermediate results to console | false   | no       |
| `precision`         | `p`   | choice | `f32` (single precision) / `f64` (double precision)      | `f64`   | no       |
| `help`              | `h`   | flag   | show help                                                |         | no       |

Example:
```shell
# start 20 repetitions on 1024x1024 single precision matrices using tensor cores
./matmul_cublas --file "path/to/output.txt" -s 1024 -e 1024 -r 20 --tensor --precision f32
```

## Results

### Comparison C vs C++

In `src/openmp_compare` are 2 files with identical code, one is a C-file, the other a CPP file.
Both files contain a tiled matrix multiplication with shared memory usage. They can be used to compare a C and a C++ compiler.

- Results:
  - clang (clang++ / clang)
    - <span style="color:green">no difference</span>
  - nvhpc (nvc++ / nvc)
    - <span style="color:orange">awaiting result</span>

## TODOs

- [ ] add results

## Third-Party code

- CmdParser to parse CLI arguments: [https://github.com/FlorianRappl/CmdParser](https://github.com/FlorianRappl/CmdParser), MIT license
- cuda_helper.h for CUDA error checking
