# OpenMP Offloading Analysis

Set of different matrix multiplication methods with different levels of optimization and loop reordering in OpenMP. 
Used to evaluate different compiler implementations of target offloading.

Currently still W.I.P.

## Compilation

### OpenMP

| compiler | command                                                       |
|----------|:--------------------------------------------------------------|
| nvhpc    | `nvc++ -std=c++17 -mp=gpu -target=gpu main.cpp -o matmul_omp` |

### CUBLAS

| compiler | command                                                                 |
|----------|:------------------------------------------------------------------------|
| nvcc     | `nvcc -arch=sm_80 matmul_cublas.cpp -lcublas -lcurand -o matmul_cublas` |

## Execution

### CUBLAS

| argument            | alias | type   | description                                              | default | required |
|---------------------|-------|--------|----------------------------------------------------------|---------|----------|
| `file`              | `p`   | string | file to write the results to                             | /       | yes      |
| `matrix_size_start` | `s`   | int    | initial matrix size, will be doubled each iteration      | 4096    | no       |
| `matrix_size_end`   | `e`   | int    | maximum matrix size, size will be included               | 16384   | no       |
| `repetitions`       | `r`   | int    | repetitions within a matrix size                         | 11      | no       |
| `tensor`            | `t`   | flag   | when enabled, force tensor operations                    | false   | no       |
| `verbose`           | `v`   | flag   | when enabled, print more intermediate results to console | false   | no       |
| `seed`              |       | int    | seed for random matrix population                        | clock() | no       |
| `precision`         |       | choice | `f32` (single precision) / `f64` (double precision)      | `f64`   | no       |

Example:
```shell
# start 20 repetitions on 1024x1024 single precision matrices using tensor cores
./matmul_cublas --file "path/to/output.txt" -s 1024 -e 1024 -r 20 --tensor --precision f32
```

## Results

W.I.P.

## TODOs

- [ ] add execution info for OpenMP
- [ ] add singularity container definition file
- [ ] test out / add more compilers
- [ ] add results

## Third-Party code

- Argparse for command line parsing: https://github.com/stdbug/argparse/blob/master/argparse/argparse.h
- cuda_helper.h for CUDA error checking