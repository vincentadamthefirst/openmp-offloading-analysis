#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
#pragma once

#include <cstdint>
#include <omp.h>
#include "preprocessor_settings.h"

enum Method {
    IJK = 0,
    IKJ = 1,
    JIK = 2,
    JKI = 3,
    IJK_COLLAPSED = 4,
    JIK_COLLAPSED = 5,
    TILED_SHMEM = 6,
    TILED_SHMEM_MEM_DIRECTIVES = 7,
    IJK_COLLAPSED_LOOP = 8
};

// lookup of method names
static std::map<std::string, Method> methodNamesMapping = {
        {"ijk",                         Method::IJK},
        {"ikj",                         Method::IKJ},
        {"jik",                         Method::JIK},
        {"jki",                         Method::JKI},
        {"ijk_collapsed",               Method::IJK_COLLAPSED},
        {"jik_collapsed",               Method::JIK_COLLAPSED},
        {"tiled_shmem",                 Method::TILED_SHMEM},
        {"tiled_shmem_mem_directives",  Method::TILED_SHMEM_MEM_DIRECTIVES},
        {"ijk_collapsed_loop",          Method::IJK_COLLAPSED_LOOP}};

static std::map<Method, std::string> methodNamesMappingReversed = {
        {Method::IJK,                        "ijk"},
        {Method::IKJ,                        "ikj"},
        {Method::JIK,                        "jik"},
        {Method::JKI,                        "jki"},
        {Method::IJK_COLLAPSED,              "ijk_collapsed"},
        {Method::JIK_COLLAPSED,              "jik_collapsed"},
        {Method::TILED_SHMEM,                "tiled_shmem"},
        {Method::TILED_SHMEM_MEM_DIRECTIVES, "tiled_shmem_mem_directives"},
        {Method::IJK_COLLAPSED_LOOP,         "ijk_collapsed_loop"}
};

namespace Target {

    double multiplyIJK(const DT *A, const DT *B, DT *C) {
        double start, end;
#pragma omp target data map(A[0:SIZE*SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
        {
            start = omp_get_wtime();
#pragma omp target teams distribute parallel for shared(A, B, C) schedule(static) default(none)
            for (size_t i = 0; i < SIZE; i++) {
                for (size_t j = 0; j < SIZE; j++) {
                    for (size_t k = 0; k < SIZE; k++) {
                        C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                    }
                }
            }
            end = omp_get_wtime();
        }
        return (end - start) * 1000.0;
    }

    double multiplyIJKCollapsed(const DT *A, const DT *B, DT *C) {
        double start, end;
#pragma omp target data map(A[0:SIZE*SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
        {
            start = omp_get_wtime();
#pragma omp target teams distribute parallel for collapse(2) shared(A, B, C) schedule(static) default(none)
            for (size_t i = 0; i < SIZE; i++) {
                for (size_t j = 0; j < SIZE; j++) {
                    for (size_t k = 0; k < SIZE; k++) {
                        C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                    }
                }
            }
            end = omp_get_wtime();
        }
        return (end - start) * 1000.0;
    }

#if !NO_LOOP_DIRECTIVES
    double multiplyIJKCollapsedLoop(const DT *A, const DT *B, DT *C) {
        double start, end;
#pragma omp target data map(A[0:SIZE*SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
        {
            start = omp_get_wtime();
#pragma omp target teams loop collapse(2) shared(A, B, C) default(none)
            for (size_t i = 0; i < SIZE; i++) {
                for (size_t j = 0; j < SIZE; j++) {
                    for (size_t k = 0; k < SIZE; k++) {
                        C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                    }
                }
            }
            end = omp_get_wtime();
        }
        return (end - start) * 1000.0;
    }
#endif

    double multiplyTiledNoBankConflict(const DT *A, const DT *B, DT *C) {
        double start, end;
#pragma omp target data map(A[0:SIZE*SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
        {
            start = omp_get_wtime();

#pragma omp target teams distribute num_teams(TAS * TAS) collapse(2) thread_limit(TILE_SIZE * TILE_SIZE)
            for (size_t blockIdx = 0; blockIdx < TAS; blockIdx++) {
                for (size_t blockIdy = 0; blockIdy < TAS; blockIdy++) {

                    // shared memory per block
                    DT a_shm[TILE_SIZE][TILE_SIZE + 1];
                    DT b_shm[TILE_SIZE][TILE_SIZE + 1];

#pragma omp parallel num_threads(TILE_SIZE * TILE_SIZE) default(none) shared(A, B, C, a_shm, b_shm, blockIdx, blockIdy)
                    {
                        DT tmp_c = 0;

                        int threadNum = omp_get_thread_num();

                        int threadIdx = threadNum % TILE_SIZE;
                        int threadIdy = threadNum / TILE_SIZE;

                        int row = blockIdy * TILE_SIZE + threadIdy;
                        int col = blockIdx * TILE_SIZE + threadIdx;

                        for (size_t k = 0; k < TAS; k++) {
                            a_shm[threadIdy][threadIdx] = A[row * SIZE + k * TILE_SIZE + threadIdx];
                            b_shm[threadIdy][threadIdx] = B[(k * TILE_SIZE + threadIdy) * SIZE + col];
#pragma omp barrier
                            for (size_t n = 0; n < TILE_SIZE; n++) {
                                tmp_c += a_shm[threadIdy][n] * b_shm[n][threadIdx];
                            }
#pragma omp barrier
                        }

                        C[((blockIdy * TILE_SIZE + threadIdy) * SIZE) + (blockIdx * TILE_SIZE) + threadIdx] = tmp_c;
                    }
                }
            }

            end = omp_get_wtime();
        }

        return (end - start) * 1000.0;
    }

    double multiplyTiled(const DT *A, const DT *B, DT *C) {
        double start, end;
#pragma omp target data map(A[0:SIZE*SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
        {
            start = omp_get_wtime();

            /*
             * Due to the limitations of the GPU a block size > 16 is not possible with OpenMP 4.5
             * 16 * 16 * 2 * 8 Byte = 16384B (double precision) which should still fit into shared memory but does not.
             * This could mean, that the shared memory is not correctly utilized by the nvc++ compiler as of now.
             */

            // start as many teams as there are blocks in the matrix
            // each team has TILE_SIZE * TILE_SIZE threads
#pragma omp target teams distribute num_teams(TAS * TAS) collapse(2) thread_limit(TILE_SIZE * TILE_SIZE)
            for (size_t blockIdx = 0; blockIdx < TAS; blockIdx++) {
                for (size_t blockIdy = 0; blockIdy < TAS; blockIdy++) {
                    // shared memory per block
                    DT a_shm[TILE_SIZE * TILE_SIZE];
                    DT b_shm[TILE_SIZE * TILE_SIZE];

                    /*
                     * NVHPC only supports OpenMP 4.5 which did not have the extensive methods for memory
                     * management that OpenMP 5.0 provides.
                     * (see https://www.alcf.anl.gov/sites/default/files/2020-01/ALCF-2019_OpenMP_Webinar-OpenMP5-v03.pdf)
                     *
                     * The impact of explicit memory management needs to be checked using the latest LLVM compiler
                     * which supports the newer specification.
                     */

                    // #pragma omp allocate(a_shm) allocator(omp_pteam_mem_alloc)
                    // T* a_shm = (T*) omp_alloc(TILE_SIZE * TILE_SIZE * sizeof(T), omp_pteam_mem_alloc);
                    // T* b_shm = (T*) omp_alloc(TILE_SIZE * TILE_SIZE * sizeof(T), omp_pteam_mem_alloc);

#pragma omp parallel num_threads(TILE_SIZE * TILE_SIZE) shared(A, B, C, a_shm, b_shm, blockIdx, blockIdy) default(none)
                    {
                        // from here on the code resembles CUDA

                        DT tmp_c = 0;

                        int threadNum = omp_get_thread_num();

                        int threadIdx = threadNum % TILE_SIZE;
                        int threadIdy = threadNum / TILE_SIZE;

                        int row = blockIdy * TILE_SIZE + threadIdy;
                        int col = blockIdx * TILE_SIZE + threadIdx;

                        for (size_t k = 0; k < TAS; k++) {
                            // read the data to shared memory, each thread stores 1 element in a_shm and 1 in b_shm
                            a_shm[threadIdy * TILE_SIZE + threadIdx] = A[row * SIZE + k * TILE_SIZE + threadIdx];
                            b_shm[threadIdy * TILE_SIZE + threadIdx] = B[(k * TILE_SIZE + threadIdy) * SIZE + col];
#pragma omp barrier

                            /*
                             * Using OpenMP 5.1 it might be possible to unroll this loop for more speedup.
                             * (see https://www.openmp.org/wp-content/uploads/OpenMP_SC20_Loop_Transformations.pdf)
                             */

                            // perform the calculations on the previously stored shared memory
                            for (size_t n = 0; n < TILE_SIZE; n++) {
                                tmp_c += a_shm[threadIdy * TILE_SIZE + n] * b_shm[n * TILE_SIZE + threadIdx];
                            }
#pragma omp barrier
                        }

                        C[((blockIdy * TILE_SIZE + threadIdy) * SIZE) + (blockIdx * TILE_SIZE) + threadIdx] = tmp_c;
                    }
                }
            }

            end = omp_get_wtime();
        }

        return (end - start) * 1000.0;
    }

#if !NO_MEM_DIRECTIVES
    double multiplyTiledAllocator(const DT *A, const DT *B, DT *C) {
        double start, end;
#pragma omp target data map(A[0:SIZE*SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
        {
            start = omp_get_wtime();

            /*
             * Due to the limitations of the GPU a block size > 16 is not possible with OpenMP 4.5
             * 16 * 16 * 2 * 8 Byte = 16384B (double precision) which should still fit into shared memory but does not.
             * This could mean, that the shared memory is not correctly utilized by the nvc++ compiler as of now.
             */

            // start as many teams as there are blocks in the matrix
            // each team has TILE_SIZE * TILE_SIZE threads
#pragma omp target teams distribute num_teams(TAS * TAS) collapse(2) thread_limit(TILE_SIZE * TILE_SIZE)
            for (size_t blockIdx = 0; blockIdx < TAS; blockIdx++) {
                for (size_t blockIdy = 0; blockIdy < TAS; blockIdy++) {
                    // shared memory per block
                    // possible allocators: omp_pteam_mem_alloc, omp_cgroup_mem_alloc
                    DT a_shm[TILE_SIZE * TILE_SIZE];
#pragma omp allocate(a_shm) allocator(omp_pteam_mem_alloc)
                    DT b_shm[TILE_SIZE * TILE_SIZE];
#pragma omp allocate(b_shm) allocator(omp_pteam_mem_alloc)

                    /*
                     * the following line does not work in Clang due to an undefined reference in nvlink
                     * DT* b_shm = (DT*) omp_alloc(TILE_SIZE * TILE_SIZE * sizeof(DT), omp_pteam_mem_alloc);
                     */

#pragma omp parallel num_threads(TILE_SIZE * TILE_SIZE) shared(A, B, C, a_shm, b_shm, blockIdx, blockIdy) default(none)
                    {
                        // from here on the code resembles CUDA

                        DT tmp_c = 0;

                        int threadNum = omp_get_thread_num();

                        int threadIdx = threadNum % TILE_SIZE;
                        int threadIdy = threadNum / TILE_SIZE;

                        int row = blockIdy * TILE_SIZE + threadIdy;
                        int col = blockIdx * TILE_SIZE + threadIdx;

                        for (size_t k = 0; k < TAS; k++) {
                            // read the data to shared memory, each thread stores 1 element in a_shm and 1 in b_shm
                            a_shm[threadIdy * TILE_SIZE + threadIdx] = A[row * SIZE + k * TILE_SIZE + threadIdx];
                            b_shm[threadIdy * TILE_SIZE + threadIdx] = B[(k * TILE_SIZE + threadIdy) * SIZE + col];
#pragma omp barrier

                            /*
                             * Using OpenMP 5.1 it might be possible to unroll this loop for more speedup.
                             * (see https://www.openmp.org/wp-content/uploads/OpenMP_SC20_Loop_Transformations.pdf)
                             */

                            // perform the calculations on the previously stored shared memory
                            for (size_t n = 0; n < TILE_SIZE; n++) {
                                tmp_c += a_shm[threadIdy * TILE_SIZE + n] * b_shm[n * TILE_SIZE + threadIdx];
                            }
#pragma omp barrier
                        }

                        C[((blockIdy * TILE_SIZE + threadIdy) * SIZE) + (blockIdx * TILE_SIZE) + threadIdx] = tmp_c;
                    }
                }
            }

            end = omp_get_wtime();
        }

        return (end - start) * 1000.0;
    }
#endif

    double multiplyIKJ(const DT *A, const DT *B, DT *C) {
        double start, end;
#pragma omp target data map(A[0:SIZE * SIZE], B[0:SIZE *  SIZE]) map(tofrom:C[0:SIZE * SIZE])
        {
            start = omp_get_wtime();
#pragma omp target teams distribute parallel for shared(A, B, C) schedule(static) default(none)
            for (size_t i = 0; i < SIZE; i++) {
                for (size_t k = 0; k < SIZE; k++) {
                    for (size_t j = 0; j < SIZE; j++) {
                        C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                    }
                }
            }
            end = omp_get_wtime();
        }
        return (end - start) * 1000.0;
    }

    double multiplyIKJCollapsed(const DT *A, const DT *B, DT *C) {
        double start, end;
#pragma omp target data map(A[0:SIZE * SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
        {
            start = omp_get_wtime();
#pragma omp target teams distribute parallel for collapse(2) shared(A, B, C) schedule(static) default(none)
            for (size_t i = 0; i < SIZE; i++) {
                for (size_t k = 0; k < SIZE; k++) {
                    for (size_t j = 0; j < SIZE; j++) {
                        C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                    }
                }
            }
            end = omp_get_wtime();
        }
        return (end - start) * 1000.0;
    }

    double multiplyJIK(const DT *A, const DT *B, DT *C) {
        double start, end;
#pragma omp target data map(A[0:SIZE * SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
        {
            start = omp_get_wtime();
#pragma omp target teams distribute parallel for shared(A, B, C) schedule(static) default(none)
            for (size_t j = 0; j < SIZE; ++j) {
                for (size_t i = 0; i < SIZE; ++i) {
                    for (size_t k = 0; k < SIZE; ++k) {
                        C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                    }
                }
            }
            end = omp_get_wtime();
        }
        return (end - start) * 1000.0;
    }

    double multiplyJIKCollapsed(const DT *A, const DT *B, DT *C) {
        double start, end;
#pragma omp target data map(A[0:SIZE * SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
        {
            start = omp_get_wtime();
#pragma omp target teams distribute parallel for collapse(2) shared(A, B, C) schedule(static) default(none)
            for (size_t j = 0; j < SIZE; ++j) {
                for (size_t i = 0; i < SIZE; ++i) {
                    for (size_t k = 0; k < SIZE; ++k) {
                        C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                    }
                }
            }
            end = omp_get_wtime();
        }
        return (end - start) * 1000.0;
    }

    double multiplyJKI(const DT *A, const DT *B, DT *C) {
        double start, end;
#pragma omp target data map(A[0:SIZE * SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
        {
            start = omp_get_wtime();
#pragma omp target teams distribute parallel for shared(A, B, C) schedule(static) default(none)
            for (size_t j = 0; j < SIZE; ++j) {
                for (size_t k = 0; k < SIZE; ++k) {
                    for (size_t i = 0; i < SIZE; ++i) {
                        C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                    }
                }
            }
            end = omp_get_wtime();
        }
        return (end - start) * 1000.0;
    }

    double multiplyJKICollapsed(const DT *A, const DT *B, DT *C) {
        double start, end;
#pragma omp target data map(A[0:SIZE * SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
        {
            start = omp_get_wtime();
#pragma omp target teams distribute parallel for collapse(2) shared(A, B, C) schedule(static) default(none)
            for (size_t j = 0; j < SIZE; ++j) {
                for (size_t k = 0; k < SIZE; ++k) {
                    for (size_t i = 0; i < SIZE; ++i) {
                        C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                    }
                }
            }
            end = omp_get_wtime();
        }
        return (end - start) * 1000.0;
    }
}
#pragma clang diagnostic pop
