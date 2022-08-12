#pragma once

#include <cstdint>
#include <omp.h>
#include "../../../include/preprocessor_settings.h"

enum Method {
    IJK,
    IJK_LOOP,
    BLOCKED_AB,
    BLOCKED_AB_MEM_ALLOCATOR,
    BLOCKED_A,
};

// lookup of method names
static std::map<Method, std::string> methodNamesMapping = {
        {Method::IJK,                      "ijk"},
        {Method::IJK_LOOP,                 "ijk_loop"},
        {Method::BLOCKED_AB,               "blocked_ab"},
        {Method::BLOCKED_AB_MEM_ALLOCATOR, "blocked_ab_mem_allocator"},
        {Method::BLOCKED_A,                "blocked_a"},
};

// reverse lookup of method names
static std::map<std::string, Method> methodNamesMappingReversed = {
        {"ijk",                      Method::IJK},
        {"ijk_loop",                 Method::IJK_LOOP},
        {"blocked_ab",               Method::BLOCKED_AB},
        {"blocked_ab_mem_allocator", Method::BLOCKED_AB_MEM_ALLOCATOR},
        {"blocked_a",                Method::BLOCKED_A},
};

// groups of methods
static std::map<std::string, std::vector<Method>> methodGroups = {
        {"basic",   {Method::IJK,        Method::IJK_LOOP}},
        {"blocked", {Method::BLOCKED_AB, Method::BLOCKED_AB_MEM_ALLOCATOR, Method::BLOCKED_A}}
};

namespace Target {
    /**
     * Basic IJK loop ordering multiplication, I is distributed across teams, J is distributed among threads, K
     * runs sequential.
     */
    double ijk_basic(const DT *A, const DT *B, DT *C) {
        double start, end;

#pragma omp target data map(A[0:SIZE*SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
        {
            start = omp_get_wtime();
#if OVERWRITE_THREAD_LIMIT
#pragma omp target teams distribute shared(A, B, C) thread_limit(1024)
#else
#pragma omp target teams distribute shared(A, B, C)
#endif
            for (int i = 0; i < SIZE; i++) {
#pragma omp parallel for
                for (int j = 0; j < SIZE; j++) {
                    for (int k = 0; k < SIZE; k++) {
                        C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                    }
                }
            }

            end = omp_get_wtime();
        }
        return (end - start) * 1000.0;
    }

    /**
     * Basic IJK loop ordering multiplication but using the new loop directive.
     */
#if !NO_LOOP_DIRECTIVES
    double ijk_loop(const DT *A, const DT *B, DT *C) {
        double start, end;
#pragma omp target data map(A[0:SIZE*SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
        {
            start = omp_get_wtime();
#pragma omp target teams loop shared(A, B, C)
            for (int i = 0; i < SIZE; i++) {
#pragma omp loop
                for (int j = 0; j < SIZE; j++) {
#pragma omp loop
                    for (int k = 0; k < SIZE; k++) {
                        C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                    }
                }
            }
            end = omp_get_wtime();
        }
        return (end - start) * 1000.0;
    }
#endif

    /**
     * Blocked matrix multiplication, similar to CUDA implementations.
     * The CUDA grid is simulated by having two for loops for the grid x and y dimension.
     *
     * Each thread is responsible of one output value of C.
     * Blocks of A and B (along the current block-row and block-column) are loaded into shared memory.
     *
     * No bounds protection! The blocks must tile evenly into the matrix.
     */
    double blocked_ab(const DT *A, const DT *B, DT *C) {
        double start, end;
#pragma omp target data map(A[0:SIZE*SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
        {
            start = omp_get_wtime();

            // start as many teams as there are blocks in the matrix
            // each team has TILE_SIZE * TILE_SIZE threads
#pragma omp target teams distribute num_teams(TAS * TAS) collapse(2) thread_limit(TILE_SIZE * TILE_SIZE)
            for (size_t blockIdx = 0; blockIdx < TAS; blockIdx++) {
                for (size_t blockIdy = 0; blockIdy < TAS; blockIdy++) {
                    // shared memory per block
                    DT a_shm[TILE_SIZE * TILE_SIZE];
                    DT b_shm[TILE_SIZE * TILE_SIZE];

#pragma omp parallel num_threads(TILE_SIZE * TILE_SIZE) shared(A, B, C, a_shm, b_shm, blockIdx, blockIdy)
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
    /**
     * Similar the blocked_ab but uses explicit memory allocators for a_shm and b_shm.
     */
    double blocked_ab_mem_allocator(const DT *A, const DT *B, DT *C) {
        double start, end;
#pragma omp target data map(A[0:SIZE*SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
        {
            start = omp_get_wtime();

            // start as many teams as there are blocks in the matrix
            // each team has TILE_SIZE * TILE_SIZE threads
#pragma omp target teams distribute num_teams(TAS * TAS) collapse(2) thread_limit(TILE_SIZE * TILE_SIZE)
            for (int blockIdx = 0; blockIdx < TAS; blockIdx++) {
                for (int blockIdy = 0; blockIdy < TAS; blockIdy++) {
                    // shared memory per block
                    // possible allocators: omp_pteam_mem_alloc, omp_cgroup_mem_alloc
                    DT a_shm[TILE_SIZE * TILE_SIZE];
#pragma omp allocate(a_shm) allocator(omp_pteam_mem_alloc)
                    DT b_shm[TILE_SIZE * TILE_SIZE];
#pragma omp allocate(b_shm) allocator(omp_pteam_mem_alloc)

#pragma omp parallel num_threads(TILE_SIZE * TILE_SIZE) shared(A, B, C, a_shm, b_shm, blockIdx, blockIdy) default(none)
                    {
                        // from here on the code resembles CUDA
                        DT tmp_c = 0;

                        int threadNum = omp_get_thread_num();

                        int threadIdx = threadNum % TILE_SIZE;
                        int threadIdy = threadNum / TILE_SIZE;

                        int row = blockIdy * TILE_SIZE + threadIdy;
                        int col = blockIdx * TILE_SIZE + threadIdx;

                        for (int k = 0; k < TAS; k++) {
                            // read the data to shared memory, each thread stores 1 element in a_shm and 1 in b_shm
                            a_shm[threadIdy * TILE_SIZE + threadIdx] = A[row * SIZE + k * TILE_SIZE + threadIdx];
                            b_shm[threadIdy * TILE_SIZE + threadIdx] = B[(k * TILE_SIZE + threadIdy) * SIZE + col];
#pragma omp barrier
                            // perform the calculations on the previously stored shared memory
                            for (int n = 0; n < TILE_SIZE; n++) {
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
#endif  // NO_MEM_DIRECTIVES

    /**
     * More portable OpenMP code for matrix multiplication, only a part of A is stored in shared memory.
     */
    double blocked_a(const DT *A, const DT *B, DT *C) {
        double start, end;
#pragma omp target data map(to: A[:SIZE * SIZE], B[:SIZE * SIZE]) map(tofrom : C[:SIZE * SIZE])
        {
            start = omp_get_wtime();

#if A_TEAMS == -1
#if A_THREAD_LIMIT == -1
#pragma omp target teams distribute
#else
#pragma omp target teams distribute num_teams(A_THREAD_LIMIT)
#endif // A_THREAD_LIMIT
#else
#if A_THREAD_LIMIT == -1
#pragma omp target teams distribute num_teams(A_TEAMS)
#else
#pragma omp target teams distribute num_teams(A_TEAMS) thread_limit(A_THREAD_LIMIT)
#endif // A_THREAD_LIMIT
#endif // A_TEAMS
            for (int i = 0; i < SIZE; ++i) {
                int in = i * SIZE;
                double a_tmp[A_BLOCK_SIZE];
                for (int kk = 0; kk < SIZE; kk += A_BLOCK_SIZE) {
                    int kk_upper = ((kk + A_BLOCK_SIZE) < SIZE) ? (kk + A_BLOCK_SIZE) : SIZE;

#pragma omp parallel for
                    for (int k = kk; k < kk_upper; ++k) {
                        a_tmp[k - kk] = A[in + k];
                    }

#pragma omp parallel for
                    for (int j = 0; j < SIZE; ++j) {
                        double tmp = 0;
                        for (int k = kk; k < kk_upper; ++k) {
                            // C[i][j] += A[i][k] * B[k][j]
                            tmp += a_tmp[k - kk] * B[k * SIZE + j];
                        }
                        C[in + j] += tmp;
                    }
                }
            }
            end = omp_get_wtime();
        }

        return (end - start) * 1000.0;
    }
}
