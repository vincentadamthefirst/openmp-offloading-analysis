#include <cstdint>
#include <omp.h>
#include "preprocessor_settings.h"

enum Method {
    IJK,
    IJK_COLLAPSED,
    IJK_COLLAPSED_LOOP,
    IJK_LOOP,
    IKJ,
    IKJ_COLLAPSED,
    JIK,
    JIK_COLLAPSED,
    JKI,
    JKI_COLLAPSED,
    TILED_SHMEM,
    TILED_SHMEM_MEM_DIRECTIVES,
    TILED_SHMEM_NO_BANK_CONFLICT,
    TILED_K
};

// lookup of method names
static std::map<Method, std::string> methodNamesMapping = {
        {Method::IJK,                          "ijk"},
        {Method::IJK_COLLAPSED,                "ijk_collapsed"},
        {Method::IJK_COLLAPSED_LOOP,           "ijk_collapsed_loop"},
        {Method::IJK_LOOP,                     "ijk_loop"},
        {Method::IKJ,                          "ikj"},
        {Method::IKJ_COLLAPSED,                "ikj_collapsed"},
        {Method::JIK,                          "jik"},
        {Method::JIK_COLLAPSED,                "jik_collapsed"},
        {Method::JKI,                          "jki"},
        {Method::JKI_COLLAPSED,                "jki_collapsed"},
        {Method::TILED_SHMEM,                  "tiled_shmem"},
        {Method::TILED_SHMEM_MEM_DIRECTIVES,   "tiled_shmem_mem_directives"},
        {Method::TILED_SHMEM_NO_BANK_CONFLICT, "tiled_shmem_no_bank_conflict"},
        {Method::TILED_K,                      "tiled_k"},
};

// reverse lookup of method names
static std::map<std::string, Method> methodNamesMappingReversed = {
        {"ijk",                          Method::IJK},
        {"ijk_collapsed",                Method::IJK_COLLAPSED},
        {"ijk_collapsed_loop",           Method::IJK_COLLAPSED_LOOP},
        {"ijk_loop",                     Method::IJK_LOOP},
        {"ikj",                          Method::IKJ},
        {"ikj_collapsed",                Method::IKJ_COLLAPSED},
        {"jik",                          Method::JIK},
        {"jik_collapsed",                Method::JIK_COLLAPSED},
        {"jki",                          Method::JKI},
        {"jki_collapsed",                Method::JKI_COLLAPSED},
        {"tiled_shmem",                  Method::TILED_SHMEM},
        {"tiled_shmem_mem_directives",   Method::TILED_SHMEM_MEM_DIRECTIVES},
        {"tiled_shmem_no_bank_conflict", Method::TILED_SHMEM_NO_BANK_CONFLICT},
        {"tiled_k",                      Method::TILED_K},
};

// groups of methods
static std::map<std::string, std::vector<Method>> methodGroups = {
        {"basic",     {Method::IJK,                        Method::IJK_COLLAPSED, Method::IKJ,                Method::JIK,           Method::JIK_COLLAPSED, Method::JKI}},
        {"collapsed", {Method::JIK_COLLAPSED,              Method::IJK_COLLAPSED, Method::IJK_COLLAPSED_LOOP, Method::IKJ_COLLAPSED, Method::JKI_COLLAPSED}},
        {"loop",      {Method::IJK_COLLAPSED_LOOP,         Method::IJK_LOOP}},
        {"tiled",     {Method::TILED_SHMEM_MEM_DIRECTIVES, Method::TILED_SHMEM,   Method::TILED_K,            Method::TILED_SHMEM_NO_BANK_CONFLICT}},
};

namespace Target {
    namespace Basic {

        double multiplyIJK(const DT *A, const DT *B, DT *C) {
            double start, end;
#pragma omp target data map(A[0:SIZE*SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
            {
                start = omp_get_wtime();
#pragma omp target teams distribute parallel for shared(A, B, C) schedule(static) default(none)
                for (int i = 0; i < SIZE; i++) {
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

        double multiplyIJKCollapsed(const DT *A, const DT *B, DT *C) {
            double start, end;
#pragma omp target data map(A[0:SIZE*SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
            {
                start = omp_get_wtime();
#pragma omp target teams distribute parallel for collapse(2) shared(A, B, C) schedule(static) default(none)
                for (int i = 0; i < SIZE; i++) {
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

        double multiplyIKJ(const DT *A, const DT *B, DT *C) {
            double start, end;
#pragma omp target data map(A[0:SIZE * SIZE], B[0:SIZE *  SIZE]) map(tofrom:C[0:SIZE * SIZE])
            {
                start = omp_get_wtime();
#pragma omp target teams distribute parallel for shared(A, B, C) schedule(static) default(none)
                for (int i = 0; i < SIZE; i++) {
                    for (int k = 0; k < SIZE; k++) {
                        for (int j = 0; j < SIZE; j++) {
                            C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                        }
                    }
                }
                end = omp_get_wtime();
            }
            return (end - start) * 1000.0;
        }

#if !NO_NESTED_PARALLEL_FOR
        double multiplyIKJCollapsed(const DT *A, const DT *B, DT *C) {
            double start, end;
#pragma omp target data map(A[0:SIZE * SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
            {
                start = omp_get_wtime();
#pragma omp target teams distribute parallel for shared(A, B, C) schedule(static) default(none)
                for (int i = 0; i < SIZE; i++) {
#pragma omp parallel for collapse(2)
                    for (int k = 0; k < SIZE; k++) {
                        for (int j = 0; j < SIZE; j++) {
                            C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                        }
                    }
                }
                end = omp_get_wtime();
            }
            return (end - start) * 1000.0;
        }
#endif

        double multiplyJIK(const DT *A, const DT *B, DT *C) {
            double start, end;
#pragma omp target data map(A[0:SIZE * SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
            {
                start = omp_get_wtime();
#pragma omp target teams distribute parallel for shared(A, B, C) schedule(static) default(none)
                for (int j = 0; j < SIZE; ++j) {
                    for (int i = 0; i < SIZE; ++i) {
                        for (int k = 0; k < SIZE; ++k) {
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
                for (int j = 0; j < SIZE; ++j) {
                    for (int i = 0; i < SIZE; ++i) {
                        for (int k = 0; k < SIZE; ++k) {
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
                for (int j = 0; j < SIZE; ++j) {
                    for (int k = 0; k < SIZE; ++k) {
                        for (int i = 0; i < SIZE; ++i) {
                            C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                        }
                    }
                }
                end = omp_get_wtime();
            }
            return (end - start) * 1000.0;
        }

#if !NO_NESTED_PARALLEL_FOR
        double multiplyJKICollapsed(const DT *A, const DT *B, DT *C) {
            double start, end;
#pragma omp target data map(A[0:SIZE * SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
            {
                start = omp_get_wtime();
#pragma omp target teams distribute parallel for shared(A, B, C) schedule(static) default(none)
                for (int j = 0; j < SIZE; ++j) {
#pragma omp parallel for collapse(2)
                    for (int k = 0; k < SIZE; ++k) {
                        for (int i = 0; i < SIZE; ++i) {
                            C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                        }
                    }
                }
                end = omp_get_wtime();
            }
            return (end - start) * 1000.0;
        }
#endif
    }

#if !NO_LOOP_DIRECTIVES
    namespace Loop {
        double multiplyIJKCollapsedLoop(const DT *A, const DT *B, DT *C) {
            double start, end;
#pragma omp target data map(A[0:SIZE*SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
            {
                start = omp_get_wtime();
#pragma omp target teams loop collapse(2) shared(A, B, C) default(none)
                for (int i = 0; i < SIZE; i++) {
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

        double multiplyIJKCompleteLoop(const DT *A, const DT *B, DT *C) {
            double start, end;
#pragma omp target data map(A[0:SIZE*SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
            {
                start = omp_get_wtime();
#pragma omp target teams loop
                for (int i = 0; i < SIZE; i++) {
#pragma omp loop //bind(parallel)
                    for (int j = 0; j < SIZE; j++) {
#pragma omp loop bind(thread)
                        for (int k = 0; k < SIZE; k++) {
                            C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                        }
                    }
                }
                end = omp_get_wtime();
            }
            return (end - start) * 1000.0;
        }
    }
#endif

    namespace Tiled {
        /**
         * Tiled matrix-matrix multiplication
         *
         * Uses a CUDA-like code to achieve a relatively high performance
         * CUDA code from: https://github.com/ZePedroResende/MatrixMultiplication/blob/master/CUDA/src/cuda.cu
         *
         * uses the TILE_SIZE to calculate the TILE_AXIS_SIZE: TILE_AXIS_SIZE = MATRIX_SIZE / TILE_SIZE
         * (requires MATRIX_SIZE to be a multiple of TILE_SIZE)
         *
         * Splits up the work on TILE_AXIS_SIZE * TILE_AXIS_SIZE teams (= CUDA blocks)
         * each team has TILE_SIZE * TILE_SIZE threads
         *
         * Each team calculates one tile of the result matrix C.
         * A thread in the team copies a tile of A and B into shared memory and calculates a temporary c value based on
         * the currently stored blocks. This is done for all blocks in a row / column of A / B and afterwards the total
         * value for C is written back.
         */
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

        /**
         *
         */
        double multiplyTiledK(const DT *A, const DT *B, DT *C) {
            double start, end;
#pragma omp target data map(to: A[:SIZE * SIZE], B[:SIZE * SIZE]) map(tofrom : C[:SIZE * SIZE])
            {
                start = omp_get_wtime();

#pragma omp target teams distribute num_teams(2048) thread_limit(512)
                for (int i = 0; i < SIZE; ++i) {
                    int in = i * SIZE;
                    double a_tmp[K_BLOCK_SIZE];
                    for (int kk = 0; kk < SIZE; kk += K_BLOCK_SIZE) {
                        int kk_upper = ((kk + K_BLOCK_SIZE) < SIZE) ? (kk + K_BLOCK_SIZE) : SIZE;

#pragma omp parallel for schedule(static, 1)
                        for (int k = kk; k < kk_upper; ++k) {
                            a_tmp[k - kk] = A[in + k];
                        }

#pragma omp parallel for schedule(static, 1)
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

    double multiplyTiledNoBankConflict(const DT *A, const DT *B, DT *C) {
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
                    DT a_shm[TILE_SIZE][TILE_SIZE + 1];
                    DT b_shm[TILE_SIZE][TILE_SIZE + 1];

#pragma omp parallel num_threads(TILE_SIZE * TILE_SIZE) shared(A, B, C, a_shm, b_shm, blockIdx, blockIdy) default(none)
                    {
                        // from here on the code resembles CUDA
                        DT tmp_c = 0;

                        int threadNum = omp_get_thread_num();

                        int threadIdx = threadNum % TILE_SIZE;
                        int threadIdy = threadNum / TILE_SIZE;

                        int row = blockIdy * TILE_SIZE + threadIdy;
                        int col = blockIdx * TILE_SIZE + threadIdx;

                        // loop over an entire row / column of tiles
                        for (int k = 0; k < TAS; k++) {
                            // read the data to shared memory, each thread stores 1 element in a_shm and 1 in b_shm
                            a_shm[threadIdy][threadIdx] = A[row * SIZE + k * TILE_SIZE + threadIdx];
                            b_shm[threadIdy][threadIdx] = B[(k * TILE_SIZE + threadIdy) * SIZE + col];
#pragma omp barrier
                            // perform the calculations on the previously stored shared memory
                            for (int n = 0; n < TILE_SIZE; n++) {
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
                for (int blockIdx = 0; blockIdx < TAS; blockIdx++) {
                    for (int blockIdy = 0; blockIdy < TAS; blockIdy++) {
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

                            for (int k = 0; k < TAS; k++) {
                                // read the data to shared memory, each thread stores 1 element in a_shm and 1 in b_shm
                                a_shm[threadIdy * TILE_SIZE + threadIdx] = A[row * SIZE + k * TILE_SIZE + threadIdx];
                                b_shm[threadIdy * TILE_SIZE + threadIdx] = B[(k * TILE_SIZE + threadIdy) * SIZE + col];
#pragma omp barrier

                                /*
                                 * Using OpenMP 5.1 it might be possible to unroll this loop for more speedup.
                                 * (see https://www.openmp.org/wp-content/uploads/OpenMP_SC20_Loop_Transformations.pdf)
                                 */

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
    }
}
