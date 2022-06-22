#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
#pragma once

#include <cstdint>
#include <omp.h>
#include <chrono>
#include <sys/time.h>

#define BLKDIM 16
#define BLOCK_SIZE_K 1024

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 32
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif


#define THREADS_PER_BLOCK BLOCK_SIZE * BLOCK_SIZE

//#define BLOCK_UPPER MATRIX_SIZE / BLOCK_SIZE
#define BLOCK_UPPER 2

namespace Target {

    template<typename T>
    double multiplyJIKSharedMemory(T *A, T *B, T *C, uint32_t size) {
#pragma omp target teams device(0) map(to:size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size]) \
    default(none) shared(A, B, C, size)
        {
            T ashm[BLKDIM][BLKDIM];
            T bshm[BLKDIM][BLKDIM];

#pragma omp distribute collapse(2)
            for (size_t j = 0; j < size / BLKDIM; ++j) {
                for (size_t i = 0; i < size / BLKDIM; ++i) {
#pragma omp parallel default(none) shared(A, B, C, size, ashm, bshm, i, j)
                    {
                        auto threadNum = omp_get_thread_num();
                        auto threadRow = threadNum % BLKDIM;
                        auto threadCol = threadNum / BLKDIM;

                        auto blockRow = i * BLKDIM;
                        auto blockCol = j * BLKDIM;

                        auto actualRow = threadRow + blockRow;
                        auto actualCol = threadCol + blockCol;

                        float cValue = C[actualRow * size + actualCol];
                        for (size_t k = 0; k < size / BLKDIM; ++k) {
                            ashm[threadCol][threadRow] = A[(k * 16 + threadCol) * size + actualRow];
                            bshm[threadCol][threadRow] = B[actualCol * size + (k * 16 + threadRow)];
#pragma omp barrier
                            cValue += ashm[0x0][threadRow] * bshm[threadCol][0x0];
                            cValue += ashm[0x1][threadRow] * bshm[threadCol][0x1];
                            cValue += ashm[0x2][threadRow] * bshm[threadCol][0x2];
                            cValue += ashm[0x3][threadRow] * bshm[threadCol][0x3];
                            cValue += ashm[0x4][threadRow] * bshm[threadCol][0x4];
                            cValue += ashm[0x5][threadRow] * bshm[threadCol][0x5];
                            cValue += ashm[0x6][threadRow] * bshm[threadCol][0x6];
                            cValue += ashm[0x7][threadRow] * bshm[threadCol][0x7];
                            cValue += ashm[0x8][threadRow] * bshm[threadCol][0x8];
                            cValue += ashm[0x9][threadRow] * bshm[threadCol][0x9];
                            cValue += ashm[0xa][threadRow] * bshm[threadCol][0xa];
                            cValue += ashm[0xb][threadRow] * bshm[threadCol][0xb];
                            cValue += ashm[0xc][threadRow] * bshm[threadCol][0xc];
                            cValue += ashm[0xd][threadRow] * bshm[threadCol][0xd];
                            cValue += ashm[0xe][threadRow] * bshm[threadCol][0xe];
                            cValue += ashm[0xf][threadRow] * bshm[threadCol][0xf];
#pragma omp barrier
                        }
                        C[actualRow * size + actualCol] += cValue;
                    } // parallel region
                } // i loop
            } // j loop
        } // target teams
    }

    template<typename T>
    double multiplyIJK(T *A, T *B, T *C, uint32_t size) {
        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
#pragma omp target data device(0) map(to:size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size])
        {
            t1 = std::chrono::high_resolution_clock::now();
#pragma omp target teams distribute parallel for shared(A, B, C, size) schedule(static) default(none)
            for (size_t i = 0; i < size; i++) {
                for (size_t j = 0; j < size; j++) {
                    for (size_t k = 0; k < size; k++) {
                        C[i * size + j] += A[i * size + k] * B[k * size + j];
                    }
                }
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t2 - t1).count();
    }

    template<typename T>
    double multiplyIJKCollapsed(T *A, T *B, T *C, uint32_t size) {
        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
#pragma omp target data device(0) map(to:size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size])
        {
            t1 = std::chrono::high_resolution_clock::now();
#pragma omp target teams distribute parallel for collapse(2) shared(A, B, C, size) schedule(static) default(none)
            for (size_t i = 0; i < size; i++) {
                for (size_t j = 0; j < size; j++) {
                    for (size_t k = 0; k < size; k++) {
                        C[i * size + j] += A[i * size + k] * B[k * size + j];
                    }
                }
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t2 - t1).count();
    }

    template<typename T>
    double multiplyIJKCollapsed3(T *A, T *B, T *C, uint32_t size) {
        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
#pragma omp target data device(0) map(to:size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size])
        {
            t1 = std::chrono::high_resolution_clock::now();
#pragma omp target teams distribute parallel for collapse(3) shared(A, B, C, size) schedule(static) default(none) // TODO collapse 3
            for (size_t i = 0; i < size; i++) {
                for (size_t j = 0; j < size; j++) {
                    for (size_t k = 0; k < size; k++) {
                        C[i * size + j] += A[i * size + k] * B[k * size + j];
                    }
                }
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t2 - t1).count();
    }

    template<typename T>
    double multiplyIJKBlocked(T *A, T *B, T *C, uint32_t size, uint32_t blockSize) {
        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
#pragma omp target data map(to:size, blockSize, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size])
        {
            t1 = std::chrono::high_resolution_clock::now();
#pragma omp target teams distribute shared(A, B, C, size, blockSize)
            for (size_t i = 0; i < size; i++) {
                T a_tmp[BLOCK_SIZE_K];
                for (size_t kk = 0; kk < size; kk += BLOCK_SIZE_K) {
                    int kk_upper = ((kk + BLOCK_SIZE_K) < size) ? (kk + BLOCK_SIZE_K) : size;

#pragma omp parallel for schedule(static, 1)
                    for (size_t k = kk; k < kk_upper; ++k) {
                        a_tmp[k - kk] = A[i * size + k];
                    }
#pragma omp parallel for schedule(static, 1)
                    for (size_t j = 0; j < size; j++) {
                        T tmp = 0;
                        for (size_t k = kk; k < kk_upper; k++) {
                            tmp += a_tmp[k - kk] * B[k * size + j];
                        }
                        C[i * size + j] += tmp;
                    }
                }
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t2 - t1).count();
    }

    template<typename T>
    double multiplyIJKBlocked3(T *A, T *B, T *C, uint32_t size, uint32_t blockSize) {
        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
#pragma omp target data map(to:size, blockSize, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size])
        {
            t1 = std::chrono::high_resolution_clock::now();

#pragma omp target teams shared(A, B, C, size, blockSize)
            {
                T a_shm[BLOCK_SIZE][BLOCK_SIZE];
                T b_shm[BLOCK_SIZE][BLOCK_SIZE];

#pragma omp distribute collapse(2)
                for (size_t block_i = 0; block_i < BLOCK_UPPER; block_i++) {
                    for (size_t block_j = 0; block_j < BLOCK_UPPER; block_j++) {
#pragma omp parallel num_threads(BLOCK_SIZE * BLOCK_SIZE) shared(A, B, C, size, blockSize, a_shm, b_shm)
                        {
                            auto threadNum = omp_get_thread_num();

                            auto threadIdx = threadNum % BLOCK_SIZE;
                            auto threadIdy = threadNum / BLOCK_SIZE;

                            auto blockIdx = block_i * BLOCK_SIZE;
                            auto blockIdy = block_j * BLOCK_SIZE;

                            auto index_i = blockIdx + threadIdx;
                            auto index_j = blockIdy + threadIdy;

                            T tmp_c = C[index_i * size + index_j];

                            auto begin_a = size * BLOCK_SIZE * blockIdy;
                            auto end_a = begin_a + size - 1;
                            auto begin_b = BLOCK_SIZE * blockIdx;

                            tmp_c = 0;

                            for (size_t i = begin_a, j = begin_b; i <= end_a; i += BLOCK_SIZE, j+= (BLOCK_SIZE * size)) {
                                a_shm[threadIdy][threadIdx] = A[i + size * threadIdy + threadIdx];
                                b_shm[threadIdx][threadIdy] = B[j + size * threadIdx + threadIdy];

#pragma omp barrier
                                for (size_t k = 0; k < BLOCK_SIZE; k++) {
                                    tmp_c += a_shm[threadIdy][k] * b_shm[k][threadIdx];
                                }
#pragma omp barrier
                            }

                            C[size * BLOCK_SIZE * blockIdy + BLOCK_SIZE * blockIdx + size * threadIdy + threadIdx] = tmp_c;

//                            for (size_t k = 0; k < BLOCK_SIZE; k++) {
//                                a_shm[threadIdy][threadIdx] = A[index_i * size + (k + threadIdy)];
//                                b_shm[threadIdy][threadIdx] = B[(k + threadIdx) * size + index_j];
//#pragma omp barrier
//                                tmp_c += a_shm[k][threadIdx] * b_shm[threadIdy][k];
//#pragma omp barrier
////
//                            }
////
//                            C[index_i * size + index_j] += tmp_c;
                        }
                    }
                }
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t2 - t1).count();
    }

    template<typename T>
    double multiplyIJKBlocked2(T *A, T *B, T *C, uint32_t size, uint32_t blockSize) {
        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
#pragma omp target data map(to:size, blockSize, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size])
        {
            t1 = std::chrono::high_resolution_clock::now();

#pragma omp target teams parallel for collapse(2)
            for (size_t block_i = 0; block_i < MATRIX_SIZE; block_i += BLOCK_SIZE) {
                for (size_t block_j = 0; block_j < MATRIX_SIZE; block_j += BLOCK_SIZE) {

#pragma omp parallel for collapse(2) schedule(static, 1)
                    for (size_t i = block_i; i < BLOCK_SIZE; i++) {
                        for (size_t j = block_j; j < BLOCK_SIZE; j++) {
                            for (size_t k = 0; k < MATRIX_SIZE; k++) {
                                C[i * size + j] += A[i * size + k] * B[k * size + j];
                            }
                        }
                    }
                }
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t2 - t1).count();
    }



    template<typename T>
    double multiplyIKJBlocked(T *A, T *B, T *C, uint32_t size, uint32_t blockSize) {
        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
#pragma omp target data map(to:size, blockSize, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size])
        {
            t1 = std::chrono::high_resolution_clock::now();
#pragma omp target teams distribute shared(A, B, C, size, blockSize)
            for (size_t i = 0; i < size; i++) {
                T a_tmp[BLOCK_SIZE_K];
                for (size_t kk = 0; kk < size; kk += BLOCK_SIZE_K) {
                    int kk_upper = ((kk + BLOCK_SIZE_K) < size) ? (kk + BLOCK_SIZE_K) : size;

#pragma omp parallel for schedule(static, 1)
                    for (size_t k = kk; k < kk_upper; ++k) {
                        a_tmp[k - kk] = A[i * size + k];
                    }
#pragma omp parallel for schedule(static, 1)
                    for (size_t k = kk; k < kk_upper; k++) {
                        for (size_t j = 0; j < size; j++) {
                            C[i * size + j] += a_tmp[k - kk] * B[k * size + j];
                        }
                    }
                }
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t2 - t1).count();
    }

    template<typename T>
    double multiplyIKJ(T *A, T *B, T *C, uint32_t size) {
        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
#pragma omp target data device(0) map(to:size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size])
        {
            t1 = std::chrono::high_resolution_clock::now();
#pragma omp target teams distribute parallel for shared(A, B, C, size) schedule(static) default(none)
            for (size_t i = 0; i < size; i++) {
                for (size_t k = 0; k < size; k++) {
                    for (size_t j = 0; j < size; j++) {
                        C[i * size + j] += A[i * size + k] * B[k * size + j];
                    }
                }
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t2 - t1).count();
    }

    template<typename T>
    double multiplyIKJCollapsed(T *A, T *B, T *C, uint32_t size) {
        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
#pragma omp target data device(0) map(to:size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size])
        {
            t1 = std::chrono::high_resolution_clock::now();
#pragma omp target teams distribute parallel for collapse(3) shared(A, B, C, size) schedule(static) default(none) // TODO collapse 3
            for (size_t i = 0; i < size; i++) {
                for (size_t k = 0; k < size; k++) {
                    for (size_t j = 0; j < size; j++) {
                        C[i * size + j] += A[i * size + k] * B[k * size + j];
                    }
                }
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t2 - t1).count();
    }

    template<typename T>
    double multiplyJIK(T *A, T *B, T *C, uint32_t size) {
        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
#pragma omp target data device(0) map(to:size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size])
        {
            t1 = std::chrono::high_resolution_clock::now();
#pragma omp target teams distribute parallel for shared(A, B, C, size) schedule(static) default(none)
            for (size_t j = 0; j < size; ++j) {
                for (size_t i = 0; i < size; ++i) {
                    for (size_t k = 0; k < size; ++k) {
                        C[i * size + j] += A[i * size + k] * B[k * size + j];
                    }
                }
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t2 - t1).count();
    }

    template<typename T>
    double multiplyJIKCollapsed(T *A, T *B, T *C, uint32_t size) {
        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
#pragma omp target data device(0) map(to:size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size])
        {
            t1 = std::chrono::high_resolution_clock::now();
#pragma omp target teams distribute parallel for collapse(2) shared(A, B, C, size) schedule(static) default(none) // TODO collapse 3
            for (size_t j = 0; j < size; ++j) {
                for (size_t i = 0; i < size; ++i) {
                    for (size_t k = 0; k < size; ++k) {
                        C[i * size + j] += A[i * size + k] * B[k * size + j];
                    }
                }
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t2 - t1).count();
    }

    template<typename T>
    double multiplyJKI(T *A, T *B, T *C, uint32_t size) {
        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
#pragma omp target data device(0) map(to:size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size])
        {
            t1 = std::chrono::high_resolution_clock::now();
#pragma omp target teams distribute parallel for shared(A, B, C, size) schedule(static) default(none)
            for (size_t j = 0; j < size; ++j) {
                for (size_t k = 0; k < size; ++k) {
                    for (size_t i = 0; i < size; ++i) {
                        C[i * size + j] += A[i * size + k] * B[k * size + j];
                    }
                }
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t2 - t1).count();
    }

    template<typename T>
    double multiplyJKICollapsed(T *A, T *B, T *C, uint32_t size) {
        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
#pragma omp target data device(0) map(to:size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size])
        {
            t1 = std::chrono::high_resolution_clock::now();
#pragma omp target teams distribute parallel for collapse(2) shared(A, B, C, size) schedule(static) default(none) // TODO collapse 3
            for (size_t j = 0; j < size; ++j) {
                for (size_t k = 0; k < size; ++k) {
                    for (size_t i = 0; i < size; ++i) {
                        C[i * size + j] += A[i * size + k] * B[k * size + j];
                    }
                }
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
}
#pragma clang diagnostic pop