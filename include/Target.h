#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
#pragma once

#include <cstdint>
#include <omp.h>

#define BLKDIM 16

namespace Target {

    template<typename T>
    void multiplyJIKSharedMemory(T *A, T *B, T *C, uint32_t size) {
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
                    C[actualRow * size + actualCol] = cValue;
                } // parallel region
            } // i loop
        } // j loop
        } // target teams
    }

    template<typename T>
    void multiplyIKJ(T *A, T *B, T *C, uint32_t size) {
// #pragma omp target teams distribute parallel for map(to: A[0:size*size], B[0:size * size], size) shared(A, B, size) map(tofrom: C[0:size * size])

#pragma omp target teams device(0) map(to:size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size]) \
    default(none) shared(A, B, C, size)
#pragma omp distribute parallel for shared(A, B, C, size) schedule(static)
        for (size_t i = 0; i < size; i++) {
            for (size_t k = 0; k < size; k++) {
                for (size_t j = 0; j < size; j++) {
                    C[i * size + j] += A[i * size + k] * B[k * size + j];
                }
            }
        }
    }

    template<typename T>
    void multiplyIKJCollapsed(T *A, T *B, T *C, uint32_t size) {
#pragma omp target teams device(0) map(to:size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size]) \
    default(none) shared(A, B, C, size)
#pragma omp distribute parallel for collapse(2) shared(A, B, C, size) schedule(static)
        for (size_t i = 0; i < size; i++) {
            for (size_t k = 0; k < size; k++) {
                for (size_t j = 0; j < size; j++) {
                    C[i * size + j] += A[i * size + k] * B[k * size + j];
                }
            }
        }
    }

    template<typename T>
    void multiplyIJK(T *A, T *B, T *C, uint32_t size) {
#pragma omp target teams device(0) map(to:size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size]) \
    default(none) shared(A, B, C, size)
#pragma omp distribute parallel for shared(A, B, C, size) schedule(static)
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                for (size_t k = 0; k < size; k++) {
                    C[i * size + j] += A[i * size + k] * B[k * size + j];
                }
            }
        }
    }

    template<typename T>
    void multiplyIJKCollapsed(T *A, T *B, T *C, uint32_t size) {
#pragma omp target teams device(0) map(to:size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size]) \
    default(none) shared(A, B, C, size)
#pragma omp distribute parallel for collapse(2) shared(A, B, C, size) schedule(static)
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                for (size_t k = 0; k < size; k++) {
                    C[i * size + j] += A[i * size + k] * B[k * size + j];
                }
            }
        }
    }

    template<typename T>
    void multiplyJIK(T *A, T *B, T *C, uint32_t size) {
#pragma omp target teams device(0) map(to:size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size]) \
    default(none) shared(A, B, C, size)
//#pragma omp distribute parallel for num_threads(512) dist_schedule(static, 512) collapse(2) shared(A, B, C, size)
#pragma omp distribute parallel for shared(A, B, C, size) schedule(static)
        for (size_t j = 0; j < size; ++j) {
            for (size_t i = 0; i < size; ++i) {
                for (size_t k = 0; k < size; ++k) {
                    C[i * size + j] += A[i * size + k] * B[k * size + j];
                }
            }
        }
    }

    template<typename T>
    void multiplyJIKCollapsed(T *A, T *B, T *C, uint32_t size) {
#pragma omp target teams device(0) map(to:size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size]) \
    default(none) shared(A, B, C, size)
//#pragma omp distribute parallel for num_threads(512) dist_schedule(static, 512) collapse(2) shared(A, B, C, size)
#pragma omp distribute parallel for collapse(2) shared(A, B, C, size) schedule(static)
        for (size_t j = 0; j < size; ++j) {
            for (size_t i = 0; i < size; ++i) {
                for (size_t k = 0; k < size; ++k) {
                    C[i * size + j] += A[i * size + k] * B[k * size + j];
                }
            }
        }
    }

    template<typename T>
    void multiplyJKI(T *A, T *B, T *C, uint32_t size) {
#pragma omp target teams device(0) map(to:size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size]) \
    default(none) shared(A, B, C, size)
//#pragma omp distribute parallel for num_threads(512) dist_schedule(static, 512) collapse(2) shared(A, B, C, size)
#pragma omp distribute parallel for shared(A, B, C, size) schedule(static)
        for (size_t j = 0; j < size; ++j) {
            for (size_t k = 0; k < size; ++k) {
                for (size_t i = 0; i < size; ++i) {
                    C[i * size + j] += A[i * size + k] * B[k * size + j];
                }
            }
        }
    }

    template<typename T>
    void multiplyJKICollapsed(T *A, T *B, T *C, uint32_t size) {
#pragma omp target teams device(0) map(to:size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size]) \
    default(none) shared(A, B, C, size)
//#pragma omp distribute parallel for num_threads(512) dist_schedule(static, 512) collapse(2) shared(A, B, C, size)
#pragma omp distribute parallel for collapse(2) shared(A, B, C, size) schedule(static)
        for (size_t j = 0; j < size; ++j) {
            for (size_t k = 0; k < size; ++k) {
                for (size_t i = 0; i < size; ++i) {
                    C[i * size + j] += A[i * size + k] * B[k * size + j];
                }
            }
        }
    }

    template <typename T>
    void multiplyIJKBlocked(T* A, T* B, T* C, uint32_t size, uint32_t b_size) {
#pragma omp target teams device(0) map(to:size, b_size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size]) \
    default(none) shared(A, B, C, size, b_size)
#pragma omp distribute parallel for schedule(static)
        for (size_t block_i = 0; block_i < size; block_i += b_size) {
            for (size_t block_j = 0; block_j < size; block_j += b_size) {
                for (size_t block_k = 0; block_k < size; block_k += b_size) {
#pragma omp distribute collapse(2)
                    for (size_t i = block_i; i < block_i + b_size; i++) {
                        for (size_t j = block_j; j < block_j + b_size; j++) {
                            for (size_t k = block_k; k < block_k + b_size; k++) {
                                C[i * size + j] += A[i * size + k] * B[k * size + j];
                            }
                        }
                    }
                }
            }
        }
    }

    template<typename T>
    void multiplyJIKBlocked(T *A, T *B, T *C, uint32_t size, uint32_t b_size) {
#pragma omp target teams device(0) map(to:size, b_size, A[0:size*size], B[0:size * size]) map(tofrom:C[0:size * size]) \
    default(none) shared(A, B, C, size, b_size)
//#pragma omp distribute parallel for num_threads(512) dist_schedule(static, 512) collapse(2) shared(A, B, C, size)
#pragma omp distribute parallel for schedule(static)
        for (size_t block_j = 0; block_j < size; block_j += b_size) {
            for (size_t block_i = 0; block_i < size; block_i += b_size) {
                for (size_t block_k = 0; block_k < size; block_k += b_size) {
                    for (size_t j = block_j; j < block_j + b_size; ++j) {
                        for (size_t i = block_i; i < block_i + b_size; ++i) {
                            for (size_t k = block_k; k < block_k + b_size; ++k) {
                                C[i * size + j] += A[i * size + k] * B[k * size + j];
                            }
                        }
                    }
                }
            }
        }
    }

    template <typename T>
    void multiply_target_ikj(T* A, T* B, T* C, unsigned size) {
#pragma omp target teams distribute parallel for map(to: A[0:size*size], B[0:size * size], size) shared(A, B, size) map(tofrom: C[0:size * size]) schedule(static)
        for (size_t i = 0; i < size; i++) {
            for (size_t k = 0; k < size; k++) {
                for (size_t j = 0; j < size; j++) {
                    C[i * size + j] += A[i * size + k] * B[k * size + j];
                }
            }
        }
    }
}
#pragma clang diagnostic pop