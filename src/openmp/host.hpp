#pragma once

namespace Host {
    template<typename T>
    void multiplyIJK(T *A, T *B, T *C, unsigned size) {
        for (unsigned i = 0; i < size; i++) {
            for (unsigned j = 0; j < size; j++) {
                for (unsigned k = 0; k < size; k++) {
                    C[i * size + j] += A[i * size + k] * B[k * size + j];
                }
            }
        }
    }

    template<typename T>
    void multiplyIKJ(T *A, T *B, T *C, unsigned size) {
        for (unsigned i = 0; i < size; i++) {
            for (unsigned k = 0; k < size; k++) {
                for (unsigned j = 0; j < size; j++) {
                    C[i * size + j] += A[i * size + k] * B[k * size + j];
                }
            }
        }
    }

    /**
     * A parallel matrix-matrix multiplication on the host (CPU). This method is used generate comparison matrices
     * for the GPU-based implementations.
     */
    template<typename T>
    void multiplyIKJParallel(T *A, T *B, T *C, unsigned size) {
#pragma omp parallel default(none) shared(size, A, B, C)
        {
            int i, j, k;
#pragma omp for collapse(2)
            for (i = 0; i < size; i++) {
                for (k = 0; k < size; k++) {
                    for (j = 0; j < size; j++) {
                        C[i * size + j] += A[i * size + k] * B[k * size + j];
                    }
                }
            }
        }
    }
}
