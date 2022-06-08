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
                    //C[i * size + j] += A[i * size + k] * B[k * size + j];
                    C[j * size + i] += A[k * size + i] * B[j * size + k];
                }
            }
        }
    }
}