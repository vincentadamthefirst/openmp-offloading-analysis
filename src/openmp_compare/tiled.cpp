/**
 * A simple test program to evaluate if there is a speed difference between C and CPP code (in OpenMP offloading).
 * Uses a tiled matrix multiplication with shared memory usage.
 */

#include <omp.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

#ifndef DATA_TYPE
#define DATA_TYPE double
#endif

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 16384
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

#if TILE_SIZE>32
#warning "Tile sizes > 32 will most likely lead to incorrect matrices!"
#endif

#ifndef REPETITIONS
#define REPETITIONS 11
#endif

// internally calculated, should not be overwritten
#define TILE_AMOUNT_PER_AXIS (MATRIX_SIZE / TILE_SIZE)

double tiledMatMul(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
    double start, end;
#pragma omp target data map(A[0:MATRIX_SIZE*MATRIX_SIZE], B[0:MATRIX_SIZE*MATRIX_SIZE]) map(tofrom:C[0:MATRIX_SIZE*MATRIX_SIZE])
    {
        start = omp_get_wtime();

#pragma omp target teams distribute num_teams(TILE_AMOUNT_PER_AXIS*TILE_AMOUNT_PER_AXIS) collapse(2) thread_limit(TILE_SIZE*TILE_SIZE)
        for (int blockIdx = 0; blockIdx < TILE_AMOUNT_PER_AXIS; blockIdx++) {
            for (int blockIdy = 0; blockIdy < TILE_AMOUNT_PER_AXIS; blockIdy++) {
                DATA_TYPE a_shm[TILE_SIZE * TILE_SIZE];
                DATA_TYPE b_shm[TILE_SIZE * TILE_SIZE];

#pragma omp parallel num_threads(TILE_SIZE * TILE_SIZE) shared(A, B, C, a_shm, b_shm, blockIdx, blockIdy) default(none)
                {
                    DATA_TYPE tmp_c = 0;

                    int threadNum = omp_get_thread_num();

                    int threadIdx = threadNum % TILE_SIZE;
                    int threadIdy = threadNum / TILE_SIZE;

                    int row = blockIdy * TILE_SIZE + threadIdy;
                    int col = blockIdx * TILE_SIZE + threadIdx;

                    for (size_t k = 0; k < TILE_AMOUNT_PER_AXIS; k++) {
                        a_shm[threadIdy * TILE_SIZE + threadIdx] = A[row * MATRIX_SIZE + k * TILE_SIZE + threadIdx];
                        b_shm[threadIdy * TILE_SIZE + threadIdx] = B[(k * TILE_SIZE + threadIdy) * MATRIX_SIZE + col];
#pragma omp barrier
                        for (size_t n = 0; n < TILE_SIZE; n++) {
                            tmp_c += a_shm[threadIdy * TILE_SIZE + n] * b_shm[n * TILE_SIZE + threadIdx];
                        }
#pragma omp barrier
                    }

                    C[((blockIdy * TILE_SIZE + threadIdy) * MATRIX_SIZE) + (blockIdx * TILE_SIZE) + threadIdx] = tmp_c;
                }
            }
        }

        end = omp_get_wtime();
    }

    return (end - start) * 1000.0;
}

int main() {
    DATA_TYPE* A = (DATA_TYPE*) malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(DATA_TYPE));
    DATA_TYPE* B = (DATA_TYPE*) malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(DATA_TYPE));
    DATA_TYPE* C = (DATA_TYPE*) malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(DATA_TYPE));

    // always use the same seed
    srand(0);

    // initialize the matrices
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            A[i * MATRIX_SIZE + j] = (((DATA_TYPE) rand() / (DATA_TYPE) RAND_MAX) * 2) - 1;
            B[i * MATRIX_SIZE + j] = (((DATA_TYPE) rand() / (DATA_TYPE) RAND_MAX) * 2) - 1;
        }
    }

    std::cout << "Performing " << REPETITIONS << " repetitions on " << MATRIX_SIZE << "x" << MATRIX_SIZE
              << " matrices with tile size " << TILE_SIZE << "." << std::endl;

    double totalExecTime = 0;

    for (int repetition = 0; repetition < REPETITIONS; repetition++) {
        double progress = (double) repetition / REPETITIONS;

        std::cout << "[";
        int pos = 70 * progress;
        for (int i = 0; i < 70; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();

        double execTime = tiledMatMul(A, B, C);
        totalExecTime += execTime;

        memset(C, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(DATA_TYPE));
    }

    std::cout << "[======================================================================]100 %" << std::endl;

    double meanExecTime = totalExecTime / REPETITIONS;
    double operations = (2.0 * (double) MATRIX_SIZE - 1.0) * (double) MATRIX_SIZE * (double) MATRIX_SIZE;
    double gflops =  (operations / 1000000000) / ((double) meanExecTime / 1000);

    std::cout.precision(3);
    std::cout << "Average execution time = " << std::fixed << meanExecTime << "ms (" << std::fixed << gflops
              << " GFLOP/s) " << std::endl;

    free(A);
    free(B);
    free(C);
}
