#include <iostream>
#include <omp.h>

#define SIZE 16777216 // 2^24
#define THREADS_PER_TEAM 512
#define NUM_TEAMS 32768 // = SIZE / THREADS_PER_TEAM

void copy(double* A, double* B) {
#pragma omp target data map(to:A[0:SIZE]) map(tofrom:B[0:SIZE])
    {
        size_t i, j;
#pragma omp target teams distribute num_teams(NUM_TEAMS) thread_limit(THREADS_PER_TEAM)
        for (i = 0; i < SIZE; i += THREADS_PER_TEAM) {
#pragma omp parallel for num_threads(THREADS_PER_TEAM)
            for (j = i; j < i + THREADS_PER_TEAM; j++) {
                B[j] = A[j];
            }
        }
    }
}

int main() {
    double* A = (double*) malloc(SIZE * sizeof(double));
    double* B = (double*) malloc(SIZE * sizeof(double));

    std::cout << "SIZE        = " << SIZE << std::endl;
    std::cout << "NUM_TEAMS   = " << NUM_TEAMS << std::endl;
    std::cout << "NUM_THREADS = " << THREADS_PER_TEAM << std::endl;

    for (uint64_t i = 0; i < SIZE; i++) {
        A[i] = i;
    }

    copy(A, B);

    bool correct = true;
    for (uint64_t i = 0; i < SIZE; i++) {
        if (std::abs(A[i] - B[i]) > 0.0001) {
            correct = false;
            break;
        }
    }

    std::cout << "copy was " << (correct ? "CORRECT" : "NOT CORRECT") << std::endl;
}