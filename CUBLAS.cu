#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdlib>

#include "include/Helper.h"

#include "cublas_v2.h"
#include "curand.h"

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 4096
#endif

#ifndef REPETITIONS
#define REPETITIONS 1
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

double MatMulCUBLAS(cublasHandle_t& handle, const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda = m, ldb = k, ldc = m;
    float alpha = 1;
    const float beta = 0;

    auto t1 = std::chrono::high_resolution_clock::now();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    cudaThreadSynchronize();
    //cudaDeviceReset();
    auto t2 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

int main() {
    //auto A = Helper::Matrix::initializeRandom<float>(MATRIX_SIZE, 0, 1);
    //auto B = Helper::Matrix::initializeRandom<float>(MATRIX_SIZE, 0, 1);

    uint64_t totalSize = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    float *d_A, *d_B;
    gpuErrchk(cudaMalloc(&d_A, totalSize));
    gpuErrchk(cudaMalloc(&d_B, totalSize));

    GPU_fill_rand(d_A, MATRIX_SIZE, MATRIX_SIZE);
    GPU_fill_rand(d_B, MATRIX_SIZE, MATRIX_SIZE);
    //gpuErrchk(cudaMemcpy(d_A, A, totalSize, cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpy(d_B, B, totalSize, cudaMemcpyHostToDevice));

    std::ofstream fileStream;
    fileStream.open("/home/s2743705/openmp/matmul/new_output.txt", std::ios_base::app);

    std::cout << "Begin timings." << std::endl;

    cublasHandle_t handle;
    cublasCreate(&handle);

    std::vector<double> runtimes;
    for (size_t repetition = 0; repetition < REPETITIONS; repetition++) {
        float *d_C;
        gpuErrchk(cudaMalloc(&d_C, totalSize));
        runtimes.push_back(MatMulCUBLAS(handle, d_A, d_B, d_C, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE));
//        auto h_C = Helper::Matrix::initializeZero<float>(16);
//        gpuErrchk(cudaMemcpy(h_C, d_C, 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost));
//        free(h_C);
        cudaFree(d_C);
    }

    for (const auto value : runtimes) {
        std::cout << value << "ms, ";
    }
    std::cout << std::endl;

    cublasDestroy(handle);

    std::cout << "Timings done." << std::endl;

    auto timeAvg = Helper::Math::calculateMean(runtimes);
    auto timeMed = Helper::Math::calculateMedian(runtimes);

    fileStream << "[CUBLAS] " << "MED= " << timeMed << "ms (" << Helper::Math::msToGFLOPs(timeMed, MATRIX_SIZE)
               << " GFLOP/s), AVG= " << timeAvg << "(" << Helper::Math::msToGFLOPs(timeAvg, MATRIX_SIZE)
               << " GFLOP/s) ms" << std::endl;

    std::cout << "[CUBLAS] " << "MED= " << timeMed << "ms (" << Helper::Math::msToGFLOPs(timeMed, MATRIX_SIZE)
              << " GFLOP/s), AVG= " << timeAvg << "(" << Helper::Math::msToGFLOPs(timeAvg, MATRIX_SIZE)
              << " GFLOP/s) ms" << std::endl;


    cudaFree(d_A);
    cudaFree(d_B);
}