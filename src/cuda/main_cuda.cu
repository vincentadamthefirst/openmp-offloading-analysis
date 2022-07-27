#ifndef SIZE
#define SIZE 8192
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

#ifndef CHECK
#define CHECK false
#endif

#define TAS (SIZE / TILE_SIZE)

#include "../../include/helper.hpp"
#include "../cublas/cuda_helper.cuh"
#include <iostream>
#include <chrono>
#include <vector>

__global__ void matmul(double* A, double* B, double* C) {
    float c_tmp = 0;

    int Row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int Col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float a_shm[TILE_SIZE][TILE_SIZE];
    __shared__ float b_shm[TILE_SIZE][TILE_SIZE];

    for (int k = 0; k < TAS; k++) {
        a_shm[threadIdx.y][threadIdx.x] = A[Row * SIZE + k * TILE_SIZE + threadIdx.x];
        b_shm[threadIdx.y][threadIdx.x] = B[(k * TILE_SIZE + threadIdx.y) * SIZE + Col];

        __syncthreads();

        for (int n = 0; n < TILE_SIZE; ++n)
            c_tmp += a_shm[threadIdx.y][n] * b_shm[n][threadIdx.x];

        __syncthreads();
    }

    if (Row < SIZE && Col < SIZE)
        C[((blockIdx.y * blockDim.y + threadIdx.y) * SIZE) + (blockIdx.x * blockDim.x)+ threadIdx.x] = c_tmp;
}

void multiplyIKJ(double *A, double *B, double *C, unsigned size) {
    int i, j, k;
    for (i = 0; i < size; i++) {
        for (k = 0; k < size; k++) {
            for (j = 0; j < size; j++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

int main() {
    double* h_A = Helper::Matrix::initializeRandom<double>(SIZE, -1, 1);
    double* h_B = Helper::Matrix::initializeRandom<double>(SIZE, -1, 1);
    double* h_C = Helper::Matrix::initializeZero<double>(SIZE);

    double *d_A, *d_B, *d_C;

    CHECK_CUDA(cudaMalloc((void**) &d_A, SIZE * SIZE * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**) &d_B, SIZE * SIZE * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**) &d_C, SIZE * SIZE * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, SIZE * SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, SIZE * SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, SIZE * SIZE * sizeof(double), cudaMemcpyHostToDevice));

    dim3 grid(TAS, TAS);
    dim3 block(TILE_SIZE, TILE_SIZE);

#if CHECK
    std::cout << "Performing check..." << std::endl;
    double* check_result = Helper::Matrix::initializeZero<double>(SIZE);

    std::cout << "Preparing host..." << std::endl;
    multiplyIKJ(h_A, h_B, check_result, SIZE);

    std::cout << "Preparing device..." << std::endl;
    matmul<<<grid, block>>>(d_A, d_B, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C, d_C, SIZE * SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    std::cout << "Done. Comapring..." << std::endl;

    std::cout << "RESULT = " << (Helper::Matrix::compare<double>(check_result, h_C, SIZE) ? "OK" : "NOT OK") << std::endl;
    std::cout << std::endl;

#endif

    std::vector<double> runtimes;

    for (int r = 0; r < 11; r++) {
        std::cout << "repetition #" << r << std::endl;

        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
        t1 = std::chrono::high_resolution_clock::now();

        matmul<<<grid, block>>>(d_A, d_B, d_C);
        CHECK_CUDA(cudaDeviceSynchronize());

        t2 = std::chrono::high_resolution_clock::now();

        CHECK_CUDA(cudaMemcpy(h_C, d_C, SIZE * SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        runtimes.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    }

    auto meanExecTimeMs = Helper::Math::calculateMean(runtimes);
    auto medianExecTimeMs = Helper::Math::calculateMedian(runtimes);

    auto meanGflops = Helper::Math::msToGFLOPs(meanExecTimeMs, SIZE);
    auto medianGflops = Helper::Math::msToGFLOPs(std::get<0>(medianExecTimeMs), SIZE);

    std::cout << "CUDA: " << "AVG=" << meanExecTimeMs << "ms, (" << meanGflops
              << " GFLOP/s) & MED=" << std::get<0>(medianExecTimeMs) << "ms, (" << medianGflops << " GFLOP/s)"
              << std::endl;
}