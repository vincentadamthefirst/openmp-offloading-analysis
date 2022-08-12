#include "../../include/preprocessor_settings.h"
#include "../../include/helper.hpp"
#include "../../include/output.hpp"
#include "../../libs/cmdparser.hpp"
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

int main(int argc, char* argv[]) {
    cli::Parser parser(argc, argv);
    Helper::IO::basicParserSetup(parser);
    parser.run_and_exit_if_error();

    auto csv = parser.get<std::string>("ft") == "csv";
    auto filename = parser.get<std::string>("o");
    auto verbose = parser.get<bool>("v");
    auto check = parser.get<bool>("c");
    auto repetitions = parser.get<int>("r");
    auto warmup = parser.get<int>("w");
    repetitions = repetitions < 1 ? 1 : repetitions;
    if (parser.get<bool>("no"))
        filename = "NO_OUTPUT_FILE";

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

    if (check) {
        if (verbose)
            std::cout << "Performing check..." << std::endl;
        double *check_result = Helper::Matrix::initializeZero<double>(SIZE);

        if (verbose)
            std::cout << "Preparing host..." << std::endl;
        multiplyIKJ(h_A, h_B, check_result, SIZE);

        if (verbose)
            std::cout << "Preparing device..." << std::endl;
        matmul<<<grid, block>>>(d_A, d_B, d_C);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_C, d_C, SIZE * SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        if (verbose)
            std::cout << "Done. Comapring..." << std::endl;

        std::cout << "RESULT = " << (Helper::Matrix::compare<double>(check_result, h_C, SIZE) ? "OK" : "NOT OK")
                  << std::endl;
        std::cout << std::endl;
    }

    std::vector<double> runtimes;

    for (int r = 0; r < 11; r++) {
        if (verbose) {
            if (r < warmup)
                std::cout << "warmup #" << r << std::endl;
            else
                std::cout << "repetition #" << r - warmup << std::endl;
        }

        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
        t1 = std::chrono::high_resolution_clock::now();

        matmul<<<grid, block>>>(d_A, d_B, d_C);
        CHECK_CUDA(cudaDeviceSynchronize());

        t2 = std::chrono::high_resolution_clock::now();

        CHECK_CUDA(cudaMemcpy(h_C, d_C, SIZE * SIZE * sizeof(double), cudaMemcpyDeviceToHost));

        if (r >= warmup)
            runtimes.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    }

    auto meanExecTimeMs = Helper::Math::calculateMean(runtimes);
    auto medianExecTimeMs = Helper::Math::calculateMedian(runtimes);

    auto meanGflops = Helper::Math::msToGFLOPs(meanExecTimeMs, SIZE);
    auto medianGflops = Helper::Math::msToGFLOPs(std::get<0>(medianExecTimeMs), SIZE);

    Output::MatrixMultiplyRunResult result;
    result.method = "CUDA";
    result.status = "1";
    result.warmup = warmup;
    result.repetitions = repetitions;
    result.matrixSize = MATRIX_SIZE;
    result.blockSize = 0;
    result.minExecutionTimeMs = std::get<1>(medianExecTimeMs);
    result.maxExecutionTimeMs = std::get<2>(medianExecTimeMs);
    result.meanExecutionTimeMs = meanExecTimeMs;
    result.medianExecutionTimeMs = std::get<0>(medianExecTimeMs);
    result.meanGflops = meanGflops;
    result.medianGflops = medianGflops;

    Output::writeOutput(std::move(filename), csv ? Output::FileType::CSV : Output::FileType::TXT, {result});

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}
