#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdlib>

#include "../include/argparse.h"
#include "../include/Helper.h"
#include "cuda_helper.cuh"

#include "cublas_v2.h"
#include "curand.h"

template<typename T>
void initMatrix(T *A, uint32_t size, unsigned long long seed);

template<>
void initMatrix<float>(float *A, uint32_t size, unsigned long long seed) {
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, seed);
    curandGenerateUniform(prng, A, size * size);
}

template<>
void initMatrix<double>(double *A, uint32_t size, unsigned long long seed) {
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, seed);
    curandGenerateUniformDouble(prng, A, size * size);
}

template<typename T>
double MatMulCUBLAS(cublasHandle_t& handle, const T *A, const T *B, T *C, const int size);

template<>
double MatMulCUBLAS<float>(cublasHandle_t& handle, const float *A, const float *B, float *C, const int size) {
    float alpha = 1;
    const float beta = 0;

    //cudaDeviceSynchronize();
    cudaThreadSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, A, size, B, size, &beta, C, size);
    cudaThreadSynchronize();
    //cudaDeviceSynchronize(); // TODO look into method functionality (can it be used as a substitute as is?)
    auto t2 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

template<>
double MatMulCUBLAS<double>(cublasHandle_t& handle, const double *A, const double *B, double *C, const int size) {
    double alpha = 1;
    const double beta = 0;

    //cudaDeviceSynchronize();
    cudaThreadSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, A, size, B, size, &beta, C, size);
    cudaThreadSynchronize();
    //cudaDeviceSynchronize(); // TODO look into method functionality (can it be used as a substitute as is?)
    auto t2 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

template<typename T>
cublasHandle_t prepareHandle(bool tensor);

template<>
cublasHandle_t prepareHandle<float>(bool tensor) {
    std::cout << "Preparing single precision CUBLAS handle with tensor math = " << (tensor ? "ON" : "OFF") << std::endl;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, tensor ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);

    std::cout << "Done." << std::endl;
    return handle;
}

template<>
cublasHandle_t prepareHandle<double>(bool tensor) {
    std::cout << "Preparing double precision CUBLAS handle with tensor math = " << (tensor ? "ON" : "OFF") << std::endl;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, tensor ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);

    std::cout << "Done." << std::endl;
    return handle;
}

template<typename T>
void run(std::string path, bool isCsv, bool verbose, uint32_t repetitions, uint32_t mStart, uint32_t mEnd, bool tensor,
         unsigned long long seed) {
    auto handle = prepareHandle<T>(tensor);

    std::ofstream fileStream;
    fileStream.open(path, std::ios_base::app);

    for (uint32_t matrixSize = mStart; matrixSize <= mEnd; matrixSize *= 2) {
        auto totalSize = matrixSize * matrixSize * sizeof(T);
        std::cout << "Matrices of size " << matrixSize << std::endl;

        T *d_A, *d_B;
        CHECK_CUDA(cudaMalloc(&d_A, totalSize));
        CHECK_CUDA(cudaMalloc(&d_B, totalSize));
        initMatrix(d_A, matrixSize, seed);
        initMatrix(d_B, matrixSize, seed);

        std::vector<double> runtimes; // runtimes in ms
        for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
            T *d_C;
            CHECK_CUDA(cudaMalloc(&d_C, totalSize));
            auto runtime = MatMulCUBLAS<T>(handle, d_A, d_B, d_C, matrixSize);
            runtimes.push_back(runtime);
            CHECK_CUDA(cudaFree(d_C));
            if (verbose)
                std::cout << "  Repetition #" << repetition << ": " << runtime << "ms ("
                          << Helper::Math::msToGFLOPs(runtime, matrixSize) << " GFLOP/s)" << std::endl;
        }

        auto timeMed = Helper::Math::calculateMedian(runtimes);
        auto timeAvg = Helper::Math::calculateMean(runtimes);

        std::string info = "MED= " + std::to_string(timeMed) + "ms (" +
                           std::to_string(Helper::Math::msToGFLOPs(timeMed, matrixSize)) + " GFLOP/s) " +
                           "AVG= " + std::to_string(timeAvg) + "ms (" +
                           std::to_string(Helper::Math::msToGFLOPs(timeAvg, matrixSize)) + " GFLOP/s) ";

        if (verbose)
            std::cout << "  RESULTS: " << info << std::endl << std::endl;

        if (isCsv) {
            // TODO
        } else {
            fileStream << "[CUBLAS SIZE=" << matrixSize << ", REP=" << repetitions << ", TENSOR="
                       << (tensor ? "1" : "0") << "] "
                       << info << std::endl;
        }

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
    }

    fileStream.close();
    cublasDestroy(handle);
}

int main(int ac, char* av[]) {
    argparse::Parser parser;
    auto matrixSizeStart =
            parser.AddArg<int>("matrix_size_start", 's',
                               "The start size of matrices (will get doubled until matrix_size_end)").Default(4096);
    auto matrixSizeEnd = parser.AddArg<int>("matrix_size_end", 'e',
                                            "The end size of matrices (will be included)").Default(16384);
    auto repetitions = parser.AddArg<int>("repetitions", 'r', "Repeat each matrix computation this amount").Default(11);
    auto tensor = parser.AddFlag("tensor", 't', "Explicitly active tensor functionality");
    auto format = parser.AddArg<std::string>("format", 'f', "The formatting of the output").Options(
            {"csv", "txt"}).Default("txt");
    auto path = parser.AddArg<std::string>("file", 'p', "The file to append the output to").Required();
    auto seed = parser.AddArg<unsigned long long>("seed", "The seed to fill the random matrices with").Default(
            (unsigned long long) clock());
    auto verbose = parser.AddFlag("verbose", 'v', "Enable verbose output.");
    auto precision = parser.AddArg<std::string>("precision", "The data type to use on the accelerator").Options(
            {"f32", "f64"}).Default("f64");

    parser.ParseArgs(ac, av);

    if (*precision == "f32") {
        run<float>(*path, *format == "csv", *verbose, *repetitions, *matrixSizeStart, *matrixSizeEnd, *tensor, *seed);
    } else {
        run<double>(*path, *format == "csv", *verbose, *repetitions, *matrixSizeStart, *matrixSizeEnd, *tensor, *seed);
    }
}