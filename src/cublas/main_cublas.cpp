#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <chrono>
#include <cstdlib>

#include "../../include/helper.hpp"
#include "../../include/output.hpp"
#include "cuda_helper.cuh"
#include "../../libs/cmdparser.hpp"

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

    cudaDeviceSynchronize();
    //cudaThreadSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, A, size, B, size, &beta, C, size);
    //cudaThreadSynchronize();
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

template<>
double MatMulCUBLAS<double>(cublasHandle_t& handle, const double *A, const double *B, double *C, const int size) {
    //double alpha = 1;
    double alpha = 1;
    const double beta = 0;

    cudaDeviceSynchronize();
    //cudaThreadSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, A, size, B, size, &beta, C, size);
//    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha,
//                 A, CUDA_R_64F, size,
//                 B, CUDA_R_64F, size, &beta,
//                 C, CUDA_R_64F, size,
//                 CUBLAS_COMPUTE_64F_PEDANTIC, CUBLAS_GEMM_ALGO0);
    //cudaThreadSynchronize();
    cudaDeviceSynchronize();
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
    cublasSetMathMode(handle, tensor ? /*CUBLAS_TENSOR_OP_MATH*/ CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);

    std::cout << "Done." << std::endl;
    return handle;
}

template<>
cublasHandle_t prepareHandle<double>(bool tensor) {
    std::cout << "Preparing double precision CUBLAS handle with tensor math = " << (tensor ? "ON" : "OFF") << std::endl;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, tensor ? CUBLAS_TENSOR_OP_MATH : /*CUBLAS_DEFAULT_MATH*/ CUBLAS_PEDANTIC_MATH);

    std::cout << "Done." << std::endl;
    return handle;
}

template<typename T>
void run(std::string path, bool isCsv, bool verbose, uint32_t repetitions, uint32_t warmup, uint32_t mStart, uint32_t mEnd, bool tensor,
         unsigned long long seed) {
    auto handle = prepareHandle<T>(tensor);

    std::vector<Output::MatrixMultiplyRunResult> runResults;

    for (uint32_t matrixSize = mStart; matrixSize <= mEnd; matrixSize *= 2) {
        auto totalSize = matrixSize * matrixSize * sizeof(T);
        std::cout << "Matrices of size " << matrixSize << std::endl;

        T *d_A, *d_B;
        CHECK_CUDA(cudaMalloc(&d_A, totalSize));
        CHECK_CUDA(cudaMalloc(&d_B, totalSize));
        initMatrix(d_A, matrixSize, seed);
        initMatrix(d_B, matrixSize, seed);

        std::vector<double> runtimes; // runtimes in ms
        for (uint32_t repetition = 0; repetition < repetitions + warmup; ++repetition) {
            T *d_C;
            CHECK_CUDA(cudaMalloc(&d_C, totalSize));
            auto runtime = MatMulCUBLAS<T>(handle, d_A, d_B, d_C, matrixSize);
            if (repetition > warmup)
                runtimes.push_back(runtime);
            CHECK_CUDA(cudaFree(d_C));
            if (verbose) {
                if (repetition >= warmup) {
                    std::cout << "  Repetition #" << (repetition - warmup) << ": " << runtime << "ms ("
                              << Helper::Math::msToGFLOPs(runtime, matrixSize) << " GFLOP/s)" << std::endl;
                } else {
                    std::cout << "  Warmup #" << repetition << std::endl;
                }
            }
        }

        auto meanExecTimeMs = Helper::Math::calculateMean(runtimes);
        auto medianExecTimeMs = Helper::Math::calculateMedian(runtimes);

        auto meanGflops = Helper::Math::msToGFLOPs(meanExecTimeMs, matrixSize);
        auto medianGflops = Helper::Math::msToGFLOPs(std::get<0>(medianExecTimeMs), matrixSize);

        runResults.push_back({"cuBLAS", "1",
                              warmup, repetitions, matrixSize,
                              std::get<1>(medianExecTimeMs), std::get<2>(medianExecTimeMs),
                              meanExecTimeMs, std::get<0>(medianExecTimeMs), meanGflops, medianGflops});

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
    }

    cublasDestroy(handle);

    Output::writeOutput(std::move(path), isCsv ? Output::FileType::CSV : Output::FileType::TXT, runResults);
}

void configureParser(cli::Parser& parser) {
    Helper::IO::basicParserSetup(parser);
    parser.set_optional<int>("s", "matrix_size_start", 4096,
                             "Start size of matrices. Will be doubled until matrix_size_end.");
    parser.set_optional<int>("e", "matrix_size_end", 16384, "End size of matrices (will be included).");
    parser.set_optional<bool>("t", "tensor", false, "Explicitly activate tensor functionality.");
    parser.set_optional<std::string>("p", "precision", "f64","The data type to use on the accelerator.");
}

int main(int argc, char* argv[]) {
    cli::Parser parser(argc, argv);
    configureParser(parser);
    parser.run_and_exit_if_error();

    auto csv = parser.get<std::string>("ft") == "csv";
    auto precision = parser.get<std::string>("p");
    auto filename = parser.get<std::string>("o");
    auto verbose = parser.get<bool>("v");
    auto repetitions = parser.get<int>("r");
    auto warmup = parser.get<int>("w");
    repetitions = repetitions < 1 ? 1 : repetitions;
    auto matrixSizeStart = parser.get<int>("s");
    auto matrixSizeEnd = parser.get<int>("e");
    auto tensor = parser.get<bool>("t");
    if (parser.get<bool>("no"))
        filename = "NO_OUTPUT_FILE";

    if (precision == "f32") {
        run<float>(filename, csv, verbose, repetitions, warmup, matrixSizeStart, matrixSizeEnd, tensor,
                   (unsigned long long) clock());
    } else {
        run<double>(filename, csv, verbose, repetitions, warmup, matrixSizeStart, matrixSizeEnd, tensor,
                    (unsigned long long) clock());
    }
}
