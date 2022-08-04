#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdlib>

#include "../../include/helper.hpp"
#include "../../include/output.hpp"
#include "../../libs/cmdparser.hpp"

#include "error_macros.h"
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>

template<typename T>
void initMatrix(T *A, uint32_t size, unsigned long long seed);

template<>
void initMatrix<float>(float *A, uint32_t size, unsigned long long seed) {
    hiprandGenerator_t prng;
    hiprandCreateGenerator(&prng, HIPRAND_RNG_PSEUDO_DEFAULT);
    hiprandSetPseudoRandomGeneratorSeed(prng, seed);
    hiprandGenerateUniform(prng, A, size * size);
}

template<>
void initMatrix<double>(double *A, uint32_t size, unsigned long long seed) {
    hiprandGenerator_t prng;
    hiprandCreateGenerator(&prng, HIPRAND_RNG_PSEUDO_DEFAULT);
    hiprandSetPseudoRandomGeneratorSeed(prng, seed);
    hiprandGenerateUniformDouble(prng, A, size * size);
}

template<typename T>
double MatMulROCBLAS(rocblas_handle& handle, const T *A, const T *B, T *C, const int size);

template<>
double MatMulROCBLAS<float>(rocblas_handle& handle, const float *A, const float *B, float *C, const int size) {
    float alpha = 1;
    const float beta = 0;

    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    CHECK_ROCBLAS_STATUS(
            rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, size, size, size, &alpha, A, size, B,
                          size, &beta, C, size));
    hipDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

template<>
double MatMulROCBLAS<double>(rocblas_handle& handle, const double *A, const double *B, double *C, const int size) {
    double alpha = 1;
    const double beta = 0;

    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    CHECK_ROCBLAS_STATUS(
            rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none, size, size, size, &alpha, A, size, B,
                          size, &beta, C, size));
    hipDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

rocblas_handle prepareHandle() {
    std::cout << "Preparing ROCBLAS handle" << std::endl;

    rocblas_handle handle;
    CHECK_ROCBLAS_STATUS(rocblas_create_handle(&handle));

    std::cout << "Done." << std::endl;
    return handle;
}

template<typename T>
void run(std::string path, bool isCsv, bool verbose, uint32_t repetitions, uint32_t warmup, uint32_t mStart, uint32_t mEnd,
         unsigned long long seed) {
    auto handle = prepareHandle();

    std::vector<Output::MatrixMultiplyRunResult> runResults;

    for (uint32_t matrixSize = mStart; matrixSize <= mEnd; matrixSize *= 2) {
        auto totalSize = matrixSize * matrixSize * sizeof(T);
        std::cout << "Matrices of size " << matrixSize << std::endl;

        T *d_A, *d_B;
        CHECK_HIP_ERROR(hipMalloc(&d_A, totalSize));
        CHECK_HIP_ERROR(hipMalloc(&d_B, totalSize));
        initMatrix(d_A, matrixSize, seed);
        initMatrix(d_B, matrixSize, seed);

        std::vector<double> runtimes; // runtimes in ms
        for (uint32_t repetition = 0; repetition < repetitions + warmup; ++repetition) {
            T *d_C;
            CHECK_HIP_ERROR(hipMalloc(&d_C, totalSize));
            auto runtime = MatMulROCBLAS<T>(handle, d_A, d_B, d_C, matrixSize);
            if (repetition > warmup)
                runtimes.push_back(runtime);
            CHECK_HIP_ERROR(hipFree(d_C));
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

        runResults.push_back({"rocBLAS", "1",
                              warmup, repetitions, matrixSize,
                              std::get<1>(medianExecTimeMs), std::get<2>(medianExecTimeMs),
                              meanExecTimeMs, std::get<0>(medianExecTimeMs), meanGflops, medianGflops});

        CHECK_HIP_ERROR(hipFree(d_A));
        CHECK_HIP_ERROR(hipFree(d_B));
    }

    rocblas_destroy_handle(handle);

    Output::writeOutput(std::move(path), isCsv ? Output::FileType::CSV : Output::FileType::TXT, runResults);
}

void configureParser(cli::Parser& parser) {
    Helper::IO::basicParserSetup(parser);
    parser.set_optional<int>("s", "matrix_size_start", 4096,
                             "Start size of matrices. Will be doubled until matrix_size_end.");
    parser.set_optional<int>("e", "matrix_size_end", 16384, "End size of matrices (will be included).");
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
    if (parser.get<bool>("no"))
        filename = "NO_OUTPUT_FILE";

    if (precision == "f32") {
        run<float>(filename, csv, verbose, repetitions, warmup, matrixSizeStart, matrixSizeEnd,
                   (unsigned long long) clock());
    } else {
        run<double>(filename, csv, verbose, repetitions, warmup, matrixSizeStart, matrixSizeEnd,
                    (unsigned long long) clock());
    }
}
