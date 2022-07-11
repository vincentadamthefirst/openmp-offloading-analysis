#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdlib>

#include "../../include/Helper.hpp"
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

        auto timeMed = std::get<0>(Helper::Math::calculateMedian(runtimes));
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

void configureParser(cli::Parser& parser) {
    parser.set_optional<int>("s", "matrix_size_start", 4096,
                             "Start size of matrices. Will be doubled until matrix_size_end.");
    parser.set_optional<int>("e", "matrix_size_end", 16384, "End size of matrices (will be included).");
    parser.set_optional<int>("r", "repetitions", 11, "Set the amount of repetitions for each matrix.");
    parser.set_optional<bool>("t", "tensor", false, "Explicitly activate tensor functionality.");
    parser.set_optional<std::string>("ft", "file_type", "txt", "Set the formatting of the output. Must be 'txt' or "
                                                               "'csv'.");
    parser.set_optional<std::string>("f", "file", "GENERATE_NEW",
                                     "File the output should be written to. If no file is given a "
                                     "new file will be generated next to the executable.");
    parser.set_optional<bool>("v", "verbose", false, "Enable verbose output.");
    parser.set_optional<std::string>("p", "precision", "f64","The data type to use on the accelerator.");
}

int main(int argc, char* argv[]) {
    cli::Parser parser(argc, argv);
    configureParser(parser);
    parser.run_and_exit_if_error();

    auto csv = parser.get<std::string>("ft") == "csv";
    auto precision = parser.get<std::string>("p");
    auto filename = parser.get<std::string>("f");
    auto verbose = parser.get<bool>("v");
    auto repetitions = parser.get<int>("r");
    repetitions = repetitions < 1 ? 1 : repetitions;
    auto matrixSizeStart = parser.get<int>("s");
    auto matrixSizeEnd = parser.get<int>("e");
    auto tensor = parser.get<bool>("t");

    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,sizeof(buffer),"%d-%m-%Y_%H:%M:%S",timeinfo);
    std::string timeString(buffer);

    std::ifstream fileCheck(filename.c_str());
    if (filename == "GENERATE_NEW" || !fileCheck.good()) {
        // no file found, create a file
        std::string newName =
                filename == "GENERATE_NEW" ? "cublas_result_" + std::to_string(matrixSizeStart) + "-" +
                                             std::to_string(matrixSizeEnd) + "_" + timeString + (csv ? ".csv" : ".txt")
                                           : filename;
        std::cout << "No output file found, generate a new one (" << newName << ")" << std::endl;

        fileCheck.close();
        std::ofstream newFile(newName);
        newFile.close();
        filename = newName;
    }

    if (precision == "f32") {
        run<float>(filename, csv, verbose, repetitions, matrixSizeStart, matrixSizeEnd, tensor,
                   (unsigned long long) clock());
    } else {
        run<double>(filename, csv, verbose, repetitions, matrixSizeStart, matrixSizeEnd, tensor,
                    (unsigned long long) clock());
    }
}
