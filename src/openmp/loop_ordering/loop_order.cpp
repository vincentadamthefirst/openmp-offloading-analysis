#include <iostream>
#include <vector>
#include <map>
#include <set>

#include "../../../include/helper.hpp"
#include "../../../include/output.hpp"
#include "../../../include/preprocessor_settings.h"
#include "../host.hpp"
#include <omp.h>

double multiplyIJK(const double *A, const double *B, double *C) {
    double start, end;

#pragma omp target data map(to:A[0:SIZE*SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
    {
        start = omp_get_wtime();

#pragma omp target teams distribute
        for (int i = 0; i < SIZE; ++i) {
#pragma omp parallel for
            for (int j = 0; j < SIZE; ++j) {
                for (int k = 0; k < SIZE; ++k) {
                    C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                }
            }
        }

        end = omp_get_wtime();
    }
    return (end - start) * 1000.0;
}

double multiplyIKJ(const double *A, const double *B, double *C) {
    double start, end;
#pragma omp target data map(to:A[0:SIZE * SIZE], B[0:SIZE *  SIZE]) map(tofrom:C[0:SIZE * SIZE])
    {
        start = omp_get_wtime();

#pragma omp target teams distribute shared(A, B, C)
        for (int i = 0; i < SIZE; i++) {
#pragma omp parallel for
            for (int k = 0; k < SIZE; k++) {
                for (int j = 0; j < SIZE; j++) {
#pragma omp atomic
                    C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                }
            }
        }
        end = omp_get_wtime();
    }
    return (end - start) * 1000.0;
}

double multiplyJIK(const double *A, const double *B, double *C) {
    double start, end;
#pragma omp target data map(to:A[0:SIZE * SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
    {
        start = omp_get_wtime();

#pragma omp target teams distribute shared(A, B, C)
        for (int j = 0; j < SIZE; ++j) {
#pragma omp parallel for
            for (int i = 0; i < SIZE; ++i) {
                for (int k = 0; k < SIZE; ++k) {
                    C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                }
            }
        }
        end = omp_get_wtime();
    }
    return (end - start) * 1000.0;
}

double multiplyJKI(const double *A, const double *B, double *C) {
    double start, end;
#pragma omp target data map(to:A[0:SIZE * SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
    {
        start = omp_get_wtime();

#pragma omp target teams distribute shared(A, B, C)
        for (int j = 0; j < SIZE; ++j) {
#pragma omp parallel for
            for (int k = 0; k < SIZE; ++k) {
                for (int i = 0; i < SIZE; ++i) {
#pragma omp atomic
                    C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                }
            }
        }
        end = omp_get_wtime();
    }
    return (end - start) * 1000.0;
}

double multiplyKIJ(const double *A, const double *B, double *C) {
    double start, end;
#pragma omp target data map(to:A[0:SIZE * SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
    {
        start = omp_get_wtime();

#pragma omp target teams distribute shared(A, B, C)
        for (int k = 0; k < SIZE; ++k) {
#pragma omp parallel for
            for (int i = 0; i < SIZE; ++i) {
                for (int j = 0; j < SIZE; ++j) {
#pragma omp atomic
                    C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                }
            }
        }
        end = omp_get_wtime();
    }

    return (end - start) * 1000.0;
}

double multiplyKJI(const double *A, const double *B, double *C) {
    double start, end;
#pragma omp target data map(to:A[0:SIZE * SIZE], B[0:SIZE * SIZE]) map(tofrom:C[0:SIZE * SIZE])
    {
        start = omp_get_wtime();

#pragma omp target teams distribute shared(A, B, C)
        for (int k = 0; k < SIZE; ++k) {
#pragma omp parallel for
            for (int j = 0; j < SIZE; ++j) {
                for (int i = 0; i < SIZE; ++i) {
#pragma omp atomic
                    C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
                }
            }
        }
        end = omp_get_wtime();
    }
    return (end - start) * 1000.0;
}



Output::MatrixMultiplyRunResult runAndTimeMethod(double (*functionPtr)(const double*, const double*, double*), std::string name, bool verbose,
                        double* A, double* B, uint32_t repetitions, uint32_t warmup, double* compare = nullptr) {
    std::vector<double> runtimes;

    auto C = Helper::Matrix::initializeZero<double>(MATRIX_SIZE);

    for (auto repetition = 0; repetition < repetitions + warmup; repetition++) {
        Helper::IO::printProgress((double) (repetition) / (repetitions + warmup),
                                  "(" + name + (repetition < warmup ? " ==WARMUP== )" : ")            "));

        auto time = functionPtr(A, B, C);
        if (repetition == 0 && compare != nullptr) {
            auto cmpResult = Helper::Matrix::compare(C, compare, MATRIX_SIZE);
            if (!cmpResult) {
                Helper::IO::printProgress((double) (repetition + 1) / (repetitions + warmup),
                                          "(" + name + " ==ABORTED== )", true);
                return {name, "0", repetitions, warmup, 0, 0, 0, 0, 0, 0, 0};
            }
        }

        Helper::IO::printProgress((double) (repetition + 1) / (repetitions + warmup), "(" + name + ")");

        if (repetition >= warmup)
            runtimes.push_back(time);

        memset(C, 0, (size_t) MATRIX_SIZE * MATRIX_SIZE * sizeof(DT));
    }

    Helper::IO::printProgress(1.0, "(" + name + ")", true);

    auto mean = Helper::Math::calculateMean(runtimes);
    auto median = Helper::Math::calculateMedian(runtimes);

    auto mean_gflops = Helper::Math::msToGFLOPs(mean, MATRIX_SIZE);
    auto median_gflops = Helper::Math::msToGFLOPs(std::get<0>(median), MATRIX_SIZE);

    if (verbose)
        std::cout << "Mean: " << mean_gflops << " GFLOPs" << std::endl;

    free(C);

    return {name, "1", repetitions, warmup, MATRIX_SIZE, 0, std::get<2>(median), std::get<1>(median), mean,
            std::get<0>(median), mean_gflops, median_gflops};
}

int main(int argc, char* argv[]) {
    // setting up the CLI parser
    cli::Parser parser(argc, argv);
    Helper::IO::basicParserSetup(parser);
    parser.run_and_exit_if_error();

    // get flags & smaller values
    auto verbose = parser.get<bool>("v");
    auto csv = parser.get<std::string>("ft") == "csv";
    auto repetitions = parser.get<int>("r");
    auto warmup = parser.get<int>("w");
    auto compare = parser.get<bool>("c");
    auto noOutput = parser.get<bool>("no");
    auto file = noOutput ? "NO_OUTPUT_FILE" : parser.get<std::string>("o");

    auto A = Helper::Matrix::initializeRandom<double>(MATRIX_SIZE, 0, 1);
    auto B = Helper::Matrix::initializeRandom<double>(MATRIX_SIZE, 0, 1);
    double *C = nullptr;
    if (compare) {
        C = Helper::Matrix::initializeZero<double>(MATRIX_SIZE);
        Host::multiplyIKJParallel(A, B, C, MATRIX_SIZE);
    }

    std::cout << MATRIX_SIZE << std::endl;

    std::vector<Output::MatrixMultiplyRunResult> res;
    res.push_back(runAndTimeMethod(multiplyIJK, "ijk", verbose, A, B, (uint32_t) repetitions, (uint32_t) warmup, C));
    res.push_back(runAndTimeMethod(multiplyIKJ, "ikj", verbose, A, B, (uint32_t) repetitions, (uint32_t) warmup, C));
    res.push_back(runAndTimeMethod(multiplyJIK, "jik", verbose, A, B, (uint32_t) repetitions, (uint32_t) warmup, C));
    res.push_back(runAndTimeMethod(multiplyJKI, "jki", verbose, A, B, (uint32_t) repetitions, (uint32_t) warmup, C));
    res.push_back(runAndTimeMethod(multiplyKIJ, "kij", verbose, A, B, (uint32_t) repetitions, (uint32_t) warmup, C));
    res.push_back(runAndTimeMethod(multiplyKJI, "kji", verbose, A, B, (uint32_t) repetitions, (uint32_t) warmup, C));

    Output::writeOutput(file, csv ? Output::FileType::CSV : Output::FileType::TXT, res);
}
