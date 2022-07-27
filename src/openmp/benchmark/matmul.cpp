#include <iostream>
#include <cstring>

#include "matmul.hpp"

void MatrixMultiplication::runMethod(double (*functionPtr)(const DT*, const DT*, DT*), Method method) {
    auto C = Helper::Matrix::initializeZero<DT>(MATRIX_SIZE);

    std::vector<double> executionTimes;

    for (size_t repetition = 0; repetition < repetitions + warmup; repetition++) {
        if (verbose) {
            Helper::IO::printProgress((double) (repetition) / (repetitions + warmup),
                                      "(" + methodNamesMapping[method] +
                                      (repetition < warmup ? " ==WARMUP== )" : ")            "));
        }

        auto execTimeMs = functionPtr(A, B, C);

        if (checkMatrix && repetition == 0) { // only check for correctness on first run
            auto correctness = Helper::Matrix::compare<DT>(C, checkMatrix, MATRIX_SIZE);
            if (!correctness) {
                Helper::IO::printProgress((double) (repetition + 1) / (repetitions + warmup),
                                          "(" + methodNamesMapping[method] + " ==ABORTED== )", true);
                std::cerr << "Method " << methodNamesMapping[method]
                          << " did not produce correct results. Aborting." << std::endl << std::endl;
                break;
            }
        }

        if (verbose) {
            Helper::IO::printProgress((double) (repetition + 1) / (repetitions + warmup),
                                      "(" + methodNamesMapping[method] + ")");
        }

        if (repetition >= warmup)
            executionTimes.push_back(execTimeMs);

        memset(C, 0, (size_t) MATRIX_SIZE * MATRIX_SIZE * sizeof(DT));
    }

    if (executionTimes.size() == repetitions) {
        if (verbose)
            Helper::IO::printProgress(1.0, "(" + methodNamesMapping[method] + ")", true); // print the final bar

        auto meanExecTimeMs = Helper::Math::calculateMean(executionTimes);
        auto medianExecTimeMs = Helper::Math::calculateMedian(executionTimes);

        auto meanGflops = Helper::Math::msToGFLOPs(meanExecTimeMs, MATRIX_SIZE);
        auto medianGflops = Helper::Math::msToGFLOPs(std::get<0>(medianExecTimeMs), MATRIX_SIZE);

        runResults.push_back({methodNamesMapping[method], "1", warmup, repetitions, std::get<1>(medianExecTimeMs),
                              std::get<2>(medianExecTimeMs), std::get<0>(medianExecTimeMs), meanExecTimeMs, meanGflops,
                              medianGflops});

        if (verbose)
            std::cout << methodNamesMapping[method] << ": " << "AVG=" << meanExecTimeMs << "ms, (" << meanGflops
                      << " GFLOP/s) & MED=" << std::get<0>(medianExecTimeMs) << "ms, (" << medianGflops << " GFLOP/s)"
                      << std::endl << std::endl;
    } else {
        runResults.push_back({methodNamesMapping[method], "0", warmup, repetitions, 0, 0, 0, 0, 0, 0});
    }

    free(C);
}

void MatrixMultiplication::execute(Method method) {
    switch (method) {
        case Method::IJK:
            runMethod(Target::Basic::ijk, method);
            break;
        case Method::IJK_COLLAPSED:
            runMethod(Target::Basic::ijkCollapsed, method);
            break;
        case Method::BLOCKED_SHMEM:
            runMethod(Target::Blocked::shmem, method);
            break;
        case Method::BLOCKED_SHMEM_MEM_DIRECTIVES:
#if NO_MEM_DIRECTIVES
            if (verbose) {
                std::cout << "Skipping tiled shared memory matrix multiplication due to compiler flag." << std::endl;
                std::cout << "To enable set NO_MEM_DIRECTIVES to false." << std::endl << std::endl;
            }
            runResults.push_back({methodNamesMapping[method], "NOT COMPILED", warmup, repetitions, 0, 0, 0, 0, 0, 0});
#else
            runMethod(Target::Blocked::memoryAllocator, method);
#endif
            break;
        case Method::IJK_COLLAPSED_LOOP:
#if NO_LOOP_DIRECTIVES
            if (verbose) {
                std::cout << "Skipping loop directive matrix multiplication (ijk_collapsed_loop) due to compiler flag."
                          << std::endl;
                std::cout << "To enable set NO_LOOP_DIRECTIVES to false." << std::endl << std::endl;
            }
            runResults.push_back({methodNamesMapping[method], "NOT COMPILED", warmup, repetitions, 0, 0, 0, 0, 0, 0});
#else
            runMethod(Target::Loop::ijkCollapsedLoop, method);
#endif
            break;
        case Method::IJK_LOOP:
#if NO_LOOP_DIRECTIVES
            if (verbose) {
                std::cout << "Skipping loop directive matrix multiplication (ikj_loop) due to compiler flag."
                          << std::endl;
                std::cout << "To enable set NO_LOOP_DIRECTIVES to false." << std::endl << std::endl;
            }
            runResults.push_back({methodNamesMapping[method], "NOT COMPILED", warmup, repetitions, 0, 0, 0, 0, 0, 0});
#else
            runMethod(Target::Loop::ijkOnlyLoop, method);
#endif
            break;
        case Method::BLOCKED_K:
            runMethod(Target::Blocked::openmpBlocking, method);
            break;
        case BLOCKED_SHMEM_REDUCED_BC:
            runMethod(Target::Blocked::reducedBankConflict, method);
            break;
        case Method::IJK_COLLAPSED_SPMD:
            runMethod(Target::Basic::ijkCollapsedSPMD, method);
            break;
        case Method::BLOCKED_SHMEM_IRREGULAR:
            runMethod(Target::Blocked::irregularBlock, method);
            break;
        case BLOCKED_K_THREAD_LIMIT:
#if !OVERWRITE_DEFAULT_NUMS
            if (verbose) {
                std::cout << "Skipping blocked with overwrite due to compiler flag."
                          << std::endl;
                std::cout << "To enable set NO_LOOP_DIRECTIVES to false." << std::endl << std::endl;
            }
            runResults.push_back({methodNamesMapping[method], "NOT COMPILED", warmup, repetitions, 0, 0, 0, 0, 0, 0});
#else
            runMethod(Target::Blocked::openmpBlockingThreadLimit, method);
#endif
            break;
    }
}

void MatrixMultiplication::writeFile() {
    Output::writeOutput(filename, csv ? Output::FileType::CSV : Output::FileType::TXT, runResults);
}