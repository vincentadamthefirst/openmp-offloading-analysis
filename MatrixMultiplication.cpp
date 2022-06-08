#include <iostream>
#include <cstring>

#include "MatrixMultiplication.h"

template<typename T>
void MatrixMultiplication<T>::runBlockedMethods() {
    // cancel if there are no block sizes
    if (blockSizeStart <= 0)
        return;

    std::cout << "Executing blocked methods:" << std::endl;
    for (size_t method = 0; method < blockedMethods.size(); method++) {
        for (uint32_t blockSize = blockSizeStart; blockSize <= blockSizeEnd; blockSize *= 2) {
            std::cout << "  " << blockedMethods[method].name << ", block size: " << blockSize << std::endl;
            std::vector<double> runtimes;

            auto C = Helper::Matrix::initializeZero<T>(matrixSize);
            bool correctness = false;

            for (uint32_t repetition = 0; repetition < repetitions; repetition++) {
                auto time = Helper::Bench::timeFuncInvocation(blockedMethods[method].method, A, B, C, matrixSize,
                                                              blockSize);
                runtimes.push_back(time);

                if (checkMatrix && repetition == 0) {
                    correctness = Helper::Matrix::compare<T>(C, checkMatrix, matrixSize);
                    std::cout << "    " << "COMPARE RESULT = " << (correctness ? "OK" : "INCORRECT") << std::endl;
                }

                if (verbose)
                    std::cout << "    " << "Repetition " << repetition << ": " << time << "ms ("
                              << Helper::Math::msToGFLOPs(time, matrixSize) << " GFLOP/s)" << std::endl;

                memset(C, 0, matrixSize * matrixSize * sizeof(T));
            }

            auto timeMed = Helper::Math::calculateMedian(runtimes);
            auto timeAvg = Helper::Math::calculateMean(runtimes);

            if (verbose)
                std::cout << "  Method " << blockedMethods[method].name << ": MED= " << timeMed << "ms ("
                          << Helper::Math::msToGFLOPs(timeMed, matrixSize) << ")" << ", AVG=" << timeAvg << "ms ("
                          << Helper::Math::msToGFLOPs(timeAvg, matrixSize) << ")" << std::endl;
            std::cout << std::endl;

            writeToFile(blockedMethods[method].name + " size " + std::to_string(blockSize), timeMed, timeAvg,
                        correctness);

            free(C);
        }
    }
}

template<typename T>
void MatrixMultiplication<T>::runBasicMethods() {
    std::cout << "Executing basic methods:" << std::endl;
    for (size_t method = 0; method < basicMethods.size(); method++) {
        std::cout << "  " << basicMethods[method].name << std::endl;
        std::vector<double> runtimes;

        auto C = Helper::Matrix::initializeZero<T>(matrixSize);
        bool correctness = false;

        for (uint32_t repetition = 0; repetition < repetitions; repetition++) {
            auto time = Helper::Bench::timeFuncInvocation(basicMethods[method].method, A, B, C, matrixSize);
            runtimes.push_back(time);

            if (checkMatrix && repetition == 0) {
                correctness = Helper::Matrix::compare<T>(C, checkMatrix, matrixSize);
                std::cout << "    " << "COMPARE RESULT = " << (correctness ? "OK" : "INCORRECT") << std::endl;
            }

            if (verbose)
                std::cout << "    " << "Repetition " << repetition << ": " << time << "ms ("
                          << Helper::Math::msToGFLOPs(time, matrixSize) << " GFLOP/s)" << std::endl;

            memset(C, 0, (size_t) matrixSize * matrixSize * sizeof(T));
        }

        auto timeMed = Helper::Math::calculateMedian(runtimes);
        auto timeAvg = Helper::Math::calculateMean(runtimes);

        if (verbose)
            std::cout << "  Method " << basicMethods[method].name << ": MED= " << timeMed << "ms ("
                      << Helper::Math::msToGFLOPs(timeMed, matrixSize) << ")" << ", AVG=" << timeAvg << "ms ("
                      << Helper::Math::msToGFLOPs(timeAvg, matrixSize) << ")" << std::endl;
        std::cout << std::endl;

        writeToFile(basicMethods[method].name, timeMed, timeAvg, correctness);

        free(C);
    }
}

template<typename T>
void MatrixMultiplication<T>::execute() {
    if (!checkMatrix) {
        // prepare A and B
        prepareMatrices();
    }

    // prepare file
    writeFileHeader();

    // execute actual matrix multiplication
    runBasicMethods();
    runBlockedMethods();
}