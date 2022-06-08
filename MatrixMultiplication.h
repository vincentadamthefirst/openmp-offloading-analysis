#pragma once

#include <string>
#include <utility>
#include <vector>

#include "include/Target.h"
#include "include/Host.h"
#include "include/Helper.h"

template<typename T, typename M>
struct MultiplicationMethod {
    M method;
    std::string name;
};

template<typename T>
class MatrixMultiplication {
public:
    MatrixMultiplication(std::string filename, bool verbose, bool csv, uint32_t matrixSize) :
            filename(std::move(filename)), verbose(verbose), csv(csv), matrixSize(matrixSize) {}

    ~MatrixMultiplication() {
        Helper::Matrix::freeAll<T>({A, B});
        if (checkMatrix)
            free(checkMatrix);
    }

    MatrixMultiplication& enableRepetitions(uint32_t r) {
        repetitions = r;
        return *this;
    }

    MatrixMultiplication& enableBlockSizes(uint32_t start, uint32_t end) {
        blockSizeStart = (int) start;
        blockSizeEnd = (int) end;
        return *this;
    }

    MatrixMultiplication& withValueRange(T lower, T upper) {
        lowerValueBound = lower;
        upperValueBound = upper;
        return *this;
    }

    MatrixMultiplication& enableCheck() {
        prepareMatrices();

        std::cout << "Preparing matrix for later checking..." << std::endl;
        checkMatrix = Helper::Matrix::initializeZero<T>(matrixSize);
        Host::multiplyIKJ(A, B, checkMatrix, matrixSize);

        std::cout << "Done." << std::endl;

        return *this;
    }

    /// Debug method. Prints the settings for this matrix multiplication.
    void printInfo() {
        std::cout << "file path:    " << filename << std::endl;
        std::cout << "verbose:      " << (verbose ? "ON" : "OFF") << std::endl;
        std::cout << "matrix size:  " << matrixSize << std::endl;
        std::cout << "repetitions:  " << repetitions << std::endl;
        std::cout << "block start:  " << blockSizeStart << std::endl;
        std::cout << "block end:    " << blockSizeEnd << std::endl;
        std::cout << "lower value:  " << lowerValueBound << std::endl;
        std::cout << "upper value:  " << upperValueBound << std::endl;
        std::cout << "check matrix: " << (checkMatrix ? "PRESENT" : "MISSING") << std::endl;
    }

    void execute();

private:

    void runBlockedMethods();

    void runBasicMethods();

    void prepareMatrices() {
        std::cout << "Preparing matrices A and B..." << std::endl;
        A = Helper::Matrix::initializeRandom<T>(matrixSize, lowerValueBound, upperValueBound);
        B = Helper::Matrix::initializeRandom<T>(matrixSize, lowerValueBound, upperValueBound);
    }

    void writeFileHeader() {
        std::ofstream fileStream;
        fileStream.open(filename, std::ios_base::app);

        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);

        fileStream << std::endl;
        fileStream << "# EVALUATION RUN " << std::put_time(&tm, "%d-%m-%Y %H-%M-%S") << std::endl;
        fileStream << "# matrix size = " << matrixSize << std::endl;
        fileStream << "# block start = " << blockSizeStart << std::endl;
        fileStream << "# block end   = " << blockSizeEnd << std::endl;
        fileStream << "# repetitions = " << repetitions << std::endl;

        if (csv) {
            fileStream << "method_name,time_median,time_mean,gflops_median,gflops_mean";
            if (checkMatrix)
                fileStream << ",correctness";
        }

        fileStream << std::endl;
        fileStream.close();
    }

    void writeToFile(std::string methodName, double timeMed, double timeAvg, bool correct) {
        std::ofstream fileStream;
        fileStream.open(filename, std::ios_base::app);
        auto gflopsMed = Helper::Math::msToGFLOPs(timeMed, matrixSize);
        auto gflopsAvg = Helper::Math::msToGFLOPs(timeAvg, matrixSize);

        if (csv) {
            fileStream << methodName << "," << timeMed << "," << timeAvg << "," << gflopsMed << "," << gflopsAvg;
            if (checkMatrix)
                fileStream << "," << (correct ? "CORRECT" : "INCORRECT");
        } else {
            fileStream << methodName
                       << ": MED= " << timeMed << "ms (" << gflopsMed << " GFLOP/s)"
                       << ", AVG=" << timeAvg << "ms (" << gflopsAvg << " GFLOP/s)";
            if (checkMatrix)
                fileStream << ", CHECK= " << (correct ? "OK" : "INCORRECT");
        }
        fileStream << std::endl;
        fileStream.close();
    }

private:
    /// private member values
    const std::string filename;
    const bool verbose;             // include more detailed output during runtime
    const bool csv = false;         // write csv compatible output
    const uint32_t matrixSize;

    uint32_t repetitions = 1;
    int blockSizeStart = -1;
    int blockSizeEnd = -1;
    T lowerValueBound = 0;
    T upperValueBound = 1;

    T* A = nullptr;
    T* B = nullptr;
    T* checkMatrix = nullptr;

private:
    /// definitions for the matrix multiplication methods to measure
    typedef void (* vBlockedMultiplication)(T *A, T *B, T *C, uint32_t size, uint32_t blockSize);
    typedef void (* vBasicMultiplication)(T *A, T *B, T *C, uint32_t size);

    const std::vector<MultiplicationMethod<T, vBasicMultiplication>> basicMethods = {
//            {Target::multiplyJIKSharedMemory<T>, "JIK (shmem)"},
            {Target::multiplyIJK<T>, "IJK"},
            {Target::multiplyIKJ<T>, "IKJ"},
            {Target::multiplyJIK<T>, "JIK"},
            {Target::multiplyJKI<T>, "JKI"},
            {Target::multiplyIJKCollapsed<T>, "IJK (collapsed)"},
            {Target::multiplyIKJCollapsed<T>, "IKJ (collapsed)"},
            {Target::multiplyJIKCollapsed<T>, "JIK (collapsed)"},
            {Target::multiplyJKICollapsed<T>, "JKI (collapsed)"},
    };

    const std::vector<MultiplicationMethod<T, vBlockedMultiplication>> blockedMethods = {
            {Target::multiplyIJKBlocked<T>, "IKJ (blocked)"},
            {Target::multiplyIJKBlocked<T>, "JIK (blocked)"},
    };
};
