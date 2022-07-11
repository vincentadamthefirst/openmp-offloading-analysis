#pragma once

#include <string>
#include <utility>
#include <vector>
#include <ctime>

#include "target.hpp"
#include "host.hpp"
#include "../../include/Helper.hpp"

struct MatrixMultiplyRunResult {
    Method method;
    std::string compareResult;
    double maxExecutionTimeMs;
    double minExecutionTimeMs;
    double meanExecutionTimeMs;
    double medianExecutionTimeMs;
};

class MatrixMultiplication {
public:
    MatrixMultiplication(std::string filename, bool verbose, bool csv) :
            filename(std::move(filename)), verbose(verbose), csv(csv) {}

    ~MatrixMultiplication() {
        Helper::Matrix::freeAll<DT>({A, B});
        if (checkMatrix)
            free(checkMatrix);
    }

    MatrixMultiplication& enableRepetitions(int r) {
        repetitions = r < 1 ? 1 : r;
        return *this;
    }

    MatrixMultiplication& enableCheck(bool check) {
        prepareMatrices();

        if (check) {
            std::cout << "Preparing matrix for later checking (this make take some time)...";
            checkMatrix = Helper::Matrix::initializeZero<DT>(MATRIX_SIZE);
            Host::multiplyIKJParallel(A, B, checkMatrix, MATRIX_SIZE);
            std::cout << "Done." << std::endl;
        }
        std::cout << std::endl;

        return *this;
    }

#define STRING(s) #s

    /// Debug method. Prints the settings for this matrix multiplication.
    void printInfo() {
        std::string dataTypeString = "";
        if (typeid(DATA_TYPE) == typeid(float)) dataTypeString = "FLOAT";
        if (typeid(DATA_TYPE) == typeid(double)) dataTypeString = "DOUBLE";
        if (typeid(DATA_TYPE) == typeid(int)) dataTypeString = "INT";

        std::cout << "file path:      " << filename << std::endl;
        std::cout << "verbose:        " << (verbose ? "ON" : "OFF") << std::endl;
        std::cout << "matrix size:    " << MATRIX_SIZE << std::endl;
        std::cout << "repetitions:    " << repetitions << std::endl;
        std::cout << "lower value:    " << VALUE_RANGE_LOWER << std::endl;
        std::cout << "upper value:    " << VALUE_RANGE_UPPER << std::endl;
        std::cout << "data type:      " << dataTypeString << std::endl;
        std::cout << "tile size:      " << TILE_SIZE << std::endl;
        std::cout << "tile axis size: " << TILE_AXIS_SIZE << std::endl;
        std::cout << "check matrix:   " << (checkMatrix ? "PRESENT" : "MISSING") << std::endl;
    }

    void execute(Method method);

    void writeFile();


private:

    void runMethod(double (*functionPtr)(const DT*, const DT*, DT*), Method method);

    void prepareMatrices() {
        std::cout << "Preparing matrices A and B..." << std::endl;
        A = Helper::Matrix::initializeRandom<DT>(MATRIX_SIZE, VALUE_RANGE_LOWER, VALUE_RANGE_UPPER);
        B = Helper::Matrix::initializeRandom<DT>(MATRIX_SIZE, VALUE_RANGE_LOWER, VALUE_RANGE_UPPER);
    }

    void writeCSV();

    void writeTXT();

private:
    /// private member values
    std::string filename;
    const bool verbose;             // include more detailed output during runtime
    const bool csv = false;         // write csv compatible output

    uint32_t repetitions = 1;

    std::vector<MatrixMultiplyRunResult> runResults;
    std::string timeString;         // for TXT output generation

    DT* A = nullptr;
    DT* B = nullptr;
    DT* checkMatrix = nullptr;
};
