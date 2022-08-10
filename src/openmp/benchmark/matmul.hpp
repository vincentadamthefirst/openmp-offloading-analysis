#pragma once

#include <string>
#include <utility>
#include <vector>
#include <ctime>

#include "target.hpp"
#include "../host.hpp"
#include "../../../include/helper.hpp"
#include "../../../include/output.hpp"

/**
 * Struct to track the results of a matrix multiplication method.
 */
struct MatrixMultiplyRunResult {
    Method method;
    std::string compareResult;
    double maxExecutionTimeMs;
    double minExecutionTimeMs;
    double meanExecutionTimeMs;
    double medianExecutionTimeMs;
    double meanGflops;
    double medianGflops;
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

    /**
     * Set the amount of repetitions to be executed per multiplication method. Values < 1 will be defaulted to 1.
     * @param r amount of repetitions
     */
    MatrixMultiplication& enableRepetitions(int r) {
        repetitions = r < 1 ? 1 : r;
        return *this;
    }

    MatrixMultiplication& enableWarmup(int w) {
        warmup = w < 0 ? 0 : w;
        return *this;
    }

    /**
     * Set if there should be a checking matrix to compare GPU and CPU results.
     * @param check if a check matrix should be calculated
     */
    MatrixMultiplication& enableCheck(bool check) {
        prepareMatrices();

        if (check) {
            std::cout << "Preparing matrix for later checking"
                      << (MATRIX_SIZE > 4096 ? " (this make take some time)" : "") << "... ";
            std::cout.flush();
            checkMatrix = Helper::Matrix::initializeZero<DT>(MATRIX_SIZE);
            Host::multiplyIKJParallel(A, B, checkMatrix, MATRIX_SIZE);
            std::cout << "Done." << std::endl;
        }
        std::cout << std::endl;

        return *this;
    }

    /// Debug method. Prints the settings for this matrix multiplication.
    void printInfo() {
        std::cout << "file path:      " << filename << std::endl;
        std::cout << "verbose:        " << (verbose ? "ON" : "OFF") << std::endl;
        std::cout << "matrix size:    " << MATRIX_SIZE << std::endl;
        std::cout << "repetitions:    " << repetitions << std::endl;
        std::cout << "warmup:         " << warmup << std::endl;
        std::cout << "lower value:    " << VALUE_RANGE_LOWER << std::endl;
        std::cout << "upper value:    " << VALUE_RANGE_UPPER << std::endl;
        std::cout << "data type:      " << typeid(DATA_TYPE).name() << std::endl;
        std::cout << "tile size:      " << TILE_SIZE << std::endl;
        std::cout << "k block size:   " << K_BLOCK_SIZE << std::endl;
        std::cout << "tile axis size: " << TILE_AXIS_SIZE << std::endl;
        std::cout << "check matrix:   " << (checkMatrix ? "PRESENT" : "MISSING") << std::endl;
        std::cout << std::endl;
    }

    /**
     * Executes the benchmark for a given method.
     * @param method the method to execute
     */
    void execute(Method method);

    /**
     * Writes the results collected until now to a file. <br>
     * The type of file and filename are specified in the constructor.
     */
    void writeFile();


private:

    /**
     * Performs the benchmark for a method. Performs the repetitions and checks (if set).
     * @param functionPtr the function to call for that method
     * @param method the type of method that is executed
     */
    void runMethod(double (*functionPtr)(const DT*, const DT*, DT*), Method method);

    /**
     * Prepares the matrices A and B to be multiplied.
     */
    void prepareMatrices() {
        std::cout << "Preparing matrices A and B... ";
        A = Helper::Matrix::initializeRandom<DT>(MATRIX_SIZE, VALUE_RANGE_LOWER, VALUE_RANGE_UPPER);
        B = Helper::Matrix::initializeRandom<DT>(MATRIX_SIZE, VALUE_RANGE_LOWER, VALUE_RANGE_UPPER);
        std::cout << "Done." << std::endl;
    }

private:
    /// private member values
    std::string filename;
    const bool verbose;             // include more detailed output during runtime
    const bool csv = false;         // write csv compatible output

    uint32_t repetitions = 1;
    uint32_t warmup = 0;

    std::vector<Output::MatrixMultiplyRunResult> runResults;
    std::string timeString;         // for TXT output generation

    DT* A = nullptr;
    DT* B = nullptr;
    DT* checkMatrix = nullptr;
};
