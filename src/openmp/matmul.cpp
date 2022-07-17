#include <iostream>
#include <cstring>

#include "matmul.hpp"

void MatrixMultiplication::runMethod(double (*functionPtr)(const DT*, const DT*, DT*), Method method) {
    auto C = Helper::Matrix::initializeZero<DT>(MATRIX_SIZE);

    std::vector<double> executionTimes;

    for (size_t repetition = 0; repetition < repetitions; repetition++) {
        if (verbose) {
            Helper::IO::printProgress((double) (repetition) / repetitions,
                                      "(" + methodNamesMapping[method] + ")");
        }

        auto execTimeMs = functionPtr(A, B, C);

        if (checkMatrix && repetition == 0) { // only check for correctness on first run
            auto correctness = Helper::Matrix::compare<DT>(C, checkMatrix, MATRIX_SIZE);
            if (!correctness) {
                Helper::IO::printProgress((double) (repetition + 1) / repetitions,
                                          "(" + methodNamesMapping[method] + " ==ABORTED== )", true);
                std::cerr << "Method " << methodNamesMapping[method]
                          << " did not produce correct results. Aborting." << std::endl << std::endl;
                break;
            }
        }

        if (verbose) {
            Helper::IO::printProgress((double) (repetition + 1) / repetitions,
                                      "(" + methodNamesMapping[method] + ")");
        }

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

        runResults.push_back({method, "1", std::get<1>(medianExecTimeMs), std::get<2>(medianExecTimeMs),
                              std::get<0>(medianExecTimeMs), meanExecTimeMs});

        if (verbose)
            std::cout << methodNamesMapping[method] << ": " << "AVG=" << meanExecTimeMs << "ms, (" << meanGflops
                      << " GFLOP/s) & MED=" << std::get<0>(medianExecTimeMs) << "ms, (" << medianGflops << " GFLOP/s)"
                      << std::endl << std::endl;
    } else {
        runResults.push_back({method, "0", -1, -1, -1, -1});
    }

    free(C);
}

void MatrixMultiplication::execute(Method method) {
    switch (method) {
        case Method::IJK:
            runMethod(Target::Basic::multiplyIJK, method);
            break;
        case Method::IKJ:
            runMethod(Target::Basic::multiplyIKJ, method);
            break;
        case Method::JIK:
            runMethod(Target::Basic::multiplyJIK, method);
            break;
        case Method::JKI:
            runMethod(Target::Basic::multiplyJKI, method);
            break;
        case Method::IJK_COLLAPSED:
            runMethod(Target::Basic::multiplyIJKCollapsed, method);
            break;
        case Method::JIK_COLLAPSED:
            runMethod(Target::Basic::multiplyJIKCollapsed, method);
            break;
        case Method::TILED_SHMEM:
            runMethod(Target::Tiled::multiplyTiled, method);
            break;
        case Method::TILED_SHMEM_MEM_DIRECTIVES:
#if NO_MEM_DIRECTIVES
            if (verbose) {
                std::cout << "Skipping tiled shared memory matrix multiplication due to compiler flag." << std::endl;
                std::cout << "To enable set NO_MEM_DIRECTIVES to false." << std::endl << std::endl;
            }
            runResults.push_back({method, "NOT COMPILED", -1, -1, -1, -1});
#else
            runMethod(Target::Tiled::multiplyTiledAllocator, method);
#endif
            break;
        case Method::IJK_COLLAPSED_LOOP:
#if NO_LOOP_DIRECTIVES
            if (verbose) {
                std::cout << "Skipping loop directive matrix multiplication due to compiler flag." << std::endl;
                std::cout << "To enable set NO_LOOP_DIRECTIVES to false." << std::endl << std::endl;
            }
            runResults.push_back({method, "NOT COMPILED", -1, -1, -1, -1});
#else
            runMethod(Target::Loop::multiplyIJKCollapsedLoop, method);
#endif
            break;
        case Method::IJK_LOOP:
#if NO_LOOP_DIRECTIVES
            if (verbose) {
                std::cout << "Skipping loop directive matrix multiplication due to compiler flag." << std::endl;
                std::cout << "To enable set NO_LOOP_DIRECTIVES to false." << std::endl << std::endl;
            }
            runResults.push_back({method, "NOT COMPILED", -1, -1, -1, -1});
#else
            runMethod(Target::Loop::multiplyIJKCompleteLoop, method);
#endif
            break;
        case Method::TILED_K:
            runMethod(Target::Tiled::multiplyTiledK, method);
            break;
        case IKJ_COLLAPSED:
            runMethod(Target::Basic::multiplyIKJCollapsed, method);
            break;
        case JKI_COLLAPSED:
            runMethod(Target::Basic::multiplyJKICollapsed, method);
            break;
        case TILED_SHMEM_NO_BANK_CONFLICT:
            runMethod(Target::Tiled::multiplyTiledNoBankConflict, method);
            break;
    }
}

void MatrixMultiplication::writeFile() {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,sizeof(buffer),"%d-%m-%Y_%H:%M:%S",timeinfo);
    std::string str(buffer);
    timeString = str;

    std::ifstream fileCheck(filename.c_str());
    if (filename == "GENERATE_NEW" || !fileCheck.good()) {
        // no file found, create a file
        std::string newName =
                filename == "GENERATE_NEW" ? "matmul_result_" + std::to_string(MATRIX_SIZE) + "_" + timeString +
                                             (csv ? ".csv" : ".txt") : filename;
        std::cout << "No output file found, generate a new one (" << newName << ")" << std::endl;

        fileCheck.close();
        std::ofstream newFile(newName);
        newFile.close();
        filename = newName;
    }

    csv ? writeCSV() : writeTXT();
}

void MatrixMultiplication::writeCSV() {
    std::ofstream fileStream;
    fileStream.open(filename, std::ios_base::app);

    std::ifstream inFileStream;
    inFileStream.open(filename, std::ios_base::app);

    // check if the file is empty (newly created) - if so write a new CSV header
    if (inFileStream.peek() == std::ifstream::traits_type::eof()) {
        fileStream << "method_name,data_type,matrix_size,tile_size,repetitions,compare_result,exec_time_min(ms),"
                      "exec_time_max(ms),exec_time_mean(ms),exec_time_median(ms)" << std::endl;
    }

    inFileStream.close();

    for (const auto& runResult : runResults) {
        auto methodName = methodNamesMapping[runResult.method];
        fileStream << methodName << ",";
        fileStream << typeid(DATA_TYPE).name() << ",";
        fileStream << MATRIX_SIZE << ",";
        fileStream << ((methodName.find("tile") != std::string::npos) ? std::to_string(TILE_SIZE) : "-1") << ",";
        fileStream << repetitions << ",";
        fileStream << runResult.compareResult << ",";
        fileStream << std::to_string(runResult.minExecutionTimeMs) << ",";
        fileStream << std::to_string(runResult.maxExecutionTimeMs) << ",";
        fileStream << std::to_string(runResult.meanExecutionTimeMs) << ",";
        fileStream << std::to_string(runResult.medianExecutionTimeMs) << std::endl;
    }

    fileStream.close();
}

void MatrixMultiplication::writeTXT() {
    std::ofstream fileStream;
    fileStream.open(filename, std::ios_base::app);

    // write general run information
    fileStream << "Matrix Multiplication Results (" << timeString << ")" << std::endl;
    fileStream << "MATRIX_SIZE=" << MATRIX_SIZE << ", TILE_SIZE=" << TILE_SIZE << ", REPETITIONS=" << repetitions
               << ", DATA_TYPE=" << typeid(DATA_TYPE).name() << std::endl;
    fileStream << std::endl;

    std::vector<std::string> tableHeader = {"method", "correctness", "exec_time_min (ms)", "exec_time_max (ms)",
                                            "exec_time_mean (ms)", "exec_time_median (ms)"};
    std::vector<size_t> columnWidths = {tableHeader[0].length(), tableHeader[1].length(), tableHeader[2].length(),
                                        tableHeader[3].length(), tableHeader[4].length(), tableHeader[5].length()};

    // find the width of the first column
    for (const auto &runResult: runResults) {
        columnWidths[0] = std::max(columnWidths[0], methodNamesMapping[runResult.method].length());
        columnWidths[1] = std::max(columnWidths[1], runResult.compareResult.length());
    }

    // write table header
    for (size_t i = 0; i < tableHeader.size(); i++) {
        fileStream << Helper::IO::padLeft(tableHeader[i], columnWidths[i]) << " ";
    }
    fileStream << std::endl;

    // write the actual table structure
    for (const auto &runResult: runResults) {
        auto methodName = methodNamesMapping[runResult.method];
        fileStream << Helper::IO::padLeft(methodName, columnWidths[0]) << " ";
        fileStream << Helper::IO::padLeft(runResult.compareResult, columnWidths[1]) << " ";
        fileStream << Helper::IO::padLeft(std::to_string(runResult.minExecutionTimeMs), columnWidths[2]) << " ";
        fileStream << Helper::IO::padLeft(std::to_string(runResult.maxExecutionTimeMs), columnWidths[3]) << " ";
        fileStream << Helper::IO::padLeft(std::to_string(runResult.meanExecutionTimeMs), columnWidths[4]) << " ";
        fileStream << Helper::IO::padLeft(std::to_string(runResult.medianExecutionTimeMs), columnWidths[5]) << " ";
        fileStream << std::endl;
    }

    fileStream << std::endl;
    fileStream.close();
}
