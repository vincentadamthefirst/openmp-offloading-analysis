#pragma once

#include <iostream>
#include <vector>
#include <cstring>
#include <ctime>
#include <utility>
#include <fstream>

#include "preprocessor_settings.h"
#include "helper.hpp"

namespace Output {
    struct MatrixMultiplyRunResult {
        std::string method;
        std::string status;

        uint32_t warmup;
        uint32_t repetitions;
        uint32_t matrixSize;
        uint32_t blockSize;

        double minExecutionTimeMs;
        double maxExecutionTimeMs;

        double meanExecutionTimeMs;
        double medianExecutionTimeMs;

        double meanGflops;
        double medianGflops;
    };

    enum FileType {
        TXT,
        CSV
    };

    void writeCSV(std::string path, std::vector<MatrixMultiplyRunResult> results, std::string suffix) {
        std::ofstream fileStream;
        fileStream.open(path, std::ios_base::app);

        std::ifstream inFileStream;
        inFileStream.open(path, std::ios_base::app);

        // check if the file is empty (newly created) - if so write a new CSV header
        if (inFileStream.peek() == std::ifstream::traits_type::eof()) {
            fileStream << "method_name,data_type,matrixSize,tile_size,repetitions,status,exec_time_min(ms),"
                          "exec_time_max(ms),exec_time_mean(ms),exec_time_median(ms),gflops_mean,gflops_median" << std::endl;
        }

        inFileStream.close();

        for (const auto& runResult : results) {
            fileStream << runResult.method << suffix << ",";
            fileStream << typeid(DATA_TYPE).name() << ",";
            fileStream << runResult.matrixSize << ",";
            fileStream << runResult.blockSize << ",";
            fileStream << runResult.repetitions << ",";
            fileStream << runResult.status << ",";
            fileStream << std::to_string(runResult.minExecutionTimeMs) << ",";
            fileStream << std::to_string(runResult.maxExecutionTimeMs) << ",";
            fileStream << std::to_string(runResult.meanExecutionTimeMs) << ",";
            fileStream << std::to_string(runResult.medianExecutionTimeMs) << ",";
            fileStream << std::to_string(runResult.meanGflops) << ",";
            fileStream << std::to_string(runResult.medianGflops) << "," << std::endl;
        }

        fileStream.close();
    }

    void writeTXT(std::string path, std::vector<MatrixMultiplyRunResult> results, std::string time, std::string suffix) {
        std::ofstream fileStream;
        fileStream.open(path, std::ios_base::app);

        // write general run information
        fileStream << "Matrix Multiplication Results (" << time << ")" << std::endl;
        fileStream << std::endl;

        std::vector<std::string> tableHeader = {"method", "status", "exec_time_min (ms)", "exec_time_max (ms)",
                                                "exec_time_mean (ms)", "exec_time_median (ms)", "gflops_mean",
                                                "gflops_median"};
        std::vector<size_t> columnWidths = {tableHeader[0].length(), tableHeader[1].length(), tableHeader[2].length(),
                                            tableHeader[3].length(), tableHeader[4].length(), tableHeader[5].length(),
                                            tableHeader[6].length(), tableHeader[7].length(),};

        // find the width of the first column
        for (const auto &runResult : results) {
            std::string tmp;
            tmp += runResult.method;
            tmp += suffix;
            columnWidths[0] = std::max(columnWidths[0], tmp.length());
            columnWidths[1] = std::max(columnWidths[1], runResult.status.length());
        }

        // write table header
        for (size_t i = 0; i < tableHeader.size(); i++) {
            fileStream << Helper::IO::padLeft(tableHeader[i], columnWidths[i]) << " ";
        }
        fileStream << std::endl;

        // write the actual table structure
        for (const auto &runResult : results) {
            fileStream << Helper::IO::padLeft(runResult.method + suffix, columnWidths[0]) << " ";
            fileStream << Helper::IO::padLeft(runResult.status, columnWidths[1]) << " ";
            fileStream << Helper::IO::padLeft(std::to_string(runResult.minExecutionTimeMs), columnWidths[2]) << " ";
            fileStream << Helper::IO::padLeft(std::to_string(runResult.maxExecutionTimeMs), columnWidths[3]) << " ";
            fileStream << Helper::IO::padLeft(std::to_string(runResult.meanExecutionTimeMs), columnWidths[4]) << " ";
            fileStream << Helper::IO::padLeft(std::to_string(runResult.medianExecutionTimeMs), columnWidths[5]) << " ";
            fileStream << Helper::IO::padLeft(std::to_string(runResult.meanGflops), columnWidths[6]) << " ";
            fileStream << Helper::IO::padLeft(std::to_string(runResult.medianGflops), columnWidths[7]) << " ";
            fileStream << std::endl;
        }

        fileStream << std::endl;
        fileStream.close();
    }

    void writeOutput(std::string path, FileType ft, std::vector<MatrixMultiplyRunResult> results, std::string suffix="") {
        if (path == "NO_OUTPUT_FILE")
            return;

        time_t rawtime;
        struct tm * timeinfo;
        char buffer[80];

        time (&rawtime);
        timeinfo = localtime(&rawtime);

        strftime(buffer,sizeof(buffer),"%d-%m-%Y_%H:%M:%S",timeinfo);
        std::string timeString(buffer);

        std::ifstream fileCheck(path.c_str());
        if (path == "GENERATE_NEW" || !fileCheck.good()) {
            // no file found, create a file
            std::string newName =
                    path == "GENERATE_NEW" ? "matmul_result_" + timeString + (ft == CSV ? ".csv" : ".txt") : path;
            std::cout << "No output file found, generate a new one (" << newName << ")" << std::endl;

            fileCheck.close();
            std::ofstream newFile(newName);
            newFile.close();
            path = newName;
        }

        ft == CSV ? writeCSV(path, results, suffix) : writeTXT(path, results, timeString, suffix);
    }
}

