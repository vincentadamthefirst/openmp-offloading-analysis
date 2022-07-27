#pragma once

#include <cstdlib>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <chrono>

#include "../libs/cmdparser.hpp"

#define EPSILON 0.001       // epsilon for float comparison
#define BAR_SIZE 70         // size of progress bars

namespace Helper {

    namespace IO {
        /**
         * Pads a string with a given char to a given length
         * From https://stackoverflow.com/a/667219
         * @param str the string to pad
         * @param num desired length
         */
        std::string padLeft(std::string str, const size_t num, char padChar = ' ') {
            if(num > str.size())
                str.insert(0, num - str.size(), padChar);
            return str;
        }

        /**
         * Pads a string with a given char to a given length
         * From https://stackoverflow.com/a/667219
         * @param str the string to pad
         * @param num desired length
         */
        std::string padRight(std::string str, const size_t num, char padChar = ' ') {
            if(num > str.size())
                str.insert(str.length(), num - str.size(), padChar);
            return str;
        }

        void printProgress(double progress, std::string info, bool isFinal = false) {
            std::cout << "[";
            int pos = BAR_SIZE * progress;
            for (int i = 0; i < BAR_SIZE; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << "% " << info << (isFinal ? " \n" : " \r");
            std::cout.flush();
        }

        /**
         * Sets up the basic arguments of the CLI.
         * @param parser the parser to add the arguments to.
         */
        void basicParserSetup(cli::Parser& parser) {
            parser.set_optional<std::string>("o", "output", "GENERATE_NEW",
                                             "File the output should be written to. If no file is given a "
                                             "new file will be generated next to the executable.");
            parser.set_optional<std::string>("ft", "file_type", "txt", "Set the formatting of the output. Must be 'txt' or "
                                                                       "'csv'.");
            parser.set_optional<bool>("v", "verbose", false, "Enable verbose output.");
            parser.set_optional<int>("r", "repetitions", 11, "Sets the amount of repetitions for the methods.");
            parser.set_optional<int>("w", "warmup", 5, "Sets the amount of warmup calculations for the methods.");
            parser.set_optional<bool>("no", "no-output", false, "Disables the output to file.");
            parser.set_optional<bool>("c", "comparison", false, "Enables result checking of "
                                                                "GPU calculations with previously generated CPU ones.");
        }
    }

    namespace Math {
        /**
         * Calculates the median of a vector of values.
         * @param values the values to calculate the median of
         * @return <median, min value, max value>
         */
        std::tuple<double, double, double> calculateMedian(std::vector<double> values) {
            auto size = values.size();
            if (size == 1)
                return std::make_tuple(values[0], values[0], values[0]);

            std::sort(values.begin(), values.end());
            if (size % 2 == 0) {
                return std::make_tuple((values[size / 2 - 1] + values[size / 2]) / 2, values[0], values[size - 1]);
            } else {
                return std::make_tuple(values[size / 2], values[0], values[size - 1]);
            }
        }

        double calculateMean(std::vector<double> values) {
            double sum = 0;
            auto size = values.size();
            std::for_each(values.begin(), values.end(), [&](double v) {
                sum += v;
            });
            return sum / size;
        }

        double msToGFLOPs(double ms, double matrixSize) {
            double operations = (2.0 * matrixSize - 1.0) * matrixSize * matrixSize;
            return (operations / 1000000000) /* 10^9 operations */ / ((double) ms / 1000) /* per second */;
        }
    }

    namespace Matrix {
        template<typename T>
        T *initializeRandom(unsigned size, T minValue, T maxValue) {
            auto output = (T *) malloc(size * size * sizeof(T));
            for (unsigned i = 0; i < size; i++) {
                for (unsigned j = 0; j < size; j++) {
                    output[i * size + j] = minValue + (((T) rand() / (T) RAND_MAX) * (maxValue - minValue));
                }
            }
            return output;
        }

        template<typename T>
        T *initializeZero(unsigned size) {
            auto output = (T *) calloc(size * size, sizeof(T));
            return output;
        }

        template<typename T>
        bool compare(T *A, T *B, unsigned size) {
            for (unsigned i = 0; i < size; i++) {
                for (unsigned j = 0; j < size; j++) {
                    if (std::abs(A[i * size + j] - B[i * size + j]) > EPSILON) {
                        return false;
                    }
                }
            }

            return true;
        }

        template<typename T>
        void print(T *matrix, unsigned size) {
            for (unsigned i = 0; i < size; i++) {
                for (unsigned j = 0; j < size; j++) {
                    std::cout << std::setw(5) << std::fixed << std::setprecision(2) << matrix[i * size + j]
                              << (j != size - 1 ? ", " : "");
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        template<typename T>
        void freeAll(const std::vector<T *> &args) {
            for (const auto &arg: args) {
                free(arg);
            }
        }
    }
}
