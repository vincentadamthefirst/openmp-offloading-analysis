#pragma once

#include <cstdlib>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <chrono>

#define EPSILON 0.001

template<typename ...Args>
void LOG(Args && ...args) {
    (std::cout << ... << args);
}

namespace Helper {

    namespace Math {
        double calculateMedian(std::vector<double> values) {
            auto size = values.size();
            if (size == 1)
                return values[0];

            std::sort(values.begin(), values.end());
            if (size % 2 == 0) {
                return (values[size / 2 - 1] + values[size / 2]) / 2;
            } else {
                return values[size / 2];
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

        double msToGFLOPs(double ms, uint32_t matrixSize) {
            double operations = (2.0 * (double) matrixSize - 1.0) * (double) matrixSize * (double) matrixSize;
            return (operations / 1000000000) /* 10^9 operations */
                   / ((double) ms / 1000) /* per second */;
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