#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <map>

#include "matmul.cpp"
#include "../../libs/cmdparser.hpp"

void configureParser(cli::Parser& parser) {
    parser.set_optional<std::string>("f", "file", "GENERATE_NEW",
                                     "File the output should be written to. If no file is given a "
                                     "new file will be generated next to the executable.");
    parser.set_optional<std::string>("ft", "file_type", "txt", "Set the formatting of the output. Must be 'txt' or "
                                                               "'csv'.");
    parser.set_optional<bool>("v", "verbose", false, "Enable verbose output.");
    parser.set_optional<int>("r", "repetitions", 11, "Set the amount of repetitions for each matrix.");
    parser.set_optional<std::vector<std::string>>("m", "methods", {"all"},
                                                  "The methods to run (comma separated list). To see a list "
                                                  "of all methods use '--print_methods'.");
    parser.set_optional<bool>("p", "print_methods", false, "Prints all available methods.");
    parser.set_optional<bool>("c", "comparison", false, "Enables result checking of "
                                                        "GPU calculations with previously generated CPU ones.");
}

int main(int argc, char* argv[]) {
    // setting up the CLI parser
    cli::Parser parser(argc, argv);
    configureParser(parser);
    parser.run_and_exit_if_error();

    // check if the user wants to print the available matrix multiplication methods
    auto printMethods = parser.get<bool>("p");
    if (printMethods) {
        for (const auto &pair: methodNamesMapping) {
            std::cout << pair.first << std::endl;
        }
        std::cout << "\nall*\ntiled*\nnon-tiled*\n\n(entries marked with '*' are groups)" << std::endl;
        exit(0);
    }

    // retrieve all methods that should be run
    auto methodsToRun = parser.get<std::vector<std::string>>("m");
    auto withTiled = std::count(methodsToRun.begin(), methodsToRun.end(), "tiled") != 0;
    auto withNonTiled = std::count(methodsToRun.begin(), methodsToRun.end(), "non_tiled") != 0;
    if (withTiled && withNonTiled) methodsToRun = {"all"};
    else if (withTiled)
        methodsToRun = {"tiled_shmem", "tiled_shmem_mem_directives"};
    else if (withNonTiled)
        methodsToRun = {"ijk", "ikj", "jik", "jki", "ijk_collapsed", "jik_collapsed", "ijk_collapsed_loop"};
    if (std::count(methodsToRun.begin(), methodsToRun.end(), "all") != 0) {
        methodsToRun.clear();
        for(auto & it : methodNamesMapping) {
            methodsToRun.push_back(it.first);
        }
    }

    // check correctness of methodsToRun
    for (auto& methodString : methodsToRun) {
        std::transform(methodString.begin(), methodString.end(), methodString.begin(), ::tolower);
        if (methodNamesMapping.find(methodString) == methodNamesMapping.end()) {
            std::cout << "The method " << methodString << " does not exist. The program will be terminated."
                      << std::endl;
            std::cout << "Use '-print_methods' to get a list of possible methods" << std::endl;
            exit(-1);
        }
    }

    // get flags & smaller values
    auto verbose = parser.get<bool>("v");
    auto csv = parser.get<std::string>("ft") == "csv";
    auto repetitions = parser.get<int>("r");
    auto file = parser.get<std::string>("f");
    auto compare = parser.get<bool>("c");

    MatrixMultiplication matrixMultiplication(file, verbose, csv);
    matrixMultiplication.enableCheck(compare).enableRepetitions(repetitions);

    // FIXME remove after debugging
    matrixMultiplication.printInfo();

    size_t largestMethodNameLength = 0;
    for (const auto& pair : methodNamesMapping) {
        largestMethodNameLength = std::max(largestMethodNameLength, pair.first.length());
    }

    for (auto i = 0; i < methodsToRun.size(); i++) {
        auto methodString = methodsToRun[i];
        matrixMultiplication.execute(methodNamesMapping[methodString]);

        if (!verbose)
            Helper::IO::printProgress((i + 1) / (double) methodsToRun.size(),
                                      Helper::IO::padRight("(" + methodString + ")", largestMethodNameLength + 2),
                                      i == methodsToRun.size() - 1);
    }

    matrixMultiplication.writeFile();
}
