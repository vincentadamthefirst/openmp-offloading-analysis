#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <map>
#include <set>

#include "matmul.cpp"
#include "../../libs/cmdparser.hpp"

static std::map<std::string, std::vector<Method>> methodGroups = {
        {"basic", {Method::IJK, Method::IJK_COLLAPSED, Method::IKJ, Method::JIK, Method::JIK_COLLAPSED, Method::JKI}},
        {"collapsed", {Method::JIK_COLLAPSED, Method::IJK_COLLAPSED, Method::IJK_COLLAPSED_LOOP}},
        {"loop", {Method::IJK_COLLAPSED_LOOP, Method::IJK_LOOP}},
        {"tiled", {Method::TILED_SHMEM_MEM_DIRECTIVES, Method::TILED_SHMEM}},
};


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
    parser.set_optional<bool>("np", "no-print", false, "Disables the printing to file.");
}

int main(int argc, char* argv[]) {
    // setting up the CLI parser
    cli::Parser parser(argc, argv);
    configureParser(parser);
    parser.run_and_exit_if_error();

    // check if the user wants to print the available matrix multiplication methods
    auto printMethods = parser.get<bool>("p");
    if (printMethods) {
        for (const auto &pair : methodGroups) {
            std::cout << pair.first << ":" << std::endl;
            for (const auto &method : pair.second) {
                std::cout << "  - " << methodNamesMappingReversed[method] << std::endl;
            }
        }
        std::cout << std::endl << "all: contains all methods" << std::endl;
        exit(0);
    }

    // retrieve all methods that should be run
    auto methodsToRunParser = parser.get<std::vector<std::string>>("m");
    auto methodsToRun = std::set<Method>();
    for (const auto& parserMethod : methodsToRunParser) {
        if (methodNamesMapping.find(parserMethod) != methodNamesMapping.end()) {
            methodsToRun.insert(methodNamesMapping[parserMethod]);
        } else if (methodGroups.find(parserMethod) != methodGroups.end()) {
            auto methodVector = methodGroups[parserMethod];
            std::copy(methodVector.begin(), methodVector.end(), std::inserter(methodsToRun, methodsToRun.end()));
        } else if (parserMethod == "all") {
            for (const auto& pair : methodGroups) {
                auto methodVector = methodGroups[pair.first];
                std::copy(methodVector.begin(), methodVector.end(), std::inserter(methodsToRun, methodsToRun.end()));
            }
            break;
        } else {
            std::cout << parserMethod << " is not a valid method, use --print_methods to see a list of methods."
                      << std::endl;
        }
    }

    // get flags & smaller values
    auto verbose = parser.get<bool>("v");
    auto csv = parser.get<std::string>("ft") == "csv";
    auto repetitions = parser.get<int>("r");
    auto file = parser.get<std::string>("f");
    auto compare = parser.get<bool>("c");
    auto noPrint = parser.get<bool>("np");

    MatrixMultiplication matrixMultiplication(file, verbose, csv);
    matrixMultiplication.enableCheck(compare).enableRepetitions(repetitions);

    // FIXME remove after debugging
    matrixMultiplication.printInfo();

    size_t largestMethodNameLength = 0;
    for (const auto& pair : methodNamesMapping) {
        largestMethodNameLength = std::max(largestMethodNameLength, pair.first.length());
    }

    auto it = methodsToRun.begin();
    for (auto i = 0; i < methodsToRun.size(); i++) {
        matrixMultiplication.execute(*it);

        if (!verbose)
            Helper::IO::printProgress((i + 1) / (double) methodsToRunParser.size(),
                                      Helper::IO::padRight("(" + methodNamesMappingReversed[*it] + ")",
                                                           largestMethodNameLength + 2),
                                      i == methodsToRunParser.size() - 1);

        it++;
    }

    if (!noPrint)
        matrixMultiplication.writeFile();
}
