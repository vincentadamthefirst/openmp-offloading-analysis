#include <iostream>
#include <vector>
#include <map>
#include <set>

#include "matmul.cpp"

void configureParser(cli::Parser& parser) {
    Helper::IO::basicParserSetup(parser);
    parser.set_optional<std::vector<std::string>>("m", "methods", {"all"},
                                                  "The methods to run (comma separated list). To see a list "
                                                  "of all methods use '--print_methods'.");
    parser.set_optional<bool>("p", "print_methods", false, "Prints all available methods.");
    parser.set_optional<std::string>("s", "suffix", "", "Adds a suffix to method output names.");
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
                std::cout << "  - " << methodNamesMapping[method] << std::endl;
            }
        }
        std::cout << std::endl << "all: contains all methods" << std::endl;
        exit(0);
    }

    // retrieve all methods that should be run
    auto methodsToRunParser = parser.get<std::vector<std::string>>("m");
    auto methodsToRun = std::set<Method>();
    for (const auto& parserMethod : methodsToRunParser) {
        if (methodNamesMappingReversed.find(parserMethod) != methodNamesMappingReversed.end()) {
            methodsToRun.insert(methodNamesMappingReversed[parserMethod]);
        } else if (methodGroups.find(parserMethod) != methodGroups.end()) {
            auto methodVector = methodGroups[parserMethod];
            std::copy(methodVector.begin(), methodVector.end(), std::inserter(methodsToRun, methodsToRun.end()));
        } else if (parserMethod == "all") {
            for (const auto& pair : methodGroups) {
                auto methodVector = methodGroups[pair.first];
                std::copy(methodVector.begin(), methodVector.end(), std::inserter(methodsToRun, methodsToRun.end()));
            }
        } else {
            std::cout << parserMethod << " is not a valid method, use --print_methods to see a list of methods."
                      << std::endl;
        }
    }

    if (methodsToRun.empty()) {
        std::cout << "Did not find any methods to run. Exiting." << std::endl;
        exit(0);
    }

    // get flags & smaller values
    auto verbose = parser.get<bool>("v");
    auto csv = parser.get<std::string>("ft") == "csv";
    auto suffix = parser.get<std::string>("s");
    auto repetitions = parser.get<int>("r");
    auto warmup = parser.get<int>("w");
    auto compare = parser.get<bool>("c");
    auto noOutput = parser.get<bool>("no");
    auto file = noOutput ? "NO_OUTPUT_FILE" : parser.get<std::string>("o");

    MatrixMultiplication matrixMultiplication(file, verbose, csv);
    matrixMultiplication.enableCheck(compare).enableRepetitions(repetitions).enableWarmup(warmup);
    matrixMultiplication.withSuffix(suffix);

    size_t largestMethodNameLength = 0;
    for (const auto& pair : methodNamesMappingReversed) {
        largestMethodNameLength = std::max(largestMethodNameLength, pair.first.length());
    }

    auto it = methodsToRun.begin();
    for (auto i = 0; i < methodsToRun.size(); i++) {
        matrixMultiplication.execute(*it);

        if (!verbose)
            Helper::IO::printProgress((i + 1) / (double) methodsToRunParser.size(),
                                      Helper::IO::padRight("(" + methodNamesMapping[*it] + ")",
                                                           largestMethodNameLength + 2),
                                      i == methodsToRunParser.size() - 1);

        it++;
    }

    if (!noOutput)
        matrixMultiplication.writeFile();
}
