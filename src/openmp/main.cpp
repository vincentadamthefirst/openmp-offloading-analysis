#include <iostream>
#include <fstream>

#include "MatrixMultiplication.cpp"
#include "../include/argparse.h"

#define MATRIX_SIZE 8192        // size of the matrices
#define TYPE float              // data type to use

#define REPETITIONS 11           // repetitions for each method
#define BLOCKSIZE_START 2       // start of the blocksize (included), multiplied by 2 until > BLOCKSIZE_END
#define BLOCKSIZE_END 8        // end of the blocksize (included)

// TODO move values into CLI arguments

void run(const std::string& filePath, bool verbose, bool csv) {
    MatrixMultiplication<TYPE> mm(filePath, verbose, csv, MATRIX_SIZE);
    //mm.enableBlockSizes(BLOCKSIZE_START, BLOCKSIZE_END).enableRepetitions(REPETITIONS).enableCheck();
    mm.enableBlockSizes(1,1);
    mm.printInfo();
    mm.execute();
}

int main(int ac, char* av[]) {
    argparse::Parser parser;
    auto flag = parser.AddFlag("verbose", 'v', "Enable verbose output.");
    auto path = parser.AddArg<std::string>("file", 'p', "The file to append the output to.").Required();
    auto type = parser.AddArg<std::string>("format", 'f', "The formatting of the output.").Options({"csv", "txt"}).Default("txt");

    parser.ParseArgs(ac, av);
    run(*path, *flag, *type == "csv");
}