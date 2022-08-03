#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


markers = ["*", ".", "x", "^", "+", "D", "v", "1", "2", "3", "4", "<", ">"]
max_gflops = {"V100": 7800, "A100": 9700, "MI50": 6600}


def get_element_by_name(row, header, column_name):
    for i in range(0, len(header)):
        if header[i] == column_name:
            return row[i]
    raise Exception("Could not find the required column.")


def get_index_by_name(header, column_name):
    for i in range(0, len(header)):
        if header[i] == column_name:
            return i
    raise Exception("Could not find the required column.")


def split_by_column(input_array, column_name):
    split = {}
    header = input_array[0]
    column_index = get_index_by_name(header, column_name)

    for row in input_array[1:]:
        if row[column_index] not in split.keys():
            split[row[column_index]] = []

        split[row[column_index]].append(row)

    return header, split


def get_list_of_values(input_array, column_name):
    column_index = get_index_by_name(input_array[0], column_name)
    values = []
    for row in input_array[1:]:
        if not values.__contains__(row[column_index]):
            values.append(row[column_index])
    return values


def plot_loop_orders():
    files = glob.glob("loop_order/*.csv")
    run_names = ["clang A100", "nvc V100", "rocm MI50", "xlc V100"]

    for run_idx, file in enumerate(files):
        csv_reader = csv.reader(open(file), delimiter=",")
        csv_data = [row for row in csv_reader]
        header, split_file = split_by_column(csv_data, "method_name")  # split the data based on methods

        gflops = {}  # method name mapped to a list of the median gflops
        for key in split_file:
            values = []
            for row in split_file[key]:
                tmp = get_element_by_name(row, header, "gflops_median")
                values.append(float(tmp))
            gflops[key] = values

        x_axis = np.array(get_list_of_values(csv_data, "matrixSize"))

        fig, axs = plt.subplots()
        for idx, key in enumerate(gflops):
            axs.plot(x_axis, gflops[key], label=key, marker=markers[idx])
        axs.legend(loc="upper left")
        axs.set_title("loop order results for " + run_names[run_idx])
        axs.set_xlabel("matrix size")
        axs.set_ylabel("GFLOPs")
        # fig.show()
        fig.savefig(os.path.splitext(file)[0] + ".svg", dpi=300)


def plot_method_over_mat_size(method_names, matrix_sizes, total_results, header, folder, ignore_compilers=[]):
    if ignore_compilers is None:
        ignore_compilers = []
    for method in method_names:
        fig, axs = plt.subplots()
        for compiler_idx, compiler in enumerate(total_results):
            # build compiler name
            c_split = compiler.split("\\")
            name = c_split[1]
            if ("ow" in compiler) or ("overwrite" in compiler):
                name += "*"
            name += " (" + c_split[2] + ")"
            if name in ignore_compilers:  # at this point check if the compiler should be ignored
                continue

            values = []  # method name mapped to a list of the median gflops
            for row in total_results[compiler][method]:
                tmp_mat_size = get_element_by_name(row, header, "matrixSize")
                if int(tmp_mat_size) not in matrix_sizes:
                    continue
                tmp = get_element_by_name(row, header, "gflops_median")
                peak = float(max_gflops[c_split[2]])
                values.append((float(tmp) / peak) * 100)

            axs.plot(np.array(matrix_sizes).astype("str"), values, marker=markers[compiler_idx], label=name)

        # axs.legend(bbox_to_anchor=(1, 1.0))
        axs.legend(loc="best")
        axs.set_title("benchmark results for " + method)
        axs.set_xlabel("matrix size")
        axs.set_ylabel("percentage of peak")
        # fig.show()
        fig.savefig(folder + "/" + method + ".svg", dpi=300)


def plot_method_comparison(method_names, matrix_sizes, total_results, header, file_name, ignore_compilers=[]):
    for matrix_size in matrix_sizes:
        fig, axs = plt.subplots()

        x = np.arange(len(method_names))
        bar_pos = x         # initial bar positions
        bar_width = .1      # constant, set to bar width
        bar_spacing = .01   # space between bars
        bar_count = 0       # to keep count of the bars displayed

        for compiler_idx, compiler in enumerate(total_results):
            # build compiler name
            c_split = compiler.split("\\")
            name = c_split[1]
            if ("ow" in compiler) or ("overwrite" in compiler):
                name += "*"
            name += " (" + c_split[2] + ")"
            if name in ignore_compilers:  # at this point check if the compiler should be ignored
                continue

            values = []
            for method in method_names:
                tup = []
                for row in total_results[compiler][method]:
                    tmp = get_element_by_name(row, header, "matrixSize")
                    if str(matrix_size) == tmp:
                        tup = row
                        break

                tmp = get_element_by_name(tup, header, "gflops_median")
                peak = float(max_gflops[c_split[2]])
                values.append((float(tmp) / peak) * 100)

            axs.bar(bar_pos, values, label=name, width=bar_width)
            bar_pos = [x + bar_width + bar_spacing for x in bar_pos]
            bar_count += 1

        axs.legend()
        axs.set_title("method comparison (matrix size " + str(matrix_size) + ")")
        axs.set_xticks([r + ((bar_width + bar_spacing) * ((bar_count - 1) / 2)) for r in x], method_names)
        axs.set_xlabel("matrix size")
        axs.set_ylabel("% of peak performance")
        # fig.show()
        fig.savefig("images/method_comparisons/" + file_name + ".svg", dpi=300)


def plot_benchmark():
    files = glob.glob("benchmark/**/*.csv", recursive=True)

    header = []
    total_results = {}
    method_names = []

    # constants
    matrix_sizes = [128, 256, 512, 1024, 2048, 4096]

    for file_idx, file in enumerate(files):
        path_without_file_ending = os.path.splitext(file)[0]
        csv_reader = csv.reader(open(file), delimiter=",")
        csv_data = [row for row in csv_reader]
        if file_idx == 0:
            method_names = get_list_of_values(csv_data, "method_name")
        h, split_file = split_by_column(csv_data, "method_name")  # split the data based on methods
        header = h
        total_results[path_without_file_ending] = split_file

    # generate plots for each method
    mpl.rcParams["figure.figsize"] = [10, 6]

    ow_compilers = ["clang* (A100)", "clang* (V100)", "xlc* (V100)"]
    non_ow_compilers = ["clang (A100)", "clang (V100)", "xlc (V100)", "rocm (MI50)", "nvc (V100)", "nvc (A100)"]

    plot_method_over_mat_size(method_names, matrix_sizes, total_results, header,
                              "images/method_over_matrix_sizes/complete")
    plot_method_over_mat_size(method_names, matrix_sizes, total_results, header,
                              "images/method_over_matrix_sizes/without_overwrite",
                              ow_compilers)
    plot_method_over_mat_size(method_names, matrix_sizes, total_results, header,
                              "images/method_over_matrix_sizes/only_overwrite", non_ow_compilers)

    plot_method_comparison(["ijk", "ijk_collapsed", "ijk_collapsed_spmd", "ijk_reduction"], matrix_sizes, total_results,
                           header, "ijk_basic", ow_compilers)
    plot_method_comparison(["blocked_shmem", "blocked_shmem_mem_directives", "blocked_shmem_reduced_bc", "blocked_k"],
                           matrix_sizes, total_results, header, "blocked", ow_compilers)
    plot_method_comparison(["ijk", "ijk_loop", "ijk_collapsed_loop"], matrix_sizes, total_results,
                           header, "ijk_loop", ["clang (A100)", "clang (V100)", "xlc (V100)", "rocm (MI50)",
                                                "clang* (A100)", "clang* (V100)", "xlc* (V100)"])


def main():
    plot_benchmark()
    # plot_loop_orders()
    exit(0)


if __name__ == "__main__":
    main()