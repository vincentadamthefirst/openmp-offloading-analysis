#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


markers = ["*", ".", "x", "^", "+", "D", "v", "1", "2", "3", "4", "<", ">"]
max_gflops = {"V100": 7800, "A100": 9700, "MI50": 6600, "A100t": 19500}


def read_files(base_path):
    files = glob.glob(base_path, recursive=True)

    header = []
    total_results = {}

    for file_idx, file in enumerate(files):
        path_without_file_ending = os.path.splitext(file)[0]
        csv_reader = csv.reader(open(file), delimiter=",")
        csv_data = [row for row in csv_reader]
        h, split_file = split_by_column(csv_data, "method_name")  # split the data based on methods
        header = h
        total_results[path_without_file_ending] = split_file

    return header, total_results


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


def plot_method_comparison(method_names, method_titles, matrix_size, total_results, header, title, file_name,
                           include_compilers, compiler_titles, compiler_colors=None, main_y="gflops",
                           secondary_y="none"):
    fig, axs = plt.subplots()
    secondary_axs = None
    if secondary_y == "%":
        secondary_axs = axs.twinx()

    x = np.arange(len(method_names))
    bar_pos = x         # initial bar positions
    bar_width = .1      # constant, set to bar width
    bar_spacing = .01   # space between bars
    bar_count = 0       # to keep count of the bars displayed
    compiler_count = 0

    for compiler_idx, compiler in enumerate(total_results):
        # build compiler name
        c_split = compiler.split("\\")
        compiler_name = c_split[1]
        if ("ow" in compiler) or ("overwrite" in compiler):
            compiler_name += "*"
        compiler_name += " (" + c_split[2] + ")"
        if compiler_name not in include_compilers:
            continue

        values = []
        percentages = []
        for method in method_names:
            rows = total_results[compiler][method]
            relevant_row = next(row for row in rows if
                                get_element_by_name(row, header, "matrixSize") == str(matrix_size))

            tmp = get_element_by_name(relevant_row, header, "gflops_median")
            peak = float(max_gflops[c_split[2]])
            percentages.append((float(tmp) / peak) * 100)
            values.append(float(tmp))

        if compiler_colors is not None:
            axs.bar(bar_pos, values if main_y == "gflops" else percentages, label=compiler_titles[compiler_name],
                    width=bar_width, color=compiler_colors[compiler_name])
        else:
            axs.bar(bar_pos, values if main_y == "gflops" else percentages, label=compiler_titles[compiler_name],
                    width=bar_width)

        if secondary_y == "%":
            # a bit hacky: add bars with the percentage values but make them invisible
            secondary_axs.bar(bar_pos, percentages, label=compiler_name, width=bar_width, color="tab:red", alpha=0)

        bar_pos = [x + bar_width + bar_spacing for x in bar_pos]
        bar_count += 1
        compiler_count += 1

    axs.legend()

    if secondary_y == "%":
        secondary_axs.set_ylabel("% of peak performance")

    axs.set_title(title)
    axs.set_xticks([r + ((bar_width + bar_spacing) * ((bar_count - 1) / 2)) for r in x],
                   method_titles if method_titles is not None else method_names)
    axs.set_ylabel("GFLOPs" if main_y == "gflops" else "% of peak performance")

    # fig.show()
    fig.savefig("images/method_comparisons/" + file_name + ".svg", dpi=300)


def plot_blocked_compiler_comparison(matrix_sizes, total_results, header, title, file_name, include_compilers,
                                     compiler_method_names, compiler_titles, main_y="gflops", secondary_y="none",
                                     compiler_colors=None, tick_rotation=0, file_type="svg"):
    fig, axs = plt.subplots()
    secondary_axs = None
    if secondary_y == "%":
        secondary_axs = axs.twinx()

    for compiler_idx, compiler in enumerate(total_results):
        # build compiler name
        c_split = compiler.split("\\")
        name = c_split[1]
        if ("ow" in compiler) or ("overwrite" in compiler):
            name += "*"
        name += " (" + c_split[2] + ")"
        if name not in include_compilers:
            continue

        rows = total_results[compiler][compiler_method_names[name]]
        rows.sort(key=lambda l: int(get_element_by_name(l, header, "matrixSize")))

        values = []
        percentages = []
        for row in rows:
            tmp = get_element_by_name(row, header, "matrixSize")
            if int(tmp) not in matrix_sizes:
                continue

            tmp = get_element_by_name(row, header, "gflops_median")
            peak = float(max_gflops[c_split[2]])
            values.append((float(tmp) / peak) * 100 if main_y == "%" else float(tmp))
            percentages.append((float(tmp) / peak) * 100)

        if compiler_colors is not None:
            axs.plot(np.array(matrix_sizes).astype("str"), values, marker=markers[compiler_idx],
                     label=compiler_titles[name], color=compiler_colors[name])
        else:
            axs.plot(np.array(matrix_sizes).astype("str"), values, marker=markers[compiler_idx],
                     label=compiler_titles[name])

        if secondary_y == "%":
            # a bit hacky: add bars with the percentage values but make them invisible
            secondary_axs.plot(np.array(matrix_sizes).astype("str"), percentages, label=compiler_titles[name],
                               color="tab:red", alpha=0)

    axs.legend()

    if secondary_y == "%":
        secondary_axs.set_ylabel("% of peak performance")

    fig.autofmt_xdate(rotation=tick_rotation)
    axs.set_title(title)
    axs.set_xlabel("matrix size")
    axs.set_ylabel("GFLOPs" if main_y == "gflops" else "% of peak performance")

    fig.savefig("images/" + file_name + "." + file_type, dpi=300)


def plot_benchmark():
    header, total_results = read_files("benchmark/**/*.csv")

    # generate plots for each method
    mpl.rcParams["figure.figsize"] = [6, 3.4]
    font = {'family': 'normal',
            'size': 8}
    plt.rc('font', **font)

    # Blocked methods, split by GPU
    plot_method_comparison(
        method_names=["blocked_shmem", "blocked_shmem_mem_directives", "blocked_k", "blocked_k_thread_limit"],
        method_titles=["blocked\n(A & B shmem)", "blocked\n(mem. allocators)", "blocked\n(A shmem)",
                       "blocked (A shmem\n+ thread limit=1024)"],
        matrix_size=4096, total_results=total_results, header=header,
        title="blocked methods on A100 (matrix size = 4096)", file_name="method_comparisons/blocked_A100",
        include_compilers=["clang* (A100)", "nvc (A100)"],
        compiler_titles={"clang* (A100)": "clang++", "nvc (A100)": "nvc++", "xlc* (A100)": "xlc++"},
        compiler_colors={"clang* (A100)": "tab:blue", "nvc (A100)": "tab:green"},
        main_y="gflops", secondary_y="%"
    )

    plot_method_comparison(
        method_names=["blocked_shmem", "blocked_shmem_mem_directives", "blocked_k", "blocked_k_thread_limit"],
        method_titles=["blocked\n(A & B shmem)", "blocked (A & B shmem\n+ mem. allocators)", "blocked\n(A shmem)",
                       "blocked (A shmem\n+ thread limit=1024)"],
        matrix_size=4096, total_results=total_results, header=header,
        title="blocked methods on V100 (matrix size = 4096)", file_name="method_comparisons/blocked_V100",
        include_compilers=["clang* (V100)", "nvc (V100)", "xlc* (V100)"],
        compiler_titles={"clang* (V100)": "clang++", "nvc (V100)": "nvc++", "xlc* (V100)": "xlc++"},
        compiler_colors={"clang* (V100)": "tab:blue", "nvc (V100)": "tab:green", "xlc* (V100)": "tab:orange"},
        main_y="gflops", secondary_y="%"
    )

    # Comparison CUDA & other compilers
    plot_blocked_compiler_comparison(
        matrix_sizes=[128, 256, 512, 1024, 2048, 4096], total_results=total_results, header=header,
        title="blocked (A & B in shared memory) on A100 (matrix size = 4096)",
        file_name="method_comparisons/cuda_comparison_A100",
        include_compilers=["clang (A100)", "nvc (A100)", "cuda (A100)"],
        compiler_colors={"clang (A100)": "tab:blue", "nvc (A100)": "tab:green", "cuda (A100)": "tab:cyan"},
        compiler_method_names={"clang (A100)": "blocked_shmem", "nvc (A100)": "blocked_shmem", "cuda (A100)": "CUDA"},
        compiler_titles={"clang (A100)": "clang++", "nvc (A100)": "nvc++", "cuda (A100)": "nvcc"},
        secondary_y="%"
    )

    plot_blocked_compiler_comparison(
        matrix_sizes=[128, 256, 512, 1024, 2048, 4096], total_results=total_results, header=header,
        title="blocked (A & B in shared memory) on V100 (matrix size = 4096)",
        file_name="method_comparisons/cuda_comparison_V100",
        include_compilers=["clang (V100)", "nvc (V100)", "xlc (V100)", "cuda (V100)"],
        compiler_colors={"clang (V100)": "tab:blue", "nvc (V100)": "tab:green", "cuda (V100)": "tab:cyan",
                         "xlc (V100)": "tab:orange"},
        compiler_method_names={"clang (V100)": "blocked_shmem", "nvc (V100)": "blocked_shmem", "cuda (V100)": "CUDA",
                               "xlc (V100)": "blocked_shmem"},
        compiler_titles={"clang (V100)": "clang++", "nvc (V100)": "nvc++", "cuda (V100)": "nvcc",
                         "xlc (V100)": "xlc++"},
        secondary_y="%"
    )


def plot_blocked():
    header, total_results = read_files("blocked/**/*.csv")

    # generate plots for each method
    mpl.rcParams["figure.figsize"] = [6, 3.4]
    font = {'family': 'normal',
            'size': 8}
    plt.rc('font', **font)

    plot_blocked_compiler_comparison(
        matrix_sizes=[256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096],
        total_results=total_results, header=header, 
        include_compilers=["clang (V100)", "nvc (V100)", "xlc (V100)"],
        title="blocked (only A in shared memory) on V100", file_name="method_comparisons/blocked_only_A_V100",
        compiler_colors={"clang (V100)": "tab:blue", "nvc (V100)": "tab:green", "xlc (V100)": "tab:orange"},
        compiler_method_names={"clang (V100)": "blocked_k_thread_limit", "nvc (V100)": "blocked_k", 
                               "xlc (V100)": "blocked_k_thread_limit"},
        compiler_titles={"clang (V100)": "clang++", "nvc (V100)": "nvc++", "xlc (V100)": "xlc++"},
        secondary_y="%", tick_rotation=45
    )

    plot_blocked_compiler_comparison(
        matrix_sizes=[256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096],
        total_results=total_results, header=header, 
        include_compilers=["clang (A100)", "nvc (A100)"],
        title="blocked (only A in shared memory) on A100", file_name="method_comparisons/blocked_only_A_A100",
        compiler_colors={"clang (A100)": "tab:blue", "nvc (A100)": "tab:green"},
        compiler_method_names={"clang (A100)": "blocked_k_thread_limit", "nvc (A100)": "blocked_k"},
        compiler_titles={"clang (A100)": "clang++", "nvc (A100)": "nvc++"},
        secondary_y="%", tick_rotation=45
    )

    plot_blocked_compiler_comparison(
        matrix_sizes=[256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096],
        total_results=total_results, header=header,
        include_compilers=["clang (V100)", "nvc (V100)", "xlc (V100)"],
        title="blocked (A & B in shared memory) on V100", file_name="method_comparisons/blocked_V100",
        compiler_colors={"clang (V100)": "tab:blue", "nvc (V100)": "tab:green", "xlc (V100)": "tab:orange"},
        compiler_method_names={"clang (V100)": "blocked_shmem", "nvc (V100)": "blocked_shmem",
                               "xlc (V100)": "blocked_shmem"},
        compiler_titles={"clang (V100)": "clang++", "nvc (V100)": "nvc++", "xlc (V100)": "xlc++"},
        secondary_y="%", tick_rotation=45
    )

    plot_blocked_compiler_comparison(
        matrix_sizes=[256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096],
        total_results=total_results, header=header,
        include_compilers=["clang (A100)", "nvc (A100)",],
        title="blocked (A & B in shared memory) on A100", file_name="method_comparisons/blocked_A100",
        compiler_colors={"clang (A100)": "tab:blue", "nvc (A100)": "tab:green"},
        compiler_method_names={"clang (A100)": "blocked_shmem", "nvc (A100)": "blocked_shmem"},
        compiler_titles={"clang (A100)": "clang++", "nvc (A100)": "nvc++"},
        secondary_y="%", tick_rotation=45
    )


def plot_cublas():
    header, total_results = read_files("blas/**/*.csv")

    mpl.rcParams["figure.figsize"] = [6, 3.4]
    font = {'family': 'normal',
            'size': 8}
    plt.rc('font', **font)

    plot_blocked_compiler_comparison(
        matrix_sizes=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        total_results=total_results, header=header,
        include_compilers=["cuBLAS (A100t)", "cuBLAS (V100)", ],
        title="cuBLAS performance", file_name="cublas/cublas",
        compiler_colors={"cuBLAS (A100t)": "tab:blue", "cuBLAS (V100)": "tab:orange"},
        compiler_method_names={"cuBLAS (A100t)": "cuBLAS", "cuBLAS (V100)": "cuBLAS"},
        compiler_titles={"cuBLAS (A100t)": "nvcc (A100)", "cuBLAS (V100)": "nvcc (V100)"},
        main_y="%", tick_rotation=0
    )

def main():
    # plot_benchmark()
    # plot_loop_orders()
    # plot_blocked()
    plot_cublas()
    exit(0)


if __name__ == "__main__":
    main()
