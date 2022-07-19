#!/bin/bash

# Quick BASH script to compile and run the code for multiple matrix sizes etc.

RUN=false

# file to compile
SRC="src/openmp/main_openmp.cpp"

# compile settings for all compilers
ALL_FLAGS="-std=c++11 -O3"

# compile settings for the different compilers
CLANG_FLAGS="-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-version=51 --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/cuda/"
NVC_FLAGS="-mp=gpu -target=gpu -gpu=cc80"
ROCM_FLAGS="-target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906"
XLC_FLAGS="-qsmp -qoffload -qtgtarch=sm_70"

# compile settings for the program itself
CLANG_SETTINGS="-DNO_LOOP_DIRECTIVES"
NVC_SETTINGS="-DNO_MEM_DIRECTIVES -DNO_NESTED_PARALLEL_FOR"
ROCM_SETTINGS="-DNO_LOOP_DIRECTIVES -DNO_MEM_DIRECTIVES"
XLC_SETTINGS="-DNO_LOOP_DIRECTIVES -DNO_MEM_DIRECTIVES"


for i in "$@"; do
  case $i in
    -m=*|--matrix_sizes=*)
      MATRIX_SIZES="${i#*=}"
      shift # past argument=value
      ;;
    -t=*|--tile_sizes=*)
      TILE_SIZES="${i#*=}"
      shift # past argument=value
      ;;
    -r=*|--repetitions=*)
      REPETITIONS="${i#*=}"
      shift # past argument=value
      ;;
    -d=*|--types=*)
      DATA_TYPES="${i#*=}"
      shift # past argument=value
      ;;
    -c=*|--compilers=*)
      COMPILERS="${i#*=}"
      shift # past argument=value
      ;;
    --run)
      RUN=true
      shift # past argument with no value
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

echo "MATRIX_SIZES = ${MATRIX_SIZES}"
echo "TILE_SIZES   = ${TILE_SIZES}"
echo "REPETITIONS  = ${REPETITIONS}"
echo "DATA_TYPES   = ${DATA_TYPES}"
echo "COMPILERS    = ${COMPILERS}"
echo "RUN          = ${RUN}"
echo ""

MATRIX_SIZES_ARRAY=(${MATRIX_SIZES//,/ })
DATA_TYPES_ARRAY=(${DATA_TYPES//,/ })
TILE_SIZES_ARRAY=(${TILE_SIZES//,/ })
COMPILER_ARRAY=(${COMPILERS//,/ })

# Compiling the code
mkdir -p tmp
echo "Generating runscript.sh"
FILE="tmp/runscript.sh"

for compiler in "${COMPILER_ARRAY[@]}"
do
  echo "Compiler: ${compiler}"
  for data_type in "${DATA_TYPES_ARRAY[@]}"
  do
    echo "  Data Type: ${data_type}"
    for matrix_size in "${MATRIX_SIZES_ARRAY[@]}"
    do
      echo "    Matrix Size: ${matrix_size}"
      CURRENT_SETTINGS="-DMATRIX_SIZE=${matrix_size} -DDATA_TYPE=${data_type}"
      
      if [ "${compiler}" == "clang" ]; then
        clang++ ${ALL_FLAGS} ${CLANG_FLAGS} ${SRC} -o tmp/${compiler}_omp_${data_type}_${matrix_size}_no_tiling ${CURRENT_SETTINGS} ${CLANG_SETTINGS}
      elif [ "${compiler}" == "nvc" ]; then
        nvc++ ${ALL_FLAGS} ${NVC_FLAGS} ${SRC} -o tmp/${compiler}_omp_${data_type}_${matrix_size}_no_tiling ${CURRENT_SETTINGS} ${NVC_SETTINGS}
      elif [ "${compiler}" == "rocm" ]; then
        /opt/rocm-5.2.0/llvm/bin/clang++ ${ALL_FLAGS} ${ROCM_FLAGS} ${SRC} -o tmp/${compiler}_omp_${data_type}_${matrix_size}_no_tiling ${CURRENT_SETTINGS} ${ROCM_SETTINGS}
      elif [ "${compiler}" == "xlc" ]; then
        xlc++ ${ALL_FLAGS} ${XLC_FLAGS} ${SRC} -o tmp/${compiler}_omp_${data_type}_${matrix_size}_no_tiling ${CURRENT_SETTINGS} ${XLC_SETTINGS}
      else
        echo "Unknown compiler: ${compiler}."
      fi
      chmod +x tmp/${compiler}_omp_${data_type}_${matrix_size}_no_tiling
      echo "./tmp/${compiler}_omp_${data_type}_${matrix_size}_no_tiling -v -r ${REPETITIONS} -m basic collapsed loop -ft csv -f results/${compiler}.csv" >> $FILE
      
      for tile_size in "${TILE_SIZES_ARRAY[@]}"
      do
        echo "      Tile Size: ${tile_size}"
        CURRENT_SETTINGS="-DMATRIX_SIZE=${matrix_size} -DDATA_TYPE=${data_type} -DTILE_SIZE=${tile_size}"

        if [ "${compiler}" == "clang" ]; then
          clang++ ${ALL_FLAGS} ${CLANG_FLAGS} ${SRC} -o tmp/${compiler}_omp_${data_type}_${matrix_size}_${tile_size} ${CURRENT_SETTINGS}  ${CLANG_SETTINGS}
        elif [ "${compiler}" == "nvc" ]; then
          nvc++ ${ALL_FLAGS} ${NVC_FLAGS} ${SRC} -o tmp/${compiler}_omp_${data_type}_${matrix_size}_${tile_size} ${CURRENT_SETTINGS} ${NVC_SETTINGS}
        elif [ "${compiler}" == "rocm" ]; then
          /opt/rocm-5.2.0/llvm/bin/clang++ ${ALL_FLAGS} ${ROCM_FLAGS} ${SRC} -o tmp/${compiler}_omp_${data_type}_${matrix_size}_${tile_size} ${CURRENT_SETTINGS} ${ROCM_SETTINGS}
        elif [ "${compiler}" == "xlc" ]; then
          xlc++ ${ALL_FLAGS} ${XLC_FLAGS} ${SRC} -o tmp/${compiler}_omp_${data_type}_${matrix_size}_${tile_size} ${CURRENT_SETTINGS} ${XLC_SETTINGS}
        else
          echo "Unknown compiler: ${compiler}."
        fi
        chmod +x tmp/${compiler}_omp_${data_type}_${matrix_size}_${tile_size}
        echo "./tmp/${compiler}_omp_${data_type}_${matrix_size}_${tile_size} -v -r ${REPETITIONS} -m tiled -ft csv -f results/${compiler}.csv" >> $FILE
      done
    done
  done
done

chmod +x ${FILE}

if ${RUN} ; then
  # Running the generated executables
  echo "Running the generated scripts..."
  echo ""
  mkdir -p results
  ./tmp/runscript.sh
fi
