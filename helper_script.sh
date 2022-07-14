#!/bin/bash

# Quick BASH script to compile and run the code for multiple matrix sizes etc.

RUN=false

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
    --methods=*)
      METHODS="${i#*=}"
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
for compiler in "${COMPILER_ARRAY[@]}"
do
  echo "Compiler: ${compiler}"
  for data_type in "${DATA_TYPES_ARRAY[@]}"
  do
    echo "  Data Type: ${data_type}"
    for matrix_size in "${MATRIX_SIZES_ARRAY[@]}"
    do
      echo "    Matrix Size: ${matrix_size}"
      for tile_size in "${TILE_SIZES_ARRAY[@]}"
      do
        echo "      Tile Size: ${tile_size}"
        if [ "${compiler}" == "clang" ]; then
          clang++ -std=c++11 -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-version=51 --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/cuda/ src/openmp/main_openmp.cpp -o tmp/${compiler}_omp_${data_type}_${matrix_size}_${tile_size} -DTILE_SIZE=${tile_size} -DMATRIX_SIZE=${matrix_size} -DDATA_TYPE=${data_type} -DNO_LOOP_DIRECTIVES
          chmod +x tmp/clang_omp_${data_type}_${matrix_size}_${tile_size}
        elif [ "${compiler}" == "nvc" ]; then
          nvc++ -std=c++11 -O3 -mp=gpu -target=gpu src/openmp/main_openmp.cpp -o tmp/${compiler}_omp_${data_type}_${matrix_size}_${tile_size} -DTILE_SIZE=${tile_size} -DMATRIX_SIZE=${matrix_size} -DDATA_TYPE=${data_type} -DNO_MEM_DIRECTIVES
          chmod +x tmp/nvc_omp_${data_type}_${matrix_size}_${tile_size}
        fi
      done
    done
  done
done

if ${RUN} ; then
  # Running the generated executables
  echo "Running the generated scripts..."
  echo ""
  mkdir -p results
  for compiler in "${COMPILER_ARRAY[@]}"
  do
    for data_type in "${DATA_TYPES_ARRAY[@]}"
    do
      for matrix_size in "${MATRIX_SIZES_ARRAY[@]}"
      do
        for tile_size in "${TILE_SIZES_ARRAY[@]}"
        do
          ./tmp/${compiler}_omp_${data_type}_${matrix_size}_${tile_size} -v -r ${REPETITIONS} -m ${METHODS} -ft csv -f results/${compiler}.csv
        done
      done
    done
  done
else
  # generate run script
  echo "Generating runscript.sh"
  FILE="tmp/runscript.sh"

  echo "#!/bin/bash" > $FILE
  for compiler in "${COMPILER_ARRAY[@]}"
  do
    for data_type in "${DATA_TYPES_ARRAY[@]}"
    do
      for matrix_size in "${MATRIX_SIZES_ARRAY[@]}"
      do
        for tile_size in "${TILE_SIZES_ARRAY[@]}"
        do
          echo "./tmp/${compiler}_omp_${data_type}_${matrix_size}_${tile_size} -v -r ${REPETITIONS} -m ${METHODS} -ft csv -f results/${compiler}.csv" >> $FILE
        done
      done
    done
  done

  chmod +x ${FILE}
fi
