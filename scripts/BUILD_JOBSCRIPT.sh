export PATH=/opt/nvidia/hpc_sdk/Linux_ppc64le/22.5/compilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/compilers/bin:/.local/bin:/opt/bin:/opt/rocm-5.2.0/llvm/bin:$PATH

# file to compile
TO_COMPILE="${BASH_SOURCE%/*}/../src/openmp/benchmark/benchmark.cpp"
# the folder to put the compiled files into
OUT_FOLDER="last"
# the base folder to put the result csvs into
RESULT_FOLDER="results"
# will pe prepended to all output csv files
RUN_NAME="blocked"
# compilation flags to be added for all compilers, some compilers already have hard-coded flags
COMPILE_FLAGS=("-DNO_MEM_DIRECTIVES" "-DNO_LOOP_DIRECTIVES")
# the methods to be run in the benchmark (passed as -m <...>), if left empty no methods will be started
METHODS="blocked_ab blocked_a"
# matrix sizes to compile on
SIZES=("4096")
#SIZES=()
#for i in {256..4096..256}
#do
#    SIZES+=("${i}")
#done
# info on how many iterations should be performed
REPETITIONS=11
WARMUP=5

SETTINGS_FOLDER="${BASH_SOURCE%/*}/settings/*"

for SETTINGS in $SETTINGS_FOLDER; do
  BASE_NAME=$(basename $SETTINGS)

  if [[ ! " $*[*] " =~ ${BASE_NAME} ]]; then
    # file is not in the provided list, skip
    continue
  fi

  source $SETTINGS

  # create necessary files
  mkdir -p ${OUT_FOLDER}
  mkdir -p ${RESULT_FOLDER}
  mkdir -p ${OUT_FOLDER}/${COMPILER}
  mkdir -p ${RESULT_FOLDER}/${COMPILER}
  mkdir -p ${OUT_FOLDER}/${COMPILER}/${GPU}
  mkdir -p ${RESULT_FOLDER}/${COMPILER}/${GPU}
  # files to write to
  JOB_FILE="${OUT_FOLDER}/${COMPILER}/${GPU}/jobscript.sh"
  SINGULARITY_FILE="${OUT_FOLDER}/${COMPILER}/${GPU}/runscript.sh"

  # write the jobscript file header
  echo "#!/bin/bash" > $JOB_FILE
  echo "" >> $JOB_FILE
  echo "#SBATCH -N 1" >> $JOB_FILE
  echo "#SBATCH -c ${NUM_CORES}" >> $JOB_FILE
  echo "#SBATCH --account=${ACCOUNT}" >> $JOB_FILE
  if [ "${MEM_PER_CORE}" != "" ]; then
    echo "#SBATCH --mem-per-cpu=${MEM_PER_CORE}" >> $JOB_FILE
  fi
  echo "#SBATCH --job-name=${COMPILER}_${GPU}" >> $JOB_FILE
  echo "#SBATCH --time=${TIME}" >> $JOB_FILE
  if [ "${RESERVATION}" != "" ]; then
    echo "#SBATCH --reservation=${RESERVATION}" >> $JOB_FILE
  fi
  echo "#SBATCH --partition=${PARTITION}" >> $JOB_FILE
  echo "#SBATCH --output=${OUT_FOLDER}/${COMPILER}/${GPU}/slurm_out.txt" >> $JOB_FILE
  echo "#SBATCH --mail-user=vincent.adam@mailbox.tu-dresden.de" >> $JOB_FILE
  for EXTRA in "${SLURM_ADDITIONAL[@]}"; do
    echo "#SBATCH ${EXTRA}" >> $JOB_FILE
  done
  echo "" >> $JOB_FILE

  # either start singularity or just add the runscript
  if [ "${SINGULARITY}" != "" ]; then
    echo "singularity exec --nv ${SINGULARITY} ./slurm/${COMPILER}/${GPU}/runscript.sh" >> $JOB_FILE
    echo "export PATH=${SINGULARITY_PATH}:\$PATH" > $SINGULARITY_FILE
  else
    echo "./${OUT_FOLDER}/${COMPILER}/${GPU}/runscript.sh" >> $JOB_FILE
    echo "" > $SINGULARITY_FILE
  fi

  # process all additional flags
  ADD_FLAGS=""
  for FLAG in "${ADDITIONAL_FLAGS[@]}"; do
    ADD_FLAGS+="-acf=${FLAG} "
  done
  for FLAG in "${COMPILE_FLAGS[@]}"; do
    ADD_FLAGS+="-acf=${FLAG} "
  done

  for MATRIX_SIZE in "${SIZES[@]}"; do
    FLAGS="-acf=-DMATRIX_SIZE=${MATRIX_SIZE} ${ADD_FLAGS}"
#    echo "-c=${COMPILER} ${FLAGS} -i=${TO_COMPILE} -ot=${TARGET_ARCH} -tt=${TARGET_TRIPLE} -v -o=${OUT_FOLDER}/${COMPILER}/${GPU}/${RUN_NAME}_${MATRIX_SIZE}"
    "${BASH_SOURCE%/*}/COMPILE.sh" -c="${COMPILER}" "${FLAGS}" -i="${TO_COMPILE}" -ot="${TARGET_ARCH}" -tt=${TARGET_TRIPLE} -v -o="${OUT_FOLDER}/${COMPILER}/${GPU}/${RUN_NAME}_${MATRIX_SIZE}"
    chmod +x "./${OUT_FOLDER}/${COMPILER}/${GPU}/${RUN_NAME}_${MATRIX_SIZE}"
    if [ "${METHODS}" == "" ]; then
      echo "./${OUT_FOLDER}/${COMPILER}/${GPU}/${RUN_NAME}_${MATRIX_SIZE} -r ${REPETITIONS} -w ${WARMUP} -ft csv -o ${RESULT_FOLDER}/${COMPILER}/${GPU}/${RUN_NAME}.csv" >> $SINGULARITY_FILE
    else
      echo "./${OUT_FOLDER}/${COMPILER}/${GPU}/${RUN_NAME}_${MATRIX_SIZE} -m ${METHODS} -r ${REPETITIONS} -w ${WARMUP} -ft csv -o ${RESULT_FOLDER}/${COMPILER}/${GPU}/${RUN_NAME}.csv" >> $SINGULARITY_FILE
    fi
  done

  chmod +x "${OUT_FOLDER}/${COMPILER}/${GPU}/runscript.sh"
done



