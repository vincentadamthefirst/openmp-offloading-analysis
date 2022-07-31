export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/compilers/bin:/.local/bin:/opt/bin:$PATH

# Set what exactly should be compiled
# TO_COMPILE="${BASH_SOURCE%/*}/../src/openmp/benchmark/benchmark.cpp"
TO_COMPILE="${BASH_SOURCE%/*}/../src/openmp/loop_ordering/loop_order.cpp"
# Base Name of the output
# OUT_BASE="benchmark_"
OUT_BASE="loop_order_"

SETTINGS_FOLDER="${BASH_SOURCE%/*}/settings/*"
IGNORE=false

for SETTINGS in $SETTINGS_FOLDER; do
  if [[ "${SETTINGS}" == *"!"* ]]; then
    continue
  fi

  BASE_NAME=$(basename $SETTINGS)

  if [[ ! " $*[*] " =~ ${BASE_NAME} ]]; then
    continue
  fi

  source $SETTINGS

  # create necessary files
  mkdir -p slurm
  mkdir -p slurm/${RUN_NAME}
  JOB_FILE="slurm/${RUN_NAME}/jobscript.sh"
  SINGULARITY_FILE="slurm/${RUN_NAME}/singularity_script.sh"

  # print file header
  echo "#!/bin/bash" > $JOB_FILE
  echo "" >> $JOB_FILE
  echo "#SBATCH -N 1" >> $JOB_FILE
  echo "#SBATCH -c ${NUM_CORES}" >> $JOB_FILE
  echo "#SBATCH --account=${ACCOUNT}" >> $JOB_FILE
  if [ "${MEM_PER_CORE}" != "" ]; then
    echo "#SBATCH --mem-per-cpu=${MEM_PER_CORE}" >> $JOB_FILE
  fi
  echo "#SBATCH --job-name=${RUN_NAME}" >> $JOB_FILE
  echo "#SBATCH --time=${TIME}" >> $JOB_FILE
  if [ "${RESERVATION}" != "" ]; then
    echo "#SBATCH --reservation=${RESERVATION}" >> $JOB_FILE
  fi
  echo "#SBATCH --partition=${PARTITION}" >> $JOB_FILE
  echo "#SBATCH --output=slurm/${RUN_NAME}/slurm_out.txt" >> $JOB_FILE
  echo "#SBATCH --mail-user=vincent.adam@mailbox.tu-dresden.de" >> $JOB_FILE

  for EXTRA in "${SLURM_ADDITIONAL[@]}"; do
    echo "#SBATCH ${EXTRA}" >> $JOB_FILE
  done

  echo "" >> $JOB_FILE

  if [ "${SINGULARITY}" != "" ]; then
    echo "singularity exec --nv ${SINGULARITY} ./slurm/${RUN_NAME}/singularity_script.sh" >> $JOB_FILE
    echo "export PATH=${SINGULARITY_PATH}:\$PATH" > $SINGULARITY_FILE
  else
    echo "./slurm/${RUN_NAME}/singularity_script.sh" >> $JOB_FILE
    echo "" > $SINGULARITY_FILE
  fi

  ADD_FLAGS=""
  for FLAG in "${ADDITIONAL_FLAGS[@]}"; do
    ADD_FLAGS+="-acf=${FLAG} "
  done

  for COMPILER in "${COMPILERS[@]}"; do
    echo "" >> $SINGULARITY_FILE
    echo "# ${COMPILER}" >> $SINGULARITY_FILE
    for MATRIX_SIZE in "${SIZES[@]}"; do
      FLAGS="-acf=-DMATRIX_SIZE=${MATRIX_SIZE} ${ADD_FLAGS}"
      "${BASH_SOURCE%/*}/COMPILE.sh" -c="${COMPILER}" "${FLAGS}" -i="${TO_COMPILE}" -tt="${TARGET_TRIPLE}" -ot="${TARGET_ARCH}" -v -o="slurm/${RUN_NAME}/${OUT_BASE}${COMPILER}_${MATRIX_SIZE}"
      chmod +x "./slurm/${RUN_NAME}/${OUT_BASE}${COMPILER}_${MATRIX_SIZE}"
      echo "./slurm/${RUN_NAME}/${OUT_BASE}${COMPILER}_${MATRIX_SIZE} -m ${METHODS} -r ${REPETITIONS} -w ${WARMUP} -ft csv -o slurm/${RUN_NAME}/${OUT_BASE}${COMPILER}.csv" >> $SINGULARITY_FILE
    done
  done

  chmod +x "slurm/${RUN_NAME}/singularity_script.sh"
done


