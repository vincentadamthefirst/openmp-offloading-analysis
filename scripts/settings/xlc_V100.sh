# slurm settings
ACCOUNT="p_sp_adam"
NUM_CORES="4"
MEM_PER_CORE="5000"
RUN_NAME="xlc_V100"
RESERVATION=""
TIME="06:00:00"
PARTITION="ml"
SLURM_ADDITIONAL=("--gres=gpu:1")
# benchmark settings
COMPILERS=("xlc")
ADDITIONAL_FLAGS=()
TARGET_TRIPLE=""
TARGET_ARCH="sm_70"
SINGULARITY=""
SINGULARITY_PATH=""