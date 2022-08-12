# slurm settings
ACCOUNT="p_sp_adam"
NUM_CORES="8"
MEM_PER_CORE="5000"
GPU="V100"
RESERVATION=""
TIME="04:00:00"
PARTITION="ml"
SLURM_ADDITIONAL=("--gres=gpu:1")
# benchmark settings
COMPILER="cuda"
ADDITIONAL_FLAGS=()
TARGET_TRIPLE=""
TARGET_ARCH="sm_70"
SINGULARITY="clang_power9.sif"
SINGULARITY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/compilers/bin:/.local/bin:/opt/bin"
