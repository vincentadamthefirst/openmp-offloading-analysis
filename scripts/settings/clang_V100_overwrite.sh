# slurm settings
ACCOUNT="p_sp_adam"
NUM_CORES="8"
MEM_PER_CORE="5000"
RUN_NAME="clang_V100_overwrite"
RESERVATION=""
TIME="04:00:00"
PARTITION="ml"
SLURM_ADDITIONAL=("--gres=gpu:1")
# benchmark settings
COMPILERS=("clang")
ADDITIONAL_FLAGS=("--cuda-path=/opt/nvidia/hpc_sdk/Linux_ppc64le/22.5/cuda/11.7/" "-DOVERWRITE_DEFAULT_NUMS")
TARGET_TRIPLE="nvptx64-nvidia-cuda"
TARGET_ARCH="sm_70"
SINGULARITY="clang_power9.sif"
SINGULARITY_PATH="/opt/nvidia/hpc_sdk/Linux_ppc64le/22.5/compilers/bin:/.local/bin"