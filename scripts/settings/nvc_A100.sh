# slurm settings
ACCOUNT="p_sp_adam"
NUM_CORES="8"
MEM_PER_CORE="5000"
GPU="A100"
RESERVATION=""
TIME="04:00:00"
PARTITION="alpha"
SLURM_ADDITIONAL=("--gres=gpu:1")
# benchmark settings
COMPILER="nvc"
ADDITIONAL_FLAGS=()
TARGET_TRIPLE="nvptx64-nvidia-cuda"
TARGET_ARCH="cc80"
SINGULARITY="clang_x86_64.sif"
SINGULARITY_PATH="/opt/nvidia/hpc_sdk/Linux_ppc64le/22.5/compilers/bin:/.local/bin"
