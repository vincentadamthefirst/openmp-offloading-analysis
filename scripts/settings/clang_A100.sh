IGNORE=true
# slurm settings
ACCOUNT="p_sp_adam"
NUM_CORES="8"
MEM_PER_CORE="5000"
RUN_NAME="clang_A100"
RESERVATION=""
TIME="04:00:00"
PARTITION="alpha"
SLURM_ADDITIONAL=("--gres=gpu:1")
# benchmark settings
COMPILERS=("clang")
SIZES=("128" "256" "512" "1024" "2048" "4096" "8192")
ADDITIONAL_FLAGS=("--cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cuda/11.7/")
TARGET_TRIPLE="nvptx64-nvidia-cuda"
TARGET_ARCH="sm_80"
METHODS="all"
REPETITIONS=11
WARMUP=5
SINGULARITY="clang_x86_64.sif"
SINGULARITY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/compilers/bin:/.local/bin:/opt/bin"