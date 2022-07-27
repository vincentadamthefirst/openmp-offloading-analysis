IGNORE=true
# slurm settings
ACCOUNT="p_sp_adam"
NUM_CORES="8"
MEM_PER_CORE="5000"
RUN_NAME="nvc_power9"
RESERVATION=""
TIME="04:00:00"
PARTITION="ml"
SLURM_ADDITIONAL=("--gres=gpu:1")
# benchmark settings
COMPILERS=("nvc")
SIZES=("128" "256" "512" "1024" "2048" "4096" "8192")
ADDITIONAL_FLAGS=()
TARGET_TRIPLE="nvptx64-nvidia-cuda"
TARGET_ARCH="cc70"
METHODS="all"
REPETITIONS=11
WARMUP=5
SINGULARITY="clang_power9.sif"
SINGULARITY_PATH="/opt/nvidia/hpc_sdk/Linux_ppc64le/22.5/compilers/bin:/.local/bin"