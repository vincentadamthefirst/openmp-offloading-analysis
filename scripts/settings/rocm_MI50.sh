IGNORE=false
# slurm settings
ACCOUNT="p_taurusamdgpu"
NUM_CORES="64"
MEM_PER_CORE="5000"
RUN_NAME="rocm_MI50"
RESERVATION="AMDGPU"
TIME="04:00:00"
PARTITION="romeo"
SLURM_ADDITIONAL=("")
# benchmark settings
COMPILERS=("rocm")
SIZES=("128" "256" "512" "1024" "2048" "4096" "8192")
ADDITIONAL_FLAGS=("")
TARGET_TRIPLE="amdgcn-amd-amdhsa"
TARGET_ARCH="gfx906"
METHODS="all"
REPETITIONS=11
WARMUP=5
SINGULARITY="rocm-2.sif"
SINGULARITY_PATH="/opt/rocm-5.2.0/llvm/bin/:/.local/bin"