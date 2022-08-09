# slurm settings
ACCOUNT="p_taurusamdgpu"
NUM_CORES="64"
MEM_PER_CORE="5000"
RUN_NAME="rocm_MI250"
RESERVATION="AMDGPU"
TIME="04:00:00"
PARTITION="romeo"
SLURM_ADDITIONAL=("-w taurusi7190")
# benchmark settings
COMPILERS=("rocm")
ADDITIONAL_FLAGS=("")
TARGET_TRIPLE="amdgcn-amd-amdhsa"
TARGET_ARCH="gfx90a"
SINGULARITY="rocm-2.sif"
SINGULARITY_PATH="/opt/rocm-5.2.0/llvm/bin/:/.local/bin"