# slurm settings
ACCOUNT="p_taurusamdgpu"
NUM_CORES="64"
MEM_PER_CORE="5000"
GPU="MI50"
RESERVATION="AMDGPU"
TIME="04:00:00"
PARTITION="romeo"
SLURM_ADDITIONAL=("-w taurusi7190")
# benchmark settings
COMPILER="rocm"
ADDITIONAL_FLAGS=()
TARGET_TRIPLE="amdgcn-amd-amdhsa"
TARGET_ARCH="gfx90a"
SINGULARITY="rocm-2.sif"
SINGULARITY_PATH="/opt/rocm-5.2.0/llvm/bin/:/.local/bin"
