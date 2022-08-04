# slurm settings
ACCOUNT="p_taurusamdgpu"
NUM_CORES="64"
MEM_PER_CORE="5000"
RUN_NAME="hip_MI50"
RESERVATION="AMDGPU"
TIME="04:00:00"
PARTITION="romeo"
SLURM_ADDITIONAL=("")
# benchmark settings
COMPILERS=("hip")
ADDITIONAL_FLAGS=("")
TARGET_TRIPLE=""
TARGET_ARCH="gfx906"
SINGULARITY="rocm-2.sif"
SINGULARITY_PATH="/opt/rocm-5.2.0/llvm/bin/:/.local/bin"