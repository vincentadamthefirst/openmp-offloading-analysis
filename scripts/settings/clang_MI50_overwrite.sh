# slurm settings
ACCOUNT="p_taurusamdgpu"
NUM_CORES="64"
MEM_PER_CORE="5000"
RUN_NAME="clang_MI50_overwrite"
RESERVATION="AMDGPU"
TIME="06:00:00"
PARTITION="romeo"
SLURM_ADDITIONAL=("")
# benchmark settings
COMPILERS=("clang")
ADDITIONAL_FLAGS=("-DOVERWRITE_DEFAULT_NUMS" "-DNO_MEM_DIRECTIVES" "-Xopenmp-target=amdgcn-amd-amdhsa" "-march=gfx906")
TARGET_TRIPLE="amdgcn-amd-amdhsa"
TARGET_ARCH="gfx906"
SINGULARITY="clang-rocm.sif"
SINGULARITY_PATH="/opt/bin:/.local/bin"