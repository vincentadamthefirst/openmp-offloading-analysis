COMPILER_NAME="NULL"
ADDITIONAL_COMPILE_FLAGS=()
OFFLOAD_TARGET="NULL"
TARGET_TRIPLE="NULL"
INPUT_FILENAME="NULL"
OUTPUT_FILENAME="NULL"
CPU_TARGET="NULL"

for i in "$@"; do
  case $i in
    -ct=*|--cpu-target=*)
      CPU_TARGET="${i#*=}"
      shift
      ;;
    -c=*|--compiler=*)
      COMPILER_NAME="${i#*=}"
      shift
      ;;
    -i=*|--input=*)
      INPUT_FILENAME="${i#*=}"
      shift
      ;;
    -ot=*|--offload-target=*)
      OFFLOAD_TARGET="${i#*=}"
      shift
      ;;
    -o=*|--output=*)
      OUTPUT_FILENAME="${i#*=}"
      shift
      ;;
    -tt=*|--target-triple=*)
      TARGET_TRIPLE="${i#*=}"
      shift
      ;;
    -v|--verbose)
      VERBOSE=true
      shift
      ;;
    -d|--debug)
      DEBUG=true
      shift
      ;;
    -acf=*|--additional_compile_flags=*)
      ADDITIONAL_COMPILE_FLAGS+=("${i#*=}")
      shift # past argument=value
      ;;
    -*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

# perform basic input checking
if [ "${COMPILER_NAME}" == "NULL" ]; then
  >&2 echo "Did not specify a compiler name."
  exit 0
fi

if [ "${OFFLOAD_TARGET}" == "NULL" ]; then
  >&2 echo "Did not specify an offload target."
  exit 0
fi

if [ "${TARGET_TRIPLE}" == "NULL" ]; then
  >&2 echo "Did not specify a target triple."
  exit 0
fi

if [ "${INPUT_FILENAME}" == "NULL" ]; then
  >&2 echo "Did not specify an input file."
  exit 0
fi

# find some important values
if [ "${CPU_TARGET}" == "NULL" ]; then
  UNAMEP=$(uname -p)
  CPU_TARGET="${UNAMEP}-pc-linux-gnu"
  if [ "${UNAMEP}" == "ppc64le" ]; then
    CPU_TARGET="ppc64le-linux-gnu"
  fi

  if [ ${VERBOSE} ]; then
    echo "No CPU_TARGET specified (-ct=<...>), using system info: ${CPU_TARGET}"
  fi
fi

# import the definitions
source "${BASH_SOURCE%/*}/SETTINGS.sh"

# uppercasing the compiler name
COMPILER=${COMPILER_NAME^^}
# the total call to the compiler
COMPILER_SETTINGS=""
# retrieving the command for the compiler
COMPILER_CMD=${COMPILER}_COMPILER

# build start of compilation sequence (-std=c++XX, -OY, ...)
if [ ${DEBUG} ]; then
  COMPILER_DEBUG_FLAGS=${COMPILER}_DEBUG_FLAGS
  COMPILER_SETTINGS="${COMPILER_SETTINGS} ${CFLAGS_DEBUG} ${!COMPILER_DEBUG_FLAGS}"
else
  COMPILER_DEFAULT_FLAGS=${COMPILER}_FLAGS
  COMPILER_SETTINGS="${COMPILER_SETTINGS} ${CFLAGS} ${!COMPILER_DEFAULT_FLAGS}"
fi

# append OpenMP related settings
COMPILER_OPENMP_FLAGS=${COMPILER}_OPENMP_FLAGS
OPENMP_FLAGS=${!COMPILER_OPENMP_FLAGS}
OPENMP_FLAGS_1=${OPENMP_FLAGS//"%HOST_TRIPLE%"/$CPU_TARGET}
OPENMP_FLAGS_2=${OPENMP_FLAGS_1//"%TARGET_TRIPLE%"/$TARGET_TRIPLE}
OPENMP_FLAGS_3=${OPENMP_FLAGS_2//"%OFFLOAD_ARCH%"/$OFFLOAD_TARGET}
COMPILER_SETTINGS="${COMPILER_SETTINGS} ${OPENMP_FLAGS_3}"

# append source code to compile
if [ "${OUTPUT_FILENAME}" == "NULL" ]; then
  OUTPUT_FILENAME="${INPUT_FILENAME%.*}"
  OUTPUT_FILENAME="${OUTPUT_FILENAME}_${COMPILER}"
  if [ ${VERBOSE} ]; then
    echo "Did not specify an output file. Using default one: "
    echo "${OUTPUT_FILENAME}"
  fi
fi
COMPILER_SETTINGS="${COMPILER_SETTINGS} ${INPUT_FILENAME} -o ${OUTPUT_FILENAME}"

# append benchmark related settings
COMPILER_BENCHMARK_FLAGS=${COMPILER}_BENCHMARK_FLAGS
COMPILER_SETTINGS="${COMPILER_SETTINGS} ${!COMPILER_BENCHMARK_FLAGS}"

# append any additional compile flags
COMPILER_SETTINGS="${COMPILER_SETTINGS} ${ADDITIONAL_COMPILE_FLAGS[*]}"

if [ ${VERBOSE} ]; then
  echo "Total compile settings: "
  echo "${COMPILER_SETTINGS}"
fi

# execute the command
${!COMPILER_CMD} ${COMPILER_SETTINGS}