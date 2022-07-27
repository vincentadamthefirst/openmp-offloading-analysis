COMPILER_NAME=""

TMP=$@

for i in "$@"; do
  case $i in
    -c=*|--compiler=*)
      COMPILER_NAME="${i#*=}"
      shift
      ;;
    *)
      ;;
  esac
done

# uppercasing the compiler name
COMPILER=${COMPILER_NAME^^}

if [ "${COMPILER}" == "CUDA" ]; then
  "${BASH_SOURCE%/*}"/COMPILE_OTHER.sh $TMP
elif [ "${COMPILER}" == "HIP" ]; then
  "${BASH_SOURCE%/*}"/COMPILE_OTHER.sh $TMP
else
  "${BASH_SOURCE%/*}"/COMPILE_OPENMP.sh $TMP
fi