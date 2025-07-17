#!/bin/bash

# Run Helios benchmarks and collect runtime information.
#
# Usage:
#   ./run_benchmarks.sh [THREAD_COUNTS] [--verbose] [--log-file <file>] [--samples <samples>] [--help]
#
# Arguments:
#   THREAD_COUNTS - Optional comma or space separated list of thread counts
#                   (e.g., "1,2,4" or "1 2 4"). If omitted the benchmarks run
#                   with a single thread.
#   --verbose     - Print build and run output to the console.
#   --log-file    - Write all output to the specified file. If used together
#                   with --verbose the output is also echoed to the console.
#   --samples     - Comma or space separated list of samples to run (e.g.,
#                   "radiation_homogeneous_canopy,energy_balance_dragon" or
#                   "radiation_homogeneous_canopy energy_balance_dragon").
#                   If omitted, all samples are run.
#   --help        - Show this help message and exit.
#
# Output:
#   Results are written to individual files in benchmarks/report named
#   <benchmark>_<hostname>_<version>_<gpu>_<cpu>.txt.
#
# Environment:
#   Requires NVIDIA GPU with drivers and CUDA toolkit installed. The
#   OMP_NUM_THREADS environment variable is used to set the thread count.
#

# Function to display usage information
usage() {
    sed -n '3,21p' "$0"
    exit 1
}

# Helper to run commands with optional logging
run_command() {
    if [ -n "$LOG_FILE" ]; then
        if [ "$VERBOSE" = "ON" ]; then
            "$@" 2>&1 | tee -a "$LOG_FILE"
            return ${PIPESTATUS[0]}
        else
            "$@" >> "$LOG_FILE" 2>&1
        fi
    elif [ "$VERBOSE" = "ON" ]; then
        "$@"
    else
        "$@" >/dev/null 2>&1
    fi
}

ALL_SAMPLES=("radiation_homogeneous_canopy" "energy_balance_dragon" "plant_architecture_bean")

BUILD_TYPES=("Debug" "Release")

# Default thread counts if not specified
THREAD_COUNTS=(1)

# Parse arguments
VERBOSE="OFF"
LOG_FILE=""
THREAD_STRING=""
SAMPLES_STRING=""

while [ $# -gt 0 ]; do
    case $1 in
        --verbose)
            VERBOSE="ON"
            ;;
        --log-file)
            [ -z "$2" ] && usage
            if [[ $2 = /* ]]; then
                LOG_FILE="$2"
            else
                LOG_FILE="$(pwd)/$2"
            fi
            shift
            ;;
        --samples)
            [ -z "$2" ] && usage
            SAMPLES_STRING="$2"
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            if [[ -z "$THREAD_STRING" ]]; then
                THREAD_STRING="$1"
            else
                echo "Unknown option: $1" >&2
                usage
            fi
            ;;
    esac
    shift
done

if [ -n "$LOG_FILE" ]; then
    > "$LOG_FILE"
fi

if [ -n "$THREAD_STRING" ]; then
    THREAD_COUNTS=()
    IFS=', ' read -r -a THREAD_COUNTS <<< "$THREAD_STRING"
fi

# Set up samples to run
if [ -n "$SAMPLES_STRING" ]; then
    SAMPLES=()
    IFS=', ' read -r -a SAMPLES <<< "$SAMPLES_STRING"
    
    # Validate that all specified samples exist in the ALL_SAMPLES array
    for sample in "${SAMPLES[@]}"; do
        found=false
        for valid_sample in "${ALL_SAMPLES[@]}"; do
            if [ "$sample" = "$valid_sample" ]; then
                found=true
                break
            fi
        done
        if [ "$found" = false ]; then
            echo "Error: invalid sample '$sample'. Valid samples are: ${ALL_SAMPLES[*]}" >&2
            exit 1
        fi
    done
else
    # Use all samples if none specified
    SAMPLES=("${ALL_SAMPLES[@]}")
fi

# Validate that all thread counts are positive integers
for t in "${THREAD_COUNTS[@]}"; do
    if ! [[ $t =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: invalid thread count '$t'" >&2
        exit 1
    fi
done

# Get the directory script is in
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$( dirname "$SCRIPT_PATH" )"

# Make sure the report directory exists
if [ ! -d report ]; then
    mkdir report
fi

if [ ! -d "${SCRIPT_DIR}/../benchmarks/report" ]; then
    mkdir -p "${SCRIPT_DIR}/../benchmarks/report"
fi

if [[ "${OSTYPE}" != "darwin"* ]] && [[ "${OSTYPE}" != "linux"* ]] && [[ "${OSTYPE}" != "msys"* ]];then
  echo "UNSUPPORTED OPERATING SYSTEM"
  exit 1
fi

if [ "$(which nvidia-smi)" == "" ];then
  echo "nvidia-smi not found. Please install the NVIDIA driver and CUDA toolkit."
  exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)

if [ "${GPU_NAME}" == "" ];then
  echo "No NVIDIA GPU found. Please install the NVIDIA driver and CUDA toolkit."
  exit 1
fi

# Get CPU information based on the operating system
if [[ "${OSTYPE}" == "darwin"* ]]; then
    # macOS
    CPU_NAME=$(sysctl -n machdep.cpu.brand_string)
elif [[ "${OSTYPE}" == "linux"* ]]; then
    # Linux
    CPU_NAME=$(grep "model name" /proc/cpuinfo | head -n 1 | cut -d: -f2 | sed 's/^ *//')
elif [[ "${OSTYPE}" == "msys"* ]]; then
    # Windows/MSYS
    CPU_NAME=$(wmic cpu get name /value | grep "Name=" | cut -d= -f2 | tr -d '\r')
else
    CPU_NAME="Unknown"
fi

if [ "${CPU_NAME}" == "" ]; then
    CPU_NAME="Unknown"
fi

# Determine the current Helios version from git
HELIOS_VERSION=$(git describe --tags 2>/dev/null)
if [ -z "$HELIOS_VERSION" ]; then
  HELIOS_VERSION="unknown"
fi
SANITIZED_VERSION=$(echo "$HELIOS_VERSION" | tr ' /:*?"<>|\\' '_')

echo "---------------------------------------------"
echo "Benchmarking Helios ${HELIOS_VERSION} on ${GPU_NAME}"
echo "CPU: ${CPU_NAME}"
echo "Using thread counts: ${THREAD_COUNTS[*]}"
echo "Running samples: ${SAMPLES[*]}"
echo "---------------------------------------------"

ERROR_COUNT=0
for i in "${SAMPLES[@]}"; do

    SAMPLE_DIR="${SCRIPT_DIR}/../benchmarks/${i}"

    if [ ! -d "${SAMPLE_DIR}" ]; then
        echo "Sample ${i} does not exist." >&2
        exit 1
    fi

    # Sanitize GPU, CPU, and hostname for filenames
    SANITIZED_GPU_NAME=$(echo "${GPU_NAME}" | tr ' /:*?"<>|\\' '_')
    SANITIZED_CPU_NAME=$(echo "${CPU_NAME}" | tr ' /:*?"<>|\\' '_')
    HOST_NAME=$(hostname)
    SANITIZED_HOST_NAME=$(echo "${HOST_NAME}" | tr ' /:*?"<>|\\' '_')

    OUTPUT_FILE="${SCRIPT_DIR}/../benchmarks/report/${i}_${SANITIZED_HOST_NAME}_${SANITIZED_VERSION}_${SANITIZED_GPU_NAME}_${SANITIZED_CPU_NAME}.txt"
    echo "$HELIOS_VERSION" > "${OUTPUT_FILE}"

    for build in "${BUILD_TYPES[@]}"; do
        UNIQUE_ID="$(date +%s)_$$"
        WORK_DIR="${SAMPLE_DIR}/run_${UNIQUE_ID}"
        BUILD_DIR="${WORK_DIR}/build"
        RESULT_DIR="${WORK_DIR}/results"

        mkdir -p "${BUILD_DIR}" "${RESULT_DIR}"
        cd "${BUILD_DIR}" || exit 1

        echo -ne "Building benchmark ${i} using build type ${build}...\n"
        run_command cmake "${SAMPLE_DIR}" -DCMAKE_BUILD_TYPE="${build}" -DENABLE_OPENMP=ON
        if (($? == 0)); then
            echo -e "\r\x1B[32mBuilding benchmark ${i}...done.\x1B[39m"
        else
            echo -e "\r\x1B[31mBuilding benchmark ${i}...failed.\x1B[39m"
            ERROR_COUNT=$((ERROR_COUNT + 1))
            rm -rf "${WORK_DIR}"
            continue
        fi

        echo -ne "Compiling benchmark ${i}..."
        run_command cmake --build ./ --target "${i}" --config "${build}"
        if (($? == 0)); then
            if [ -e "${i}" ] || [ -e "${i}.exe" ]; then
                echo -e "\r\x1B[32mCompiling benchmark ${i}...done.\x1B[39m"
            else
                echo -e "\r\x1B[31mCompiling benchmark ${i}...failed.\x1B[39m"
                ERROR_COUNT=$((ERROR_COUNT + 1))
                rm -rf "${WORK_DIR}"
                continue
            fi
        else
            echo -e "\r\x1B[31mCompiling benchmark ${i}...failed.\x1B[39m"
            ERROR_COUNT=$((ERROR_COUNT + 1))
            rm -rf "${WORK_DIR}"
            continue
        fi

        for threads in "${THREAD_COUNTS[@]}"; do
            echo -ne "Running benchmark ${i} with ${threads} threads..."
            export OMP_NUM_THREADS="${threads}"

            if [[ "${OSTYPE}" == "msys"* ]]; then
                run_command "./${i}.exe"
                exit_code=$?
            else
                run_command "./${i}"
                exit_code=$?
            fi

            if [[ $exit_code -ne 0 ]]; then
                echo -e "\r\x1B[31mRunning benchmark ${i} with ${threads} threads...failed with exit code ${exit_code}.\x1B[39m"
                ERROR_COUNT=$((ERROR_COUNT + 1))
                continue
            fi

            echo -e "\r\x1B[32mRunning benchmark ${i} with ${threads} threads...done.\x1B[39m"

            if [ ! -e "${RESULT_DIR}/runtime.txt" ]; then
                echo -e "\r\x1B[31mResults file was not produced for ${threads} threads.\x1B[39m"
                ERROR_COUNT=$((ERROR_COUNT + 1))
                continue
            fi

            while IFS=',' read -r label runtime; do
                echo "${i},${GPU_NAME},${CPU_NAME},${build},${threads},${label},${runtime}" >> "${OUTPUT_FILE}"
            done < "${RESULT_DIR}/runtime.txt"
            rm -f "${RESULT_DIR}/runtime.txt"
        done

        rm -rf "${WORK_DIR}"

    done #loop over build types

done #loop over samples

cat "${OUTPUT_FILE}"

# Print a message about where the results were saved
echo -e "\n\x1B[32mResults have been saved to individual files in the benchmarks directory.\x1B[39m"
for i in "${SAMPLES[@]}"; do
    SANITIZED_GPU_NAME=$(echo "${GPU_NAME}" | tr ' /:*?"<>|\\' '_')
    SANITIZED_CPU_NAME=$(echo "${CPU_NAME}" | tr ' /:*?"<>|\\' '_')
    HOST_NAME=$(hostname)
    SANITIZED_HOST_NAME=$(echo "${HOST_NAME}" | tr ' /:*?"<>|\\' '_')
    echo " - ${SCRIPT_DIR}/../benchmarks/report/${i}_${SANITIZED_HOST_NAME}_${SANITIZED_VERSION}_${SANITIZED_GPU_NAME}_${SANITIZED_CPU_NAME}.txt"
done