#!/bin/bash
# Usage:
#   ./benchmark_script.sh [THREAD_COUNTS]
#
# Arguments:
#   THREAD_COUNTS - Optional. A comma or space-separated list of thread counts to use
#                   (e.g., "1,2,4,8,16" or "1 2 4 8 16").
#                   If not provided, the benchmark will run with 1 thread only.
#
# Examples:
#   ./benchmark_script.sh "1,2,4,8,16"    # Run with 1, 2, 4, 8, and 16 threads
#   ./benchmark_script.sh "1 4 16"        # Run with 1, 4, and 16 threads
#   ./benchmark_script.sh                 # Run with 1 thread only (default)
#
# Output:
#   Results are saved to individual files in the benchmarks/report directory.
#   The output includes benchmark name, GPU, build type, thread count, 
#   benchmark label, and runtime.
#
# Environment:
#   Requires NVIDIA GPU with drivers and CUDA toolkit installed.
#   Sets OMP_NUM_THREADS environment variable to control OpenMP threading.
#

SAMPLES=("radiation_homogeneous_canopy")

BUILD_TYPES=("Debug" "Release")

# Default thread counts if not specified
THREAD_COUNTS=(1)

# Parse command-line arguments for thread counts
if [ $# -gt 0 ]; then
    # Clear default and use provided values
    THREAD_COUNTS=()
    # Split the argument by commas and spaces
    IFS=', ' read -r -a THREAD_COUNTS <<< "$1"
fi

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

echo "---------------------------------------------"
echo "Benchmarking on ${GPU_NAME}"
echo "Using thread counts: ${THREAD_COUNTS[*]}"
echo "---------------------------------------------"

ERROR_COUNT=0
for i in "${SAMPLES[@]}"; do

    SAMPLE_DIR="${SCRIPT_DIR}/../benchmarks/${i}"

    if [ ! -e "${SAMPLE_DIR}" ]; then
        echo "Sample ${i} does not exist."
        exit 1
    fi
    if [ ! -e "${SAMPLE_DIR}/build" ]; then
        mkdir -p "${SAMPLE_DIR}/build"
    fi

    cd "${SAMPLE_DIR}" || exit 1

    # Create a new consolidated results file for this benchmark
    RESULTS_FILE="${SAMPLE_DIR}/results/runtime.txt"
    # Initialize the file (clear it if it already exists)
    : > "${RESULTS_FILE}"

    RUNTIME_FILE="../results/runtime.txt"

    # Create sanitized GPU name for filename (replace spaces and special chars with underscores)
    SANITIZED_GPU_NAME=$(echo "${GPU_NAME}" | tr ' /:*?"<>|\\' '_')

    # Define output file with benchmark name and GPU name
    OUTPUT_FILE="${SCRIPT_DIR}/../benchmarks/report/${i}_${SANITIZED_GPU_NAME}.txt"
    git describe --tags > "${OUTPUT_FILE}"

    for build in "${BUILD_TYPES[@]}"; do

        cd "${SAMPLE_DIR}"/build || exit 1

        rm -rf "${SAMPLE_DIR}"/build/*glob*

        echo -ne "Building benchmark ${i} using build type ${build}...\n"

        cmake .. -DCMAKE_BUILD_TYPE="${build}" -DENABLE_OPENMP=ON &>/dev/null

        if (($? == 0)); then
            echo -e "\r\x1B[32mBuilding benchmark ${i}...done.\x1B[39m"
        else
            echo -e "\r\x1B[31mBuilding benchmark ${i}...failed.\x1B[39m"
            ERROR_COUNT=$((ERROR_COUNT + 1))
            rm -rf "${SAMPLE_DIR}"/build/*glob*
            continue
        fi

        echo -ne "Compiling benchmark ${i}..."

        cmake --build ./ --target "${i}" --config "${build}" &>/dev/null

        if (($? == 0)); then
            if [ -e "${i}" ]; then
                echo -e "\r\x1B[32mCompiling benchmark ${i}...done.\x1B[39m"
            else
                echo -e "\r\x1B[31mCompiling benchmark ${i}...failed.\x1B[39m"
                ERROR_COUNT=$((ERROR_COUNT + 1))
                rm -rf "${SAMPLE_DIR}"/build/*glob*
                continue
            fi
        else
            echo -e "\r\x1B[31mCompiling benchmark ${i}...failed.\x1B[39m"
            ERROR_COUNT=$((ERROR_COUNT + 1))
            rm -rf "${SAMPLE_DIR}"/build/*glob*
            continue
        fi

        # Loop over different thread counts
        for threads in "${THREAD_COUNTS[@]}"; do
            echo -ne "Running benchmark ${i} with ${threads} threads..."

            # Set the OpenMP threads environment variable
            export OMP_NUM_THREADS="${threads}"

            if [[ "${OSTYPE}" == "msys"* ]]; then
                "./${i}.exe" &>/dev/null
                exit_code=$?
            else
                "./${i}" &>/dev/null
                exit_code=$?
            fi

            if [[ $exit_code -ne 0 ]]; then
                echo -e "\r\x1B[31mRunning benchmark ${i} with ${threads} threads...failed with exit code ${exit_code}.\x1B[39m"
                echo -e "\r\x1B[31mThe benchmark ${i} did not run successfully.\x1B[39m"
                return 1
            fi

            echo -e "\r\x1B[32mRunning benchmark ${i} with ${threads} threads...done.\x1B[39m"

            if [ ! -e "../results/runtime.txt" ]; then
                echo -e "\r\x1B[31mResults file was not produced for ${threads} threads.\x1B[39m"
                ERROR_COUNT=$((ERROR_COUNT + 1))
                continue
            fi

            # Process runtime.txt and append to consolidated results file
            while IFS=',' read -r label runtime; do
                # Append with GPU name, build type, and thread count
                echo "${i},${GPU_NAME},${build},${threads},${label},${runtime}" >> "${OUTPUT_FILE}"
            done < "${RUNTIME_FILE}"
        done

        rm -rf "${SAMPLE_DIR}"/build/*glob*

    done #loop over build types

done #loop over samples

cat "${OUTPUT_FILE}"

# Print a message about where the results were saved
echo -e "\n\x1B[32mResults have been saved to individual files in the benchmarks directory.\x1B[39m"
for i in "${SAMPLES[@]}"; do
    SANITIZED_GPU_NAME=$(echo "${GPU_NAME}" | tr ' /:*?"<>|\\' '_')
    echo " - ${SCRIPT_DIR}/../benchmarks/report/${i}_${SANITIZED_GPU_NAME}.txt"
done