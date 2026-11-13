#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo
    echo "Options:"
    echo "  --checkout       Clone the latest Helios repo to /tmp and run tests"
    echo "  --force-vulkan   Force use of Vulkan backend (passes -DFORCE_VULKAN_BACKEND to cmake)"
    echo "  --nogpu          Run the tests as if no GPU were present (sets HELIOS_NO_GPU=1)."
    echo "                   Reproduces the non-GPU CI runners on a GPU-equipped machine."
    echo "  --requiregpu     Fail instead of skipping when no usable GPU is found"
    echo "                   (sets HELIOS_REQUIRE_GPU=1). For GPU-dedicated CI runners."
    echo "  --visbuildonly   Build only, do not run visualizer tests"
    echo "  --memcheck       Enable memory checking tools (requires leaks on macOS or valgrind on Linux)"
    echo "  --debugbuild     Build with Debug configuration"
    echo "  --verbose        Show full build/compile output"
    echo "  --log-file <file>  Redirect all output to a specified log file"
    echo "  --test <name>    Run only the specified test (e.g., --test radiation)"
    echo "  --tests <list>   Run specified tests (comma-separated, e.g., --tests \"radiation,lidar\")"
    echo "  --testcase <case> Pass specific test case(s) to doctest, comma-separated, with * wildcards (e.g., --testcase \"My Test Case\" or --testcase \"*first*,*second*\")"
    echo "  --doctestargs <args> Pass arguments directly to doctest (e.g., --doctestargs \"--help --list-test-cases\"); words are split on spaces, so quote any containing one (e.g., --doctestargs \"'--subcase=*a b*'\")"
    echo "  --list-tests         Build test executables and list available test cases without running them"
    echo "  --project-dir <dir>  Use specified directory for project (persistent, not cleaned up)"
    echo "  --shards <N>         Run each test executable as N concurrent processes, each over a share of its test cases."
    echo "                       Each process can use several GB of memory for the heavier plugins (e.g. plantarchitecture),"
    echo "                       and GPU plugins share one device, so choose N with that in mind. Not combinable with"
    echo "                       --memcheck or a doctest --out file."
    echo "  --disableopenmp      Disable OpenMP by passing -DENABLE_OPENMP=OFF to cmake"
    echo "  --cmake-args <args>  Pass custom arguments to cmake (e.g., --cmake-args \"-DSOME_OPTION=ON -DANOTHER=OFF\")"
    echo
    exit ${1:-1}
}

# Function to run commands with proper redirection (compatible with all bash versions)
#
# In quiet mode the output is captured rather than discarded, and replayed if the command fails.
# Previously it went to /dev/null, so a fatal error (a failed CMake configure, say) surfaced only as
# a bare "...failed." with nothing to act on, and the run had to be repeated with --verbose to learn
# anything at all.
run_command() {
    if [ -n "$LOG_FILE" ]; then
        if [ "$VERBOSE" == "ON" ]; then
            "$@" 2>&1 | tee -a "$LOG_FILE"
            return "${PIPESTATUS[0]}"
        else
            "$@" >> "$LOG_FILE" 2>&1
        fi
    elif [ "$VERBOSE" == "ON" ]; then
        "$@"
    else
        local quiet_output quiet_status
        quiet_output="$("$@" 2>&1)"
        quiet_status=$?
        if [ ${quiet_status} -ne 0 ]; then
            echo "" >&2
            echo "--- output of failed command: $* ---" >&2
            echo "${quiet_output}" >&2
            echo "--- end of output (re-run with --verbose for the full log) ---" >&2
        fi
        return ${quiet_status}
    fi
}

# Run one test executable as several concurrent processes ("shards"), each over its own share of the test cases.
#
# doctest has no parallel mode, but it can run a contiguous range of the test cases that pass the filters (--first/--last,
# 1-based and inclusive). The filtered list is cut into SHARD_CHUNKS_PER_WORKER ranges per worker, and worker w runs ranges
# w, w + workers, w + 2 * workers, ... one after another. Interleaving the ranges spreads a cluster of slow neighbouring
# tests across the workers, which one contiguous range per worker would not, and each range still runs in file order.
#
# Each worker runs in its own directory under the build directory, with TMPDIR (TMP and TEMP on Windows) inside it, so
# tests that write a file by a relative name or into the system temp directory cannot collide with another worker.
# HELIOS_BUILD is set to the build directory so that assets still resolve from there.
#
# Arguments: the executable's file name in the current (build) directory. Reads the global arrays TEST_ARGS_ARRAY (all
# doctest arguments) and FILTER_ARGS (the test-selecting subset, used to count the cases). Returns non-zero if any range
# failed or did not finish.
SHARD_CHUNKS_PER_WORKER=8
SHARD_PIDS=()
run_sharded_test() {
  local executable_file="$1"
  local build_dir
  build_dir="$(pwd)"

  local case_count
  case_count=$("./${executable_file}" --count "${FILTER_ARGS[@]}" 2>/dev/null | grep -o 'passing the current filters: [0-9]*' | grep -o '[0-9]*$')
  if [ -z "$case_count" ]; then
    echo "" >&2
    echo "Error: could not count the test cases of ${executable_file} with 'doctest --count'." >&2
    return 1
  fi
  if [ "$case_count" -eq 0 ]; then
    return 0
  fi

  local chunk_count=$((SHARDS * SHARD_CHUNKS_PER_WORKER))
  if [ "$chunk_count" -gt "$case_count" ]; then
    chunk_count=$case_count
  fi
  local worker_count=$SHARDS
  if [ "$worker_count" -gt "$chunk_count" ]; then
    worker_count=$chunk_count
  fi

  local shard_root="${build_dir}/shards_${executable_file%.exe}"
  rm -rf "$shard_root"
  mkdir -p "$shard_root" || return 1

  SHARD_PIDS=()
  local worker
  for ((worker = 0; worker < worker_count; worker++)); do
    local worker_dir="${shard_root}/worker_${worker}"
    mkdir -p "${worker_dir}/tmp" || return 1
    (
      cd "$worker_dir" || exit 1
      export HELIOS_BUILD="$build_dir" TMPDIR="${worker_dir}/tmp" TMP="${worker_dir}/tmp" TEMP="${worker_dir}/tmp"
      for ((chunk = worker; chunk < chunk_count; chunk += worker_count)); do
        first=$((chunk * case_count / chunk_count + 1))
        last=$(((chunk + 1) * case_count / chunk_count))
        "${build_dir}/${executable_file}" "${TEST_ARGS_ARRAY[@]}" "--first=${first}" "--last=${last}" > "${shard_root}/chunk_${chunk}.log" 2>&1
        echo $? > "${shard_root}/chunk_${chunk}.status"
      done
    ) &
    SHARD_PIDS+=($!)
  done
  local pid
  for pid in "${SHARD_PIDS[@]}"; do
    wait "$pid"
  done
  SHARD_PIDS=()

  # Replay the output in test order: all of it under --verbose or --log-file, otherwise only the ranges that failed. A range
  # with no status file was cut short (its worker was killed), which counts as a failure.
  local failed_chunks=0
  local total_assertions=0
  local failed_assertions=0
  local chunk
  for ((chunk = 0; chunk < chunk_count; chunk++)); do
    # doctest's "test cases:" summary line counts every test case passing the filters, including those outside the range, so
    # only the assertion counts are summed across ranges.
    local assertion_line
    assertion_line=$(grep '^\[doctest\] assertions:' "${shard_root}/chunk_${chunk}.log" 2>/dev/null | tail -1)
    if [ -n "$assertion_line" ]; then
      total_assertions=$((total_assertions + $(echo "$assertion_line" | awk -F'[:|]' '{print $2+0}')))
      failed_assertions=$((failed_assertions + $(echo "$assertion_line" | awk -F'|' '{print $3+0}')))
    fi
    local chunk_status
    chunk_status=$(cat "${shard_root}/chunk_${chunk}.status" 2>/dev/null)
    local chunk_header="--- ${executable_file} shard range $((chunk + 1))/${chunk_count}: test cases $((chunk * case_count / chunk_count + 1))-$(((chunk + 1) * case_count / chunk_count)) of ${case_count} ---"
    if [ "$chunk_status" != "0" ]; then
      failed_chunks=$((failed_chunks + 1))
    fi
    if [ -n "$LOG_FILE" ]; then
      { echo "$chunk_header"; cat "${shard_root}/chunk_${chunk}.log" 2>/dev/null; } >> "$LOG_FILE"
    fi
    if [ "$VERBOSE" == "ON" ] || { [ "$chunk_status" != "0" ] && [ -z "$LOG_FILE" ]; }; then
      echo "" >&2
      echo "$chunk_header" >&2
      cat "${shard_root}/chunk_${chunk}.log" >&2 2>/dev/null || echo "(no output: the range did not run)" >&2
    fi
  done

  local shard_summary="[shards] ${case_count} test cases in ${chunk_count} ranges over ${worker_count} processes: ${failed_chunks} ranges failed | assertions: ${total_assertions} | ${failed_assertions} failed"
  if [ -n "$LOG_FILE" ]; then
    echo "$shard_summary" >> "$LOG_FILE"
  fi
  if [ "$VERBOSE" == "ON" ] || [ "$failed_chunks" -gt 0 ]; then
    echo "$shard_summary" >&2
  fi

  if [ "$failed_chunks" -eq 0 ]; then
    rm -rf "$shard_root"
    return 0
  fi
  # Kept so that files the failing tests wrote can be inspected.
  echo "" >&2
  echo "${failed_chunks} of ${chunk_count} shard ranges of ${executable_file} failed. Their working directories are kept under ${shard_root}" >&2
  return 1
}

# Test plugins to include in unified build
TEST_PLUGINS="energybalance lidar photosynthesis radiation leafoptics solarposition stomatalconductance visualizer weberpenntree canopygenerator boundarylayerconductance syntheticannotation plantarchitecture projectbuilder planthydraulics soiltransport parameteroptimization collisiondetection"

BUILD_TYPE="Release"

# Detect number of processors for parallel compilation
if command -v nproc >/dev/null 2>&1; then
    NPROC=$(nproc)
elif [[ "${OSTYPE}" == "darwin"* ]]; then
    NPROC=$(sysctl -n hw.ncpu)
elif [[ "${OSTYPE}" == "msys"* ]] || [[ "${OSTYPE}" == "cygwin"* ]] || [[ -n "${NUMBER_OF_PROCESSORS}" ]]; then
    # Windows environment (Git Bash, MSYS2, Cygwin, or GitHub Actions)
    NPROC=${NUMBER_OF_PROCESSORS:-$(nproc 2>/dev/null || echo "1")}
else
    NPROC=1
fi

# Save the original working directory before changing directories
ORIGINAL_DIR="$(pwd)"

# Determine the Helios base directory by finding the script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELIOS_BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Change to the samples directory relative to the Helios base
cd "$HELIOS_BASE_DIR/samples" || exit 1

case "${OSTYPE}" in
  darwin*|linux*|msys*|cygwin*|mingw*) ;;
  *)
    echo "UNSUPPORTED OPERATING SYSTEM (OSTYPE='${OSTYPE}', uname -s='$(uname -s 2>/dev/null || echo unknown)')"
    exit 1
    ;;
esac

while [ $# -gt 0 ]; do
  case $1 in
  --checkout)
    CHECKOUT_MODE="ON"
    cd /tmp || exit 1

    if [ -e "./helios_test" ]; then
      chmod -R 777 ./helios_test
      rm -rf ./helios_test
    fi

    git clone https://www.github.com/PlantSimulationLab/Helios ./helios_test

    if [ ! -e "./helios_test" ]; then
      echo "Git checkout unsuccessful..exiting."
      exit 1
    fi

    chmod -R 777 ./helios_test
    cd ./helios_test/samples || exit 1
    ;;

  --force-vulkan)
    FORCE_VULKAN="ON"
    ;;

  --nogpu)
    # HELIOS_NO_GPU is a runtime-only veto consulted by every backend probe (nothing in the
    # CMake configuration reads it), so exporting it here changes only what the test binaries
    # see. This makes a GPU-equipped development machine behave exactly like the non-GPU CI
    # runners, which is the only way to catch a GPU-dependent test that was never guarded
    # with GPU_TEST_CASE before it reaches CI.
    export HELIOS_NO_GPU=1
    NOGPU="ON"
    ;;

  --requiregpu)
    # Inverse of --nogpu: on a runner whose whole purpose is exercising GPU code, a missing
    # driver must be a failure rather than a silent skip of every GPU test. Without this the
    # job reports success having tested nothing, which is how a broken GPU instance can go
    # unnoticed for days.
    export HELIOS_REQUIRE_GPU=1
    REQUIREGPU="ON"
    ;;

  --visbuildonly)
    VISRUN="OFF"
    ;;

  --memcheck)
    MEMCHECK="ON"
    ;;

  --debugbuild)
    BUILD_TYPE="Debug"
    ;;

  --verbose)
    VERBOSE="ON"
    ;;

  --log-file)
    if [ -z "$2" ]; then
      echo "Error: --log-file requires a file path."
      usage
    fi
    # Use an absolute path for the log file so subsequent
    # directory changes do not affect where output is written
    if [[ "$2" = /* ]]; then
      LOG_FILE="$2"
    else
      LOG_FILE="$ORIGINAL_DIR/$2"
    fi
    shift
    ;;

  --test)
    if [ -z "$2" ]; then
      echo "Error: --test requires a test name."
      usage
    fi
    SPECIFIC_TEST="$2"
    shift
    ;;

  --tests)
    if [ -z "$2" ]; then
      echo "Error: --tests requires a comma-separated list."
      usage
    fi
    SPECIFIC_TESTS="$2"
    shift
    ;;

  --testcase)
    if [ -z "$2" ]; then
      echo "Error: --testcase requires a test case name."
      usage
    fi
    TESTCASE_FILTER="$2"
    shift
    ;;

  --doctestargs)
    if [ -z "$2" ]; then
      echo "Error: --doctestargs requires arguments."
      usage
    fi
    DOCTEST_ARGS="$2"
    shift
    ;;

  --project-dir)
    if [ -z "$2" ]; then
      echo "Error: --project-dir requires a directory path."
      usage
    fi
    PROJECT_DIR="$2"
    shift
    ;;

  --list-tests)
    LIST_TESTS="ON"
    ;;

  --shards)
    if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
      echo "Error: --shards requires a positive whole number of processes (got '$2')."
      usage
    fi
    SHARDS="$2"
    shift
    ;;

  --disableopenmp)
    DISABLE_OPENMP="ON"
    ;;

  --cmake-args)
    if [ -z "$2" ]; then
      echo "Error: --cmake-args requires arguments."
      usage
    fi
    CUSTOM_CMAKE_ARGS="$2"
    shift
    ;;

  --help|-h)
    usage 0
    ;;

  *)
    echo "Error: Unknown option: $1"
    usage
    ;;
  esac
  shift
done

# doctest takes only options, so a word of --doctestargs that is not one is a piece of an option split at a space, and the
# filter it came from no longer selects the tests that were asked for. Checked before anything is built. Filename expansion
# is off, as where the arguments are used, so that a '*' is not matched against files.
if [ -n "$DOCTEST_ARGS" ]; then
  set -f
  eval "DOCTEST_ARGS_CHECK=($DOCTEST_ARGS)"
  set +f
  for test_arg in "${DOCTEST_ARGS_CHECK[@]}"; do
    if [[ "$test_arg" != -* ]]; then
      echo -e "\x1B[31mError: --doctestargs \"$DOCTEST_ARGS\" contains '$test_arg', which is not a doctest option: an option containing a space was split there. Quote that option inside the argument (e.g., --doctestargs \"'--test-case=*a b*'\"), or give test cases with --testcase \"*first*,*second*\".\x1B[39m"
      exit 1
    fi
  done
fi

# Sharded processes would all write the same doctest --out file, and memcheck wraps a single process.
if [ -n "$SHARDS" ] && [ "$SHARDS" -gt 1 ]; then
  if [ "${MEMCHECK}" == "ON" ]; then
    echo "Error: --shards cannot be combined with --memcheck, which checks a single test process."
    exit 1
  fi
  for test_arg in "${DOCTEST_ARGS_CHECK[@]}"; do
    case "$test_arg" in
      -o|-o=*|--out|--out=*|-dt-o|-dt-o=*|--dt-out|--dt-out=*)
        echo "Error: --shards cannot be combined with a doctest --out file: every shard would write the same file."
        exit 1
        ;;
    esac
  done
fi

# Validate that only one test specification method is used
TEST_SPEC_COUNT=0
if [ -n "$SPECIFIC_TEST" ]; then
  TEST_SPEC_COUNT=$((TEST_SPEC_COUNT + 1))
fi
if [ -n "$SPECIFIC_TESTS" ]; then
  TEST_SPEC_COUNT=$((TEST_SPEC_COUNT + 1))
fi

if [ $TEST_SPEC_COUNT -gt 1 ]; then
  echo "Error: Only one test specification method can be used at a time:"
  echo "  --test <name>     (run single test)"
  echo "  --tests <list>    (run multiple specific tests)"
  echo ""
  usage
fi

if [ -n "$LOG_FILE" ]; then
  true > "$LOG_FILE"  # Clear the log file
fi

if [ "${MEMCHECK}" == "ON" ];then
  if [[ "${OSTYPE}" == "darwin"* ]];then
    if [[ $(which leaks) == "" ]];then
      echo "Leaks memory checker tool not installed...ignoring --memcheck argument."
      MEMCHECK="OFF"
    fi
  elif [[ "${OSTYPE}" == "linux"* ]];then
    if [[ $(which valgrind) == "" ]];then
      echo "Valgrind memory checker tool not installed...ignoring --memcheck argument."
      MEMCHECK="OFF"
    fi
  fi
fi

# Check if cmake command is available
if ! command -v cmake >/dev/null 2>&1; then
  echo "ERROR: cmake command not found. Please install cmake and make sure it's in your PATH."
  exit 1
fi

if [ "${MEMCHECK}" == "ON" ];then
  BUILD_TYPE="Debug"
fi

# Configure Vulkan for GPU tests
if [[ "${OSTYPE}" == "linux"* ]]; then
    # NVIDIA Linux driver workaround for sequential instance creation bug
    # See: https://forums.developer.nvidia.com/t/issue-with-repeated-instance-creation-in-one-process/176978
    # After ~25-35 vkCreateInstance/vkDestroyInstance cycles, NVIDIA driver fails with VK_ERROR_INITIALIZATION_FAILED

    # Solution 1: Use only NVIDIA ICD (modern Vulkan loader method)
    # Only where an NVIDIA driver is installed, and never over a driver choice the caller already made: the loader applies
    # this filter on top of VK_ICD_FILENAMES / VK_DRIVER_FILES, so on a machine without an NVIDIA driver (an AMD or Intel GPU,
    # or lavapipe in CI) it leaves no driver at all and every Vulkan test skips for "no GPU".
    if [ -z "${VK_ICD_FILENAMES:-}${VK_DRIVER_FILES:-}${VK_LOADER_DRIVERS_SELECT:-}" ] &&
       ls /usr/share/vulkan/icd.d/nvidia*.json /etc/vulkan/icd.d/nvidia*.json /usr/local/share/vulkan/icd.d/nvidia*.json >/dev/null 2>&1; then
        export VK_LOADER_DRIVERS_SELECT="nvidia*"
    fi

    # Solution 2: Increase file descriptor limit
    # NVIDIA driver has FD leak issue with VkFence objects
    # See: https://github.com/ValveSoftware/gamescope/pull/454
    if command -v ulimit >/dev/null 2>&1; then
        ulimit -n 65536 2>/dev/null
    fi
fi

ERROR_COUNT=0

# Determine directory to use for project - either temporary or persistent
TEMP_BASE="$HELIOS_BASE_DIR"
if [ -n "$PROJECT_DIR" ]; then
  # Convert to absolute path if relative
  if [[ "$PROJECT_DIR" = /* ]]; then
    TEMP_DIR="$PROJECT_DIR"
  else
    TEMP_DIR="$HELIOS_BASE_DIR/$PROJECT_DIR"
  fi
  PERSISTENT_PROJECT="ON"
else
  TEMP_DIR="$HELIOS_BASE_DIR/temp_$$_$(date +%s)"
  PERSISTENT_PROJECT="OFF"
fi

# Cleanup function for errors and interruptions
# Remove the temporary project and exit. The optional argument is the exit status; without it the status is the error
# count, which every failure path increments before calling here. The signal traps pass 128 + the signal number (the shell
# convention), because an interrupted run has tested nothing and must not report success: with the error count it exited 0
# whenever nothing had failed yet.
cleanup() {
  local exit_status="${1:-$ERROR_COUNT}"
  echo ""
  # Stop any shard workers and the test processes they started, so an interrupted sharded run leaves nothing running. Each
  # worker is stopped before its test process: stopping the test process first would let the worker start its next range.
  for shard_pid in "${SHARD_PIDS[@]}"; do
    shard_children=$(pgrep -P "$shard_pid" 2>/dev/null)
    kill "$shard_pid" 2>/dev/null
    if [ -n "$shard_children" ]; then
      kill $shard_children 2>/dev/null
    fi
  done
  if [ -d "$TEMP_DIR" ] && [ "$PERSISTENT_PROJECT" != "ON" ]; then
    chmod -R 755 "$TEMP_DIR" 2>/dev/null || true
    rm -rf "$TEMP_DIR"
  fi
  exit "$exit_status"
}

# Set up signal traps for cleanup on interruption only. Bash runs a trap once the foreground command finishes, so a signal
# sent to this script alone (rather than to its process group, as Ctrl-C does) takes effect after the current build or
# test command completes.
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

# Validate requested test names up front so an invalid name (e.g. "core" instead of "context", or a
# misspelled plugin) fails immediately with a clear message instead of building every other target
# first and then dying on a cryptic "No rule to make target `<name>_tests'" from make.
validate_test_name() {
  local name="$1"
  # "context" is the core test suite; it is valid but is not a plugin directory.
  if [[ "$name" == "context" ]]; then
    return 0
  fi
  # Any other name must be a plugin that actually provides a test (plugins/<name>/tests/selfTest.cpp).
  if [ -f "$HELIOS_BASE_DIR/plugins/$name/tests/selfTest.cpp" ]; then
    return 0
  fi
  return 1
}

report_invalid_test_name() {
  local name="$1"
  echo -e "\x1B[31mError: '$name' is not a valid test name.\x1B[39m"
  # Catch the common core/context mix-up explicitly.
  if [[ "$name" == "core" ]]; then
    echo "  Did you mean 'context'? The core test suite is selected with --test context (not 'core')."
  fi
  echo "  Valid test names are 'context' (core tests) or a plugin that provides tests:"
  local plugin
  local available=""
  for plugin in "$HELIOS_BASE_DIR"/plugins/*/tests/selfTest.cpp; do
    [ -e "$plugin" ] || continue
    plugin="${plugin#"$HELIOS_BASE_DIR"/plugins/}"
    available="$available ${plugin%%/tests/selfTest.cpp}"
  done
  echo "    context$available"
}

INVALID_TEST_NAME=""
if [ -n "$SPECIFIC_TEST" ]; then
  if ! validate_test_name "$SPECIFIC_TEST"; then
    INVALID_TEST_NAME="$SPECIFIC_TEST"
  fi
elif [ -n "$SPECIFIC_TESTS" ]; then
  IFS=',' read -ra VALIDATE_ARRAY <<< "$SPECIFIC_TESTS"
  for test in "${VALIDATE_ARRAY[@]}"; do
    test=$(echo "$test" | xargs)  # trim whitespace
    if ! validate_test_name "$test"; then
      INVALID_TEST_NAME="$test"
      break
    fi
  done
fi
if [ -n "$INVALID_TEST_NAME" ]; then
  report_invalid_test_name "$INVALID_TEST_NAME"
  exit 1
fi

# Determine which plugins are needed for current test selection
PLUGINS_TO_BUILD=${TEST_PLUGINS}
if [ -n "$SPECIFIC_TEST" ]; then
  if [[ "$SPECIFIC_TEST" == "context" ]]; then
    PLUGINS_TO_BUILD=""  # Context test needs no plugins
  else
    PLUGINS_TO_BUILD="$SPECIFIC_TEST"  # Only build the plugin being tested
  fi
elif [ -n "$SPECIFIC_TESTS" ]; then
  PLUGINS_TO_BUILD=""
  IFS=',' read -ra TESTS_ARRAY <<< "$SPECIFIC_TESTS"
  for test in "${TESTS_ARRAY[@]}"; do
    test=$(echo "$test" | xargs)  # trim whitespace
    if [[ "$test" != "context" ]]; then
      PLUGINS_TO_BUILD="$PLUGINS_TO_BUILD $test"
    fi
  done
  PLUGINS_TO_BUILD=$(echo "$PLUGINS_TO_BUILD" | xargs)  # trim leading/trailing whitespace
fi

# Create or verify project directory
PROJECT_EXISTS="OFF"
if [ -e "$TEMP_DIR" ]; then
  if [ "$PERSISTENT_PROJECT" == "ON" ]; then
    # Check if the directory has a proper project setup
    if [ -e "$TEMP_DIR/main.cpp" ] && [ -e "$TEMP_DIR/CMakeLists.txt" ] && [ -e "$TEMP_DIR/build" ]; then
      
      # Check if existing project has the right plugin configuration
      if [ -n "$PLUGINS_TO_BUILD" ]; then
        # Extract current plugins from CMakeLists.txt
        CURRENT_PLUGINS=$(grep '^set( PLUGINS' "$TEMP_DIR/CMakeLists.txt" | sed 's/.*PLUGINS "\(.*\)" ).*/\1/' | tr ';' ' ')
        CURRENT_PLUGINS=$(echo "$CURRENT_PLUGINS" | xargs)  # normalize whitespace
        NEEDED_PLUGINS=$(echo "$PLUGINS_TO_BUILD" | tr ' ' '\n' | sort | tr '\n' ' ' | xargs)
        EXISTING_PLUGINS=$(echo "$CURRENT_PLUGINS" | tr ' ' '\n' | sort | tr '\n' ' ' | xargs)
        
        if [ "$NEEDED_PLUGINS" == "$EXISTING_PLUGINS" ]; then
          echo "Using existing project in $TEMP_DIR (plugins match: ${NEEDED_PLUGINS:-none})"
          PROJECT_EXISTS="ON"
        else
          echo "Project directory exists but plugin configuration changed (was: ${EXISTING_PLUGINS:-none}, need: ${NEEDED_PLUGINS:-none}), recreating..."
          rm -rf "$TEMP_DIR"
          mkdir -p "$TEMP_DIR"
        fi
      else
        # No plugins needed - check if existing project has no plugins
        CURRENT_PLUGINS=$(grep '^set( PLUGINS' "$TEMP_DIR/CMakeLists.txt" | sed 's/.*PLUGINS "\(.*\)" ).*/\1/' | tr ';' ' ')
        CURRENT_PLUGINS=$(echo "$CURRENT_PLUGINS" | xargs)  # normalize whitespace
        
        if [ -z "$CURRENT_PLUGINS" ]; then
          echo "Using existing project in $TEMP_DIR (no plugins needed)"
          PROJECT_EXISTS="ON"
        else
          echo "Project directory exists but plugin configuration changed (was: $CURRENT_PLUGINS, need: none), recreating..."
          rm -rf "$TEMP_DIR"
          mkdir -p "$TEMP_DIR"
        fi
      fi
    else
      echo "Project directory exists but incomplete, recreating..."
      rm -rf "$TEMP_DIR"
      mkdir -p "$TEMP_DIR"
    fi
  else
    rm -rf "$TEMP_DIR"
    mkdir -p "$TEMP_DIR"
  fi
else
  mkdir -p "$TEMP_DIR"
fi

if [ ! -e "$HELIOS_BASE_DIR/utilities/create_project.sh" ]; then
  echo -e "\r\x1B[31mProject creation script create_project.sh does not exist...failed.\x1B[39m"
  ERROR_COUNT=$((ERROR_COUNT + 1))
  rm -rf temp
else
  if [ "$NOGPU" == "ON" ] && [ "$REQUIREGPU" == "ON" ]; then
    echo "Error: --nogpu and --requiregpu are contradictory. --nogpu hides the GPU; --requiregpu demands one."
    exit 1
  fi

  if [ "$NOGPU" == "ON" ]; then
    echo "HELIOS_NO_GPU=1 is set: running as if no GPU were present. GPU tests will skip."
  fi

  if [ "$REQUIREGPU" == "ON" ]; then
    echo "HELIOS_REQUIRE_GPU=1 is set: tests that would skip for lack of a GPU will fail instead."
  fi

  # Only create project if it doesn't already exist
  if [ "$PROJECT_EXISTS" == "OFF" ]; then
    echo "Building project with plugins: ${PLUGINS_TO_BUILD:-none}"
    "$HELIOS_BASE_DIR/utilities/create_project.sh" "$TEMP_DIR" ${PLUGINS_TO_BUILD}
  fi
  
  cd "$TEMP_DIR" || exit 1
  if [ ! -e "main.cpp" ] || [ ! -e "CMakeLists.txt" ] || [ ! -e "build" ]; then
    echo -e "\r\x1B[31mProject creation or setup failed - missing required files...failed.\x1B[39m"
    ERROR_COUNT=$((ERROR_COUNT + 1))
    cleanup
  else

    cd build || exit 1

    echo -ne "Building unified test project..."

    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DBUILD_TESTS=ON"
    if [ "${DISABLE_OPENMP}" == "ON" ]; then
      CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_OPENMP=OFF"
    fi
    if [ "${FORCE_VULKAN}" == "ON" ]; then
      CMAKE_ARGS="${CMAKE_ARGS} -DFORCE_VULKAN_BACKEND=ON"
    fi
    if [ -n "${CUSTOM_CMAKE_ARGS}" ]; then
      CMAKE_ARGS="${CMAKE_ARGS} ${CUSTOM_CMAKE_ARGS}"
    fi

    run_command cmake .. ${CMAKE_ARGS}

    if (($? == 0)); then
      echo -e "\r\x1B[32mBuilding unified test project...done.\x1B[39m"
    else
      echo -e "\r\x1B[31mBuilding unified test project...failed.\x1B[39m"
      ERROR_COUNT=$((ERROR_COUNT + 1))
      cleanup
    fi

    # Determine which test targets to build based on user selection
    BUILD_TARGETS=()
    if [ -n "$SPECIFIC_TEST" ]; then
      if [[ "$SPECIFIC_TEST" == "context" ]]; then
        BUILD_TARGETS+=("context_tests")
      else
        BUILD_TARGETS+=("${SPECIFIC_TEST}_tests")
      fi
    elif [ -n "$SPECIFIC_TESTS" ]; then
      IFS=',' read -ra TESTS_ARRAY <<< "$SPECIFIC_TESTS"
      for test in "${TESTS_ARRAY[@]}"; do
        test=$(echo "$test" | xargs)  # trim whitespace
        if [[ "$test" == "context" ]]; then
          BUILD_TARGETS+=("context_tests")
        else
          BUILD_TARGETS+=("${test}_tests")
        fi
      done
    else
      # Build all available test targets
      BUILD_TARGETS+=("context_tests")
      for plugin in ${TEST_PLUGINS}; do
        BUILD_TARGETS+=("${plugin}_tests")
      done
    fi
    
    echo "Building ${#BUILD_TARGETS[@]} test target(s): ${BUILD_TARGETS[*]}"
    
    
    # Note: Skip pre-build target validation since cmake --build --target help 
    # is unreliable across different generators (especially Visual Studio on Windows).
    # Instead, we'll attempt to build the targets and validate executables afterward.
    
    # Build only the required test targets
    echo -ne "Compiling test targets..."
    
    
    for target in "${BUILD_TARGETS[@]}"; do
      run_command cmake --build ./ --target "$target" --config "${BUILD_TYPE}" -j "${NPROC}"
      if (($? != 0)); then
        echo -e "\r\x1B[31mCompiling test target $target...failed.\x1B[39m"
        echo
        echo "This could indicate:"
        echo "  1. The target '$target' does not exist (check plugin name spelling)"
        echo "  2. BUILD_TESTS=ON was not set properly during cmake configuration"
        echo "  3. There's a compilation error in the test code"
        echo "  4. Missing dependencies or incorrect CMake generator"
        echo
        echo "Example usage:"
        echo "  --test context                    # for context_tests"  
        echo "  --test radiation                  # for radiation_tests"
        echo "  --tests \"radiation,lidar\"         # for multiple tests"
        ERROR_COUNT=$((ERROR_COUNT + 1))
        cleanup
      fi
    done
    
    echo -e "\r\x1B[32mCompiling test targets...done.\x1B[39m"
    
    # Discover available test executables after compilation
    
    TEST_EXECUTABLES=()
    for target in "${BUILD_TARGETS[@]}"; do
      if [ -e "$target" ] || [ -e "$target.exe" ]; then
        TEST_EXECUTABLES+=("$target")
      else
        echo "Warning: Expected executable '$target' not found after compilation"
        ERROR_COUNT=$((ERROR_COUNT + 1))
      fi
    done
    
    echo "Verified ${#TEST_EXECUTABLES[@]} test executable(s): ${TEST_EXECUTABLES[*]}"

    # If --list-tests is specified, list test cases and exit
    if [ "$LIST_TESTS" == "ON" ]; then
      echo ""
      for test_exe in "${TEST_EXECUTABLES[@]}"; do
        echo -e "\x1B[32m=== Test cases in $test_exe ===\x1B[39m"
        if [[ "${OSTYPE}" == "msys"* ]]; then
          "./${test_exe}.exe" --list-test-cases
        else
          "./${test_exe}" --list-test-cases
        fi
        echo ""
      done

      cd "$TEMP_BASE" || cleanup 1

      # Clean up temp directory if not persistent
      if [ -d "$TEMP_DIR" ] && [ "$PERSISTENT_PROJECT" != "ON" ]; then
        chmod -R 755 "$TEMP_DIR" 2>/dev/null || true
        rm -rf "$TEMP_DIR"
      fi

      exit 0
    fi

    # Run each test executable
    for test_exe in "${TEST_EXECUTABLES[@]}"; do
      
      # Skip visualizer tests if requested
      if [ -n "${VISRUN}" ] && [[ "$test_exe" == *"visualizer"* ]]; then
        echo "Skipping $test_exe because it uses the visualizer..."
        continue
      fi
      
      # Prepare test arguments using arrays for proper quoting
      TEST_ARGS_ARRAY=()
      if [ -n "$TESTCASE_FILTER" ]; then
        TEST_ARGS_ARRAY+=("--test-case=$TESTCASE_FILTER")
      fi
      if [ -n "$DOCTEST_ARGS" ]; then
        # Use eval to properly split DOCTEST_ARGS into array elements, honouring quotes. Filename expansion is turned off
        # around it, or a '*' in a test filter would be matched against the files in the build directory.
        set -f
        eval "TEST_ARGS_ARRAY+=($DOCTEST_ARGS)"
        set +f
      fi

      # Validate that the test filters match at least one test before running, whether they came from --testcase or
      # --doctestargs. A filter that matches nothing otherwise runs zero tests and is reported as a pass. This is easy
      # to do by accident with --doctestargs, whose words are split on spaces: "--test-case=*a b*" becomes
      # "--test-case=*a" and a stray "b*".
      FILTER_ARGS=()
      for test_arg in "${TEST_ARGS_ARRAY[@]}"; do
        case "$test_arg" in
          --test-case=*|-tc=*|--test-case-exclude=*|-tce=*|--test-suite=*|-ts=*|--test-suite-exclude=*|-tse=*|--source-file=*|-sf=*|--source-file-exclude=*|-sfe=*)
            FILTER_ARGS+=("$test_arg")
            ;;
        esac
      done
      if [ ${#FILTER_ARGS[@]} -gt 0 ]; then
        if [[ "${OSTYPE}" == "msys"* ]]; then
          MATCHED_CASES=$("./${test_exe}.exe" --list-test-cases "${FILTER_ARGS[@]}" 2>/dev/null)
        else
          MATCHED_CASES=$("./${test_exe}" --list-test-cases "${FILTER_ARGS[@]}" 2>/dev/null)
        fi
        # Check if doctest reports 0 matching test cases
        if echo "$MATCHED_CASES" | grep -q "current filters: 0"; then
          echo -e "\x1B[31mError: No test cases in $test_exe match the filters: ${FILTER_ARGS[*]}\x1B[39m"
          echo "Use --doctestargs \"--list-test-cases\" to see available test cases. Give several test cases as one comma-separated --testcase \"*first*,*second*\"."
          ERROR_COUNT=$((ERROR_COUNT + 1))
          continue
        fi
      fi

      SHARD_NOTE=""
      if [ -n "$SHARDS" ] && [ "$SHARDS" -gt 1 ]; then
        SHARD_NOTE=" in ${SHARDS} shards"
      fi
      if [ ${#TEST_ARGS_ARRAY[@]} -gt 0 ]; then
        echo -ne "Running test $test_exe${SHARD_NOTE} with args: ${TEST_ARGS_ARRAY[*]}..."
      else
        echo -ne "Running test $test_exe${SHARD_NOTE}..."
      fi

      if [ -n "$SHARD_NOTE" ]; then
        if [[ "${OSTYPE}" == "msys"* ]]; then
          run_sharded_test "${test_exe}.exe"
        else
          run_sharded_test "${test_exe}"
        fi
      elif [[ "${OSTYPE}" == "msys"* ]]; then
        if [ ${#TEST_ARGS_ARRAY[@]} -gt 0 ]; then
          run_command "./${test_exe}.exe" "${TEST_ARGS_ARRAY[@]}"
        else
          run_command "./${test_exe}.exe"
        fi
      else
        if [ ${#TEST_ARGS_ARRAY[@]} -gt 0 ]; then
          run_command "./${test_exe}" "${TEST_ARGS_ARRAY[@]}"
        else
          run_command "./${test_exe}"
        fi
      fi
      
      if (($? == 0)); then
        echo -e "\r\x1B[32mRunning test $test_exe...passed.\x1B[39m"
      else
        echo -e "\r\x1B[31mRunning test $test_exe...failed.\x1B[39m"
        ERROR_COUNT=$((ERROR_COUNT + 1))
        continue
      fi
      
      # Run memory check if enabled
      if [[ "${MEMCHECK}" == "ON" ]] && [[ "${OSTYPE}" != "msys"* ]]; then
        
        # Skip visualizer tests on Linux due to memory leak issues
        if [[ "${OSTYPE}" == "linux"* ]] && [[ "$test_exe" == *"visualizer"* ]]; then
          continue
        fi
        
        if [ ${#TEST_ARGS_ARRAY[@]} -gt 0 ]; then
          echo -ne "Running memcheck for test $test_exe with args: ${TEST_ARGS_ARRAY[*]}..."
        else
          echo -ne "Running memcheck for test $test_exe..."
        fi
        
        # No ".exe" handling here: this whole block is already gated on OSTYPE != msys* above, so
        # Windows never reaches it. The nested "msys" checks that used to sit inside each branch
        # were unreachable dead code.
        if [[ "${OSTYPE}" == "darwin"* ]]; then
          if [ ${#TEST_ARGS_ARRAY[@]} -gt 0 ]; then
            run_command leaks --atExit -- "./${test_exe}" "${TEST_ARGS_ARRAY[@]}"
          else
            run_command leaks --atExit -- "./${test_exe}"
          fi
        else
          if [ ${#TEST_ARGS_ARRAY[@]} -gt 0 ]; then
            run_command valgrind --leak-check=full --error-exitcode=1 "./${test_exe}" "${TEST_ARGS_ARRAY[@]}"
          else
            run_command valgrind --leak-check=full --error-exitcode=1 "./${test_exe}"
          fi
        fi
        
        if (($? == 0)); then
          echo -e "\r\x1B[32mRunning memcheck for test $test_exe...passed.\x1B[39m"
        else
          echo -e "\r\x1B[31mRunning memcheck for test $test_exe...failed.\x1B[39m"
          ERROR_COUNT=$((ERROR_COUNT + 1))
        fi
        
      fi

    done

    cd "$TEMP_BASE" || cleanup 1

  fi
fi

if [ "$CHECKOUT_MODE" == "ON" ]; then
  cd ../..
  rm -rf ./helios_test
fi

# Disable signal traps since we're about to exit normally
trap - INT TERM

# Clean up temp directory on successful completion (only if not persistent)
if [ -d "$TEMP_DIR" ] && [ "$PERSISTENT_PROJECT" != "ON" ]; then
  chmod -R 755 "$TEMP_DIR" 2>/dev/null || true
  rm -rf "$TEMP_DIR"
fi

if ((ERROR_COUNT == 0)); then
  echo -e "\r\x1B[32mAll cases ran successfully.\x1B[39m"
  exit 0
else
  echo -e "\r\x1B[31mFailed ${ERROR_COUNT} cases.\x1B[39m"
  exit 1
fi