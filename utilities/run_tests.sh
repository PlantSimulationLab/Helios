#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo
    echo "Options:"
    echo "  --checkout       Clone the latest Helios repo to /tmp and run tests"
    echo "  --nogpu          Run only tests that do not require GPU"
    echo "  --visbuildonly   Build only, do not run visualizer tests"
    echo "  --memcheck       Enable memory checking tools (requires leaks on macOS or valgrind on Linux)"
    echo "  --debugbuild     Build with Debug configuration"
    echo "  --verbose        Show full build/compile output"
    echo "  --log-file <file>  Redirect all output to a specified log file"
    echo "  --test <name>    Run only the specified test (e.g., --test radiation)"
    echo "  --tests <list>   Run specified tests (comma-separated, e.g., --tests \"radiation,lidar\")"
    echo "  --testcase <case> Pass specific test case to doctest (e.g., --testcase \"My Test Case\")"
    echo "  --doctestargs <args> Pass arguments directly to doctest (e.g., --doctestargs \"--help --list-test-cases\")"
    echo "  --project-dir <dir>  Use specified directory for project (persistent, not cleaned up)"
    echo
    exit ${1:-1}
}

# Function to run commands with proper redirection (compatible with all bash versions)
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
        "$@" >/dev/null 2>&1
    fi
}

# Test plugins to include in unified build
TEST_PLUGINS="energybalance lidar aeriallidar photosynthesis radiation leafoptics solarposition stomatalconductance visualizer voxelintersection weberpenntree canopygenerator boundarylayerconductance syntheticannotation plantarchitecture projectbuilder planthydraulics parameteroptimization collisiondetection"
TEST_PLUGINS_NOGPU="leafoptics photosynthesis solarposition stomatalconductance visualizer weberpenntree canopygenerator boundarylayerconductance syntheticannotation plantarchitecture projectbuilder planthydraulics parameteroptimization collisiondetection"

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

if [[ "${OSTYPE}" != "darwin"* ]] && [[ "${OSTYPE}" != "linux"* ]] && [[ "${OSTYPE}" != "msys"* ]];then
  echo "UNSUPPORTED OPERATING SYSTEM"
  exit 1
fi

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

  --nogpu)
    TEST_PLUGINS=${TEST_PLUGINS_NOGPU}
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

# Validate that only one test specification method is used
TEST_SPEC_COUNT=0
if [ -n "$SPECIFIC_TEST" ]; then
  TEST_SPEC_COUNT=$((TEST_SPEC_COUNT + 1))
fi
if [ -n "$SPECIFIC_TESTS" ]; then
  TEST_SPEC_COUNT=$((TEST_SPEC_COUNT + 1))
fi
if [ "$TEST_PLUGINS" = "$TEST_PLUGINS_NOGPU" ]; then
  TEST_SPEC_COUNT=$((TEST_SPEC_COUNT + 1))
fi

if [ $TEST_SPEC_COUNT -gt 1 ]; then
  echo "Error: Only one test specification method can be used at a time:"
  echo "  --test <name>     (run single test)"
  echo "  --tests <list>    (run multiple specific tests)"
  echo "  --nogpu           (run all non-GPU tests)"
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
cleanup() {
  echo ""
  if [ -d "$TEMP_DIR" ] && [ "$PERSISTENT_PROJECT" != "ON" ]; then
    chmod -R 755 "$TEMP_DIR" 2>/dev/null || true
    rm -rf "$TEMP_DIR"
  fi
  exit $ERROR_COUNT
}

# Set up signal traps for cleanup on interruption only
trap cleanup INT TERM

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

    run_command cmake .. -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DBUILD_TESTS=ON

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
        # Use eval to properly split DOCTEST_ARGS into array elements
        eval "TEST_ARGS_ARRAY+=($DOCTEST_ARGS)"
      fi
      
      if [ ${#TEST_ARGS_ARRAY[@]} -gt 0 ]; then
        echo -ne "Running test $test_exe with args: ${TEST_ARGS_ARRAY[*]}..."
      else
        echo -ne "Running test $test_exe..."
      fi
      
      if [[ "${OSTYPE}" == "msys"* ]]; then
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
        
        if [[ "${OSTYPE}" == "darwin"* ]]; then
          if [[ "${OSTYPE}" == "msys"* ]]; then
            if [ ${#TEST_ARGS_ARRAY[@]} -gt 0 ]; then
              run_command leaks --atExit -- "./${test_exe}.exe" "${TEST_ARGS_ARRAY[@]}"
            else
              run_command leaks --atExit -- "./${test_exe}.exe"
            fi
          else
            if [ ${#TEST_ARGS_ARRAY[@]} -gt 0 ]; then
              run_command leaks --atExit -- "./${test_exe}" "${TEST_ARGS_ARRAY[@]}"
            else
              run_command leaks --atExit -- "./${test_exe}"
            fi
          fi
        else
          if [[ "${OSTYPE}" == "msys"* ]]; then
            if [ ${#TEST_ARGS_ARRAY[@]} -gt 0 ]; then
              run_command valgrind --leak-check=full --error-exitcode=1 "./${test_exe}.exe" "${TEST_ARGS_ARRAY[@]}"
            else
              run_command valgrind --leak-check=full --error-exitcode=1 "./${test_exe}.exe"
            fi
          else
            if [ ${#TEST_ARGS_ARRAY[@]} -gt 0 ]; then
              run_command valgrind --leak-check=full --error-exitcode=1 "./${test_exe}" "${TEST_ARGS_ARRAY[@]}"
            else
              run_command valgrind --leak-check=full --error-exitcode=1 "./${test_exe}"
            fi
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

    cd "$TEMP_BASE" || cleanup

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