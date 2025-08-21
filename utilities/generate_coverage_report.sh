#!/usr/bin/env bash
# generate_coverage_report.sh
# Generate code-coverage reports for a CMake C/C++ project.
#
# NOTE: This script can be run from any directory.

set -euo pipefail

#######################################
# Defaults
#######################################
report_type="html"
subset_files=()                   # Initialize as empty array
ignore_regex='^.*/lib/.*'         # for LLVM/Clang
ignore_pattern='*/lib/*'          # for GCC/lcov (glob pattern)
profdata="coverage.profdata"
coverage_dir="coverage"
log_file=""

# Test plugins available for coverage (same as run_tests.sh)
TEST_PLUGINS="energybalance lidar aeriallidar photosynthesis radiation leafoptics solarposition stomatalconductance visualizer voxelintersection weberpenntree canopygenerator boundarylayerconductance syntheticannotation plantarchitecture projectbuilder planthydraulics parameteroptimization collisiondetection"

BUILD_TYPE="Debug"  # Coverage requires debug build

# Initialize variables
SPECIFIC_TEST=""
SPECIFIC_TESTS=""
PROJECT_DIR=""

#######################################
# Helper function to run commands with optional logging
#######################################
run_command() {
    if [ -n "$log_file" ]; then
        "$@" >> "$log_file" 2>&1
    else
        "$@" >/dev/null 2>&1
    fi
}

#######################################
# Help
#######################################
usage() {
    cat <<EOF
Usage: ./generate_coverage_report.sh --test <name> [options]
   or: ./generate_coverage_report.sh --tests <list> [options]

This script can be run from any directory.

Required (one of):
  --test <name>          Run coverage for the specified test (e.g., --test radiation)
  --tests <list>         Run coverage for specified tests (comma-separated, e.g., --tests "radiation,lidar")

Options:
  -r {html|text}         Report format (default: html)
  -f <file> [...]        One or more source files to limit a *text* report.
  -l <log_file>          Write command output to specified log file.
  --project-dir <dir>    Use specified directory for project (default: coverage_<testname>)
  -h                     Show this help.

Available tests:
  context, energybalance, lidar, aeriallidar, photosynthesis, radiation, leafoptics,
  solarposition, stomatalconductance, visualizer, voxelintersection, weberpenntree,
  canopygenerator, boundarylayerconductance, syntheticannotation, plantarchitecture,
  projectbuilder, planthydraulics, parameteroptimization, collisiondetection

Examples:
  # HTML report for radiation test in current directory
  ./generate_coverage_report.sh --test radiation

  # Text report for specific test
  ./generate_coverage_report.sh --test context -r text

  # Multiple tests with custom project directory
  ./generate_coverage_report.sh --tests \"radiation,lidar\" --project-dir ./my_coverage

  # Default project directory will be named coverage_<testname>
  ./generate_coverage_report.sh --test radiation  # Creates coverage_radiation/ directory
EOF
    exit 1
}

#######################################
# Parse arguments
#######################################
while [ $# -gt 0 ]; do
    case $1 in
        -r)
            if [ -z "$2" ]; then
                echo "Error: -r requires a report type."
                usage
            fi
            report_type=$2
            shift
            ;;
        -f)
            if [ -z "$2" ]; then
                echo "Error: -f requires a file path."
                usage
            fi
            subset_files+=("$2")
            shift
            ;;
        -l)
            if [ -z "$2" ]; then
                echo "Error: -l requires a log file path."
                usage
            fi
            log_file=$2
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
        --project-dir)
            if [ -z "$2" ]; then
                echo "Error: --project-dir requires a directory path."
                usage
            fi
            PROJECT_DIR="$2"
            shift
            ;;
        -h|--help)
            usage 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            usage
            ;;
    esac
    shift
done

# Require either --test or --tests to be specified
if [ -z "$SPECIFIC_TEST" ] && [ -z "$SPECIFIC_TESTS" ]; then
    echo "Error: You must specify either --test <name> or --tests <list>"
    echo "Use --help to see available tests and usage examples."
    exit 1
fi

# Initialize log file if specified
if [ -n "$log_file" ]; then
    > "$log_file"
    echo "Coverage report generation started at $(date)" >> "$log_file"
fi

# Find the Helios root directory by looking for utilities/create_project.sh
HELIOS_ROOT=""
CURRENT_DIR="$(pwd)"
SEARCH_DIR="$CURRENT_DIR"

# Search up the directory tree for utilities/create_project.sh
while [ "$SEARCH_DIR" != "/" ]; do
    if [ -e "$SEARCH_DIR/utilities/create_project.sh" ]; then
        HELIOS_ROOT="$SEARCH_DIR"
        break
    fi
    SEARCH_DIR="$(dirname "$SEARCH_DIR")"
done

# If not found in parent directories, check if we're in the utilities directory
if [ -z "$HELIOS_ROOT" ] && [ -e "./create_project.sh" ]; then
    HELIOS_ROOT="$(dirname "$(pwd)")"
fi

if [ -z "$HELIOS_ROOT" ]; then
    echo "Error: Could not find Helios root directory"
    echo "Please run this script from within a Helios repository directory tree"
    echo "Looking for utilities/create_project.sh"
    exit 1
fi

echo "Found Helios root: $HELIOS_ROOT"

# Detect number of processors for parallel compilation
if command -v nproc >/dev/null 2>&1; then
    NPROC=$(nproc)
elif [[ "${OSTYPE}" == "darwin"* ]]; then
    NPROC=$(sysctl -n hw.ncpu)
elif [[ "${OSTYPE}" == "msys"* ]] || [[ "${OSTYPE}" == "cygwin"* ]] || [[ -n "${NUMBER_OF_PROCESSORS}" ]]; then
    NPROC=${NUMBER_OF_PROCESSORS:-$(nproc 2>/dev/null || echo "1")}
else
    NPROC=1
fi

# Determine directory to use for project (always persistent to preserve coverage reports)
TEMP_BASE="$CURRENT_DIR"
if [ -n "$PROJECT_DIR" ]; then
    # Convert to absolute path if relative
    if [[ "$PROJECT_DIR" = /* ]]; then
        TEMP_DIR="$PROJECT_DIR"
    else
        TEMP_DIR="$CURRENT_DIR/$PROJECT_DIR"
    fi
else
    # Create a descriptive default project directory name
    if [ -n "$SPECIFIC_TEST" ]; then
        TEMP_DIR="$CURRENT_DIR/coverage_${SPECIFIC_TEST}"
    elif [ -n "$SPECIFIC_TESTS" ]; then
        # Use first test name for directory
        FIRST_TEST=$(echo "$SPECIFIC_TESTS" | cut -d',' -f1 | xargs)
        TEMP_DIR="$CURRENT_DIR/coverage_${FIRST_TEST}_etc"
    else
        TEMP_DIR="$CURRENT_DIR/coverage_project"
    fi
fi
PERSISTENT_PROJECT="ON"  # All projects are now persistent to preserve coverage reports

# Cleanup function (only for error cases - projects are always kept for coverage reports)
cleanup() {
    echo ""
    echo "Coverage project directory preserved at: $TEMP_DIR"
    exit 1
}

# Set up signal traps for cleanup on interruption
trap cleanup INT TERM

#######################################
# Create and setup unified test project
#######################################

# Create or verify project directory
PROJECT_EXISTS="OFF"
if [ -e "$TEMP_DIR" ]; then
    # Check if the directory has a proper project setup
    if [ -e "$TEMP_DIR/main.cpp" ] && [ -e "$TEMP_DIR/CMakeLists.txt" ] && [ -e "$TEMP_DIR/build" ]; then
        echo "Using existing project in $TEMP_DIR"
        PROJECT_EXISTS="ON"
    else
        echo "Project directory exists but incomplete, recreating..."
        rm -rf "$TEMP_DIR"
        mkdir -p "$TEMP_DIR"
    fi
else
    echo "Creating coverage project directory: $TEMP_DIR"
    mkdir -p "$TEMP_DIR"
fi

# Only create project if it doesn't already exist
if [ "$PROJECT_EXISTS" == "OFF" ]; then
    # Determine which plugins to build based on test selection
    PLUGINS_TO_BUILD=""
    if [ -n "$SPECIFIC_TEST" ]; then
        if [[ "$SPECIFIC_TEST" == "context" ]]; then
            PLUGINS_TO_BUILD=""  # Context test needs no plugins
        else
            PLUGINS_TO_BUILD="$SPECIFIC_TEST"  # Only build the plugin being tested
        fi
    elif [ -n "$SPECIFIC_TESTS" ]; then
        IFS=',' read -ra TESTS_ARRAY <<< "$SPECIFIC_TESTS"
        for test in "${TESTS_ARRAY[@]}"; do
            test=$(echo "$test" | xargs)  # trim whitespace
            if [[ "$test" != "context" ]]; then
                PLUGINS_TO_BUILD="$PLUGINS_TO_BUILD $test"
            fi
        done
        PLUGINS_TO_BUILD=$(echo "$PLUGINS_TO_BUILD" | xargs)  # trim leading/trailing whitespace
    fi
    
    echo "Creating coverage project with plugins: ${PLUGINS_TO_BUILD:-none}"
    "$HELIOS_ROOT/utilities/create_project.sh" "$TEMP_DIR" ${PLUGINS_TO_BUILD}
fi

cd "$TEMP_DIR" || cleanup
if [ ! -e "main.cpp" ] || [ ! -e "CMakeLists.txt" ] || [ ! -e "build" ]; then
    echo "Project creation or setup failed - missing required files"
    cleanup
fi

cd build || cleanup

# Set absolute paths
source_dir="$(cd .. && pwd)"
build_dir="$(pwd)"

#######################################
# Detect compiler
#######################################
detect_compiler() {
    local compiler_path
    # Check if CXX is set
    if [[ -n "${CXX:-}" ]]; then
        compiler_path="$CXX"
    elif [[ -n "${CC:-}" ]]; then
        compiler_path="$CC"
    else
        # Try to find the default compiler
        compiler_path=$(command -v c++ 2>/dev/null || command -v g++ 2>/dev/null || command -v gcc 2>/dev/null || command -v clang++ 2>/dev/null || command -v clang 2>/dev/null || echo "")
    fi
    
    if [[ -z "$compiler_path" ]]; then
        echo "Error: Could not detect compiler" >&2
        exit 1
    fi
    
    local compiler_name
    compiler_name=$(basename "$compiler_path")
    case "$compiler_name" in
        clang*|clang++*)
            echo "clang" ;;
        gcc*|g++*)
            echo "gcc" ;;
        c++|*)
            # Fallback to parsing version string for c++ and unknown compilers
            local compiler_version
            compiler_version=$($compiler_path --version 2>/dev/null | head -1)
            
            if [[ $compiler_version =~ [Cc]lang ]]; then
                echo "clang"
            elif [[ $compiler_version =~ [Gg][Cc][Cc] || $compiler_version =~ g\+\+ ]]; then
                echo "gcc"
            else
                echo "unknown"
            fi
            ;;
    esac
}

compiler_type=$(detect_compiler)
echo "Detected compiler: $compiler_type"

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
fi

echo "Building coverage for ${#BUILD_TARGETS[@]} test target(s): ${BUILD_TARGETS[*]}"

#######################################
# CMake configure & build with coverage
#######################################
if [[ "$compiler_type" == "clang" ]]; then
    # LLVM/Clang configuration
    echo "Configuring CMake for Clang coverage..."
    run_command cmake .. -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DBUILD_TESTS=ON \
      -DCMAKE_C_FLAGS="-fprofile-instr-generate -fcoverage-mapping -O0 -g" \
      -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping -O0 -g" \
      -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate -fcoverage-mapping"
elif [[ "$compiler_type" == "gcc" ]]; then
    # GCC configuration
    echo "Configuring CMake for GCC coverage..."
    run_command cmake .. -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DBUILD_TESTS=ON \
      -DCMAKE_C_FLAGS="--coverage -O0 -g" \
      -DCMAKE_CXX_FLAGS="--coverage -O0 -g" \
      -DCMAKE_EXE_LINKER_FLAGS="--coverage"
else
    echo "Error: Unsupported compiler type: $compiler_type" >&2
    cleanup
fi

if (($? != 0)); then
    echo "CMake configuration failed"
    cleanup
fi

echo "Building test targets..."
for target in "${BUILD_TARGETS[@]}"; do
    run_command cmake --build ./ --target "$target" --config "${BUILD_TYPE}" -j "${NPROC}"
    if (($? != 0)); then
        echo "Failed to build test target: $target"
        cleanup
    fi
done

#######################################
# Run tests to generate coverage files
#######################################

# Discover available test executables after compilation
TEST_EXECUTABLES=()
for target in "${BUILD_TARGETS[@]}"; do
    if [ -e "$target" ] || [ -e "$target.exe" ]; then
        TEST_EXECUTABLES+=("$target")
    else
        echo "Warning: Expected executable '$target' not found after compilation"
    fi
done

echo "Running ${#TEST_EXECUTABLES[@]} test executable(s) to generate coverage data: ${TEST_EXECUTABLES[*]}"

for test_exe in "${TEST_EXECUTABLES[@]}"; do
    echo "Running $test_exe..."
    if [[ "${OSTYPE}" == "msys"* ]]; then
        run_command "./${test_exe}.exe"
    else
        run_command "./${test_exe}"
    fi
    
    if (($? != 0)); then
        echo "Test $test_exe failed, but continuing with coverage generation..."
    fi
done

#######################################
# Process coverage data and create report
#######################################
if [[ "$compiler_type" == "clang" ]]; then
    # LLVM/Clang coverage processing
    echo "Processing coverage data with LLVM tools..."
    rm -f "$profdata"
    
    # Use xcrun on macOS, direct commands on other platforms
    if [[ "$OSTYPE" == "darwin"* ]]; then
        run_command xcrun llvm-profdata merge --sparse "$build_dir"/*.profraw -o "$profdata"
        
        llvm_cov_common=(
            xcrun llvm-cov show
            -instr-profile="$profdata"
            -ignore-filename-regex="$ignore_regex"
            "${TEST_EXECUTABLES[0]}"
        )
    else
        run_command llvm-profdata merge --sparse "$build_dir"/*.profraw -o "$profdata"
        
        llvm_cov_common=(
            llvm-cov show
            -instr-profile="$profdata"
            -ignore-filename-regex="$ignore_regex"
            "${TEST_EXECUTABLES[0]}"
        )
    fi
    
    case "$report_type" in
        html)
            echo "Generating HTML coverage report..."
            rm -rf "$coverage_dir"
            run_command "${llvm_cov_common[@]}" -format=html -output-dir="$coverage_dir"
            echo "HTML report generated in $coverage_dir/index.html"
            ;;
        text)
            echo "Generating text coverage report..."
            if [ ${#subset_files[@]} -gt 0 ]; then
                "${llvm_cov_common[@]}" -format=text "${subset_files[@]}" > coverage_details.txt
            else
                "${llvm_cov_common[@]}" -format=text > coverage_details.txt
            fi
            echo "Text report written to coverage_details.txt"
            ;;
        *)
            echo "Unknown report type: $report_type" >&2
            cleanup
            ;;
    esac
    
elif [[ "$compiler_type" == "gcc" ]]; then
    # GCC coverage processing
    case "$report_type" in
        html)
            echo "Generating HTML coverage report with lcov..."
            rm -rf "$coverage_dir"
            mkdir -p "$coverage_dir"
            
            # Generate HTML report using lcov and genhtml
            if command -v lcov >/dev/null 2>&1 && command -v genhtml >/dev/null 2>&1; then
                # Capture coverage data
                run_command lcov --capture --directory "$build_dir" --output-file coverage.info
                
                # Remove files matching the ignore pattern
                run_command lcov --remove coverage.info "$ignore_pattern" --output-file coverage_filtered.info
                
                # Generate HTML report
                run_command genhtml coverage_filtered.info --output-directory "$coverage_dir"
                echo "HTML report generated in $coverage_dir/index.html"
                
                # Clean up intermediate files
                rm -f coverage.info coverage_filtered.info
            else
                echo "Error: lcov and genhtml are required for HTML reports with GCC" >&2
                echo "Please install lcov package (e.g., 'sudo apt-get install lcov' or 'brew install lcov')" >&2
                cleanup
            fi
            ;;
        text)
            echo "Generating text coverage report with gcov..."
            
            # Create a subdirectory for gcov files
            gcov_dir="gcov_files"
            rm -rf "$gcov_dir"
            mkdir -p "$gcov_dir"
            
            # Generate text report using gcov
            if [ ${#subset_files[@]} -gt 0 ]; then
                # Convert relative paths to absolute paths
                subset_files_abs=()
                for file in "${subset_files[@]}"; do
                    if [[ "$file" = /* ]]; then
                        subset_files_abs+=("$file")
                    else
                        subset_files_abs+=("$(cd "$(dirname "$file")" && pwd)/$(basename "$file")")
                    fi
                done
                
                # Find corresponding .gcno files in the build directory
                gcno_files=()
                for file in "${subset_files_abs[@]}"; do
                    # Get relative path from source directory
                    rel_path="${file#$source_dir/}"
                    # Look for .gcno files in any test target directory
                    obj_path=""
                    for target in "${BUILD_TARGETS[@]}"; do
                        candidate="$build_dir/CMakeFiles/${target}.dir/${rel_path}.gcno"
                        if [ -f "$candidate" ]; then
                            obj_path="$candidate"
                            break
                        fi
                    done
                    if [ -n "$obj_path" ] && [ -f "$obj_path" ]; then
                        gcno_files+=("$obj_path")
                    else
                        echo "Warning: Coverage file not found for $file" >&2
                        for target in "${BUILD_TARGETS[@]}"; do
                            echo "  Checked: $build_dir/CMakeFiles/${target}.dir/${rel_path}.gcno" >&2
                        done
                    fi
                done
                
                if [ ${#gcno_files[@]} -gt 0 ]; then
                    (
                        cd "$gcov_dir"
                        for gcno_file in "${gcno_files[@]}"; do
                            source_file="${gcno_file%.gcno}"
                            # Extract source file path from gcno path (remove CMakeFiles/target.dir/ prefix)
                            source_file=$(echo "$source_file" | sed 's|.*/CMakeFiles/[^/]*\.dir/||')
                            echo "Coverage for $source_file:"
                            run_command gcov -b -c -o "$(dirname "$gcno_file")" "$source_file"
                        done
                    )
                    
                    # Generate the coverage details file
                    (
                        cd "$gcov_dir"
                        for gcno_file in "${gcno_files[@]}"; do
                            source_file="${gcno_file%.gcno}"
                            # Extract source file path from gcno path (remove CMakeFiles/target.dir/ prefix)
                            source_file=$(echo "$source_file" | sed 's|.*/CMakeFiles/[^/]*\.dir/||')
                            echo "Coverage for $source_file:"
                            gcov -b -c -o "$(dirname "$gcno_file")" "$source_file" 2>/dev/null || echo "No coverage data found"
                        done
                    ) > coverage_details.txt
                else
                    echo "Error: No coverage files found for specified source files" >&2
                    cleanup
                fi
            else
                echo "Generating text coverage report for all files..."
                (
                    cd "$gcov_dir"
                    run_command find "$build_dir" -name "*.gcno" -exec gcov -b -c {} \;
                )
                
                # Generate the coverage details file
                (
                    cd "$gcov_dir"
                    find "$build_dir" -name "*.gcno" -exec gcov -b -c {} \; 2>/dev/null || true
                ) > coverage_details.txt
            fi
            
            echo "Text report written to coverage_details.txt"
            echo "Individual gcov files written to $gcov_dir/"
            ;;
        *)
            echo "Unknown report type: $report_type" >&2
            cleanup
            ;;
    esac
fi

if [ -n "$log_file" ]; then
    echo "Coverage report generation completed at $(date)" >> "$log_file"
    echo "Command output has been logged to: $log_file"
fi

# Return to original directory and cleanup
cd "$TEMP_BASE" || exit 1

# Disable signal traps since we're about to exit normally
trap - INT TERM

echo "Coverage report generation completed successfully."
echo "Coverage project directory: $TEMP_DIR"
if [[ "$report_type" == "html" ]]; then
    echo "Coverage report location: $TEMP_DIR/build/coverage/index.html"
else
    echo "Coverage report location: $TEMP_DIR/build/coverage_details.txt"
fi