#!/usr/bin/env bash
# generate_coverage_report.sh
# Generate code-coverage reports for a CMake C/C++ project.
#
# NOTE: This script must be run from the build directory.

set -euo pipefail

#######################################
# Defaults
#######################################
build_dir="."
source_dir=".."
report_type="html"
subset_files=()                   # Initialize as empty array
ignore_regex='^.*/lib/.*'         # for LLVM/Clang
ignore_pattern='*/lib/*'          # for GCC/lcov (glob pattern)
profdata="coverage.profdata"
coverage_dir="coverage"
log_file=""

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
Usage: ./generate_coverage_report.sh -t <test_target> [options]

IMPORTANT: This script must be run from the build directory.

Required:
  -t <test_target>       Name of the test executable to run.

Options:
  -s <source_dir>        Directory containing CMakeLists.txt (default: ..)
  -r {html|text}         Report format (default: html)
  -f <file> [...]        One or more source files to limit a *text* report.
  -l <log_file>          Write command output to specified log file.
  -h                     Show this help.

Examples:
  # Run from build directory - Default source directory, HTML report
  ./generate_coverage_report.sh -t visualizer_tests

  # Run from build directory - Custom source directory, text report for two files
  ./generate_coverage_report.sh -s \$HOME/project -t helios_tests -r text \\
      -f src/foo.cpp src/bar.cpp

  # Run from build directory - Write output to log file
  ./generate_coverage_report.sh -t visualizer_tests -l coverage.log
EOF
    exit 1
}

#######################################
# Parse arguments
#######################################
while getopts ":s:t:r:f:l:h" opt; do
    case "$opt" in
        s) source_dir=$OPTARG ;;
        t) test_target=$OPTARG ;;
        r) report_type=$OPTARG ;;
        f) subset_files+=("$OPTARG") ;;
        l) log_file=$OPTARG ;;
        h|\?) usage ;;
    esac
done

: "${test_target:?Missing -t <test_target> (executable to run).}"

# Initialize log file if specified
if [ -n "$log_file" ]; then
    > "$log_file"
    echo "Coverage report generation started at $(date)" >> "$log_file"
fi

#######################################
# Absolute paths
#######################################
build_dir=$(cd "$build_dir" && pwd)
source_dir=$(cd "$source_dir" && pwd)
test_exe="${build_dir}/${test_target}"

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

#######################################
# CMake configure & build
#######################################
if [[ "$compiler_type" == "clang" ]]; then
    # LLVM/Clang configuration
    echo "Configuring CMake for Clang coverage..."
    run_command cmake -B "$build_dir" -S "$source_dir" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_C_FLAGS="-fprofile-instr-generate -fcoverage-mapping -O0 -g" \
      -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping -O0 -g" \
      -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate -fcoverage-mapping"
elif [[ "$compiler_type" == "gcc" ]]; then
    # GCC configuration
    echo "Configuring CMake for GCC coverage..."
    run_command cmake -B "$build_dir" -S "$source_dir" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_C_FLAGS="--coverage -O0 -g" \
      -DCMAKE_CXX_FLAGS="--coverage -O0 -g" \
      -DCMAKE_EXE_LINKER_FLAGS="--coverage"
else
    echo "Error: Unsupported compiler type: $compiler_type" >&2
    exit 1
fi

echo "Building test target..."
run_command cmake --build "$build_dir" --target "$test_target"

#######################################
# Run tests to generate coverage files
#######################################
echo "Running tests to generate coverage data..."
run_command "$test_exe"

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
            "$test_exe"
        )
    else
        run_command llvm-profdata merge --sparse "$build_dir"/*.profraw -o "$profdata"
        
        llvm_cov_common=(
            llvm-cov show
            -instr-profile="$profdata"
            -ignore-filename-regex="$ignore_regex"
            "$test_exe"
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
            exit 2
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
                exit 1
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
                    # Convert to object file path in build directory
                    obj_path="$build_dir/CMakeFiles/${test_target}.dir/${rel_path}.gcno"
                    if [ -f "$obj_path" ]; then
                        gcno_files+=("$obj_path")
                    else
                        echo "Warning: Coverage file not found for $file" >&2
                        echo "  Expected: $obj_path" >&2
                    fi
                done
                
                if [ ${#gcno_files[@]} -gt 0 ]; then
                    (
                        cd "$gcov_dir"
                        for gcno_file in "${gcno_files[@]}"; do
                            source_file="${gcno_file%.gcno}"
                            source_file="${source_file#$build_dir/CMakeFiles/${test_target}.dir/}"
                            echo "Coverage for $source_file:"
                            run_command gcov -b -c -o "$(dirname "$gcno_file")" "$source_file"
                        done
                    )
                    
                    # Generate the coverage details file
                    (
                        cd "$gcov_dir"
                        for gcno_file in "${gcno_files[@]}"; do
                            source_file="${gcno_file%.gcno}"
                            source_file="${source_file#$build_dir/CMakeFiles/${test_target}.dir/}"
                            echo "Coverage for $source_file:"
                            gcov -b -c -o "$(dirname "$gcno_file")" "$source_file" 2>/dev/null || echo "No coverage data found"
                        done
                    ) > coverage_details.txt
                else
                    echo "Error: No coverage files found for specified source files" >&2
                    exit 1
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
            exit 2
            ;;
    esac
fi

if [ -n "$log_file" ]; then
    echo "Coverage report generation completed at $(date)" >> "$log_file"
    echo "Command output has been logged to: $log_file"
fi