#!/usr/bin/env bash
# generate_coverage_report.sh
# Generate code-coverage reports for a CMake C/C++ project.

set -euo pipefail

#######################################
# Defaults
#######################################
build_dir="."
source_dir=".."
report_type="html"
declare -a subset_files           # empty → all files
ignore_regex='^.*/lib/.*'         # for LLVM/Clang
ignore_pattern='*/lib/*'          # for GCC/lcov (glob pattern)
profdata="coverage.profdata"
coverage_dir="coverage"

#######################################
# Help
#######################################
usage() {
    cat <<EOF
Usage: ./generate_coverage_report.sh -t <test_target> [options]

Required:
  -t <test_target>       Name of the test executable to run.

Options:
  -b <build_dir>         Build directory (default: .)
  -s <source_dir>        Directory containing CMakeLists.txt (default: ..)
  -r {html|text}         Report format (default: html)
  -f <file> [...]        One or more source files to limit a *text* report.
  -h                     Show this help.

Examples:
  # Default directories, HTML report
  ./generate_coverage_report.sh -t visualizer_tests

  # Custom dirs, text report for two files
  ./generate_coverage_report.sh -b build -s \$HOME/project -t helios_tests -r text \\
      -f src/foo.cpp src/bar.cpp
EOF
    exit 1
}

#######################################
# Parse arguments
#######################################
while getopts ":b:s:t:r:f:h" opt; do
    case "$opt" in
        b) build_dir=$OPTARG ;;
        s) source_dir=$OPTARG ;;
        t) test_target=$OPTARG ;;
        r) report_type=$OPTARG ;;
        f) subset_files+=("$OPTARG") ;;
        h|\?) usage ;;
    esac
done

: "${test_target:?Missing -t <test_target> (executable to run).}"

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
        compiler_path=$(command -v c++ 2>/dev/null || command -v gcc 2>/dev/null || command -v clang++ 2>/dev/null || echo "")
    fi
    
    if [[ -z "$compiler_path" ]]; then
        echo "Error: Could not detect compiler" >&2
        exit 1
    fi
    
    # Get compiler version info
    local compiler_version
    compiler_version=$($compiler_path --version 2>/dev/null | head -1)
    
    if [[ $compiler_version =~ [Cc]lang ]]; then
        echo "clang"
    elif [[ $compiler_version =~ [Gg]cc ]]; then
        echo "gcc"
    else
        echo "unknown"
    fi
}

compiler_type=$(detect_compiler)
echo "Detected compiler: $compiler_type"

#######################################
# CMake configure & build
#######################################
if [[ "$compiler_type" == "clang" ]]; then
    # LLVM/Clang configuration
    cmake -B "$build_dir" -S "$source_dir" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_C_FLAGS="-fprofile-instr-generate -fcoverage-mapping -O0 -g" \
      -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping -O0 -g" \
      -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate -fcoverage-mapping"
elif [[ "$compiler_type" == "gcc" ]]; then
    # GCC configuration
    cmake -B "$build_dir" -S "$source_dir" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_C_FLAGS="--coverage -O0 -g" \
      -DCMAKE_CXX_FLAGS="--coverage -O0 -g" \
      -DCMAKE_EXE_LINKER_FLAGS="--coverage"
else
    echo "Error: Unsupported compiler type: $compiler_type" >&2
    exit 1
fi

cmake --build "$build_dir" --target "$test_target"

#######################################
# Run tests to generate coverage files
#######################################
"$test_exe"

#######################################
# Process coverage data and create report
#######################################
if [[ "$compiler_type" == "clang" ]]; then
    # LLVM/Clang coverage processing
    rm -f "$profdata"
    xcrun llvm-profdata merge --sparse "$build_dir"/*.profraw -o "$profdata"
    
    llvm_cov_common=(
        xcrun llvm-cov show
        -instr-profile="$profdata"
        -ignore-filename-regex="$ignore_regex"
        "$test_exe"
    )
    
    case "$report_type" in
        html)
            rm -rf "$coverage_dir"
            "${llvm_cov_common[@]}" -format=html -output-dir="$coverage_dir"
            echo "HTML report generated in $coverage_dir/index.html"
            ;;
        text)
            if ((${#subset_files[@]})); then
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
            rm -rf "$coverage_dir"
            mkdir -p "$coverage_dir"
            
            # Generate HTML report using lcov and genhtml
            if command -v lcov >/dev/null 2>&1 && command -v genhtml >/dev/null 2>&1; then
                # Capture coverage data
                lcov --capture --directory "$build_dir" --output-file coverage.info
                
                # Remove files matching the ignore pattern
                lcov --remove coverage.info "$ignore_pattern" --output-file coverage_filtered.info
                
                # Generate HTML report
                genhtml coverage_filtered.info --output-directory "$coverage_dir"
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
            # Generate text report using gcov
            if ((${#subset_files[@]})); then
                for file in "${subset_files[@]}"; do
                    echo "Coverage for $file:"
                    gcov -b -c "$build_dir"/"$(basename "$file" .cpp)".gcno 2>/dev/null || true
                done > coverage_details.txt
            else
                echo "Generating text coverage report for all files..."
                find "$build_dir" -name "*.gcno" -exec gcov -b -c {} \; > coverage_details.txt 2>/dev/null || true
            fi
            echo "Text report written to coverage_details.txt"
            ;;
        *)
            echo "Unknown report type: $report_type" >&2
            exit 2
            ;;
    esac
fi