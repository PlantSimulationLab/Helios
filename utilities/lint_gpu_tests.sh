#!/bin/bash
#
# Detect GPU-dependent test cases that were not declared with GPU_TEST_CASE.
#
# Why this exists:
#   The RadiationModel constructor calls RayTracingBackend::create("auto"), so merely
#   constructing a model throws on a machine with no usable GPU - even in a test that only
#   exercises CPU-side logic such as an error-message guard or a default flag value. A test
#   written as a plain DOCTEST_TEST_CASE therefore passes on a developer's GPU-equipped
#   machine and fails on the non-GPU CI runners, roughly 15-30 minutes after the push.
#   GPU_TEST_CASE (plugins/radiation/tests/test_helpers.h) skips the body when no backend is
#   available, which is what every such test needs.
#
# Usage:
#   utilities/lint_gpu_tests.sh            # lint the known test files
#   utilities/lint_gpu_tests.sh <file>...  # lint specific files
#
# Exit status:
#   0 - no unguarded GPU-dependent test cases found
#   1 - at least one found (details on stderr)
#
# Suppressing a legitimate case:
#   A handful of tests deliberately construct a model without a GPU - for example to assert
#   that construction against a broken driver throws cleanly rather than crashing. Mark those
#   with a GPU-LINT-OK comment carrying a reason, anywhere inside the test case body:
#
#       DOCTEST_TEST_CASE("Repeated construction never crashes") {
#           // GPU-LINT-OK: deliberately constructs without a GPU to assert a clean throw.
#
#   Do not add the marker just to silence the lint. If the test needs a GPU, it needs
#   GPU_TEST_CASE.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELIOS_BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Test files known to contain GPU-dependent test cases. Plugins whose tests never construct a
# GPU-backed model do not need to be listed.
DEFAULT_FILES=(
  "$HELIOS_BASE_DIR/plugins/radiation/tests/selfTest.cpp"
)

if [ $# -gt 0 ]; then
  FILES=("$@")
else
  FILES=("${DEFAULT_FILES[@]}")
fi

# Constructs that require a live ray tracing backend. Extend this list rather than weakening
# the check if another GPU-backed type gains a test. The subclass patterns matter: a test that
# derives a probe type from RadiationModel to reach a protected member (e.g. FlagProbe) still
# runs the base constructor, and so still needs a backend, but never names RadiationModel in a
# declaration of its own.
# Parentheses are doubly escaped: awk processes the string's escapes once when compiling it
# into a dynamic regex, so '\\(' here is what reaches the matcher as a literal '('.
GPU_CONSTRUCTION='RadiationModel[ \t]+[A-Za-z_]+[ \t]*\\(|createWithSharedDevice|public[ \t]+RadiationModel|:[ \t]*RadiationModel[ \t]*\\('

FOUND=0

for file in "${FILES[@]}"; do
  if [ ! -f "$file" ]; then
    echo "ERROR (lint_gpu_tests.sh): file not found: $file" >&2
    exit 1
  fi

  # Walk the file tracking the enclosing test case. A DOCTEST_TEST_CASE whose body constructs a
  # GPU-backed model, and which carries no GPU-LINT-OK marker, is reported.
  output=$(awk -v gpu_ctor="$GPU_CONSTRUCTION" '
    function flush_case() {
      if (tc_line && !guarded && uses_gpu && !allowed) {
        printf "%s:%d: %s\n", FILENAME, tc_line, tc_name
      }
    }
    /^(DOCTEST_TEST_CASE|GPU_TEST_CASE)[ \t]*\(/ {
      flush_case()
      guarded = ($0 ~ /^GPU_TEST_CASE[ \t]*\(/)
      tc_line = NR
      tc_name = $0
      sub(/[ \t]*\{[ \t]*$/, "", tc_name)
      uses_gpu = 0
      allowed = 0
      next
    }
    /GPU-LINT-OK/ { allowed = 1 }
    $0 ~ gpu_ctor { if (tc_line) uses_gpu = 1 }
    END { flush_case() }
  ' "$file")

  if [ -n "$output" ]; then
    FOUND=1
    {
      echo "ERROR (lint_gpu_tests.sh): GPU-dependent test case(s) not declared with GPU_TEST_CASE:"
      echo
      echo "$output" | sed 's/^/  /'
      echo
      echo "  Constructing a RadiationModel requires a working ray tracing backend, so these"
      echo "  test cases throw on the non-GPU CI runners even when the logic under test is"
      echo "  CPU-only. Change DOCTEST_TEST_CASE to GPU_TEST_CASE, or - if the test genuinely"
      echo "  must run without a GPU - add a 'GPU-LINT-OK: <reason>' comment inside the body."
      echo
      echo "  Reproduce the CI failure locally with:  utilities/run_tests.sh --nogpu --test radiation"
    } >&2
  fi
done

if [ "$FOUND" -eq 0 ]; then
  echo "lint_gpu_tests.sh: no unguarded GPU-dependent test cases found."
fi

exit "$FOUND"
