#!/bin/bash
#
# Detect POSIX-only constructs in test code that do not compile under MSVC.
#
# Why this exists:
#   The Windows CI runners build the same test sources as Linux and macOS, but with MSVC, which
#   provides no setenv/unsetenv/fork/waitpid/unistd.h. A test using them compiles cleanly on a
#   developer's Mac and on two of the four runners, then fails the Windows build with
#   "error C3861: identifier not found" - roughly 10-30 minutes after the push, and for the
#   whole job, since the failure is in the core library target rather than in a single test.
#   That is exactly what happened in v1.3.79 with setenv()/unsetenv().
#
#   This is the portability counterpart to lint_gpu_tests.sh: a mechanical check that costs
#   seconds locally instead of a CI cycle.
#
# Usage:
#   utilities/lint_test_portability.sh            # lint the known test files
#   utilities/lint_test_portability.sh <file>...  # lint specific files
#
# Exit status:
#   0 - no unguarded POSIX-only constructs found
#   1 - at least one found (details on stderr)
#
# What counts as guarded:
#   Code inside '#ifndef _WIN32' or '#ifdef __unix__'-style blocks is compiled out on Windows and
#   is fine. plugins/radiation/tests/selfTest.cpp does this correctly for its fork()-based test:
#   both the includes and the whole test case sit inside '#ifndef _WIN32'. The checker tracks
#   preprocessor nesting so that pattern reports nothing.
#
#   A POSIX call that has no Windows equivalent and cannot be guarded away usually means the test
#   itself should be Windows-exempt - wrap the test case, not just the call.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELIOS_BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Helios's own test sources. Vendored third-party trees (e.g. plugins/topography/lib/curl-*) are
# deliberately excluded: they ship their own platform configuration and are not built as tests here.
# Built with a while-read loop rather than mapfile/readarray: macOS ships bash 3.2, which has
# neither, and this script must run on a developer's Mac as readily as on the CI runners.
FILES=()
if [ $# -gt 0 ]; then
  FILES=("$@")
else
  while IFS= read -r discovered_file; do
    FILES+=("$discovered_file")
  done < <(
    find "$HELIOS_BASE_DIR/core/tests" "$HELIOS_BASE_DIR/plugins" \
      -path '*/lib/*' -prune -o \
      -path '*/tests/*' -type f \( -name '*.cpp' -o -name '*.h' \) -print 2>/dev/null | sort
  )
fi

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "ERROR (lint_test_portability.sh): no test files found to lint." >&2
  exit 1
fi

# POSIX-only identifiers absent from the MSVC runtime. Each needs either a Windows branch or a
# guard that compiles it out. Deliberately conservative: only entries that genuinely fail to
# compile under MSVC belong here, so that a report is always actionable and never noise.
#
# Not listed, and intentionally so:
#   - std::getenv: standard C++, MSVC provides it (only a C4996 deprecation warning, and the
#     build sets no /WX, so it cannot break).
#   - M_PI and friends: core/include/global.h defines them unconditionally, so the usual
#     _USE_MATH_DEFINES trap does not apply to this codebase.
#
# Word boundaries are spelled out as an explicit "not preceded/followed by an identifier
# character" class rather than \b or \< : BSD awk (the awk on macOS, where developers run this)
# supports neither, and silently treating the pattern as invalid is how a lint ends up reporting
# a clean tree it never actually searched.
NB='[^A-Za-z0-9_]'
# The open parenthesis is quadruply escaped, which is not a typo. Two levels are consumed here:
# bash collapses '\\\\(' to '\\(' inside double quotes, then awk collapses that to '\(' when it
# compiles the -v string into a dynamic regex - so the matcher finally sees a literal '('. Written
# with fewer backslashes, awk receives a bare '(' and dies with "illegal primary in regular
# expression", which is how the first draft of this script silently linted nothing.
POSIX_ONLY="(^|$NB)(setenv|unsetenv|fork|waitpid|pipe2|usleep|gettimeofday|mkdtemp|mkstemp|dlopen|dlsym|dlclose|strcasecmp|strncasecmp|getpid|sysconf|readlink|symlink|opendir|readdir|closedir)[ \t]*\\\\("
# The literal dot before "h>" is written as the character class [.] rather than \. because gawk
# warns ("escape sequence `\.' treated as plain `.'") on the latter while BSD awk stays silent -
# a difference invisible on a developer's Mac that turned into a red Linux CI job. A one-character
# class means the same thing to every awk and needs no escaping at all.
POSIX_HEADERS='#[ \t]*include[ \t]*<(unistd|sys/wait|sys/time|sys/mman|sys/socket|dirent|pthread|dlfcn|libgen|strings|netinet/in|arpa/inet)[.]h>'
POSIX_TYPES="(^|$NB)(pid_t|ssize_t|mode_t|uid_t|gid_t|nlink_t)($NB|\$)"

FOUND=0
awk_error_file="$(mktemp)"
trap 'rm -f "$awk_error_file"' EXIT

for file in "${FILES[@]}"; do
  if [ ! -f "$file" ]; then
    echo "ERROR (lint_test_portability.sh): file not found: $file" >&2
    exit 1
  fi

  # Walk the file tracking preprocessor nesting, so that anything inside a block which excludes
  # Windows is skipped. Only the non-Windows-excluded regions are matched against the patterns.
  output=$(awk \
    -v posix_only="$POSIX_ONLY" \
    -v posix_headers="$POSIX_HEADERS" \
    -v posix_types="$POSIX_TYPES" '
    # Preprocessor state. Two distinct cases must be tracked, because a region can be
    # non-Windows either by being inside a Windows-excluding block or by being the #else of a
    # Windows-only block:
    #   excluded_at    - depth of an open "#ifndef _WIN32"-style block (non-Windows region NOW)
    #   windows_at     - depth of an open "#ifdef _MSC_VER"-style block (Windows region now, so
    #                    its #else branch is the non-Windows one and must be skipped instead)
    BEGIN { depth = 0; excluded_at = 0; windows_at = 0 }

    /^[ \t]*#[ \t]*(if|ifdef|ifndef)/ {
      depth++
      # Compiled out on Windows: #ifndef _WIN32 / #ifndef _MSC_VER, or a positive test for a
      # non-Windows platform such as #ifdef __linux__ / #if defined(__APPLE__).
      if (excluded_at == 0 && windows_at == 0 &&
          ($0 ~ /#[ \t]*ifndef[ \t]+(_WIN32|_MSC_VER|WIN32)/ ||
           $0 ~ /#[ \t]*if[ \t]+!defined[ \t]*\([ \t]*(_WIN32|_MSC_VER|WIN32)/ ||
           $0 ~ /#[ \t]*if(def)?[ \t]+.*(__linux__|__APPLE__|__unix__|__MACH__)/)) {
        excluded_at = depth
      }
      # Compiled ONLY on Windows: #ifdef _WIN32 / #ifdef _MSC_VER / #if defined(_MSC_VER).
      # The body is fine (MSVC is meant to see it); the #else branch is the POSIX one.
      else if (excluded_at == 0 && windows_at == 0 &&
               ($0 ~ /#[ \t]*ifdef[ \t]+(_WIN32|_MSC_VER|WIN32)/ ||
                $0 ~ /#[ \t]*if[ \t]+defined[ \t]*\([ \t]*(_WIN32|_MSC_VER|WIN32)/)) {
        windows_at = depth
      }
      next
    }
    /^[ \t]*#[ \t]*else/ {
      # Flip: the else-branch of a Windows-excluding block is the Windows branch (start checking
      # again), and the else-branch of a Windows-only block is the POSIX branch (stop checking).
      if (excluded_at == depth) { excluded_at = 0 }
      else if (windows_at == depth) { windows_at = 0; excluded_at = depth }
      next
    }
    /^[ \t]*#[ \t]*endif/ {
      if (excluded_at == depth) { excluded_at = 0 }
      if (windows_at == depth) { windows_at = 0 }
      if (depth > 0) { depth-- }
      next
    }

    # Inside a region Windows never compiles - nothing here can break the MSVC build.
    excluded_at > 0 { next }

    # Strip // comments and string literals so prose and messages do not trigger matches.
    {
      line = $0
      sub(/\/\/.*$/, "", line)
      gsub(/"[^"]*"/, "\"\"", line)
    }

    line ~ posix_headers { printf "%s:%d: POSIX-only header: %s\n", FILENAME, NR, trim($0) }
    line ~ posix_only    { printf "%s:%d: POSIX-only function: %s\n", FILENAME, NR, trim($0) }
    line ~ posix_types   { printf "%s:%d: POSIX-only type: %s\n", FILENAME, NR, trim($0) }

    function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
  ' "$file" 2>"$awk_error_file")
  awk_status=$?

  # A lint that cannot run must never look like a lint that found nothing: an awk failure here
  # (an unsupported regex construct, an unreadable file) would otherwise be swallowed and the
  # script would go on to print "no unguarded POSIX-only constructs found" and exit 0, which is
  # precisely the false all-clear this check exists to prevent.
  #
  # Warnings are reported but are NOT fatal, while anything else on stderr is. The distinction
  # matters in both directions: gawk warns where BSD awk is silent, so treating every byte of
  # stderr as fatal (the first version of this check) makes a harmless diagnostic on one platform
  # fail the build on that platform only - which is the same class of bug this script exists to
  # catch, committed by the script itself. Ignoring stderr wholesale would be the opposite error.
  awk_problems=""
  if [ -s "$awk_error_file" ]; then
    awk_problems=$(grep -v -i 'warning' "$awk_error_file" || true)
  fi

  if [ "$awk_status" -ne 0 ] || [ -n "$awk_problems" ]; then
    {
      echo "ERROR (lint_test_portability.sh): the checker itself failed on $file."
      echo "This is a bug in the lint script, not a finding. Its result cannot be trusted."
      [ -s "$awk_error_file" ] && sed 's/^/  /' "$awk_error_file"
    } >&2
    exit 2
  fi

  # Non-fatal, but still surfaced: a warning means one of the patterns above is not written the
  # way this awk expects, so it should be fixed rather than left to accumulate.
  if [ -s "$awk_error_file" ]; then
    {
      echo "WARNING (lint_test_portability.sh): the checker emitted diagnostics on $file."
      echo "The scan still ran; fix the pattern so this stays quiet on every awk."
      sed 's/^/  /' "$awk_error_file"
    } >&2
  fi

  # Truncate rather than delete: the file is reused by the next iteration and removed by the
  # EXIT trap. Deleting it here would leave later iterations writing to a path that no longer
  # exists, and the -s test would then never fire again.
  : > "$awk_error_file"

  if [ -n "$output" ]; then
    FOUND=1
    {
      echo "ERROR (lint_test_portability.sh): POSIX-only construct(s) that MSVC cannot compile:"
      echo
      echo "$output" | sed 's/^/  /'
      echo
      echo "  These build on Linux and macOS and fail the Windows CI job. Either provide a"
      echo "  Windows branch (e.g. _putenv_s for setenv, as core/tests/Test_functions.h does),"
      echo "  or wrap the whole test case in '#ifndef _WIN32' when it has no Windows equivalent"
      echo "  (as the fork()-based test in plugins/radiation/tests/selfTest.cpp does)."
    } >&2
  fi
done

if [ "$FOUND" -eq 0 ]; then
  echo "lint_test_portability.sh: no unguarded POSIX-only constructs found in test code."
fi

exit "$FOUND"
