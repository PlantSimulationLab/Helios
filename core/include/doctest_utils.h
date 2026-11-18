/*
 * Copyright (C) 2016-2026 Brian Bailey
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef DOCTEST_UTILS_H
#define DOCTEST_UTILS_H

#include <doctest.h>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace helios {

    /**
     * @brief Validates doctest command line arguments and provides helpful warnings
     *
     * This utility function checks if provided command line arguments are valid doctest
     * options. It warns users about invalid arguments and suggests corrections.
     *
     * @param argc Number of command line arguments
     * @param argv Array of command line argument strings
     * @return true if all arguments are valid, false if any are invalid (but execution continues)
     */
    inline bool validateDoctestArguments(int argc, char **argv) {
        // List of valid doctest arguments (short and long forms)
        // Reference: doctest.h parseOption/parseFlag/DOCTEST_PARSE_AS_BOOL_OR_FLAG calls
        static const std::vector<std::string> valid_prefixes = {
                // Help, version, counting, listing
                "-?", "--help", "-h", "-v", "--version", "-c", "--count",
                "-ltc", "--list-test-cases", "-lts", "--list-test-suites", "-lr", "--list-reporters",
                // Filters (comma-separated args)
                "-tc", "--test-case", "-tce", "--test-case-exclude",
                "-sf", "--source-file", "-sfe", "--source-file-exclude",
                "-ts", "--test-suite", "-tse", "--test-suite-exclude",
                "-sc", "--subcase", "-sce", "--subcase-exclude",
                "-r", "--reporters",
                // String/int options
                "-o", "--out", "-ob", "--order-by", "-rs", "--rand-seed",
                "-f", "--first", "-l", "--last", "-aa", "--abort-after",
                "-scfl", "--subcase-filter-levels",
                // Bool/flag options
                "-s", "--success", "-cs", "--case-sensitive", "-e", "--exit",
                "-d", "--duration", "-m", "--minimal", "-q", "--quiet",
                "-nt", "--no-throw", "-ne", "--no-exitcode", "-nr", "--no-run",
                "-ni", "--no-intro", "-nv", "--no-version", "-nc", "--no-colors",
                "-fc", "--force-colors", "-nb", "--no-breaks", "-ns", "--no-skip",
                "-gfl", "--gnu-file-line", "-npf", "--no-path-filenames",
                "-sfp", "--strip-file-prefixes", "-nln", "--no-line-numbers",
                "-ndo", "--no-debug-output", "-nss", "--no-skipped-summary",
                "-ntio", "--no-time-in-output",
                // dt- prefixed versions (help, version, counting, listing)
                "-dt-?", "--dt-help", "-dt-h", "-dt-v", "--dt-version", "-dt-c", "--dt-count",
                "-dt-ltc", "--dt-list-test-cases", "-dt-lts", "--dt-list-test-suites", "-dt-lr", "--dt-list-reporters",
                // dt- prefixed filters
                "-dt-tc", "--dt-test-case", "-dt-tce", "--dt-test-case-exclude",
                "-dt-sf", "--dt-source-file", "-dt-sfe", "--dt-source-file-exclude",
                "-dt-ts", "--dt-test-suite", "-dt-tse", "--dt-test-suite-exclude",
                "-dt-sc", "--dt-subcase", "-dt-sce", "--dt-subcase-exclude",
                "-dt-r", "--dt-reporters",
                // dt- prefixed string/int options
                "-dt-o", "--dt-out", "-dt-ob", "--dt-order-by", "-dt-rs", "--dt-rand-seed",
                "-dt-f", "--dt-first", "-dt-l", "--dt-last", "-dt-aa", "--dt-abort-after",
                "-dt-scfl", "--dt-subcase-filter-levels",
                // dt- prefixed bool/flag options
                "-dt-s", "--dt-success", "-dt-cs", "--dt-case-sensitive", "-dt-e", "--dt-exit",
                "-dt-d", "--dt-duration", "-dt-m", "--dt-minimal", "-dt-q", "--dt-quiet",
                "-dt-nt", "--dt-no-throw", "-dt-ne", "--dt-no-exitcode", "-dt-nr", "--dt-no-run",
                "-dt-ni", "--dt-no-intro", "-dt-nv", "--dt-no-version", "-dt-nc", "--dt-no-colors",
                "-dt-fc", "--dt-force-colors", "-dt-nb", "--dt-no-breaks", "-dt-ns", "--dt-no-skip",
                "-dt-gfl", "--dt-gnu-file-line", "-dt-npf", "--dt-no-path-filenames",
                "-dt-sfp", "--dt-strip-file-prefixes", "-dt-nln", "--dt-no-line-numbers",
                "-dt-ndo", "--dt-no-debug-output", "-dt-nss", "--dt-no-skipped-summary",
                "-dt-ntio", "--dt-no-time-in-output"};

        auto isValidDoctestArgument = [&](const std::string &arg) -> bool {
            // Check for exact matches or prefix matches with =
            for (const auto &prefix: valid_prefixes) {
                if (arg == prefix || (arg.length() > prefix.length() && arg.substr(0, prefix.length()) == prefix && arg[prefix.length()] == '=')) {
                    return true;
                }
            }
            return false;
        };

        // Validate command line arguments
        std::vector<std::string> invalid_args;
        for (int i = 1; i < argc; ++i) { // Skip program name (argv[0])
            std::string arg(argv[i]);
            if (!isValidDoctestArgument(arg)) {
                invalid_args.push_back(arg);
            }
        }

        // Warn about invalid arguments
        if (!invalid_args.empty()) {
            std::cerr << "WARNING: Invalid or unrecognized test arguments detected:" << std::endl;
            for (const auto &invalid_arg: invalid_args) {
                std::cerr << "  " << invalid_arg << std::endl;
            }
            std::cerr << std::endl;
            std::cerr << "Common valid patterns:" << std::endl;
            std::cerr << "  -tc=\"pattern\"           : Run tests matching pattern" << std::endl;
            std::cerr << "  --test-case=\"pattern\"   : Run tests matching pattern" << std::endl;
            std::cerr << "  -s                      : Include successful assertions in output" << std::endl;
            std::cerr << "  -c                      : Count matching tests" << std::endl;
            std::cerr << "  --help                  : Show full help" << std::endl;
            std::cerr << std::endl;

            // Try to suggest corrections for common mistakes
            for (const auto &invalid_arg: invalid_args) {
                if (!invalid_arg.empty() && invalid_arg[0] != '-') {
                    std::cerr << "Suggestion: Did you mean -tc=\"" << invalid_arg << "\" ?" << std::endl;
                }
            }
            std::cerr << std::endl;
            std::cerr << "Proceeding with test execution..." << std::endl;
            std::cerr << std::endl;

            return false;
        }

        return true;
    }

    /**
     * @brief Check if test filters match any actual tests
     *
     * This function runs doctest in count mode to check if the provided filters
     * match any tests, and warns if no tests are found.
     *
     * @param argc Number of command line arguments
     * @param argv Array of command line argument strings
     * @return true if tests were found or no filters specified, false if filters match no tests
     */
    inline bool checkTestFiltersMatchTests(int argc, char **argv) {
        // Create a separate context to count matching tests
        doctest::Context count_context;
        count_context.applyCommandLine(argc, argv);

        // Add the count option to see how many tests match
        count_context.setOption("count", true);
        count_context.setOption("no-run", true);

        // Capture the output by redirecting cout temporarily
        std::ostringstream count_output;
        std::streambuf *orig_cout = std::cout.rdbuf();
        std::cout.rdbuf(count_output.rdbuf());

        // Run in count mode
        int count_result = count_context.run();

        // Restore cout
        std::cout.rdbuf(orig_cout);

        // Parse the output to get the count
        std::string output = count_output.str();

        // Look for patterns indicating 0 tests found
        if (output.find("unskipped test cases passing the current filters: 0") != std::string::npos || output.find("number of tests: 0") != std::string::npos) {
            // Check if any test filters were actually specified
            bool has_test_filters = false;
            for (int i = 1; i < argc; ++i) {
                std::string arg(argv[i]);
                if (arg.find("-tc=") == 0 || arg.find("--test-case=") == 0 || arg.find("-ts=") == 0 || arg.find("--test-suite=") == 0 || arg.find("-sc=") == 0 || arg.find("--subcase=") == 0 || arg.find("-dt-tc=") == 0 ||
                    arg.find("--dt-test-case=") == 0 || arg.find("-dt-ts=") == 0 || arg.find("--dt-test-suite=") == 0 || arg.find("-dt-sc=") == 0 || arg.find("--dt-subcase=") == 0) {
                    has_test_filters = true;
                    break;
                }
            }

            if (has_test_filters) {
                std::cerr << "WARNING: No tests match the specified filters!" << std::endl;
                std::cerr << "Your filter criteria resulted in 0 matching tests." << std::endl;
                std::cerr << "Use -ltc or --list-test-cases to see available tests." << std::endl;
                std::cerr << std::endl;
                return false;
            }
        }

        return true;
    }

    /**
     * @brief Enhanced selfTest wrapper that validates arguments before running doctest
     *
     * This template function provides a standardized way for plugins to implement selfTest
     * with argument validation. It should be used in plugin selfTest implementations.
     * The executable name is automatically derived from argv[0].
     *
     * @param argc Number of command line arguments
     * @param argv Array of command line argument strings
     * @return Test result code from doctest
     */
    template<typename ContextType = doctest::Context>
    inline int runDoctestWithValidation(int argc, char **argv) {
        // Validate arguments first
        validateDoctestArguments(argc, argv);

        // Check if test filters match any tests
        checkTestFiltersMatchTests(argc, argv);

        // Run the tests with command line arguments
        ContextType context;
        context.applyCommandLine(argc, argv);
        int res = context.run();

        if (context.shouldExit()) {
            return res;
        }

        return res;
    }

    /**
     * @brief Test helper that runs a scope from a working directory that is not the build directory
     *
     * Asset files shipped with Helios are opened by paths relative to the build directory (e.g. "plugins/solarposition/..."). They must
     * resolve through the HELIOS_BUILD environment variable when the process working directory is somewhere else, which is the case
     * for applications that do not run from the build directory (e.g. the Python bindings). On construction this helper sets
     * HELIOS_BUILD to the build directory, if it is not already set, and changes the working directory to a new, empty temporary
     * directory. On destruction it restores the original working directory, unsets HELIOS_BUILD if this helper set it, and removes the temporary directory.
     *
     * @note The working directory belongs to the whole process, so this must only be used from single-threaded test code.
     * @note When HELIOS_BUILD is not already set, the test must be started from the build directory, which is what run_tests.sh does.
     */
    class ScopedForeignWorkingDirectory {
    public:
        ScopedForeignWorkingDirectory() : original_directory(std::filesystem::current_path()) {
            std::random_device random_source;
            foreign_directory = std::filesystem::temp_directory_path() / ("helios_foreign_cwd_" + std::to_string(random_source()) + std::to_string(random_source()));
            std::filesystem::create_directories(foreign_directory);

            if (std::getenv("HELIOS_BUILD") != nullptr) {
                had_helios_build = true;
            } else {
                setEnvironmentVariable("HELIOS_BUILD", original_directory.string());
            }

            // Changing directory is the last step that can fail; undo the environment change if it does, since the destructor will not run
            try {
                std::filesystem::current_path(foreign_directory);
            } catch (...) {
                if (!had_helios_build) {
                    unsetEnvironmentVariable("HELIOS_BUILD");
                }
                throw;
            }
        }

        ~ScopedForeignWorkingDirectory() {
            std::filesystem::current_path(original_directory);
            if (!had_helios_build) {
                unsetEnvironmentVariable("HELIOS_BUILD");
            }
            std::error_code ignored_error;
            std::filesystem::remove_all(foreign_directory, ignored_error);
        }

        ScopedForeignWorkingDirectory(const ScopedForeignWorkingDirectory &) = delete;
        ScopedForeignWorkingDirectory &operator=(const ScopedForeignWorkingDirectory &) = delete;

    private:
        // POSIX setenv()/unsetenv() do not exist in the MSVC runtime, which provides _putenv_s() instead.
        static void setEnvironmentVariable(const char *name, const std::string &value) {
#ifdef _MSC_VER
            _putenv_s(name, value.c_str());
#else
            setenv(name, value.c_str(), 1);
#endif
        }

        static void unsetEnvironmentVariable(const char *name) {
#ifdef _MSC_VER
            _putenv_s(name, ""); // an empty value deletes the variable on Windows
#else
            unsetenv(name);
#endif
        }

        std::filesystem::path original_directory;
        std::filesystem::path foreign_directory;
        bool had_helios_build = false;
    };

} // namespace helios

#endif // DOCTEST_UTILS_H
