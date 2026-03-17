#ifndef DOCTEST_UTILS_H
#define DOCTEST_UTILS_H

#include <doctest.h>
#include <iostream>
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
        static const std::vector<std::string> valid_prefixes = {"-?", "--help", "-h", "-v", "--version", "-c", "--count", "-ltc", "--list-test-cases", "-lts", "--list-test-suites", "-lr", "--list-reporters", "-tc", "--test-case", "-tce",
                                                                "--test-case-exclude", "-sf", "--source-file", "-sfe", "--source-file-exclude", "-ts", "--test-suite", "-tse", "--test-suite-exclude", "-sc", "--subcase", "-sce", "--subcase-exclude",
                                                                "-r", "--reporters", "-o", "--out", "-ob", "--order-by", "-rs", "--rand-seed", "-f", "--first", "-l", "--last", "-aa", "--abort-after", "-scfl", "--subcase-filter-levels",
                                                                // dt- prefixed versions
                                                                "-dt-?", "--dt-help", "-dt-h", "-dt-v", "--dt-version", "-dt-c", "--dt-count", "-dt-ltc", "--dt-list-test-cases", "-dt-lts", "--dt-list-test-suites", "-dt-lr", "--dt-list-reporters",
                                                                "-dt-tc", "--dt-test-case", "-dt-tce", "--dt-test-case-exclude", "-dt-sf", "--dt-source-file", "-dt-sfe", "--dt-source-file-exclude", "-dt-ts", "--dt-test-suite", "-dt-tse",
                                                                "--dt-test-suite-exclude", "-dt-sc", "--dt-subcase", "-dt-sce", "--dt-subcase-exclude", "-dt-r", "--dt-reporters", "-dt-o", "--dt-out", "-dt-ob", "--dt-order-by", "-dt-rs",
                                                                "--dt-rand-seed", "-dt-f", "--dt-first", "-dt-l", "--dt-last", "-dt-aa", "--dt-abort-after", "-dt-scfl", "--dt-subcase-filter-levels"};

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

} // namespace helios

#endif // DOCTEST_UTILS_H
