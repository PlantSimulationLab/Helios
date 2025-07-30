## Common Mistakes to Avoid
- **Never create build directories** - They already exist in all sample projects
- **Never build from `core/` or `plugins/` directories** - Always use `samples/`
- **Never use `mkdir` for build directories** - Just `cd` into the existing build directory

## Code Style
- Code style should follow the style given in `.clang-format`.
- Standard C++ library include headers are listed in the file `core/include/global.h`. Check these includes to make sure you do not add unnecessary includes.
- Prefer descriptive variable names that 'self-document' the code.
- Prefer clearer code over clever optimized code that provides only marginal performance improvements.
- When implementing new function/method definitions, be aware of the organization of the source file. Don't just automatically add them at the end of the file, but keep them grouped with similar definitions if possible.

## Code Structure
- The code is organized into a core library and plugins. The core library contains the main functionality, while plugins provide additional features.
- The folder `samples` contains many examples that illustrate how to run the code. Each of these examples contains a `CMakeLists.txt` file that can be used to build the example. This file references `core/CMake_project.cmake`, which is the highest level CMake file for the core library, and also links any plugin CMake files.
- CMakeLists.txt files inside `core` and `plugins` do not build by themselves. You need to build a project that uses `core/CMake_project.cmake` to build the core library and any plugins.

## Testing

### Test File Organization
- Core tests are located in `core/tests/` with 5 main test header files:
    - `Test_utilities.h`: Vector types, colors, dates/times, coordinate systems
    - `Test_functions.h`: Global helper functions and math utilities
    - `Test_XML.h`: XML parsing functions
    - `Test_context.h`: Context class methods
    - `Test_data.h`: Context data management (primitive, object, global data)
- Plugin tests are in `plugins/[plugin_name]/tests/selfTest.cpp`
- Tests use the doctest framework with specific patterns (DOCTEST_TEST_CASE, DOCTEST_CHECK, etc.)
- When adding new functions or classes, always add a test for it in the appropriate test file.

### Build and Test Workflow
1. **Building Tests**: Always build from a sample directory, not from core or plugins directly
   ```bash
   cd samples/[project_name]/
   cd build  # The build directory already exists - DO NOT create it with mkdir
   cmake .. && make
   ```
   **IMPORTANT**: Never use `mkdir` to create build directories - they already exist in all sample projects
   
2. **Running Tests**: 
   - For core tests: `./context_tests` (from context_selftest/build/)
   - For plugin tests: `./{plugin_name}_selftest` or `./{plugin_name}_tests`
   - If you need to run custom test code from a new .cpp file, add it to the `CMakeLists.txt` in the appropriate sample directory and rebuild.

3. **Common Build Issues**:
   - Always check compilation errors carefully for missing includes or function signature mismatches
   - Plugin tests may have different data label expectations than actual implementation (verify against source code)
   - Tests failing with "does not exist" errors usually indicate incorrect data labels in tests

### Test Coverage
- The script `utilities/generate_coverage_report.sh` can be used to check test coverage.
- The script must be run from the build directory. For example, `cd samples/context_selftest/build && ../../../utilities/generate_coverage_report.sh -t context_tests -r html -l logfile.log`.
- **Prefer HTML output (`-r html`)** over text output - it provides a clear summary table with color-coded coverage percentages and clickable file navigation in `coverage/index.html`.
- Coverage analysis requires tests to compile and run successfully first
- Coverage improvements should focus on:
    - Adding tests for uncovered functions (check function names against test coverage)
    - Verifying data manipulation functions use correct parameter types and labels
    - Testing edge cases and error conditions
    - Ensuring tests are organized with similar functionality and non-redundant

### Debugging Plugin Tests
- When plugin tests fail, first check if the test expectations match the actual implementation
- Always verify test expectations against the source code in `plugins/[name]/src/`
- Tests should not write any errors messages to std::cerr. Use the struct `capture_cerr` (defined in `core/include/global.h`) to capture any error messages and check them in the test.
- Make sure that any functions marked `[[nodiscard]]` assign their return value to a variable in the test, otherwise the compiler will issue a warning.
   
## Documentation
- When adding new functions or classes, always add a docstring to the header file, and consider whether additional documentation is needed (in `doc/*.dox` for the core or `plugins/[plugin_name]/doc/*.dox` for plugins).
- If changes are made to docstrings or function signatures, build the documentation file `doc/Doxyfile` using doxygen to test it. Some notes on this are given below:
- Run `doxygen doc/Doxyfile` to generate the documentation from the root directory not the `doc` directory.
- Check doxygen output for warnings. It is ok to ignore warnings of the form " warning: Member XXX is not documented." 
- The main concern when changing docstrings or function signatures is the potential for breaking \ref references, which produces a warning like:  warning: unable to resolve reference to XXX for \ref command".
    When fixing these, don't just remove the funciton signature or use markdown `` ticks to suppress the warning. It is important to keep the signature in the reference for overloaded functions/methods. Figure out why the function signature is not matching and thus causing Doxygen to treat it as plain text.