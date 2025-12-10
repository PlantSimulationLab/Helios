## Common Mistakes to Avoid
- **Never create build directories manually** - Use the unified test system with `utilities/run_tests.sh`
- **Never build from `core/` or `plugins/` directories** - Always use the unified test script which handles all building automatically
- **Don't be too agreeable** - Scrutinize the user's assumptions and suggestions. If they suggest something you do not agree with, don't just automatically implement it. Bring this to their attention and wait for a rebuttal.  
- **Don't implement fallbacks** - Helios follows a fail-fast philosophy - never implement silent fallbacks that hide issues from users. NEVER return fake values (0.0, empty lists, fake IDs), silently catch and ignore exceptions, or continue with misleading fallback functionality when core features fail. Instead, always raise explicit `helios_runtime_error()` with clear, actionable error messages that explain what failed, why it failed, and how to fix it. 
- **Don't implement placeholders** - When implementing new functionality, do not leave placeholder code that does not actually do anything. If you are not sure how to implement a function, leave it unimplemented and raise a `helios_runtime_error()` with a clear message explaining that the function is not yet implemented. This will help users understand that they need to implement the functionality before it can be used.

## CRITICAL Git Safety Rules
**NEVER USE THESE DESTRUCTIVE GIT COMMANDS:**
- **NEVER use `git checkout HEAD`** - This overwrites ALL uncommitted changes and destroys work-in-progress
- **NEVER use `git reset --hard`** - This permanently destroys uncommitted changes 
- **NEVER use `git clean -fd`** - This deletes untracked files that may contain important work
- **NEVER use `git checkout .`** - This discards all local changes in the working directory
- **NEVER use `git stash --include-untracked && git stash drop`** - This permanently destroys stashed changes

**Safe Git Commands Only:**
- Use `git status` to check repository state
- Use `git diff` to see changes before committing
- Use `git add` to stage specific files
- Use `git commit` to save changes with descriptive messages
- If you need to discard changes to a specific file, ask the user first and use `git checkout -- filename` only for that specific file
- If repository state is unclear, ask the user how they want to proceed rather than making destructive changes

**Before Any Git Command:**
1. Always run `git status` first to understand the current state
2. If there are uncommitted changes, ask the user what they want to do with them
3. Never make assumptions about what changes the user wants to keep or discard
4. When in doubt, do nothing and ask the user for explicit instructions

## Sub-Agents
- Make sure to familiarize yourself with the available sub-agents (@.claude/agents/) and use them efficiently:
  - context-gatherer: Expert code archaeologist and information synthesizer specializing in rapidly discovering, analyzing, and summarizing relevant context for development tasks.
  - research-specialist: Expert technical web researcher specializing in finding and evaluating open source codebases, scientific literature, and algorithmic implementation.
  - code-architect: Analyzes existing codebases holistically and create comprehensive implementation plans that consider architectural integrity, maintainability, and scalability.
  - debug-specialist: Expert software debugging specialist with deep C++ and cmake expertise.
  - performance-optimizer: Specialist in code profiling, performance analysis, and bottleneck identification.
  - test-coverage-maximizer: Writes tests that achieve maximum code coverage while ensuring robust functional testing that catches real bugs and edge cases.

## Code Style
- Code style should follow the style given in `.clang-format`.
- Standard C++ library include headers are listed in the file `core/include/global.h`. Check these includes to make sure you do not add unnecessary includes.
- Prefer descriptive variable names that 'self-document' the code.
- Prefer clearer code over clever optimized code that provides only marginal performance improvements.
- When implementing new function/method definitions, be aware of the organization of the source file. Don't just automatically add them at the end of the file, but keep them grouped with similar definitions if possible.
- Use `helios_runtime_error()` for runtime errors. Don't manually throw exceptions or write to std::cerr and exit.

## Code Structure
- The code is organized into a core library and plugins. The core library contains the main functionality, while plugins provide additional features.
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

### Building and Running Tests
1. **Use the Unified Test System**: Always use `utilities/run_tests.sh` to build and run tests
    ```bash
    cd utilities
    ./run_tests.sh  # Runs all tests
    ```
   The script automatically:
   - Creates temporary build directories
   - Configures CMake with appropriate plugins
   - Builds only the required test targets
   - Runs the tests and cleans up
   - Builds in Release mode by default (use `--debugbuild` for Debug mode)

### Test Workflow
1. **Running Tests** (from `utilities/` directory):
    ```bash
    ./run_tests.sh                              # Run all tests
    ./run_tests.sh --test photosynthesis        # Run single plugin test
    ./run_tests.sh --tests "radiation,lidar"    # Run multiple plugin tests
    ./run_tests.sh --test context               # Run core context tests
    ```
   
When debugging, it is recommended to use the --verbose option to see the full build output, which can help identify issues.
It is also recommended when debugging to build to a persistent project directory using the `--project-dir` option, which allows you to reuse the build artifacts and avoid recompilation from scratch on subsequent runs.

2. **Advanced Test Options**:
    ```bash
    ./run_tests.sh --testcase "Test Name"       # Run specific doctest case
    ./run_tests.sh --doctestargs "--help"       # Pass args directly to doctest
    ./run_tests.sh --debugbuild                 # Build in Debug mode
    ./run_tests.sh --nogpu                      # Run only non-GPU tests
    ./run_tests.sh --memcheck                   # Enable memory checking
    ./run_tests.sh --verbose                    # Show full build output
    ./run_tests.sh --log-file test.log          # Log output to file
    ./run_tests.sh --project-dir my_project     # Use persistent project directory
    ```

3. **Optimized Testing Workflow with Persistent Projects**:
   For iterative testing and development, use `--project-dir` to avoid recompilation:
    ```bash
    # Initial setup - creates project and compiles from scratch
    ./run_tests.sh --project-dir my_test_project --test photosynthesis
    
    # Subsequent runs are much faster (reuses compiled project)
    ./run_tests.sh --project-dir my_test_project --doctestargs "--list-test-cases"
    ./run_tests.sh --project-dir my_test_project --testcase "specific test name"
    ./run_tests.sh --project-dir my_test_project --doctestargs "--help"
    
    # IMPORTANT: Clean up when done to avoid lingering directories
    rm -rf my_test_project
    
    # Project directory persists after script completion for reuse
    # Only libraries/changed files are recompiled on subsequent runs
    ```
   
   **Benefits of Persistent Projects**:
   - **Speed**: Subsequent runs are 5-10x faster (no library recompilation)
   - **Efficiency**: Perfect for exploring test cases with `--list-test-cases` then running specific ones
   - **Development**: Ideal for debugging specific tests or running different doctest arguments
   - **Clean**: Still automatically cleans up temporary projects when `--project-dir` not used
   - **CRITICAL**: **Always clean up persistent project directories when finished** - use `rm -rf project_name` to remove them

4**Custom Tests**:
    - It is sometimes necessary to write custom tests for specific functionality such as testing performance. In this case, add a custom project in `samples/`.
    - Follow the pattern of existing projects in `samples/` to create a new test project.
    - You can also use the `utilities/create_project.sh` script to automate creation of new projects.
    - Be sure to clean up the files after you are done.

5**Common Build Issues**:
    - Always check compilation errors carefully for missing includes or function signature mismatches
    - Plugin tests may have different data label expectations than actual implementation (verify against source code)
    - Tests failing with "does not exist" errors usually indicate incorrect data labels in tests

6**Assessing Success/Failure**:
   - Always check for error/warning messages first before declaring success 
   - Look for specific failure indicators like "WARNING", "ERROR", "FAILED"
   - Follow the "Don't be too agreeable" principle - be critical and thorough 
   - Stop and analyze failures instead of proceeding when things are clearly broken
   - All tests should be passing before considering the implementation complete. 100% success rate is the only acceptable outcome.

### Test Coverage
- The script `utilities/generate_coverage_report.sh` generates comprehensive coverage reports for specific tests
- **Usage requires specifying which tests to analyze**:
    ```bash
    cd utilities
    ./generate_coverage_report.sh --test context           # Core context tests
    ./generate_coverage_report.sh --test radiation         # Radiation plugin tests
    ./generate_coverage_report.sh --tests "radiation,lidar" # Multiple plugin tests
    ```
- The script can be run from any directory within the Helios repository
- **Coverage projects are persistent** - reports and build artifacts are preserved in descriptively named directories:
    - `coverage_context/` for single tests
    - `coverage_radiation_etc/` for multiple tests
    - Custom directories with `--project-dir my_coverage`
- **Prefer HTML output (default)** over text output - provides color-coded coverage percentages and clickable file navigation in `coverage/index.html`
- Coverage analysis automatically builds and runs tests, then generates reports
- Coverage improvements should focus on:
    - Adding tests for uncovered functions (check function names against test coverage)
    - Verifying data manipulation functions use correct parameter types and labels
    - Testing edge cases and error conditions
    - Ensuring tests are organized with similar functionality and non-redundant

**Coverage Report Options**:
```bash
./generate_coverage_report.sh --test context -r html        # HTML report (default)
./generate_coverage_report.sh --test context -r text        # Text report
./generate_coverage_report.sh --test context --project-dir my_coverage  # Custom directory
```

### Debugging Plugin Tests
- When plugin tests fail, first check if the test expectations match the actual implementation
- Always verify test expectations against the source code in `plugins/[name]/src/`
- Tests should not write any errors messages to std::cerr. Use the struct `capture_cerr` (defined in `core/include/global.h`) to capture any error messages and check them in the test. There is a similar method `capture_cout` if needed.
- Make sure that any functions marked `[[nodiscard]]` assign their return value to a variable in the test, otherwise the compiler will issue a warning.
- **Plugin `selfTest.cpp` files must include `#define DOCTEST_CONFIG_IMPLEMENT` before including `doctest.h`** - without this, doctest will auto-discover ALL tests linked into the binary (including core tests), causing the plugin to run hundreds of unrelated tests instead of just its own.

## Documentation
- When adding new functions or classes, always add a docstring to the header file, and consider whether additional documentation is needed (in `doc/*.dox` for the core or `plugins/[plugin_name]/doc/*.dox` for plugins).
- If changes are made to docstrings or function signatures, build the documentation file `doc/Doxyfile` using doxygen to test it. Some notes on this are given below:
- Run `doxygen doc/Doxyfile` to generate the documentation from the root directory not the `doc` directory.
- Check doxygen output for warnings. It is ok to ignore warnings of the form " warning: Member XXX is not documented."
- The main concern when changing docstrings or function signatures is the potential for breaking \ref references, which produces a warning like:  warning: unable to resolve reference to XXX for \ref command".
  When fixing these, don't just remove the funciton signature or use markdown `` ticks to suppress the warning. It is important to keep the signature in the reference for overloaded functions/methods. Figure out why the function signature is not matching and thus causing Doxygen to treat it as plain text.

**Before Completing an Implementation Task**:
1. Add any tests to the relevant `selfTest.cpp` file based on new code.
2. Review the documentation to ensure it is up-to-date (either in `doc/*.dox` or in `plugins/*/doc/*.dox`).

## MCP: Knowledge-graph memory policy

- Server alias: `memory` (added via `claude mcp add memory npx:@modelcontextprotocol/server-memory`).
- Purpose: persist structured facts and relationships about this repo, projects, and collaborators using the knowledge-graph memory tools.
- Safety: summarize what you plan to store before writing; do not store secrets or API keys.

### When to write memory
Trigger a write when any of the following occur:
1. A new project, module, or dataset is introduced.
2. A design decision or convention is finalized.
3. A collaborator’s role, preference, or responsibility is clarified.

### How to write memory
Use the server’s tools rather than free-form text. Prefer the smallest useful graph entries.

1. Create entities  
   Run the MCP tool `create_entities` with fields `name`, `entityType`, and `observations`. Example:
- “Create an entity for the library ‘Helios EnergyBalanceModel’ with observation summarizing the inputs, outputs, and key files.”

2. Add relations  
   Run `create_relations` to connect entities. Example:
- “Link ‘Helios EnergyBalanceModel’ to ‘SurfaceEnergyBalance’ with relationType ‘implements’.”

3. Update or annotate  
   Use `append_observations` to add a brief dated note when behavior or conventions change.

### When to read memory
Before large refactors, onboarding explanations, or when the task mentions prior decisions, call `search_entities` or `search_relations` with a concise query, then cite what you found.

### Usage examples
- “Search memory for entities about ‘Helios’ and ‘stomatal conductance’ and summarize relevant observations.”
- “Create entities for ‘GEMINI project’ (type: project) and ‘Nonpareil orchard dataset’ (type: dataset), then relate them with relationType ‘uses’.”

### References inside prompts
- To reference MCP resources or trigger tools, you can type `/mcp` in Claude Code to view available servers and tools, or mention the server by name in your instruction, e.g., “Using the `memory` server, run `search_entities` for ‘trellis’.” See Anthropic’s MCP guide for listing and managing servers. 