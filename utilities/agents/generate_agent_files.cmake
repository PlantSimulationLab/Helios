# Generates AGENTS.md and CLAUDE.md in a Helios project directory, so that an AI coding agent started
# in that directory is told where Helios lives, how to build the project, which conventions and
# pitfalls matter, and where the documentation for each plug-in is.
#
# The hand-written guidance lives in helios_agent_guide.md next to this file; the rest of AGENTS.md
# (install path, version, enabled plug-ins, documentation index) is generated from the Helios tree
# so that it always matches the installed version.
#
# This file is used in two ways:
#   1. include()d from core/CMake_project.cmake, so every project gets the files at configure time.
#   2. Run in script mode by utilities/create_project.sh and utilities/generate_agent_files.sh:
#        cmake -DHELIOS_PROJECT_DIR=<project directory> -P generate_agent_files.cmake
#
# A file is only (re)written if it does not exist or still begins with the helios-managed marker
# line. Removing that line makes the file the user's own, and it is never touched again.
#
# Opt-out: configure with -DHELIOS_GENERATE_AGENT_FILES=OFF, set(HELIOS_GENERATE_AGENT_FILES OFF)
# in the project's CMakeLists.txt before the include of CMake_project.cmake, or set the environment
# variable HELIOS_GENERATE_AGENT_FILES=OFF to disable it for every project.

if (CMAKE_SCRIPT_MODE_FILE)
    cmake_minimum_required(VERSION 3.15)
endif ()

set(HELIOS_AGENT_FILE_MARKER "<!-- helios-managed-file")

function(helios_generate_agent_files project_dir helios_base_dir project_name project_plugins)

    get_filename_component(project_dir "${project_dir}" ABSOLUTE)
    get_filename_component(helios_base_dir "${helios_base_dir}" ABSOLUTE)
    list(REMOVE_ITEM project_plugins "")

    # Projects inside the Helios source tree (samples, benchmarks, test builds) are part of the
    # repository and must not gain untracked files. The exception is projects/, which is where
    # create_project.sh suggests putting user projects and which is git-ignored.
    file(RELATIVE_PATH project_rel_to_helios "${helios_base_dir}" "${project_dir}")
    if (NOT project_rel_to_helios MATCHES "^\\.\\./" AND NOT IS_ABSOLUTE "${project_rel_to_helios}")
        if (NOT project_rel_to_helios MATCHES "^projects/")
            return()
        endif ()
    endif ()

    set(guide_file "${helios_base_dir}/utilities/agents/helios_agent_guide.md")
    if (NOT EXISTS "${guide_file}")
        message(FATAL_ERROR "[Helios] Cannot generate AI agent files: '${guide_file}' is missing from the Helios install. "
                            "Restore it from the Helios repository, or disable generation with -DHELIOS_GENERATE_AGENT_FILES=OFF.")
    endif ()
    file(READ "${guide_file}" guide_text)

    # Helios version from the top entry of the changelog.
    set(helios_version "unknown")
    if (EXISTS "${helios_base_dir}/doc/CHANGELOG.md")
        file(STRINGS "${helios_base_dir}/doc/CHANGELOG.md" version_lines REGEX "^# \\[[0-9.]+\\]" LIMIT_COUNT 1)
        if (version_lines MATCHES "^# \\[([0-9.]+)\\]")
            set(helios_version "${CMAKE_MATCH_1}")
        endif ()
    endif ()

    # ---- Documentation index ---- #

    set(index "## Documentation index (generated)\n\n")
    string(APPEND index "Paths are relative to the Helios install at `${helios_base_dir}`.\n\n")
    string(APPEND index "- Core: headers in `core/include/` (start with `Context.h`); user guide (geometry, data, file I/O) `doc/UserGuide.dox`\n")
    file(GLOB tutorial_docs RELATIVE "${helios_base_dir}/doc/tutorials" "${helios_base_dir}/doc/tutorials/tutorial_*.dox")
    if (tutorial_docs)
        list(SORT tutorial_docs)
        list(GET tutorial_docs 0 first_tutorial)
        list(GET tutorial_docs -1 last_tutorial)
        string(REGEX REPLACE "^tutorial_(.*)\\.dox$" "\\1" first_tutorial "${first_tutorial}")
        string(REGEX REPLACE "^tutorial_(.*)\\.dox$" "\\1" last_tutorial "${last_tutorial}")
        string(APPEND index "- Tutorials: `doc/tutorials/tutorial_NN.dox` (${first_tutorial}-${last_tutorial}), walk-throughs of `samples/tutorialN/`\n")
    endif ()

    # Which samples use which plug-in, read from each sample's PLUGINS list.
    file(GLOB sample_cmake_files "${helios_base_dir}/samples/*/CMakeLists.txt")
    list(SORT sample_cmake_files)

    string(APPEND index "\nPlug-in `<name>`: docs in `plugins/<name>/doc/*.dox`, headers in `plugins/<name>/include/`. ")
    string(APPEND index "Format: name - description (additional doc pages) | projects in `samples/` that use it. **[enabled]** = in this project's `PLUGINS`.\n\n")

    file(GLOB plugin_dirs LIST_DIRECTORIES true "${helios_base_dir}/plugins/*")
    list(SORT plugin_dirs)
    foreach (plugin_dir ${plugin_dirs})
        if (NOT IS_DIRECTORY "${plugin_dir}" OR NOT EXISTS "${plugin_dir}/CMakeLists.txt")
            continue()
        endif ()
        get_filename_component(plugin_name "${plugin_dir}" NAME)

        file(GLOB plugin_docs RELATIVE "${plugin_dir}/doc" "${plugin_dir}/doc/*.dox")
        if (NOT plugin_docs)
            continue() # undocumented (e.g., in-development) plug-ins are left out of the index
        endif ()
        list(SORT plugin_docs)

        # Description: the \page title of the plug-in's main doc file, i.e. the one titled
        # "... Plugin Documentation" (a plug-in can have additional pages, such as sub-models).
        set(description "")
        set(main_doc "")
        foreach (doc_name ${plugin_docs})
            file(STRINGS "${plugin_dir}/doc/${doc_name}" page_lines REGEX "\\\\page[ \t]" LIMIT_COUNT 1)
            if (page_lines MATCHES "\\\\page[ \t]+\"?[A-Za-z0-9_]+\"?[ \t]+(.*)$")
                set(page_title "${CMAKE_MATCH_1}")
                if (NOT description OR page_title MATCHES "Plug-?in Documentation")
                    set(description "${page_title}")
                    set(main_doc "${doc_name}")
                endif ()
                if (page_title MATCHES "Plug-?in Documentation")
                    break()
                endif ()
            endif ()
        endforeach ()
        string(REPLACE "%" "" description "${description}")
        string(REGEX REPLACE "[ \t]*(Plug-?in )?Documentation[ \t]*$" "" description "${description}")
        string(STRIP "${description}" description)

        set(using_samples "")
        foreach (sample_cmake ${sample_cmake_files})
            file(STRINGS "${sample_cmake}" plugin_line REGEX "^[ \t]*set[ \t]*\\([ \t]*PLUGINS")
            if (plugin_line MATCHES "\"([^\"]*)\"")
                set(sample_plugins "${CMAKE_MATCH_1}")
                list(FIND sample_plugins "${plugin_name}" found_index)
                if (NOT found_index EQUAL -1)
                    get_filename_component(sample_dir "${sample_cmake}" DIRECTORY)
                    get_filename_component(sample_name "${sample_dir}" NAME)
                    list(APPEND using_samples "${sample_name}")
                endif ()
            endif ()
        endforeach ()

        set(line "- `${plugin_name}`")
        list(FIND project_plugins "${plugin_name}" enabled_index)
        if (NOT enabled_index EQUAL -1)
            string(APPEND line " **[enabled]**")
        endif ()
        if (description)
            string(APPEND line " - ${description}")
        endif ()
        set(extra_docs ${plugin_docs})
        list(REMOVE_ITEM extra_docs "${main_doc}")
        if (extra_docs)
            string(REPLACE ";" ", " extra_docs_text "${extra_docs}")
            string(APPEND line " (${extra_docs_text})")
        endif ()
        if (using_samples)
            string(REPLACE ";" ", " samples_text "${using_samples}")
            string(APPEND line " | ${samples_text}")
        endif ()
        string(APPEND index "${line}\n")
    endforeach ()

    # ---- Assemble the files ---- #

    if (project_plugins)
        string(REPLACE ";" ", " enabled_text "${project_plugins}")
    else ()
        set(enabled_text "none")
    endif ()

    set(marker_line "${HELIOS_AGENT_FILE_MARKER}: regenerated on every CMake configure. Delete this line to keep your own edits; see 'Working with AI Coding Agents' in the Helios user guide. -->")

    set(agents_text "${marker_line}\n\n# Helios project: ${project_name}\n\n")
    string(APPEND agents_text "- Helios install: `${helios_base_dir}` (version ${helios_version})\n")
    string(APPEND agents_text "- Plug-ins enabled in `CMakeLists.txt`: ${enabled_text}\n")
    string(APPEND agents_text "- New projects are created with `${helios_base_dir}/utilities/create_project.sh <dir> [plugins...]`\n\n")
    string(APPEND agents_text "${guide_text}\n${index}")

    set(claude_text "${marker_line}\n\n@AGENTS.md\n")

    _helios_write_managed_file("${project_dir}/AGENTS.md" "${agents_text}" agents_status)
    _helios_write_managed_file("${project_dir}/CLAUDE.md" "${claude_text}" claude_status)

    if (agents_status STREQUAL "written" OR claude_status STREQUAL "written")
        message(STATUS "[Helios] Wrote AI coding agent instructions (AGENTS.md, CLAUDE.md) to ${project_dir}. "
                       "Disable with -DHELIOS_GENERATE_AGENT_FILES=OFF.")
    endif ()

    # Claude Code reads AGENTS.md only when there is no CLAUDE.md, so a user's own CLAUDE.md hides
    # the generated instructions unless it imports them.
    if (claude_status STREQUAL "user-owned" AND NOT agents_status STREQUAL "user-owned")
        file(READ "${project_dir}/CLAUDE.md" user_claude_text)
        if (NOT user_claude_text MATCHES "@AGENTS\\.md")
            message(STATUS "[Helios] ${project_dir}/CLAUDE.md is your own file, so Claude Code will not read the generated AGENTS.md. "
                           "Add the line '@AGENTS.md' to your CLAUDE.md to include it.")
        endif ()
    endif ()

endfunction()

# Writes content to path if the file is absent or is still Helios-managed. Leaves the file untouched
# when the content is unchanged, so its timestamp only moves when something actually changed.
# Sets status_var to "written", "unchanged", or "user-owned".
function(_helios_write_managed_file path content status_var)
    if (EXISTS "${path}")
        file(READ "${path}" existing_content)
        string(FIND "${existing_content}" "${HELIOS_AGENT_FILE_MARKER}" marker_position)
        if (NOT marker_position EQUAL 0)
            set(${status_var} "user-owned" PARENT_SCOPE)
            return()
        endif ()
        if (existing_content STREQUAL content)
            set(${status_var} "unchanged" PARENT_SCOPE)
            return()
        endif ()
    endif ()
    file(WRITE "${path}" "${content}")
    set(${status_var} "written" PARENT_SCOPE)
endfunction()

# Returns OFF if generation was disabled through the environment variable.
function(_helios_agent_files_env_enabled result_var)
    set(enabled ON)
    if (DEFINED ENV{HELIOS_GENERATE_AGENT_FILES})
        set(env_value "$ENV{HELIOS_GENERATE_AGENT_FILES}")
        if (NOT env_value)
            set(enabled OFF)
        endif ()
    endif ()
    set(${result_var} ${enabled} PARENT_SCOPE)
endfunction()

# ---- Script mode: cmake -DHELIOS_PROJECT_DIR=<dir> -P generate_agent_files.cmake ---- #

if (CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
    if (NOT DEFINED HELIOS_PROJECT_DIR)
        message(FATAL_ERROR "Usage: cmake -DHELIOS_PROJECT_DIR=<project directory> -P ${CMAKE_CURRENT_LIST_FILE}")
    endif ()
    get_filename_component(HELIOS_PROJECT_DIR "${HELIOS_PROJECT_DIR}" ABSOLUTE)
    if (NOT EXISTS "${HELIOS_PROJECT_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "[Helios] '${HELIOS_PROJECT_DIR}' has no CMakeLists.txt, so it is not a Helios project. "
                            "Create the project with utilities/create_project.sh first.")
    endif ()

    _helios_agent_files_env_enabled(env_enabled)
    if (NOT env_enabled)
        message(STATUS "[Helios] HELIOS_GENERATE_AGENT_FILES is OFF in the environment; not generating AI agent files.")
        return()
    endif ()

    # Read the project name and plug-in list from the project's CMakeLists.txt.
    file(STRINGS "${HELIOS_PROJECT_DIR}/CMakeLists.txt" name_line REGEX "^[ \t]*set[ \t]*\\([ \t]*EXECUTABLE_NAME")
    if (name_line MATCHES "\"([^\"]*)\"")
        set(script_project_name "${CMAKE_MATCH_1}")
    else ()
        get_filename_component(script_project_name "${HELIOS_PROJECT_DIR}" NAME)
    endif ()
    set(script_plugins "")
    file(STRINGS "${HELIOS_PROJECT_DIR}/CMakeLists.txt" plugin_line REGEX "^[ \t]*set[ \t]*\\([ \t]*PLUGINS")
    if (plugin_line MATCHES "\"([^\"]*)\"")
        set(script_plugins "${CMAKE_MATCH_1}")
        list(REMOVE_ITEM script_plugins "")
    endif ()

    get_filename_component(script_helios_base "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
    helios_generate_agent_files("${HELIOS_PROJECT_DIR}" "${script_helios_base}" "${script_project_name}" "${script_plugins}")
endif ()
