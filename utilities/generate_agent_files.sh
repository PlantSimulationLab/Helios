#!/bin/bash

# Writes (or refreshes) AGENTS.md and CLAUDE.md with Helios instructions for AI coding agents in an
# existing Helios project directory. Projects get these files automatically when they are configured
# with CMake; this script is for projects that have not been configured yet.
#
# Usage: utilities/generate_agent_files.sh <project directory>

if [ "$#" != 1 ] || [ "${1}" == "--help" ]; then
    echo "Usage: $0 <project directory>"
    echo ""
    echo "Writes AGENTS.md and CLAUDE.md for AI coding agents into an existing Helios project."
    echo "Files you have edited and whose first line is no longer the helios-managed marker are left untouched."
    exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "Error: cmake was not found on the PATH. Install CMake 3.15 or newer, which is also needed to build Helios."
    exit 1
fi

UTILPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

PROJECT_DIR="$( cd -- "${1}" &> /dev/null && pwd )"
if [ -z "${PROJECT_DIR}" ]; then
    echo "Error: Project directory '${1}' does not exist."
    exit 1
fi

cmake -DHELIOS_PROJECT_DIR="${PROJECT_DIR}" -P "${UTILPATH}/agents/generate_agent_files.cmake"
