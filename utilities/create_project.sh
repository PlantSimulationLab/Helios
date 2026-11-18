#!/bin/bash

# This script sets up a new project. To use it, first create a new directory where you want the project to be located. Then, run this script and give it the path to that directory, for example: $ ./create_project.sh projects/myProject.

if [ "$#" == 0 ] || [ "${1}" == "--help" ]; then
    echo "Usage: $0 [path] '[plugins]'";
    echo "";
    echo "[path] = Path to project directory.";
    echo "[plugins] = List of plugins to be included in project (separated by spaces).";
    echo "";
    exit 1;
fi

DIRPATH="${1}"

# Validate input path for security
if [[ "${DIRPATH}" =~ [\;\&\|\`\$] ]]; then
    echo "Error: Invalid characters in path. Path cannot contain: ; & | \` $";
    exit 1;
fi
UTILPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
HELIOS_BASE_DIR="$(dirname "$UTILPATH")"

# Convert a shell path to one that native cmake.exe can resolve.
#
# Under Git Bash/MSYS/Cygwin on Windows, pwd yields a POSIX-style path such as
# /c/Users/name/Helios. That form is only meaningful to the MSYS runtime: cmake.exe is a native
# Windows binary and cannot resolve it, so writing it into CMakeLists.txt as BASE_DIRECTORY makes
# configure fail with "include could not find requested file". MSYS auto-converts POSIX paths in
# command-line arguments to native executables, but not text written into a file, so the
# conversion has to be explicit here. On Linux/macOS this is a no-op.
to_cmake_path() {
    case "${OSTYPE}" in
        msys*|cygwin*|mingw*)
            if command -v cygpath >/dev/null 2>&1; then
                cygpath -m "${1}"    # -m gives C:/foo/bar (forward slashes, no escaping needed)
            else
                echo "${1}"
            fi
            ;;
        *) echo "${1}" ;;
    esac
}

# Header that a project includes to use a plug-in. Kept as one name-to-header list: two parallel arrays of names and headers
# fell out of step when a plug-in was added to one and not the other, and new projects then included the wrong header.
plugin_header() {
    case "${1}" in
        aeriallidar) echo "AerialLiDAR.h" ;;
        boundarylayerconductance) echo "BoundaryLayerConductanceModel.h" ;;
        canopygenerator) echo "CanopyGenerator.h" ;;
        collisiondetection) echo "CollisionDetection.h" ;;
        energybalance) echo "EnergyBalanceModel.h" ;;
        leafoptics) echo "LeafOptics.h" ;;
        lidar) echo "LiDAR.h" ;;
        parameteroptimization) echo "ParameterOptimization.h" ;;
        photosynthesis) echo "PhotosynthesisModel.h" ;;
        plantarchitecture) echo "PlantArchitecture.h" ;;
        planthydraulics) echo "PlantHydraulicsModel.h" ;;
        projectbuilder) echo "ProjectBuilder.h" ;;
        radiation) echo "RadiationModel.h" ;;
        soiltransport) echo "SoilTransportModel.h" ;;
        solarposition) echo "SolarPosition.h" ;;
        stomatalconductance) echo "StomatalConductanceModel.h" ;;
        syntheticannotation) echo "SyntheticAnnotation.h" ;;
        topography) echo "Topography.h" ;;
        visualizer) echo "Visualizer.h" ;;
        voxelintersection) echo "VoxelIntersection.h" ;;
        weberpenntree) echo "WeberPennTree.h" ;;
        *) echo "" ;;
    esac
}

# Check every requested plug-in before anything is written, so that a misspelled name stops the script instead of producing a project without it.
for argin in "${@:2}"; do
    header="$(plugin_header "${argin}")"
    if [ -z "${header}" ]; then
        echo "Error: Unknown plug-in '${argin}'."
        exit 1
    fi
    if [ ! -e "${HELIOS_BASE_DIR}/plugins/${argin}/include/${header}" ]; then
        echo "Error: Plug-in '${argin}' was not found: ${HELIOS_BASE_DIR}/plugins/${argin}/include/${header} does not exist."
        exit 1
    fi
done

if [ ! -e "${DIRPATH}" ]; then
    echo "Directory ""${DIRPATH}"" does not exist. Creating it...";
    if ! mkdir -p "${DIRPATH}"; then
        echo "Error: Failed to create directory ""${DIRPATH}""";
        exit 1;
    fi
fi

if ! cd "${DIRPATH}"; then
    echo "Error: Failed to change to directory ""${DIRPATH}""";
    exit 1;
fi

if [ ! -e "build" ]; then
    mkdir build;
fi

FILEBASE=$(basename "${DIRPATH}")

if [ "${FILEBASE}" == '.' ]; then
  FILEBASE=$(pwd)
  FILEBASE=$(basename "${FILEBASE}")
fi

#----- build the CMakeLists.txt file ------#

echo -e '# Helios standard CMakeLists.txt file version 1.10' > CMakeLists.txt

echo -e 'cmake_minimum_required(VERSION 3.15)\nproject(helios)\n' >> CMakeLists.txt

echo -e '#-------- USER INPUTS ---------#\n' >> CMakeLists.txt

# For the common case where we're running from utilities/ and creating a project in samples/
# we can simplify this significantly
if [[ "${DIRPATH}" == "../samples/"* ]]; then
    # Extract just the project name from the path like ../samples/project_name
    PROJECT_NAME="${DIRPATH#../samples/}"
    BASE_DIR_REL="../.."  # From samples/project_name/ to Helios root is ../..
elif [[ "${DIRPATH}" = /* ]]; then
    # Absolute path - calculate normally
    PROJECT_ABS_PATH="${DIRPATH}"
    if [[ "${PROJECT_ABS_PATH}" == "${HELIOS_BASE_DIR}"* ]]; then
        RELATIVE_PATH="${PROJECT_ABS_PATH#${HELIOS_BASE_DIR}}"
        RELATIVE_PATH="${RELATIVE_PATH#/}"
        if [ -z "$RELATIVE_PATH" ]; then
            BASE_DIR_REL="."
        else
            IFS='/' read -ra PATH_PARTS <<< "$RELATIVE_PATH"
            LEVEL_COUNT=0
            for part in "${PATH_PARTS[@]}"; do
                if [ -n "$part" ]; then
                    LEVEL_COUNT=$((LEVEL_COUNT + 1))
                fi
            done
            BASE_DIR_REL=""
            for ((i=0; i<LEVEL_COUNT; i++)); do
                if [ $i -eq 0 ]; then
                    BASE_DIR_REL=".."
                else
                    BASE_DIR_REL="${BASE_DIR_REL}/.."
                fi
            done
        fi
    else
        BASE_DIR_REL="$(to_cmake_path "${HELIOS_BASE_DIR}")"  # Use absolute path (native form for cmake.exe)
    fi
else
    # Other relative paths - use absolute path as fallback for reliability
    BASE_DIR_REL="$(to_cmake_path "${HELIOS_BASE_DIR}")"
fi

echo -e '#provide the path to Helios base directory, either as an absolute path or a path relative to the location of this file\nset( BASE_DIRECTORY "'"${BASE_DIR_REL}"'" )\n'  >> CMakeLists.txt

echo -e '#define the name of the executable to be created\nset( EXECUTABLE_NAME "'$FILEBASE'" )\n' >> CMakeLists.txt

echo -e '#provide name of source file(s) (separate multiple file names with semicolon)\nset( SOURCE_FILES "main.cpp" )\n' >> CMakeLists.txt

if [ "$#" == 1 ]; then
    echo -e '#specify which plug-ins to use (separate plug-in names with semicolon)\nset( PLUGINS "" )\n' >> CMakeLists.txt
else
    echo -ne '#specify which plug-ins to use (separate plug-in names with semicolon)\nset( PLUGINS "' >> CMakeLists.txt

    for argin in "${@:2}"; do
        echo -ne "${argin};" >> CMakeLists.txt
    done
    echo -e '" )\n\n' >> CMakeLists.txt
fi

echo -e '#-------- DO NOT MODIFY ---------#\ninclude( "${BASE_DIRECTORY}/core/CMake_project.cmake" )\n' >> CMakeLists.txt

#----- build the main.cpp file ------#

if [ "$#" == 1 ]; then
  echo -e '#include "Context.h"' > main.cpp
fi

for argin in "${@:2}"; do
    echo -e '#include "'"$(plugin_header "${argin}")"'"' >> main.cpp
done

echo -e '\nusing namespace helios;\n' >> main.cpp

echo -e 'int main(){\n\n\n}' >> main.cpp

#----- write instructions for AI coding agents (AGENTS.md, CLAUDE.md) ------#
# The same files are also (re)generated every time the project is configured with CMake; writing them
# here as well means an agent started in the new directory has them before the first build.
# Disable with the environment variable HELIOS_GENERATE_AGENT_FILES=OFF.

if command -v cmake >/dev/null 2>&1; then
    if ! cmake -DHELIOS_PROJECT_DIR="$(pwd)" -P "${HELIOS_BASE_DIR}/utilities/agents/generate_agent_files.cmake"; then
        echo "Error: Failed to write AI agent instruction files (AGENTS.md, CLAUDE.md) to ${DIRPATH}."
        exit 1
    fi
else
    echo "Note: cmake was not found, so AGENTS.md and CLAUDE.md were not written yet. They will be generated the first time the project is configured with CMake."
fi
