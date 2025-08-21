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

PLUGINS=("energybalance" "lidar" "aeriallidar" "photosynthesis" "radiation" "solarposition" "stomatalconductance" "topography" "visualizer" "voxelintersection" "weberpenntree" "canopygenerator" "boundarylayerconductance" "leafoptics" "syntheticannotation" "plantarchitecture" "projectbuilder" "planthydraulics" "parameteroptimization" "collisiondetection")
HEADERS=("EnergyBalanceModel.h" "LiDAR.h" "AerialLiDAR.h" "PhotosynthesisModel.h" "RadiationModel.h" "SolarPosition.h" "StomatalConductanceModel.h" "Topography.h" "Visualizer.h" "VoxelIntersection.h" "WeberPennTree.h" "CanopyGenerator.h" "BoundaryLayerConductanceModel.h" "LeafOptics.h" "SyntheticAnnotation.h" "PlantArchitecture.h" "ProjectBuilder.h" "PlantHydraulicsModel.h" "ParameterOptimization.h" "CollisionDetection.h")

FILEBASE=$(basename "${DIRPATH}")

if [ "${FILEBASE}" == '.' ]; then
  FILEBASE=$(pwd)
  FILEBASE=$(basename "${FILEBASE}")
fi

#----- build the CMakeLists.txt file ------#

echo -e '# Helios standard CMakeLists.txt file version 1.10' > CMakeLists.txt

echo -e 'cmake_minimum_required(VERSION 3.15)\nproject(helios)\n' >> CMakeLists.txt

echo -e '#-------- USER INPUTS ---------#\n' >> CMakeLists.txt

# Determine if we should use relative or absolute path
# If the project is within the Helios repository structure, use relative path
# Convert DIRPATH to absolute path
if [[ "${DIRPATH}" = /* ]]; then
    PROJECT_ABS_PATH="${DIRPATH}"
else
    PROJECT_ABS_PATH="$(pwd)/${DIRPATH}"
fi
if [[ "${PROJECT_ABS_PATH}" == "${HELIOS_BASE_DIR}"* ]]; then
    # Project is within Helios repository, calculate relative path manually
    # Count directory levels from project to Helios base
    RELATIVE_PATH="${PROJECT_ABS_PATH#${HELIOS_BASE_DIR}}"  # Remove base path prefix
    RELATIVE_PATH="${RELATIVE_PATH#/}"  # Remove leading slash if present
    
    if [ -z "$RELATIVE_PATH" ]; then
        # Project is at Helios base directory
        BASE_DIR_REL="."
    else
        # Count directory levels and build relative path
        # Split on '/' and count non-empty parts
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
    echo -e '#provide the path to Helios base directory, either as an absolute path or a path relative to the location of this file\nset( BASE_DIRECTORY "'"${BASE_DIR_REL}"'" )\n'  >> CMakeLists.txt
else
    # Project is outside Helios repository, use absolute path
    echo -e '#provide the path to Helios base directory, either as an absolute path or a path relative to the location of this file\nset( BASE_DIRECTORY "'"${HELIOS_BASE_DIR}"'" )\n'  >> CMakeLists.txt
fi

echo -e '#define the name of the executable to be created\nset( EXECUTABLE_NAME "'$FILEBASE'" )\n' >> CMakeLists.txt

echo -e '#provide name of source file(s) (separate multiple file names with semicolon)\nset( SOURCE_FILES "main.cpp" )\n' >> CMakeLists.txt

if [ "$#" == 1 ]; then
    echo -e '#specify which plug-ins to use (separate plug-in names with semicolon)\nset( PLUGINS "" )\n' >> CMakeLists.txt
else
    echo -ne '#specify which plug-ins to use (separate plug-in names with semicolon)\nset( PLUGINS "' >> CMakeLists.txt

    for argin in "${@:2}"; do
	FOUND=0
	for i in "${PLUGINS[@]}";do
	    if [ "${i}" == "${argin}" ]; then
		echo -ne "${argin};" >> CMakeLists.txt
		FOUND=1
		break;
	    fi
	done
	if [ $FOUND == "0" ]; then
	    echo "Unknown plug-in '${argin}'. Skipping..."
	fi
    done
    echo -e '" )\n\n' >> CMakeLists.txt
fi

echo -e '#-------- DO NOT MODIFY ---------#\ninclude( "${BASE_DIRECTORY}/core/CMake_project.cmake" )\n' >> CMakeLists.txt

#----- build the main.cpp file ------#

if [ "$#" == 1 ]; then
  echo -e '#include "Context.h"' > main.cpp
fi

for argin in "${@:2}"; do
    FOUND=0
    IND=0
    for i in "${PLUGINS[@]}";do
	if [ "${i}" == "${argin}" ]; then
	    echo -e '#include "'${HEADERS[${IND}]}'"' >> main.cpp
	    FOUND=1
	    break;
	fi
	IND=$((IND+1))
    done
    if [ $FOUND == "0" ]; then
	echo "Unknown plug-in '${argin}'. Skipping..."
    fi
done

echo -e '\nusing namespace helios;\n' >> main.cpp

echo -e 'int main(){\n\n\n}' >> main.cpp
