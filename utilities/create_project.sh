#!/bin/bash

# This script sets up a new project. To use it, first create a new directory where you want the project to be located. Then, run this script and give it the path to that directory, for example: $ ./create_project.sh projects/myProject.

if [ "$#" == 0 ] || [ "${2}" == "--help" ]; then
    echo "Usage: $0 [path] '[plugins]'";
    echo "";
    echo "[path] = Path to project directory.";
    echo "[plugins] = List of plugins to be included in project (separated by spaces).";
    echo "";
    exit 1;
fi

DIRPATH=${1}

if [ ! -e ${DIRPATH} ]; then
    echo "Directory ""${DIRPATH}"" does not exist.";
    exit 1;
fi

cd ${DIRPATH}

if [ ! -e "build" ]; then
    mkdir build;
fi

PLUGINS=("energybalance" "lidar" "photosynthesis" "radiation" "solarposition" "stomatalconductance" "topography" "visualizer" "voxelintersection" "weberpenntree")
HEADERS=("EnergyBalanceModel.h" "LiDAR.h" "PhotosynthesisModel.h" "RadiationModel.h" "SolarPosition.h" "StomatalConductanceModel.h" "Topography.h" "Visualizer.h" "VoxelIntersection.h" "WeberPennTree.h")

FILEBASE=`basename $DIRPATH`

if [ "${FILEBASE}" == '.' ]; then
    FILEBASE="executable";
fi

#----- build the CMakeLists.txt file ------#

echo -e '# Helios standard CMakeLists.txt file version 1.2\n' > CMakeLists.txt

echo -e '#-------- USER INPUTS ---------#\n' >> CMakeLists.txt

echo -e '#provide the path to Helios base directory, either as an absolute path or a path relative to the location of this file\nset( BASE_DIRECTORY "../.." )\n'  >> CMakeLists.txt

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
    echo -e '" )\n' >> CMakeLists.txt
fi

echo -e '#-------- MAIN CODE (Dont Modify) ---------#\ncmake_minimum_required(VERSION 2.4)\nproject(helios)' >> CMakeLists.txt

echo -e 'SET(CMAKE_CXX_COMPILER_ID "GNU")\nif( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7 )\n\tSET(CMAKE_CXX_FLAGS "-g -std=c++0x")\nelse()\n\tSET(CMAKE_CXX_FLAGS "-g -std=c++11")\nendif()' >> CMakeLists.txt

echo -e 'if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)\nset(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")\nendif()' >> CMakeLists.txt

echo -e 'set( EXECUTABLE_NAME_EXT ${EXECUTABLE_NAME}_exe )\nset( LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib )\nadd_executable( ${EXECUTABLE_NAME_EXT} ${SOURCE_FILES} )\nadd_subdirectory( ${BASE_DIRECTORY}/core "lib" )\ntarget_link_libraries( ${EXECUTABLE_NAME_EXT} helios)' >> CMakeLists.txt

echo -e 'LIST(LENGTH PLUGINS PLUGIN_COUNT)\nmessage("-- Loading ${PLUGIN_COUNT} plug-ins")\nforeach(PLUGIN ${PLUGINS})\n\tmessage("-- loading plug-in ${PLUGIN}")\n\tadd_subdirectory( ${BASE_DIRECTORY}/plugins/${PLUGIN} "plugins/${PLUGIN}" )\n\ttarget_link_libraries( ${EXECUTABLE_NAME_EXT} ${PLUGIN} )\nendforeach(PLUGIN)' >> CMakeLists.txt

echo -e 'include_directories( "${PLUGIN_INCLUDE_PATHS};${CMAKE_CURRENT_SOURCE_DIRECTORY}" )\nadd_custom_command( TARGET ${EXECUTABLE_NAME_EXT} POST_BUILD COMMAND ${CMAKE_COMMAND} -E rename ${EXECUTABLE_NAME_EXT} ${EXECUTABLE_NAME} )' >> CMakeLists.txt

#----- build the main.cpp file ------#

echo -e '#include "Context.h"' > main.cpp

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

echo -e 'int main( void ){\n\n\n}' >> main.cpp
