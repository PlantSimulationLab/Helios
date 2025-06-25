#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo
    echo "Options:"
    echo "  --checkout       Clone the latest Helios repo to /tmp and run tests"
    echo "  --nogpu          Run only samples that do not require GPU"
    echo "  --visbuildonly   Build only, do not run visualizer samples"
    echo "  --memcheck       Enable memory checking tools (requires leaks on macOS or valgrind on Linux)"
    echo "  --debugbuild     Build with Debug configuration"
    echo "  --verbose        Show full build/compile output"
    echo
    exit 1
}

SAMPLES=("context_selftest" "visualizer_selftest" "radiation_selftest" "energybalance_selftest" "leafoptics_selftest" "solarposition_selftest" "stomatalconductance_selftest" "photosynthesis_selftest" "weberpenntree_selftest" "lidar_selftest" "aeriallidar_selftest" "voxelintersection_selftest" "canopygenerator_selftest" "boundarylayerconductance_selftest" "syntheticannotation_selftest" "plantarchitecture_selftest" "projectbuilder_selftest" "tutorial0" "tutorial1" "tutorial2" "tutorial5" )
SAMPLES_NOGPU=("context_selftest" "visualizer_selftest" "leafoptics_selftest" "solarposition_selftest" "stomatalconductance_selftest" "photosynthesis_selftest" "weberpenntree_selftest" "canopygenerator_selftest" "boundarylayerconductance_selftest" "syntheticannotation_selftest" "plantarchitecture_selftest" "projectbuilder_selftest" "tutorial0" "tutorial1" "tutorial2" "tutorial5")

TEST_PLUGINS="energybalance lidar aeriallidar photosynthesis radiation leafoptics solarposition stomatalconductance visualizer voxelintersection weberpenntree canopygenerator boundarylayerconductance syntheticannotation plantarchitecture projectbuilder"
TEST_PLUGINS_NOGPU="leafoptics photosynthesis solarposition stomatalconductance visualizer weberpenntree canopygenerator boundarylayerconductance syntheticannotation plantarchitecture projectbuilder"

BUILD_TYPE="Release"
OUTPUT_REDIRECT=" &>/dev/null"

cd ../samples || exit 1

if [[ "${OSTYPE}" != "darwin"* ]] && [[ "${OSTYPE}" != "linux"* ]] && [[ "${OSTYPE}" != "msys"* ]];then
  echo "UNSUPPORTED OPERATING SYSTEM"
  exit 1
fi

while [ $# -gt 0 ]; do
  case $1 in
  --checkout)
    cd /tmp || exit 1

    if [ -e "./helios_test" ]; then
      chmod -R 777 ./helios_test
      rm -r ./helios_test
    fi

    git clone https://www.github.com/PlantSimulationLab/Helios ./helios_test

    if [ ! -e "./helios_test" ]; then
      echo "Git checkout unsuccessful..exiting."
      exit 1
    fi

    chmod -R 777 ./helios_test
    cd ./helios_test/samples || exit 1
    ;;

  --nogpu)
    SAMPLES=("${SAMPLES_NOGPU[@]}")
    TEST_PLUGINS=${TEST_PLUGINS_NOGPU}
    ;;

  --visbuildonly)
    VISRUN="OFF"
    ;;

  --memcheck)
    MEMCHECK="ON"
    #export MallocStackLogging=1
    ;;

  --debugbuild)
    BUILD_TYPE="Debug"
    ;;

  --verbose)
      unset OUTPUT_REDIRECT
      ;;

  --help|-h)
    usage
    ;;

  *)
    echo "Error: Unknown option: $1"
    usage
    ;;
  esac
  shift
done

if [ "${MEMCHECK}" == "ON" ];then
  if [[ "${OSTYPE}" == "darwin"* ]];then
    if [[ $(which leaks) == "" ]];then
      echo "Leaks memory checker tool not installed...ignoring --memcheck argument."
      MEMCHECK="OFF"
    fi
  elif [[ "${OSTYPE}" == "linux"* ]];then
    if [[ $(which valgrind) == "" ]];then
      echo "Valgrind memory checker tool not installed...ignoring --memcheck argument."
      MEMCHECK="OFF"
    fi
  fi
fi

# Check if cmake command is available
if ! command -v cmake &>/dev/null; then
  echo "ERROR: cmake command not found. Please install cmake and make sure it's in your PATH."
  exit 1
fi

if [ "${MEMCHECK}" == "ON" ];then
  BUILD_TYPE="Debug"
fi

ERROR_COUNT=0

if [ -e "temp" ]; then
  rm -rf temp/*
else
  mkdir temp
fi

if [ ! -e "../utilities/create_project.sh" ]; then
  echo -e "\r\x1B[31mProject creation script create_project.sh does not exist...failed.\x1B[39m"
  ERROR_COUNT=$((ERROR_COUNT + 1))
  rm -rf temp
else
  ../utilities/create_project.sh temp ${TEST_PLUGINS}
  cd temp || exit 1
  if [ ! -e "main.cpp" ] || [ ! -e "CMakeLists.txt" ] || [ ! -e "build" ]; then
    echo -e "\r\x1B[31mProject creation script failed to create correct structure...failed.\x1B[39m"
    ERROR_COUNT=$((ERROR_COUNT + 1))
    cd ..
    rm -rf temp
  else

    cd build || exit 1

    echo -ne "Building project creation script test..."

    eval cmake .. -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" ${OUTPUT_REDIRECT}

    if (($? == 0)); then
      echo -e "\r\x1B[32mBuilding project creation script test...done.\x1B[39m"
    else
      echo -e "\r\x1B[31mBuilding project creation script test...failed.\x1B[39m"
      ERROR_COUNT=$((ERROR_COUNT + 1))
    fi

    echo -ne "Compiling project creation script test..."

    eval cmake --build ./ --target temp ${OUTPUT_REDIRECT}

    if (($? == 0)); then
      if [ -e "temp" ]; then
        echo -e "\r\x1B[32mCompiling project creation script test...done.\x1B[39m"
      else
        echo -e "\r\x1B[31mCompiling project creation script test...failed.\x1B[39m"
        ERROR_COUNT=$((ERROR_COUNT + 1))
      fi
    else
      echo -e "\r\x1B[31mCompiling project creation script test...failed.\x1B[39m"
      ERROR_COUNT=$((ERROR_COUNT + 1))
    fi

    echo -ne "Running project creation script test..."

    if [[ "${OSTYPE}" == "msys"* ]];then
      eval ./temp.exe ${OUTPUT_REDIRECT}
    else
      eval ./temp ${OUTPUT_REDIRECT}
    fi

    if (($? == 0)); then
      echo -e "\r\x1B[32mRunning project creation script test...passed.\x1B[39m"
    else
      echo -e "\r\x1B[31mRunning project creation script test...failed.\x1B[39m"
      ERROR_COUNT=$((ERROR_COUNT + 1))
    fi

    cd ../..

    rm -rf temp

  fi
fi

for i in "${SAMPLES[@]}"; do

  if [ ! -e "${i}" ]; then
    echo "Sample ${i} does not exist."
    exit 1
  fi
  if [ ! -e "${i}/build" ]; then
    echo "Build directory does not exist for sample ${i}."
    exit 1
  fi

  cd "$i"/build || exit 1

  rm -rf *

  echo -ne "Building sample ${i}..."

  eval cmake .. -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" ${OUTPUT_REDIRECT}

  if (($? == 0)); then
    echo -e "\r\x1B[32mBuilding sample ${i}...done.\x1B[39m"
  else
    echo -e "\r\x1B[31mBuilding sample ${i}...failed.\x1B[39m"
    ERROR_COUNT=$((ERROR_COUNT + 1))
    rm -rf "$i"/build/*
    cd ../..
    continue
  fi

  echo -ne "Compiling sample ${i}..."

  eval cmake --build ./ --target "${i}" --config "${BUILD_TYPE}" ${OUTPUT_REDIRECT}

  if (($? == 0)); then
    if [ -e "${i}" ]; then
      echo -e "\r\x1B[32mCompiling sample ${i}...done.\x1B[39m"
    else
      echo -e "\r\x1B[31mCompiling sample ${i}...failed.\x1B[39m"
      ERROR_COUNT=$((ERROR_COUNT + 1))
      rm -rf "$i"/build/*
      cd ../..
      continue
    fi
  else
    echo -e "\r\x1B[31mCompiling sample ${i}...failed.\x1B[39m"
    ERROR_COUNT=$((ERROR_COUNT + 1))
    rm -rf "$i"/build/*
    cd ../..
    continue
  fi

  if [ -n "${VISRUN}" ];then
    if  grep -qi visualizer ../CMakeLists.txt ;then
      echo "Skipping run for case ${i} because it uses the visualizer..."
      cd ../..
      continue
    fi
  fi

  echo -ne "Running sample ${i}..."

  if [[ "${OSTYPE}" == "msys"* ]];then
    eval "./${i}.exe" ${OUTPUT_REDIRECT}
  else
    eval "./${i}" ${OUTPUT_REDIRECT}
  fi

  if (($? == 0)); then
    echo -e "\r\x1B[32mRunning sample ${i}...passed.\x1B[39m"
  else
    echo -e "\r\x1B[31mRunning sample ${i}...failed.\x1B[39m"
    ERROR_COUNT=$((ERROR_COUNT + 1))
    rm -rf "$i"/build/*
    cd ../..
    continue
  fi

  if [[ "${MEMCHECK}" == "ON" ]] && [[ "${OSTYPE}" != "msys"* ]];then

    #there are memory leak issues on Linux associated with visualizer libraries
    if [[ "${OSTYPE}" == "linux"* ]];then
      if  grep -qi visualizer ../CMakeLists.txt ;then
          cd ../..
          continue
      fi
    fi

    echo -ne "Running memcheck for sample ${i}..."

    if [[ "${OSTYPE}" == "darwin"* ]];then
      eval leaks --atExit -- "./${i}" ${OUTPUT_REDIRECT}
    else
      eval valgrind --leak-check=full --error-exitcode=1 "./${i}" ${OUTPUT_REDIRECT}
    fi

    if (($? == 0)); then
      echo -e "\r\x1B[32mRunning memcheck for sample ${i}...passed.\x1B[39m"
    else
      echo -e "\r\x1B[31mRunning memcheck for sample ${i}...failed.\x1B[39m"
      ERROR_COUNT=$((ERROR_COUNT + 1))
      rm -rf "$i"/build/*
      cd ../..
      continue
    fi

  fi

  rm -rf ./*

  cd ../..

done

if [ "$1" == "-checkout" ]; then
  cd ../..
  rm -r ./helios_test
fi

if ((ERROR_COUNT == 0)); then
  echo -e "\r\x1B[32mAll cases ran successfully.\x1B[39m"
  exit 0
else
  echo -e "\r\x1B[31mFailed ${ERROR_COUNT} cases.\x1B[39m"
  exit 1
fi
