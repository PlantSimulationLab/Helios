#!/bin/bash

SAMPLES=("context_selftest" "visualizer_selftest" "radiation_selftest" "energybalance_selftest" "solarposition_selftest" "stomatalconductance_selftest" "photosynthesis_selftest" "weberpenntree_selftest" "lidar_selftest" "aeriallidar_selftest" "voxelintersection_selftest" "canopygenerator_selftest" "boundarylayerconductance_selftest" "tutorial0" "tutorial1" "tutorial2" "tutorial5")
SAMPLES_NOGPU=("context_selftest" "solarposition_selftest" "stomatalconductance_selftest" "photosynthesis_selftest" "weberpenntree_selftest" "canopygenerator_selftest" "boundarylayerconductance_selftest" "tutorial0" "tutorial1" "tutorial2" "tutorial5")

TEST_PLUGINS="energybalance lidar aeriallidar photosynthesis radiation solarposition stomatalconductance visualizer voxelintersection weberpenntree canopygenerator boundarylayerconductance"
TEST_PLUGINS_NOGPU="photosynthesis solarposition stomatalconductance visualizer weberpenntree canopygenerator boundarylayerconductance"

cd ../samples

while [ $# -gt 0 ]; do
  case $1 in
  --checkout)

    cd /tmp

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
    cd ./helios_test/samples
    ;;

  --nogpu)
    SAMPLES=("${SAMPLES_NOGPU[@]}")
    TEST_PLUGINS=${TEST_PLUGINS_NOGPU}
    ;;

  esac
  shift
done

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
  cd temp
  if [ ! -e "main.cpp" ] || [ ! -e "CMakeLists.txt" ] || [ ! -e "build" ]; then
    echo -e "\r\x1B[31mProject creation script failed to create correct structure...failed.\x1B[39m"
    ERROR_COUNT=$((ERROR_COUNT + 1))
    cd ..
    rm -rf temp
  else

    cd build

    echo -ne "Building project creation script test..."
    #if  cmake .. 2> /dev/null | grep -q 'Build files have been written to'  ;then
    if ! cmake .. 2>/dev/null | grep -qi 'CMake Error'; then
      #if ! cmake .. | grep -qi 'CMake Error'  ;then
      echo -e "\r\x1B[32mBuilding project creation script test...done.\x1B[39m"
    else
      echo -e "\r\x1B[31mBuilding project creation script test...failed.\x1B[39m"
      ERROR_COUNT=$((ERROR_COUNT + 1))
    fi

    echo -ne "Compiling project creation script test..."

    if make 2>/dev/null | grep -Fq '[100%]'; then
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

    ./temp &>/dev/null

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

  ERROR_CASE=0

  if [ ! -e "${i}" ]; then
    echo "Sample ${i} does not exist."
    exit 0
  fi
  if [ ! -e "${i}/build" ]; then
    echo "Build directory does not exist for sample ${i}."
    exit 0
  fi

  cd "$i"/build

  rm -rf *

  echo -ne "Building sample ${i}..."

  if cmake .. 2>/dev/null | grep -q 'Build files have been written to'; then
    echo -e "\r\x1B[32mBuilding sample ${i}...done.\x1B[39m"
  else
    echo -e "\r\x1B[31mBuilding sample ${i}...failed.\x1B[39m"
    ERROR_CASE=1
    rm -rf "$i"/build/*
    cd ../..
    continue
  fi

  echo -ne "Compiling sample ${i}..."

  if make 2>/dev/null | grep -Fq '[100%]'; then
    if [ -e "${i}" ]; then
      echo -e "\r\x1B[32mCompiling sample ${i}...done.\x1B[39m"
    else
      echo -e "\r\x1B[31mCompiling sample ${i}...failed.\x1B[39m"
      ERROR_CASE=1
      rm -rf "$i"/build/*
      cd ../..
      continue
    fi
  else
    echo -e "\r\x1B[31mCompiling sample ${i}...failed.\x1B[39m"
    ERROR_CASE=1
    rm -rf "$i"/build/*
    cd ../..
    continue
  fi

  echo -ne "Running sample ${i}..."

  "./${i}" &>/dev/null

  if (($? == 0)); then
    echo -e "\r\x1B[32mRunning sample ${i}...passed.\x1B[39m"
  else
    echo -e "\r\x1B[31mRunning sample ${i}...failed.\x1B[39m"
    ERROR_CASE=1
    rm -rf "$i"/build/*
    cd ../..
    continue
  fi

  rm -rf *

  cd ../..

  if ((ERROR_CASE == 1)); then
    ERROR_COUNT=$((ERROR_COUNT + 1))
  fi

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
