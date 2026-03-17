#!/usr/bin/env bash

################################################
##  Install package dependencies for Helios.  ##
#
## Use one of the following arguments to choose
## which dependencies to install. Default (if
## no argument provided) is "all".
#
# ARGUMENT
#   option: Choice of which dependencies to
#           install (default is "all").
#           See below.
#
# OPTIONS
#   BASE: Install GCC, G++, CMake, and OpenMP
#         Required to run Helios.
#   VIS:  Install base dependencies + X11/xorg
#         Required for Visualizer plugin.
#   CUDA: Install base dependencies + CUDA
#         Required for Aerial LiDAR plugin.
#         Optional for Radiation plugin (enables
#         OptiX backend on NVIDIA systems).
#   ALL:  Install dependencies for ALL plugins
#
# EXAMPLE
#   source dependencies.sh BASE
#
################################################

DEPENDENCIES_PATH=("gcc" "g++" "cmake" "wget" "jq") # Base PATH dependencies

# Run bash script as root
if command -v nvcc &> /dev/null; then
    ROOT="sudo"
else
    echo "'sudo' command not found. Please run this script as root."
    ROOT=""
fi

# Runs command and clears output from terminal after completion.
run_command_clear_output() {
    run_command="$1"
    out_file=$(mktemp)
    # Install package and store output to temporary file
    eval "$run_command" 2>&1 | tee "$out_file"
    # Clear output from terminal
    num_lines=$(wc -l < "$out_file")
    for ((i=0; i<num_lines; i++)); do
        tput cuu1
        tput el
    done
    # Remove temporary file
    rm "$out_file"
}

# Checks if element is in a list
is_in_list() {
    search="$1"
    shift
    list=("$@")
    for element in "${list[@]}"; do
        if [[ "$element" == "$search" ]]; then
            return 0
        fi
    done
    return 1
}

arg=$(echo "$1" | tr '[:upper:]' '[:lower:]') # case-insensitive command-line argument

ARGS=("base" "vis" "cuda" "all") # valid arguments; default is 'all'

# Determine packages to install (base, vis, cuda, or all)
if [ -z "$1" ]; then
    MODE="all"
else
    if is_in_list "$arg" "${ARGS[@]}"; then
        MODE="$arg"
    else
        MODE="all"
    fi
fi

# Check if the host is Windows, macOS, or Linux
if [[ "$(uname -s)" == *"MINGW"* || "$(uname -s)" == *"CYGWIN"* ]]; then
    echo -e "Host is Windows. Dependencies need to be installed manually."
    echo -e "Please install Visual Studio: https://visualstudio.microsoft.com/downloads/"
    echo -e "Please install CUDA: https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64"
    exit 0
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "Installing $MODE dependencies for macOS host...\n"
    PACKAGE_MANAGER="brew"
    DEPENDENCIES_PATH+=("pybind11" "libomp")
    # Note: Caskroom and cask are built into modern Homebrew, no need to install separately
    if [[ "$MODE" == "all" || "$MODE" == "vis" ]]; then
        DEPENDENCIES_PATH+=("xquartz")
    fi
    if [[ "$MODE" == "all" || "$MODE" == "cuda" ]]; then
#        DEPENDENCIES_PATH+=("cuda")
        echo "Host is macOS. CUDA cannot be installed."
    fi
    CHECK_EXISTS="brew list"
    FLAG=""
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "Installing $MODE dependencies for Linux host...\n"
    if command -v apt &> /dev/null; then
        PACKAGE_MANAGER="$ROOT apt-get"
        CHECK_EXISTS="dpkg -l | grep -w -m 1"
    elif command -v yum &> /dev/null; then
        PACKAGE_MANAGER="$ROOT yum"
        CHECK_EXISTS="rpm -qa | grep -w -m 1"
    elif command -v dnf &> /dev/null; then
        PACKAGE_MANAGER="$ROOT dnf"
        CHECK_EXISTS="rpm -qa | grep -w -m 1"
    elif command -v tdnf &> /dev/null; then
        PACKAGE_MANAGER="$ROOT tdnf"
        CHECK_EXISTS="rpm -qa | grep -w -m 1"
    elif command -v zypper &> /dev/null; then
        PACKAGE_MANAGER="$ROOT zypper"
        CHECK_EXISTS="zypper se --installed-only | grep -w -m 1"
    elif command -v pacman &> /dev/null; then
        PACKAGE_MANAGER="$ROOT pacman"
        CHECK_EXISTS="pacman -Qs | grep -w -m 1"
    else
        echo "No package manager detected. Exiting..."
        exit 1
    fi
    # Add OpenMP development packages for Linux based on package manager
    if command -v apt &> /dev/null; then
        DEPENDENCIES_PATH+=("libomp-dev")
    elif command -v yum &> /dev/null || command -v dnf &> /dev/null || command -v tdnf &> /dev/null; then
        DEPENDENCIES_PATH+=("libgomp-devel")
    elif command -v zypper &> /dev/null; then
        DEPENDENCIES_PATH+=("libgomp-devel")
    elif command -v pacman &> /dev/null; then
        DEPENDENCIES_PATH+=("openmp")
    fi
    if [[ "$MODE" == "all" || "$MODE" == "vis" ]]; then
        DEPENDENCIES_PATH+=("libx11-dev" "xorg-dev" "libgl1-mesa-dev" "libglu1-mesa-dev" "libxrandr-dev" "python3-dev" "pybind11-dev")
    fi
    FLAG="-y"
    export DEBIAN_FRONTEND=noninteractive # Avoid timezone prompts
else
    echo "Unsupported OS."
    exit 1
fi

# Update package list
run_command_clear_output "$PACKAGE_MANAGER update $FLAG"

# Install PATH dependencies
for package in "${DEPENDENCIES_PATH[@]}"; do
    if command -v "$package" &> /dev/null; then
        echo "$package already installed at: $(command -v $package)"
    elif eval "$CHECK_EXISTS \"$package\" &> /dev/null"; then
        echo "$package already installed."
    else
        echo "Installing $package..."
        run_command_clear_output "$PACKAGE_MANAGER install $FLAG $package"
        if command -v "$package" &> /dev/null; then
            echo "$package installed at: $(command -v $package)"
        else
            echo "$package installed."
        fi
    fi
done

# If host is macOS or host is Linux and CUDA not needed, dependencies already installed successfully.
if [[ "$OSTYPE" == "darwin"* || "$MODE" == "base" || "$MODE" == "vis" ]]; then
  echo "Finished installing dependencies."
  exit 0
fi

# Get Linux distribution & version
os_name=$(cat /etc/os-release | grep "^NAME=" | cut -d '"' -f 2 | awk '{print $1}')
version_id=$(cat /etc/os-release | grep "^VERSION_ID=" | cut -d '"' -f 2)
architecture=$(uname -m)
distro="${os_name}_${version_id}_${architecture}"
if cat /proc/version | grep -o WSL &> /dev/null; then
    if cat /proc/version | grep -o WSL2 &> /dev/null; then
        distro="WSL"
    else
        echo "Install the latest version of WSL2 for running Linux GUI applications!"
        exit 1
    fi
fi

# If host is Linux, need to install CUDA
if command -v nvcc &> /dev/null; then
    echo "CUDA version $( nvcc --version | grep -oP 'V\d+\.\d+\.\d+' | awk -F'V' '{print $2}' ) already installed at $(command -v nvcc)"
else
    echo "Installing CUDA for $distro..."
    mapfile -t CUDA_COMMANDS < <(jq -r ".\"$distro\"[]" CUDA_install.json)

    # Verify that CUDA_COMMANDS were loaded correctly
    if [ ${#CUDA_COMMANDS[@]} -eq 0 ]; then
        echo "Error: No CUDA installation for $distro found in CUDA_install.json. Exiting..."
        exit 1
    fi

    # Install CUDA
    for install_command in "${CUDA_COMMANDS[@]}"; do
        run_command_clear_output "$ROOT $install_command"
    done
fi

# Add nvcc to path
export PATH=/usr/local/cuda/bin:$PATH

# Fix OptiX drivers for WSL
if [[ "$distro" == "WSL" ]]; then
    # Automatically install Linux drivers version 470.256.02
    DRIVER_URL="https://us.download.nvidia.com/XFree86/Linux-x86_64/470.256.02/NVIDIA-Linux-x86_64-470.256.02.run"
    DRIVER_FILE="NVIDIA-Linux-x86_64-470.256.02.run"
    wget -O $DRIVER_FILE $DRIVER_URL

    LINUX_DRIVER=$(find . -name "NVIDIA-Linux-x86_64-*.run" -print -quit)
    if [[ -n "$LINUX_DRIVER" ]]; then
        VERSION=$(echo "$LINUX_DRIVER" | sed -E 's/.*NVIDIA-Linux-x86_64-([0-9.]+)\.run/\1/')
        echo "Linux Driver Version $VERSION Found. Installing..."
        run_command_clear_output "./$LINUX_DRIVER -x"
        LINUX_DRIVER_PATH="${LINUX_DRIVER%.run}"
    else
        echo "Linux Driver not found. Please place Linux Driver .run file in Helios root directory and rerun this script."
        echo -e "Linux drivers can be downloaded \e]8;;https://www.nvidia.com/en-in/drivers/unix/\aHERE\e]8;;\a. Version \e]8;;https://www.nvidia.in/Download/driverResults.aspx/227064/en-in\a470.256.02\e]8;;\a (https://www.nvidia.in/Download/driverResults.aspx/227064/en-in) recommended."
        exit 1
    fi
    LXSS="/mnt/c/Windows/System32/lxss/lib/"
    OPTIX_DRIVERS_PATH="$(pwd)/$LINUX_DRIVER_PATH/"
    DRIVERS=("libnvoptix.so.1" "libnvidia-ptxjitcompiler.so.1")
    if [[ ! -f "$LXSS/libnvidia-ptxjitcompiler.so.1" || ! -f "$LXSS/libnvoptix.so.1" ]]; then
        mkdir -p "$LXSS"
        ln -s "$OPTIX_DRIVERS_PATH/libnvidia-rtcore.so.$VERSION" "$LXSS/libnvidia-rtcore.so.$VERSION"
        ln -s "$OPTIX_DRIVERS_PATH/libnvidia-ptxjitcompiler.so.$VERSION" "$LXSS/libnvidia-ptxjitcompiler.so.1"
        ln -s "$OPTIX_DRIVERS_PATH/libnvoptix.so.$VERSION" "$LXSS/libnvoptix.so.1"
        export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    fi
fi

echo "Finished installing $MODE dependencies."
exit 0
