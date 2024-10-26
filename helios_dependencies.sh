#!/usr/bin/env bash

DEPENDENCIES_PATH=("gcc" "g++" "cmake" "wget" "jq") # Universal PATH dependencies

# Define function to run command and clear output from terminal after completion
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

# Check if the host is Windows, macOS, or Linux
if [[ "$(uname -s)" == *"MINGW"* || "$(uname -s)" == *"CYGWIN"* ]]; then
    echo "Host is Windows."
    echo -e "Please install \e]8;;https://visualstudio.microsoft.com/downloads/\aVisual Studio\e]8;;\a and \e]8;;https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64\aCUDA\e]8;;\a."
    exit 0
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Host is macOS."
    PACKAGE_MANAGER="brew"
    DEPENDENCIES_VIS=("Caskroom" "xquartz" "cuda") # required for visualizer + radiation
    CHECK_EXISTS="brew list"
    FLAG=""
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Host is Linux."
    PACKAGE_MANAGER="sudo apt-get"
    DEPENDENCIES_VIS=("libx11-dev" "xorg-dev" "libgl1-mesa-dev" "libglu1-mesa-dev" "libxrandr-dev") # required for visualizer
    CHECK_EXISTS="dpkg -l | grep -w -m 1"
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
    else
        echo "Installing $package..."
        run_command_clear_output "$PACKAGE_MANAGER install $FLAG $package"
        echo "$package installed at: $(command -v $package)"
    fi
done

# Install visualizer dependencies
for package in "${DEPENDENCIES_VIS[@]}"; do
    if eval "$CHECK_EXISTS \"$package\" &> /dev/null"; then
        echo "$package already installed."
    else
        echo "Installing $package..."
        run_command_clear_output "$PACKAGE_MANAGER install $FLAG $package"
        echo "$package installed."
    fi
done

# If host is macOS, dependencies installed successfully.
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Finished installing dependencies."
  exit 0
fi

# Get Linux distribution & version
os_name=$(cat /etc/os-release | grep "^NAME=" | cut -d '"' -f 2)
version_id=$(cat /etc/os-release | grep "^VERSION_ID=" | cut -d '"' -f 2)
distro="${os_name}_${version_id}"
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
    echo "Installing CUDA..."
    mapfile -t CUDA_COMMANDS < <(jq -r ".\"$distro\"[]" CUDA_install.json)

    # Verify that CUDA_COMMANDS were loaded correctly
    if [ ${#CUDA_COMMANDS[@]} -eq 0 ]; then
        echo "Can't install CUDA"
        exit 1
    fi

    # Install CUDA
    for install_command in "${CUDA_COMMANDS[@]}"; do
        run_command_clear_output "$install_command"
    done

    # Add nvcc to path
    export PATH=/usr/local/cuda/bin:$PATH
fi

echo "Finished installing dependencies."
exit 0
