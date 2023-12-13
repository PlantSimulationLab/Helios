[![Build and Run on Linux](https://github.com/PlantSimulationLab/Helios/actions/workflows/linux_selftests.yaml/badge.svg?branch=master)](https://github.com/PlantSimulationLab/Helios/actions/workflows/linux_selftests.yaml) [![Build and Run on Windows](https://github.com/PlantSimulationLab/Helios/actions/workflows/windows_selftests.yaml/badge.svg?branch=master)](https://github.com/PlantSimulationLab/Helios/actions/workflows/windows_selftests.yaml) [![Build and Run on MacOS](https://github.com/PlantSimulationLab/Helios/actions/workflows/mac_selftests.yaml/badge.svg?branch=master)](https://github.com/PlantSimulationLab/Helios/actions/workflows/mac_selftests.yaml)

[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/channel/UCUlyE4rKNGokwH5a-TioS0A)

For complete documentation of this software, please consult <a href="https://baileylab.ucdavis.edu/software/helios">https://baileylab.ucdavis.edu/software/helios, or open the file doc/html/index.html in a web browser.

Helios is a C++ API for 3D physical simulation of plant and environmental systems. In order to build and compile the core library, you will need to install a C/C++ compiler (recommended are the GNU C compilers version 5.5+), and CMake. In order to run many of the model plug-ins, you will need to install NVIDIA CUDA 9.0+, and a GPU with compute capability 3.5+. The software has been tested on Linux, Mac, and Windows platforms.

![Almond Reconstruction](doc/images/AlmondVarietyReconstruction.png)