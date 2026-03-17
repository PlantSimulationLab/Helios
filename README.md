[![Build and Run on Linux](https://github.com/PlantSimulationLab/Helios/actions/workflows/linux_selftests.yaml/badge.svg?branch=master)](https://github.com/PlantSimulationLab/Helios/actions/workflows/linux_selftests.yaml) [![Build and Run on Windows](https://github.com/PlantSimulationLab/Helios/actions/workflows/windows_selftests.yaml/badge.svg?branch=master)](https://github.com/PlantSimulationLab/Helios/actions/workflows/windows_selftests.yaml) [![Build and Run on MacOS](https://github.com/PlantSimulationLab/Helios/actions/workflows/mac_selftests.yaml/badge.svg?branch=master)](https://github.com/PlantSimulationLab/Helios/actions/workflows/mac_selftests.yaml)

[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/channel/UCUlyE4rKNGokwH5a-TioS0A)

For complete documentation of this software, please consult <a href="https://plantsimulationlab.github.io/Helios/">https://plantsimulationlab.github.io/Helios/.

Helios is a C++ library for 3D physical simulation of plant and environmental systems. It can generate and manipulate plant and other geometric objects, which can feed into biophysical model plug-ins such as radiation transfer, photosynthesis, and evapotranspiration, among others. 

In order to build and compile the core library, you will need to install a C/C++ compiler (recommended are the GNU C compilers version 7.0+), and CMake. In order to run many of the model plug-ins, you will need to install NVIDIA CUDA 10.2+, and a GPU with compute capability 5.0+. The software has been tested on Linux, Mac, and Windows platforms. The YouTube channel linked above has a number of tutorials for getting started.

**NEW** : Helios now has a Python API! Please see the [PyHelios API documentation](https://plantsimulationlab.github.io/PyHelios/) for more information.

<div align="center">
  <img src="https://raw.githubusercontent.com/PlantSimulationLab/PyHelios/master/docs/images/PyHelios_logo_whiteborder.png"  alt="" width="100" />
</div>

![Almond Reconstruction](doc/images/AlmondVarietyReconstruction.png)