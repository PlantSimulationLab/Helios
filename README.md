[![Build and Run on Linux](https://github.com/PlantSimulationLab/Helios/actions/workflows/linux_selftests.yaml/badge.svg?branch=master)](https://github.com/PlantSimulationLab/Helios/actions/workflows/linux_selftests.yaml) [![Build and Run on Windows](https://github.com/PlantSimulationLab/Helios/actions/workflows/windows_selftests.yaml/badge.svg?branch=master)](https://github.com/PlantSimulationLab/Helios/actions/workflows/windows_selftests.yaml) [![Build and Run on MacOS](https://github.com/PlantSimulationLab/Helios/actions/workflows/mac_selftests.yaml/badge.svg?branch=master)](https://github.com/PlantSimulationLab/Helios/actions/workflows/mac_selftests.yaml)

[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/channel/UCUlyE4rKNGokwH5a-TioS0A)

For complete documentation of this software, please consult <a href="https://baileylab.ucdavis.edu/software/helios">https://baileylab.ucdavis.edu/software/helios, or open the file doc/html/index.html in a web browser.

Helios is a C++ library for 3D physical simulation of plant and environmental systems. It can generate and manipulate plant and other geometric objects, which can feed into biophysical model plug-ins such as radiation transfer, photosynthesis, and evapotranspiration, among others. 

In order to build and compile the core library, you will need to install a C/C++ compiler (recommended are the GNU C compilers version 7.0+), and CMake. In order to run many of the model plug-ins, you will need to install NVIDIA CUDA 10.2+, and a GPU with compute capability 5.0+. The software has been tested on Linux, Mac, and Windows platforms. The YouTube channel linked above has a number of tutorials for getting started.

![Almond Reconstruction](doc/images/AlmondVarietyReconstruction.png)

## Benchmarking

To compare runtime across different machines run the benchmark script on each
system.  You can optionally provide a list of thread counts or log the output:

```bash
./utilities/run_benchmarks.sh "1,4,16" --log-file my_benchmark.log --verbose
```

This produces one results file per benchmark in `benchmarks/report` named
`<benchmark>_<hostname>_<version>_<gpu>.txt`. The first line of each
results file records the Helios version. After collecting reports from multiple
machines they can be plotted with:

```bash
python3 utilities/plot_benchmarks.py benchmarks/report/*.txt
```
