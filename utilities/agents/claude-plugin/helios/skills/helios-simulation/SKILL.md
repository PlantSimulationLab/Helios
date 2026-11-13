---
name: helios-simulation
description: Use when creating, writing, building, running, or debugging a simulation that uses the Helios C++ plant and environmental simulation library (helios::Context, RadiationModel, PlantArchitecture, EnergyBalanceModel, LiDAR, Visualizer and other Helios plug-ins), or when a project's CMakeLists.txt sets BASE_DIRECTORY and includes core/CMake_project.cmake.
---

# Helios simulation projects

The instructions for working with Helios live in the Helios install itself, so that they always match
the installed version. This skill only finds them and makes sure the project has them.

1. **Find the Helios install.**
   - If the working directory (or the project the user named) has a `CMakeLists.txt` that includes
     `core/CMake_project.cmake`, the install is its `BASE_DIRECTORY` value, resolved relative to that
     `CMakeLists.txt` if it is not absolute.
   - Otherwise ask the user where Helios is installed. If they do not have it, point them to
     https://github.com/PlantSimulationLab/Helios and ask before cloning anything.
   - Confirm the path by checking that `<helios>/core/include/Context.h` exists.

2. **Make sure the project has its agent instructions.**
   - New project: create it with `<helios>/utilities/create_project.sh <project dir> [plugins...]`.
     This writes `AGENTS.md` and `CLAUDE.md` into the project.
   - Existing project without an `AGENTS.md`: run `<helios>/utilities/generate_agent_files.sh <project dir>`.
     (The files are also regenerated whenever the project is configured with CMake.)
   - If the project has its own `CLAUDE.md` that does not import `@AGENTS.md`, tell the user that
     adding that line makes Claude Code load the Helios instructions automatically in future sessions.

3. **Read the project's `AGENTS.md` in full and follow it.** It gives the Helios version, build
   commands, conventions, pitfalls, and an index of the documentation, headers, and samples for every
   plug-in. Read the relevant plug-in documentation before writing code that uses that plug-in.

If the generation script or `AGENTS.md` is missing from the Helios install, the install predates this
skill. Tell the user and work from `<helios>/doc/UserGuide.dox` and `<helios>/plugins/<name>/doc/*.dox`.
