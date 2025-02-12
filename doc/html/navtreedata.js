/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ ".", "index.html", [
    [ "Helios Documentation", "index.html", null ],
    [ "User's Guide", "usergroup0.html", [
      [ "Overview", "_overview.html", null ],
      [ "Install and Set-up", "_dependent_software.html", null ],
      [ "User/API Guide", "_a_p_i.html", null ],
      [ "File I/O", "_i_o.html", null ],
      [ "Tutorials", "_tutorials.html", null ]
    ] ],
    [ "Plug-ins", "_plug_ins.html", null ],
    [ "Using the CLion IDE with Helios", "_c_lion_i_d_e.html", [
      [ "Setting up CMake and compilers (Toolchains)", "_c_lion_i_d_e.html#CLionToolchains", null ],
      [ "Opening and building a project", "_c_lion_i_d_e.html#CLionProject", null ],
      [ "Debugging code errors", "_c_lion_i_d_e.html#CLionDebug", null ],
      [ "Git/GitHub integration", "_c_lion_i_d_e.html#CLionGit", null ],
      [ "Doxygen", "_c_lion_i_d_e.html#CLionDox", null ]
    ] ],
    [ "Tutorials", "_tutorials.html", "_tutorials" ],
    [ "Overview", "_overview.html", [
      [ "What is an API?", "_overview.html#whatisAPI", null ],
      [ "C++ Prerequisites", "_overview.html#Prereqs", null ],
      [ "Helios Prerequisites", "_overview.html#PrereqsHelios", null ],
      [ "Model Geometry and Data", "_overview.html#ModelGeom", null ],
      [ "The Helios Context", "_overview.html#ContextOverview", null ],
      [ "Model Plug-ins", "_overview.html#PluginsOverview", null ],
      [ "Using the Documentation", "_overview.html#DocOverview", null ]
    ] ],
    [ "Plug-ins", "_plug_ins.html", "_plug_ins" ],
    [ "Install and Set-up", "_dependent_software.html", [
      [ "Which platform to use for Helios programming?", "_dependent_software.html#WhichPlatform", null ],
      [ "Set-up on Windows PC", "_dependent_software.html#SetupPC", [
        [ "Install Microsoft Visual Studio C++ compiler tools", "_dependent_software.html#SetupPCMSVC", null ],
        [ "Setting up basic build functionality", "_dependent_software.html#SetupPCCLion", null ],
        [ "Setting up NVIDIA CUDA", "_dependent_software.html#SetupPCCUDA", [
          [ "Increasing Timout Detection (TDR) Delay", "_dependent_software.html#TdrDelay", null ],
          [ "Manually installing OptiX if using Windows Subsystem for Linux (WSL)", "_dependent_software.html#OptiXWSL", null ]
        ] ]
      ] ],
      [ "Set-up on Linux", "_dependent_software.html#SetupLinux", [
        [ "Setting up basic build functionality", "_dependent_software.html#SetupLinuxCLion", null ],
        [ "Setting up NVIDIA CUDA", "_dependent_software.html#SetupLinuxCUDA", null ],
        [ "Dependencies of the Visualizer Plug-in", "_dependent_software.html#SetupLinuxVis", null ]
      ] ],
      [ "Set-up on Mac", "_dependent_software.html#SetupMac", [
        [ "Setting up basic build functionality", "_dependent_software.html#SetupMacCLion", null ]
      ] ]
    ] ],
    [ "User/API Guide", "_a_p_i.html", [
      [ "Building and Compiling Your Own Projects", "_a_p_i.html#BuildCompile", [
        [ "Basic Directory Structure", "_a_p_i.html#DirStruct", null ],
        [ "Build Directory", "_a_p_i.html#BuildDir", null ],
        [ "Main and auxiliary .cpp files", "_a_p_i.html#Source", null ],
        [ "CMakeLists.txt File", "_a_p_i.html#CMake", null ],
        [ "New Project Script", "_a_p_i.html#DirScript", null ],
        [ "C++ Standard Library Include Files", "_a_p_i.html#GlobalInclude", null ]
      ] ],
      [ "Context", "_a_p_i.html#ContextSect", null ],
      [ "Vector Types", "_a_p_i.html#VecTypes", [
        [ "R-G-B(-A) color vectors", "_a_p_i.html#RGB", null ]
      ] ],
      [ "Coordinate System", "_a_p_i.html#Coord", null ],
      [ "Geometry", "_a_p_i.html#Geom", [
        [ "Primitive Types", "_a_p_i.html#PrimitiveTypes", null ],
        [ "Adding Primitives", "_a_p_i.html#AddingPrims", [
          [ "Adding Patches", "_a_p_i.html#AddingPatch", null ],
          [ "Adding Triangles", "_a_p_i.html#AddingTriangle", null ],
          [ "Adding Voxels", "_a_p_i.html#AddingVoxel", null ]
        ] ],
        [ "Primitive Transformations", "_a_p_i.html#PrimTransform", null ],
        [ "Primitive Properties", "_a_p_i.html#PrimProps", null ],
        [ "Texture Mapping", "_a_p_i.html#Texture", null ],
        [ "Coloring Primitives by Texture Map", "_a_p_i.html#TextureColor", null ],
        [ "Masking Primitives by Image Transparency Channel", "_a_p_i.html#TextureMask", null ],
        [ "Compound Geometry", "_a_p_i.html#Compound", null ],
        [ "Objects", "_a_p_i.html#Objects", null ]
      ] ],
      [ "Data Structures", "_a_p_i.html#Data", [
        [ "Primitive Data", "_a_p_i.html#PrimData", [
          [ "Setting Primitive Data Values", "_a_p_i.html#SetPrimData", null ],
          [ "Getting Primitive Data Values", "_a_p_i.html#GetPrimData", null ],
          [ "Primitive Data Query Functions", "_a_p_i.html#PrimDataHelpers", null ]
        ] ],
        [ "Global Data", "_a_p_i.html#GlobalData", null ],
        [ "Data Timeseries (Weather Inputs)", "_a_p_i.html#DataTimeseries", null ]
      ] ]
    ] ],
    [ "File Input/Output", "_i_o.html", [
      [ "XML File Structure", "_i_o.html#XMLstructure", [
        [ "Adding Primitives", "_i_o.html#PrimXML", null ],
        [ "Adding Timeseries Data", "_i_o.html#TimeXML", null ]
      ] ],
      [ "Adding Timeseries (Weather) Data from Tabular Text Files", "_i_o.html#ASCIItimeseries", null ],
      [ "Reading XML Files", "_i_o.html#XMLread", null ],
      [ "Reading Standard Polygon File Formats", "_i_o.html#Poly", [
        [ "Reading PLY (Stanford Polygon) Files", "_i_o.html#PLYread", null ],
        [ "Reading OBJ (Wavefront) Files", "_i_o.html#OBJread", null ],
        [ "Writing PLY (Stanford Polygon) Files", "_i_o.html#PLYwrite", null ],
        [ "Writing OBJ (Wavefront) Files", "_i_o.html#OBJwrite", null ]
      ] ],
      [ "Exporting Project to XML File Format", "_i_o.html#Export", null ],
      [ "Exporting Primitive Data to Text File", "_i_o.html#ExportASCII", null ]
    ] ],
    [ "Writing Plugins", "_plugins.html", [
      [ "Introduction", "_plugins.html#PluginIntro", null ],
      [ "Writing Documentation", "_plugins.html#PluginWriting", null ],
      [ "CMakeLists.txt file", "_plugins.html#cmake", [
        [ "Include Directories", "_plugins.html#include", null ]
      ] ]
    ] ],
    [ "Making texture mask files with transparency using GIMP", "_making_masks.html", [
      [ "Choosing the image", "_making_masks.html#One", null ],
      [ "Rotating and cropping (OPTIONAL)", "_making_masks.html#Two", null ],
      [ "Removing the background", "_making_masks.html#Three", null ],
      [ "Exporting to .png format", "_making_masks.html#Four", null ]
    ] ],
    [ "Converting polygon file formats to .ply using Blender", "_convert_p_l_y.html", null ],
    [ "Choosing the right CUDA and OptiX version", "_choosing_c_u_d_a.html", [
      [ "CUDA Version", "_choosing_c_u_d_a.html#chooseCUDA", null ],
      [ "OptiX Version", "_choosing_c_u_d_a.html#chooseOptiX", null ]
    ] ],
    [ "Increasing graphics driver timeout", "_p_c_g_p_u_timeout.html", null ],
    [ "Dummy Model Plugin Documentation", "_dummy.html", [
      [ "Introduction", "_dummy.html#DummyIntro", null ],
      [ "Dependencies", "_dummy.html#DummyDepends", null ]
    ] ],
    [ "Data Structures", "annotated.html", [
      [ "Data Structures", "annotated.html", "annotated_dup" ],
      [ "Data Structure Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Data Fields", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", "functions_func" ],
        [ "Variables", "functions_vars.html", "functions_vars" ],
        [ "Enumerations", "functions_enum.html", null ],
        [ "Enumerator", "functions_eval.html", null ],
        [ "Related Symbols", "functions_rela.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "Globals", "globals.html", [
        [ "All", "globals.html", null ],
        [ "Functions", "globals_func.html", null ],
        [ "Enumerations", "globals_enum.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"_a_p_i.html",
"_photosynthesis_doc.html#FvCBPhotoParams",
"annotated.html",
"class_li_d_a_rcloud.html#abf2f7677b1a92cba5b51ad0336e3743e",
"class_synthetic_annotation.html#a04117857ddc185b3b3595fcc2cfa97ae",
"classhelios_1_1_context.html#a074b0b3ba7069e416206ab3c84138b93",
"classhelios_1_1_context.html#a8c4ec2d90ccd3d44a97597ecd80d78b6",
"classhelios_1_1_tube.html#aceabd5281c442abf90725bb3728caf8b",
"struct_base_grape_vine_parameters.html#a400d6647ac782d95810095028b3fc896",
"struct_sorghum_canopy_parameters.html#a016f6dbe9526e4411d9b7f1480e562b6"
];

var SYNCONMSG = 'click to disable panel synchronization';
var SYNCOFFMSG = 'click to enable panel synchronization';