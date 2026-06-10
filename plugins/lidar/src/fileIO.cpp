/** \file "fileIO.cpp" Declarations for LiDAR plug-in related to file input/output.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "LiDAR.h"
#include "pugixml.hpp"

using namespace helios;
using namespace std;

// Helper function to ensure output directory exists
void ensureOutputDirectoryExists(const char *filename) {
    std::filesystem::path file_path(filename);
    std::filesystem::path dir_path = file_path.parent_path();

    if (!dir_path.empty() && !std::filesystem::exists(dir_path)) {
        std::error_code ec;
        if (!std::filesystem::create_directories(dir_path, ec)) {
            helios_runtime_error("ERROR: Could not create output directory '" + dir_path.string() + "': " + ec.message());
        }
    }
}

void LiDARcloud::loadXML(const char *filename) {
    loadXML(filename, false);
}

void LiDARcloud::loadXML(const char *filename, bool load_grid_only) {

    if (printmessages) {
        cout << "Reading XML file: " << filename << "..." << flush;
    }

    // Check if file exists
    ifstream f(filename);
    if (!f.good()) {
        cerr << "failed.\n";
        helios_runtime_error("ERROR (LiDARcloud::loadXML): XML file does not exist.");
    }

    // Using "pugixml" parser.  See pugixml.org
    pugi::xml_document xmldoc;

    // Resolve file path using project-based resolution
    std::filesystem::path resolved_path = resolveProjectFile(filename);
    std::string resolved_filename = resolved_path.string();
    std::filesystem::path xml_parent_dir = resolved_path.parent_path();

    // load file
    pugi::xml_parse_result result = xmldoc.load_file(resolved_filename.c_str());

    // error checking
    if (!result) {
        cout << "failed." << endl;
        cerr << "XML  file " << filename << " parsed with errors, attribute value: [" << xmldoc.child("node").attribute("attr").value() << "]\n";
        helios_runtime_error("ERROR (LiDARcloud::loadXML): Errors were found while parsing XML file. Error description: " + std::string(result.description()));
    }

    pugi::xml_node helios = xmldoc.child("helios");

    if (helios.empty()) {
        std::cout << "failed." << std::endl;
        helios_runtime_error("ERROR (LiDARcloud::loadXML): XML file must have tag '<helios> ... </helios>' bounding all other tags.");
    }

    //-------------- Scans ---------------//

    uint scan_count = 0; // counter variable for scans
    size_t total_hits = 0;

    if (load_grid_only == false) {

        // looping over any scans specified in XML file
        for (pugi::xml_node s = helios.child("scan"); s; s = s.next_sibling("scan")) {

            // ----- scan origin ------//
            const char *origin_str = s.child_value("origin");

            if (strlen(origin_str) == 0) {
                cerr << "failed.\n";
                helios_runtime_error("ERROR (LiDARcloud::loadXML): An origin was not specified for scan #" + std::to_string(scan_count));
            }

            vec3 origin = string2vec3(origin_str); // note: pugi loads xml data as a character.  need to separate it into 3 floats

            // ----- scan pattern ------//
            // Optional. Default is a raster scan (uniform angular grid). 'spinning_multibeam' models a rotating multi-channel
            // sensor (e.g. Velodyne/Ouster/Hesai) whose channels are specified via <beamElevationAngles>.
            std::string scan_pattern_str = deblank(s.child_value("scanPattern"));
            if (scan_pattern_str.empty()) {
                scan_pattern_str = deblank(s.child_value("scanpattern"));
            }
            std::transform(scan_pattern_str.begin(), scan_pattern_str.end(), scan_pattern_str.begin(), [](unsigned char ch) { return std::tolower(ch); });
            const bool spinning_multibeam = (scan_pattern_str == "spinning_multibeam" || scan_pattern_str == "spinning-multibeam" || scan_pattern_str == "spinningmultibeam");
            if (!scan_pattern_str.empty() && scan_pattern_str != "raster" && !spinning_multibeam) {
                cerr << "failed.\n";
                helios_runtime_error("ERROR (LiDARcloud::loadXML): Unrecognized scanPattern '" + scan_pattern_str + "' for scan #" + std::to_string(scan_count) + ". Valid values are 'raster' and 'spinning_multibeam'.");
            }

            // ----- beam (channel) elevation angles for spinning multibeam ------//
            std::vector<float> beamZenithAngles;
            if (spinning_multibeam) {
                const char *beam_angles_str = s.child_value("beamElevationAngles");
                if (strlen(beam_angles_str) == 0) {
                    beam_angles_str = s.child_value("beamelevationangles");
                }
                if (strlen(beam_angles_str) == 0) {
                    cerr << "failed.\n";
                    helios_runtime_error("ERROR (LiDARcloud::loadXML): A spinning_multibeam scan (#" + std::to_string(scan_count) + ") requires <beamElevationAngles> (space-separated channel elevation angles, in degrees above the horizon).");
                }
                std::istringstream beam_stream(beam_angles_str);
                float elev_deg;
                while (beam_stream >> elev_deg) {
                    // Manufacturer spec sheets list channel angles as elevation above the horizon; convert to Helios zenith (0 = up).
                    beamZenithAngles.push_back(0.5f * float(M_PI) - elev_deg * float(M_PI) / 180.f);
                }
                if (beamZenithAngles.empty()) {
                    cerr << "failed.\n";
                    helios_runtime_error("ERROR (LiDARcloud::loadXML): Could not parse any channel angles from <beamElevationAngles> for scan #" + std::to_string(scan_count) + ".");
                }
            }

            // ----- scan size (resolution) ------//
            // Raster scans require <size> = "Ntheta Nphi". Spinning multibeam scans derive Ntheta from the number of channels
            // (beamElevationAngles) and take the azimuth-step count Nphi from <Nphi> (or the second component of <size>).
            helios::int2 size;
            if (spinning_multibeam) {
                size.x = int(beamZenithAngles.size()); // Ntheta = number of laser channels
                const char *nphi_str = s.child_value("Nphi");
                if (strlen(nphi_str) == 0) {
                    nphi_str = s.child_value("nphi");
                }
                if (strlen(nphi_str) > 0) {
                    size.y = atoi(nphi_str);
                } else {
                    const char *size_str = s.child_value("size");
                    if (strlen(size_str) == 0) {
                        cerr << "failed.\n";
                        helios_runtime_error("ERROR (LiDARcloud::loadXML): A spinning_multibeam scan (#" + std::to_string(scan_count) + ") requires the number of azimuth steps, given as <Nphi> or the second component of <size>.");
                    }
                    size = string2int2(size_str);
                    size.x = int(beamZenithAngles.size());
                }
                if (size.y <= 0) {
                    cerr << "failed.\n";
                    helios_runtime_error("ERROR (LiDARcloud::loadXML): The number of azimuth steps (Nphi) must be positive (check scan #" + std::to_string(scan_count) + ").");
                }
            } else {
                const char *size_str = s.child_value("size");
                if (strlen(size_str) == 0) {
                    cerr << "failed.\n";
                    helios_runtime_error("ERROR (LiDARcloud::loadXML): A size was not specified for scan #" + std::to_string(scan_count));
                } else {
                    size = string2int2(size_str); // note: pugi loads xml data as a character.  need to separate it into 2 ints
                }
                if (size.x <= 0 || size.y <= 0) {
                    cerr << "failed.\n";
                    helios_runtime_error("ERROR (LiDARcloud::loadXML): The scan size must be positive (check scan #" + std::to_string(scan_count) + ").");
                }
            }

            // ----- scan translation ------//
            const char *offset_str = s.child_value("translation");

            vec3 translation = make_vec3(0, 0, 0);
            if (strlen(offset_str) > 0) {
                translation = string2vec3(offset_str); // note: pugi loads xml data as a character.  need to separate it into 3 floats
            }

            // ----- scan rotation ------//
            const char *rotation_str = s.child_value("rotation");

            SphericalCoord rotation_sphere(0, 0, 0);
            if (strlen(rotation_str) > 0) {
                vec2 rotation = string2vec2(rotation_str); // note: pugi loads xml data as a character.  need to separate it into 2 floats
                rotation = rotation * M_PI / 180.f;
                rotation_sphere = make_SphericalCoord(rotation.x, rotation.y);
            }

            // ----- thetaMin ------//
            const char *thetaMin_str = s.child_value("thetaMin");

            float thetaMin;
            if (strlen(thetaMin_str) == 0) {
                // cerr << "WARNING (loadXML): A minimum zenithal scan angle was not specified for scan #" << scan_count << "...assuming thetaMin = 0." << flush;
                thetaMin = 0.f;
            } else {
                thetaMin = atof(thetaMin_str) * M_PI / 180.f;
            }

            if (thetaMin < 0) {
                helios_runtime_error("ERROR (LiDARcloud::loadXML): thetaMin cannot be less than 0.");
            }

            // ----- thetaMax ------//
            const char *thetaMax_str = s.child_value("thetaMax");

            float thetaMax;
            if (strlen(thetaMax_str) == 0) {
                thetaMax = M_PI;
            } else {
                thetaMax = atof(thetaMax_str) * M_PI / 180.f;
            }

            if (thetaMax - 1e-5 > M_PI) {
                helios_runtime_error("ERROR (LiDARcloud::loadXML): thetaMax cannot be greater than 180 degrees.");
            }

            // ----- phiMin ------//
            const char *phiMin_str = s.child_value("phiMin");

            float phiMin;
            if (strlen(phiMin_str) == 0) {
                phiMin = 0.f;
            } else {
                phiMin = atof(phiMin_str) * M_PI / 180.f;
            }

            if (phiMin < 0) {
                helios_runtime_error("ERROR (LiDARcloud::loadXML): phiMin cannot be less than 0.");
            }

            // ----- phiMax ------//
            const char *phiMax_str = s.child_value("phiMax");

            float phiMax;
            if (strlen(phiMax_str) == 0) {
                phiMax = 2.f * M_PI;
            } else {
                phiMax = atof(phiMax_str) * M_PI / 180.f;
            }

            if (phiMax - 1e-5 > 4.f * M_PI) {
                helios_runtime_error("ERROR (LiDARcloud::loadXML): phiMax cannot be greater than 720 degrees.");
            }

            // ----- exitDiameter ------//
            const char *exitDiameter_str_uc = s.child_value("exitDiameter");
            const char *exitDiameter_str_lc = s.child_value("exitdiameter");

            float exitDiameter;
            if (strlen(exitDiameter_str_uc) == 0 && strlen(exitDiameter_str_lc) == 0) {
                exitDiameter = 0;
            } else if (strlen(exitDiameter_str_uc) > 0) {
                exitDiameter = fmax(0, atof(exitDiameter_str_uc));
            } else {
                exitDiameter = fmax(0, atof(exitDiameter_str_lc));
            }

            // ----- beamDivergence ------//
            const char *beamDivergence_str_uc = s.child_value("beamDivergence");
            const char *beamDivergence_str_lc = s.child_value("beamdivergence");

            float beamDivergence;
            if (strlen(beamDivergence_str_uc) == 0 && strlen(beamDivergence_str_lc) == 0) {
                beamDivergence = 0;
            } else if (strlen(beamDivergence_str_uc) > 0) {
                beamDivergence = fmax(0, atof(beamDivergence_str_uc));
            } else {
                beamDivergence = fmax(0, atof(beamDivergence_str_lc));
            }

            // ----- rangeNoiseStdDev ------//
            const char *rangeNoise_str_uc = s.child_value("rangeNoiseStdDev");
            const char *rangeNoise_str_lc = s.child_value("rangenoisestddev");

            float rangeNoiseStdDev;
            if (strlen(rangeNoise_str_uc) == 0 && strlen(rangeNoise_str_lc) == 0) {
                rangeNoiseStdDev = 0;
            } else if (strlen(rangeNoise_str_uc) > 0) {
                rangeNoiseStdDev = fmax(0, atof(rangeNoise_str_uc));
            } else {
                rangeNoiseStdDev = fmax(0, atof(rangeNoise_str_lc));
            }

            // ----- angleNoiseStdDev ------//
            const char *angleNoise_str_uc = s.child_value("angleNoiseStdDev");
            const char *angleNoise_str_lc = s.child_value("anglenoisestddev");

            float angleNoiseStdDev;
            if (strlen(angleNoise_str_uc) == 0 && strlen(angleNoise_str_lc) == 0) {
                angleNoiseStdDev = 0;
            } else if (strlen(angleNoise_str_uc) > 0) {
                angleNoiseStdDev = fmax(0, atof(angleNoise_str_uc));
            } else {
                angleNoiseStdDev = fmax(0, atof(angleNoise_str_lc));
            }

            // ----- scanTilt (global scanner tilt: roll pitch, in degrees) ------//
            const char *scanTilt_str_uc = s.child_value("scanTilt");
            const char *scanTilt_str_lc = s.child_value("scantilt");

            float scanTiltRoll = 0.f;
            float scanTiltPitch = 0.f;
            const char *scanTilt_str = (strlen(scanTilt_str_uc) > 0) ? scanTilt_str_uc : scanTilt_str_lc;
            if (strlen(scanTilt_str) > 0) {
                vec2 scanTilt = string2vec2(scanTilt_str); // "roll pitch" in degrees
                scanTilt = scanTilt * float(M_PI) / 180.f;
                scanTiltRoll = scanTilt.x;
                scanTiltPitch = scanTilt.y;
            }

            // ----- distanceFilter ------//
            const char *dFilter_str = s.child_value("distanceFilter");

            float distanceFilter = -1;
            if (strlen(dFilter_str) > 0) {
                distanceFilter = atof(dFilter_str);
            }

            // ------ ASCII data file format ------- //

            const char *data_format = s.child_value("ASCII_format");

            std::vector<std::string> column_format;
            if (strlen(data_format) != 0) {

                std::string tmp;

                std::istringstream stream(data_format);
                while (stream >> tmp) {
                    column_format.push_back(tmp);
                }
            }

            // create a temporary scan object
            ScanMetadata scan = spinning_multibeam ? ScanMetadata(origin, beamZenithAngles, size.y, phiMin, phiMax, exitDiameter, beamDivergence, rangeNoiseStdDev, angleNoiseStdDev, column_format, scanTiltRoll, scanTiltPitch)
                                                   : ScanMetadata(origin, size.x, thetaMin, thetaMax, size.y, phiMin, phiMax, exitDiameter, beamDivergence, rangeNoiseStdDev, angleNoiseStdDev, column_format, scanTiltRoll, scanTiltPitch);

            addScan(scan);

            uint scanID = getScanCount() - 1;

            // ----- ASCII data file name ------//
            std::string data_filename = deblank(s.child_value("filename"));

            if (!data_filename.empty()) {

                // Resolve the data file. Try in order:
                //   1. input/<data_filename>     (legacy convention)
                //   2. <data_filename>            (cwd-relative or absolute)
                //   3. <xml_parent_dir>/<data_filename>  (sibling of the XML file)
                std::string resolved_data_file;
                std::vector<std::string> candidates;
                candidates.push_back("input/" + data_filename);
                candidates.push_back(data_filename);
                if (!xml_parent_dir.empty()) {
                    candidates.push_back((xml_parent_dir / data_filename).string());
                }
                for (const std::string &candidate: candidates) {
                    ifstream f(candidate);
                    if (f.good()) {
                        resolved_data_file = candidate;
                        break;
                    }
                }
                if (resolved_data_file.empty()) {
                    cout << "failed.\n";
                    helios_runtime_error("ERROR (LiDARcloud::loadXML): Data file `" + data_filename + "' given for scan #" + std::to_string(scan_count) + " does not exist.");
                }

                scan.data_file = resolved_data_file; // set the data file for the scan

                // add hit points to scan if data file was given

                total_hits += loadASCIIFile(scanID, scan.data_file);

                if (translation.magnitude() > 0.f) {
                    coordinateShift(scanID, translation);
                }
                if (rotation_sphere.elevation != 0 || rotation_sphere.azimuth != 0) {
                    coordinateRotation(scanID, rotation_sphere);
                }
            }

            scan_count++;
        }
    }

    //------------ Grids ------------//

    uint cell_count = 0; // counter variable for scans

    // looping over any grids specified in XML file
    for (pugi::xml_node s = helios.child("grid"); s; s = s.next_sibling("grid")) {

        // ----- grid center ------//
        const char *center_str = s.child_value("center");

        if (strlen(center_str) == 0) {
            cerr << "failed.\n";
            helios_runtime_error("ERROR (LiDARcloud::loadXML): A center was not specified for grid #" + std::to_string(cell_count));
        }

        vec3 center = string2vec3(center_str); // note: pugi loads xml data as a character.  need to separate it into 3 floats

        // ----- grid size ------//
        const char *gsize_str = s.child_value("size");

        if (strlen(gsize_str) == 0) {
            cerr << "failed.\n";
            helios_runtime_error("ERROR (LiDARcloud::loadXML): A size was not specified for grid cell #" + std::to_string(cell_count));
        }

        vec3 gsize = string2vec3(gsize_str); // note: pugi loads xml data as a character.  need to separate it into 3 floats

        if (gsize.x <= 0 || gsize.y <= 0 || gsize.z <= 0) {
            cerr << "failed.\n";
            helios_runtime_error("ERROR (LiDARcloud::loadXML): The grid cell size must be positive.");
        }

        // ----- grid rotation ------//
        float rotation;
        const char *grot_str = s.child_value("rotation");

        if (strlen(grot_str) == 0) {
            rotation = 0; // if no rotation specified, assume = 0
        } else {
            rotation = atof(grot_str);
        }

        // ----- grid cells ------//
        uint Nx, Ny, Nz;

        const char *Nx_str = s.child_value("Nx");

        if (strlen(Nx_str) == 0) { // If no Nx specified, assume Nx=1;
            Nx = 1;
        } else {
            Nx = atof(Nx_str);
        }
        if (Nx <= 0) {
            cerr << "failed.\n";
            helios_runtime_error("ERROR (LiDARcloud::loadXML): The number of grid cells must be positive.");
        }

        const char *Ny_str = s.child_value("Ny");

        if (strlen(Ny_str) == 0) { // If no Ny specified, assume Ny=1;
            Ny = 1;
        } else {
            Ny = atof(Ny_str);
        }
        if (Ny <= 0) {
            cerr << "failed.\n";
            helios_runtime_error("ERROR (LiDARcloud::loadXML): The number of grid cells must be positive.");
        }

        const char *Nz_str = s.child_value("Nz");

        if (strlen(Nz_str) == 0) { // If no Nz specified, assume Nz=1;
            Nz = 1;
        } else {
            Nz = atof(Nz_str);
        }
        if (Nz <= 0) {
            cerr << "failed.\n";
            helios_runtime_error("ERROR (LiDARcloud::loadXML): The number of grid cells must be positive.");
        }

        int3 gridDivisions = helios::make_int3(Nx, Ny, Nz);

        // add cells to grid

        vec3 gsubsize = make_vec3(float(gsize.x) / float(Nx), float(gsize.y) / float(Ny), float(gsize.z) / float(Nz));

        float x, y, z;
        uint count = 0;
        for (int k = 0; k < Nz; k++) {
            z = -0.5f * float(gsize.z) + (float(k) + 0.5f) * float(gsubsize.z);
            for (int j = 0; j < Ny; j++) {
                y = -0.5f * float(gsize.y) + (float(j) + 0.5f) * float(gsubsize.y);
                for (int i = 0; i < Nx; i++) {
                    x = -0.5f * float(gsize.x) + (float(i) + 0.5f) * float(gsubsize.x);

                    vec3 subcenter = make_vec3(x, y, z);

                    vec3 subcenter_rot = rotatePoint(subcenter, make_SphericalCoord(0, rotation * M_PI / 180.f));

                    if (printmessages) {
                        cout << "Adding grid cell #" << count << " with center " << subcenter_rot.x + center.x << "," << subcenter_rot.y + center.y << "," << subcenter.z + center.z << " and size " << gsubsize.x << " x " << gsubsize.y << " x "
                             << gsubsize.z << endl;
                    }

                    addGridCell(subcenter + center, center, gsubsize, gsize, rotation * M_PI / 180.f, make_int3(i, j, k), make_int3(Nx, Ny, Nz));

                    count++;
                }
            }
        }
    }

    if (printmessages) {

        cout << "done." << endl;

        cout << "Successfully read " << getScanCount() << " scan(s), which contain " << total_hits << " total hit points." << endl;
    }
}

size_t LiDARcloud::loadASCIIFile(uint scanID, const std::string &ASCII_data_file) {

    // Resolve file path using project-based resolution
    std::filesystem::path resolved_path = resolveProjectFile(ASCII_data_file);
    std::string resolved_filename = resolved_path.string();

    ifstream datafile(resolved_filename); // open the file

    if (!datafile.is_open()) { // check that file exists
        helios_runtime_error("ERROR (LiDARcloud::loadASCIIFile): ASCII data file '" + ASCII_data_file + "' does not exist.");
    }

    if (scanID >= getScanCount()) {
        helios_runtime_error("ERROR (LiDARcloud::loadASCIIFile): Scan #" + std::to_string(scanID) + " does not exist.");
    }
    const ScanMetadata &scan_data = scans.at(scanID);

    vec3 temp_xyz;
    float temp_zenith, temp_azimuth;
    RGBcolor temp_rgb;
    float temp_row, temp_column;
    double temp_data;
    std::map<std::string, double> data;

    std::size_t hit_count = 0;
    while (datafile.good()) { // loop through file to read scan data

        temp_xyz = make_vec3(-9999, -9999, -9999);
        temp_rgb = make_RGBcolor(1, 0, 0); // default color: red
        temp_row = -1;
        temp_column = -1;
        temp_zenith = -9999;
        temp_azimuth = -9999;

        // Skip comment/header lines (e.g. the '#'-prefixed column-name header written by
        // exportPointCloud). The loader keys columns off the XML ASCII_format, so the header is
        // informational and is simply discarded.
        datafile >> std::ws;
        if (datafile.peek() == '#') {
            std::string discard;
            std::getline(datafile, discard);
            continue;
        }
        if (!datafile.good()) { // EOF reached after trailing whitespace
            break;
        }

        for (uint i = 0; i < scan_data.columnFormat.size(); i++) {
            if (scan_data.columnFormat.at(i) == "row") {
                datafile >> temp_row;
            } else if (scan_data.columnFormat.at(i) == "column") {
                datafile >> temp_column;
            } else if (scan_data.columnFormat.at(i) == "zenith") {
                datafile >> temp_zenith;
                temp_zenith = deg2rad(temp_zenith);
            } else if (scan_data.columnFormat.at(i) == "azimuth") {
                datafile >> temp_azimuth;
                temp_azimuth = deg2rad(temp_azimuth);
            } else if (scan_data.columnFormat.at(i) == "zenith_rad") {
                datafile >> temp_zenith;
            } else if (scan_data.columnFormat.at(i) == "azimuth_rad") {
                datafile >> temp_azimuth;
            } else if (scan_data.columnFormat.at(i) == "x") {
                datafile >> temp_xyz.x;
            } else if (scan_data.columnFormat.at(i) == "y") {
                datafile >> temp_xyz.y;
            } else if (scan_data.columnFormat.at(i) == "z") {
                datafile >> temp_xyz.z;
            } else if (scan_data.columnFormat.at(i) == "r") {
                datafile >> temp_rgb.r;
            } else if (scan_data.columnFormat.at(i) == "g") {
                datafile >> temp_rgb.g;
            } else if (scan_data.columnFormat.at(i) == "b") {
                datafile >> temp_rgb.b;
            } else if (scan_data.columnFormat.at(i) == "r255") {
                datafile >> temp_rgb.r;
                temp_rgb.r /= 255.f;
            } else if (scan_data.columnFormat.at(i) == "g255") {
                datafile >> temp_rgb.g;
                temp_rgb.g /= 255.f;
            } else if (scan_data.columnFormat.at(i) == "b255") {
                datafile >> temp_rgb.b;
                temp_rgb.b /= 255.f;
            } else { // assume that rest is data
                datafile >> temp_data;
                data[scan_data.columnFormat.at(i)] = temp_data;
            }
        }

        if (!datafile.good()) { // if the whole line was not read successfully, stop
            if (hit_count == 0) {
                std::cerr << "WARNING: Something is likely wrong with the data file " << ASCII_data_file << ". Check that the format is consistent with that specified in the XML metadata file." << std::endl;
            }
            break;
        }

        // -- Checks to make sure everything was specified correctly -- //

        // hit point
        if (temp_xyz.x == -9999) {
            helios_runtime_error("ERROR (LiDARcloud::loadASCIIFile): x-coordinate not specified for hit point #" + std::to_string(hit_count) + " of scan #" + std::to_string(scanID));
        } else if (temp_xyz.y == -9999) {
            helios_runtime_error("ERROR (LiDARcloud::loadASCIIFile): y-coordinate not specified for hit point #" + std::to_string(hit_count) + " of scan #" + std::to_string(scanID));
        } else if (temp_xyz.z == -9999) {
            helios_runtime_error("ERROR (LiDARcloud::loadASCIIFile): z-coordinate not specified for hit point #" + std::to_string(hit_count) + " of scan #" + std::to_string(scanID));
        }

        // direction
        SphericalCoord temp_direction(1.f, 0.5f * M_PI - temp_zenith, temp_azimuth);
        if (temp_direction.elevation == -9999 || temp_direction.azimuth == -9999) {
            temp_direction = cart2sphere(temp_xyz - scan_data.origin);
        }

        // Carry the native scan-grid row/column indices onto the hit as hit data so that
        // the row/column-based gap filler (gapfillMisses) can reconstruct miss directions
        // without timestamps. Only stored when the columns were actually present and read.
        if (temp_row >= 0) {
            data["row"] = temp_row;
        }
        if (temp_column >= 0) {
            data["column"] = temp_column;
        }

        // add hit point to the scan
        addHitPoint(scanID, temp_xyz, temp_direction, temp_rgb, data);

        hit_count++;
    }

    datafile.close();

    return hit_count;
}

void LiDARcloud::exportTriangleNormals(const char *filename) {

    ensureOutputDirectoryExists(filename);

    ofstream file;

    file.open(filename);

    if (!file.is_open()) {
        helios_runtime_error("ERROR (LiDARcloud::exportTriangleNormals): Could not open file '" + std::string(filename) + "' for writing.");
    }

    for (std::size_t t = 0; t < triangles.size(); t++) {

        Triangulation tri = triangles.at(t);

        vec3 v0 = tri.vertex0;
        vec3 v1 = tri.vertex1;
        vec3 v2 = tri.vertex2;

        vec3 normal = cross(v1 - v0, v2 - v0);
        normal.normalize();

        file << normal.x << " " << normal.y << " " << normal.z << std::endl;
    }

    file.close();
}

void LiDARcloud::exportTriangleNormals(const char *filename, int gridcell) {

    ensureOutputDirectoryExists(filename);

    ofstream file;

    file.open(filename);

    if (!file.is_open()) {
        helios_runtime_error("ERROR (LiDARcloud::exportTriangleNormals): Could not open file '" + std::string(filename) + "' for writing.");
    }

    for (std::size_t t = 0; t < triangles.size(); t++) {

        Triangulation tri = triangles.at(t);

        if (tri.gridcell == gridcell) {

            vec3 v0 = tri.vertex0;
            vec3 v1 = tri.vertex1;
            vec3 v2 = tri.vertex2;

            vec3 normal = cross(v1 - v0, v2 - v0);
            normal.normalize();

            file << normal.x << " " << normal.y << " " << normal.z << std::endl;
        }
    }

    file.close();
}

void LiDARcloud::exportTriangleAreas(const char *filename) {

    ensureOutputDirectoryExists(filename);

    ofstream file;

    file.open(filename);

    if (!file.is_open()) {
        helios_runtime_error("ERROR (LiDARcloud::exportTriangleAreas): Could not open file '" + std::string(filename) + "' for writing.");
    }

    for (std::size_t t = 0; t < triangles.size(); t++) {

        Triangulation tri = triangles.at(t);

        file << tri.area << std::endl;
    }

    file.close();
}

void LiDARcloud::exportTriangleAreas(const char *filename, int gridcell) {

    ensureOutputDirectoryExists(filename);

    ofstream file;

    file.open(filename);

    if (!file.is_open()) {
        helios_runtime_error("ERROR (LiDARcloud::exportTriangleAreas): Could not open file '" + std::string(filename) + "' for writing.");
    }

    for (std::size_t t = 0; t < triangles.size(); t++) {

        Triangulation tri = triangles.at(t);

        if (tri.gridcell == gridcell) {

            file << tri.area << std::endl;
        }
    }

    file.close();
}

void LiDARcloud::exportTriangleInclinationDistribution(const char *filename, uint Nbins) {

    ensureOutputDirectoryExists(filename);

    std::vector<std::vector<float>> inclinations(getGridCellCount());
    for (int i = 0; i < getGridCellCount(); i++) {
        inclinations.at(i).resize(Nbins);
    }
    std::vector<float> cell_area(inclinations.size(), 0);

    float db = 0.5f * M_PI / float(Nbins); // bin width

    for (std::size_t t = 0; t < triangles.size(); t++) {

        Triangulation tri = triangles.at(t);

        int cell = tri.gridcell;

        if (cell < 0) {
            continue;
        }

        vec3 v0 = tri.vertex0;
        vec3 v1 = tri.vertex1;
        vec3 v2 = tri.vertex2;

        vec3 normal = cross(v1 - v0, v2 - v0);
        normal.normalize();

        float angle = acos_safe(fabs(normal.z));

        float area = tri.area;

        uint bin = floor(angle / db);
        if (bin >= Nbins) {
            bin = Nbins - 1;
        }

        inclinations.at(cell).at(bin) += area;

        cell_area.at(cell) += area;
    }

    ofstream file;

    file.open(filename);

    if (!file.is_open()) {
        helios_runtime_error("ERROR (LiDARcloud::exportTriangleInclinationDistribution): Could not open file '" + std::string(filename) + "' for writing.");
    }

    for (int cell = 0; cell < getGridCellCount(); cell++) {
        for (int bin = 0; bin < Nbins; bin++) {
            file << inclinations.at(cell).at(bin) / cell_area.at(cell) << " ";
        }
        file << std::endl;
    }

    file.close();
}

void LiDARcloud::exportTriangleAzimuthDistribution(const char *filename, uint Nbins) {

    ensureOutputDirectoryExists(filename);

    std::vector<std::vector<float>> azimuths(getGridCellCount());
    for (int i = 0; i < getGridCellCount(); i++) {
        azimuths.at(i).resize(Nbins);
    }
    std::vector<float> cell_area(azimuths.size(), 0);

    float db = 2 * M_PI / float(Nbins); // bin width

    for (std::size_t t = 0; t < triangles.size(); t++) {

        Triangulation tri = triangles.at(t);

        int cell = tri.gridcell;

        if (cell < 0) {
            continue;
        }

        vec3 v0 = tri.vertex0;
        vec3 v1 = tri.vertex1;
        vec3 v2 = tri.vertex2;

        vec3 normal = cross(v1 - v0, v2 - v0);
        normal.normalize();
        SphericalCoord n_sph = cart2sphere(normal);

        float azimuth = n_sph.azimuth;

        if (normal.z < 0) {
            azimuth = azimuth + M_PI;
            if (azimuth > M_PI * 2) {
                azimuth = azimuth - M_PI * 2;
            }
        }


        float area = tri.area;

        uint bin = floor(azimuth / db);
        if (bin >= Nbins) {
            bin = Nbins - 1;
        }

        azimuths.at(cell).at(bin) += area;
        cell_area.at(cell) += area;
    }

    ofstream file;

    file.open(filename);

    if (!file.is_open()) {
        helios_runtime_error("ERROR (LiDARcloud::exportTriangleAzimuthDistribution): Could not open file '" + std::string(filename) + "' for writing.");
    }

    for (int cell = 0; cell < getGridCellCount(); cell++) {
        for (int bin = 0; bin < Nbins; bin++) {
            file << azimuths.at(cell).at(bin) / cell_area.at(cell) << " ";
        }
        file << std::endl;
    }

    file.close();
}

void LiDARcloud::exportLeafAreas(const char *filename) {

    ensureOutputDirectoryExists(filename);

    ofstream file;

    file.open(filename);

    if (!file.is_open()) {
        helios_runtime_error("ERROR (LiDARcloud::exportLeafAreas): Could not open file '" + std::string(filename) + "' for writing.");
    }

    for (uint i = 0; i < getGridCellCount(); i++) {

        file << getCellLeafArea(i) << std::endl;
    }

    file.close();
}

void LiDARcloud::exportLeafAreaDensities(const char *filename) {

    ensureOutputDirectoryExists(filename);

    ofstream file;

    file.open(filename);

    if (!file.is_open()) {
        helios_runtime_error("ERROR (LiDARcloud::exportLeafAreaDensities): Could not open file '" + std::string(filename) + "' for writing.");
    }

    for (uint i = 0; i < getGridCellCount(); i++) {

        file << getCellLeafAreaDensity(i) << std::endl;
    }

    file.close();
}

void LiDARcloud::exportGtheta(const char *filename) {

    ensureOutputDirectoryExists(filename);

    ofstream file;

    file.open(filename);

    if (!file.is_open()) {
        helios_runtime_error("ERROR (LiDARcloud::exportGtheta): Could not open file '" + std::string(filename) + "' for writing.");
    }

    for (uint i = 0; i < getGridCellCount(); i++) {

        file << getCellGtheta(i) << std::endl;
    }

    file.close();
}

void LiDARcloud::exportLeafAreaUncertainty(const char *filename) {

    ensureOutputDirectoryExists(filename);

    ofstream file;

    file.open(filename);

    if (!file.is_open()) {
        helios_runtime_error("ERROR (LiDARcloud::exportLeafAreaUncertainty): Could not open file '" + std::string(filename) + "' for writing.");
    }

    // SAMPLING uncertainty of the leaf-area inversion (Pimont et al. 2018), conditional on the
    // beams that entered each voxel; does NOT capture occlusion/coverage bias. Undefined values
    // are written as the sentinel -1.
    file << "# cell_index leaf_area beam_count I_rdi LAD_std_error ci_valid" << std::endl;
    for (uint i = 0; i < getGridCellCount(); i++) {
        const float lad_variance = getCellLADVariance(i);
        const float lad_std_error = (lad_variance >= 0.f) ? std::sqrt(lad_variance) : -1.f;
        file << i << " " << getCellLeafArea(i) << " " << getCellBeamCount(i) << " " << grid_cells.at(i).I_rdi << " " << lad_std_error << " " << (grid_cells.at(i).ci_valid ? 1 : 0) << std::endl;
    }

    file.close();
}

void LiDARcloud::exportPointCloud(const char *filename, bool write_header) {

    if (getScanCount() == 1) {
        exportPointCloud(filename, 0, write_header);
    } else {

        for (int i = 0; i < getScanCount(); i++) {

            std::string filename_a = filename;
            char scan[20];
            snprintf(scan, sizeof(scan), "%d", i);

            size_t dotindex = filename_a.find_last_of(".");
            if (dotindex == filename_a.size() - 1 || filename_a.size() - 1 - dotindex > 4) { // no file extension was provided
                filename_a = filename_a + "_" + scan;
            } else { // has file extension
                std::string ext = filename_a.substr(dotindex, filename_a.size() - 1);
                filename_a = filename_a.substr(0, dotindex) + "_" + scan + ext;
            }

            exportPointCloud(filename_a.c_str(), i, write_header);
        }
    }
}


void LiDARcloud::exportPointCloud(const char *filename, uint scanID, bool write_header) {

    ensureOutputDirectoryExists(filename);

    if (scanID > getScanCount()) {
        std::cerr << "ERROR (LiDARcloud::exportPointCloud): Cannot export scan " << scanID << " because this scan does not exist." << std::endl;
        throw 1;
    }

    ofstream file;

    file.open(filename);

    if (!file.is_open()) {
        helios_runtime_error("ERROR (LiDARcloud::exportPointCloud): Could not open file '" + std::string(filename) + "' for writing.");
    }

    std::vector<std::string> hit_data;
    for (int r = 0; r < getHitCount(); r++) {
        std::map<std::string, double> data = hits.at(r).data;
        for (std::map<std::string, double>::iterator iter = data.begin(); iter != data.end(); ++iter) {
            std::vector<std::string>::iterator it = find(hit_data.begin(), hit_data.end(), iter->first);
            if (it == hit_data.end()) {
                hit_data.push_back(iter->first);
            }
        }
    }

    std::vector<std::string> ASCII_format = getScanColumnFormat(scanID);

    if (ASCII_format.size() == 0) {
        ASCII_format.push_back("x");
        ASCII_format.push_back("y");
        ASCII_format.push_back("z");
    }

    // Write a leading comment-line header listing the column field names. This follows the
    // conventional '#'-prefixed ASCII point-cloud header (e.g. accepted by CloudCompare). The
    // tokens are the resolved ASCII_format columns, so the header always matches the data
    // columns, including any user-defined scalar fields. loadASCIIFile() skips '#' lines.
    if (write_header) {
        file << "#";
        for (const std::string &col: ASCII_format) {
            file << " " << col;
        }
        file << std::endl;
    }

    for (int r = 0; r < getHitCount(); r++) {

        if (getHitScanID(r) != scanID) {
            continue;
        }

        vec3 xyz = getHitXYZ(r);
        RGBcolor color = getHitColor(r);

        for (int c = 0; c < ASCII_format.size(); c++) {

            if (ASCII_format.at(c).compare("x") == 0) {
                file << xyz.x;
            } else if (ASCII_format.at(c).compare("y") == 0) {
                file << xyz.y;
            } else if (ASCII_format.at(c).compare("z") == 0) {
                file << xyz.z;
            } else if (ASCII_format.at(c).compare("r") == 0) {
                file << color.r;
            } else if (ASCII_format.at(c).compare("g") == 0) {
                file << color.g;
            } else if (ASCII_format.at(c).compare("b") == 0) {
                file << color.b;
            } else if (ASCII_format.at(c).compare("r255") == 0) {
                file << round(color.r * 255);
            } else if (ASCII_format.at(c).compare("g255") == 0) {
                file << round(color.g * 255);
            } else if (ASCII_format.at(c).compare("b255") == 0) {
                file << round(color.b * 255);
            } else if (ASCII_format.at(c).compare("zenith") == 0) {
                file << getHitRaydir(r).zenith;
            } else if (ASCII_format.at(c).compare("azimuth") == 0) {
                file << getHitRaydir(r).azimuth;
            } else if (hits.at(r).data.find(ASCII_format.at(c)) != hits.at(r).data.end()) { // hit scalar data
                file << getHitData(r, ASCII_format.at(c).c_str());
            } else {
                file << -9999;
            }

            if (c < ASCII_format.size() - 1) {
                file << " ";
            }
        }

        file << std::endl;
    }

    file.close();
}

void LiDARcloud::exportScans(const char *filename) {

    if (getScanCount() == 0) {
        helios_runtime_error("ERROR (LiDARcloud::exportScans): No scans to export.");
    }

    ensureOutputDirectoryExists(filename);

    std::filesystem::path xml_path(filename);
    std::filesystem::path parent_dir = xml_path.parent_path();
    std::string stem = xml_path.stem().string();

    pugi::xml_document xmldoc;
    pugi::xml_node helios_node = xmldoc.append_child("helios");

    for (uint i = 0; i < getScanCount(); i++) {

        std::string xyz_basename = stem + "_" + std::to_string(i) + ".xyz";
        std::filesystem::path xyz_path = parent_dir / xyz_basename;
        std::string xyz_path_str = xyz_path.string();

        // Write the ASCII point cloud for this scan using the existing exporter
        exportPointCloud(xyz_path_str.c_str(), i);

        // Build the <scan> entry
        pugi::xml_node scan_node = helios_node.append_child("scan");

        vec3 origin = getScanOrigin(i);
        uint Ntheta = getScanSizeTheta(i);
        uint Nphi = getScanSizePhi(i);
        vec2 theta_range = getScanRangeTheta(i);
        vec2 phi_range = getScanRangePhi(i);
        float exit_diameter = getScanBeamExitDiameter(i);
        float beam_divergence = getScanBeamDivergence(i);
        float range_noise_stddev = getScanRangeNoiseStdDev(i);
        float angle_noise_stddev = getScanAngleNoiseStdDev(i);
        float scan_tilt_roll = getScanTiltRoll(i);
        float scan_tilt_pitch = getScanTiltPitch(i);
        std::vector<std::string> column_format = getScanColumnFormat(i);
        if (column_format.empty()) {
            column_format = {"x", "y", "z"};
        }

        auto append_text_child = [&](const char *tag, const std::string &text) {
            pugi::xml_node child = scan_node.append_child(tag);
            child.append_child(pugi::node_pcdata).set_value(text.c_str());
        };

        std::ostringstream origin_ss;
        origin_ss << origin.x << " " << origin.y << " " << origin.z;
        append_text_child("origin", origin_ss.str());

        // Spinning multibeam scans store the per-channel zenith angles rather than a uniform theta range, so write the pattern
        // and channel elevation angles (in degrees above the horizon) to round-trip the scan geometry on re-import.
        if (getScanPattern(i) == SCAN_PATTERN_SPINNING_MULTIBEAM) {
            append_text_child("scanPattern", "spinning_multibeam");
            std::ostringstream elev_ss;
            const std::vector<float> beam_zenith_angles = getScanBeamZenithAngles(i);
            for (size_t c = 0; c < beam_zenith_angles.size(); c++) {
                if (c > 0) {
                    elev_ss << " ";
                }
                elev_ss << (0.5f * float(M_PI) - beam_zenith_angles[c]) * 180.f / float(M_PI); // zenith -> elevation (deg)
            }
            append_text_child("beamElevationAngles", elev_ss.str());
            append_text_child("Nphi", std::to_string(Nphi));
        } else {
            std::ostringstream size_ss;
            size_ss << Ntheta << " " << Nphi;
            append_text_child("size", size_ss.str());
        }

        append_text_child("thetaMin", std::to_string(theta_range.x * 180.f / float(M_PI)));
        append_text_child("thetaMax", std::to_string(theta_range.y * 180.f / float(M_PI)));
        append_text_child("phiMin", std::to_string(phi_range.x * 180.f / float(M_PI)));
        append_text_child("phiMax", std::to_string(phi_range.y * 180.f / float(M_PI)));
        append_text_child("exitDiameter", std::to_string(exit_diameter));
        append_text_child("beamDivergence", std::to_string(beam_divergence));
        append_text_child("rangeNoiseStdDev", std::to_string(range_noise_stddev));
        append_text_child("angleNoiseStdDev", std::to_string(angle_noise_stddev));

        if (scan_tilt_roll != 0.f || scan_tilt_pitch != 0.f) {
            std::ostringstream tilt_ss;
            tilt_ss << scan_tilt_roll * 180.f / float(M_PI) << " " << scan_tilt_pitch * 180.f / float(M_PI);
            append_text_child("scanTilt", tilt_ss.str());
        }

        std::ostringstream format_ss;
        for (size_t c = 0; c < column_format.size(); c++) {
            if (c > 0) {
                format_ss << " ";
            }
            format_ss << column_format[c];
        }
        append_text_child("ASCII_format", format_ss.str());

        append_text_child("filename", xyz_basename);
    }

    if (!xmldoc.save_file(filename)) {
        helios_runtime_error("ERROR (LiDARcloud::exportScans): Could not write XML metadata file '" + std::string(filename) + "'.");
    }
}

void LiDARcloud::exportPointCloudPTX(const char *filename, uint scanID) {

    ensureOutputDirectoryExists(filename);

    if (scanID > getScanCount()) {
        std::cerr << "ERROR (LiDARcloud::exportPointCloudPTX): Cannot export scan " << scanID << " because this scan does not exist." << std::endl;
        throw 1;
    }

    ofstream file;

    file.open(filename);

    if (!file.is_open()) {
        helios_runtime_error("ERROR (LiDARcloud::exportPointCloudPTX): Could not open file '" + std::string(filename) + "' for writing.");
    }

    std::vector<std::string> ASCII_format = getScanColumnFormat(scanID);

    uint Nx = getScanSizeTheta(scanID);
    uint Ny = getScanSizePhi(scanID);

    file << Nx << std::endl;
    file << Ny << std::endl;
    file << "0 0 0" << std::endl;
    file << "1 0 0" << std::endl;
    file << "0 1 0" << std::endl;
    file << "0 0 1" << std::endl;
    file << "1 0 0 0" << std::endl;
    file << "0 1 0 0" << std::endl;
    file << "0 0 1 0" << std::endl;
    file << "0 0 0 1" << std::endl;

    std::vector<std::vector<vec4>> xyzi(Ny);
    for (int j = 0; j < Ny; j++) {
        xyzi.at(j).resize(Nx);
        for (int i = 0; i < Nx; i++) {
            xyzi.at(j).at(i) = make_vec4(0, 0, 0, 1);
        }
    }

    vec3 origin = getScanOrigin(scanID);

    for (int r = 0; r < getHitCount(); r++) {

        if (getHitScanID(r) != scanID) {
            continue;
        }

        SphericalCoord raydir = getHitRaydir(r);

        int2 row_column = scans.at(scanID).direction2rc(raydir);

        assert(row_column.x >= 0 && row_column.x < Nx && row_column.y >= 0 && row_column.y < Ny);

        vec3 xyz = getHitXYZ(r);

        if ((xyz - origin).magnitude() >= 1e4) {
            continue;
        }

        float intensity = 1.f;
        if (hits.at(r).data.find("intensity") != hits.at(r).data.end()) {
            intensity = getHitData(r, "intensity");
        }

        xyzi.at(row_column.y).at(row_column.x) = make_vec4(xyz.x, xyz.y, xyz.z, intensity);
    }

    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            file << xyzi.at(j).at(i).x << " " << xyzi.at(j).at(i).y << " " << xyzi.at(j).at(i).z << " " << xyzi.at(j).at(i).w << std::endl;
        }
    }

    file.close();
}

std::vector<uint> LiDARcloud::loadTreeQSM(helios::Context *context, const std::string &filename, uint radial_subdivisions, const std::string &texture_file) {
    return loadTreeQSM_impl(context, filename, radial_subdivisions, false, texture_file);
}

std::vector<uint> LiDARcloud::loadTreeQSMColormap(helios::Context *context, const std::string &filename, uint radial_subdivisions, const std::string &colormap_name) {
    return loadTreeQSM_impl(context, filename, radial_subdivisions, true, colormap_name);
}

std::vector<uint> LiDARcloud::loadTreeQSM_impl(helios::Context *context, const std::string &filename, uint radial_subdivisions, bool use_colormap, const std::string &colormap_or_texture) {

    if (printmessages) {
        if (use_colormap) {
            std::cout << "Loading TreeQSM cylinder file with colormap: " << filename << " (colormap: " << colormap_or_texture << ")" << std::endl;
        } else {
            std::cout << "Loading TreeQSM cylinder file: " << filename << std::endl;
        }
    }

    std::vector<uint> tube_UUIDs;

    // Open the file
    std::ifstream file(filename);
    if (!file.is_open()) {
        helios_runtime_error("ERROR (LiDARcloud::loadTreeQSM): Could not open TreeQSM file: " + filename);
    }

    // Structure to hold cylinder data
    struct CylinderData {
        float radius;
        float length;
        helios::vec3 start_point;
        helios::vec3 axis_direction;
        int parent;
        int extension;
        int branch_id;
        int branch_order;
        int position_in_branch;
        float mad;
        float surf_cov;
        int added;
        float unmod_radius;
    };

    std::vector<CylinderData> cylinders;
    std::string line;

    // Skip the header line
    if (!std::getline(file, line)) {
        helios_runtime_error("ERROR (LiDARcloud::loadTreeQSM): Empty file or failed to read header: " + filename);
    }

    // Read cylinder data
    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        std::istringstream iss(line);
        CylinderData cylinder;

        // Parse the tab-separated values
        if (!(iss >> cylinder.radius >> cylinder.length >> cylinder.start_point.x >> cylinder.start_point.y >> cylinder.start_point.z >> cylinder.axis_direction.x >> cylinder.axis_direction.y >> cylinder.axis_direction.z >> cylinder.parent >>
              cylinder.extension >> cylinder.branch_id >> cylinder.branch_order >> cylinder.position_in_branch >> cylinder.mad >> cylinder.surf_cov >> cylinder.added >> cylinder.unmod_radius)) {
            std::cerr << "WARNING (LiDARcloud::loadTreeQSM): Failed to parse line: " << line << std::endl;
            continue;
        }

        cylinders.push_back(cylinder);
    }

    file.close();

    if (printmessages) {
        std::cout << "Read " << cylinders.size() << " cylinders from TreeQSM file" << std::endl;
    }

    // Group cylinders by branch ID
    std::map<int, std::vector<CylinderData>> branches;
    for (const auto &cylinder: cylinders) {
        branches[cylinder.branch_id].push_back(cylinder);
    }

    if (printmessages) {
        std::cout << "Found " << branches.size() << " branches" << std::endl;
    }

    // Generate the colormap if needed
    std::vector<helios::RGBcolor> colormap;
    if (use_colormap) {
        uint num_colors = std::max(static_cast<uint>(branches.size()), 10u); // At least 10 colors for variety
        try {
            colormap = context->generateColormap(colormap_or_texture, num_colors);
        } catch (const std::exception &e) {
            helios_runtime_error("ERROR (LiDARcloud::loadTreeQSM): Invalid colormap name '" + colormap_or_texture + "'. Valid options are: hot, cool, rainbow, lava, parula, gray, green");
        }
    }

    // Create tube objects for each branch
    for (const auto &branch_pair: branches) {
        int branch_id = branch_pair.first;
        const auto &branch_cylinders = branch_pair.second;

        if (branch_cylinders.empty())
            continue;

        // Sort cylinders by position in branch
        std::vector<CylinderData> sorted_cylinders = branch_cylinders;
        std::sort(sorted_cylinders.begin(), sorted_cylinders.end(), [](const CylinderData &a, const CylinderData &b) { return a.position_in_branch < b.position_in_branch; });

        // Create nodes and radii for the tube
        std::vector<helios::vec3> nodes;
        std::vector<float> radii;

        for (const auto &cylinder: sorted_cylinders) {
            // Add start point
            nodes.push_back(cylinder.start_point);
            radii.push_back(cylinder.radius);

            // Add end point (start + length * axis_direction)
            helios::vec3 end_point = cylinder.start_point + cylinder.length * cylinder.axis_direction;
            nodes.push_back(end_point);
            radii.push_back(cylinder.radius);
        }

        // Remove duplicate consecutive nodes (where end of one cylinder = start of next)
        std::vector<helios::vec3> final_nodes;
        std::vector<float> final_radii;

        if (!nodes.empty()) {
            final_nodes.push_back(nodes[0]);
            final_radii.push_back(radii[0]);

            for (size_t i = 1; i < nodes.size(); i++) {
                // Check if this node is significantly different from the previous
                if ((nodes[i] - final_nodes.back()).magnitude() > 1e-6) {
                    final_nodes.push_back(nodes[i]);
                    final_radii.push_back(radii[i]);
                }
            }
        }

        // Create the tube object
        if (final_nodes.size() >= 2) {
            uint tube_UUID;

            if (use_colormap) {
                // Sample color from colormap based on branch ID
                // Use branch_id modulo colormap size for deterministic color selection
                int color_index = std::abs(branch_id) % colormap.size();
                helios::RGBcolor branch_color = colormap[color_index];

                // Create a color vector for all nodes in this branch
                std::vector<helios::RGBcolor> tube_colors(final_nodes.size(), branch_color);
                tube_UUID = context->addTubeObject(radial_subdivisions, final_nodes, final_radii, tube_colors);

                // if( printmessages ){
                //     std::cout << "Created tube for branch " << branch_id << " with " << final_nodes.size()
                //              << " nodes, branch_order " << sorted_cylinders[0].branch_order << ", color RGB("
                //              << branch_color.r << "," << branch_color.g << "," << branch_color.b << ")" << std::endl;
                // }
            } else {
                // Use texture file or solid color
                if (colormap_or_texture.empty()) {
                    // Create a color vector for the tube (red color for all nodes)
                    std::vector<helios::RGBcolor> tube_colors(final_nodes.size(), helios::RGB::red);
                    tube_UUID = context->addTubeObject(radial_subdivisions, final_nodes, final_radii, tube_colors);
                } else {
                    tube_UUID = context->addTubeObject(radial_subdivisions, final_nodes, final_radii, colormap_or_texture.c_str());
                }

                if (printmessages) {
                    std::cout << "Created tube for branch " << branch_id << " with " << final_nodes.size() << " nodes, branch_order " << sorted_cylinders[0].branch_order << std::endl;
                }
            }

            // Add object data for branch rank (branch_order)
            int branch_order = sorted_cylinders[0].branch_order; // All cylinders in a branch should have same order
            context->setObjectData(tube_UUID, "branch_order", branch_order);
            context->setObjectData(tube_UUID, "branch_id", branch_id);

            tube_UUIDs.push_back(tube_UUID);
        }
    }

    if (printmessages) {
        if (use_colormap) {
            std::cout << "Successfully created " << tube_UUIDs.size() << " tube objects from TreeQSM file using colormap" << std::endl;
        } else {
            std::cout << "Successfully created " << tube_UUIDs.size() << " tube objects from TreeQSM file" << std::endl;
        }
    }

    return tube_UUIDs;
}
