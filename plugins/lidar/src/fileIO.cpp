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

namespace {

    //! Build a Hamilton quaternion (qx,qy,qz,qw) from intrinsic Z-Y-X (yaw-pitch-roll) Tait-Bryan angles in radians.
    /** Mirror of the canonical helper in LiDAR.cpp (same pinned convention: Hamilton body->world, q = qz(yaw)*qy(pitch)*qx(roll)).
     *  Duplicated here only so the XML trajectory loader can convert Euler-form rows without exposing LiDAR.cpp internals. */
    helios::vec4 trajQuatFromRPY(float roll, float pitch, float yaw) {
        const float cr = std::cos(roll * 0.5f), sr = std::sin(roll * 0.5f);
        const float cp = std::cos(pitch * 0.5f), sp = std::sin(pitch * 0.5f);
        const float cy = std::cos(yaw * 0.5f), sy = std::sin(yaw * 0.5f);
        helios::vec4 q;
        q.w = cr * cp * cy + sr * sp * sy;
        q.x = sr * cp * cy - cr * sp * sy;
        q.y = cr * sp * cy + sr * cp * sy;
        q.z = cr * cp * sy - sr * sp * cy;
        return q;
    }

    //! Convert per-channel zenith angles (Helios convention, radians) back to elevation above the horizon (radians).
    /** The XML loader parses <beamElevationAngles> directly into zenith angles; addScanSpinning expects elevation, so
     *  this inverts the conversion (elevation = pi/2 - zenith) without losing precision. */
    std::vector<float> beamElevationAnglesRad(const std::vector<float> &beamZenithAngles) {
        std::vector<float> elevation;
        elevation.reserve(beamZenithAngles.size());
        for (float zenith: beamZenithAngles) {
            elevation.push_back(0.5f * float(M_PI) - zenith);
        }
        return elevation;
    }

    //! Parse a stream of whitespace-separated trajectory rows into (times, positions, quaternions).
    /**
     * Each non-empty, non-comment ('#') row is either 8 numbers (t x y z qx qy qz qw, quaternion form) or 7 numbers
     * (t x y z roll pitch yaw, Euler form in DEGREES, intrinsic Z-Y-X). The row width is auto-detected per row but must
     * be consistent across the whole stream. Euler rows are converted to Hamilton body->world quaternions. The caller
     * supplies a context string (e.g. "scan #0") for error messages. Fails fast on ragged/short rows.
     */
    void parseTrajectoryStream(std::istream &stream, const std::string &context, std::vector<double> &traj_t, std::vector<helios::vec3> &traj_pos, std::vector<helios::vec4> &traj_quat) {

        traj_t.clear();
        traj_pos.clear();
        traj_quat.clear();

        int row_width = 0; // 0 = undetermined, otherwise 7 or 8
        std::string line;
        size_t line_number = 0;
        while (std::getline(stream, line)) {
            line_number++;
            // Strip a trailing comment and skip blank/comment lines.
            const size_t hash = line.find('#');
            if (hash != std::string::npos) {
                line = line.substr(0, hash);
            }
            std::istringstream ls(line);
            std::vector<double> vals;
            double v;
            while (ls >> v) {
                vals.push_back(v);
            }
            if (vals.empty()) {
                continue; // blank or comment-only line
            }
            if (row_width == 0) {
                if (vals.size() != 7 && vals.size() != 8) {
                    helios_runtime_error("ERROR (LiDARcloud::loadXML): trajectory for " + context + " has a row with " + std::to_string(vals.size()) +
                                         " numbers on line " + std::to_string(line_number) + ". Each row must have 8 numbers (t x y z qx qy qz qw) or 7 numbers (t x y z roll pitch yaw, degrees).");
                }
                row_width = int(vals.size());
            } else if (int(vals.size()) != row_width) {
                helios_runtime_error("ERROR (LiDARcloud::loadXML): trajectory for " + context + " is ragged: line " + std::to_string(line_number) + " has " + std::to_string(vals.size()) + " numbers but earlier rows have " +
                                     std::to_string(row_width) + ". All trajectory rows must have the same number of columns.");
            }

            traj_t.push_back(vals[0]);
            traj_pos.push_back(helios::make_vec3(float(vals[1]), float(vals[2]), float(vals[3])));
            if (row_width == 8) {
                traj_quat.push_back(helios::make_vec4(float(vals[4]), float(vals[5]), float(vals[6]), float(vals[7])));
            } else {
                // Euler degrees -> radians -> Hamilton body->world quaternion (intrinsic Z-Y-X).
                const float roll = float(vals[4]) * float(M_PI) / 180.f;
                const float pitch = float(vals[5]) * float(M_PI) / 180.f;
                const float yaw = float(vals[6]) * float(M_PI) / 180.f;
                traj_quat.push_back(trajQuatFromRPY(roll, pitch, yaw));
            }
        }

        if (traj_t.empty()) {
            helios_runtime_error("ERROR (LiDARcloud::loadXML): trajectory for " + context + " contains no pose rows.");
        }
    }

} // namespace

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
            // A scan must define its beam emission origin in one of two ways: a single static <origin> tag (a fixed
            // scanner), or per-point origin columns (origin_x/origin_y/origin_z) in the ASCII data file (a moving
            // platform, where each pulse has its own origin). Exactly which one is present is validated below, once the
            // ASCII column format has been parsed. When only per-point origins are present the static origin is unused
            // and defaults to (0,0,0).
            const char *origin_str = s.child_value("origin");
            const bool has_static_origin = (strlen(origin_str) != 0);
            vec3 origin = has_static_origin ? string2vec3(origin_str) : make_vec3(0, 0, 0); // note: pugi loads xml as characters; split into 3 floats

            // ----- scan pattern ------//
            // Optional. Default is a raster scan (uniform angular grid). 'spinning_multibeam' models a rotating multi-channel
            // sensor (e.g. Velodyne/Ouster/Hesai) whose channels are specified via <beamElevationAngles>.
            std::string scan_pattern_str = deblank(s.child_value("scanPattern"));
            if (scan_pattern_str.empty()) {
                scan_pattern_str = deblank(s.child_value("scanpattern"));
            }
            std::transform(scan_pattern_str.begin(), scan_pattern_str.end(), scan_pattern_str.begin(), [](unsigned char ch) { return std::tolower(ch); });
            const bool spinning_multibeam = (scan_pattern_str == "spinning_multibeam" || scan_pattern_str == "spinning-multibeam" || scan_pattern_str == "spinningmultibeam");
            // 'risley' / 'risley_prism' models a rotating-Risley-prism (Livox-style rosette) scanner whose prisms are specified
            // via <prism> children.
            const bool risley = (scan_pattern_str == "risley" || scan_pattern_str == "risley_prism" || scan_pattern_str == "risley-prism" || scan_pattern_str == "risleyprism");
            if (!scan_pattern_str.empty() && scan_pattern_str != "raster" && !spinning_multibeam && !risley) {
                cerr << "failed.\n";
                helios_runtime_error("ERROR (LiDARcloud::loadXML): Unrecognized scanPattern '" + scan_pattern_str + "' for scan #" + std::to_string(scan_count) + ". Valid values are 'raster', 'spinning_multibeam', and 'risley'.");
            }

            // ----- Risley prism stack ------//
            // A risley scan refracts a single beam through a stack of rotating wedge prisms, each given as a <prism> child with
            // "wedgeAngle(deg) refractiveIndex rotorRate(Hz, signed) [phase(deg)]". The optional <refractiveIndexAir> sets the
            // surrounding medium index (default 1.0).
            std::vector<RisleyPrism> risley_prisms;
            double risley_refractive_index_air = 1.0;
            if (risley) {
                const char *air_str = s.child_value("refractiveIndexAir");
                if (strlen(air_str) == 0) {
                    air_str = s.child_value("refractiveindexair");
                }
                if (strlen(air_str) > 0) {
                    risley_refractive_index_air = atof(air_str);
                }
                for (pugi::xml_node prism_node = s.child("prism"); prism_node; prism_node = prism_node.next_sibling("prism")) {
                    // NOTE: do not deblank() here - the prism is several space-separated values, and deblank() strips ALL
                    // spaces (it is meant for single tokens), which would concatenate the fields. The istringstream handles
                    // the internal whitespace itself.
                    std::istringstream prism_stream(prism_node.child_value());
                    double wedge_deg, refr_index, rotor_hz;
                    if (!(prism_stream >> wedge_deg >> refr_index >> rotor_hz)) {
                        cerr << "failed.\n";
                        helios_runtime_error("ERROR (LiDARcloud::loadXML): A <prism> of risley scan #" + std::to_string(scan_count) +
                                             " must give at least 'wedgeAngle(deg) refractiveIndex rotorRate(Hz)' (an optional fourth value sets the initial phase in degrees).");
                    }
                    double phase_deg = 0.0;
                    prism_stream >> phase_deg; // optional; left at 0 if absent
                    risley_prisms.emplace_back(wedge_deg * M_PI / 180.0, refr_index, rotor_hz * 2.0 * M_PI, phase_deg * M_PI / 180.0);
                }
                if (risley_prisms.empty()) {
                    cerr << "failed.\n";
                    helios_runtime_error("ERROR (LiDARcloud::loadXML): A risley scan (#" + std::to_string(scan_count) + ") requires at least one <prism> child (a Livox-style sensor uses two counter-rotating prisms).");
                }
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
            // Raster scans require <size> = "Ntheta Nphi". Spinning multibeam scans set Ntheta from the number of channels
            // (beamElevationAngles) and derive the azimuth-step count Nphi internally from <azimuthStep>, <PRF>, and the
            // trajectory (see the dispatch below), so they do not specify a <size>.
            helios::int2 size = make_int2(0, 0);
            if (spinning_multibeam) {
                size.x = int(beamZenithAngles.size()); // Ntheta = number of laser channels
            } else if (risley) {
                // A risley scan stores a single direction per pulse: Ntheta=1 and Nphi=Npulses are derived in addScanRisley
                // from the PRF and the trajectory duration, so no <size> is specified.
                size.x = 1;
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

            // ----- scanAzimuthOffset (global scanner azimuth/heading offset, in degrees) ------//
            const char *scanAzimuth_str_uc = s.child_value("scanAzimuthOffset");
            const char *scanAzimuth_str_lc = s.child_value("scanazimuthoffset");

            float scanAzimuthOffset = 0.f;
            const char *scanAzimuth_str = (strlen(scanAzimuth_str_uc) > 0) ? scanAzimuth_str_uc : scanAzimuth_str_lc;
            if (strlen(scanAzimuth_str) > 0) {
                scanAzimuthOffset = atof(scanAzimuth_str) * float(M_PI) / 180.f; // degrees -> radians
            }

            // ----- returnMode (analytic-waveform return reporting: 'multi' (default) or 'single') ------//
            std::string returnMode_str = deblank(s.child_value("returnMode"));
            if (returnMode_str.empty()) {
                returnMode_str = deblank(s.child_value("returnmode"));
            }
            std::transform(returnMode_str.begin(), returnMode_str.end(), returnMode_str.begin(), [](unsigned char ch) { return std::tolower(ch); });
            ReturnMode returnMode = RETURN_MODE_MULTI;
            if (returnMode_str == "single") {
                returnMode = RETURN_MODE_SINGLE;
            } else if (!returnMode_str.empty() && returnMode_str != "multi") {
                cerr << "failed.\n";
                helios_runtime_error("ERROR (LiDARcloud::loadXML): Unrecognized returnMode '" + returnMode_str + "' for scan #" + std::to_string(scan_count) + ". Valid values are 'multi' and 'single'.");
            }

            // ----- singleReturnSelection ('strongest' (default), 'first', 'last', or 'strongest_plus_last' / 'dual') ------//
            std::string singleSel_str = deblank(s.child_value("singleReturnSelection"));
            if (singleSel_str.empty()) {
                singleSel_str = deblank(s.child_value("singlereturnselection"));
            }
            std::transform(singleSel_str.begin(), singleSel_str.end(), singleSel_str.begin(), [](unsigned char ch) { return std::tolower(ch); });
            SingleReturnSelection singleReturnSelection = SINGLE_RETURN_STRONGEST;
            if (singleSel_str == "first") {
                singleReturnSelection = SINGLE_RETURN_FIRST;
            } else if (singleSel_str == "last") {
                singleReturnSelection = SINGLE_RETURN_LAST;
            } else if (singleSel_str == "strongest_plus_last" || singleSel_str == "dual") {
                singleReturnSelection = SINGLE_RETURN_STRONGEST_PLUS_LAST;
            } else if (!singleSel_str.empty() && singleSel_str != "strongest") {
                cerr << "failed.\n";
                helios_runtime_error("ERROR (LiDARcloud::loadXML): Unrecognized singleReturnSelection '" + singleSel_str + "' for scan #" + std::to_string(scan_count) + ". Valid values are 'strongest', 'first', 'last', and 'strongest_plus_last' (alias 'dual').");
            }

            // ----- maxReturns (returns per pulse in single/limited mode: 1=single, 2=dual, N=N-return) ------//
            int maxReturns = 1;
            const char *maxReturns_str = s.child_value("maxReturns");
            if (strlen(maxReturns_str) == 0) {
                maxReturns_str = s.child_value("maxreturns");
            }
            if (strlen(maxReturns_str) > 0) {
                maxReturns = atoi(maxReturns_str);
                if (maxReturns < 1) {
                    cerr << "failed.\n";
                    helios_runtime_error("ERROR (LiDARcloud::loadXML): maxReturns must be at least 1, but '" + std::string(maxReturns_str) + "' was given for scan #" + std::to_string(scan_count) + ".");
                }
            }

            // ----- pulseWidth (range resolution, meters) or pulseDuration (seconds, converted via c*tau/2) ------//
            float pulseWidth = 0.f;
            const char *pulseWidth_str = s.child_value("pulseWidth");
            if (strlen(pulseWidth_str) == 0) {
                pulseWidth_str = s.child_value("pulsewidth");
            }
            const char *pulseDuration_str = s.child_value("pulseDuration");
            if (strlen(pulseDuration_str) == 0) {
                pulseDuration_str = s.child_value("pulseduration");
            }
            if (strlen(pulseWidth_str) > 0) {
                pulseWidth = fmax(0, atof(pulseWidth_str));
            } else if (strlen(pulseDuration_str) > 0) {
                // Round-trip range extent of a pulse of duration tau: R = c * tau / 2 (c = speed of light in m/s).
                pulseWidth = fmax(0.f, float(atof(pulseDuration_str)) * 299792458.f * 0.5f);
            }

            // ----- detectionThreshold (minimum return energy fraction) ------//
            float detectionThreshold = 0.f;
            const char *detThresh_str = s.child_value("detectionThreshold");
            if (strlen(detThresh_str) == 0) {
                detThresh_str = s.child_value("detectionthreshold");
            }
            if (strlen(detThresh_str) > 0) {
                detectionThreshold = fmax(0, atof(detThresh_str));
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

            // Require an emission origin. A trajectory-driven scan (a spinning scan, or any scan with a <trajectory> /
            // <trajectoryFile>) supplies the per-pulse origin from the trajectory, so it needs neither a static <origin>
            // nor per-point origin columns. A static scan must provide one of: a static <origin> tag, or per-point origin
            // columns (origin_x/origin_y/origin_z) in the ASCII data.
            const bool has_perpoint_origin = (std::find(column_format.begin(), column_format.end(), "origin_x") != column_format.end() && std::find(column_format.begin(), column_format.end(), "origin_y") != column_format.end() &&
                                              std::find(column_format.begin(), column_format.end(), "origin_z") != column_format.end());
            const bool has_trajectory_tag = (!s.child("trajectory").empty() || strlen(s.child_value("trajectoryFile")) != 0 || strlen(s.child_value("trajectoryfile")) != 0);
            if (!has_static_origin && !has_perpoint_origin && !spinning_multibeam && !risley && !has_trajectory_tag) {
                cerr << "failed.\n";
                helios_runtime_error("ERROR (LiDARcloud::loadXML): Scan #" + std::to_string(scan_count) +
                                     " has no beam origin. Specify either a static <origin> tag, or per-point origin columns (origin_x origin_y origin_z) in the <ASCII_format> and data file.");
            }

            // ----- physical-parameter (moving-platform / spinning) tags ------//
            // These describe a moving-platform or spinning-multibeam instrument using its physical parameters; when present,
            // Helios derives the internal sampling grid (see addScanSpinning / addScanMovingRaster). Their presence switches
            // the scan from the legacy static-grid path to the trajectory-driven path.

            // Azimuth resolution (degrees per firing step) -> radians. Primary azimuth control for a spinning sensor.
            float azimuthStep_rad = 0.f;
            const char *azStep_str = s.child_value("azimuthStep");
            if (strlen(azStep_str) == 0) {
                azStep_str = s.child_value("azimuthstep");
            }
            if (strlen(azStep_str) > 0) {
                azimuthStep_rad = float(atof(azStep_str)) * float(M_PI) / 180.f;
            }

            // Pulse repetition frequency (Hz).
            float PRF = 0.f;
            const char *prf_str = s.child_value("PRF");
            if (strlen(prf_str) == 0) {
                prf_str = s.child_value("pulseRate");
            }
            if (strlen(prf_str) == 0) {
                prf_str = s.child_value("pulserate");
            }
            if (strlen(prf_str) > 0) {
                PRF = float(atof(prf_str));
            }

            // Lever arm (sensor optical center in the body frame, meters) and boresight (roll pitch yaw, degrees).
            vec3 lever_arm = make_vec3(0, 0, 0);
            const char *lever_str = s.child_value("leverArm");
            if (strlen(lever_str) == 0) {
                lever_str = s.child_value("leverarm");
            }
            if (strlen(lever_str) > 0) {
                lever_arm = string2vec3(lever_str);
            }
            vec3 boresight_rpy = make_vec3(0, 0, 0);
            const char *boresight_str = s.child_value("boresight");
            if (strlen(boresight_str) > 0) {
                boresight_rpy = string2vec3(boresight_str) * float(M_PI) / 180.f; // degrees -> radians
            }

            // t0 (time of first pulse, seconds). Optional; defaults to the trajectory start.
            double t0 = 0.0;
            bool t0_specified = false;
            const char *t0_str = s.child_value("t0");
            if (strlen(t0_str) > 0) {
                t0 = atof(t0_str);
                t0_specified = true;
            }

            // Trajectory: either an inline <trajectory> block of <pose> children, or a referenced <trajectoryFile>.
            std::vector<double> traj_t;
            std::vector<vec3> traj_pos;
            std::vector<vec4> traj_quat;
            bool has_trajectory = false;
            pugi::xml_node traj_node = s.child("trajectory");
            const char *trajFile_str = s.child_value("trajectoryFile");
            if (strlen(trajFile_str) == 0) {
                trajFile_str = s.child_value("trajectoryfile");
            }
            if (!traj_node.empty()) {
                // Inline trajectory: concatenate the text of all <pose> children into a stream of rows.
                std::ostringstream poses;
                for (pugi::xml_node p = traj_node.child("pose"); p; p = p.next_sibling("pose")) {
                    poses << p.child_value() << "\n";
                }
                std::istringstream traj_stream(poses.str());
                parseTrajectoryStream(traj_stream, "scan #" + std::to_string(scan_count), traj_t, traj_pos, traj_quat);
                has_trajectory = true;
            } else if (strlen(trajFile_str) > 0) {
                // Referenced trajectory file: resolve relative to the XML file, then cwd.
                std::string resolved_traj;
                std::vector<std::string> candidates;
                candidates.emplace_back(trajFile_str);
                if (!xml_parent_dir.empty()) {
                    candidates.push_back((xml_parent_dir / trajFile_str).string());
                }
                candidates.push_back("input/" + std::string(trajFile_str));
                for (const std::string &candidate: candidates) {
                    ifstream tf(candidate);
                    if (tf.good()) {
                        resolved_traj = candidate;
                        break;
                    }
                }
                if (resolved_traj.empty()) {
                    cerr << "failed.\n";
                    helios_runtime_error("ERROR (LiDARcloud::loadXML): trajectory file `" + std::string(trajFile_str) + "' given for scan #" + std::to_string(scan_count) + " does not exist.");
                }
                ifstream tf(resolved_traj);
                parseTrajectoryStream(tf, "scan #" + std::to_string(scan_count), traj_t, traj_pos, traj_quat);
                has_trajectory = true;
            }

            // Dispatch by scan type. A spinning_multibeam scan is always set up from its physical parameters (channel
            // elevations + <azimuthStep> + <PRF> + a trajectory); there is no static azimuth-grid form. A stationary "spin
            // in place" capture is just a trajectory of two coincident poses whose time gap sets the acquisition duration.
            // A non-spinning scan with a trajectory is a moving raster; otherwise it is a static raster.
            const bool physical_moving_raster = !spinning_multibeam && !risley && has_trajectory && (PRF > 0.f);

            uint scanID;

            if (risley) {

                if (PRF <= 0.f) {
                    cerr << "failed.\n";
                    helios_runtime_error("ERROR (LiDARcloud::loadXML): risley scan #" + std::to_string(scan_count) + " requires a pulse repetition rate given as <PRF> (Hz).");
                }
                if (!has_trajectory) {
                    cerr << "failed.\n";
                    helios_runtime_error("ERROR (LiDARcloud::loadXML): risley scan #" + std::to_string(scan_count) +
                                         " requires a <trajectory> or <trajectoryFile>. For a stationary capture, give two coincident poses with the same position and orientation, separated in time by the acquisition duration.");
                }

                scanID = addScanRisley(risley_prisms, risley_refractive_index_air, PRF, traj_t, traj_pos, traj_quat, lever_arm, boresight_rpy, exitDiameter, beamDivergence, rangeNoiseStdDev, angleNoiseStdDev, column_format,
                                       t0_specified ? t0 : (traj_t.empty() ? 0.0 : traj_t.front()));

                // Apply analytic-waveform return parameters (not constructor arguments of the new entry points).
                setScanReturnMode(scanID, returnMode);
                setScanSingleReturnSelection(scanID, singleReturnSelection);
                setScanMaxReturns(scanID, maxReturns);
                setScanPulseWidth(scanID, pulseWidth);
                setScanDetectionThreshold(scanID, detectionThreshold);

            } else if (spinning_multibeam) {

                if (azimuthStep_rad <= 0.f) {
                    cerr << "failed.\n";
                    helios_runtime_error("ERROR (LiDARcloud::loadXML): spinning_multibeam scan #" + std::to_string(scan_count) + " requires an azimuth resolution given as <azimuthStep> (degrees per firing step).");
                }
                if (PRF <= 0.f) {
                    cerr << "failed.\n";
                    helios_runtime_error("ERROR (LiDARcloud::loadXML): spinning_multibeam scan #" + std::to_string(scan_count) + " requires a pulse repetition rate given as <PRF> (Hz).");
                }
                if (!has_trajectory) {
                    cerr << "failed.\n";
                    helios_runtime_error("ERROR (LiDARcloud::loadXML): spinning_multibeam scan #" + std::to_string(scan_count) +
                                         " requires a <trajectory> or <trajectoryFile>. For a stationary spin in place, give two coincident poses with the same position and orientation, separated in time by the acquisition duration.");
                }

                scanID = addScanSpinning(beamElevationAnglesRad(beamZenithAngles), azimuthStep_rad, PRF, traj_t, traj_pos, traj_quat, lever_arm, boresight_rpy, exitDiameter, beamDivergence, rangeNoiseStdDev, angleNoiseStdDev, column_format,
                                         t0_specified ? t0 : (traj_t.empty() ? 0.0 : traj_t.front()));

                // Apply analytic-waveform return parameters (not constructor arguments of the new entry points).
                setScanReturnMode(scanID, returnMode);
                setScanSingleReturnSelection(scanID, singleReturnSelection);
                setScanMaxReturns(scanID, maxReturns);
                setScanPulseWidth(scanID, pulseWidth);
                setScanDetectionThreshold(scanID, detectionThreshold);

            } else if (physical_moving_raster) {

                scanID = addScanMovingRaster(size.x, thetaMin, thetaMax, size.y, phiMin, phiMax, PRF, traj_t, traj_pos, traj_quat, lever_arm, boresight_rpy, exitDiameter, beamDivergence, rangeNoiseStdDev, angleNoiseStdDev, column_format,
                                             t0_specified ? t0 : (traj_t.empty() ? 0.0 : traj_t.front()));

                setScanReturnMode(scanID, returnMode);
                setScanSingleReturnSelection(scanID, singleReturnSelection);
                setScanMaxReturns(scanID, maxReturns);
                setScanPulseWidth(scanID, pulseWidth);
                setScanDetectionThreshold(scanID, detectionThreshold);

            } else {

                // Static raster scan (single fixed origin, uniform Ntheta x Nphi angular grid).
                ScanMetadata scan(origin, size.x, thetaMin, thetaMax, size.y, phiMin, phiMax, exitDiameter, beamDivergence, rangeNoiseStdDev, angleNoiseStdDev, column_format, scanTiltRoll, scanTiltPitch, scanAzimuthOffset);

                // Analytic-waveform return parameters (not constructor arguments)
                scan.returnMode = returnMode;
                scan.singleReturnSelection = singleReturnSelection;
                scan.maxReturns = maxReturns;
                scan.pulseWidth = pulseWidth;
                scan.detectionThreshold = detectionThreshold;

                addScan(scan);

                scanID = getScanCount() - 1;
            }

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

                scans.at(scanID).data_file = resolved_data_file; // set the data file for the registered scan

                // add hit points to scan if data file was given

                total_hits += loadASCIIFile(scanID, scans.at(scanID).data_file);

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

    // The union of per-hit scalar-data labels is exactly the set of columns in the columnar store. This
    // used to be computed by copying every hit's whole std::map and scanning it (an O(N*keys) pass that
    // was a second hidden export hot spot); with columnar storage it is just the label list. (This
    // collected list is presently unused downstream — the export columns come from getScanColumnFormat
    // below — but is kept for parity with the prior behavior.)
    std::vector<std::string> hit_data = hit_data_labels;

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

    // Resolve every output column ONCE to either a built-in field code or a scalar-data column slot, so
    // the per-hit inner loop does no string comparisons and no per-cell map/hash lookups. A scalar column
    // resolves to its slot index in the columnar store (>= 0), or COL_ABSENT if the label is not a known
    // column at all. Built-ins (x/y/z/r/g/b/...) get distinct negative codes.
    enum ColCode {
        COL_X = -1,
        COL_Y = -2,
        COL_Z = -3,
        COL_R = -4,
        COL_G = -5,
        COL_B = -6,
        COL_R255 = -7,
        COL_G255 = -8,
        COL_B255 = -9,
        COL_ZENITH = -10,
        COL_AZIMUTH = -11,
        COL_ABSENT = -12
    };
    std::vector<int> col_resolved(ASCII_format.size());
    for (size_t c = 0; c < ASCII_format.size(); c++) {
        const std::string &tok = ASCII_format[c];
        if (tok == "x") {
            col_resolved[c] = COL_X;
        } else if (tok == "y") {
            col_resolved[c] = COL_Y;
        } else if (tok == "z") {
            col_resolved[c] = COL_Z;
        } else if (tok == "r") {
            col_resolved[c] = COL_R;
        } else if (tok == "g") {
            col_resolved[c] = COL_G;
        } else if (tok == "b") {
            col_resolved[c] = COL_B;
        } else if (tok == "r255") {
            col_resolved[c] = COL_R255;
        } else if (tok == "g255") {
            col_resolved[c] = COL_G255;
        } else if (tok == "b255") {
            col_resolved[c] = COL_B255;
        } else if (tok == "zenith") {
            col_resolved[c] = COL_ZENITH;
        } else if (tok == "azimuth") {
            col_resolved[c] = COL_AZIMUTH;
        } else {
            int slot = getHitDataColumnIndex(tok.c_str());
            col_resolved[c] = (slot >= 0) ? slot : COL_ABSENT;
        }
    }

    for (int r = 0; r < getHitCount(); r++) {

        if (getHitScanID(r) != scanID) {
            continue;
        }

        vec3 xyz = getHitXYZ(r);
        RGBcolor color = getHitColor(r);

        for (int c = 0; c < ASCII_format.size(); c++) {

            const int code = col_resolved[c];
            if (code >= 0) { // scalar-data column slot
                if (hit_data_present[code][r] != char(0)) {
                    file << hit_data_columns[code][r];
                } else {
                    file << -9999;
                }
            } else if (code == COL_X) {
                file << xyz.x;
            } else if (code == COL_Y) {
                file << xyz.y;
            } else if (code == COL_Z) {
                file << xyz.z;
            } else if (code == COL_R) {
                file << color.r;
            } else if (code == COL_G) {
                file << color.g;
            } else if (code == COL_B) {
                file << color.b;
            } else if (code == COL_R255) {
                file << round(color.r * 255);
            } else if (code == COL_G255) {
                file << round(color.g * 255);
            } else if (code == COL_B255) {
                file << round(color.b * 255);
            } else if (code == COL_ZENITH) {
                file << getHitRaydir(r).zenith;
            } else if (code == COL_AZIMUTH) {
                file << getHitRaydir(r).azimuth;
            } else { // COL_ABSENT
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
        float scan_azimuth_offset = getScanAzimuthOffset(i);
        std::vector<std::string> column_format = getScanColumnFormat(i);
        if (column_format.empty()) {
            column_format = {"x", "y", "z"};
        }

        auto append_text_child = [&](const char *tag, const std::string &text) {
            pugi::xml_node child = scan_node.append_child(tag);
            child.append_child(pugi::node_pcdata).set_value(text.c_str());
        };

        // Write a static <origin> only when the scan has no per-point origin columns. A moving-platform scan records a
        // per-pulse origin (origin_x/origin_y/origin_z) in the data file, so a single static origin would be misleading;
        // it is omitted and the per-point origins are the source of truth (loadXML accepts either, but requires one).
        const bool has_perpoint_origin = (std::find(column_format.begin(), column_format.end(), "origin_x") != column_format.end() && std::find(column_format.begin(), column_format.end(), "origin_y") != column_format.end() &&
                                          std::find(column_format.begin(), column_format.end(), "origin_z") != column_format.end());
        if (!has_perpoint_origin) {
            std::ostringstream origin_ss;
            origin_ss << origin.x << " " << origin.y << " " << origin.z;
            append_text_child("origin", origin_ss.str());
        }

        const ScanMode scan_mode = getScanMode(i);
        const bool is_spinning = (scan_mode == SCAN_MODE_SPINNING);
        const bool is_moving_raster = (scan_mode == SCAN_MODE_MOVING_RASTER);
        const bool is_risley = (scan_mode == SCAN_MODE_RISLEY_PRISM);

        // Spinning multibeam scans store the per-channel zenith angles rather than a uniform theta range, so write the pattern
        // and channel elevation angles (in degrees above the horizon) to round-trip the scan geometry on re-import. A Risley
        // scan stores its rotating prism stack instead; both derive their grid internally and emit no <size>.
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
            // A spinning scan derives Nphi (and the azimuth sweep) internally from the azimuth resolution, PRF, and
            // trajectory, which are written below; no <size>/<Nphi>/<phiMax> is emitted.
        } else if (is_risley) {
            append_text_child("scanPattern", "risley");
            if (getScanRisleyRefractiveIndexAir(i) != 1.0) {
                append_text_child("refractiveIndexAir", std::to_string(getScanRisleyRefractiveIndexAir(i)));
            }
            // One <prism> child per rotating wedge: "wedgeAngle(deg) refractiveIndex rotorRate(Hz, signed) phase(deg)". The
            // grid (Ntheta=1, Nphi=Npulses) and circular FoV are derived internally from these on reload; no <size> is emitted.
            const std::vector<RisleyPrism> prisms = getScanRisleyPrisms(i);
            for (const RisleyPrism &prism : prisms) {
                std::ostringstream prism_ss;
                prism_ss << std::setprecision(12) << prism.wedge_angle * 180.0 / M_PI << " " << prism.refractive_index << " " << prism.rotor_rate / (2.0 * M_PI) << " " << prism.phase * 180.0 / M_PI;
                append_text_child("prism", prism_ss.str());
            }
        } else if (!is_moving_raster) {
            std::ostringstream size_ss;
            size_ss << Ntheta << " " << Nphi;
            append_text_child("size", size_ss.str());
        }

        // The angular bounds describe the per-frame fan. For a physical spinning or Risley scan they are derived and must NOT
        // be written (the geometry comes from the channels / prisms); for a moving raster the fan resolution is written via
        // <size> below.
        if (is_moving_raster) {
            std::ostringstream size_ss;
            size_ss << Ntheta << " " << Nphi;
            append_text_child("size", size_ss.str());
        }
        if (!is_spinning && !is_risley) {
            append_text_child("thetaMin", std::to_string(theta_range.x * 180.f / float(M_PI)));
            append_text_child("thetaMax", std::to_string(theta_range.y * 180.f / float(M_PI)));
            append_text_child("phiMin", std::to_string(phi_range.x * 180.f / float(M_PI)));
            append_text_child("phiMax", std::to_string(phi_range.y * 180.f / float(M_PI)));
        }

        // ----- physical-parameter (moving / spinning / Risley) round-trip ------//
        // Emit the physical instrument parameters and a trajectory sidecar CSV so a moving-platform, spinning, or Risley scan
        // reloads through the same physical-parameter path it was created with (addScanSpinning / addScanMovingRaster /
        // addScanRisley).
        if (is_spinning || is_moving_raster || is_risley) {
            const ScanMetadata &sm = scans.at(i);

            // PRF (Hz) from the per-pulse period.
            if (sm.pulse_period > 0.0) {
                append_text_child("PRF", std::to_string(1.0 / sm.pulse_period));
            }
            if (is_spinning && sm.steps_per_rev > 0) {
                // Azimuth resolution in degrees per step (360 / steps_per_rev).
                append_text_child("azimuthStep", std::to_string(360.0 / double(sm.steps_per_rev)));
            }
            if (sm.lever_arm.x != 0.f || sm.lever_arm.y != 0.f || sm.lever_arm.z != 0.f) {
                std::ostringstream lever_ss;
                lever_ss << sm.lever_arm.x << " " << sm.lever_arm.y << " " << sm.lever_arm.z;
                append_text_child("leverArm", lever_ss.str());
            }
            if (sm.boresight_rpy.x != 0.f || sm.boresight_rpy.y != 0.f || sm.boresight_rpy.z != 0.f) {
                std::ostringstream boresight_ss;
                boresight_ss << sm.boresight_rpy.x * 180.f / float(M_PI) << " " << sm.boresight_rpy.y * 180.f / float(M_PI) << " " << sm.boresight_rpy.z * 180.f / float(M_PI);
                append_text_child("boresight", boresight_ss.str());
            }
            if (sm.t0 != 0.0) {
                append_text_child("t0", std::to_string(sm.t0));
            }

            // Trajectory sidecar CSV: "<stem>_<i>_traj.csv" alongside the XML, columns t x y z qx qy qz qw.
            std::string traj_basename = stem + "_" + std::to_string(i) + "_traj.csv";
            std::filesystem::path traj_path = parent_dir / traj_basename;
            std::ofstream traj_file(traj_path.string());
            if (!traj_file.is_open()) {
                helios_runtime_error("ERROR (LiDARcloud::exportScans): Could not write trajectory file '" + traj_path.string() + "'.");
            }
            traj_file << "# t x y z qx qy qz qw\n";
            for (size_t k = 0; k < sm.traj_t.size(); k++) {
                const vec3 &p = sm.traj_pos.at(k);
                const vec4 &q = sm.traj_quat.at(k);
                traj_file << sm.traj_t.at(k) << " " << p.x << " " << p.y << " " << p.z << " " << q.x << " " << q.y << " " << q.z << " " << q.w << "\n";
            }
            traj_file.close();
            append_text_child("trajectoryFile", traj_basename);
        }

        append_text_child("exitDiameter", std::to_string(exit_diameter));
        append_text_child("beamDivergence", std::to_string(beam_divergence));
        append_text_child("rangeNoiseStdDev", std::to_string(range_noise_stddev));
        append_text_child("angleNoiseStdDev", std::to_string(angle_noise_stddev));

        // Analytic-waveform return parameters: write only non-default values to keep the metadata file uncluttered.
        if (getScanReturnMode(i) == RETURN_MODE_SINGLE) {
            append_text_child("returnMode", "single");
            SingleReturnSelection sel = getScanSingleReturnSelection(i);
            if (sel == SINGLE_RETURN_FIRST) {
                append_text_child("singleReturnSelection", "first");
            } else if (sel == SINGLE_RETURN_LAST) {
                append_text_child("singleReturnSelection", "last");
            } else if (sel == SINGLE_RETURN_STRONGEST_PLUS_LAST) {
                append_text_child("singleReturnSelection", "strongest_plus_last");
            } else {
                append_text_child("singleReturnSelection", "strongest");
            }
            if (getScanMaxReturns(i) != 1) {
                append_text_child("maxReturns", std::to_string(getScanMaxReturns(i)));
            }
        }
        if (getScanPulseWidth(i) > 0.f) {
            append_text_child("pulseWidth", std::to_string(getScanPulseWidth(i)));
        }
        if (getScanDetectionThreshold(i) > 0.f) {
            append_text_child("detectionThreshold", std::to_string(getScanDetectionThreshold(i)));
        }

        if (scan_tilt_roll != 0.f || scan_tilt_pitch != 0.f) {
            std::ostringstream tilt_ss;
            tilt_ss << scan_tilt_roll * 180.f / float(M_PI) << " " << scan_tilt_pitch * 180.f / float(M_PI);
            append_text_child("scanTilt", tilt_ss.str());
        }

        if (scan_azimuth_offset != 0.f) {
            append_text_child("scanAzimuthOffset", std::to_string(scan_azimuth_offset * 180.f / float(M_PI)));
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

    // The PTX format encodes a single scanner position/transform per scan, which cannot represent a scanner that moved
    // during acquisition. Fail fast rather than write a file with a misleading single origin. Use exportScans() /
    // exportPointCloud() (which write the per-pulse origin columns) for a moving-platform scan instead.
    if (scanID < scans.size() && scans.at(scanID).isMoving) {
        helios_runtime_error("ERROR (LiDARcloud::exportPointCloudPTX): the PTX format cannot represent a moving-platform scan (see addScanMoving), which has no single scanner origin. Use exportScans() or exportPointCloud() instead.");
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
        if (doesHitDataExist(r, "intensity")) {
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
