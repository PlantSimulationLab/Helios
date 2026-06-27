/** \file "LiDAR.h" Primary header file for LiDAR plug-in.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef LIDARPLUGIN
#define LIDARPLUGIN

#include "CollisionDetection.h"
#include "Context.h"
#include "Visualizer.h"

#include "triangulation_cdt.h"

#include <functional>

template<class datatype>
class HitTable {
public:
    uint Ntheta, Nphi;

    HitTable(void) {
        Ntheta = 0;
        Nphi = 0;
    }
    HitTable(const int nx, const int ny) {
        Ntheta = nx;
        Nphi = ny;
        data.resize(Nphi);
        for (int j = 0; j < Nphi; j++) {
            data.at(j).resize(Ntheta);
        }
    }
    HitTable(const int nx, const int ny, const datatype initval) {
        Ntheta = nx;
        Nphi = ny;
        data.resize(Nphi);
        for (int j = 0; j < Nphi; j++) {
            data.at(j).resize(Ntheta, initval);
        }
    }

    datatype get(const int i, const int j) const {
        if (i >= 0 && i < Ntheta && j >= 0 && j < Nphi) {
            return data.at(j).at(i);
        } else {
            helios::helios_runtime_error("ERROR (hit_map.get): get index out of range. Attempting to get index map at (" + std::to_string(i) + "," + std::to_string(j) + "), but size of scan is " + std::to_string(Ntheta) + " x " +
                                         std::to_string(Nphi) + ".");
        }
        return data.at(j).at(i); // unreachable; silences control-reaches-end-of-non-void-function warning
    }
    void set(const int i, const int j, const datatype value) {
        if (i >= 0 && i < Ntheta && j >= 0 && j < Nphi) {
            data.at(j).at(i) = value;
        } else {
            helios::helios_runtime_error("ERROR (hit_map.set): set index out of range. Attempting to set index map at (" + std::to_string(i) + "," + std::to_string(j) + "), but size of scan is " + std::to_string(Ntheta) + " x " +
                                         std::to_string(Nphi) + ".");
        }
    }
    void resize(const int nx, const int ny, const datatype initval) {
        Ntheta = nx;
        Nphi = ny;
        data.resize(Nphi);
        for (int j = 0; j < Nphi; j++) {
            data.at(j).resize(Ntheta);
            for (int i = 0; i < Ntheta; i++) {
                data.at(j).at(i) = initval;
            }
        }
    }

private:
    std::vector<std::vector<datatype>> data;
};

struct HitPoint {
    helios::vec3 position;
    helios::SphericalCoord direction;
    helios::int2 row_column;
    helios::RGBcolor color;
    int gridcell;
    int scanID;
    // NOTE: per-hit scalar data (intensity, distance, timestamp, custom labels, ...) is NOT stored
    // here. It lives in cloud-level columnar storage on LiDARcloud (hit_data_columns/hit_data_present),
    // indexed by the hit's position in the `hits` vector. Storing it as N independent
    // std::map<std::string,double> trees (one per hit) made bulk field extraction O(K*N) cache-cold
    // tree descents and dominated export time. See LiDARcloud::getHitData / getHitDataColumn.
    HitPoint(void) {
        position = helios::make_vec3(0, 0, 0);
        direction = helios::make_SphericalCoord(0, 0);
        row_column = helios::make_int2(0, 0);
        color = helios::RGB::red;
        gridcell = -2;
        scanID = -1;
    }
    HitPoint(int __scanID, helios::vec3 __position, helios::SphericalCoord __direction, helios::int2 __row_column, helios::RGBcolor __color) {
        scanID = __scanID;
        position = __position;
        direction = __direction;
        row_column = __row_column;
        color = __color;
        gridcell = -2;
    }
};

struct Triangulation {
    helios::vec3 vertex0, vertex1, vertex2;
    int ID0, ID1, ID2;
    int scanID;
    int gridcell;
    helios::RGBcolor color;
    float area;
    Triangulation(void) {
        vertex0 = helios::make_vec3(0, 0, 0);
        vertex1 = helios::make_vec3(0, 0, 0);
        vertex2 = helios::make_vec3(0, 0, 0);
        ID0 = 0;
        ID1 = 0;
        ID2 = 0;
        scanID = -1;
        gridcell = -2;
        color = helios::RGB::green;
        area = 0;
    }
    Triangulation(int __scanID, helios::vec3 __vertex0, helios::vec3 __vertex1, helios::vec3 __vertex2, int __ID0, int __ID1, int __ID2, helios::RGBcolor __color, int __gridcell) {
        scanID = __scanID;
        vertex0 = __vertex0;
        vertex1 = __vertex1;
        vertex2 = __vertex2;
        ID0 = __ID0;
        ID1 = __ID1;
        ID2 = __ID2;
        gridcell = __gridcell;
        color = __color;

        // calculate area
        helios::vec3 s0 = vertex1 - vertex0;
        helios::vec3 s1 = vertex2 - vertex0;
        helios::vec3 s2 = vertex2 - vertex1;

        float a = s0.magnitude();
        float b = s1.magnitude();
        float c = s2.magnitude();
        float s = 0.5f * (a + b + c);

        area = sqrt(s * (s - a) * (s - b) * (s - c));
    }
};

struct GridCell {
    helios::vec3 center;
    helios::vec3 global_anchor;
    helios::vec3 size;
    helios::vec3 global_size;
    helios::int3 global_ijk;
    helios::int3 global_count;
    //! Rotation of the grid cell about the z-axis in radians
    float azimuthal_rotation;
    float leaf_area;
    float Gtheta;
    float ground_height;
    float vegetation_height;
    float maximum_height;
    // ---- LAD inversion uncertainty (sufficient statistics + sampling variance) ----
    // These quantify the statistical SAMPLING uncertainty of the leaf-area inversion,
    // conditional on the beams that entered the voxel (Pimont et al. 2018, RSE 215:343-370).
    // They do NOT capture occlusion/coverage bias (voxels shadowed so beams never penetrate).
    int beam_count = -1; //!< N: number of beams that entered the voxel (-1 if not computed)
    float I_rdi = 0.f; //!< Relative density index I = 1 - P (fraction of beams intercepted)
    float zbar_e = 0.f; //!< Mean beam path length through the voxel [m]
    float var_path = 0.f; //!< Empirical variance of per-beam path lengths [m^2]
    float L1_element = -1.f; //!< Single-element optical depth L1 = lambda1*delta (-1 if element size unknown)
    float LAD_variance = -1.f; //!< Sampling variance of LAD [1/m]^2 (terms a+b); -1 if undefined
    bool ci_valid = false; //!< True if (L, L1, N) fall within the Pimont Table-3 CI-validity envelope
    GridCell(helios::vec3 __center, helios::vec3 __global_anchor, helios::vec3 __size, helios::vec3 __global_size, float __azimuthal_rotation, helios::int3 __global_ijk, helios::int3 __global_count) {
        center = __center;
        global_anchor = __global_anchor;
        size = __size;
        global_size = __global_size;
        azimuthal_rotation = __azimuthal_rotation;
        global_ijk = __global_ijk;
        global_count = __global_count;
        leaf_area = 0;
        Gtheta = 0;
        ground_height = 0;
        vegetation_height = 0;
        maximum_height = 0;
    }
};

//! Geometric pattern that maps a scan's (row,column) table indices to beam directions
/**
 * Determines how the (row, column) scan-table indices are converted to beam directions. The default \ref SCAN_PATTERN_RASTER
 * reproduces the original Helios behavior of a uniform angular grid in zenith and azimuth (panoramic terrestrial laser scanner).
 * \ref SCAN_PATTERN_SPINNING_MULTIBEAM models a rotating multi-channel sensor (e.g. Velodyne, Ouster, Hesai) where each table
 * row corresponds to a laser channel fired at a fixed - and generally non-uniformly spaced - zenith angle, while each column is
 * a uniform azimuth step. \ref SCAN_PATTERN_RISLEY_PRISM models a rotating-Risley-prism scanner (the optical mechanism used by
 * Livox rosette-pattern sensors such as the Mid-40/Mid-70/Avia): a single beam is refracted through a stack of continuously
 * rotating wedge prisms, tracing a non-repetitive rosette inside a circular field of view. It is non-separable - each column is
 * one pulse whose direction comes from the prism optics at that pulse's time, not from a row x column angular grid - so it is
 * stored as an Ntheta=1 (single-row) table and is always trajectory-driven. The raster and spinning patterns share the same
 * Ntheta x Nphi table storage, so all downstream processing (ray tracing, hit tables, leaf-area and leaf-angle inversion) is
 * common to both.
 */
enum ScanPattern {
    //! Uniform angular grid: zenith uniformly spaced over [thetaMin, thetaMax], azimuth over [phiMin, phiMax]
    SCAN_PATTERN_RASTER = 0,
    //! Rotating multi-channel sensor: each row is a laser channel at a fixed zenith angle (see \ref ScanMetadata::beamZenithAngles), each column a uniform azimuth step
    SCAN_PATTERN_SPINNING_MULTIBEAM = 1,
    //! Rotating-Risley-prism scanner (Livox-style rosette): each column is one pulse whose body-frame direction is computed by refracting the beam through the rotating prisms (see \ref ScanMetadata::risley_prisms) at that pulse's time. Stored as a single-row (Ntheta=1) table; always trajectory-driven.
    SCAN_PATTERN_RISLEY_PRISM = 2
};

//! High-level, introspective descriptor of how a scan was acquired
/**
 * Combines the geometric beam pattern (see \ref ScanPattern) and the platform motion into a single descriptor so a scan
 * can be queried ("how was this acquired?") without reverse-engineering it from \ref ScanMetadata::isMoving,
 * \ref ScanMetadata::scanPattern, and \ref ScanMetadata::phiMax. It is set by the scan-creation entry points and is
 * purely descriptive: the underlying mechanism is still driven by \ref ScanMetadata::scanPattern and
 * \ref ScanMetadata::isMoving.
 */
enum ScanMode {
    //! Uniform angular grid acquired from a single fixed origin (static terrestrial / tripod scan)
    SCAN_MODE_STATIC_RASTER = 0,
    //! Uniform angular grid acquired while the platform moves along a trajectory (mobile/airborne raster sensor)
    SCAN_MODE_MOVING_RASTER = 1,
    //! Continuously-spinning multi-channel sensor (always trajectory-driven; a stationary capture is two coincident poses)
    SCAN_MODE_SPINNING = 2,
    //! Rotating-Risley-prism rosette sensor (Livox-style; always trajectory-driven; a stationary capture is two coincident poses separated in time by the acquisition duration)
    SCAN_MODE_RISLEY_PRISM = 3
};

//! How the analytic-waveform synthetic scanner reports the returns detected along each pulse
/**
 * Selects how many points a pulse contributes when synthetic-scan waveform processing is active (more than one ray per
 * pulse; see \ref LiDARcloud::syntheticScan). With \ref RETURN_MODE_MULTI every detected return above the detection
 * threshold is reported (discrete multi-return instrument with no return limit). With \ref RETURN_MODE_SINGLE a limited
 * number of returns per pulse is reported - at most \ref ScanMetadata::maxReturns of them (default 1 = classic
 * single-return, 2 = dual-return, N = N-return) - selected by \ref SingleReturnSelection. In single-return mode (maxReturns
 * = 1) two surfaces falling within the pulse range-resolution merge into one blended return at an intermediate range,
 * reproducing the "ghost"/"mixed pixel" point that a real single-return instrument records at an edge. A single ray per
 * pulse always produces an idealized exact intersection regardless of mode.
 */
enum ReturnMode {
    //! Report every detected return above the detection threshold (discrete multi-return, no return limit)
    RETURN_MODE_MULTI = 0,
    //! Report at most \ref ScanMetadata::maxReturns detector-selected returns per pulse (see \ref SingleReturnSelection)
    RETURN_MODE_SINGLE = 1
};

//! Which return(s) a limited-return instrument reports when a pulse resolves more returns than the return limit
/** Used only when \ref ReturnMode is \ref RETURN_MODE_SINGLE. When a pulse produces more returns than
 *  \ref ScanMetadata::maxReturns, this policy selects which subset to keep (the kept subset is always reported nearest-first).
 *  Mirrors the configurable first/last/strongest selection of real discrete-return scanners. \ref SINGLE_RETURN_STRONGEST,
 *  \ref SINGLE_RETURN_FIRST and \ref SINGLE_RETURN_LAST each keep the top \ref ScanMetadata::maxReturns returns by a single
 *  key; \ref SINGLE_RETURN_STRONGEST_PLUS_LAST instead keeps the strongest-plus-last pair (1 or 2 returns) and ignores
 *  \ref ScanMetadata::maxReturns. */
enum SingleReturnSelection {
    //! Keep the return(s) with the largest intensity (echo amplitude)
    SINGLE_RETURN_STRONGEST = 0,
    //! Keep the nearest return(s) (smallest range)
    SINGLE_RETURN_FIRST = 1,
    //! Keep the farthest return(s) (largest range)
    SINGLE_RETURN_LAST = 2,
    //! Dual return: keep the strongest return AND the last (farthest) return of the pulse, deduplicated to one when they
    //! are the same return. Models the "strongest + last" dual-return mode of real discrete-return scanners. Intrinsically
    //! yields 1 or 2 returns and ignores \ref ScanMetadata::maxReturns.
    SINGLE_RETURN_STRONGEST_PLUS_LAST = 3
};

//! A single rotating wedge prism in a Risley-prism beam deflector
/**
 * One element of the rotating-prism stack used by a \ref SCAN_PATTERN_RISLEY_PRISM scan. Each prism is a glass wedge with one
 * face perpendicular to the optical axis and one face tilted by the wedge angle; it deflects the transmitted beam by refraction
 * and rotates continuously about the optical axis. A pair of such prisms with different (and generally incommensurate) rotation
 * rates traces the characteristic non-repetitive rosette of a Livox sensor. The beam direction is computed by non-paraxial ray
 * tracing - refracting the beam through both faces of each prism with the vector form of Snell's law - so the maximum beam
 * deflection, and therefore the circular field of view, is an emergent property of the wedge angles and refractive indices,
 * not a directly specified parameter.
 */
struct RisleyPrism {

    //! Default Risley prism (no deflection)
    RisleyPrism() : wedge_angle(0.0), refractive_index(1.0), rotor_rate(0.0), phase(0.0) {
    }

    //! Construct a Risley prism
    /**
     * \param[in] wedge_angle  Wedge (inclination) angle of the prism in radians
     * \param[in] refractive_index  Refractive index of the prism glass
     * \param[in] rotor_rate  Rotation rate of the prism about the optical axis in radians per second (the sign sets the direction of rotation; a counter-rotating pair traces a rosette)
     * \param[in] phase  Initial clocking angle of the wedge about the optical axis in radians at scan time t=0 (defaults to 0)
     */
    RisleyPrism(double wedge_angle, double refractive_index, double rotor_rate, double phase = 0.0) : wedge_angle(wedge_angle), refractive_index(refractive_index), rotor_rate(rotor_rate), phase(phase) {
    }

    //! Wedge (inclination) angle of the prism in radians
    double wedge_angle;
    //! Refractive index of the prism glass
    double refractive_index;
    //! Rotation rate of the prism about the optical axis in radians per second (sign sets the rotation direction)
    double rotor_rate;
    //! Initial clocking angle of the wedge about the optical axis in radians at scan time t=0
    double phase;
};

//! Structure containing metadata for a scan
/** A static raster scan is initialized by providing 1) the origin of the scan (see \ref origin), 2) the number of zenithal scan directions (see \ref Ntheta), 3) the range of zenithal scan angles (see \ref thetaMin, \ref thetaMax), 4) the number of
azimuthal scan directions (see \ref Nphi), 5) the range of azimuthal scan angles (see \ref phiMin, \ref phiMax). This creates a grid of Ntheta x Nphi scan points which are all initialized as misses.  Points are set as hits using the addHitPoint()
function. There are various functions to query the scan data. \note During synthetic scan generation the raster pattern models the continuous azimuth rotation of a real terrestrial scanner: the azimuth drifts across each zenith column by one column step
((phiMax-phiMin)/(Nphi-1)), so the emitted zenith columns are slightly skewed (tilted) rather than perfectly vertical. The drift is sub-cell ((phiMax-phiMin)/((Nphi-1)*Ntheta) per zenith step) and is not reflected in the nominal (row,column) grid
mapping used by \ref rc2direction() / \ref direction2rc(). The same Ntheta x Nphi grid storage backs the other scan patterns and modes (see \ref ScanPattern and \ref ScanMode): a spinning multibeam scan stores per-channel zenith angles in
\ref beamZenithAngles, and a moving-platform scan additionally carries a 6-DOF trajectory (\ref traj_t, \ref traj_pos, \ref traj_quat) with the derived rotation rate and revolution count (\ref rotation_rate, \ref n_revolutions). Prefer the
high-level entry points \ref LiDARcloud::addScanSpinning() and \ref LiDARcloud::addScanMovingRaster() to set up moving/spinning scans from physical instrument parameters.
*/
struct ScanMetadata {

    //! Default LiDAR scan data structure
    ScanMetadata();

    //! Create a LiDAR scan data structure
    /**
     * \param[in] origin  (x,y,z) position of the scanner
     * \param[in] Ntheta  Number of scan points in the theta (zenithal) direction
     * \param[in] thetaMin  Minimum scan angle in the theta (zenithal) direction in radians
     * \param[in] thetaMax  Maximum scan angle in the theta (zenithal) direction in radians
     * \param[in] Nphi  Number of scan points in the phi (azimuthal) direction
     * \param[in] phiMin  Minimum scan angle in the phi (azimuthal) direction in radians
     * \param[in] phiMax  Maximum scan angle in the phi (azimuthal) direction in radians
     * \param[in] exitDiameter  Diameter of the laser pulse at exit from the scanner in meters
     * \param[in] beamDivergence  Divergence angle of the laser beam in radians
     * \param[in] rangeNoiseStdDev  Standard deviation of Gaussian range (along-beam) measurement noise in meters (0 disables noise)
     * \param[in] angleNoiseStdDev  Standard deviation of Gaussian angular (beam-pointing) jitter in radians (0 disables jitter)
     * \param[in] columnFormat  Vector of strings specifying the columns of the scan ASCII file for input/output
     * \param[in] scanTiltRoll  Global scanner tilt roll angle in radians (right-hand rotation about the body lateral axis; 0 = perfectly level) [optional]
     * \param[in] scanTiltPitch  Global scanner tilt pitch angle in radians (right-hand rotation about the body forward/azimuth-zero axis; 0 = perfectly level) [optional]
     * \param[in] scanAzimuthOffset  Global scanner azimuth (heading) offset in radians (right-hand rotation about the world +z axis applied on top of the azimuth sweep; 0 = no offset) [optional]
     */
    ScanMetadata(const helios::vec3 &origin, uint Ntheta, float thetaMin, float thetaMax, uint Nphi, float phiMin, float phiMax, float exitDiameter, float beamDivergence, float rangeNoiseStdDev, float angleNoiseStdDev,
                 const std::vector<std::string> &columnFormat, float scanTiltRoll = 0.f, float scanTiltPitch = 0.f, float scanAzimuthOffset = 0.f);

    //! Create a spinning multibeam LiDAR scan data structure (e.g. Velodyne/Ouster/Hesai rotating multi-channel sensor)
    /**
     * Each laser channel is fired at a fixed zenith angle (taken from \p beamZenithAngles) as the sensor head rotates through a
     * uniform sequence of azimuth angles. The scan is stored as an Ntheta x Nphi table where Ntheta equals the number of channels
     * (rows) and Nphi is the number of azimuth steps (columns), so all downstream processing is shared with raster scans. The
     * \ref scanPattern is set to \ref SCAN_PATTERN_SPINNING_MULTIBEAM and \ref thetaMin / \ref thetaMax are set to the minimum and
     * maximum of \p beamZenithAngles.
     * \note This is the low-level grid primitive. To set up a spinning multibeam scan from physical instrument parameters
     * (channel elevations, azimuth resolution, PRF, trajectory) with the azimuth grid, rotation rate, and revolution count
     * derived internally, use \ref LiDARcloud::addScanSpinning() instead.
     * \param[in] origin  (x,y,z) position of the scanner
     * \param[in] beamZenithAngles  Per-channel zenith angles in radians (zenith convention: 0 = upward, pi/2 = horizontal, pi = downward). Its size sets Ntheta. Manufacturer spec sheets typically list channel angles as elevation above the horizon;
     * zenith = pi/2 - elevation.
     * \param[in] Nphi  Number of azimuth steps (columns) per rotation
     * \param[in] phiMin  Minimum scan angle in the phi (azimuthal) direction in radians
     * \param[in] phiMax  Maximum scan angle in the phi (azimuthal) direction in radians
     * \param[in] exitDiameter  Diameter of the laser pulse at exit from the scanner in meters
     * \param[in] beamDivergence  Divergence angle of the laser beam in radians
     * \param[in] rangeNoiseStdDev  Standard deviation of Gaussian range (along-beam) measurement noise in meters (0 disables noise)
     * \param[in] angleNoiseStdDev  Standard deviation of Gaussian angular (beam-pointing) jitter in radians (0 disables jitter)
     * \param[in] columnFormat  Vector of strings specifying the columns of the scan ASCII file for input/output
     * \param[in] scanTiltRoll  Global scanner tilt roll angle in radians (right-hand rotation about the body lateral axis; 0 = perfectly level) [optional]
     * \param[in] scanTiltPitch  Global scanner tilt pitch angle in radians (right-hand rotation about the body forward/azimuth-zero axis; 0 = perfectly level) [optional]
     * \param[in] scanAzimuthOffset  Global scanner azimuth (heading) offset in radians (right-hand rotation about the world +z axis applied on top of the azimuth sweep; 0 = no offset) [optional]
     */
    ScanMetadata(const helios::vec3 &origin, const std::vector<float> &beamZenithAngles, uint Nphi, float phiMin, float phiMax, float exitDiameter, float beamDivergence, float rangeNoiseStdDev, float angleNoiseStdDev,
                 const std::vector<std::string> &columnFormat, float scanTiltRoll = 0.f, float scanTiltPitch = 0.f, float scanAzimuthOffset = 0.f);

    //! File containing hit point data
    std::string data_file;

    //! Number of zenithal angles in scan (rows)
    uint Ntheta;
    //! Minimum zenithal angle of scan in radians
    /**
     * \note Zenithal angles range from 0 (upward) to pi (downward).
     */
    float thetaMin;
    //! Maximum zenithal angle of scan in radians
    /**
     * \note Zenithal angles range from 0 (upward) to pi (downward).
     */
    float thetaMax;

    //! Number of azimuthal angles in scan (columns)
    uint Nphi;
    //! Minimum azimuthal angle of scan in radians
    /**
     * \note Azimuthal angles start at 0 (x=+,y=0) to 2pi (x=0,y=+) through 2pi.
     */
    float phiMin;
    //! Maximum azimuthal angle of scan in radians
    /**
     * \note Azimuthal angles start at 0 (x=+,y=0) to 2pi (x=0,y=+) through 2pi.
     */
    float phiMax;

    //!(x,y,z) coordinate of scanner location
    helios::vec3 origin;

    //! Diameter of laser pulse at exit from the scanner
    /**
     * \note This is not needed for single-return instruments, and is only used for synthetic scan generation.
     */
    float exitDiameter;

    //! Divergence angle of the laser beam in radians
    /**
     * \note This is not needed for single-return instruments, and is only used for synthetic scan generation.
     */
    float beamDivergence;

    //! Standard deviation of Gaussian range (along-beam) measurement noise in meters
    /**
     * Realistic LiDAR positional error is anisotropic: it is dominated by an error in the measured range (distance), which
     * displaces the hit point along the beam direction rather than isotropically in (x,y,z). During synthetic scan generation
     * the measured range of each return is perturbed by a zero-mean Gaussian draw with this standard deviation, and the hit
     * point is reconstructed along the nominal beam direction. A value of 0 disables range noise.
     * \note This is only used for synthetic scan generation.
     */
    float rangeNoiseStdDev;

    //! Standard deviation of Gaussian angular (beam-pointing) jitter in radians
    /**
     * Real LiDAR instruments have a small random error in the pointing direction of each emitted pulse. This contributes
     * the across-beam (lateral) component of the per-point positional error, which grows with range as approximately
     * range times this standard deviation, and is distinct from the within-beam spread produced by beam divergence. During
     * synthetic scan generation the nominal beam direction of each pulse is perturbed by an independent zero-mean Gaussian
     * angular offset with this standard deviation before ray tracing, so the whole beam (including any divergence cone and
     * finite aperture) is rotated together. A value of 0 disables angular jitter.
     * \note This is only used for synthetic scan generation.
     */
    float angleNoiseStdDev;

    //! Return-reporting mode for analytic-waveform synthetic scans
    /**
     * Selects whether a pulse reports all detected returns (\ref RETURN_MODE_MULTI, the default) or a single
     * detector-selected return (\ref RETURN_MODE_SINGLE). Only takes effect when more than one ray per pulse is fired (see
     * \ref LiDARcloud::syntheticScan); a single ray per pulse always produces an idealized exact intersection.
     * \note This is only used for synthetic scan generation.
     */
    ReturnMode returnMode = RETURN_MODE_MULTI;

    //! Which return(s) to keep in single/limited-return mode (see \ref SingleReturnSelection)
    /**
     * Used only when \ref returnMode is \ref RETURN_MODE_SINGLE. When a pulse resolves more returns than \ref maxReturns,
     * this policy selects which subset to keep (\ref SINGLE_RETURN_FIRST keeps the nearest, \ref SINGLE_RETURN_LAST the
     * farthest, \ref SINGLE_RETURN_STRONGEST the highest-amplitude). \ref SINGLE_RETURN_STRONGEST_PLUS_LAST is special: it
     * keeps the strongest return plus the last (farthest) return (deduplicated to one when they coincide) and ignores
     * \ref maxReturns, reporting 1 or 2 returns.
     * \note This is only used for synthetic scan generation.
     */
    SingleReturnSelection singleReturnSelection = SINGLE_RETURN_STRONGEST;

    //! Maximum number of returns reported per pulse in single/limited-return mode
    /**
     * Used only when \ref returnMode is \ref RETURN_MODE_SINGLE. 1 = classic single-return (the default), 2 = dual-return,
     * N = N-return; the kept returns are the subset chosen by \ref singleReturnSelection, always ordered nearest-first.
     * Ignored when \ref returnMode is \ref RETURN_MODE_MULTI (which reports every detected return), and also ignored when
     * \ref singleReturnSelection is \ref SINGLE_RETURN_STRONGEST_PLUS_LAST (which intrinsically reports 1 or 2 returns).
     * Must be >= 1.
     * \note This is only used for synthetic scan generation.
     */
    int maxReturns = 1;

    //! Range resolution (transmit pulse range-extent) in meters, used to merge sub-ray hits into discrete returns
    /**
     * Two surfaces separated by less than this distance fall within one transmit pulse and merge into a single detected
     * return at the energy-weighted range, reproducing the range-resolution / dead-zone limit of a real instrument (roughly
     * c*pulse_duration/2). When 0 the synthetic scanner falls back to the pulse_distance_threshold argument of
     * \ref LiDARcloud::syntheticScan.
     * \note This is only used for synthetic scan generation.
     */
    float pulseWidth = 0.f;

    //! Minimum return energy fraction for a return to be detected (noise floor)
    /**
     * A detected return whose intensity (range-normalized echo amplitude, expressed as a fraction of the total per-pulse
     * beam energy) is below this value is discarded, modeling the minimum detectable signal above the noise floor of a real
     * instrument. Set to 0 to disable suppression so every return is reported.
     *
     * The default of 0.05 models a realistic ~5% noise floor and pairs with the recommended ~40 rays/pulse: with N
     * sub-rays the weakest possible return is 1/N of pulse energy (a single sub-ray), so a 0.05 threshold at N=40 keeps
     * only returns supported by at least ~2 sub-rays. This suppresses the single-sub-ray "phantom" returns whose Bernoulli
     * appearance/disappearance otherwise forces very high ray counts to converge, letting ~40 rays/pulse reach a stable
     * multi-return statistic. Raise N if you lower this threshold below ~1/N (a threshold under 1/N has no effect, since no
     * return can be weaker than one sub-ray); a value of 0 reproduces the previous "report every return" behavior.
     * \note This is only used for synthetic scan generation.
     */
    float detectionThreshold = 0.05f;

    //! Global scanner tilt roll angle in radians (right-hand rotation about the body lateral axis)
    /**
     * Real terrestrial laser scanners are never perfectly leveled; a dual-axis inclinometer reports the residual tilt of the
     * scanner spin axis away from true vertical (plumb) as two orthogonal angles. This field is the roll component: a right-hand
     * rotation of the entire scan frame (the fan of ray directions) about the body lateral axis, applied about the scanner
     * origin during synthetic scan generation. The body frame is right-handed and Z-up, matching commercial scanners (e.g.
     * RIEGL SOCS): the forward axis is the horizontal projection of the azimuth-zero (\ref phiMin) scan direction, and the
     * lateral axis completes the right-handed frame. When \ref phiMin is 0 the lateral axis is the world +x axis. A value of 0
     * corresponds to a perfectly level scanner.
     * \note The sign convention is right-handed; commercial scanners do not share a universal inclinometer sign convention, so
     *       when matching a specific instrument the sign should be verified against that instrument's reported tilt.
     * \note This is only used for synthetic scan generation. See also \ref scanTilt_pitch.
     */
    float scanTilt_roll;

    //! Global scanner tilt pitch angle in radians (right-hand rotation about the body forward/azimuth-zero axis)
    /**
     * Pitch component of the global scanner tilt (see \ref scanTilt_roll): a right-hand rotation of the entire scan frame about
     * the body forward axis (the horizontal projection of the azimuth-zero / \ref phiMin scan direction), applied about the
     * scanner origin during synthetic scan generation. When \ref phiMin is 0 the forward axis is the world +y axis. Together
     * \ref scanTilt_roll and \ref scanTilt_pitch model the two angles reported by a real dual-axis inclinometer. A value of 0
     * corresponds to a perfectly level scanner. Roll is applied before pitch.
     * \note This is only used for synthetic scan generation.
     */
    float scanTilt_pitch;

    //! Global scanner azimuth (heading) offset in radians (right-hand rotation about the world vertical +z axis)
    /**
     * Yaw component of the global scanner orientation that completes the roll/pitch/yaw (\ref scanTilt_roll,
     * \ref scanTilt_pitch, azimuth) triad used by real instruments. While roll and pitch come from the dual-axis
     * inclinometer, the azimuth (heading) is the compass orientation of the scanner about the local vertical. This field
     * applies a right-hand rotation of the entire scan frame (the fan of ray directions and the tilt body axes) about the
     * world +z axis, about the scanner origin, during synthetic scan generation. It is applied as a heading offset on top
     * of the per-scan azimuth sweep [\ref phiMin, \ref phiMax]: the rotation order is yaw (azimuth) first, then pitch, then
     * roll. A value of 0 leaves the azimuth-zero (\ref phiMin) direction unchanged.
     * \note The sign convention is right-handed (counter-clockwise about +z when viewed from above). Commercial scanners do
     *       not share a universal heading sign convention, so when matching a specific instrument the sign should be
     *       verified against that instrument's reported azimuth.
     * \note This is only used for synthetic scan generation. See also \ref scanTilt_roll and \ref scanTilt_pitch.
     */
    float scanTilt_azimuth;

    //! Vector of strings specifying the columns of the scan ASCII file for input/output
    std::vector<std::string> columnFormat;

    //! Geometric pattern that maps scan-table (row,column) indices to beam directions
    /** Defaults to \ref SCAN_PATTERN_RASTER (uniform angular grid). When set to \ref SCAN_PATTERN_SPINNING_MULTIBEAM the
     *  per-channel zenith angles in \ref beamZenithAngles define the row directions instead of uniform [thetaMin,thetaMax] spacing.
     */
    ScanPattern scanPattern;

    //! Per-channel zenith angles for a spinning multibeam scan, in radians
    /** Used only when \ref scanPattern is \ref SCAN_PATTERN_SPINNING_MULTIBEAM. The number of entries equals \ref Ntheta (one per
     *  laser channel / row) and the values need not be uniformly spaced. Zenith convention: 0 = upward, pi/2 = horizontal,
     *  pi = downward. Manufacturer spec sheets typically list channel angles as elevation above the horizon, so zenith =
     *  pi/2 - elevation. Empty for \ref SCAN_PATTERN_RASTER scans.
     */
    std::vector<float> beamZenithAngles;

    //! Rotating wedge prisms of a Risley-prism (Livox-style rosette) scan
    /** Used only when \ref scanPattern is \ref SCAN_PATTERN_RISLEY_PRISM. The beam is refracted in order through each prism in
     *  this stack (a Livox sensor uses two counter-rotating prisms; the underlying model supports any number). The per-pulse
     *  body-frame beam direction is computed by full Snell's-law refraction through the rotating prisms at the pulse's time
     *  (see \ref rc2direction). Empty for the other scan patterns.
     */
    std::vector<RisleyPrism> risley_prisms;

    //! Refractive index of the medium surrounding the Risley prisms (air)
    /** Used only when \ref scanPattern is \ref SCAN_PATTERN_RISLEY_PRISM. Defaults to 1.0 (vacuum/air). */
    double risley_refractive_index_air = 1.0;

    //! Convert the (row,column) of hit point in a scan to a direction vector
    /**
     * \param[in] row  Index of hit point in the theta (zenithal) direction.
     * \param[in] column  Index of hit point in the phi (azimuthal) direction.
     * \return Spherical vector corresponding to the ray direction for the given hit point.
     */
    helios::SphericalCoord rc2direction(uint row, uint column) const;

    //! Convert the scan ray direction into (row,column) table index
    /**
     * \param[in] direction  Spherical vector corresponding to the ray direction for the given hit point.
     * \return (row,column) table index for the given hit point
     */
    helios::int2 direction2rc(const helios::SphericalCoord &direction) const;

    // ---- Moving-platform (mobile/airborne) scan support ----
    // When isMoving is true the scanner pose is driven by a dense timestamped 6-DOF trajectory and the
    // synthetic scan generates a per-pulse origin and orientation instead of a single static origin. The
    // static scanTilt_roll/pitch/azimuth fields are NOT applied in this mode: attitude comes entirely from
    // the trajectory quaternions composed with the fixed boresight misalignment. When isMoving is false the
    // scan behaves exactly as before (single static origin). Populated via \ref LiDARcloud::addScanMoving().

    //! Whether this scan uses a moving-platform trajectory (true) or a single static origin (false)
    bool isMoving = false;

    //! Monotonically increasing trajectory sample times in seconds (size M)
    std::vector<double> traj_t;

    //! Trajectory platform (body-frame origin) positions in world coordinates, one per \ref traj_t entry (size M)
    std::vector<helios::vec3> traj_pos;

    //! Trajectory platform orientations as quaternions (qx,qy,qz,qw), one per \ref traj_t entry (size M)
    /**
     * Hamilton convention, body->world: the quaternion rotates a vector expressed in the platform body frame
     * into the world frame. Stored as (x,y,z,w) in a \ref helios::vec4.
     */
    std::vector<helios::vec4> traj_quat;

    //! Position of the sensor optical center in the platform body frame, in meters (lever arm)
    helios::vec3 lever_arm = helios::make_vec3(0, 0, 0);

    //! Fixed sensor rotational misalignment (boresight) as roll/pitch/yaw in radians, in the body frame
    helios::vec3 boresight_rpy = helios::make_vec3(0, 0, 0);

    //! Time between consecutive pulses in seconds (= 1 / pulse_rate_hz). For static scans this defaults to 1.0
    //! so per-pulse timestamps equal the historical pulse ordinal (i.e. no change to existing behavior).
    double pulse_period = 1.0;

    //! Time of the first pulse (pulse ordinal 0) in seconds. Relative time; no absolute GPS epoch.
    double t0 = 0.0;

    // ---- Self-describing acquisition descriptors ----
    // These make a scan introspectable ("how was it acquired / how fast did it spin / how many revolutions?")
    // without reverse-engineering the answer from scanPattern, isMoving, and phiMax. They are set by the
    // high-level scan-creation entry points (\ref LiDARcloud::addScanSpinning, \ref LiDARcloud::addScanMovingRaster,
    // etc.) and are purely descriptive; the underlying mechanism is still scanPattern + isMoving.

    //! High-level acquisition-mode descriptor (see \ref ScanMode)
    ScanMode scanMode = SCAN_MODE_STATIC_RASTER;

    //! Number of azimuth firing steps per full 360-degree revolution (spinning multibeam scans only; 0 otherwise)
    uint steps_per_rev = 0;

    //! Sensor-head rotation rate in revolutions per second (spinning multibeam scans only; 0 otherwise)
    /** Derived internally as PRF / (channels * steps_per_rev). */
    double rotation_rate = 0.0;

    //! Total number of full 360-degree revolutions collected (spinning multibeam scans only; may be fractional; 0 otherwise)
    /** Derived internally as rotation_rate * trajectory_duration. */
    double n_revolutions = 0.0;

    //! Interpolate the platform pose at trajectory time \p t (moving-platform scans only)
    /**
     * Brackets \p t in \ref traj_t by binary search, linearly interpolates \ref traj_pos, and spherically
     * interpolates (SLERP) \ref traj_quat. Times outside [traj_t.front(), traj_t.back()] are clamped to the
     * nearest endpoint. Throws via helios_runtime_error if the trajectory is empty, the three trajectory
     * vectors differ in length, or \ref traj_t is not strictly increasing.
     * \param[in] t  Query time in seconds.
     * \param[out] pos  Interpolated platform position in world coordinates.
     * \param[out] quat  Interpolated platform orientation quaternion (qx,qy,qz,qw), Hamilton body->world.
     */
    void poseAt(double t, helios::vec3 &pos, helios::vec4 &quat) const;
};

//! Primary class for terrestrial LiDAR scan
class LiDARcloud {
private:
    size_t Nhits;

    std::vector<ScanMetadata> scans;

    std::vector<HitPoint> hits;

    // ---- Columnar per-hit scalar data ----
    // Per-hit scalar fields are stored column-wise (one contiguous array per label) rather than as one
    // std::map<std::string,double> per hit. This makes bulk field extraction a cache-linear pass instead
    // of N cache-cold red-black-tree lookups, matching the speed of XYZ/RGB export.
    // INVARIANT: every column in hit_data_columns and every mask in hit_data_present has length
    // hits.size() at all times. Column slot `s` holds label hit_data_labels[s]; a value is meaningful
    // only where hit_data_present[s][i] != 0 (a hit may be missing a value for a label). The values
    // for slot s of hit i are hit_data_columns[s][i]. These structures are kept in lockstep with the
    // `hits` vector everywhere it is mutated (addHitPoint, deleteHitPoint's swap-and-pop, clearHits).
    std::vector<std::string> hit_data_labels; //!< Column order; hit_data_labels[s] is the label of slot s.
    std::unordered_map<std::string, size_t> hit_data_label_index; //!< label -> column slot.
    std::vector<std::vector<double>> hit_data_columns; //!< [slot][hit_index] scalar values.
    std::vector<std::vector<char>> hit_data_present; //!< [slot][hit_index] presence flag (0 = absent).

    //! Resolve a label to its column slot, creating a new full-length, absent-back-filled column if it
    //! does not yet exist. Returns the slot index.
    size_t getOrCreateHitDataColumn(const std::string &label);

    //! Append one hit's scalar data as a new row across all columns. Call AFTER the HitPoint has been
    //! pushed onto `hits` (it uses index hits.size()-1). Keeps every column length-aligned with `hits`.
    void appendHitData(const std::map<std::string, double> &data);

    //! Clear all hits AND their columnar scalar data, keeping the two in lockstep.
    void clearHits();

    //! Apply an in-place transform to hit `index`'s per-pulse emission origin (labels origin_x/y/z), if
    //! it carries one. All-three-or-none semantics. Used by coordinateShift/coordinateRotation.
    void transformHitOrigin(uint index, const std::function<helios::vec3(const helios::vec3 &)> &transform);

    //! Return hit `index`'s per-pulse origin if it carries one (labels origin_x/y/z), else `fallback`.
    helios::vec3 hitOriginOrFallback(uint index, const helios::vec3 &fallback) const;

    std::vector<GridCell> grid_cells;

    //! Test/diagnostic hook: when true, \ref calculateLeafArea_inner() uses the brute-force per-cell slab loop even for
    //! regular lattices that would otherwise use the fast DDA path. Used by the self-tests to A/B the two paths against
    //! each other; has no effect on results, only on which code path computes them.
    bool force_bruteforce_LAD = false;

    std::vector<Triangulation> triangles;

    //! 2D map of hits, one value for each (theta,phi) combo of scan. = -1 if no hit, = index if hit - size = (Ntheta)x(Nphi)
    std::vector<HitTable<int>> hit_tables;

    //! Flag denoting whether \ref LiDARcloud::calculateHitGridCell[*]() has been called previously.
    bool hitgridcellcomputed;

    //! Flag denoting whether triangulation has been performed previously
    bool triangulationcomputed;

    //! Diagnostics from the most recent \ref triangulateHitPoints() call. Each
    //! dropped triangle is attributed to ONE primary reason (Lmax, then aspect,
    //! then degenerate), so candidates == kept + dropped_lmax + dropped_aspect +
    //! dropped_degenerate. `kept` mirrors triangles.size() for the run.
    std::size_t triangulation_candidate_count;
    std::size_t triangulation_dropped_lmax;
    std::size_t triangulation_dropped_aspect;
    std::size_t triangulation_dropped_degenerate;

    //! Return the index of the grid cell containing point \p p, or -1 if \p p lies outside every
    //! cell. Uses the same axis-aligned containment test (with inverse rotation for rotated cells)
    //! as \ref calculateHitGridCell(), so hit-point binning and external-triangle binning agree.
    int getContainingGridCell(const helios::vec3 &p) const;

    //! Flag denoting whether messages should be printed to screen
    bool printmessages;

    //! Optional callback fired with (progress_fraction, message) during syntheticScan
    std::function<void(float, const std::string &)> progress_callback;

    //! Soft cap (bytes) on the transient ray-tracing buffers held live at once during \ref syntheticScan. The per-scan beam
    //! fan-out is processed in chunks so that only chunk_beams * rays_per_pulse sub-rays are materialized simultaneously,
    //! bounding peak working memory to roughly this budget regardless of scan size. See \ref setSyntheticScanMemoryBudget.
    //! A value of 0 (the default) selects an automatic, path-dependent cap: \ref SYNTHETIC_SCAN_DEFAULT_BUDGET_GPU on a
    //! GPU build (the GPU ray-tracer handles larger batches efficiently) and \ref SYNTHETIC_SCAN_DEFAULT_BUDGET_CPU
    //! otherwise. This bounds only the trace-time scratch buffers; the output point cloud grows with the number of
    //! recorded returns and is not affected by this setting.
    size_t synthetic_scan_memory_budget_bytes = 0; // 0 => automatic (see syntheticScan)

    //! Automatic \ref syntheticScan memory cap used on the CPU/OpenMP ray-tracing path when no explicit budget is set.
    static constexpr size_t SYNTHETIC_SCAN_DEFAULT_BUDGET_CPU = size_t(4) * 1024 * 1024 * 1024; // 4 GiB

    //! Automatic \ref syntheticScan memory cap used on the GPU ray-tracing path when no explicit budget is set.
    static constexpr size_t SYNTHETIC_SCAN_DEFAULT_BUDGET_GPU = size_t(8) * 1024 * 1024 * 1024; // 8 GiB

    //! Collision detection plugin for unified ray-tracing
    CollisionDetection *collision_detection;

    //! External cancellation flag forwarded to the collision-detection ray loop
    //! so a long syntheticScan can be aborted mid-trace (nullptr = none).
    volatile int *cancel_flag = nullptr;

    //! Optional caller-owned counter that syntheticScan writes the index of the
    //! scan currently being ray-traced into (0-based), updated at the start of each
    //! scan (nullptr = none). Lets a host poll per-scan progress from another thread
    //! while the blocking syntheticScan call runs. Not owned by LiDARcloud — the
    //! caller manages its lifetime.
    volatile int *synthetic_scan_progress = nullptr;

    //! Prepare the collision-detection engine for a batch of ray-tracing calls over static geometry.
    /**
     * Initializes the collision-detection plugin, disables automatic BVH rebuilds, and builds the BVH once. Use this once
     * before issuing many \ref castRaysUnified() calls (e.g. one per beam chunk) so the BVH is not rebuilt on every call;
     * pair with \ref finishUnifiedRayTracing() after the batch. The geometry must not change between prepare and finish.
     * \param[in] context Pointer to the Helios context.
     */
    void prepareUnifiedRayTracing(helios::Context *context);

    //! Re-enable automatic BVH rebuilds after a batch of ray-tracing calls (see \ref prepareUnifiedRayTracing).
    void finishUnifiedRayTracing();

    //! Trace a batch of already-prepared rays into the caller's result buffers (no BVH (re)build).
    /**
     * Casts total_rays rays and writes per-ray results into hit_t/hit_fnorm/hit_ID using the low-memory SoA path
     * (\ref CollisionDetection::castRaysSoA), avoiding any per-call full-length RayQuery/HitResult vectors. A miss writes
     * hit_t = \ref LIDAR_RAYTRACE_MISS_T, hit_ID = -1, hit_fnorm = 1e6; a hit writes the hit distance, primitive UUID, and
     * the dot product of the ray direction with the surface normal. The BVH must already be current — call
     * \ref prepareUnifiedRayTracing() once before the first batch.
     * \param[in] total_rays Number of rays to trace.
     * \param[in] ray_origins Array of ray origins (length total_rays).
     * \param[in] direction Array of ray directions (length total_rays).
     * \param[out] hit_t Array (length total_rays) receiving the hit distance (or miss sentinel) per ray.
     * \param[out] hit_fnorm Array (length total_rays) receiving dot(direction, normal) per hit (1e6 on a miss).
     * \param[out] hit_ID Array (length total_rays) receiving the hit primitive UUID (-1 on a miss).
     * \param[in] packet_size Rays per coherent packet (the rays are laid out so each consecutive group of packet_size
     *                        rays is one pulse's sub-rays). >1 enables coherent packet traversal
     *                        (\ref CollisionDetection::castRaysSoA_packets); 1 (default) uses per-ray traversal.
     */
    void castRaysUnified(size_t total_rays, helios::vec3 *ray_origins, helios::vec3 *direction, float *hit_t, float *hit_fnorm, int *hit_ID, size_t packet_size = 1);

    // -------- I/O --------- //

    // -------- RECONSTRUCTION --------- //

    // first index: leaf group, second index: triangle #
    std::vector<std::vector<Triangulation>> reconstructed_triangles;

    std::vector<std::vector<Triangulation>> reconstructed_trunk_triangles;

    std::vector<helios::vec3> reconstructed_alphamasks_center;
    std::vector<helios::vec2> reconstructed_alphamasks_size;
    std::vector<helios::SphericalCoord> reconstructed_alphamasks_rotation;
    std::vector<uint> reconstructed_alphamasks_gridcell;
    std::string reconstructed_alphamasks_maskfile;
    std::vector<uint> reconstructed_alphamasks_direct_flag;

    void leafReconstructionFloodfill();

    void backfillLeavesAlphaMask(const std::vector<float> &leaf_size, float leaf_aspect_ratio, float solidfraction, const std::vector<bool> &group_filter_flag);

    void calculateLeafAngleCDF(uint Nbins, std::vector<std::vector<float>> &CDF_theta, std::vector<std::vector<float>> &CDF_phi);

    void floodfill(size_t t, std::vector<Triangulation> &cloud_triangles, std::vector<int> &fill_flag, std::vector<std::vector<int>> &nodes, int tag, int depth, int maxdepth);

    // -------- HELPERS --------- //

    //! Compute G(theta) values for all grid cells from triangulation data
    /**
     * \param[in] Ncells Number of grid cells
     * \param[in] Nscans Number of scans
     * \param[out] Gtheta G(theta) value for each cell (size = Ncells)
     * \param[out] Gtheta_bar Average G(theta) for each cell (size = Ncells)
     */
    void computeGtheta(uint Ncells, uint Nscans, std::vector<float> &Gtheta, std::vector<float> &Gtheta_bar);

    //! True if any scan in the cloud is a moving-platform scan (see \ref addScanMoving())
    /**
     * Used to guard functions that assume a single static scan origin (triangulation, ray-direction validation,
     * distance filtering) and therefore cannot operate correctly on a moving-platform scan.
     */
    bool anyScanMoving() const;

    //! Shared implementation of the beam-based leaf-area inversion (see the public \ref calculateLeafArea overloads)
    /**
     * Performs the per-voxel beam classification and Beer-Lambert LAD inversion. Beam geometry is computed from the
     * per-pulse emission origin recorded on each hit (\ref getHitOrigin()), so it is correct for both static and
     * moving-platform scans.
     * \param[in] context Pointer to the Helios context
     * \param[in] min_voxel_hits Minimum number of allowable LiDAR hits per voxel
     * \param[in] element_width Characteristic vegetation element width [m] (<= 0 reports sampling-only uncertainty)
     * \param[in] supplied_Gtheta Controls the source of G(theta):
     *            - empty: G(theta) is computed per voxel from triangulation, which must have been performed.
     *            - size 1: the single value is broadcast to every voxel; triangulation is NOT required.
     *            - size == grid-cell count: the value is used per voxel in cell order; triangulation is NOT required.
     *            Any other size is an error. Every supplied value must be in (0,1].
     */
    void calculateLeafArea_inner(helios::Context *context, int min_voxel_hits, float element_width, const std::vector<float> &supplied_Gtheta);

    //! Perform LAD inversion for a single voxel using secant method
    /**
     * \param[in] voxel_index Grid cell index
     * \param[in] P Transmission probability for this voxel
     * \param[in] Gtheta G(theta) value for this voxel
     * \param[in] dr_samples Vector of path length samples through this voxel
     * \param[in] min_voxel_hits Minimum number of hits required
     * \param[in] gridsize Voxel dimensions (x, y, z)
     * \param[out] leaf_area Computed leaf area for this voxel
     * \param[in,out] warnings Warning aggregator for per-voxel convergence warnings
     * \return True if inversion succeeded, false otherwise
     */
    bool invertLAD(uint voxel_index, float P, float Gtheta, const std::vector<float> &dr_samples, int min_voxel_hits, const helios::vec3 &gridsize, float &leaf_area, helios::WarningAggregator &warnings);

    //! Result of a per-voxel leaf-area-density inversion, including sampling uncertainty
    struct LADInversionResult {
        float leaf_area = 0.f; //!< Point estimate of leaf area in the voxel [m^2]
        float LAD_variance = -1.f; //!< Sampling variance of LAD [1/m]^2 (terms a+b); -1 if undefined
        int beam_count = 0; //!< N: beams that entered the voxel
        float I_rdi = 0.f; //!< Relative density index I = 1 - P
        float zbar_e = 0.f; //!< Mean beam path length through the voxel [m]
        float var_path = 0.f; //!< Empirical variance of per-beam path lengths [m^2]
        float L1_element = -1.f; //!< Single-element optical depth L1 (-1 if element size unknown)
        bool converged = false; //!< True if the secant solve converged (false => fallback used)
        bool element_size_known = false; //!< False => variance is sampling-only (term a; term b omitted)
    };

    //! Invert Beer-Lambert per voxel for leaf area AND its statistical sampling variance
    /** Wraps the point-estimate inversion (\ref invertLAD()) and additionally computes the
        per-voxel sampling variance of LAD following Pimont et al. (2018), RSE 215:343-370.
        The variance has two components: (a) a finite-beam sampling term that decays as 1/N, and
        (b) an N-independent element-position-variability term that requires the element size.
        \param[in] voxel_index Index of the voxel being inverted (for warning messages)
        \param[in] P Transmission probability for this voxel
        \param[in] Gtheta G(theta) value for this voxel
        \param[in] dr_samples Per-beam path length samples through this voxel [m]
        \param[in] sum_frac_sq Sum over beams of the squared per-beam transmittance fraction (for the empirical-variance guard)
        \param[in] element_width Characteristic vegetation element width [m] (<=0 => term (b) omitted, sampling-only)
        \param[in] min_voxel_hits Minimum number of beams required to attempt inversion
        \param[in] gridsize Voxel dimensions (x, y, z) [m]
        \param[in,out] warnings Warning aggregator for per-voxel convergence warnings
        \return LADInversionResult with the point estimate and its sampling variance
     */
    LADInversionResult invertLADWithVariance(uint voxel_index, float P, float Gtheta, const std::vector<float> &dr_samples, float sum_frac_sq, float element_width, int min_voxel_hits, const helios::vec3 &gridsize,
                                             helios::WarningAggregator &warnings);

    //! Test whether a voxel's (L, L1, N) fall within the Pimont (2018) Table-3 CI-validity envelope
    /** \param[in] L Voxel optical depth L = lambda*delta (= a*Gtheta*zbar_e)
        \param[in] L1 Single-element optical depth L1 = lambda1*delta
        \param[in] N Number of beams that entered the voxel
        \param[in] confidence_level Confidence level (0.90 and 0.95 are tabulated; others use the 0.95 envelope)
        \return True if the Wald confidence interval is trustworthy for this voxel
     */
    bool ciValidPimont(float L, float L1, int N, float confidence_level) const;

    // -------- MULTI-RETURN HELPERS --------- //

    //! Detect if point cloud contains multi-return data
    /**
     * \return True if multi-return data detected (target_count > 1), false otherwise
     */
    bool isMultiReturnData() const;

    //! Beam grouping structure for multi-return data (compressed-sparse-row layout)
    /** A pulse ("beam") may produce several returns. Rather than a vector-of-vectors (one small heap allocation per
        beam - prohibitive for the tens of millions of beams in a dense scan), the returns are stored CSR-style: \ref
        beam_members holds every return's global hit index, grouped contiguously by beam, and \ref beam_offsets gives
        the [start,end) range of each beam within it. Beam \c k owns members[beam_offsets[k] .. beam_offsets[k+1]). */
    struct BeamGrouping {
        uint Nbeams = 0;
        std::vector<uint> beam_members; //!< All returns' global hit indices, grouped contiguously by beam
        std::vector<uint> beam_offsets; //!< Size Nbeams+1; beam k spans beam_members[beam_offsets[k] .. beam_offsets[k+1])

        //! Number of returns in beam \p k
        uint beamSize(uint k) const {
            return beam_offsets[k + 1] - beam_offsets[k];
        }
    };

    //! Group hit points by timestamp into beams
    /**
     * \param[in] scan_indices Vector of hit indices for a specific scan
     * \return BeamGrouping structure with beam organization
     */
    BeamGrouping groupHitsByTimestamp(const std::vector<uint> &scan_indices) const;

    //! Description of a regular voxel lattice reconstructed from the grid cells, used by the fast LAD inversion path
    /** When every grid cell shares a common anchor, size, division count, and azimuthal rotation (the case produced by
        \ref addGrid()), the cells form a regular axis-aligned lattice and the leaf-area inversion can walk each beam
        through only the voxels it pierces (3D-DDA) instead of testing every hit against every voxel. \ref detectVoxelLattice()
        validates this and fills in the geometry below. \p valid is false when the cells do not form such a lattice (e.g. a
        grid assembled cell-by-cell with mixed sizes/rotations), in which case the inversion falls back to the brute-force
        per-cell slab test. */
    struct VoxelLattice {
        bool valid = false; //!< True if the cells form a regular lattice that the DDA path can use
        helios::vec3 origin; //!< Minimum corner of the lattice in the un-rotated lattice frame (= global_anchor - 0.5*global_size)
        helios::vec3 anchor; //!< Rotation pivot (= shared global_anchor)
        helios::vec3 cell_extent; //!< Per-cell dimensions (= global_size / global_count)
        float rotation = 0.f; //!< Shared azimuthal rotation about z [rad]
        helios::int3 count; //!< Number of cells along each axis (= global_count)
        std::vector<int> ijk_to_index; //!< Dense (i,j,k)->cell-index map (size count.x*count.y*count.z); -1 if no cell occupies that slot
    };

    //! Detect whether the current grid cells form a regular lattice and reconstruct its geometry
    /** \return A \ref VoxelLattice with \p valid set appropriately. When valid, the DDA fast path in
        \ref calculateLeafArea_inner() is used; otherwise the brute-force fallback is used. */
    VoxelLattice detectVoxelLattice() const;

    //! Accumulate one beam's already-classified returns into the equal-weighting transmission statistics for a voxel
    /** Shared by both the DDA fast path and the brute-force fallback in \ref calculateLeafArea_inner() so the two paths
        produce identical results. The caller classifies each of the beam's returns relative to the voxel and supplies the
        per-return path length \p dr (\c |t1-t0|, 0 if the return's ray misses the voxel) and \p hit_location
        (0=miss-voxel, 1=before, 2=inside, 3=after). This helper folds those into the per-beam transmittance fraction
        P = E_after/(E_inside+E_after) (Eq. 7) and the per-voxel path-length sample (mean dr over the beam's intersecting
        returns), exactly mirroring the original inlined logic.
        \param[in] return_indices Pointer to this beam's return indices (into \p dr / \p hit_location)
        \param[in] Nreturns Number of returns belonging to this beam
        \param[in] dr Path length through the voxel per return index [m] (0 if that return's ray does not intersect the voxel)
        \param[in] hit_location Classification per return index (0=miss-voxel, 1=before, 2=inside, 3=after)
        \param[in,out] P_equal_numerator Running sum of per-beam transmittance fractions for this voxel
        \param[in,out] P_equal_denominator Running count of beams contributing to this voxel
        \param[in,out] P_equal_sumsq Running sum of squared per-beam fractions (sampling-variance guard)
        \param[in,out] dr_array_cell Per-voxel path-length samples (one push_back per contributing beam) */
    static void accumulateBeamCell(const uint *return_indices, size_t Nreturns, const std::vector<float> &dr, const std::vector<uint> &hit_location, float &P_equal_numerator, float &P_equal_denominator, float &P_equal_sumsq,
                                   std::vector<float> &dr_array_cell);

    //! Helper method for loading TreeQSM cylinder files with different coloring strategies
    /**
     * \param[in] context Pointer to the Helios context where tube objects will be added.
     * \param[in] filename Path to the TreeQSM cylinder text file.
     * \param[in] radial_subdivisions Number of radial subdivisions for the tube objects.
     * \param[in] use_colormap If true, use colormap coloring; if false, use texture or solid color.
     * \param[in] colormap_or_texture Either colormap name (if use_colormap=true) or texture file path (if use_colormap=false).
     * \return Vector of tube object UUIDs that were created.
     */
    std::vector<uint> loadTreeQSM_impl(helios::Context *context, const std::string &filename, uint radial_subdivisions, bool use_colormap, const std::string &colormap_or_texture);

    //! Timestamp-based implementation of gap filling (see \ref gapfillMisses). Reconstructs the scan grid from per-hit timestamps.
    /**
     * \param[in] scanID ID of scan to gapfill
     * \param[in] gapfill_grid_only if true, missing points are gapfilled only within the axis-aligned bounding box of the voxel grid
     * \param[in] add_flags if true, gapfillMisses_code is added as hitpoint data
     * \return (x,y,z) of missing points added to the scan from gapfilling
     */
    std::vector<helios::vec3> gapfillMisses_timestamp(uint scanID, const bool gapfill_grid_only, const bool add_flags);

    //! Row/column-based implementation of gap filling (see \ref gapfillMisses).
    /**
     * Reconstructs miss-pulse directions from the native scan-grid row/column indices when per-hit timestamps are
     * unavailable. A robust per-row generative model is fit from the returns: each row's zenith is the median of the
     * measured zeniths of its returns, and each row's azimuth is a robust (Theil-Sen) line fit azimuth = intercept + slope*column,
     * which absorbs scanner tilt and the continuous azimuth sweep (azimuth shear across rows). Rows with too few returns
     * inherit their model parameters by robustly extrapolating the per-row parameters across the row axis, which allows
     * large blank near-zenith regions to be extrapolated rather than only interpolated. Every empty grid cell is then
     * emitted as a miss along its reconstructed direction.
     *
     * \param[in] scanID ID of scan to gapfill
     * \param[in] add_flags if true, gapfillMisses_code is added as hitpoint data (0 = original points, 1 = gapfilled interior, 4 = extrapolated row)
     * \return (x,y,z) of missing points added to the scan from gapfilling
     */
    std::vector<helios::vec3> gapfillMisses_rowcolumn(uint scanID, const bool add_flags);

public:
    //! LiDAR point cloud constructor
    LiDARcloud();

    //! LiDAR point cloud destructor
    ~LiDARcloud();

    //! Self-test (unit test) function
    static int selfTest(int argc = 0, char **argv = nullptr);

    void validateRayDirections();

    //! Disable all print messages to the screen except for fatal error messages
    void disableMessages();

    //! Enable all print messages to the screen
    void enableMessages();

    //! Register a callback to receive progress updates during syntheticScan
    /**
     * progress_fraction is in [0, 1]. message describes the current operation phase.
     * Pass an empty std::function to clear the callback.
     * \param[in] callback Function that receives (progress_fraction, message_string).
     */
    void setProgressCallback(std::function<void(float, const std::string &)> callback);

    //! Register an external cancellation flag polled during long-running operations
    /**
     * When the pointed-to int becomes non-zero, the current long-running operation aborts at the
     * next poll point. \ref syntheticScan() stops its parallel ray loop and returns early with
     * whatever hits were recorded so far; \ref triangulateHitPoints() discards any partial mesh
     * and returns an empty triangulation. The flag is owned by the caller (e.g. a ctypes int
     * shared with Python) and must outlive the operation; pass nullptr to clear. Set this before
     * calling the operation to be cancelled.
     * \param[in] flag Pointer to a 0/non-zero cancellation flag, or nullptr.
     */
    void setCancelFlag(volatile int *flag);

    //! Register an external counter for per-scan syntheticScan progress
    /**
     * syntheticScan writes the 0-based index of the scan it is currently ray-tracing
     * into the pointed-to int, updated at the start of each scan, and sets it to
     * getScanCount() when the batch finishes. The counter is owned by the caller
     * (e.g. a ctypes int shared with Python) and must outlive the scan; pass nullptr
     * to clear. Set this before calling syntheticScan().
     * \param[in] ptr Pointer to a caller-owned progress counter, or nullptr.
     */
    void setSyntheticScanProgressPointer(volatile int *ptr);

    //! Initialize collision detection plugin for unified ray-tracing (called automatically when needed)
    void initializeCollisionDetection(helios::Context *context);

    //! Perform unified ray-tracing using collision detection plugin (replaces CUDA kernels)
    void performUnifiedRayTracing(helios::Context *context, size_t N, int Npulse, helios::vec3 *ray_origins, helios::vec3 *direction, float *hit_t, float *hit_fnorm, int *hit_ID);

    //! Normalize a synthetic return intensity for range
    /** Helios reports <b>range-normalized</b> intensity: the range-independent return amplitude
     * \f$\rho\,\cos\theta\f$ (per-primitive reflectivity times the incidence-angle cosine), as if the geometric
     * \f$1/R^2\f$ loss of the LiDAR range equation had been measured and then divided back out
     * (\f$(\rho\,\cos\theta/R^2)\cdot R^2 = \rho\,\cos\theta\f$). Two identical surfaces at different ranges
     * therefore return the same intensity. Because \ref syntheticScan() generates intensity directly as
     * \f$\rho\,\cos\theta\f$ without ever applying the \f$1/R^2\f$ loss, this normalization is the identity on
     * the value; the helper exists to make the convention explicit and is the single place to change should the
     * raw (range-dependent) convention be wanted instead. Partial-footprint attenuation of sub-footprint
     * returns (in multi-return mode) is carried separately by the fraction of beam sub-rays that strike the target and is
     * deliberately preserved, as it reflects a target property rather than a range-geometry loss.
     * \param[in] intensity Return intensity (\f$\rho\,\cos\theta\f$).
     * \param[in] distance Measured range from the scanner to the return, in meters (accepted for interface
     * symmetry and future raw-mode use).
     * \return Range-normalized intensity.
     */
    [[nodiscard]] static float applyRangeIntensityCorrection(float intensity, float distance);

    // ------- SCANS -------- //

    //! Get number of scans in point cloud
    uint getScanCount();

    //! Add a LiDAR scan to the point cloud
    /**
     * \param[in] newscan LiDAR scan data structure
     * \return ID for scan that was created
     */
    uint addScan(ScanMetadata &newscan);

    //! Add a moving-platform (mobile/airborne) LiDAR scan driven by a 6-DOF pose trajectory
    /**
     * Registers a scan whose scanner pose changes during the sweep. The synthetic scan generator (see
     * \ref syntheticScan()) computes, for each pulse, its acquisition time \f$t = t_0 + \mathrm{ordinal}\cdot
     * \mathrm{pulse\_period}\f$ (where pulse_period = 1/\p pulse_rate_hz and the ordinal is the pulse's position
     * in the scan-grid firing sequence), interpolates the platform pose at that time via \ref ScanMetadata::poseAt(),
     * and emits a per-pulse origin \f$\mathbf{o} = \mathbf{pos} + R(\mathbf{q})\,\mathbf{lever\_arm}\f$ and direction
     * \f$\mathbf{d} = R(\mathbf{q})\,R(\mathbf{boresight})\,\mathbf{d}_{body}\f$. Every resulting hit and miss stores
     * its own origin (data labels "origin_x"/"origin_y"/"origin_z"), real timestamp ("timestamp"), and firing index
     * ("pulse_id"). The static \ref ScanMetadata::scanTilt_roll / scanTilt_pitch / scanTilt_azimuth fields are not
     * applied in this mode and must be zero (attitude is defined entirely by \p traj_quat and \p boresight_rpy).
     * \param[in] scan  Scan metadata defining the angular sampling grid and beam parameters (origin is ignored; the trajectory supplies position).
     * \param[in] traj_t  Monotonically increasing trajectory sample times in seconds (size M).
     * \param[in] traj_pos  Platform positions in world coordinates, one per \p traj_t entry (size M).
     * \param[in] traj_quat  Platform orientation quaternions (qx,qy,qz,qw), Hamilton body->world, one per \p traj_t entry (size M).
     * \param[in] lever_arm  Sensor optical center in the platform body frame (meters).
     * \param[in] boresight_rpy  Fixed sensor rotational misalignment as roll/pitch/yaw in radians (body frame).
     * \param[in] pulse_rate_hz  Pulse repetition rate in Hz (must be > 0); sets the time between consecutive pulses.
     * \param[in] t0  Time of the first pulse in seconds (relative time; defaults to 0).
     * \return ID for scan that was created
     */
    uint addScanMoving(ScanMetadata scan, const std::vector<double> &traj_t, const std::vector<helios::vec3> &traj_pos, const std::vector<helios::vec4> &traj_quat, const helios::vec3 &lever_arm, const helios::vec3 &boresight_rpy,
                       float pulse_rate_hz, double t0 = 0.0);

    //! Add a moving-platform (mobile/airborne) LiDAR scan with the orientation trajectory given as Euler angles
    /**
     * Convenience overload of \ref addScanMoving() that takes the per-sample platform orientation as roll/pitch/yaw
     * Euler angles (radians) instead of quaternions. Each \p traj_rpy entry is converted to a Hamilton body->world
     * quaternion using the same intrinsic Z-Y-X (yaw-pitch-roll) convention as \p boresight_rpy, then the scan is
     * registered exactly as the quaternion overload. Use this when hand-authoring a trajectory; prefer the quaternion
     * overload when the trajectory comes from an INS/IMU that natively reports quaternions (avoids a round-trip through
     * Euler angles and the associated gimbal-lock ambiguity).
     * \param[in] scan  Scan metadata defining the angular sampling grid and beam parameters (origin is ignored; the trajectory supplies position).
     * \param[in] traj_t  Monotonically increasing trajectory sample times in seconds (size M).
     * \param[in] traj_pos  Platform positions in world coordinates, one per \p traj_t entry (size M).
     * \param[in] traj_rpy  Platform orientations as roll/pitch/yaw Euler angles in radians (intrinsic Z-Y-X), one per \p traj_t entry (size M).
     * \param[in] lever_arm  Sensor optical center in the platform body frame (meters).
     * \param[in] boresight_rpy  Fixed sensor rotational misalignment as roll/pitch/yaw in radians (body frame).
     * \param[in] pulse_rate_hz  Pulse repetition rate in Hz (must be > 0); sets the time between consecutive pulses.
     * \param[in] t0  Time of the first pulse in seconds (relative time; defaults to 0).
     * \return ID for scan that was created
     */
    uint addScanMoving(ScanMetadata scan, const std::vector<double> &traj_t, const std::vector<helios::vec3> &traj_pos, const std::vector<helios::vec3> &traj_rpy, const helios::vec3 &lever_arm, const helios::vec3 &boresight_rpy,
                       float pulse_rate_hz, double t0 = 0.0);

    //! Add a continuously-spinning multibeam (Velodyne/Ouster/Hesai-style) scan driven by a 6-DOF platform trajectory
    /**
     * Sets up a rotating multi-channel sensor from its physical instrument parameters; Helios derives the internal
     * sampling grid, rotation rate, and revolution count rather than requiring the caller to hand-flatten them into an
     * Ntheta x Nphi grid. A spinning sensor rotates continuously through 360 degrees while the platform moves along the
     * trajectory, so there is no partial-arc azimuth range: the only azimuth control is the angular resolution
     * (\p azimuthStep_rad). The number of points is dictated by the pulse repetition rate and the total trajectory
     * duration: n_pulses = pulse_rate_hz * (traj_t.back() - traj_t.front()), distributed across the channels and the
     * derived number of azimuth steps.
     *
     * From the physical parameters Helios derives: steps_per_rev = round(2*pi / azimuthStep_rad);
     * rotation_rate = pulse_rate_hz / (channels * steps_per_rev); n_revolutions = rotation_rate * duration;
     * Nphi = round(steps_per_rev * n_revolutions); Ntheta = number of channels. The per-pulse timestamp, origin, and
     * orientation are produced exactly as in \ref addScanMoving(), and each pulse fires at the EXACT per-channel
     * elevation (not a resampled uniform grid).
     *
     * \note This is the only way to set up a spinning scan. A stationary "spin in place" capture (e.g. a tripod) is
     * expressed as a trajectory of two coincident poses with the same position and orientation, separated in time by the
     * desired acquisition duration; the duration determines the number of revolutions, exactly as for a moving capture.
     *
     * \param[in] beamElevationAngles  Per-channel beam elevation angles above the horizon, in radians (zenith = pi/2 - elevation). Its size sets the channel count. \note This is ELEVATION, unlike the zenith-angle \ref ScanMetadata spinning constructor.
     * \param[in] azimuthStep_rad  Azimuth angular resolution in radians per firing step (e.g. 0.2 degrees = 0.2*pi/180). Must be > 0.
     * \param[in] pulse_rate_hz  Pulse repetition rate (PRF) in Hz (must be > 0).
     * \param[in] traj_t  Monotonically increasing trajectory sample times in seconds (size M).
     * \param[in] traj_pos  Platform positions in world coordinates, one per \p traj_t entry (size M).
     * \param[in] traj_quat  Platform orientation quaternions (qx,qy,qz,qw), Hamilton body->world, one per \p traj_t entry (size M).
     * \param[in] lever_arm  Sensor optical center in the platform body frame (meters).
     * \param[in] boresight_rpy  Fixed sensor rotational misalignment as roll/pitch/yaw in radians (body frame).
     * \param[in] exitDiameter  Diameter of the laser pulse at exit from the scanner in meters.
     * \param[in] beamDivergence  Divergence angle of the laser beam in radians.
     * \param[in] rangeNoiseStdDev  Standard deviation of Gaussian range (along-beam) measurement noise in meters (0 disables).
     * \param[in] angleNoiseStdDev  Standard deviation of Gaussian angular (beam-pointing) jitter in radians (0 disables).
     * \param[in] columnFormat  Vector of strings specifying the columns of the scan ASCII file for input/output.
     * \param[in] t0  Time of the first pulse (pulse ordinal 0) in seconds (relative time; defaults to 0).
     * \return ID for scan that was created
     */
    uint addScanSpinning(const std::vector<float> &beamElevationAngles, float azimuthStep_rad, float pulse_rate_hz, const std::vector<double> &traj_t, const std::vector<helios::vec3> &traj_pos, const std::vector<helios::vec4> &traj_quat,
                         const helios::vec3 &lever_arm, const helios::vec3 &boresight_rpy, float exitDiameter, float beamDivergence, float rangeNoiseStdDev, float angleNoiseStdDev,
                         const std::vector<std::string> &columnFormat = {"x", "y", "z"}, double t0 = 0.0);

    //! Add a continuously-spinning multibeam scan with the orientation trajectory given as Euler angles
    /**
     * Convenience overload of \ref addScanSpinning() that takes the per-sample platform orientation as roll/pitch/yaw
     * Euler angles (radians, intrinsic Z-Y-X) instead of quaternions. Each \p traj_rpy entry is converted to a Hamilton
     * body->world quaternion, then the scan is registered exactly as the quaternion overload.
     * \param[in] beamElevationAngles  Per-channel beam elevation angles above the horizon, in radians.
     * \param[in] azimuthStep_rad  Azimuth angular resolution in radians per firing step. Must be > 0.
     * \param[in] pulse_rate_hz  Pulse repetition rate (PRF) in Hz (must be > 0).
     * \param[in] traj_t  Monotonically increasing trajectory sample times in seconds (size M).
     * \param[in] traj_pos  Platform positions in world coordinates, one per \p traj_t entry (size M).
     * \param[in] traj_rpy  Platform orientations as roll/pitch/yaw Euler angles in radians (intrinsic Z-Y-X), one per \p traj_t entry (size M).
     * \param[in] lever_arm  Sensor optical center in the platform body frame (meters).
     * \param[in] boresight_rpy  Fixed sensor rotational misalignment as roll/pitch/yaw in radians (body frame).
     * \param[in] exitDiameter  Diameter of the laser pulse at exit from the scanner in meters.
     * \param[in] beamDivergence  Divergence angle of the laser beam in radians.
     * \param[in] rangeNoiseStdDev  Standard deviation of Gaussian range (along-beam) measurement noise in meters (0 disables).
     * \param[in] angleNoiseStdDev  Standard deviation of Gaussian angular (beam-pointing) jitter in radians (0 disables).
     * \param[in] columnFormat  Vector of strings specifying the columns of the scan ASCII file for input/output.
     * \param[in] t0  Time of the first pulse (pulse ordinal 0) in seconds (relative time; defaults to 0).
     * \return ID for scan that was created
     */
    uint addScanSpinning(const std::vector<float> &beamElevationAngles, float azimuthStep_rad, float pulse_rate_hz, const std::vector<double> &traj_t, const std::vector<helios::vec3> &traj_pos, const std::vector<helios::vec3> &traj_rpy,
                         const helios::vec3 &lever_arm, const helios::vec3 &boresight_rpy, float exitDiameter, float beamDivergence, float rangeNoiseStdDev, float angleNoiseStdDev,
                         const std::vector<std::string> &columnFormat = {"x", "y", "z"}, double t0 = 0.0);

    //! Add a moving-platform raster scan: a fixed uniform angular fan swept while the platform moves along a trajectory
    /**
     * Convenience wrapper around \ref addScanMoving() for a non-spinning sensor on a moving platform. The caller specifies
     * the per-frame angular fan resolution (Ntheta x Nphi over [thetaMin,thetaMax] x [phiMin,phiMax]) plus the trajectory
     * and PRF, and Helios derives the per-pulse time sampling along the trajectory (t = t0 + ordinal/pulse_rate_hz). Unlike
     * the low-level \ref addScanMoving(), the caller does not pre-build a \ref ScanMetadata or compute pulse
     * counts to make the sweep span the flight. Sets the scan's \ref ScanMode to \ref SCAN_MODE_MOVING_RASTER.
     * \param[in] Ntheta  Number of zenith samples in the angular fan.
     * \param[in] thetaMin  Minimum zenith angle in radians.
     * \param[in] thetaMax  Maximum zenith angle in radians.
     * \param[in] Nphi  Number of azimuth samples in the angular fan.
     * \param[in] phiMin  Minimum azimuth angle in radians.
     * \param[in] phiMax  Maximum azimuth angle in radians.
     * \param[in] pulse_rate_hz  Pulse repetition rate (PRF) in Hz (must be > 0).
     * \param[in] traj_t  Monotonically increasing trajectory sample times in seconds (size M).
     * \param[in] traj_pos  Platform positions in world coordinates, one per \p traj_t entry (size M).
     * \param[in] traj_quat  Platform orientation quaternions (qx,qy,qz,qw), Hamilton body->world, one per \p traj_t entry (size M).
     * \param[in] lever_arm  Sensor optical center in the platform body frame (meters).
     * \param[in] boresight_rpy  Fixed sensor rotational misalignment as roll/pitch/yaw in radians (body frame).
     * \param[in] exitDiameter  Diameter of the laser pulse at exit from the scanner in meters.
     * \param[in] beamDivergence  Divergence angle of the laser beam in radians.
     * \param[in] rangeNoiseStdDev  Standard deviation of Gaussian range (along-beam) measurement noise in meters (0 disables).
     * \param[in] angleNoiseStdDev  Standard deviation of Gaussian angular (beam-pointing) jitter in radians (0 disables).
     * \param[in] columnFormat  Vector of strings specifying the columns of the scan ASCII file for input/output.
     * \param[in] t0  Time of the first pulse (pulse ordinal 0) in seconds (relative time; defaults to 0).
     * \return ID for scan that was created
     */
    uint addScanMovingRaster(uint Ntheta, float thetaMin, float thetaMax, uint Nphi, float phiMin, float phiMax, float pulse_rate_hz, const std::vector<double> &traj_t, const std::vector<helios::vec3> &traj_pos,
                             const std::vector<helios::vec4> &traj_quat, const helios::vec3 &lever_arm, const helios::vec3 &boresight_rpy, float exitDiameter, float beamDivergence, float rangeNoiseStdDev, float angleNoiseStdDev,
                             const std::vector<std::string> &columnFormat = {"x", "y", "z"}, double t0 = 0.0);

    //! Add a rotating-Risley-prism (Livox-style rosette) scan from physical instrument parameters
    /**
     * Sets up a non-repetitive rosette scan produced by a stack of continuously rotating wedge prisms (the optical mechanism
     * used by Livox rosette-pattern sensors such as the Mid-40, Mid-70, and Avia). Each pulse fires a single beam that is
     * refracted through the rotating prisms; with the prisms rotating at different (and generally incommensurate) rates the
     * beam traces a non-repetitive pattern that fills a circular field of view, denser toward the center. The body-frame beam
     * direction of each pulse is computed by full Snell's-law refraction through the prisms at that pulse's time, then composed
     * with the boresight and platform-trajectory orientation exactly as in \ref addScanMoving(). The scan is stored as an
     * Ntheta=1, Nphi=Npulses table (one direction per pulse), where Npulses = round(pulse_rate_hz * trajectory_duration). The
     * \ref ScanPattern is set to \ref SCAN_PATTERN_RISLEY_PRISM and the \ref ScanMode to \ref SCAN_MODE_RISLEY_PRISM.
     *
     * Like a spinning scan, a Risley-prism scan is always trajectory-driven. A stationary capture (e.g. on a tripod) is
     * expressed as a trajectory of two coincident poses with the same position and orientation, separated in time by the
     * desired acquisition duration; the duration determines how many pulses (and how much of the rosette) are collected.
     *
     * \note Only the rotating-Risley-prism rosette mechanism is modeled. Livox's deterministic line-scan mode and the newer
     *       non-Risley (MEMS/galvanometer) non-repetitive patterns are different mechanisms not covered by this entry point.
     * \note The field of view is an emergent property of the prism wedge angles and refractive indices; it is not specified
     *       directly. The beam direction is computed by non-paraxial ray tracing through the rotating wedges (vector-form
     *       Snell's law at each face).
     *
     * \param[in] prisms  Rotating wedge prisms in the order the beam passes through them (see \ref RisleyPrism). At least one is required; a Livox sensor uses two counter-rotating prisms.
     * \param[in] refractive_index_air  Refractive index of the medium surrounding the prisms (typically 1.0 for air).
     * \param[in] pulse_rate_hz  Pulse repetition rate (PRF) in Hz (must be > 0).
     * \param[in] traj_t  Monotonically increasing trajectory sample times in seconds (size M).
     * \param[in] traj_pos  Platform positions in world coordinates, one per \p traj_t entry (size M).
     * \param[in] traj_quat  Platform orientation quaternions (qx,qy,qz,qw), Hamilton body->world, one per \p traj_t entry (size M).
     * \param[in] lever_arm  Sensor optical center in the platform body frame (meters).
     * \param[in] boresight_rpy  Fixed sensor rotational misalignment as roll/pitch/yaw in radians (body frame).
     * \param[in] exitDiameter  Diameter of the laser pulse at exit from the scanner in meters.
     * \param[in] beamDivergence  Divergence angle of the laser beam in radians.
     * \param[in] rangeNoiseStdDev  Standard deviation of Gaussian range (along-beam) measurement noise in meters (0 disables).
     * \param[in] angleNoiseStdDev  Standard deviation of Gaussian angular (beam-pointing) jitter in radians (0 disables).
     * \param[in] columnFormat  Vector of strings specifying the columns of the scan ASCII file for input/output.
     * \param[in] t0  Time of the first pulse (pulse ordinal 0) in seconds (relative time; defaults to 0).
     * \return ID for scan that was created
     */
    uint addScanRisley(const std::vector<RisleyPrism> &prisms, double refractive_index_air, float pulse_rate_hz, const std::vector<double> &traj_t, const std::vector<helios::vec3> &traj_pos, const std::vector<helios::vec4> &traj_quat,
                       const helios::vec3 &lever_arm, const helios::vec3 &boresight_rpy, float exitDiameter, float beamDivergence, float rangeNoiseStdDev, float angleNoiseStdDev, const std::vector<std::string> &columnFormat = {"x", "y", "z"},
                       double t0 = 0.0);

    //! Add a rotating-Risley-prism (Livox-style rosette) scan with the orientation trajectory given as Euler angles
    /**
     * Convenience overload of \ref addScanRisley() that takes the per-sample platform orientation as roll/pitch/yaw Euler
     * angles (radians, intrinsic Z-Y-X) instead of quaternions; it converts each to a quaternion and delegates to the
     * quaternion overload.
     * \param[in] prisms  Rotating wedge prisms in the order the beam passes through them (see \ref RisleyPrism).
     * \param[in] refractive_index_air  Refractive index of the medium surrounding the prisms (typically 1.0 for air).
     * \param[in] pulse_rate_hz  Pulse repetition rate (PRF) in Hz (must be > 0).
     * \param[in] traj_t  Monotonically increasing trajectory sample times in seconds (size M).
     * \param[in] traj_pos  Platform positions in world coordinates, one per \p traj_t entry (size M).
     * \param[in] traj_rpy  Platform orientations as roll/pitch/yaw Euler angles in radians, one per \p traj_t entry (size M).
     * \param[in] lever_arm  Sensor optical center in the platform body frame (meters).
     * \param[in] boresight_rpy  Fixed sensor rotational misalignment as roll/pitch/yaw in radians (body frame).
     * \param[in] exitDiameter  Diameter of the laser pulse at exit from the scanner in meters.
     * \param[in] beamDivergence  Divergence angle of the laser beam in radians.
     * \param[in] rangeNoiseStdDev  Standard deviation of Gaussian range (along-beam) measurement noise in meters (0 disables).
     * \param[in] angleNoiseStdDev  Standard deviation of Gaussian angular (beam-pointing) jitter in radians (0 disables).
     * \param[in] columnFormat  Vector of strings specifying the columns of the scan ASCII file for input/output.
     * \param[in] t0  Time of the first pulse (pulse ordinal 0) in seconds (relative time; defaults to 0).
     * \return ID for scan that was created
     */
    uint addScanRisley(const std::vector<RisleyPrism> &prisms, double refractive_index_air, float pulse_rate_hz, const std::vector<double> &traj_t, const std::vector<helios::vec3> &traj_pos, const std::vector<helios::vec3> &traj_rpy,
                       const helios::vec3 &lever_arm, const helios::vec3 &boresight_rpy, float exitDiameter, float beamDivergence, float rangeNoiseStdDev, float angleNoiseStdDev, const std::vector<std::string> &columnFormat = {"x", "y", "z"},
                       double t0 = 0.0);

    //! Specify a scan point as a hit by providing the (x,y,z) coordinates and scan ray direction
    /**
     * \param[in] scanID ID of scan hit point to which hit point should be added.
     * \param[in] xyz (x,y,z) coordinates of hit point.
     * \param[in] direction Spherical coordinate corresponding to the scanner ray direction for the hit point.
     * \note If only the (row,column) scan table coordinates are available, use \ref ScanMetadata::rc2direction() to convert them to a spherical scan direction coordinate.
     */
    void addHitPoint(uint scanID, const helios::vec3 &xyz, const helios::SphericalCoord &direction);

    //! Specify a scan point as a hit by providing the (x,y,z) coordinates and scan ray direction
    /**
     * \param[in] scanID ID of scan hit point to which hit point should be added.
     * \param[in] xyz (x,y,z) coordinates of hit point.
     * \param[in] direction Spherical coordinate corresponding to the scanner ray direction for the hit point.
     * \param[in] color r-g-b color of the hit point
     * \note If only the (row,column) scan table coordinates are available, use \ref ScanMetadata::rc2direction() to convert them to a spherical scan direction coordinate.
     */
    void addHitPoint(uint scanID, const helios::vec3 &xyz, const helios::SphericalCoord &direction, const helios::RGBcolor &color);

    //! Specify a scan point as a hit by providing the (x,y,z) coordinates and scan ray direction
    /**
     * \param[in] scanID ID of scan hit point to which hit point should be added.
     * \param[in] xyz (x,y,z) coordinates of hit point.
     * \param[in] direction Spherical coordinate corresponding to the scanner ray direction for the hit point.
     * \param[in] data Map data structure containing floating point data values for the hit point. E.g., "reflectance" could be mapped to a value of 965.2.
     */
    void addHitPoint(uint scanID, const helios::vec3 &xyz, const helios::SphericalCoord &direction, const std::map<std::string, double> &data);

    //! Specify a scan point as a hit by providing the (x,y,z) coordinates and scan ray direction
    /**
     * \param[in] scanID ID of scan hit point to which hit point should be added.
     * \param[in] xyz (x,y,z) coordinates of hit point.
     * \param[in] direction Spherical coordinate corresponding to the scanner ray direction for the hit point.
     * \param[in] color r-g-b color of the hit point
     * \param[in] data Map data structure containing floating point data values for the hit point. E.g., "reflectance" could be mapped to a value of 965.2.
     */
    void addHitPoint(uint scanID, const helios::vec3 &xyz, const helios::SphericalCoord &direction, const helios::RGBcolor &color, const std::map<std::string, double> &data);

    //! Specify a scan point as a hit by providing the (x,y,z) coordinates and row,column in scan table
    /**
     * \param[in] scanID ID of scan hit point to which hit point should be added.
     * \param[in] xyz (x,y,z) coordinates of hit point.
     * \param[in] row_column row (theta index) and column (phi index) for point in scan table
     * \param[in] color r-g-b color of the hit point
     * \param[in] data Map data structure containing floating point data values for the hit point. E.g., "reflectance" could be mapped to a value of 965.2.
     */
    void addHitPoint(uint scanID, const helios::vec3 &xyz, const helios::int2 &row_column, const helios::RGBcolor &color, const std::map<std::string, double> &data);

    //! Delete a hit point in the scan
    /**
     * \param[in] index Index of hit point in the point cloud
     */
    void deleteHitPoint(uint index);

    //! Get the number of hit points in the point cloud
    uint getHitCount() const;

    //! Get the (x,y,z) scan origin
    /**
     * \param[in] scanID ID of scan.
     */
    helios::vec3 getScanOrigin(uint scanID) const;

    //! Get the number of scan points in the theta (zenithal) direction
    /**
     * \param[in] scanID ID of scan.
     */
    uint getScanSizeTheta(uint scanID) const;

    //! Get the number of scan points in the phi (azimuthal) direction
    /**
     * \param[in] scanID ID of scan.
     */
    uint getScanSizePhi(uint scanID) const;

    //! Get the range of scan directions in the theta (zenithal) direction
    /**
     * \param[in] scanID ID of scan.
     * \return vec2.x is the minimum scan zenithal angle, and vec2.y is the maximum scan zenithal angle, both in radians
     */
    helios::vec2 getScanRangeTheta(uint scanID) const;

    //! Get the range of scan directions in the phi (azimuthal) direction
    /**
     * \param[in] scanID ID of scan.
     * \return vec2.x is the minimum scan azimuthal angle, and vec2.y is the maximum scan azimuthal angle, both in radians
     */
    helios::vec2 getScanRangePhi(uint scanID) const;

    //! Get the diameter of the laser beam at exit from the instrument
    /**
     * \param[in] scanID ID of scan.
     * \return Diameter of the beam at exit.
     */
    float getScanBeamExitDiameter(uint scanID) const;

    //! Get the labels for columns in ASCII input/output file
    /**
     * \param[in] scanID ID of scan.
     */
    std::vector<std::string> getScanColumnFormat(uint scanID) const;

    //! Get the geometric beam pattern of a scan
    /**
     * \param[in] scanID ID of scan.
     * \return \ref SCAN_PATTERN_RASTER for a uniform angular grid, or \ref SCAN_PATTERN_SPINNING_MULTIBEAM for a rotating multi-channel sensor.
     */
    ScanPattern getScanPattern(uint scanID) const;

    //! Get the per-channel zenith angles of a spinning multibeam scan
    /**
     * \param[in] scanID ID of scan.
     * \return Vector of per-channel zenith angles in radians (one per row). Empty for a \ref SCAN_PATTERN_RASTER scan.
     */
    std::vector<float> getScanBeamZenithAngles(uint scanID) const;

    //! Get the high-level acquisition-mode descriptor of a scan
    /**
     * \param[in] scanID ID of scan.
     * \return \ref SCAN_MODE_STATIC_RASTER, \ref SCAN_MODE_MOVING_RASTER, \ref SCAN_MODE_SPINNING, or \ref SCAN_MODE_RISLEY_PRISM.
     */
    ScanMode getScanMode(uint scanID) const;

    //! Get the number of azimuth firing steps per full 360-degree revolution of a spinning multibeam scan
    /**
     * \param[in] scanID ID of scan.
     * \return Azimuth steps per revolution. 0 for scans that are not \ref SCAN_MODE_SPINNING.
     */
    uint getScanStepsPerRev(uint scanID) const;

    //! Get the sensor-head rotation rate of a spinning multibeam scan
    /**
     * \param[in] scanID ID of scan.
     * \return Rotation rate in revolutions per second (PRF / (channels * steps_per_rev)). 0 for non-spinning scans.
     */
    double getScanRotationRate(uint scanID) const;

    //! Get the total number of full 360-degree revolutions collected by a spinning multibeam scan
    /**
     * \param[in] scanID ID of scan.
     * \return Number of revolutions (may be fractional). 0 for non-spinning scans.
     */
    double getScanRevolutions(uint scanID) const;

    //! Get the rotating wedge prisms of a Risley-prism (Livox-style rosette) scan
    /**
     * \param[in] scanID ID of scan.
     * \return The prism stack in beam-traversal order (see \ref RisleyPrism). Empty for scans that are not \ref SCAN_MODE_RISLEY_PRISM.
     */
    std::vector<RisleyPrism> getScanRisleyPrisms(uint scanID) const;

    //! Get the refractive index of the medium surrounding the prisms of a Risley-prism scan
    /**
     * \param[in] scanID ID of scan.
     * \return Refractive index of air (typically 1.0). Returns 1.0 for non-Risley scans.
     */
    double getScanRisleyRefractiveIndexAir(uint scanID) const;

    //! Divergence angle of the laser beam in radians
    /**
     * \param[in] scanID ID of scan.
     * \return Divergence angle of the beam, in radians.
     */
    float getScanBeamDivergence(uint scanID) const;

    //! Standard deviation of Gaussian range (along-beam) measurement noise in meters
    /**
     * \param[in] scanID ID of scan.
     * \return Standard deviation of the range measurement noise applied during synthetic scan generation, in meters (0 if disabled).
     */
    float getScanRangeNoiseStdDev(uint scanID) const;

    //! Standard deviation of Gaussian angular (beam-pointing) jitter in radians
    /**
     * \param[in] scanID ID of scan.
     * \return Standard deviation of the beam-pointing jitter applied during synthetic scan generation, in radians (0 if disabled).
     */
    float getScanAngleNoiseStdDev(uint scanID) const;

    //! Get the return-reporting mode of a scan
    /**
     * \param[in] scanID ID of scan.
     * \return \ref RETURN_MODE_MULTI or \ref RETURN_MODE_SINGLE used during analytic-waveform synthetic scan generation.
     */
    ReturnMode getScanReturnMode(uint scanID) const;

    //! Set the return-reporting mode of a scan
    /**
     * \param[in] scanID ID of scan.
     * \param[in] returnMode \ref RETURN_MODE_MULTI to report all detected returns, or \ref RETURN_MODE_SINGLE for one return per pulse.
     */
    void setScanReturnMode(uint scanID, ReturnMode returnMode);

    //! Get the single-return selection policy of a scan
    /**
     * \param[in] scanID ID of scan.
     * \return The \ref SingleReturnSelection used when \ref getScanReturnMode is \ref RETURN_MODE_SINGLE.
     */
    SingleReturnSelection getScanSingleReturnSelection(uint scanID) const;

    //! Set the single-return selection policy of a scan
    /**
     * \param[in] scanID ID of scan.
     * \param[in] selection Which return to report in single-return mode (\ref SINGLE_RETURN_STRONGEST, \ref SINGLE_RETURN_FIRST, \ref SINGLE_RETURN_LAST, or \ref SINGLE_RETURN_STRONGEST_PLUS_LAST).
     */
    void setScanSingleReturnSelection(uint scanID, SingleReturnSelection selection);

    //! Get the maximum number of returns reported per pulse in single/limited-return mode
    /**
     * \param[in] scanID ID of scan.
     * \return Maximum returns per pulse used when \ref getScanReturnMode is \ref RETURN_MODE_SINGLE (1 = single-return, 2 = dual-return, N = N-return). Ignored in \ref RETURN_MODE_MULTI.
     */
    int getScanMaxReturns(uint scanID) const;

    //! Set the maximum number of returns reported per pulse in single/limited-return mode
    /**
     * \param[in] scanID ID of scan.
     * \param[in] maxReturns Maximum returns per pulse (must be >= 1): 1 = single-return, 2 = dual-return, N = N-return. The kept returns are the subset chosen by \ref setScanSingleReturnSelection. Ignored in \ref RETURN_MODE_MULTI.
     */
    void setScanMaxReturns(uint scanID, int maxReturns);

    //! Get the range resolution (transmit pulse range-extent) of a scan in meters
    /**
     * \param[in] scanID ID of scan.
     * \return Range resolution used to merge sub-ray hits into discrete returns, in meters (0 if the syntheticScan pulse_distance_threshold argument is used instead).
     */
    float getScanPulseWidth(uint scanID) const;

    //! Set the range resolution (transmit pulse range-extent) of a scan in meters
    /**
     * \param[in] scanID ID of scan.
     * \param[in] pulseWidth Range resolution in meters; surfaces closer than this merge into one return. 0 falls back to the syntheticScan pulse_distance_threshold argument.
     */
    void setScanPulseWidth(uint scanID, float pulseWidth);

    //! Get the detection threshold (minimum return energy fraction) of a scan
    /**
     * \param[in] scanID ID of scan.
     * \return Minimum return energy fraction below which a return is discarded (0 if suppression is disabled).
     */
    float getScanDetectionThreshold(uint scanID) const;

    //! Set the detection threshold (minimum return energy fraction) of a scan
    /**
     * \param[in] scanID ID of scan.
     * \param[in] detectionThreshold Minimum return energy fraction in [0,1]; returns weaker than this are discarded. 0 disables suppression.
     */
    void setScanDetectionThreshold(uint scanID, float detectionThreshold);

    //! Get the global scanner tilt roll angle for a scan
    /**
     * \param[in] scanID ID of scan.
     * \return Scanner tilt roll angle (rotation about the world x-axis) applied during synthetic scan generation, in radians (0 if level).
     */
    float getScanTiltRoll(uint scanID) const;

    //! Get the global scanner tilt pitch angle for a scan
    /**
     * \param[in] scanID ID of scan.
     * \return Scanner tilt pitch angle (rotation about the world y-axis) applied during synthetic scan generation, in radians (0 if level).
     */
    float getScanTiltPitch(uint scanID) const;

    //! Get the global scanner azimuth (heading) offset for a scan
    /**
     * \param[in] scanID ID of scan.
     * \return Scanner azimuth offset (right-hand rotation about the world z-axis) applied during synthetic scan generation, in radians (0 if none).
     */
    float getScanAzimuthOffset(uint scanID) const;

    //! Get (x,y,z) coordinate of hit point by index
    /**
     * \param [in] index Hit number
     */
    helios::vec3 getHitXYZ(uint index) const;

    //! Get the (x,y,z) origin from which the beam producing this hit point was emitted
    /**
     * For moving-platform scans (see \ref addScanMoving()) each hit stores its own per-pulse emission origin in the
     * data labels "origin_x"/"origin_y"/"origin_z"; this function returns that origin. For static scans (which store
     * no per-hit origin) it falls back to the single scan origin \ref getScanOrigin() of the hit's scan.
     * \param [in] index Hit number
     * \return (x,y,z) world-coordinate origin of the beam that produced this hit.
     */
    helios::vec3 getHitOrigin(uint index) const;

    //! Get ray direction of hit point in the scan based on its index
    /**
     * \param [in] index Hit number
     */
    helios::SphericalCoord getHitRaydir(uint index) const;

    //! Get floating point data value associated with a hit point.
    /**
     * \param[in] index Hit number.
     * \param[in] label Label of the data value (e.g., "reflectance").
     * \return Value of scalar data.
     */
    double getHitData(uint index, const char *label) const;

    //! Set floating point data value associated with a hit point.
    /**
     * \param[in] index Hit number.
     * \param[in] label Label of the data value (e.g., "reflectance").
     * \param[in] value Value of scalar data.
     */
    void setHitData(uint index, const char *label, double value);

    //! Check if scalar data exists for a hit point
    /**
     * \param[in] index Hit number.
     * \param[in] label Label of the data value (e.g., "reflectance").
     */
    bool doesHitDataExist(uint index, const char *label) const;

    //! Get the internal column index for a hit-data label.
    /**
     * Per-hit scalar data is stored column-wise (see \ref getHitDataColumn()). This returns the column
     * slot for a label, which is useful for repeated bulk access without re-resolving the label.
     * \param[in] label Label of the data value (e.g., "intensity").
     * \return Column index for the label, or -1 if the label has never been set on any hit.
     */
    int getHitDataColumnIndex(const char *label) const;

    //! Bulk-read a per-hit scalar field across all hits into a contiguous array.
    /**
     * This is the fast path for extracting a whole scalar field from a large cloud: it is a single
     * cache-linear pass over the field's storage column, equivalent in cost to \ref getHitXYZ over all
     * hits, rather than N separate \ref getHitData lookups. The output has one entry per hit, in hit-index
     * order; hits that have no value for the label receive `absent_value`.
     * \param[in] label Label of the data value (e.g., "intensity").
     * \param[out] data Filled with one value per hit (resized to \ref getHitCount()).
     * \param[in] absent_value Value written for hits that lack the label (default -9999, matching the
     *            sentinel used by ASCII export).
     */
    void getHitDataColumn(const char *label, std::vector<double> &data, double absent_value = -9999) const;

    //! Distance (m) at which a "miss" point is placed along its beam direction.
    /** A fired pulse that returns nothing (transmitted to the sky) is represented as a
     * point at this distance from the scan origin along the beam. The value is far beyond
     * any real target so misses are unambiguously classified as transmitted beams in the
     * leaf-area inversion. Shared by \ref gapfillMisses(), \ref syntheticScan(), and the
     * miss classification (see \ref isHitMiss()). This is the distance at which the miss
     * POINT is positioned in the cloud; it is distinct from \ref LIDAR_RAYTRACE_MISS_T,
     * the ray-tracer's internal no-hit parameter. */
    static constexpr float LIDAR_MISS_DISTANCE = 20000.f;

    //! Ray-tracer "no hit" parameter (m): the maximum ray length passed to the backend.
    /** A traced ray that intersects nothing returns this value as its hit distance t. The
     * synthetic-scan miss detection compares the returned t against this sentinel to decide
     * whether a beam hit a primitive. This is an internal ray-tracing threshold and is NOT
     * the distance at which a miss point is placed in the cloud (that is
     * \ref LIDAR_MISS_DISTANCE). Used by \ref performUnifiedRayTracing() and
     * \ref syntheticScan(). */
    static constexpr float LIDAR_RAYTRACE_MISS_T = 1001.f;

    //! Determine whether a hit point is a "miss" (a fired pulse that returned nothing)
    /**
     * A miss represents a laser beam transmitted through the scene to the sky. Misses are
     * stored as points placed along the beam direction at \ref LIDAR_MISS_DISTANCE.
     * \param[in] index Hit point index
     * \return True if the hit is flagged as a miss (per-hit `is_miss` data == 1), or, for
     *         legacy data lacking the flag, if its range reaches \ref LIDAR_MISS_DISTANCE.
     */
    bool isHitMiss(uint index) const;

    //! Determine whether the point cloud contains any miss points
    /**
     * Leaf-area inversion (\ref calculateLeafArea()) requires misses to count the beams
     * transmitted through each voxel.
     * \return True if at least one hit is a miss (see \ref isHitMiss()).
     */
    bool hasMisses() const;

    //! Get color of hit point
    /**
     * \param[in] index Hit number
     */
    helios::RGBcolor getHitColor(uint index) const;

    //! Get the scan with which a hit is associated
    /**
     * \param[in] index Hit number
     */
    int getHitScanID(uint index) const;

    //! Get the index of a scan point based on its row and column in the hit table
    /**
     * \param[in] scanID ID of scan.
     * \param[in] row Row in the 2D scan data table (elevation angle).
     * \param[in] column Column in the 2D scan data table (azimuthal angle).
     * \note If the point was not a hit, the function will return `-1'.
     */
    int getHitIndex(uint scanID, uint row, uint column) const;

    //! Get the grid cell in which the hit point resides
    /**
     * \param[in] index Hit number
     * \note If the point does not reside in any grid cells, this function returns `-1'.
     * \note Calling this function requires that the function calculateHitGridCell[*]() has been called previously.
     */
    int getHitGridCell(uint index) const;

    //! Set the grid cell in which the hit point resides
    /**
     * \param[in] index Hit number
     * \param[in] cell Cell number
     */
    void setHitGridCell(uint index, int cell);

    //! Apply a translation to all points in the point cloud
    /**
     * \param[in] shift Distance to translate in x-, y-, and z- direction
     */
    void coordinateShift(const helios::vec3 &shift);

    //! Apply a translation to all points in a given scan
    /**
     * \param[in] scanID ID of scan to be shifted
     * \param[in] shift Distance to translate in x-, y-, and z- direction
     */
    void coordinateShift(uint scanID, const helios::vec3 &shift);

    //! Rotate all points in the point cloud about the origin
    /**
     * \param[in] rotation Spherical rotation angle
     */
    void coordinateRotation(const helios::SphericalCoord &rotation);

    //! Rotate all points in the point cloud about the origin
    /**
     * \param[in] scanID ID of scan to be shifted
     * \param[in] rotation Spherical rotation angle
     */
    void coordinateRotation(uint scanID, const helios::SphericalCoord &rotation);

    //! Rotate all points in the point cloud about an arbitrary line
    /**
     * \param[in] rotation Rotation angle in radians
     * \param[in] line_base (x,y,z) coordinate of a point on the line about which points will be rotated
     * \param[in] line_direction Unit vector pointing in the direction of the line about which points will be rotated
     */
    void coordinateRotation(float rotation, const helios::vec3 &line_base, const helios::vec3 &line_direction);

    //! Get the number of triangles formed by the triangulation
    uint getTriangleCount() const;

    //! Number of candidate triangles the Delaunay pass produced in the most
    //! recent triangulateHitPoints() call, before edge-length/aspect/degenerate
    //! filtering. Zero if triangulation has not been run.
    std::size_t getTriangulationCandidateCount() const;

    //! Number of candidate triangles dropped because an edge exceeded Lmax in
    //! the most recent triangulateHitPoints() call.
    std::size_t getTriangulationDroppedByLmax() const;

    //! Number of candidate triangles dropped by the aspect-ratio test (and, for
    //! multi-return data, the adaptive separation-ratio test) in the most recent
    //! triangulateHitPoints() call. Triangles already dropped by Lmax are not
    //! double-counted here.
    std::size_t getTriangulationDroppedByAspect() const;

    //! Number of candidate triangles dropped because their computed area was
    //! degenerate (NaN) in the most recent triangulateHitPoints() call.
    std::size_t getTriangulationDroppedByDegenerate() const;

    //! Get hit point corresponding to first vertex of triangle
    /**
     * \param[in] index Triangulation index (0 thru Ntriangles-1)
     * \return Hit point index (0 thru Nhits-1)
     */
    Triangulation getTriangle(uint index) const;

    // ------- FILE I/O --------- //

    //! Read an XML file containing scan information
    /**
     * \param[in] filename Path to XML file
     */
    void loadXML(const char *filename);

    //! Read an XML file containing scan information
    /**
     * \param[in] filename Path to XML file
     * \param[in] load_grid_only if true only the voxel grid defined in the xml file will be loaded, the scans themselves will not be loaded.
     */
    void loadXML(const char *filename, bool load_grid_only);

    //! Load point cloud data from a tabular ASCII text file into an existing scan
    /**
     * \param[in] scanID  ID of the scan to which the point cloud data should be added.
     * \param[in]  ASCII_data_file  Metadata for point cloud data contained in the ASCII text file.
     * \return Number of points loaded from the file.
     */
    size_t loadASCIIFile(uint scanID, const std::string &ASCII_data_file);

    //! Export to file the normal vectors (nx,ny,nz) for all triangles formed
    /**
     * \param[in] filename Name of file
     */
    void exportTriangleNormals(const char *filename);

    //! Export to file the normal vectors (nx,ny,nz) for triangles formed within a single gridcell
    /**
     * \param[in] filename Name of file
     * \param[in] gridcell Index of gridcell to get triangles from
     */
    void exportTriangleNormals(const char *filename, int gridcell);

    //! Export to file the area of all triangles formed
    /**
     * \param[in] filename Name of file
     */
    void exportTriangleAreas(const char *filename);

    //! Export to file the area of all triangles formed within a single grid cell
    /**
     * \param[in] filename Name of file
     * \param[in] gridcell Index of gridcell to get triangles from
     */
    void exportTriangleAreas(const char *filename, int gridcell);

    //! Export to file discrete area-weighted inclination angle probability distribution based on the triangulation. Inclination angles are between 0 and 90 degrees. The probability distribution is normalized such that the sine-weighted integral over
    //! all angles is 1. The value of each bin is written as a column in the output file; lines correspond to each voxel grid cell.
    /**
     * \param[in] filename Name of file
     * \param[in] Nbins Number of bins to use for the histogram
     */
    void exportTriangleInclinationDistribution(const char *filename, uint Nbins);

    //! Export to file discrete azimuthal angle probability distribution based on the triangulation. Azimuthal angles are between 0 and 360 degrees. The probability distribution is normalized such that the integral over all angles is 1. The value of
    //! each bin is written as a column in the output file; lines correspond to each voxel grid cell.
    /**
     * \param[in] filename Name of file
     * \param[in] Nbins Number of bins to use for the histogram
     */
    void exportTriangleAzimuthDistribution(const char *filename, uint Nbins);

    //! Export to file the leaf area within each grid cell. Lines of the file correspond to each grid cell
    /**
     * \param[in] filename Name of file
     */
    void exportLeafAreas(const char *filename);

    //! Export to file the leaf area density within each grid cell. Lines of the file correspond to each grid cell
    /**
     * \param[in] filename Name of file
     */
    void exportLeafAreaDensities(const char *filename);

    //! Export to file the G(theta) value within each grid cell. Lines of the file correspond to each grid cell
    /**
     * \param[in] filename Name of file
     */
    void exportGtheta(const char *filename);

    //! Export to file the per-voxel leaf-area inversion sampling uncertainty. Lines of the file correspond to each grid cell
    /** Columns: cell_index leaf_area beam_count I_rdi LAD_std_error ci_valid. The standard error
        column reports the SAMPLING standard error of LAD [1/m] (sqrt of the per-voxel variance);
        undefined values are written as the sentinel -1. This is statistical sampling uncertainty
        conditional on the beams that entered each voxel and does NOT capture occlusion/coverage bias.
        \param[in] filename Name of file
     */
    void exportLeafAreaUncertainty(const char *filename);

    //! Export to file all points in the point cloud to an ASCII text file following the column format specified by the \<ASCII_format>\</ASCII_format> tag in the scan XML file
    /**
     * \param[in] filename Name of file
     * \param[in] write_header [optional] If true (default), a leading comment line beginning with '#' that lists the column field names is written at the top of each file. The loader ignores '#' comment lines, so headered files round-trip through
     * \ref loadXML().
     * \note If there are multiple scans in the point cloud, each scan will be exported to a different file with the scan ID appended to the filename. This is because different scans may have a different column format.
     */
    void exportPointCloud(const char *filename, bool write_header = true);

    //! Export to file all points from a given scan to an ASCII text file following the column format specified by the \<ASCII_format>\</ASCII_format> tag in the scan XML file
    /**
     * \param[in] filename Name of file
     * \param[in] scanID Identifier of scan to be exported
     * \param[in] write_header [optional] If true (default), a leading comment line beginning with '#' that lists the column field names is written at the top of the file. The loader ignores '#' comment lines, so headered files round-trip through
     * \ref loadXML().
     */
    void exportPointCloud(const char *filename, uint scanID, bool write_header = true);

    //! Export to file all points from a given scan to PTX file.
    /**
     * \param[in] filename Name of file
     * \param[in] scanID Identifier of scan to be exported
     */
    void exportPointCloudPTX(const char *filename, uint scanID);

    //! Export all scans in the point cloud to an XML metadata file plus one ASCII data file per scan
    /**
     * \param[in] filename Name of the XML metadata file to write (e.g. "output/scans.xml")
     * \note One ASCII point cloud data file is auto-generated per scan, named by stripping the extension of \p filename and appending "_\<scanID>.xyz". For example, passing "output/scans.xml" with three scans produces "output/scans_0.xyz",
     * "output/scans_1.xyz", and "output/scans_2.xyz" alongside the XML. The ASCII column format follows the per-scan \<ASCII_format>\</ASCII_format> tag, and the XML output can be re-loaded with \ref LiDARcloud::loadXML() when invoked from the same
     * working directory used at export time.
     */
    void exportScans(const char *filename);

    // ------- VISUALIZER --------- //

    //! Add all hit points to the visualizer plug-in, and color them by their r-g-b color
    /**
     * \param[in] visualizer Pointer to the Visualizer plugin object.
     * \param[in] pointsize Size of scan point in font points.
     */
    void addHitsToVisualizer(Visualizer *visualizer, uint pointsize) const;

    //! Add all hit points to the visualizer plug-in, and color them by a specified r-g-b color
    /**
     * \param[in] visualizer Pointer to the Visualizer plugin object.
     * \param[in] pointsize Size of scan point in font points.
     * \param[in] point_color r-g-b color of the hit points.
     */
    void addHitsToVisualizer(Visualizer *visualizer, uint pointsize, const helios::RGBcolor &point_color) const;

    //! Add all hit points to the visualizer plug-in, and color them by a hit scalar data value
    /**
     * \param[in] visualizer Pointer to the Visualizer plugin object.
     * \param[in] pointsize Size of scan point in font points.
     * \param[in] color_value Label for scalar hit data value to be used for coloring the points based on a pseudocolor mapping (e.g., "reflectance"). If the label does not exist, the function will print a warning and use the default color.
     */
    void addHitsToVisualizer(Visualizer *visualizer, uint pointsize, const char *color_value) const;

    //! Add all grid cells to the visualizer plug-in
    /**
     * \param[in] visualizer Pointer to the Visualizer plug-in object.
     */
    void addGridToVisualizer(Visualizer *visualizer) const;

    //! Add wire frame of the grid to the visualizer plug-in
    /**
     * \param[in] visualizer Pointer to the Visualizer plug-in object.
     * \param[in] linewidth_pixels Width of the wire frame lines in pixels (default = 1.0).
     */
    void addGridWireFrametoVisualizer(Visualizer *visualizer, float linewidth_pixels = 1.0f) const;

    //! Add a grid to point cloud instead of reading in from an xml file
    /**
     * \param[in] center center of the grid.
     * \param[in] size Size of the grid in each dimension.
     * \param[in] ndiv number of cells in the grid in each dimension.
     * \param[in] rotation horizontal rotation in degrees.
     */
    void addGrid(const helios::vec3 &center, const helios::vec3 &size, const helios::int3 &ndiv, float rotation);

    //! Add a grid to the point cloud with per-column vertical offsets (terrain following)
    /**
     * Identical to the four-argument overload, but each vertical column of voxels is shifted in z by a
     * per-column offset so the grid can follow a terrain surface (e.g. a DEM). Cell centers are stored
     * unrotated (rotation is applied downstream), so the offset is a pure vertical shift.
     * \param[in] center center of the grid.
     * \param[in] size Size of the grid in each dimension.
     * \param[in] ndiv number of cells in the grid in each dimension.
     * \param[in] rotation horizontal rotation in degrees.
     * \param[in] column_z_offsets per-(x,y)-column vertical offset, row-major as [j*ndiv.x + i], length ndiv.x*ndiv.y. Empty for no offset.
     */
    void addGrid(const helios::vec3 &center, const helios::vec3 &size, const helios::int3 &ndiv, float rotation, const std::vector<float> &column_z_offsets);

    //! Add all triangles to the visualizer plug-in, and color them by their r-g-b color
    /**
     * \param[in] visualizer Pointer to the Visualizer plug-in object.
     */
    void addTrianglesToVisualizer(Visualizer *visualizer) const;

    //! Add triangles within a given grid cell to the visualizer plug-in, and color them by their r-g-b color
    /**
     * \param[in] visualizer Pointer to the Visualizer plugin object.
     * \param[in] gridcell Index of grid cell.
     */
    void addTrianglesToVisualizer(Visualizer *visualizer, uint gridcell) const;

    //! Add reconstructed leaves (triangles or alpha masks) to the visualizer plug-in
    /**
     * \param[in] visualizer Pointer to the Visualizer plugin object.
     */
    void addLeafReconstructionToVisualizer(Visualizer *visualizer) const;

    //! Add trunk reconstruction to the visualizer plug-in.  Colors reconstructed triangles by hit point color.
    /**
     * \param[in] visualizer Pointer to the Visualizer plugin object.
     */
    void addTrunkReconstructionToVisualizer(Visualizer *visualizer) const;

    //! Add trunk reconstruction to the visualizer plug-in
    /**
     * \param[in] visualizer Pointer to the Visualizer plugin object.
     * \param[in] trunk_color r-g-b color of trunk.
     */
    void addTrunkReconstructionToVisualizer(Visualizer *visualizer, const helios::RGBcolor &trunk_color) const;

    //! Add reconstructed leaves (texture-masked patches) to the Context
    /**
     * \param[in] context Pointer to the Helios context
     * \note This function creates the following primitive data for each patch 1) "gridCell" which indicates the index of the gridcell that contains the patch, 2) "directFlag" which equals 1 if the leaf was part of the direct reconstruction, and 0 if
     * the leaf was backfilled.
     */
    std::vector<uint> addLeafReconstructionToContext(helios::Context *context) const;

    //! Add reconstructed leaves (texture-masked patches) to the Context with leaves divided into sub-patches (tiled)
    /**
     * \param[in] context Pointer to the Helios context
     * \param[in] subpatches Number of leaf sub-patches (tiles) in the x- and y- directions.
     * \note This function creates the following primitive data for each patch 1) "gridCell" which indicates the index of the gridcell that contains the patch, 2) "directFlag" which equals 1 if the leaf was part of the direct reconstruction, and 0 if
     * the leaf was backfilled.
     */
    std::vector<uint> addLeafReconstructionToContext(helios::Context *context, const helios::int2 &subpatches) const;

    //! Add triangle groups used in the direct reconstruction to the Context
    /**
     * \param[in] context Pointer to the Helios context.
     * \note This function creates primitive data called "leafGroup" which provides an identifier for each triangle based on the fill group it is in.
     */
    std::vector<uint> addReconstructedTriangleGroupsToContext(helios::Context *context) const;

    //! Add reconstructed trunk triangles to the Context
    /**
     * \param[in] context Pointer to the Helios context
     */
    std::vector<uint> addTrunkReconstructionToContext(helios::Context *context) const;

    //! Form an axis-aligned bounding box for all hit points in the point cloud
    /**
     * \param[out] boxmin Coordinates of the bounding box vertex in the (-x,-y,-z) direction.
     * \param[out] boxmax Coordinates of the bounding box vertex in the (+x,+y,+z) direction.
     */
    void getHitBoundingBox(helios::vec3 &boxmin, helios::vec3 &boxmax) const;

    //! Form an axis-aligned bounding box for all grid cells in the point cloud
    /**
     * \param[out] boxmin Coordinates of the bounding box vertex in the (-x,-y,-z) direction.
     * \param[out] boxmax Coordinates of the bounding box vertex in the (+x,+y,+z) direction.
     */
    void getGridBoundingBox(helios::vec3 &boxmin, helios::vec3 &boxmax) const;

    // --------- POINT FILTERING ----------- //

    //! Filter scan by imposing a maximum distance from the scanner
    /**
     * \param[in] maxdistance Maximum hit point distance from scanner
     */
    void distanceFilter(float maxdistance);

    //! overloaded version of xyzFilter that defaults to deleting points outside the provided bounding box
    /**
     * \param[in] xmin minimum x coordinate of bounding box
     * \param[in] xmax maximum x coordinate of bounding box
     * \param[in] ymin minimum y coordinate of bounding box
     * \param[in] ymax maximum y coordinate of bounding box
     * \param[in] zmin minimum z coordinate of bounding box
     * \param[in] zmax maximum z coordinate of bounding box
     * \note points outside the provided bounding box are deleted by default
     */
    void xyzFilter(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax);

    //! Filter scan with a bounding box
    /**
     * \param[in] xmin minimum x coordinate of bounding box
     * \param[in] xmax maximum x coordinate of bounding box
     * \param[in] ymin minimum y coordinate of bounding box
     * \param[in] ymax maximum y coordinate of bounding box
     * \param[in] zmin minimum z coordinate of bounding box
     * \param[in] zmax maximum z coordinate of bounding box
     * \param[in] deleteOutside if true, deletes points outside the bounding box, if false deletes points inside the bounding box
     * \note points outside the provided bounding box are deleted
     */
    void xyzFilter(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, bool deleteOutside);


    //! Filter scan by imposing a minimum reflectance value
    /**
     * \param[in] minreflectance Miniimum hit point reflectance value
     * \note If `reflectance' data was not provided for a hit point when calling \ref LiDARcloud::addHitPoint(), the point will not be filtered.
     */
    void reflectanceFilter(float minreflectance);

    //! Filter hit points based on a scalar field given by a column in the ASCII data
    /**
     * \param[in] scalar_field Name of a scalar field defined in the ASCII point cloud data (e.g., "reflectance")
     * \param[in] threshold Value for filter threshold
     * \param[in] comparator Points will be filtered if "scalar (comparator) threshold", where (comparator) is one of ">", "<", or "="
     * \note As an example, imagine we wanted to remove all hit points where the reflectance is less than -10. In this case we would call scalarFilter( "reflectance", -10, "<" );
     */
    void scalarFilter(const char *scalar_field, float threshold, const char *comparator);

    //! Filter multi-return data according to the maximum scalar value along each pulse. Any scalar value can be used, provided it is a field in the hit point data file. The resulting point cloud will have only one hit point per laser pulse.
    /**
     * \param[in] scalar Name of hit point scalar data in the hit data file.
     * \note This function is only applicable for multi-return data and requires that the scalar field "timestamp" is provided in the ASCII hit point data file.
     */
    void maxPulseFilter(const char *scalar);

    //! Filter multi-return data according to the minimum scalar value along each pulse. Any scalar value can be used, provided it is a field in the hit point data file. The resulting point cloud will have only one hit point per laser pulse.
    /**
     * \param[in] scalar Name of hit point scalar data in the ASCII hit data file.
     * \note This function is only applicable for multi-return data and requires that the scalar field "timestamp" is provided in the hit point data file.
     */
    void minPulseFilter(const char *scalar);

    //! Filter multi-return data to include only the first hit per laser pulse. The resulting point cloud will have only one hit point per laser pulse (first hits).
    /**
     * \note This function is only applicable for multi-return data and requires that the scalar field "target_index" is provided in the hit point data file. The "target_index" values can start at 0 or 1 for first hits as long as it is consistent
     * throughout the point cloud.
     */
    void firstHitFilter();

    //! Filter multi-return data to include only the last hit per laser pulse. The resulting point cloud will have only one hit point per laser pulse (last hits).
    /**
     * \note This function is only applicable for multi-return data and requires that the scalar fields "target_index" and "target_count" are provided in the hit point data file. The "target_index" values can start at 0 or 1 for first hits as long
     * as it is consistent throughout the point cloud.
     */
    void lastHitFilter();

    // ------- TRIANGULATION --------- //

    //! Perform triangulation on all hit points in point cloud
    /**
     * \param[in] Lmax Maximum allowable length of triangle sides.
     * \param[in] max_aspect_ratio Maximum allowable aspect ratio of triangles.
     * \note This call honors the cancellation flag registered via \ref setCancelFlag(): if the flag becomes non-zero (set from another thread) the triangulation aborts, discards any partial mesh, and returns an empty triangulation
     * (\ref getTriangleCount() == 0) rather than running to completion.
     */
    void triangulateHitPoints(float Lmax, float max_aspect_ratio);

    // ERK
    //! Perform triangulation on hit points in point cloud that meet some filtering criteria based on scalar data
    /**
     * \param[in] Lmax Maximum allowable length of triangle sides.
     * \param[in] max_aspect_ratio Maximum allowable aspect ratio of triangles.
     * \param[in] scalar_field Name of a scalar field defined in the ASCII point cloud data (e.g., "deviation")
     * \param[in] threshold Value for filter threshold
     * \param[in] comparator Points will not be used in triangulation if "scalar (comparator) threshold", where (comparator) is one of ">", "<", or "="
     * \note As an example, imagine we wanted to remove all hit points where the deviation is greater than 15 for the purposes of the triangulation. In this case we would call triangulateHitPoints(Lmax, max_aspect_ratio, "deviation", 15, ">" );
     * \note This call honors the cancellation flag registered via \ref setCancelFlag(): if the flag becomes non-zero (set from another thread) the triangulation aborts, discards any partial mesh, and returns an empty triangulation
     * (\ref getTriangleCount() == 0) rather than running to completion.
     */
    void triangulateHitPoints(float Lmax, float max_aspect_ratio, const char *scalar_field, float threshold, const char *comparator);

    //! Replace the internal triangulation with an externally-supplied world-space mesh
    /**
     * Bypasses the internal Constrained-Delaunay triangulation so a mesh produced elsewhere
     * (e.g. a re-used Helios triangulation, or a per-scan open3d Ball-Pivot mesh) can drive
     * leaf-area inversion. Leaf-area inversion only consumes triangulation through the per-voxel
     * G(theta) leaf-angle term (see \ref calculateLeafArea()), which needs each triangle's three
     * vertices, its source scan (to recover the ray zenith from \ref getScanOrigin()), and the
     * grid cell its centroid falls in -- nothing about the triangulation topology. This method
     * supplies exactly that, then sets the triangulation-computed flag so \ref calculateLeafArea()
     * runs unchanged.
     *
     * A grid must already be defined (see \ref addGrid()). Each triangle's grid cell is determined
     * by centroid containment; triangles whose centroid lies outside every cell are kept but
     * contribute to no cell (same as the internal path). Degenerate (zero/NaN-area) triangles are
     * dropped. Any previously-computed triangulation is discarded.
     *
     * \param[in] triangle_vertices Flat list of triangle vertices in world coordinates, three
     *            consecutive entries (v0, v1, v2) per triangle. Size must be a multiple of 3.
     * \param[in] scanIDs Source scan index for each triangle (size = triangle_vertices.size()/3).
     *            Used to recover the ray direction for G(theta); every entry must be a valid scan
     *            index in [0, getScanCount()). Per-scan provenance is required -- a merged mesh
     *            with no scan association is not a valid input for leaf-area inversion.
     */
    void setExternalTriangulation(const std::vector<helios::vec3> &triangle_vertices, const std::vector<int> &scanIDs);


    //! Add triangle geometry to Helios context
    /**
     * \param[in] context Pointer to Helios context
     */
    void addTrianglesToContext(helios::Context *context) const;

    // -------- GRID ----------- //

    //! Get the number of cells in the grid
    uint getGridCellCount() const;

    //! Add a cell to the grid
    /**
     * \param[in] center (x,y,z) coordinate of grid center.
     * \param[in] size size of the grid cell in the x,y,z directions.
     * \param[in] rotation rotation angle (in radians) of the grid cell about the z-axis.
     */
    void addGridCell(const helios::vec3 &center, const helios::vec3 &size, float rotation);

    //! Add a cell to the grid, where the cell is part of a larger global rectangular grid
    /**
     * \param[in] center (x,y,z) coordinate of grid center
     * \param[in] global_anchor (x,y,z) coordinate of grid global anchor, i.e., this is the 'center' coordinate entered in the xml file.  If grid Nx=Ny=Nz=1, global_anchor=center
     * \param[in] size size of the grid cell in the x,y,z directions
     * \param[in] global_size size of the global grid in the x,y,z directions
     * \param[in] rotation rotation angle (in radians) of the grid cell about the z-axis
     * \param[in] global_ijk index within the global grid in the x,y,z directions
     * \param[in] global_count total number of cells in global grid in the x,y,z directions
     */
    void addGridCell(const helios::vec3 &center, const helios::vec3 &global_anchor, const helios::vec3 &size, const helios::vec3 &global_size, float rotation, const helios::int3 &global_ijk, const helios::int3 &global_count);

    //! Get the (x,y,z) coordinate of a grid cell by its index
    /**
     * \param[in] index Index of a grid cell.  Note: the index of a grid cell is given by the order in which it was added to the grid. E.g., the first cell's index is 0, and the last cell's index is Ncells-1.
     */
    helios::vec3 getCellCenter(uint index) const;

    //! Get the (x,y,z) coordinate of a grid global anchor by its index
    /**
     * \param[in] index Index of a grid cell.  Note: the index of a grid cell is given by the order in which it was added to the grid. E.g., the first cell's index is 0, and the last cell's index is Ncells-1.
     */
    helios::vec3 getCellGlobalAnchor(uint index) const;

    //! Get the size of a grid cell by its index
    /**
     * \param[in] index Index of a grid cell.  Note: the index of a grid cell is given by the order in which it was added to the grid. E.g., the first cell's index is 0, an
  d the last cell's index is Ncells-1.
     */
    helios::vec3 getCellSize(uint index) const;

    //! Get the rotation angle of a grid cell about the z-axis by its index
    /**
     * \param[in] index Index of a grid cell.  Note: the index of a grid cell is given by the order in which it was added to the grid. E.g., the first cell's index is 0, and the last cell's index is Ncells-1.
     * \return Rotation angle of the cell about the z-axis, in radians.
     */
    float getCellRotation(uint index) const;

    // ------- SYNTHETIC SCAN ------ //

    //! Run a single-return synthetic LiDAR scan based on scan parameters given in an XML file, returning one laser hit per pulse
    /**
     * \param[in] context Pointer to the Helios context
     * \note This overload does NOT record miss points (transmitted beams). The resulting cloud cannot be used with calculateLeafArea(), which requires misses. To record misses, use the overload with the record_misses argument: syntheticScan(context,
     * scan_grid_only, record_misses).
     * \note Any non-standard label listed in the scan column format is used to label each hit point with that scalar data field of the intersected primitive. The value is taken from the primitive's own primitive data when present, otherwise from
     * the object data of the primitive's parent compound object (per-primitive data takes precedence). See \ref LiDARsynthetic.
     */
    void syntheticScan(helios::Context *context);

    //! Run a single-return synthetic LiDAR scan based on scan parameters given in an XML file, returning one laser hit per pulse
    /**
     * \param[in] context Pointer to the Helios context
     * \param[in] append If true, new hit points are appended to existing data. If false, existing hit points are cleared before adding new ones.
     */
    void syntheticScan(helios::Context *context, bool append);

    //! Run a single-return synthetic LiDAR scan based on scan parameters given in an XML file, returning one laser hit per pulse
    /**
     * \param[in] context Pointer to the Helios context.
     * \param[in] scan_grid_only If true, only record hit points for rays that intersect the voxel grid.
     * \param[in] record_misses If true, "miss" points (i.e., beam did not hit any primitives) are recorded in the scan.
     * \note Calling syntheticScan() with scan_grid_only=true can save substantial memory for contexts with large domains.
     */
    void syntheticScan(helios::Context *context, bool scan_grid_only, bool record_misses);

    //! Run a single-return synthetic LiDAR scan based on scan parameters given in an XML file, returning one laser hit per pulse
    /**
     * \param[in] context Pointer to the Helios context.
     * \param[in] scan_grid_only If true, only record hit points for rays that intersect the voxel grid.
     * \param[in] record_misses If true, "miss" points (i.e., beam did not hit any primitives) are recorded in the scan.
     * \param[in] append If true, new hit points are appended to existing data. If false, existing hit points are cleared before adding new ones.
     * \note Calling syntheticScan() with scan_grid_only=true can save substantial memory for contexts with large domains.
     */
    void syntheticScan(helios::Context *context, bool scan_grid_only, bool record_misses, bool append);

    //! Run a multi-return synthetic LiDAR scan based on scan parameters given in an XML file, returning multiple laser hits per pulse
    /**
     * \param[in] context Pointer to the Helios context.
     * \param[in] rays_per_pulse Number of ray launches per laser pulse direction.
     * \param[in] pulse_distance_threshold Threshold distance for determining laser hit locations. Hits within pulse_distance_threshold of each other will be grouped into a single hit.
     * \note Calling syntheticScan() with rays_per_pulse=1 will effectively run a single-return synthetic scan.
     * \note This overload does NOT record miss points (transmitted beams). The resulting cloud cannot be used with calculateLeafArea(), which requires misses. To record misses, use the overload with the record_misses argument: syntheticScan(context,
     * rays_per_pulse, pulse_distance_threshold, scan_grid_only, record_misses).
     */
    void syntheticScan(helios::Context *context, int rays_per_pulse, float pulse_distance_threshold);

    //! Run a multi-return synthetic LiDAR scan based on scan parameters given in an XML file, returning multiple laser hits per pulse
    /**
     * \param[in] context Pointer to the Helios context.
     * \param[in] rays_per_pulse Number of ray launches per laser pulse direction.
     * \param[in] pulse_distance_threshold Threshold distance for determining laser hit locations. Hits within pulse_distance_threshold of each other will be grouped into a single hit.
     * \param[in] append If true, new hit points are appended to existing data. If false, existing hit points are cleared before adding new ones.
     * \note Calling syntheticScan() with rays_per_pulse=1 will effectively run a single-return synthetic scan.
     */
    void syntheticScan(helios::Context *context, int rays_per_pulse, float pulse_distance_threshold, bool append);

    //! Run a multi-return synthetic LiDAR scan based on scan parameters given in an XML file, returning multiple laser hits per pulse
    /**
     * \param[in] context Pointer to the Helios context.
     * \param[in] rays_per_pulse Number of ray launches per laser pulse direction.
     * \param[in] pulse_distance_threshold Threshold distance for determining laser hit locations. Hits within pulse_distance_threshold of each other will be grouped into a single hit.
     * \param[in] scan_grid_only If true, only considers context geometry within the scan grid. scan_grid_only=true can save substantial memory for contexts with large domains.
     * \param[in] record_misses If true, "miss" points (i.e., beam did not hit any primitives) are recorded in the scan.
     * \note Calling syntheticScan() with rays_per_pulse=1 will effectively run a single-return synthetic scan.
     */
    void syntheticScan(helios::Context *context, int rays_per_pulse, float pulse_distance_threshold, bool scan_grid_only, bool record_misses);

    //! Run a multi-return synthetic LiDAR scan based on scan parameters given in an XML file, returning multiple laser hits per pulse
    /**
     * \param[in] context Pointer to the Helios context.
     * \param[in] rays_per_pulse Number of ray launches per laser pulse direction.
     * \param[in] pulse_distance_threshold Threshold distance for determining laser hit locations. Hits within pulse_distance_threshold of each other will be grouped into a single hit.
     * \param[in] scan_grid_only If true, only considers context geometry within the scan grid. scan_grid_only=true can save substantial memory for contexts with large domains.
     * \param[in] record_misses If true, "miss" points (i.e., beam did not hit any primitives) are recorded in the scan.
     * \param[in] append If true, new hit points are appended to existing data. If false, existing hit points are cleared before adding new ones.
     * \note Calling syntheticScan() with rays_per_pulse=1 will effectively run a single-return synthetic scan.
     */
    void syntheticScan(helios::Context *context, int rays_per_pulse, float pulse_distance_threshold, bool scan_grid_only, bool record_misses, bool append);

    //! Run a synthetic LiDAR scan with an explicit return-reporting mode (analytic-waveform processing)
    /**
     * Fires \p rays_per_pulse sub-rays per pulse and forms an analytic (sum-of-Gaussians) waveform whose detected returns are
     * reported according to \p return_mode. With \ref RETURN_MODE_MULTI all detected returns are reported; with
     * \ref RETURN_MODE_SINGLE up to the scan's \ref getScanMaxReturns returns per pulse are reported (see
     * \ref setScanSingleReturnSelection), and (in single-return mode, maxReturns=1) two surfaces within the pulse
     * range-resolution blend into one return at an intermediate range (a "ghost"/"mixed pixel" point). The range-resolution
     * used to merge returns is the scan's \ref getScanPulseWidth when set, otherwise \p pulse_distance_threshold; the noise
     * floor is the scan's \ref getScanDetectionThreshold. This overrides each scan's stored \ref getScanReturnMode for this
     * call only (the stored value is restored afterward); the per-scan \ref getScanMaxReturns and selection policy still apply.
     * \param[in] context Pointer to the Helios context.
     * \param[in] rays_per_pulse Number of ray launches per laser pulse direction. A value of 1 produces an idealized exact-intersection scan regardless of \p return_mode.
     * \param[in] pulse_distance_threshold Range-resolution distance used to merge sub-ray hits into returns when the scan's pulse width is 0. Hits within this distance merge into one return.
     * \param[in] return_mode \ref RETURN_MODE_MULTI to report all detected returns, or \ref RETURN_MODE_SINGLE for one return per pulse.
     * \param[in] scan_grid_only If true, only considers context geometry within the scan grid. [optional]
     * \param[in] record_misses If true, "miss" points (beam did not hit any primitives) are recorded in the scan. [optional]
     * \param[in] append If true, new hit points are appended to existing data; if false, existing hit points are cleared first. [optional]
     */
    void syntheticScan(helios::Context *context, int rays_per_pulse, float pulse_distance_threshold, ReturnMode return_mode, bool scan_grid_only = false, bool record_misses = false, bool append = true);

    //! Set the soft memory budget (in bytes) for the transient ray-tracing buffers used during \ref syntheticScan.
    /**
     * \ref syntheticScan fans each laser pulse out into rays_per_pulse sub-rays; for a large multi-return scan the
     * total number of simultaneously-traced sub-rays (beams x rays_per_pulse) can demand tens of gigabytes if traced in
     * one batch. To bound this, the per-scan beam fan-out is processed in chunks sized so that the live trace buffers stay
     * near this budget, independent of the scan resolution. Larger rays_per_pulse automatically yields fewer beams per
     * chunk. The budget bounds only the transient scratch buffers, not the output point cloud (which grows with the number
     * of recorded returns). A very small budget is clamped up internally so each chunk still contains at least one beam and
     * stays large enough for efficient batched ray tracing.
     *
     * If never called, the budget is automatic and path-dependent: \ref SYNTHETIC_SCAN_DEFAULT_BUDGET_GPU (8 GiB) on a
     * GPU build and \ref SYNTHETIC_SCAN_DEFAULT_BUDGET_CPU (4 GiB) otherwise. Call this to override that with a fixed cap
     * (typically to lower peak memory on a constrained host).
     * \param[in] bytes Soft cap in bytes on the live ray-tracing scratch buffers. Must be > 0.
     */
    void setSyntheticScanMemoryBudget(size_t bytes);

    //! Get the soft memory budget (in bytes) for the transient ray-tracing buffers used during \ref syntheticScan.
    /**
     * \return The explicitly configured budget in bytes, or 0 if using the automatic path-dependent default (8 GiB on a
     * GPU build, 4 GiB otherwise; see \ref setSyntheticScanMemoryBudget).
     */
    [[nodiscard]] size_t getSyntheticScanMemoryBudget() const;

    //! Calculate the surface area of all primitives in the context
    /**
     * \param[in] context Pointer to the Helios context
     */
    std::vector<float> calculateSyntheticLeafArea(helios::Context *context);

    //! Calculate the G(theta) of all primitives in the context
    /**
     * \param[in] context Pointer to the Helios context
     */
    std::vector<float> calculateSyntheticGtheta(helios::Context *context);

    // -------- LEAF AREA -------- //

    //! Set the leaf area of a grid cell in m^2
    /**
     * \param[in] area Leaf area in cell in m^2.
     * \param[in] index Index of a grid cell.  Note: the index of a grid cell is given by the order in which it was added to the grid. E.g., the first cell's index is 0, and the last cell's index is Ncells-1.
     */
    void setCellLeafArea(float area, uint index);

    //! Get the leaf area of a grid cell in m^2
    /**
     * \param[in] index Index of a grid cell.  Note: the index of a grid cell is given by the order in which it was added to the grid. E.g., the first cell's index is 0, and the last cell's index is Ncells-1.
     */
    float getCellLeafArea(uint index) const;

    //! Get the leaf area density of a grid cell in 1/m
    /**
     * \param[in] index Index of a grid cell.  Note: the index of a grid cell is given by the order in which it was added to the grid. E.g., the first cell's index is 0, and the last cell's index is Ncells-1.
     */
    float getCellLeafAreaDensity(uint index) const;

    //! Set the average G(theta) value of a grid cell
    /**
     * \param[in] Gtheta G(theta) in cell.
     * \param[in] index Index of a grid cell.  Note: the index of a grid cell is given by the order in which it was added to the grid. E.g., the first cell's index is 0, and the last cell's index is Ncells-1.
     */
    void setCellGtheta(float Gtheta, uint index);

    //! Get the G(theta) of a grid cell
    /**
     * \param[in] index Index of a grid cell.  Note: the index of a grid cell is given by the order in which it was added to the grid. E.g., the first cell's index is 0, and the last cell's index is Ncells-1.
     */
    float getCellGtheta(uint index) const;

    // -------- LEAF AREA INVERSION UNCERTAINTY -------- //
    //
    // The following accessors expose the per-voxel statistical SAMPLING uncertainty of the
    // leaf-area inversion (Pimont et al. 2018, RSE 215:343-370). This is the uncertainty owing
    // to the finite number of beams that sampled the voxel and to vegetation-element position
    // variability. It is CONDITIONAL on the beams that entered the voxel and does NOT capture
    // occlusion/coverage bias (voxels shadowed so that beams never penetrate). The group form
    // (\ref getGroupLADConfidenceInterval()) is the recommended path: single-voxel intervals are
    // routinely +-50-100%, whereas group intervals (a vertical slice, a whole plant) are +-5-10%.

    //! Get the number of beams that entered a grid cell during the leaf-area inversion
    /**
     * \param[in] index Index of a grid cell.
     * \return Beam count N, or -1 if \ref calculateLeafArea() has not been run for this cell.
     */
    int getCellBeamCount(uint index) const;

    //! Get the relative density index of a grid cell
    /** The relative density index is RDI = 1 - P, where P is the transmission probability.
     * \param[in] index Index of a grid cell.
     * \return RDI between 0 and 1 (the fraction of beams intercepted within the voxel).
     */
    float getCellRelativeDensityIndex(uint index) const;

    //! Get the mean beam path length through a grid cell in meters
    /**
     * \param[in] index Index of a grid cell.
     * \return Mean per-beam path length through the voxel [m].
     */
    float getCellMeanPathLength(uint index) const;

    //! Get the sampling variance of leaf area density for a grid cell
    /**
     * \param[in] index Index of a grid cell.
     * \return Sampling variance of LAD in (1/m)^2, or -1 if undefined (too few beams, etc.).
     */
    float getCellLADVariance(uint index) const;

    //! Get the single-voxel sampling confidence interval on leaf area
    /** Returns the interval centered on the cell's leaf-area point estimate. Returns false (no
        interval written) when the voxel falls outside the Pimont (2018) Table-3 validity envelope,
        rather than emitting an untrustworthy interval.
        \param[in] index Index of a grid cell.
        \param[in] confidence_level Confidence level in (0,1), e.g. 0.95.
        \param[out] lower Lower bound of the leaf-area confidence interval [m^2].
        \param[out] upper Upper bound of the leaf-area confidence interval [m^2].
        \return True if a valid confidence interval was produced, false otherwise.
     */
    bool getCellLeafAreaConfidenceInterval(uint index, float confidence_level, float &lower, float &upper) const;

    //! Get the group-scale sampling confidence interval on mean leaf area density - the recommended path
    /** Computes the confidence interval on the mean LAD over a set of voxels, assuming voxel
        independence (Pimont et al. 2018, Eq. 39): mean_LAD +- z * sqrt(sum(sigma^2)) / n_v.
        Voxels outside the Table-3 validity envelope are skipped (not counted in n_v).
        \param[in] indices Indices of the grid cells in the group.
        \param[in] confidence_level Confidence level between 0 and 1, e.g. 0.95.
        \param[out] mean_lad Mean leaf area density over the valid voxels in the group, in 1/m.
        \param[out] lower Lower bound of the mean-LAD confidence interval, in 1/m.
        \param[out] upper Upper bound of the mean-LAD confidence interval, in 1/m.
        \return True if at least one valid voxel contributed and an interval was produced.
     */
    bool getGroupLADConfidenceInterval(const std::vector<uint> &indices, float confidence_level, float &mean_lad, float &lower, float &upper) const;

    //! For scans that are missing points (e.g., sky points), this function will attempt to fill in missing points for all scans. This increases the accuracy of LAD calculations because it makes sure all pulses are accounted for.
    /**
     * \return (x,y,z) of missing points added to the scan from gapfilling
     */
    std::vector<helios::vec3> gapfillMisses();

    //! For scans that are missing points (e.g., sky points), this function will attempt to fill in missing points. This increases the accuracy of LAD calculations because it makes sure all pulses are accounted for.
    /**
     * \param[in] scanID ID of scan to gapfill
     * \return (x,y,z) of missing points added to the scan from gapfilling
     */
    std::vector<helios::vec3> gapfillMisses(uint scanID);

    //! For scans that are missing points (e.g., sky points), this function will attempt to fill in missing points. This increases the accuracy of LAD calculations because it makes sure all pulses are accounted for.
    /**
     * \param[in] scanID ID of scan to gapfill
     * \param[in] gapfill_grid_only if true, missing points are gapfilled only within the axis-aligned bounding box of the voxel grid. If false missing points are gap filled across the range of phi and theta values specified in the scan xml file.
     * \param[in] add_flags if true, gapfillMisses_code is added as hitpoint data. 0 = original points, 1 = gapfilled, 2 = extrapolated at downward edge, 3 = extrapolated at upward edge
     * \return (x,y,z) of missing points added to the scan from gapfilling
     */
    std::vector<helios::vec3> gapfillMisses(uint scanID, const bool gapfill_grid_only, const bool add_flags);


    //! Test/diagnostic hook: force the leaf-area inversion to use the brute-force per-cell slab loop
    /** By default \ref calculateLeafArea() uses a fast per-beam 3D-DDA traversal of the voxel lattice. When this is set
        true, the slower brute-force per-cell path is used instead (it produces identical results). This exists so the
        self-tests can verify the two paths agree; it is not needed in normal use.
        \param[in] force True to force the brute-force path, false (default) to use the fast DDA path when applicable. */
    void forceBruteForceLeafArea(bool force) {
        force_bruteforce_LAD = force;
    }

    //! Calculate the leaf area for each grid volume
    /**
     * \param[in] context Pointer to the Helios context
     * \note Requires that the point cloud contains miss points (transmitted beams); see
     *       \ref hasMisses() and \ref gapfillMisses(). Throws if no misses are present.
     */
    void calculateLeafArea(helios::Context *context);

    //! Calculate the leaf area for each grid volume
    /**
     * \param[in] context Pointer to the Helios context
     * \param[in] min_voxel_hits Minimum number of allowable LiDAR hits per voxel
     * \note Requires that the point cloud contains miss points (transmitted beams), which
     *       count the beams transmitted through each voxel for the transmission-probability
     *       inversion. Supply them via a miss-retaining scan format or \ref gapfillMisses();
     *       throws (fail-fast) if no misses are present. Misses are identified by the per-hit
     *       `is_miss` flag (see \ref isHitMiss()). Handles single- and multi-return data with a
     *       unified beam-based equal-weighting algorithm.
     */
    void calculateLeafArea(helios::Context *context, int min_voxel_hits);

    //! Calculate the leaf area for each grid volume, with element size for uncertainty estimation
    /**
     * \param[in] context Pointer to the Helios context
     * \param[in] min_voxel_hits Minimum number of allowable LiDAR hits per voxel
     * \param[in] element_width Characteristic vegetation element width [m] (e.g. mean leaf width),
     *            used by the per-voxel LAD sampling-uncertainty estimate (the element-position
     *            variance term; Pimont et al. 2018, Appendix A). Pass a value <= 0 to omit that
     *            term and report SAMPLING-ONLY uncertainty. The leaf-area point estimate is
     *            identical regardless of this argument.
     * \note Requires miss points; see the two-argument overload. After calling, per-voxel sampling
     *       uncertainty is available via \ref getCellLADVariance, \ref getCellBeamCount,
     *       \ref getCellLeafAreaConfidenceInterval, and \ref getGroupLADConfidenceInterval.
     */
    void calculateLeafArea(helios::Context *context, int min_voxel_hits, float element_width);

    //! Calculate the leaf area for each grid volume using a caller-supplied G(theta), without requiring triangulation
    /**
     * Beam-based leaf-area inversion for scans that cannot be triangulated - in particular moving-platform
     * (mobile/airborne) scans (see \ref addScanMoving()), whose pulses do not lie on a fixed theta-phi grid and so
     * cannot be Delaunay-triangulated. Triangulation is normally required only to estimate the per-voxel mean
     * leaf-projection coefficient G(theta); this overload takes G(theta) directly instead, so it does NOT require
     * \ref triangulateHitPoints() to have been called.
     *
     * The inversion uses the per-pulse beam origin recorded on each hit (see \ref getHitOrigin()) when classifying
     * beams against voxels, so it is geometrically correct for a scanner that moved during acquisition. For a static
     * scan the per-hit origin equals the scan origin, so this overload also works (with a supplied G(theta)) there.
     *
     * \param[in] context Pointer to the Helios context
     * \param[in] Gtheta Mean leaf-projection coefficient G(theta), applied to every voxel. Must be in (0,1]. Use 0.5
     *            for a spherical (random) leaf-angle distribution; supply a measured/assumed value otherwise.
     * \param[in] min_voxel_hits Minimum number of allowable LiDAR hits per voxel
     * \param[in] element_width Characteristic vegetation element width [m]; see the three-argument overload. Pass <= 0
     *            to report sampling-only uncertainty.
     * \note Requires miss points (transmitted beams), like the other overloads; supply a miss-retaining scan format or
     *       call \ref gapfillMisses() first.
     */
    void calculateLeafArea(helios::Context *context, float Gtheta, int min_voxel_hits, float element_width);

    //! Calculate the leaf area for each grid volume using a caller-supplied PER-VOXEL G(theta), without requiring triangulation
    /**
     * Identical to the single-G(theta) overload above, but takes one G(theta) per grid cell instead of a single value
     * applied everywhere. This supports a vertically-varying (or otherwise spatially-varying) leaf-angle distribution -
     * e.g. a canopy whose leaf inclination changes with height - without triangulating. Like the scalar overload it
     * inverts Beer's law from each beam's own origin and does NOT require \ref triangulateHitPoints().
     *
     * \param[in] context Pointer to the Helios context
     * \param[in] Gtheta_per_cell Mean leaf-projection coefficient G(theta) for each grid cell, in grid-cell order (the
     *            same order as \ref getCellCenter()). Its length must equal \ref getGridCellCount(). Every value must be
     *            in (0,1].
     * \param[in] min_voxel_hits Minimum number of allowable LiDAR hits per voxel
     * \param[in] element_width Characteristic vegetation element width [m]; see the three-argument overload. Pass <= 0
     *            to report sampling-only uncertainty.
     * \note Requires miss points (transmitted beams), like the other overloads; supply a miss-retaining scan format or
     *       call \ref gapfillMisses() first.
     */
    void calculateLeafArea(helios::Context *context, const std::vector<float> &Gtheta_per_cell, int min_voxel_hits, float element_width);

    //! Calculate the leaf area for each grid volume (DEPRECATED - use calculateLeafArea)
    /**
     * \deprecated This function has been renamed to calculateLeafArea(). The GPU-specific implementation has been replaced with CollisionDetection plugin integration. Use calculateLeafArea() instead. For GPU acceleration, call
     * enableCDGPUAcceleration() before calculateLeafArea().
     * \param[in] context Pointer to the Helios context
     */
    [[deprecated("Use calculateLeafArea() instead. GPU functionality is now provided by the CollisionDetection plugin.")]]
    void calculateLeafAreaGPU(helios::Context *context);

    //! Calculate the leaf area for each grid volume (DEPRECATED - use calculateLeafArea)
    /**
     * \deprecated This function has been renamed to calculateLeafArea(). The GPU-specific implementation has been replaced with CollisionDetection plugin integration. Use calculateLeafArea() instead. For GPU acceleration, call
     * enableCDGPUAcceleration() before calculateLeafArea().
     * \param[in] context Pointer to the Helios context
     * \param[in] min_voxel_hits Minimum number of allowable LiDAR hits per voxel
     */
    [[deprecated("Use calculateLeafArea(context, min_voxel_hits) instead. GPU functionality is now provided by the CollisionDetection plugin.")]]
    void calculateLeafAreaGPU(helios::Context *context, int min_voxel_hits);

    //! Enable GPU acceleration in CollisionDetection plugin
    void enableGPUAcceleration();

    //! Disable GPU acceleration in CollisionDetection plugin (use CPU/OpenMP only)
    void disableGPUAcceleration();

    //! Check whether a CUDA-capable GPU is available for acceleration
    /** Returns true only if the CollisionDetection plugin was compiled with CUDA support, a
        CUDA device is present at runtime, and the GPU path is not disabled via the
        HELIOS_NO_GPU environment variable. Reports capability; use isGPUAccelerationEnabled()
        to query whether GPU acceleration is currently toggled on. */
    [[nodiscard]] bool isGPUAvailable() const;

    //! Check whether GPU acceleration is currently enabled
    [[nodiscard]] bool isGPUAccelerationEnabled() const;

    //! Determine which grid cell each hit point resides in
    void calculateHitGridCell();

    // -------- RECONSTRUCTION --------- //

    //! Perform a leaf reconstruction based on texture-masked Patches within each gridcell.  The reconstruction produces Patches for each reconstructed leaf surface, with leaf size automatically estimated algorithmically.
    /**
     * \param[in] minimum_leaf_group_area Minimum allowable area of leaf triangular fill groups. Leaf fill groups with total areas less than minimum_leaf_group_area are not considered in the reconstruction.
     * \param[in] maximum_leaf_group_area Maximum area of leaf triangular fill groups. Leaf fill groups with total areas greater than maximum_leaf_group_area are not considered in the reconstruction.
     * \param[in] leaf_aspect_ratio Ratio of length of leaf along midrib to with of leaf perpendicular to leaf midrib.  This will generally be the length/width of leaf mask.
     * \param[in] mask_file Path to PNG image file to be used with Alpha Mask.
     */
    void leafReconstructionAlphaMask(float minimum_leaf_group_area, float maximum_leaf_group_area, float leaf_aspect_ratio, const char *mask_file);

    //! Perform a leaf reconstruction based on texture-masked Patches within each gridcell.  The reconstruction produces Patches for each reconstructed leaf surface, with leaf size set to a constant value.
    /**
     * \param[in] minimum_leaf_group_area Minimum allowable area of leaf triangular fill groups. Leaf fill groups with total areas less than minimum_leaf_group_area are not considered in the reconstruction.
     * \param[in] maximum_leaf_group_area Maximum area of leaf triangular fill groups. Leaf fill groups with total areas greater than maximum_leaf_group_area are not considered in the reconstruction.
     * \param[in] leaf_aspect_ratio Ratio of length of leaf along midrib to with of leaf perpendicular to leaf midrib.  This will generally be the length/width of leaf mask.
     * \param[in] leaf_length_constant Constant length of all reconstructed leaves.
     * \param[in] mask_file Path to PNG image file to be used with Alpha Mask.
     */
    void leafReconstructionAlphaMask(float minimum_leaf_group_area, float maximum_leaf_group_area, float leaf_aspect_ratio, float leaf_length_constant, const char *mask_file);

    //! Reconstruct the trunk of the tree. In order to do this, you must specify the center and size of a rectangular box that encompasses the tree trunk. This routine will then try to find the largest continuous triangle group, which is assumed to
    //! correspond to the trunk.
    /**
     * \param[in] box_center (x,y,z) coordinates of the center of a rectangular box that encompasses the tree trunk.
     * \param[in] box_size Dimension of the trunk box in the x-, y-, and z- directions.
     * \param[in] Lmax maximum dimension of triangles (see also triangulateHitPoints()).
     * \param[in] max_aspect_ratio Maximum allowable aspect ratio of triangles (see also triangulateHitPoints())
     */
    void trunkReconstruction(const helios::vec3 &box_center, const helios::vec3 &box_size, float Lmax, float max_aspect_ratio);

    //! Read a TreeQSM cylinder file and add tube objects to the context for each branch
    /**
     * \param[in] context Pointer to the Helios context where tube objects will be added.
     * \param[in] filename Path to the TreeQSM cylinder text file.
     * \param[in] radial_subdivisions Number of radial subdivisions for the tube objects.
     * \param[in] texture_file Optional path to texture image file for the tube objects. If empty, tubes will be colored red.
     * \return Vector of tube object IDs that were created.
     */
    std::vector<uint> loadTreeQSM(helios::Context *context, const std::string &filename, uint radial_subdivisions, const std::string &texture_file = "");

    //! Read a TreeQSM cylinder file and add tube objects to the context for each branch with colormap-based coloring
    /**
     * \param[in] context Pointer to the Helios context where tube objects will be added.
     * \param[in] filename Path to the TreeQSM cylinder text file.
     * \param[in] radial_subdivisions Number of radial subdivisions for the tube objects.
     * \param[in] colormap_name Name of the Helios colormap to use for coloring branches (e.g., "hot", "cool", "rainbow").
     * \return Vector of tube object UUIDs that were created.
     * \note Each branch will be colored with a color randomly sampled from the colormap based on the branch ID.
     */
    std::vector<uint> loadTreeQSMColormap(helios::Context *context, const std::string &filename, uint radial_subdivisions, const std::string &colormap_name);

    //! Delete hitpoints that do not pass through / intersect the voxel grid
    /**
     * \param[in] source the scan index
     */
    void cropBeamsToGridAngleRange(uint source);

    //! find the indices of the peaks of a vector of floats
    /**
     * \param[in] signal the signal we want to detect peaks in
     */
    std::vector<uint> peakFinder(std::vector<float> signal);
};

bool sortcol0(const std::vector<double> &v0, const std::vector<double> &v1);

bool sortcol1(const std::vector<double> &v0, const std::vector<double> &v1);

#endif
