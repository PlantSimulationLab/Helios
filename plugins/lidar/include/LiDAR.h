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
    std::map<std::string, double> data;
    int gridcell;
    int scanID;
    HitPoint(void) {
        position = helios::make_vec3(0, 0, 0);
        direction = helios::make_SphericalCoord(0, 0);
        row_column = helios::make_int2(0, 0);
        color = helios::RGB::red;
        gridcell = -2;
        scanID = -1;
    }
    HitPoint(int __scanID, helios::vec3 __position, helios::SphericalCoord __direction, helios::int2 __row_column, helios::RGBcolor __color, std::map<std::string, double> __data) {
        scanID = __scanID;
        position = __position;
        direction = __direction;
        row_column = __row_column;
        color = __color;
        data = __data;
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
 * a uniform azimuth step. Both patterns share the same Ntheta x Nphi table storage, so all downstream processing (ray tracing,
 * hit tables, leaf-area and leaf-angle inversion) is common to both.
 */
enum ScanPattern {
    //! Uniform angular grid: zenith uniformly spaced over [thetaMin, thetaMax], azimuth over [phiMin, phiMax]
    SCAN_PATTERN_RASTER = 0,
    //! Rotating multi-channel sensor: each row is a laser channel at a fixed zenith angle (see \ref ScanMetadata::beamZenithAngles), each column a uniform azimuth step
    SCAN_PATTERN_SPINNING_MULTIBEAM = 1
};

//! Structure containing metadata for a terrestrial scan
/** A scan is initialized by providing 1) the origin of the scan (see \ref origin), 2) the number of zenithal scan directions (see \ref Ntheta), 3) the range of zenithal scan angles (see \ref thetaMin, \ref thetaMax), 4) the number of azimuthal scan
directions (see \ref Nphi), 5) the range of azimuthal scan angles (see \ref phiMin, \ref phiMax). This creates a grid of Ntheta x Nphi scan points which are all initialized as misses.  Points are set as hits using the addHitPoint() function. There
are various functions to query the scan data.
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

    std::vector<GridCell> grid_cells;

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

    //! Flag denoting whether messages should be printed to screen
    bool printmessages;

    //! Collision detection plugin for unified ray-tracing
    CollisionDetection *collision_detection;

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
     * \param[in] supplied_Gtheta If > 0, this G(theta) is used for every voxel and triangulation is NOT required. If <=
     *            0 (the sentinel), G(theta) is computed per voxel from triangulation, which must have been performed.
     */
    void calculateLeafArea_inner(helios::Context *context, int min_voxel_hits, float element_width, float supplied_Gtheta);

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

    //! Beam grouping structure for multi-return data
    struct BeamGrouping {
        uint Nbeams;
        std::vector<std::vector<uint>> beam_array;
    };

    //! Group hit points by timestamp into beams
    /**
     * \param[in] scan_indices Vector of hit indices for a specific scan
     * \return BeamGrouping structure with beam organization
     */
    BeamGrouping groupHitsByTimestamp(const std::vector<uint> &scan_indices) const;

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
     */
    void triangulateHitPoints(float Lmax, float max_aspect_ratio, const char *scalar_field, float threshold, const char *comparator);


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
