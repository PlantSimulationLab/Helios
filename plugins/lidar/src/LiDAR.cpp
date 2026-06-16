/** \file "LiDAR.cpp" Primary source file for LiDAR plug-in.

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

#include <set>

using namespace std;
using namespace helios;

namespace {
    //! Column-format tokens that map to geometry/standard hit fields handled directly by file I/O.
    //! Any other column-format label is treated as a primitive-data field to sample onto hits.
    bool isStandardColumnToken(const std::string &label) {
        static const std::set<std::string> standard_tokens = {"x", "y", "z", "r", "g", "b", "r255", "g255", "b255", "row", "column", "zenith", "azimuth", "zenith_rad", "azimuth_rad",
                                                              // "raydir" is reserved here so it is never treated as a primitive-data label,
                                                              // even though the ASCII file reader treats it as a generic scalar column.
                                                              "raydir"};
        return standard_tokens.find(label) != standard_tokens.end();
    }

    //! Inverse of the standard normal CDF (quantile function), Acklam's rational approximation.
    //! Returns z such that Phi(z) = p, for p in (0,1). Relative error < 1.15e-9.
    //! Used to convert a confidence level into the two-sided z-multiplier z_{alpha/2}.
    double normalQuantile(double p) {
        if (p <= 0.0 || p >= 1.0) {
            helios_runtime_error("ERROR (normalQuantile): probability must be strictly between 0 and 1.");
        }
        // Coefficients for Acklam's algorithm
        static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
        static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01};
        static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
        static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00};
        const double p_low = 0.02425;
        const double p_high = 1.0 - p_low;
        double q, r;
        if (p < p_low) { // lower tail
            q = std::sqrt(-2.0 * std::log(p));
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        } else if (p <= p_high) { // central region
            q = p - 0.5;
            r = q * q;
            return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
        } else { // upper tail
            q = std::sqrt(-2.0 * std::log(1.0 - p));
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        }
    }

    // ---- Quaternion math for moving-platform LiDAR ----
    // Convention (pinned): Hamilton quaternions, body->world, stored as helios::vec4 components (x,y,z,w) = (qx,qy,qz,qw).
    // quat_rotate(q, v) rotates a vector v expressed in the body frame into the world frame. quat_from_rpy builds a
    // quaternion from intrinsic Z-Y-X (yaw-pitch-roll) Tait-Bryan angles: q = qz(yaw) * qy(pitch) * qx(roll), so the
    // resulting rotation applies roll first, then pitch, then yaw (the standard aerospace convention). Helios stores
    // geometry in single precision, so these operate on float vec4/vec3 to match the rest of the pipeline.

    //! Rotate a body-frame vector into the world frame by a Hamilton quaternion (qx,qy,qz,qw).
    helios::vec3 quat_rotate(const helios::vec4 &q, const helios::vec3 &v) {
        // v' = v + 2*qw*(qv x v) + 2*(qv x (qv x v)), with qv = (qx,qy,qz). Assumes q is unit-norm.
        const helios::vec3 qv = helios::make_vec3(q.x, q.y, q.z);
        const helios::vec3 t = helios::cross(qv, v) * 2.f;
        return v + t * q.w + helios::cross(qv, t);
    }

    //! Spherically interpolate (SLERP) between two Hamilton quaternions; falls back to normalized lerp when nearly parallel.
    helios::vec4 quat_slerp(helios::vec4 q0, helios::vec4 q1, double u) {
        q0.normalize();
        q1.normalize();
        double dot = double(q0.x) * q1.x + double(q0.y) * q1.y + double(q0.z) * q1.z + double(q0.w) * q1.w;
        // Quaternions q and -q represent the same rotation; choose the shorter arc.
        if (dot < 0.0) {
            q1 = helios::make_vec4(-q1.x, -q1.y, -q1.z, -q1.w);
            dot = -dot;
        }
        if (dot > 0.9995) {
            // Nearly parallel: normalized linear interpolation avoids division by ~zero sin(theta).
            helios::vec4 result = helios::make_vec4(q0.x + float(u) * (q1.x - q0.x), q0.y + float(u) * (q1.y - q0.y), q0.z + float(u) * (q1.z - q0.z), q0.w + float(u) * (q1.w - q0.w));
            result.normalize();
            return result;
        }
        const double theta_0 = std::acos(dot);
        const double theta = theta_0 * u;
        const double sin_theta_0 = std::sin(theta_0);
        const double s0 = std::sin(theta_0 - theta) / sin_theta_0;
        const double s1 = std::sin(theta) / sin_theta_0;
        helios::vec4 result = helios::make_vec4(float(s0 * q0.x + s1 * q1.x), float(s0 * q0.y + s1 * q1.y), float(s0 * q0.z + s1 * q1.z), float(s0 * q0.w + s1 * q1.w));
        result.normalize();
        return result;
    }

    //! Build a Hamilton quaternion (qx,qy,qz,qw) from intrinsic Z-Y-X (yaw-pitch-roll) Tait-Bryan angles in radians.
    helios::vec4 quat_from_rpy(float roll, float pitch, float yaw) {
        const float cr = std::cos(roll * 0.5f), sr = std::sin(roll * 0.5f);
        const float cp = std::cos(pitch * 0.5f), sp = std::sin(pitch * 0.5f);
        const float cy = std::cos(yaw * 0.5f), sy = std::sin(yaw * 0.5f);
        // q = qz(yaw) * qy(pitch) * qx(roll) (Hamilton product), components (x,y,z,w).
        helios::vec4 q;
        q.w = cr * cp * cy + sr * sp * sy;
        q.x = sr * cp * cy - cr * sp * sy;
        q.y = cr * sp * cy + sr * cp * sy;
        q.z = cr * cp * sy - sr * sp * cy;
        return q;
    }
} // namespace

ScanMetadata::ScanMetadata() {

    origin = make_vec3(0, 0, 0);
    Ntheta = 100;
    thetaMin = 0;
    thetaMax = M_PI;
    Nphi = 200;
    phiMin = 0;
    phiMax = 2.f * M_PI;
    exitDiameter = 0;
    beamDivergence = 0;
    rangeNoiseStdDev = 0;
    angleNoiseStdDev = 0;
    scanTilt_roll = 0;
    scanTilt_pitch = 0;
    scanTilt_azimuth = 0;
    columnFormat = {"x", "y", "z"};
    scanPattern = SCAN_PATTERN_RASTER;

    data_file = "";
}

ScanMetadata::ScanMetadata(const vec3 &a_origin, uint a_Ntheta, float a_thetaMin, float a_thetaMax, uint a_Nphi, float a_phiMin, float a_phiMax, float a_exitDiameter, float a_beamDivergence, float a_rangeNoiseStdDev, float a_angleNoiseStdDev,
                           const vector<string> &a_columnFormat, float a_scanTiltRoll, float a_scanTiltPitch, float a_scanAzimuthOffset) {

    // Copy arguments into structure variables
    origin = a_origin;
    Ntheta = a_Ntheta;
    thetaMin = a_thetaMin;
    thetaMax = a_thetaMax;
    Nphi = a_Nphi;
    phiMin = a_phiMin;
    phiMax = a_phiMax;
    exitDiameter = a_exitDiameter;
    beamDivergence = a_beamDivergence;
    rangeNoiseStdDev = a_rangeNoiseStdDev;
    angleNoiseStdDev = a_angleNoiseStdDev;
    scanTilt_roll = a_scanTiltRoll;
    scanTilt_pitch = a_scanTiltPitch;
    scanTilt_azimuth = a_scanAzimuthOffset;
    columnFormat = a_columnFormat;
    scanPattern = SCAN_PATTERN_RASTER;

    data_file = "";
}

ScanMetadata::ScanMetadata(const vec3 &a_origin, const std::vector<float> &a_beamZenithAngles, uint a_Nphi, float a_phiMin, float a_phiMax, float a_exitDiameter, float a_beamDivergence, float a_rangeNoiseStdDev, float a_angleNoiseStdDev,
                           const vector<string> &a_columnFormat, float a_scanTiltRoll, float a_scanTiltPitch, float a_scanAzimuthOffset) {

    if (a_beamZenithAngles.empty()) {
        helios_runtime_error("ERROR (ScanMetadata): A spinning multibeam scan requires at least one beam (channel) zenith angle, but the provided beamZenithAngles vector is empty.");
    }

    origin = a_origin;
    scanPattern = SCAN_PATTERN_SPINNING_MULTIBEAM;
    beamZenithAngles = a_beamZenithAngles;
    Ntheta = uint(a_beamZenithAngles.size()); // one row per laser channel
    // thetaMin/thetaMax bracket the channel angles so range queries and bounding logic remain valid for non-uniform spacing.
    thetaMin = *std::min_element(a_beamZenithAngles.begin(), a_beamZenithAngles.end());
    thetaMax = *std::max_element(a_beamZenithAngles.begin(), a_beamZenithAngles.end());
    Nphi = a_Nphi;
    phiMin = a_phiMin;
    phiMax = a_phiMax;
    exitDiameter = a_exitDiameter;
    beamDivergence = a_beamDivergence;
    rangeNoiseStdDev = a_rangeNoiseStdDev;
    angleNoiseStdDev = a_angleNoiseStdDev;
    scanTilt_roll = a_scanTiltRoll;
    scanTilt_pitch = a_scanTiltPitch;
    scanTilt_azimuth = a_scanAzimuthOffset;
    columnFormat = a_columnFormat;

    data_file = "";
}

helios::SphericalCoord ScanMetadata::rc2direction(uint row, uint column) const {

    float zenith;
    if (scanPattern == SCAN_PATTERN_SPINNING_MULTIBEAM) {
        // Each row is a laser channel fired at its own fixed (generally non-uniform) zenith angle.
        uint clamped_row = (row < beamZenithAngles.size()) ? row : uint(beamZenithAngles.size()) - 1;
        zenith = beamZenithAngles.at(clamped_row);
    } else {
        zenith = thetaMin + (thetaMax - thetaMin) / float(Ntheta) * float(row);
    }
    float elevation = 0.5f * M_PI - zenith;
    float phi = phiMin - (phiMax - phiMin) / float(Nphi) * float(column);
    return make_SphericalCoord(1, elevation, phi);
};

helios::int2 ScanMetadata::direction2rc(const SphericalCoord &direction) const {

    float theta = direction.zenith;
    float phi = direction.azimuth;

    int row;
    if (scanPattern == SCAN_PATTERN_SPINNING_MULTIBEAM) {
        // Channel zenith angles are not uniformly spaced, so map to the nearest channel rather than interpolating linearly.
        int nearest = 0;
        float best = std::fabs(theta - beamZenithAngles.at(0));
        for (uint k = 1; k < beamZenithAngles.size(); k++) {
            float d = std::fabs(theta - beamZenithAngles.at(k));
            if (d < best) {
                best = d;
                nearest = int(k);
            }
        }
        row = nearest;
    } else {
        row = std::round((theta - thetaMin) / (thetaMax - thetaMin) * float(Ntheta));
    }
    int column = std::round(fabs(phi - phiMin) / (phiMax - phiMin) * float(Nphi));

    if (row <= -1) {
        row = 0;
    } else if (row >= Ntheta) {
        row = Ntheta - 1;
    }
    if (column <= -1) {
        column = 0;
    } else if (column >= Nphi) {
        column = Nphi - 1;
    }

    return helios::make_int2(row, column);
};

void ScanMetadata::poseAt(double t, helios::vec3 &pos, helios::vec4 &quat) const {

    const size_t M = traj_t.size();
    if (M == 0) {
        helios_runtime_error("ERROR (ScanMetadata::poseAt): the scan has no trajectory samples. This scan was not created as a moving-platform scan (see LiDARcloud::addScanMoving).");
    }
    if (traj_pos.size() != M || traj_quat.size() != M) {
        helios_runtime_error("ERROR (ScanMetadata::poseAt): trajectory arrays have inconsistent lengths (traj_t=" + std::to_string(M) + ", traj_pos=" + std::to_string(traj_pos.size()) + ", traj_quat=" + std::to_string(traj_quat.size()) +
                             "). All three must have the same number of samples.");
    }

    // Single sample: constant pose (also covers a degenerate zero-velocity trajectory).
    if (M == 1) {
        pos = traj_pos.at(0);
        quat = traj_quat.at(0);
        return;
    }

    // Clamp to the trajectory endpoints rather than extrapolating.
    if (t <= traj_t.front()) {
        pos = traj_pos.front();
        quat = traj_quat.front();
        return;
    }
    if (t >= traj_t.back()) {
        pos = traj_pos.back();
        quat = traj_quat.back();
        return;
    }

    // Binary search for the bracketing interval [traj_t[i], traj_t[i+1]] containing t.
    // upper_bound returns the first sample strictly greater than t, so the lower index is one before it.
    const auto upper = std::upper_bound(traj_t.begin(), traj_t.end(), t);
    const size_t i1 = size_t(upper - traj_t.begin());
    const size_t i0 = i1 - 1;

    const double t0_s = traj_t.at(i0);
    const double t1_s = traj_t.at(i1);
    const double denom = t1_s - t0_s;
    if (denom <= 0.0) {
        helios_runtime_error("ERROR (ScanMetadata::poseAt): trajectory times are not strictly increasing (traj_t[" + std::to_string(i0) + "]=" + std::to_string(t0_s) + " >= traj_t[" + std::to_string(i1) + "]=" + std::to_string(t1_s) + ").");
    }
    const double u = (t - t0_s) / denom;

    // Linear interpolation of position; SLERP of orientation.
    pos = traj_pos.at(i0) + (traj_pos.at(i1) - traj_pos.at(i0)) * float(u);
    quat = quat_slerp(traj_quat.at(i0), traj_quat.at(i1), u);
};

LiDARcloud::LiDARcloud() {

    Nhits = 0;
    hitgridcellcomputed = false;
    triangulationcomputed = false;
    triangulation_candidate_count = 0;
    triangulation_dropped_lmax = 0;
    triangulation_dropped_aspect = 0;
    triangulation_dropped_degenerate = 0;
    printmessages = true;
    collision_detection = nullptr;
}

LiDARcloud::~LiDARcloud(void) {
    delete collision_detection;
}

void LiDARcloud::disableMessages() {
    printmessages = false;
}

void LiDARcloud::initializeCollisionDetection(helios::Context *context) {
    if (collision_detection == nullptr) {
        collision_detection = new CollisionDetection(context);
        collision_detection->disableMessages();
    }
}

void LiDARcloud::performUnifiedRayTracing(helios::Context *context, size_t N, int Npulse, helios::vec3 *ray_origins, helios::vec3 *direction, float *hit_t, float *hit_fnorm, int *hit_ID) {
    const float miss_distance = LIDAR_RAYTRACE_MISS_T;

    size_t total_rays = N * Npulse;

    // Disable automatic BVH rebuilds during batch ray tracing (geometry is static during scan)
    collision_detection->disableAutomaticBVHRebuilds();

    // Manually ensure BVH is current once for the entire batch
    collision_detection->buildBVH();

    // Convert LiDAR rays to CollisionDetection format
    std::vector<CollisionDetection::RayQuery> ray_queries;
    ray_queries.reserve(total_rays);

    for (size_t i = 0; i < total_rays; i++) {
        ray_queries.emplace_back(ray_origins[i], direction[i], miss_distance);
    }

    // Use the collision detection ray casting (this replaces the old CUDA kernels)
    std::vector<CollisionDetection::HitResult> hit_results = collision_detection->castRays(ray_queries);

    // Re-enable automatic BVH rebuilds for future operations
    collision_detection->enableAutomaticBVHRebuilds();

    // Convert results back to LiDAR format
    size_t hit_count = 0;
    for (size_t i = 0; i < total_rays; i++) {
        const auto &result = hit_results[i];
        if (result.hit) {
            hit_count++;
            hit_t[i] = result.distance;
            hit_ID[i] = static_cast<int>(result.primitive_UUID);

            // Calculate dot product for surface normal (approximation)
            helios::vec3 ray_dir = direction[i];
            hit_fnorm[i] = ray_dir.x * result.normal.x + ray_dir.y * result.normal.y + ray_dir.z * result.normal.z;

        } else {
            hit_t[i] = miss_distance;
            hit_ID[i] = -1;
            hit_fnorm[i] = 1e6;
        }
    }
}

float LiDARcloud::applyRangeIntensityCorrection(float intensity, float distance) {
    // Helios reports RANGE-NORMALIZED intensity: the range-independent return amplitude rho*cos(theta), as if the
    // geometric 1/R^2 loss of the LiDAR range equation had been measured and then divided back out. Equivalently,
    // the physical raw return rho*cos(theta)/R^2 is multiplied by R^2 to remove the range dependence:
    //
    //     I_norm = (rho*cos(theta) / R^2) * R^2 = rho*cos(theta)
    //
    // Because the synthetic intensity is generated directly as rho*cos(theta) (no 1/R^2 loss is ever applied), the
    // two operations cancel and the normalization is the identity on the value. This helper makes that contract
    // explicit and is the single place to change if the raw (range-dependent) convention is ever desired instead.
    // The partial-footprint (point-target) attenuation of sub-footprint returns (in multi-return mode) is carried by the
    // fraction of beam sub-rays that strike the target and is intentionally retained (it is a target property, not
    // a range-geometry loss). The distance argument is accepted for interface symmetry and future raw-mode use.
    (void) distance;
    return intensity;
}

void LiDARcloud::enableMessages() {
    printmessages = true;
}

bool LiDARcloud::anyScanMoving() const {
    for (const auto &scan: scans) {
        if (scan.isMoving) {
            return true;
        }
    }
    return false;
}

void LiDARcloud::validateRayDirections() {

    // This validation reconstructs each hit's direction from the single scan origin, which is not meaningful for a
    // moving-platform scan (each pulse has its own origin). Fail fast rather than report spurious mismatches.
    if (anyScanMoving()) {
        helios_runtime_error("ERROR (LiDARcloud::validateRayDirections): ray-direction validation is not supported for moving-platform scans (see addScanMoving), because the per-pulse origins make a single-origin direction check meaningless.");
    }

    for (uint s = 0; s < getScanCount(); s++) {

        for (int j = 0; j < getScanSizePhi(s); j++) {
            for (int i = 0; i < getScanSizeTheta(s); i++) {
                if (getHitIndex(s, i, j) >= 0) {
                    SphericalCoord direction1 = scans.at(s).rc2direction(i, j);
                    SphericalCoord direction2 = cart2sphere(getHitXYZ(getHitIndex(s, i, j)) - getScanOrigin(s));
                    SphericalCoord direction3 = getHitRaydir(getHitIndex(s, i, j));


                    float err_theta = max(fabs(direction1.zenith - direction2.zenith), fabs(direction1.zenith - direction3.zenith));

                    float err_phi = max(fabs(direction1.azimuth - direction2.azimuth), fabs(direction1.azimuth - direction3.azimuth));

                    if (err_theta > 1e-6 || err_phi > 1e-6) {
                        helios_runtime_error("ERROR (LiDARcloud::validateRayDirections): validation of ray directions failed.");
                    }
                }
            }
        }
    }
}

uint LiDARcloud::getScanCount() {
    return scans.size();
}

uint LiDARcloud::addScan(ScanMetadata &newscan) {

    float epsilon = 1e-5;

    if (newscan.thetaMin < 0) {
        std::cerr << "WARNING (LiDARcloud::addScan): Specified scan minimum zenith angle of " << newscan.thetaMin << " is less than 0. Truncating to 0." << std::endl;
        newscan.thetaMin = 0;
    }
    if (newscan.phiMin < 0) {
        std::cerr << "WARNING (LiDARcloud::addScan): Specified scan minimum azimuth angle of " << newscan.phiMin << " is less than 0. Truncating to 0." << std::endl;
        newscan.phiMin = 0;
    }
    if (newscan.thetaMax > M_PI + epsilon) {
        std::cerr << "WARNING (LiDARcloud::addScan): Specified scan maximum zenith angle of " << newscan.thetaMax << " is greater than pi. Setting thetaMin to 0 and truncating thetaMax to pi. Did you mistakenly use degrees instead of radians?"
                  << std::endl;
        newscan.thetaMax = M_PI;
        newscan.thetaMin = 0;
    }
    if (newscan.phiMax > 4.f * M_PI + epsilon) {
        std::cerr << "WARNING (LiDARcloud::addScan): Specified scan maximum azimuth angle of " << newscan.phiMax << " is greater than 2pi. Did you mistakenly use degrees instead of radians?" << std::endl;
    }

    // initialize the hit table to `-1' (all misses)
    HitTable<int> table;
    table.resize(newscan.Ntheta, newscan.Nphi, -1);
    hit_tables.push_back(table);

    scans.emplace_back(newscan);

    return scans.size() - 1;
}

uint LiDARcloud::addScanMoving(ScanMetadata scan, const std::vector<double> &traj_t, const std::vector<vec3> &traj_pos, const std::vector<vec4> &traj_quat, const vec3 &lever_arm, const vec3 &boresight_rpy, float pulse_rate_hz, double t0) {

    const size_t M = traj_t.size();
    if (M == 0) {
        helios_runtime_error("ERROR (LiDARcloud::addScanMoving): the trajectory is empty. At least one pose sample is required.");
    }
    if (traj_pos.size() != M || traj_quat.size() != M) {
        helios_runtime_error("ERROR (LiDARcloud::addScanMoving): trajectory arrays have inconsistent lengths (traj_t=" + std::to_string(M) + ", traj_pos=" + std::to_string(traj_pos.size()) + ", traj_quat=" + std::to_string(traj_quat.size()) +
                             "). All three must have the same number of samples.");
    }
    for (size_t k = 1; k < M; k++) {
        if (traj_t.at(k) <= traj_t.at(k - 1)) {
            helios_runtime_error("ERROR (LiDARcloud::addScanMoving): trajectory times must be strictly increasing, but traj_t[" + std::to_string(k - 1) + "]=" + std::to_string(traj_t.at(k - 1)) + " >= traj_t[" + std::to_string(k) + "]=" +
                                 std::to_string(traj_t.at(k)) + ".");
        }
    }
    // Reject non-finite trajectory data up front: a single NaN/inf would otherwise propagate silently through the SLERP
    // interpolation and produce NaN origins/directions for a whole range of pulses. Quaternions must also be non-zero so
    // they can be normalized.
    for (size_t k = 0; k < M; k++) {
        if (!std::isfinite(traj_t.at(k))) {
            helios_runtime_error("ERROR (LiDARcloud::addScanMoving): trajectory time traj_t[" + std::to_string(k) + "] is not finite (NaN or infinity).");
        }
        const helios::vec3 &p = traj_pos.at(k);
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            helios_runtime_error("ERROR (LiDARcloud::addScanMoving): trajectory position traj_pos[" + std::to_string(k) + "] is not finite (NaN or infinity).");
        }
        const helios::vec4 &q = traj_quat.at(k);
        if (!std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.z) || !std::isfinite(q.w)) {
            helios_runtime_error("ERROR (LiDARcloud::addScanMoving): trajectory quaternion traj_quat[" + std::to_string(k) + "] is not finite (NaN or infinity).");
        }
        if (q.magnitude() < 1e-6f) {
            helios_runtime_error("ERROR (LiDARcloud::addScanMoving): trajectory quaternion traj_quat[" + std::to_string(k) + "] has near-zero magnitude and cannot be normalized to a valid rotation.");
        }
    }
    if (!std::isfinite(t0)) {
        helios_runtime_error("ERROR (LiDARcloud::addScanMoving): t0 must be finite, but a non-finite value was provided.");
    }
    if (pulse_rate_hz <= 0.f) {
        helios_runtime_error("ERROR (LiDARcloud::addScanMoving): pulse_rate_hz must be greater than 0, but " + std::to_string(pulse_rate_hz) + " was provided.");
    }
    // Trajectory replaces scanTilt: the static roll/pitch/yaw tilt is incompatible with a trajectory-driven attitude.
    if (scan.scanTilt_roll != 0.f || scan.scanTilt_pitch != 0.f || scan.scanTilt_azimuth != 0.f) {
        helios_runtime_error("ERROR (LiDARcloud::addScanMoving): the scan specifies a non-zero static tilt (scanTilt_roll/pitch/azimuth), which is not applied for moving-platform scans. Platform attitude must be supplied entirely "
                             "through the trajectory quaternions and the boresight; leave the static tilt at zero.");
    }

    scan.isMoving = true;
    scan.traj_t = traj_t;
    scan.traj_pos = traj_pos;
    scan.traj_quat = traj_quat;
    scan.lever_arm = lever_arm;
    scan.boresight_rpy = boresight_rpy;
    scan.pulse_period = 1.0 / double(pulse_rate_hz);
    scan.t0 = t0;
    // The trajectory supplies position; the static origin field is unused for moving scans.
    scan.origin = traj_pos.front();

    return addScan(scan);
}

uint LiDARcloud::addScanMoving(ScanMetadata scan, const std::vector<double> &traj_t, const std::vector<vec3> &traj_pos, const std::vector<vec3> &traj_rpy, const vec3 &lever_arm, const vec3 &boresight_rpy, float pulse_rate_hz, double t0) {

    // Convert the per-sample roll/pitch/yaw Euler angles to Hamilton body->world quaternions (intrinsic Z-Y-X, the same
    // convention used for the boresight), then delegate to the quaternion overload so all validation lives in one place.
    // The length check is duplicated here only so the error message names traj_rpy rather than the converted traj_quat.
    if (traj_rpy.size() != traj_t.size()) {
        helios_runtime_error("ERROR (LiDARcloud::addScanMoving): trajectory arrays have inconsistent lengths (traj_t=" + std::to_string(traj_t.size()) + ", traj_rpy=" + std::to_string(traj_rpy.size()) + "). All trajectory arrays must have the same number of samples.");
    }

    std::vector<vec4> traj_quat;
    traj_quat.reserve(traj_rpy.size());
    for (const vec3 &rpy: traj_rpy) {
        traj_quat.push_back(quat_from_rpy(rpy.x, rpy.y, rpy.z));
    }

    return addScanMoving(scan, traj_t, traj_pos, traj_quat, lever_arm, boresight_rpy, pulse_rate_hz, t0);
}

void LiDARcloud::addHitPoint(uint scanID, const vec3 &xyz, const SphericalCoord &direction) {

    // default color
    RGBcolor color = make_RGBcolor(1, 0, 0);

    // empty data
    std::map<std::string, double> data;

    addHitPoint(scanID, xyz, direction, color, data);
}

void LiDARcloud::addHitPoint(uint scanID, const vec3 &xyz, const SphericalCoord &direction, const map<string, double> &data) {

    // default color
    RGBcolor color = make_RGBcolor(1, 0, 0);

    addHitPoint(scanID, xyz, direction, color, data);
}

void LiDARcloud::addHitPoint(uint scanID, const vec3 &xyz, const SphericalCoord &direction, const RGBcolor &color) {

    // empty data
    std::map<std::string, double> data;

    addHitPoint(scanID, xyz, direction, color, data);
}

void LiDARcloud::addHitPoint(uint scanID, const vec3 &xyz, const SphericalCoord &direction, const RGBcolor &color, const map<string, double> &data) {

    // error checking
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::addHitPoint): Hit point cannot be added to scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }

    ScanMetadata scan = scans.at(scanID);
    int2 row_column = scan.direction2rc(direction);

    HitPoint hit(scanID, xyz, direction, row_column, color, data);

    hits.push_back(hit);
}

void LiDARcloud::addHitPoint(uint scanID, const vec3 &xyz, const int2 &row_column, const RGBcolor &color, const map<string, double> &data) {

    ScanMetadata scan = scans.at(scanID);
    SphericalCoord direction = scan.rc2direction(row_column.x, row_column.y);

    HitPoint hit(scanID, xyz, direction, row_column, color, data);

    hits.push_back(hit);
}

void LiDARcloud::deleteHitPoint(uint index) {

    if (index >= hits.size()) {
        cerr << "WARNING (deleteHitPoint): Hit point #" << index << " cannot be deleted from the scan because there have only been " << hits.size() << " hit points added." << endl;
        return;
    }

    HitPoint hit = hits.at(index);

    int scanID = hit.scanID;

    // erase from vector of hits (use swap-and-pop method)
    std::swap(hits.at(index), hits.back());
    hits.pop_back();
}

uint LiDARcloud::getHitCount() const {
    return hits.size();
}

helios::vec3 LiDARcloud::getScanOrigin(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanOrigin): Cannot get origin of scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).origin;
}

uint LiDARcloud::getScanSizeTheta(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanSizeTheta): Cannot get theta size for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).Ntheta;
}

uint LiDARcloud::getScanSizePhi(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanSizePhi): Cannot get phi size for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).Nphi;
}

helios::vec2 LiDARcloud::getScanRangeTheta(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanRangeTheta): Cannot get theta range for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return helios::make_vec2(scans.at(scanID).thetaMin, scans.at(scanID).thetaMax);
}

helios::vec2 LiDARcloud::getScanRangePhi(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanRangePhi): Cannot get phi range for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return helios::make_vec2(scans.at(scanID).phiMin, scans.at(scanID).phiMax);
}

float LiDARcloud::getScanBeamExitDiameter(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanBeamExitDiameter): Cannot get exit diameter for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).exitDiameter;
}

float LiDARcloud::getScanBeamDivergence(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanBeamDivergence): Cannot get beam divergence for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).beamDivergence;
}

float LiDARcloud::getScanRangeNoiseStdDev(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanRangeNoiseStdDev): Cannot get range noise standard deviation for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).rangeNoiseStdDev;
}

float LiDARcloud::getScanAngleNoiseStdDev(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanAngleNoiseStdDev): Cannot get angular noise standard deviation for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).angleNoiseStdDev;
}

float LiDARcloud::getScanTiltRoll(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanTiltRoll): Cannot get scanner tilt roll for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).scanTilt_roll;
}

float LiDARcloud::getScanTiltPitch(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanTiltPitch): Cannot get scanner tilt pitch for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).scanTilt_pitch;
}

float LiDARcloud::getScanAzimuthOffset(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanAzimuthOffset): Cannot get scanner azimuth offset for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).scanTilt_azimuth;
}

std::vector<std::string> LiDARcloud::getScanColumnFormat(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanColumnFormat): Cannot get column format for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).columnFormat;
}

ScanPattern LiDARcloud::getScanPattern(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanPattern): Cannot get scan pattern for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).scanPattern;
}

std::vector<float> LiDARcloud::getScanBeamZenithAngles(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanBeamZenithAngles): Cannot get beam zenith angles for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).beamZenithAngles;
}

helios::vec3 LiDARcloud::getHitXYZ(uint index) const {

    if (index >= hits.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getHitXYZ): Hit point index out of bounds. Requesting hit #" + std::to_string(index) + " but scan only has " + std::to_string(hits.size()) + " hits.");
    }

    return hits.at(index).position;
}

helios::vec3 LiDARcloud::getHitOrigin(uint index) const {

    if (index >= hits.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getHitOrigin): Hit point index out of bounds. Requesting hit #" + std::to_string(index) + " but scan only has " + std::to_string(hits.size()) + " hits.");
    }

    // Moving-platform scans store the per-pulse emission origin on each hit. Static scans do not, so fall back to the
    // single scan origin.
    if (doesHitDataExist(index, "origin_x") && doesHitDataExist(index, "origin_y") && doesHitDataExist(index, "origin_z")) {
        return helios::make_vec3(float(getHitData(index, "origin_x")), float(getHitData(index, "origin_y")), float(getHitData(index, "origin_z")));
    }

    return getScanOrigin(uint(getHitScanID(index)));
}

helios::SphericalCoord LiDARcloud::getHitRaydir(uint index) const {

    if (index >= hits.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getHitRaydir): Hit point index out of bounds. Requesting hit #" + std::to_string(index) + " but scan only has " + std::to_string(hits.size()) + " hits.");
    }

    // Use the beam's own emission origin (per-pulse for moving-platform scans; the scan origin for static scans) so the
    // recovered ray direction is correct regardless of platform motion.
    vec3 direction_cart = getHitXYZ(index) - getHitOrigin(index);
    return cart2sphere(direction_cart);
}

void LiDARcloud::setHitData(uint index, const char *label, double value) {

    if (index >= hits.size()) {
        helios_runtime_error("ERROR (LiDARcloud::setHitScalarData): Hit point index out of bounds. Tried to set hit #" + std::to_string(index) + " but scan only has " + std::to_string(hits.size()) + " hits.");
    }

    hits.at(index).data[label] = value;
}

double LiDARcloud::getHitData(uint index, const char *label) const {

    if (index >= hits.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getHitData): Hit point index out of bounds. Requesting hit #" + std::to_string(index) + " but scan only has " + std::to_string(hits.size()) + " hits.");
    }

    std::map<std::string, double> hit_data = hits.at(index).data;
    if (hit_data.find(label) == hit_data.end()) {
        helios_runtime_error("ERROR (LiDARcloud::getHitData): Data value ``" + std::string(label) + "'' does not exist.");
    }

    return hit_data.at(label);
}

bool LiDARcloud::doesHitDataExist(uint index, const char *label) const {

    if (index >= hits.size()) {
        return false;
    }

    std::map<std::string, double> hit_data = hits.at(index).data;
    if (hit_data.find(label) == hit_data.end()) {
        return false;
    } else {
        return true;
    }
}

RGBcolor LiDARcloud::getHitColor(uint index) const {

    if (index >= hits.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getHitColor): Hit point index out of bounds. Requesting hit #" + std::to_string(index) + " but scan only has " + std::to_string(hits.size()) + " hits.");
    }

    return hits.at(index).color;
}

int LiDARcloud::getHitScanID(uint index) const {

    if (index >= hits.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getHitColor): Hit point index out of bounds. Requesting hit #" + std::to_string(index) + " but scan only has " + std::to_string(hits.size()) + " hits.");
    }

    return hits.at(index).scanID;
}

int LiDARcloud::getHitIndex(uint scanID, uint row, uint column) const {

    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::deleteHitPoint): Hit point cannot be deleted from scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    if (row >= getScanSizeTheta(scanID)) {
        helios_runtime_error("ERROR (LiDARcloud::getHitIndex): Row in scan data table out of range.");
    } else if (column >= getScanSizePhi(scanID)) {
        helios_runtime_error("ERROR (LiDARcloud::getHitIndex): Column in scan data table out of range.");
    }

    int hit = hit_tables.at(scanID).get(row, column);

    assert(hit < getScanSizeTheta(scanID) * getScanSizePhi(scanID));

    return hit;
}

int LiDARcloud::getHitGridCell(uint index) const {

    if (index >= hits.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getHitGridCell): Hit point index out of bounds. Requesting hit #" + std::to_string(index) + " but scan only has " + std::to_string(hits.size()) + " hits.");
    } else if (hits.at(index).gridcell == -2) {
        cerr << "WARNING (LiDARcloud::getHitGridCell): hit grid cell for point #" << index << " was never set.  Returning a value of `-1'.  Did you forget to call calculateHitGridCell[*] first?" << endl;
        return -1;
    }

    return hits.at(index).gridcell;
}

void LiDARcloud::setHitGridCell(uint index, int cell) {

    if (index >= hits.size()) {
        helios_runtime_error("ERROR (LiDARcloud::setHitGridCell): Hit point index out of bounds. Tried to set hit #" + std::to_string(index) + " but scan only has " + std::to_string(hits.size()) + " hits.");
    }

    hits.at(index).gridcell = cell;
}

namespace {
    //! Apply an in-place transform to a hit's per-pulse emission origin (data labels origin_x/origin_y/origin_z), if it
    //! carries one. Moving-platform hits store their own origin, which must be transformed together with the hit
    //! position so the two stay in the same coordinate frame. Static hits carry no such labels and are left unchanged.
    void transformHitOrigin(HitPoint &hit, const std::function<helios::vec3(const helios::vec3 &)> &transform) {
        auto itx = hit.data.find("origin_x");
        auto ity = hit.data.find("origin_y");
        auto itz = hit.data.find("origin_z");
        if (itx != hit.data.end() && ity != hit.data.end() && itz != hit.data.end()) {
            helios::vec3 o = helios::make_vec3(float(itx->second), float(ity->second), float(itz->second));
            o = transform(o);
            itx->second = o.x;
            ity->second = o.y;
            itz->second = o.z;
        }
    }

    //! Return a hit's per-pulse origin if it carries one (data labels origin_x/y/z), else the supplied fallback origin.
    helios::vec3 hitOriginOrFallback(const HitPoint &hit, const helios::vec3 &fallback) {
        auto itx = hit.data.find("origin_x");
        auto ity = hit.data.find("origin_y");
        auto itz = hit.data.find("origin_z");
        if (itx != hit.data.end() && ity != hit.data.end() && itz != hit.data.end()) {
            return helios::make_vec3(float(itx->second), float(ity->second), float(itz->second));
        }
        return fallback;
    }
} // namespace

void LiDARcloud::coordinateShift(const vec3 &shift) {

    for (auto &scan: scans) {
        scan.origin = scan.origin + shift;
    }

    for (auto &hit: hits) {
        hit.position = hit.position + shift;
        transformHitOrigin(hit, [&](const vec3 &o) { return o + shift; });
    }
}

void LiDARcloud::coordinateShift(uint scanID, const vec3 &shift) {

    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::coordinateShift): Cannot apply coordinate shift to scan " + std::to_string(scanID) + " because it does not exist.");
    }

    scans.at(scanID).origin = scans.at(scanID).origin + shift;

    for (auto &hit: hits) {
        if (hit.scanID == scanID) {
            hit.position = hit.position + shift;
            transformHitOrigin(hit, [&](const vec3 &o) { return o + shift; });
        }
    }
}

void LiDARcloud::coordinateRotation(const SphericalCoord &rotation) {

    for (auto &scan: scans) {
        scan.origin = rotatePoint(scan.origin, rotation);
    }

    for (auto &hit: hits) {
        hit.position = rotatePoint(hit.position, rotation);
        transformHitOrigin(hit, [&](const vec3 &o) { return rotatePoint(o, rotation); });
        // Recompute the stored ray direction from the hit's own (transformed) origin so it remains correct for moving scans.
        hit.direction = cart2sphere(hit.position - hitOriginOrFallback(hit, scans.at(hit.scanID).origin));
    }
}

void LiDARcloud::coordinateRotation(uint scanID, const SphericalCoord &rotation) {

    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::coordinateRotation): Cannot apply rotation to scan " + std::to_string(scanID) + " because it does not exist.");
    }

    scans.at(scanID).origin = rotatePoint(scans.at(scanID).origin, rotation);

    for (auto &hit: hits) {
        if (hit.scanID == scanID) {
            hit.position = rotatePoint(hit.position, rotation);
            transformHitOrigin(hit, [&](const vec3 &o) { return rotatePoint(o, rotation); });
            hit.direction = cart2sphere(hit.position - hitOriginOrFallback(hit, scans.at(scanID).origin));
        }
    }
}

void LiDARcloud::coordinateRotation(float rotation, const vec3 &line_base, const vec3 &line_direction) {

    for (auto &scan: scans) {
        scan.origin = rotatePointAboutLine(scan.origin, line_base, line_direction, rotation);
    }

    for (auto &hit: hits) {
        hit.position = rotatePointAboutLine(hit.position, line_base, line_direction, rotation);
        transformHitOrigin(hit, [&](const vec3 &o) { return rotatePointAboutLine(o, line_base, line_direction, rotation); });
        hit.direction = cart2sphere(hit.position - hitOriginOrFallback(hit, scans.at(hit.scanID).origin));
    }
}

uint LiDARcloud::getTriangleCount() const {
    return triangles.size();
}

std::size_t LiDARcloud::getTriangulationCandidateCount() const {
    return triangulation_candidate_count;
}

std::size_t LiDARcloud::getTriangulationDroppedByLmax() const {
    return triangulation_dropped_lmax;
}

std::size_t LiDARcloud::getTriangulationDroppedByAspect() const {
    return triangulation_dropped_aspect;
}

std::size_t LiDARcloud::getTriangulationDroppedByDegenerate() const {
    return triangulation_dropped_degenerate;
}

Triangulation LiDARcloud::getTriangle(uint index) const {
    if (index >= triangles.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getTriangle): Triangle index out of bounds. Tried to get triangle #" + std::to_string(index) + " but point cloud only has " + std::to_string(triangles.size()) + " triangles.");
    }

    return triangles.at(index);
}

void LiDARcloud::addHitsToVisualizer(Visualizer *visualizer, uint pointsize) const {
    addHitsToVisualizer(visualizer, pointsize, "");
}

void LiDARcloud::addHitsToVisualizer(Visualizer *visualizer, uint pointsize, const RGBcolor &point_color) const {

    if (printmessages && scans.size() == 0) {
        std::cout << "WARNING (LiDARcloud::addHitsToVisualizer): There are no scans in the point cloud, and thus there is no geometry to add...skipping." << std::endl;
        return;
    }

    for (uint i = 0; i < getHitCount(); i++) {
        vec3 center = getHitXYZ(i);

        visualizer->addPoint(center, point_color, pointsize, Visualizer::COORDINATES_CARTESIAN);
    }
}

void LiDARcloud::addHitsToVisualizer(Visualizer *visualizer, uint pointsize, const char *color_value) const {

    if (printmessages && scans.size() == 0) {
        std::cout << "WARNING (LiDARcloud::addHitsToVisualizer): There are no scans in the point cloud, and thus there is no geometry to add...skipping." << std::endl;
        return;
    }

    //-- hit points --//
    float minval = 1e9;
    float maxval = -1e9;
    if (strcmp(color_value, "gridcell") == 0) {
        minval = 0;
        maxval = getGridCellCount() - 1;
    } else if (strcmp(color_value, "") != 0) {
        for (uint i = 0; i < getHitCount(); i++) {
            if (doesHitDataExist(i, color_value)) {
                float data = float(getHitData(i, color_value));
                if (data < minval) {
                    minval = data;
                }
                if (data > maxval) {
                    maxval = data;
                }
            }
        }
    }

    RGBcolor color;
    Colormap cmap = visualizer->getCurrentColormap();
    if (minval != 1e9 && maxval != -1e9) {
        cmap.setRange(minval, maxval);
    }

    for (uint i = 0; i < getHitCount(); i++) {

        if (strcmp(color_value, "") == 0) {
            color = getHitColor(i);
        } else if (strcmp(color_value, "gridcell") == 0) {
            if (getHitGridCell(i) < 0) {
                color = RGB::red;
            } else {
                color = cmap.query(getHitGridCell(i));
            }
        } else {
            if (!doesHitDataExist(i, color_value)) {
                color = RGB::red;
            } else {
                float data = float(getHitData(i, color_value));
                color = cmap.query(data);
            }
        }

        vec3 center = getHitXYZ(i);

        visualizer->addPoint(center, color, pointsize, Visualizer::COORDINATES_CARTESIAN);
    }
}

void LiDARcloud::addGridToVisualizer(Visualizer *visualizer) const {

    if (printmessages && scans.size() == 0) {
        std::cout << "WARNING (LiDARcloud::addGridToVisualizer): There are no scans in the point cloud, and thus there is no geometry to add...skipping." << std::endl;
        return;
    }

    float minval = 1e9;
    float maxval = -1e9;
    for (uint i = 0; i < getGridCellCount(); i++) {
        float data = getCellLeafAreaDensity(i);
        if (data < minval) {
            minval = data;
        }
        if (data > maxval) {
            maxval = data;
        }
    }

    Colormap cmap = visualizer->getCurrentColormap();
    if (minval != 1e9 && maxval != -1e9) {
        cmap.setRange(minval, maxval);
    }

    vec3 origin;
    for (uint i = 0; i < getGridCellCount(); i++) {

        if (getCellLeafAreaDensity(i) == 0) {
            continue;
        }

        vec3 center = getCellCenter(i);

        vec3 anchor = getCellGlobalAnchor(i);

        SphericalCoord rotation = make_SphericalCoord(0, getCellRotation(i));

        center = rotatePointAboutLine(center, anchor, make_vec3(0, 0, 1), rotation.azimuth);
        vec3 size = getCellSize(i);

        RGBAcolor color = make_RGBAcolor(cmap.query(getCellLeafAreaDensity(i)), 0.5);

        visualizer->addVoxelByCenter(center, size, rotation, color, Visualizer::COORDINATES_CARTESIAN);

        origin = origin + center / float(getGridCellCount());
    }

    vec3 boxmin, boxmax;
    getHitBoundingBox(boxmin, boxmax);

    float R = 2.f * sqrt(pow(boxmax.x - boxmin.x, 2) + pow(boxmax.y - boxmin.y, 2) + pow(boxmax.z - boxmin.z, 2));
}

void LiDARcloud::addTrianglesToVisualizer(Visualizer *visualizer) const {

    if (printmessages && scans.size() == 0) {
        std::cout << "WARNING (LiDARcloud::addGeometryToVisualizer): There are no scans in the point cloud, and thus there is no geometry to add...skipping." << std::endl;
        return;
    }

    for (uint i = 0; i < triangles.size(); i++) {

        Triangulation tri = triangles.at(i);

        visualizer->addTriangle(tri.vertex0, tri.vertex1, tri.vertex2, tri.color, Visualizer::COORDINATES_CARTESIAN);
    }
}

void LiDARcloud::addTrianglesToVisualizer(Visualizer *visualizer, uint gridcell) const {

    if (printmessages && scans.size() == 0) {
        std::cout << "WARNING (LiDARcloud::addTrianglesToVisualizer): There are no scans in the point cloud, and thus there is no geometry to add...skipping." << std::endl;
        return;
    }

    for (uint i = 0; i < triangles.size(); i++) {

        Triangulation tri = triangles.at(i);

        if (tri.gridcell == gridcell) {
            visualizer->addTriangle(tri.vertex0, tri.vertex1, tri.vertex2, tri.color, Visualizer::COORDINATES_CARTESIAN);
        }
    }
}

void LiDARcloud::addGrid(const vec3 &center, const vec3 &size, const int3 &ndiv, float rotation) {
    if (size.x <= 0 || size.y <= 0 || size.z <= 0) {
        cerr << "failed.\n";
        helios_runtime_error("ERROR (LiDARcloud::addGrid): The grid cell size must be positive.");
    }

    if (ndiv.x <= 0 || ndiv.y <= 0 || ndiv.z <= 0) {
        cerr << "failed.\n";
        helios_runtime_error("ERROR (LiDARcloud::addGrid): The number of grid cells in each direction must be positive.");
    }

    // add cells to grid
    vec3 gsubsize = make_vec3(float(size.x) / float(ndiv.x), float(size.y) / float(ndiv.y), float(size.z) / float(ndiv.z));

    float x, y, z;
    uint count = 0;
    for (int k = 0; k < ndiv.z; k++) {
        z = -0.5f * float(size.z) + (float(k) + 0.5f) * float(gsubsize.z);
        for (int j = 0; j < ndiv.y; j++) {
            y = -0.5f * float(size.y) + (float(j) + 0.5f) * float(gsubsize.y);
            for (int i = 0; i < ndiv.x; i++) {
                x = -0.5f * float(size.x) + (float(i) + 0.5f) * float(gsubsize.x);

                vec3 subcenter = make_vec3(x, y, z);

                vec3 subcenter_rot = rotatePoint(subcenter, make_SphericalCoord(0, rotation * M_PI / 180.f));

                if (printmessages) {
                    cout << "Adding grid cell #" << count << " with center " << subcenter_rot.x + center.x << "," << subcenter_rot.y + center.y << "," << subcenter.z + center.z << " and size " << gsubsize.x << " x " << gsubsize.y << " x "
                         << gsubsize.z << endl;
                }

                addGridCell(subcenter + center, center, gsubsize, size, rotation * M_PI / 180.f, make_int3(i, j, k), ndiv);

                count++;
            }
        }
    }
}

void LiDARcloud::addGridWireFrametoVisualizer(Visualizer *visualizer, float linewidth_pixels) const {


    for (int i = 0; i < getGridCellCount(); i++) {
        helios::vec3 center = getCellCenter(i);
        helios::vec3 size = getCellSize(i);

        helios::vec3 boxmin, boxmax;
        boxmin = make_vec3(center.x - 0.5 * size.x, center.y - 0.5 * size.y, center.z - 0.5 * size.z);
        boxmax = make_vec3(center.x + 0.5 * size.x, center.y + 0.5 * size.y, center.z + 0.5 * size.z);

        // vertical edges of the cell
        visualizer->addLine(make_vec3(boxmin.x, boxmin.y, boxmin.z), make_vec3(boxmin.x, boxmin.y, boxmax.z), RGB::black, linewidth_pixels, Visualizer::COORDINATES_CARTESIAN);
        visualizer->addLine(make_vec3(boxmin.x, boxmax.y, boxmin.z), make_vec3(boxmin.x, boxmax.y, boxmax.z), RGB::black, linewidth_pixels, Visualizer::COORDINATES_CARTESIAN);
        visualizer->addLine(make_vec3(boxmax.x, boxmin.y, boxmin.z), make_vec3(boxmax.x, boxmin.y, boxmax.z), RGB::black, linewidth_pixels, Visualizer::COORDINATES_CARTESIAN);
        visualizer->addLine(make_vec3(boxmax.x, boxmax.y, boxmin.z), make_vec3(boxmax.x, boxmax.y, boxmax.z), RGB::black, linewidth_pixels, Visualizer::COORDINATES_CARTESIAN);

        // horizontal top edges
        visualizer->addLine(make_vec3(boxmin.x, boxmin.y, boxmax.z), make_vec3(boxmin.x, boxmax.y, boxmax.z), RGB::black, linewidth_pixels, Visualizer::COORDINATES_CARTESIAN);
        visualizer->addLine(make_vec3(boxmin.x, boxmin.y, boxmax.z), make_vec3(boxmax.x, boxmin.y, boxmax.z), RGB::black, linewidth_pixels, Visualizer::COORDINATES_CARTESIAN);
        visualizer->addLine(make_vec3(boxmax.x, boxmin.y, boxmax.z), make_vec3(boxmax.x, boxmax.y, boxmax.z), RGB::black, linewidth_pixels, Visualizer::COORDINATES_CARTESIAN);
        visualizer->addLine(make_vec3(boxmin.x, boxmax.y, boxmax.z), make_vec3(boxmax.x, boxmax.y, boxmax.z), RGB::black, linewidth_pixels, Visualizer::COORDINATES_CARTESIAN);

        // horizontal bottom edges
        visualizer->addLine(make_vec3(boxmin.x, boxmin.y, boxmin.z), make_vec3(boxmin.x, boxmax.y, boxmin.z), RGB::black, linewidth_pixels, Visualizer::COORDINATES_CARTESIAN);
        visualizer->addLine(make_vec3(boxmin.x, boxmin.y, boxmin.z), make_vec3(boxmax.x, boxmin.y, boxmin.z), RGB::black, linewidth_pixels, Visualizer::COORDINATES_CARTESIAN);
        visualizer->addLine(make_vec3(boxmax.x, boxmin.y, boxmin.z), make_vec3(boxmax.x, boxmax.y, boxmin.z), RGB::black, linewidth_pixels, Visualizer::COORDINATES_CARTESIAN);
        visualizer->addLine(make_vec3(boxmin.x, boxmax.y, boxmin.z), make_vec3(boxmax.x, boxmax.y, boxmin.z), RGB::black, linewidth_pixels, Visualizer::COORDINATES_CARTESIAN);
    }
}

void LiDARcloud::addLeafReconstructionToVisualizer(Visualizer *visualizer) const {

    size_t Ngroups = reconstructed_triangles.size();

    std::vector<helios::RGBcolor> ctable;
    std::vector<float> clocs;

    ctable.push_back(RGB::violet);
    ctable.push_back(RGB::blue);
    ctable.push_back(RGB::green);
    ctable.push_back(RGB::yellow);
    ctable.push_back(RGB::orange);
    ctable.push_back(RGB::red);

    clocs.push_back(0.f);
    clocs.push_back(0.2f);
    clocs.push_back(0.4f);
    clocs.push_back(0.6f);
    clocs.push_back(0.8f);
    clocs.push_back(1.f);

    Colormap colormap(ctable, clocs, 100, 0, Ngroups - 1);

    for (size_t g = 0; g < Ngroups; g++) {

        float randi = randu() * (Ngroups - 1);
        RGBcolor color = colormap.query(randi);

        for (size_t t = 0; t < reconstructed_triangles.at(g).size(); t++) {

            helios::vec3 v0 = reconstructed_triangles.at(g).at(t).vertex0;
            helios::vec3 v1 = reconstructed_triangles.at(g).at(t).vertex1;
            helios::vec3 v2 = reconstructed_triangles.at(g).at(t).vertex2;

            // RGBcolor color = reconstructed_triangles.at(g).at(t).color;

            visualizer->addTriangle(v0, v1, v2, color, Visualizer::COORDINATES_CARTESIAN);
        }
    }

    Ngroups = reconstructed_alphamasks_center.size();

    for (size_t g = 0; g < Ngroups; g++) {

        visualizer->addRectangleByCenter(reconstructed_alphamasks_center.at(g), reconstructed_alphamasks_size.at(g), reconstructed_alphamasks_rotation.at(g), reconstructed_alphamasks_maskfile.c_str(), Visualizer::COORDINATES_CARTESIAN);
    }
}

void LiDARcloud::addTrunkReconstructionToVisualizer(Visualizer *visualizer) const {

    size_t Ngroups = reconstructed_trunk_triangles.size();

    for (size_t g = 0; g < Ngroups; g++) {

        for (size_t t = 0; t < reconstructed_trunk_triangles.at(g).size(); t++) {

            helios::vec3 v0 = reconstructed_trunk_triangles.at(g).at(t).vertex0;
            helios::vec3 v1 = reconstructed_trunk_triangles.at(g).at(t).vertex1;
            helios::vec3 v2 = reconstructed_trunk_triangles.at(g).at(t).vertex2;

            RGBcolor color = reconstructed_trunk_triangles.at(g).at(t).color;

            visualizer->addTriangle(v0, v1, v2, color, Visualizer::COORDINATES_CARTESIAN);
        }
    }
}

void LiDARcloud::addTrunkReconstructionToVisualizer(Visualizer *visualizer, const RGBcolor &trunk_color) const {

    size_t Ngroups = reconstructed_trunk_triangles.size();

    for (size_t g = 0; g < Ngroups; g++) {

        for (size_t t = 0; t < reconstructed_trunk_triangles.at(g).size(); t++) {

            helios::vec3 v0 = reconstructed_trunk_triangles.at(g).at(t).vertex0;
            helios::vec3 v1 = reconstructed_trunk_triangles.at(g).at(t).vertex1;
            helios::vec3 v2 = reconstructed_trunk_triangles.at(g).at(t).vertex2;

            visualizer->addTriangle(v0, v1, v2, trunk_color, Visualizer::COORDINATES_CARTESIAN);
        }
    }
}

std::vector<uint> LiDARcloud::addLeafReconstructionToContext(Context *context) const {
    return addLeafReconstructionToContext(context, helios::make_int2(1, 1));
}

std::vector<uint> LiDARcloud::addLeafReconstructionToContext(Context *context, const int2 &subpatches) const {

    std::vector<uint> UUIDs;

    std::vector<uint> UUID_leaf_template;
    if (subpatches.x > 1 || subpatches.y > 1) {
        UUID_leaf_template = context->addTile(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), subpatches, reconstructed_alphamasks_maskfile.c_str());
    }

    size_t Ngroups = reconstructed_alphamasks_center.size();

    for (size_t g = 0; g < Ngroups; g++) {

        helios::RGBcolor color = helios::RGB::red;

        uint zone = reconstructed_alphamasks_gridcell.at(g);

        if (reconstructed_alphamasks_size.at(g).x > 0 && reconstructed_alphamasks_size.at(g).y > 0) {
            std::vector<uint> UUIDs_leaf;
            if (subpatches.x == 1 && subpatches.y == 1) {
                UUIDs_leaf.push_back(context->addPatch(reconstructed_alphamasks_center.at(g), reconstructed_alphamasks_size.at(g), reconstructed_alphamasks_rotation.at(g), reconstructed_alphamasks_maskfile.c_str()));
            } else {
                UUIDs_leaf = context->copyPrimitive(UUID_leaf_template);
                context->scalePrimitive(UUIDs_leaf, make_vec3(reconstructed_alphamasks_size.at(g).x, reconstructed_alphamasks_size.at(g).y, 1));
                context->rotatePrimitive(UUIDs_leaf, -reconstructed_alphamasks_rotation.at(g).elevation, "x");
                context->rotatePrimitive(UUIDs_leaf, -reconstructed_alphamasks_rotation.at(g).azimuth, "z");
                context->translatePrimitive(UUIDs_leaf, reconstructed_alphamasks_center.at(g));
            }
            context->setPrimitiveData(UUIDs_leaf, "gridCell", zone);
            uint flag = reconstructed_alphamasks_direct_flag.at(g);
            context->setPrimitiveData(UUIDs_leaf, "directFlag", flag);
            UUIDs.insert(UUIDs.end(), UUIDs_leaf.begin(), UUIDs_leaf.end());
        }
    }

    context->deletePrimitive(UUID_leaf_template);

    return UUIDs;
}

std::vector<uint> LiDARcloud::addReconstructedTriangleGroupsToContext(helios::Context *context) const {

    std::vector<uint> UUIDs;

    size_t Ngroups = reconstructed_triangles.size();

    for (size_t g = 0; g < Ngroups; g++) {

        int leafGroup = round(context->randu() * (Ngroups - 1));

        for (size_t t = 0; t < reconstructed_triangles.at(g).size(); t++) {

            helios::vec3 v0 = reconstructed_triangles.at(g).at(t).vertex0;
            helios::vec3 v1 = reconstructed_triangles.at(g).at(t).vertex1;
            helios::vec3 v2 = reconstructed_triangles.at(g).at(t).vertex2;

            RGBcolor color = reconstructed_triangles.at(g).at(t).color;

            UUIDs.push_back(context->addTriangle(v0, v1, v2, color));

            uint zone = reconstructed_triangles.at(g).at(t).gridcell;
            context->setPrimitiveData(UUIDs.back(), "gridCell", zone);

            context->setPrimitiveData(UUIDs.back(), "leafGroup", leafGroup);
        }
    }

    return UUIDs;
}

std::vector<uint> LiDARcloud::addTrunkReconstructionToContext(Context *context) const {

    std::vector<uint> UUIDs;

    size_t Ngroups = reconstructed_trunk_triangles.size();

    for (size_t g = 0; g < Ngroups; g++) {

        for (size_t t = 0; t < reconstructed_trunk_triangles.at(g).size(); t++) {

            helios::vec3 v0 = reconstructed_trunk_triangles.at(g).at(t).vertex0;
            helios::vec3 v1 = reconstructed_trunk_triangles.at(g).at(t).vertex1;
            helios::vec3 v2 = reconstructed_trunk_triangles.at(g).at(t).vertex2;

            RGBcolor color = reconstructed_trunk_triangles.at(g).at(t).color;

            UUIDs.push_back(context->addTriangle(v0, v1, v2, color));
        }
    }

    return UUIDs;
}

void LiDARcloud::getHitBoundingBox(helios::vec3 &boxmin, helios::vec3 &boxmax) const {

    if (printmessages && hits.size() == 0) {
        std::cout << "WARNING (getHitBoundingBox): There are no hit points in the point cloud, cannot determine bounding box...skipping." << std::endl;
        return;
    }

    boxmin = make_vec3(1e6, 1e6, 1e6);
    boxmax = make_vec3(-1e6, -1e6, -1e6);

    for (std::size_t i = 0; i < hits.size(); i++) {

        vec3 xyz = getHitXYZ(i);

        if (xyz.x < boxmin.x) {
            boxmin.x = xyz.x;
        }
        if (xyz.x > boxmax.x) {
            boxmax.x = xyz.x;
        }
        if (xyz.y < boxmin.y) {
            boxmin.y = xyz.y;
        }
        if (xyz.y > boxmax.y) {
            boxmax.y = xyz.y;
        }
        if (xyz.z < boxmin.z) {
            boxmin.z = xyz.z;
        }
        if (xyz.z > boxmax.z) {
            boxmax.z = xyz.z;
        }
    }
}

void LiDARcloud::getGridBoundingBox(helios::vec3 &boxmin, helios::vec3 &boxmax) const {

    if (printmessages && getGridCellCount() == 0) {
        std::cout << "WARNING (getGridBoundingBox): There are no grid cells in the point cloud, cannot determine bounding box...skipping." << std::endl;
        return;
    }

    boxmin = make_vec3(1e6, 1e6, 1e6);
    boxmax = make_vec3(-1e6, -1e6, -1e6);

    std::size_t count = 0;
    for (uint c = 0; c < getGridCellCount(); c++) {

        vec3 center = getCellCenter(c);
        vec3 size = getCellSize(c);
        vec3 cellanchor = getCellGlobalAnchor(c);
        float rotation = getCellRotation(c);

        vec3 xyz_min = center - 0.5f * size;
        xyz_min = rotatePointAboutLine(xyz_min, cellanchor, make_vec3(0, 0, 1), rotation);
        vec3 xyz_max = center + 0.5f * size;
        xyz_max = rotatePointAboutLine(xyz_max, cellanchor, make_vec3(0, 0, 1), rotation);

        if (xyz_min.x < boxmin.x) {
            boxmin.x = xyz_min.x;
        }
        if (xyz_max.x > boxmax.x) {
            boxmax.x = xyz_max.x;
        }
        if (xyz_min.y < boxmin.y) {
            boxmin.y = xyz_min.y;
        }
        if (xyz_max.y > boxmax.y) {
            boxmax.y = xyz_max.y;
        }
        if (xyz_min.z < boxmin.z) {
            boxmin.z = xyz_min.z;
        }
        if (xyz_max.z > boxmax.z) {
            boxmax.z = xyz_max.z;
        }
    }
}

void LiDARcloud::distanceFilter(float maxdistance) {

    std::size_t delete_count = 0;
    for (int i = (getHitCount() - 1); i >= 0; i--) {

        vec3 xyz = getHitXYZ(i);
        // Range is measured from the beam's own emission origin (per-pulse for moving scans, scan origin for static).
        vec3 r = xyz - getHitOrigin(i);

        if (r.magnitude() > maxdistance) {
            deleteHitPoint(i);
            delete_count++;
        }
    }

    if (printmessages) {
        std::cout << "Removed " << delete_count << " hit points based on distance filter." << std::endl;
    }
}

void LiDARcloud::reflectanceFilter(float minreflectance) {

    std::size_t delete_count = 0;
    for (int r = (getHitCount() - 1); r >= 0; r--) {
        if (hits.at(r).data.find("reflectance") != hits.at(r).data.end()) {
            double R = getHitData(r, "reflectance");
            if (R < minreflectance) {
                deleteHitPoint(r);
                delete_count++;
            }
        }
    }

    if (printmessages) {
        std::cout << "Removed " << delete_count << " hit points based on reflectance filter." << std::endl;
    }
}

void LiDARcloud::scalarFilter(const char *scalar_field, float threshold, const char *comparator) {

    std::size_t delete_count = 0;
    for (int r = (getHitCount() - 1); r >= 0; r--) {
        if (hits.at(r).data.find(scalar_field) != hits.at(r).data.end()) {
            double R = getHitData(r, scalar_field);
            if (strcmp(comparator, "<") == 0) {
                if (R < threshold) {
                    deleteHitPoint(r);
                    delete_count++;
                }
            } else if (strcmp(comparator, ">") == 0) {
                if (R > threshold) {
                    deleteHitPoint(r);
                    delete_count++;
                }
            } else if (strcmp(comparator, "=") == 0) {
                if (R == threshold) {
                    deleteHitPoint(r);
                    delete_count++;
                }
            }
        }
    }

    if (printmessages) {
        std::cout << "Removed " << delete_count << " hit points based on scalar filter." << std::endl;
    }
}

void LiDARcloud::xyzFilter(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax) {

    xyzFilter(xmin, xmax, ymin, ymax, zmin, zmax, true);
}

void LiDARcloud::xyzFilter(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, bool deleteOutside) {

    if (xmin > xmax || ymin > ymax || zmin > zmax) {
        std::cout << "WARNING: at least one minimum value provided is greater than one maximum value. " << std::endl;
    }

    std::size_t delete_count = 0;

    if (deleteOutside) {
        for (int i = (getHitCount() - 1); i >= 0; i--) {
            vec3 xyz = getHitXYZ(i);

            if (xyz.x < xmin || xyz.x > xmax || xyz.y < ymin || xyz.y > ymax || xyz.z < zmin || xyz.z > zmax) {
                deleteHitPoint(i);
                delete_count++;
            }
        }
    } else {
        for (int i = (getHitCount() - 1); i >= 0; i--) {
            vec3 xyz = getHitXYZ(i);

            if (xyz.x >= xmin && xyz.x < xmax && xyz.y > ymin && xyz.y < ymax && xyz.z > zmin && xyz.z < zmax) {
                deleteHitPoint(i);
                delete_count++;
            }
        }
    }


    if (printmessages) {
        std::cout << "Removed " << delete_count << " hit points based on provided bounding box." << std::endl;
    }
}

// bool sortcol0( const std::vector<float>& v0, const std::vector<float>& v1 ){
//   return v0.at(0)<v1.at(0);
// }

// bool sortcol1( const std::vector<float>& v0, const std::vector<float>& v1 ){
//   return v0.at(1)<v1.at(1);
// }

bool sortcol0(const std::vector<double> &v0, const std::vector<double> &v1) {
    return v0.at(0) < v1.at(0);
}

bool sortcol1(const std::vector<double> &v0, const std::vector<double> &v1) {
    return v0.at(1) < v1.at(1);
}

namespace {

    //! Median of a vector of doubles (modifies the input via nth_element). Caller guarantees non-empty.
    double median_double(std::vector<double> &v) {
        const size_t n = v.size();
        const size_t mid = n / 2;
        std::nth_element(v.begin(), v.begin() + mid, v.end());
        const double hi = v.at(mid);
        if (n % 2 == 1) {
            return hi;
        }
        // even count: average the two central order statistics
        const double lo = *std::max_element(v.begin(), v.begin() + mid);
        return 0.5 * (lo + hi);
    }

    //! Robust line fit y = slope*x + intercept using the Theil-Sen estimator (median of pairwise slopes).
    /**
     * The slope is the median over all sample pairs of (y_j - y_i)/(x_j - x_i); the intercept is the
     * median of (y_i - slope*x_i). Theil-Sen has a ~29% breakdown point and requires no tuning, which
     * makes it well suited to noisy encoder angles with occasional gross outliers. Pairs with x_i == x_j
     * are skipped. To bound the O(n^2) pair count, when the sample is large a deterministic strided subset
     * of pairs is used (fixed stride, not RNG) so results are reproducible.
     *
     * \param[in] x Predictor samples (e.g. column index).
     * \param[in] y Response samples (e.g. measured azimuth).
     * \param[out] slope Fitted slope.
     * \param[out] intercept Fitted intercept.
     * \return true if a fit was produced (>=2 samples with distinct x), false otherwise.
     */
    bool theilSenFit(const std::vector<double> &x, const std::vector<double> &y, double &slope, double &intercept) {
        const size_t n = x.size();
        if (n < 2) {
            return false;
        }

        // Cap the number of point pairs considered. The full estimator is O(n^2); above this many samples
        // we walk pairs with a deterministic stride so the slope/intercept remain reproducible run-to-run.
        const size_t pair_cap = 1000;
        const size_t stride = (n > pair_cap) ? (n / pair_cap) : 1;

        const size_t n_strided = n / stride + 1; // approximate number of sampled indices
        std::vector<double> slopes;
        slopes.reserve(n_strided * n_strided / 2);
        for (size_t i = 0; i < n; i += stride) {
            for (size_t j = i + 1; j < n; j += stride) {
                const double dx = x.at(j) - x.at(i);
                if (dx == 0.0) {
                    continue;
                }
                slopes.push_back((y.at(j) - y.at(i)) / dx);
            }
        }

        if (slopes.empty()) {
            return false;
        }

        slope = median_double(slopes);

        std::vector<double> intercepts;
        intercepts.reserve(n);
        for (size_t i = 0; i < n; i++) {
            intercepts.push_back(y.at(i) - slope * x.at(i));
        }
        intercept = median_double(intercepts);

        return true;
    }

} // namespace

void LiDARcloud::maxPulseFilter(const char *scalar) {

    if (printmessages) {
        std::cout << "Filtering point cloud by maximum " << scalar << " per pulse..." << std::flush;
    }

    std::vector<std::vector<double>> timestamps;
    timestamps.resize(getHitCount());

    std::size_t delete_count = 0;
    for (std::size_t r = 0; r < getHitCount(); r++) {

        if (!doesHitDataExist(r, "timestamp")) {
            helios_runtime_error("ERROR (LiDARcloud::maxPulseFilter): Hit point " + std::to_string(r) + " does not have scalar data 'timestamp', which is required for max pulse filtering.");
        } else if (!doesHitDataExist(r, scalar)) {
            helios_runtime_error("ERROR (LiDARcloud::maxPulseFilter): Hit point " + std::to_string(r) + " does not have scalar data '" + scalar + "', which is required for max pulse filtering.");
        }

        // Store the original hit index as double(r): see minPulseFilter for the precision rationale.
        std::vector<double> v{getHitData(r, "timestamp"), getHitData(r, scalar), double(r)};

        timestamps.at(r) = v;
    }

    std::sort(timestamps.begin(), timestamps.end(), sortcol0);

    std::vector<std::vector<double>> isort;
    std::vector<int> to_delete;
    double time_old = timestamps.at(0).at(0);
    for (std::size_t r = 0; r < timestamps.size(); r++) {

        if (timestamps.at(r).at(0) != time_old) {

            if (isort.size() > 1) {

                std::sort(isort.begin(), isort.end(), sortcol1);

                for (int i = 0; i < isort.size() - 1; i++) {
                    to_delete.push_back(int(isort.at(i).at(2)));
                }
            }

            isort.resize(0);
            time_old = timestamps.at(r).at(0);
        }

        isort.push_back(timestamps.at(r));
    }

    std::sort(to_delete.begin(), to_delete.end());

    for (int i = to_delete.size() - 1; i >= 0; i--) {
        deleteHitPoint(to_delete.at(i));
    }

    if (printmessages) {
        std::cout << "done." << std::endl;
    }
}

void LiDARcloud::minPulseFilter(const char *scalar) {

    if (printmessages) {
        std::cout << "Filtering point cloud by minimum " << scalar << " per pulse..." << std::flush;
    }

    std::vector<std::vector<double>> timestamps;
    timestamps.resize(getHitCount());

    std::size_t delete_count = 0;
    for (std::size_t r = 0; r < getHitCount(); r++) {

        if (!doesHitDataExist(r, "timestamp")) {
            helios_runtime_error("ERROR (LiDARcloud::minPulseFilter): Hit point " + std::to_string(r) + " does not have scalar data 'timestamp', which is required for min pulse filtering.");
        } else if (!doesHitDataExist(r, scalar)) {
            helios_runtime_error("ERROR (LiDARcloud::minPulseFilter): Hit point " + std::to_string(r) + " does not have scalar data '" + scalar + "', which is required for min pulse filtering.");
        }

        // Store the original hit index as double(r), not float(r): the index round-trips
        // through this double vector and is read back via int(...). float has only 24 bits of
        // mantissa, so indices above 2^24 would alias and corrupt the delete mapping.
        std::vector<double> v{getHitData(r, "timestamp"), getHitData(r, scalar), double(r)};

        timestamps.at(r) = v;
    }

    std::sort(timestamps.begin(), timestamps.end(), sortcol0);

    std::vector<std::vector<double>> isort;
    std::vector<int> to_delete;
    double time_old = timestamps.at(0).at(0);
    for (std::size_t r = 0; r < timestamps.size(); r++) {

        if (timestamps.at(r).at(0) != time_old) {

            if (isort.size() > 1) {

                std::sort(isort.begin(), isort.end(), sortcol1);

                for (int i = 1; i < isort.size(); i++) {
                    to_delete.push_back(int(isort.at(i).at(2)));
                }
            }

            isort.resize(0);
            time_old = timestamps.at(r).at(0);
        }

        isort.push_back(timestamps.at(r));
    }

    std::sort(to_delete.begin(), to_delete.end());

    for (int i = to_delete.size() - 1; i >= 0; i--) {
        deleteHitPoint(to_delete.at(i));
    }

    if (printmessages) {
        std::cout << "done." << std::endl;
    }
}

void LiDARcloud::firstHitFilter() {

    if (printmessages) {
        std::cout << "Filtering point cloud to only first hits per pulse..." << std::flush;
    }

    std::vector<float> target_index;
    target_index.resize(getHitCount());
    int min_tindex = 1;

    for (std::size_t r = 0; r < target_index.size(); r++) {

        if (!doesHitDataExist(r, "target_index")) {
            std::cerr << "failed\nERROR (LiDARcloud::firstHitFilter): Hit point " << r
                      << " does not have scalar data "
                         "target_index"
                         ". No filtering will be performed."
                      << std::endl;
            return;
        }

        target_index.at(r) = getHitData(r, "target_index");

        if (target_index.at(r) == 0) {
            min_tindex = 0;
        }
    }

    for (int r = target_index.size() - 1; r >= 0; r--) {

        if (target_index.at(r) != min_tindex) {
            deleteHitPoint(r);
        }
    }

    if (printmessages) {
        std::cout << "done." << std::endl;
    }
}

void LiDARcloud::lastHitFilter() {

    if (printmessages) {
        std::cout << "Filtering point cloud to only last hits per pulse..." << std::flush;
    }

    std::vector<float> target_index;
    target_index.resize(getHitCount());
    int min_tindex = 1;

    for (std::size_t r = 0; r < target_index.size(); r++) {

        if (!doesHitDataExist(r, "target_index")) {
            std::cout << "failed\n";
            std::cerr << "ERROR (LiDARcloud::lastHitFilter): Hit point " << r
                      << " does not have scalar data "
                         "target_index"
                         ". No filtering will be performed."
                      << std::endl;
            return;
        } else if (!doesHitDataExist(r, "target_count")) {
            std::cout << "failed\n";
            std::cerr << "ERROR (LiDARcloud::lastHitFilter): Hit point " << r
                      << " does not have scalar data "
                         "target_count"
                         ". No filtering will be performed."
                      << std::endl;
            return;
        }

        target_index.at(r) = getHitData(r, "target_index");

        if (target_index.at(r) == 0) {
            min_tindex = 0;
        }
    }

    for (int r = target_index.size() - 1; r >= 0; r--) {

        float target_count = getHitData(r, "target_count");

        if (target_index.at(r) == target_count - 1 + min_tindex) {
            deleteHitPoint(r);
        }
    }

    if (printmessages) {
        std::cout << "done." << std::endl;
    }
}

std::vector<helios::vec3> LiDARcloud::gapfillMisses_rowcolumn(uint scanID, const bool add_flags) {

    if (printmessages) {
        std::cout << "Gap filling complete misses in scan " << scanID << " using row/column indices..." << std::flush;
    }

    // The row/column path places filled misses from a single static scan origin and carries no per-point time from
    // which a moving platform's pose could be recovered, so it cannot correctly place misses for a moving-platform
    // scan. Moving scans carry per-pulse timestamps and are handled by the timestamp path (gapfillMisses_timestamp);
    // fail fast here rather than synthesize misses at the wrong origin.
    if (scans.at(scanID).isMoving) {
        helios_runtime_error("ERROR (LiDARcloud::gapfillMisses): the row/column gap-filling path does not support moving-platform scans (see addScanMoving). A moving scan should be gap-filled via its per-pulse timestamps; ensure the scan "
                             "data carries 'timestamp' (and not 'row'/'column') so the timestamp-based path is used.");
    }

    const float gap_distance = LIDAR_MISS_DISTANCE; // place gapfilled miss points at the canonical miss distance
    const helios::vec3 origin = getScanOrigin(scanID);
    const int Ntheta = (int) scans.at(scanID).Ntheta;
    const int Nphi = (int) scans.at(scanID).Nphi;

    std::vector<helios::vec3> xyz_filled;

    // ---- 1. Collect this scan's returns that carry row/column indices ---- //
    // Per row, accumulate the measured (column, zenith, azimuth) of each return. The measured direction comes
    // from getHitRaydir(), so the fit is grounded in the actual beam geometry (including tilt and sweep), not
    // the idealized rc2direction model.
    std::vector<std::vector<double>> row_cols(Ntheta); // column index per return, bucketed by row
    std::vector<std::vector<double>> row_zeniths(Ntheta); // measured zenith per return, bucketed by row
    std::vector<std::vector<double>> row_azimuths(Ntheta); // measured (unwrapped later) azimuth per return, bucketed by row
    std::set<std::pair<int, int>> occupied; // (row,column) cells that already contain a return

    for (size_t r = 0; r < getHitCount(); r++) {
        if (getHitScanID(r) != (int) scanID) {
            continue;
        }

        // Canonical miss flag: existing points are returns unless already flagged as misses.
        if (!doesHitDataExist(r, "is_miss")) {
            setHitData(r, "is_miss", 0.0);
        }
        if (add_flags) {
            setHitData(r, "gapfillMisses_code", 0.0); // 0 = original point
        }

        if (!doesHitDataExist(r, "row") || !doesHitDataExist(r, "column")) {
            continue;
        }

        const int row = (int) std::lround(getHitData(r, "row"));
        const int col = (int) std::lround(getHitData(r, "column"));
        if (row < 0 || row >= Ntheta || col < 0 || col >= Nphi) {
            continue; // index out of declared scan grid; ignore
        }

        const helios::SphericalCoord raydir = getHitRaydir(r);
        row_cols.at(row).push_back((double) col);
        row_zeniths.at(row).push_back(raydir.zenith);
        row_azimuths.at(row).push_back(raydir.azimuth);
        occupied.insert(std::make_pair(row, col));
    }

    // ---- 2. Per-row robust fit of the generative model ---- //
    // For each row with enough returns: zenith[row] = median(zeniths); azimuth = intercept[row] + slope[row]*column
    // via Theil-Sen. The azimuth samples in a row are unwrapped about their median first so the 0/2pi seam does
    // not corrupt the slope fit.
    const int min_returns_for_fit = 4; // K: rows with fewer returns are filled by cross-row extrapolation
    std::vector<double> zenith_lut(Ntheta, 0.0);
    std::vector<double> az_intercept_lut(Ntheta, 0.0);
    std::vector<double> az_slope_lut(Ntheta, 0.0);
    std::vector<bool> row_fitted(Ntheta, false);

    for (int row = 0; row < Ntheta; row++) {
        if ((int) row_cols.at(row).size() < min_returns_for_fit) {
            continue;
        }

        // robust zenith for this row
        std::vector<double> zeniths_copy = row_zeniths.at(row);
        const double zen = median_double(zeniths_copy);

        // unwrap azimuths about a robust center so the seam does not split the samples
        std::vector<double> az_center_copy = row_azimuths.at(row);
        const double az_center = median_double(az_center_copy);
        std::vector<double> az_unwrapped = row_azimuths.at(row);
        for (double &a: az_unwrapped) {
            while (a - az_center > M_PI) {
                a -= 2.0 * M_PI;
            }
            while (a - az_center < -M_PI) {
                a += 2.0 * M_PI;
            }
        }

        double slope = 0.0, intercept = 0.0;
        if (!theilSenFit(row_cols.at(row), az_unwrapped, slope, intercept)) {
            continue; // e.g. all returns in one column; treat as unfitted, extrapolate later
        }

        zenith_lut.at(row) = zen;
        az_slope_lut.at(row) = slope;
        az_intercept_lut.at(row) = intercept;
        row_fitted.at(row) = true;
    }

    // ---- 3. Extrapolate the per-row model across the row axis to cover sparse/empty rows ---- //
    // Collect the directly-fitted rows and robustly fit zenith-vs-row and intercept-vs-row (Theil-Sen). The
    // azimuth slope (sweep rate) is approximately constant across rows, so its robust median over fitted rows
    // is used. Evaluating these across-row fits at every row index defines a complete model over the whole grid,
    // including blank near-zenith rows (extrapolation).
    std::vector<double> fitted_row_idx, fitted_zenith, fitted_intercept, fitted_slope;
    for (int row = 0; row < Ntheta; row++) {
        if (row_fitted.at(row)) {
            fitted_row_idx.push_back((double) row);
            fitted_zenith.push_back(zenith_lut.at(row));
            fitted_intercept.push_back(az_intercept_lut.at(row));
            fitted_slope.push_back(az_slope_lut.at(row));
        }
    }

    if ((int) fitted_row_idx.size() < 2) {
        helios_runtime_error("ERROR (LiDARcloud::gapfillMisses): scan " + std::to_string(scanID) + " has too few populated scan rows (" + std::to_string(fitted_row_idx.size()) +
                             ") to robustly reconstruct the row/column scan-grid model. At least 2 rows with >= " + std::to_string(min_returns_for_fit) + " returns are required.");
    }

    double zen_slope = 0.0, zen_intercept = 0.0;
    double int_slope = 0.0, int_intercept = 0.0;
    const bool zen_ok = theilSenFit(fitted_row_idx, fitted_zenith, zen_slope, zen_intercept);
    const bool int_ok = theilSenFit(fitted_row_idx, fitted_intercept, int_slope, int_intercept);
    const double median_slope = median_double(fitted_slope);

    for (int row = 0; row < Ntheta; row++) {
        if (row_fitted.at(row)) {
            continue;
        }
        zenith_lut.at(row) = zen_ok ? (zen_intercept + zen_slope * (double) row) : fitted_zenith.front();
        az_intercept_lut.at(row) = int_ok ? (int_intercept + int_slope * (double) row) : fitted_intercept.front();
        az_slope_lut.at(row) = median_slope;
    }

    // ---- 4. Emit a miss for every empty grid cell along its reconstructed direction ---- //
    uint npoints_interior = 0;
    uint npoints_extrapolated = 0;
    for (int row = 0; row < Ntheta; row++) {
        for (int col = 0; col < Nphi; col++) {

            if (occupied.find(std::make_pair(row, col)) != occupied.end()) {
                continue; // cell already has a return
            }

            const double zenith = zenith_lut.at(row);
            double azimuth = az_intercept_lut.at(row) + az_slope_lut.at(row) * (double) col;
            // wrap azimuth to [0, 2pi)
            azimuth = std::fmod(azimuth, 2.0 * M_PI);
            if (azimuth < 0.0) {
                azimuth += 2.0 * M_PI;
            }

            const helios::SphericalCoord spherical(gap_distance, 0.5 * M_PI - (float) zenith, (float) azimuth);
            const helios::vec3 xyz = origin + helios::sphere2cart(spherical);
            xyz_filled.push_back(xyz);

            std::map<std::string, double> data;
            data.insert(std::make_pair("is_miss", 1.0)); // gapfilled points are misses (transmitted beams)
            data.insert(std::make_pair("row", (double) row));
            data.insert(std::make_pair("column", (double) col));
            data.insert(std::make_pair("nRaysHit", 0.0)); // a miss: zero sub-rays of the pulse returned a hit
            if (add_flags) {
                // 1 = interior gapfill (row had its own direct fit); 4 = extrapolated row (model came from cross-row fit)
                data.insert(std::make_pair("gapfillMisses_code", row_fitted.at(row) ? 1.0 : 4.0));
            }

            addHitPoint(scanID, xyz, spherical, data);
            if (row_fitted.at(row)) {
                npoints_interior++;
            } else {
                npoints_extrapolated++;
            }
        }
    }

    if (printmessages) {
        std::cout << "filled " << xyz_filled.size() << " points (" << npoints_interior << " interior, " << npoints_extrapolated << " extrapolated-row)." << std::endl;
    }

    return xyz_filled;
}

std::vector<helios::vec3> LiDARcloud::gapfillMisses() {
    std::vector<helios::vec3> xyz_filled;
    for (uint scanID = 0; scanID < getScanCount(); scanID++) {
        std::vector<helios::vec3> filled_this_scan = gapfillMisses(scanID, false, false);
        xyz_filled.insert(xyz_filled.end(), filled_this_scan.begin(), filled_this_scan.end());
    }
    return xyz_filled;
}

std::vector<helios::vec3> LiDARcloud::gapfillMisses(uint scanID) {
    return gapfillMisses(scanID, false, false);
}

std::vector<helios::vec3> LiDARcloud::gapfillMisses(uint scanID, const bool gapfill_grid_only, const bool add_flags) {

    // Validate scanID
    if (scanID >= getScanCount()) {
        helios_runtime_error("ERROR (LiDARcloud::gapfillMisses): Invalid scanID " + std::to_string(scanID) + ". Only " + std::to_string(getScanCount()) + " scans exist.");
    }

    // Auto-detect which reconstruction path to use based on the data available on this scan's returns.
    // The row/column path reconstructs miss directions from the native scan-grid indices and is robust to
    // scanner tilt, angular noise, and azimuth sweep; the timestamp path reconstructs the grid from per-hit
    // timestamps. When both are available we prefer row/column. When neither is available we fail fast, since
    // there is no way to reconstruct miss directions.
    bool has_rowcolumn = false;
    bool has_timestamp = false;
    size_t scan_hit_count = 0;
    for (size_t r = 0; r < getHitCount(); r++) {
        if (getHitScanID(r) != (int) scanID) {
            continue;
        }
        scan_hit_count++;
        if (doesHitDataExist(r, "row") && doesHitDataExist(r, "column")) {
            has_rowcolumn = true;
        }
        if (doesHitDataExist(r, "timestamp")) {
            has_timestamp = true;
        }
        if (has_rowcolumn && has_timestamp) {
            break; // both present; row/column will be preferred
        }
    }

    // A scan with no returns at all (e.g. rays that all missed empty geometry) has nothing to reconstruct;
    // return gracefully. The fail-fast below applies only when returns exist but carry neither timestamp nor
    // row/column indices, which is a genuine data-format problem.
    if (scan_hit_count == 0) {
        if (printmessages) {
            std::cout << "Gap filling complete misses in scan " << scanID << "...scan has no hits. Skipping gap fill." << std::endl;
        }
        return {};
    }

    if (has_rowcolumn) {
        return gapfillMisses_rowcolumn(scanID, add_flags);
    } else if (has_timestamp) {
        return gapfillMisses_timestamp(scanID, gapfill_grid_only, add_flags);
    } else {
        helios_runtime_error("ERROR (LiDARcloud::gapfillMisses): scan " + std::to_string(scanID) +
                             " has neither 'timestamp' nor 'row'/'column' hit data; cannot reconstruct miss directions. "
                             "Provide either per-hit timestamps or scan row/column indices.");
        return {}; // unreachable; silences compiler warning
    }
}

std::vector<helios::vec3> LiDARcloud::gapfillMisses_timestamp(uint scanID, const bool gapfill_grid_only, const bool add_flags) {

    if (printmessages) {
        std::cout << "Gap filling complete misses in scan " << scanID << "..." << std::flush;
    }

    float gap_distance = LIDAR_MISS_DISTANCE; // place gapfilled miss points at the canonical miss distance

    helios::vec3 origin = getScanOrigin(scanID);
    std::vector<helios::vec3> xyz_filled;

    // For a moving-platform scan each filled miss must be emitted from the platform pose at that miss's (interpolated)
    // timestamp, not from the single static origin. originAtTime() returns the per-pulse origin for moving scans and the
    // static scan origin otherwise; for moving scans the synthesized miss also stores its own origin_x/y/z so it carries
    // correct beam geometry into the leaf-area inversion (which reads getHitOrigin()).
    const ScanMetadata &gapfill_scan = scans.at(scanID);
    const bool gapfill_is_moving = gapfill_scan.isMoving;
    auto originAtTime = [&](double timestep) -> helios::vec3 {
        if (!gapfill_is_moving) {
            return origin;
        }
        helios::vec3 pos;
        helios::vec4 quat;
        gapfill_scan.poseAt(timestep, pos, quat);
        return pos + quat_rotate(quat, gapfill_scan.lever_arm);
    };

    // Populating a hit table for each scan:
    // Column 0 - hit index; Column 1 - timestamp; Column 2 - ray zenith; Column 3 - ray azimuth
    std::vector<std::vector<double>> hit_table;
    for (size_t r = 0; r < getHitCount(); r++) {
        if (getHitScanID(r) == scanID) {

            // Canonical miss flag: existing points are returns unless already flagged
            // as misses (e.g. imported misses from a miss-retaining format).
            if (!doesHitDataExist(r, "is_miss")) {
                setHitData(r, "is_miss", 0.0);
            }

            if (add_flags) {
                // gapfillMisses_code = 0: original points
                setHitData(r, "gapfillMisses_code", 0.0);
            }

            helios::SphericalCoord raydir = getHitRaydir(r);

            if (!doesHitDataExist(r, "timestamp")) {
                helios_runtime_error("ERROR (LiDARcloud::gapfillMisses): Hit " + std::to_string(r) + " is missing required 'timestamp' data. Cannot perform gap filling.");
            }

            double timestamp = getHitData(r, "timestamp");
            std::vector<double> data;
            data.resize(4);
            data.at(0) = float(r);
            data.at(1) = timestamp;
            data.at(2) = raydir.zenith;
            data.at(3) = raydir.azimuth;
            hit_table.push_back(data);
        }
    }

    // Check for empty scan
    if (hit_table.empty()) {
        if (printmessages) {
            std::cout << "scan has no hits. Skipping gap fill." << std::endl;
        }
        return xyz_filled; // Return empty vector
    }

    // sorting, initial dt and dtheta calculations, and determining minimum target index in the scan

    // sort the hit table by column 1 (timestamp)
    std::sort(hit_table.begin(), hit_table.end(), sortcol1);

    int min_tindex = 1;
    for (size_t r = 0; r < hit_table.size(); r++) {

        // this is to figure out if target indexing uses 0 or 1 offset
        if (min_tindex == 1 && doesHitDataExist(hit_table.at(r).at(0), "target_index") && doesHitDataExist(hit_table.at(r).at(0), "target_count")) {
            if (getHitData(hit_table.at(r).at(0), "target_index") == 0) {
                min_tindex = 0;
            }
        }
    }

    // getting rid of points with target index greater than the minimum

    int ndup_target = 0;
    // create new array without duplicate timestamps (keep only first hits). Iterate the full
    // table - including the last element - so the target_index filter is applied uniformly; a
    // previous version excluded the last element from the loop and then appended it unfiltered,
    // which could admit a non-first-hit return.
    std::vector<std::vector<double>> hit_table_semiclean;
    for (size_t r = 0; r < hit_table.size(); r++) {

        // only consider first hits
        if (doesHitDataExist(hit_table.at(r).at(0), "target_index") && doesHitDataExist(hit_table.at(r).at(0), "target_count")) {
            if (getHitData(hit_table.at(r).at(0), "target_index") > min_tindex) {
                ndup_target++;
                continue;
            }
        }

        hit_table_semiclean.push_back(hit_table.at(r));
    }

    //  re-calculating dt

    std::vector<double> dt_semiclean;
    dt_semiclean.resize(hit_table_semiclean.size(), 0.0);
    for (size_t r = 0; r + 1 < hit_table_semiclean.size(); r++) {

        dt_semiclean.at(r) = hit_table_semiclean.at(r + 1).at(1) - hit_table_semiclean.at(r).at(1);
        // set the hit index of the new array
        hit_table_semiclean.at(r).at(0) = r;
    }

    //  checking for duplicate timestamps in the remaining data

    int ndup = 0;
    // create new array without duplicate timestamps
    std::vector<std::vector<double>> hit_table_clean;
    for (size_t r = 0; r + 1 < hit_table_semiclean.size(); r++) {

        // if there are still rows with duplicate timestamps, it probably means there is no "target_index" column, but multiple hits per timestamp are still included
        // proceed using this assumption, just get rid of the rows where dt = 0 for simplicity (last hits probably are what remain).
        if (dt_semiclean.at(r) == 0) {
            ndup++;
            continue;
        }

        hit_table_clean.push_back(hit_table_semiclean.at(r));
    }

    // The 2D-grid reconstruction below requires at least two cleaned hits to
    // compute dt/dtheta between consecutive beams. Clouds loaded from ASCII files
    // without row/column indices can collapse to zero or one cleaned hit after the
    // duplicate-timestamp filter above; without this guard the subsequent
    // `size() - 1` style loops underflow and read out of bounds.
    if (hit_table_clean.size() < 2) {
        if (printmessages) {
            std::cout << "insufficient hits to reconstruct scan grid. Skipping gap fill." << std::endl;
        }
        return xyz_filled; // Return empty vector
    }

    // recalculate dt and dtheta with only one hit per beam
    // and calculate the minimum dt value
    std::vector<double> dt_clean;
    std::vector<float> dtheta_clean;
    dt_clean.resize(hit_table_clean.size(), 0.0);
    dtheta_clean.resize(hit_table_clean.size(), 0.f);

    double dt_clean_min = 1e6;
    for (size_t r = 0; r + 1 < hit_table_clean.size(); r++) {

        dt_clean.at(r) = hit_table_clean.at(r + 1).at(1) - hit_table_clean.at(r).at(1);
        dtheta_clean.at(r) = hit_table_clean.at(r + 1).at(2) - hit_table_clean.at(r).at(2);
        // set the hit index of the new array
        hit_table_clean.at(r).at(0) = r;

        if (dt_clean.at(r) < dt_clean_min) {
            dt_clean_min = dt_clean.at(r);
        }
    }

    // configuration of 2D map
    // reconfigure hit table into 2D (theta,phi) map
    std::vector<std::vector<std::vector<double>>> hit_table2D;

    int column = 0;
    hit_table2D.resize(1);
    for (size_t r = 0; r + 1 < hit_table_clean.size(); r++) {

        hit_table2D.at(column).push_back(hit_table_clean.at(r));
        // for small scans (like the rectangle test case, this needs to change to < 0 or some smaller angle (that is larger than noise))
        //  if( dtheta_clean.at(r) < 0 ){
        //  for normal scans, this threshold allows for 10 degrees drops in theta within a given sweep as noise. This can be adjusted as appropriate.
        if (dtheta_clean.at(r) < -0.1745329f) {
            column++;
            hit_table2D.resize(column + 1);
        }
    }

    // calculate average dt and dtheta for subsequent points

    // calculate average dt
    float dt_avg = 0;
    int dt_sum = 0;

    // calculate the average dtheta to use for extrapolation
    float dtheta_avg = 0;
    int dtheta_sum = 0;

    for (int j = 0; j < hit_table2D.size(); j++) {
        for (int i = 0; i < hit_table2D.at(j).size(); i++) {
            int r = int(hit_table2D.at(j).at(i).at(0));
            if (dt_clean.at(r) >= dt_clean_min && dt_clean.at(r) < 1.5 * dt_clean_min) {
                dt_avg += dt_clean.at(r);
                dt_sum++;

                // calculate the average dtheta to use for extrapolation
                dtheta_avg += dtheta_clean.at(r);
                dtheta_sum++;
            }
        }
    }

    if (dt_sum == 0 || dtheta_sum == 0) {
        if (printmessages) {
            std::cout << "insufficient valid hit pairs. Skipping gap fill." << std::endl;
        }
        return xyz_filled; // Return empty vector
    }

    dt_avg = dt_avg / float(dt_sum);
    // Calculate the average dtheta to use for extrapolation
    dtheta_avg = dtheta_avg / float(dtheta_sum);

    // dt_avg and dt_clean_min are used below as divisors and gap-spacing thresholds.
    // A degenerate scan grid (e.g. an ASCII cloud whose angular sampling can't be
    // reconstructed) can leave them zero, negative, or non-finite, which makes the
    // Ngap computation below blow up. Bail out rather than fill garbage.
    if (!std::isfinite(dt_avg) || dt_avg <= 0.f || !std::isfinite(dt_clean_min) || dt_clean_min <= 0.0) {
        if (printmessages) {
            std::cout << "degenerate timestamp spacing. Skipping gap fill." << std::endl;
        }
        return xyz_filled; // Return empty vector
    }

    // Get theta range for grid position calculations (needed early for filled_positions)
    helios::vec2 theta_range = getScanRangeTheta(scanID);

    // Track which grid positions have been filled (to avoid duplicates)
    std::set<std::pair<int, int>> filled_positions;

    // Pre-populate with existing hit positions using proper direction2rc conversion
    for (size_t r = 0; r < getHitCount(); r++) {
        if (getHitScanID(r) == scanID) {
            helios::SphericalCoord raydir = getHitRaydir(r);
            helios::int2 rc = scans.at(scanID).direction2rc(raydir);
            filled_positions.insert(std::make_pair(rc.x, rc.y));
        }
    }

    // identify gaps and fill
    for (int j = 0; j < hit_table2D.size(); j++) {

        if (hit_table2D.at(j).size() > 0) {
            for (size_t i = 0; i + 1 < hit_table2D.at(j).size(); i++) {

                double dt = hit_table2D.at(j).at(i + 1).at(1) - hit_table2D.at(j).at(i).at(1);

                if (dt > 1.5f * dt_clean_min) { // missing hit(s)

                    // calculate number of missing hits
                    int Ngap = round(dt / dt_avg) - 1;

                    // A gap can never span more beams than the scan has rows. Cap Ngap
                    // to guard against a runaway fill loop if the reconstructed grid
                    // spacing is much finer than the real one.
                    int Ngap_max = (int) scans.at(scanID).Ntheta;
                    if (Ngap > Ngap_max) {
                        Ngap = Ngap_max;
                    }

                    // fill missing points
                    for (int k = 1; k <= Ngap; k++) {

                        float timestep = hit_table2D.at(j).at(i).at(1) + dt_avg * float(k);

                        // interpolate theta and phi
                        float theta = hit_table2D.at(j).at(i).at(2) + (hit_table2D.at(j).at(i + 1).at(2) - hit_table2D.at(j).at(i).at(2)) * float(k) / float(Ngap + 1);
                        float phi = hit_table2D.at(j).at(i).at(3) + (hit_table2D.at(j).at(i + 1).at(3) - hit_table2D.at(j).at(i).at(3)) * float(k) / float(Ngap + 1);
                        // Wrap phi to [0, 2π] range
                        if (phi > 2.f * M_PI) {
                            phi = phi - 2.f * M_PI;
                        } else if (phi < 0.f) {
                            phi = phi + 2.f * M_PI;
                        }

                        // Convert to grid indices using proper direction2rc method
                        helios::SphericalCoord dir_to_check(gap_distance, 0.5 * M_PI - theta, phi);
                        helios::int2 rc = scans.at(scanID).direction2rc(dir_to_check);

                        // Skip grid positions outside the scan's row/column range
                        // (mirrors the bounds check in the extrapolation passes below).
                        if (rc.x < 0 || rc.x >= (int) scans.at(scanID).Ntheta || rc.y < 0 || rc.y >= (int) scans.at(scanID).Nphi) {
                            continue;
                        }

                        auto grid_key = std::make_pair(rc.x, rc.y);

                        // Only add if this grid position hasn't been filled yet
                        if (filled_positions.find(grid_key) == filled_positions.end()) {

                            helios::SphericalCoord spherical(gap_distance, 0.5 * M_PI - theta, phi);
                            helios::vec3 fill_origin = originAtTime(timestep);
                            helios::vec3 xyz = fill_origin + helios::sphere2cart(spherical);
                            xyz_filled.push_back(xyz);

                            std::map<std::string, double> data;
                            data.insert(std::pair<std::string, double>("timestamp", timestep));
                            data.insert(std::pair<std::string, double>("target_index", min_tindex));
                            data.insert(std::pair<std::string, double>("nRaysHit", 0.0)); // a miss: zero sub-rays of the pulse returned a hit
                            data.insert(std::pair<std::string, double>("is_miss", 1.0)); // gapfilled points are misses (transmitted beams)
                            if (gapfill_is_moving) {
                                // Store the per-pulse emission origin so the synthesized miss carries correct beam geometry (getHitOrigin).
                                data.insert(std::pair<std::string, double>("origin_x", fill_origin.x));
                                data.insert(std::pair<std::string, double>("origin_y", fill_origin.y));
                                data.insert(std::pair<std::string, double>("origin_z", fill_origin.z));
                            }
                            if (add_flags) {
                                // gapfillMisses_code = 1: gapfilled points
                                data.insert(std::pair<std::string, double>("gapfillMisses_code", 1.0));
                            }
                            addHitPoint(scanID, xyz, spherical, data);
                            filled_positions.insert(grid_key); // Mark as filled
                        }
                    }
                }
            }
        }
    }
    uint npointsfilled = xyz_filled.size();

    // Get actual grid spacing for proper edge extrapolation (theta_range already declared above)
    float grid_dtheta = (theta_range.y - theta_range.x) / float(scans.at(scanID).Ntheta - 1);
    float grid_dphi = (scans.at(scanID).phiMax - scans.at(scanID).phiMin) / float(scans.at(scanID).Nphi - 1);

    if (gapfill_grid_only == true) {
        // instead of extrapolating to the angle ranges given in the xml file, we can extrapolate to the angle range of the voxel grid to save time.
        //  to do this we loop through the vertices of the voxel grid.
        std::vector<helios::vec3> grid_vertices;
        helios::vec3 boxmin, boxmax;
        getGridBoundingBox(boxmin, boxmax); // axis aligned bounding box of all grid cells
        grid_vertices.push_back(boxmin);
        grid_vertices.push_back(boxmax);
        grid_vertices.push_back(helios::make_vec3(boxmin.x, boxmin.y, boxmax.z));
        grid_vertices.push_back(helios::make_vec3(boxmax.x, boxmax.y, boxmin.z));
        grid_vertices.push_back(helios::make_vec3(boxmin.x, boxmax.y, boxmin.z));
        grid_vertices.push_back(helios::make_vec3(boxmin.x, boxmax.y, boxmax.z));
        grid_vertices.push_back(helios::make_vec3(boxmax.x, boxmin.y, boxmin.z));
        grid_vertices.push_back(helios::make_vec3(boxmax.x, boxmin.y, boxmax.z));

        float max_theta = 0;
        float min_theta = M_PI;
        float max_phi = 0;
        float min_phi = 2 * M_PI;
        for (uint gg = 0; gg < grid_vertices.size(); gg++) {
            helios::vec3 direction_cart = grid_vertices.at(gg) - getScanOrigin(scanID);
            helios::SphericalCoord sc = cart2sphere(direction_cart);
            if (sc.azimuth < min_phi) {
                min_phi = sc.azimuth;
            }

            if (sc.azimuth > max_phi) {
                max_phi = sc.azimuth;
            }

            if (sc.zenith < min_theta) {
                min_theta = sc.zenith;
            }

            if (sc.zenith > max_theta) {
                max_theta = sc.zenith;
            }
        }

        // if the min or max theta is outside of the values provided in xml, use the xml values
        if (min_theta < theta_range.x) {
            min_theta = theta_range.x;
        }

        if (max_theta > theta_range.y) {
            max_theta = theta_range.y;
        }

        theta_range = helios::make_vec2(min_theta, max_theta);
    }

    // extrapolate missing points
    for (int j = 0; j < hit_table2D.size(); j++) {

        if (hit_table2D.at(j).size() > 0) {

            // upward edge points
            if (hit_table2D.at(j).front().at(2) > theta_range.x) {

                float dtheta = dtheta_avg;
                float theta = hit_table2D.at(j).at(0).at(2) - dtheta;
                // just use the last value of phi in the sweep
                float phi = hit_table2D.at(j).at(0).at(3);
                float timestep = hit_table2D.at(j).at(0).at(1) - dt_avg;
                if (dtheta == 0) {
                    continue;
                }

                while (theta > theta_range.x) {

                    // Convert to grid indices using proper direction2rc method
                    helios::SphericalCoord dir_to_check(gap_distance, 0.5 * M_PI - theta, phi);
                    helios::int2 rc = scans.at(scanID).direction2rc(dir_to_check);

                    // Only add if this grid position is actually empty (avoid duplicates)
                    if (rc.x >= 0 && rc.x < (int) scans.at(scanID).Ntheta && rc.y >= 0 && rc.y < (int) scans.at(scanID).Nphi) {

                        // Check if this grid position has already been filled (avoid duplicates)
                        auto grid_key = std::make_pair(rc.x, rc.y);
                        if (filled_positions.find(grid_key) == filled_positions.end()) {

                            helios::SphericalCoord spherical(gap_distance, 0.5 * M_PI - theta, phi);
                            helios::vec3 fill_origin = originAtTime(timestep);
                            helios::vec3 xyz = fill_origin + helios::sphere2cart(spherical);
                            xyz_filled.push_back(xyz);

                            std::map<std::string, double> data;
                            data.insert(std::pair<std::string, double>("timestamp", timestep));
                            data.insert(std::pair<std::string, double>("target_index", min_tindex));
                            data.insert(std::pair<std::string, double>("nRaysHit", 0.0)); // a miss: zero sub-rays of the pulse returned a hit
                            data.insert(std::pair<std::string, double>("is_miss", 1.0)); // gapfilled points are misses (transmitted beams)
                            if (gapfill_is_moving) {
                                data.insert(std::pair<std::string, double>("origin_x", fill_origin.x));
                                data.insert(std::pair<std::string, double>("origin_y", fill_origin.y));
                                data.insert(std::pair<std::string, double>("origin_z", fill_origin.z));
                            }
                            if (add_flags) {
                                // gapfillMisses_code = 3: upward edge points
                                data.insert(std::pair<std::string, double>("gapfillMisses_code", 3.0));
                            }

                            addHitPoint(scanID, xyz, spherical, data);
                            filled_positions.insert(grid_key); // Mark this position as filled
                        }
                    }

                    theta = theta - dtheta;
                    timestep = timestep - dt_avg;
                }
            }

            // downward edge points
            if (hit_table2D.at(j).back().at(2) < theta_range.y) {

                int sz = hit_table2D.at(j).size();
                // same concept as above for downward edge points
                float dtheta = dtheta_avg;
                float theta = hit_table2D.at(j).at(sz - 1).at(2) + dtheta;
                float phi = hit_table2D.at(j).at(sz - 1).at(3);
                float timestep = hit_table2D.at(j).at(sz - 1).at(1) + dt_avg;
                while (theta < theta_range.y) {

                    // Convert to grid indices using proper direction2rc method
                    helios::SphericalCoord dir_to_check(gap_distance, 0.5 * M_PI - theta, phi);
                    helios::int2 rc = scans.at(scanID).direction2rc(dir_to_check);

                    // Only add if this grid position is actually empty (avoid duplicates)
                    if (rc.x >= 0 && rc.x < (int) scans.at(scanID).Ntheta && rc.y >= 0 && rc.y < (int) scans.at(scanID).Nphi) {

                        // Check if this grid position has already been filled (avoid duplicates)
                        auto grid_key = std::make_pair(rc.x, rc.y);
                        if (filled_positions.find(grid_key) == filled_positions.end()) {

                            helios::SphericalCoord spherical(gap_distance, 0.5 * M_PI - theta, phi);
                            helios::vec3 fill_origin = originAtTime(timestep);
                            helios::vec3 xyz = fill_origin + helios::sphere2cart(spherical);
                            xyz_filled.push_back(xyz);

                            std::map<std::string, double> data;
                            data.insert(std::pair<std::string, double>("timestamp", timestep));
                            data.insert(std::pair<std::string, double>("target_index", min_tindex));
                            data.insert(std::pair<std::string, double>("nRaysHit", 0.0)); // a miss: zero sub-rays of the pulse returned a hit
                            data.insert(std::pair<std::string, double>("is_miss", 1.0)); // gapfilled points are misses (transmitted beams)
                            if (gapfill_is_moving) {
                                data.insert(std::pair<std::string, double>("origin_x", fill_origin.x));
                                data.insert(std::pair<std::string, double>("origin_y", fill_origin.y));
                                data.insert(std::pair<std::string, double>("origin_z", fill_origin.z));
                            }
                            if (add_flags) {
                                // gapfillMisses_code = 2: downward edge points
                                data.insert(std::pair<std::string, double>("gapfillMisses_code", 2.0));
                            }

                            addHitPoint(scanID, xyz, spherical, data);
                            filled_positions.insert(grid_key); // Mark this position as filled
                        }
                    }

                    theta = theta + dtheta;
                    timestep = timestep + dt_avg;
                }
            }
        }
    }

    uint npointsextrapolated = xyz_filled.size() - npointsfilled;

    if (printmessages) {
        std::cout << "filled " << xyz_filled.size() << " points (" << npointsfilled << " interior, " << npointsextrapolated << " edge)." << std::endl;
        std::cout << "  Processed " << hit_table2D.size() << " scan columns" << std::endl;
    }
    return xyz_filled;
}

void LiDARcloud::triangulateHitPoints(float Lmax, float max_aspect_ratio) {

    // Triangulation projects hits into the scan's (zenith, azimuth) grid space from a single origin. A moving-platform
    // scan has no fixed theta-phi grid (each pulse fires from a different pose), so the projection is meaningless and
    // would produce garbage triangles. Fail fast. For leaf-area inversion of a moving scan use the calculateLeafArea
    // overload that takes a supplied G(theta), which does not require triangulation.
    if (anyScanMoving()) {
        helios_runtime_error("ERROR (LiDARcloud::triangulateHitPoints): triangulation is not supported for moving-platform scans (see addScanMoving), which have no fixed theta-phi scan grid to triangulate. For leaf-area inversion of a "
                             "moving scan, call the calculateLeafArea overload that takes a G(theta) argument (it does not require triangulation).");
    }

    if (printmessages && getScanCount() == 0) {
        cout << "WARNING (triangulateHitPoints): No scans have been added to the point cloud.  Skipping triangulation..." << endl;
        return;
    } else if (printmessages && getHitCount() == 0) {
        cout << "WARNING (triangulateHitPoints): No hit points have been added to the point cloud.  Skipping triangulation..." << endl;
        return;
    }

    if (!hitgridcellcomputed) {
        calculateHitGridCell();
    }

    int Ntriangles = 0;

    // Reset triangulation diagnostics for this run (see getTriangulation* getters).
    triangulation_candidate_count = 0;
    triangulation_dropped_lmax = 0;
    triangulation_dropped_aspect = 0;
    triangulation_dropped_degenerate = 0;

    // For multi-return data, calculate adaptive separation ratio threshold
    bool use_adaptive_threshold = isMultiReturnData();
    float adaptive_sep_threshold = 0.0f;

    if (use_adaptive_threshold) {
        if (printmessages) {
            std::cout << "Multi-return data detected - calculating adaptive separation ratio threshold..." << std::endl;
        }

        // First pass: collect separation ratios from all potential triangles
        std::vector<float> all_separation_ratios;

        for (uint s = 0; s < getScanCount(); s++) {
            std::vector<int> Delaunay_inds_pass1;
            std::vector<Shx> pts_pass1, pts_copy_pass1;
            int count_pass1 = 0;

            for (int r = 0; r < getHitCount(); r++) {
                if (getHitScanID(r) == s && getHitGridCell(r) >= 0) {
                    // Auto-filter first returns for multi-return data
                    if (use_adaptive_threshold) {
                        if (doesHitDataExist(r, "target_index") && getHitData(r, "target_index") != 0.0) {
                            continue; // Skip non-first returns
                        }
                    }

                    helios::SphericalCoord direction = getHitRaydir(r);
                    helios::vec3 direction_cart = getHitXYZ(r) - getScanOrigin(s);
                    direction = cart2sphere(direction_cart);

                    Shx pt;
                    pt.id = count_pass1;
                    pt.r = direction.zenith;
                    pt.c = direction.azimuth;
                    pts_pass1.push_back(pt);
                    Delaunay_inds_pass1.push_back(r);
                    count_pass1++;
                }
            }

            if (pts_pass1.size() == 0)
                continue;

            // Handle coordinate wrapping
            float h[2] = {0, 0};
            for (int r = 0; r < pts_pass1.size(); r++) {
                if (pts_pass1.at(r).c < 0.5 * M_PI)
                    h[0] += 1.f;
                else if (pts_pass1.at(r).c > 1.5 * M_PI)
                    h[1] += 1.f;
            }
            h[0] /= float(pts_pass1.size());
            h[1] /= float(pts_pass1.size());
            if (h[0] + h[1] > 0.4) {
                for (int r = 0; r < pts_pass1.size(); r++) {
                    pts_pass1.at(r).c += M_PI;
                    if (pts_pass1.at(r).c > 2.f * M_PI)
                        pts_pass1.at(r).c -= 2.f * M_PI;
                }
            }

            std::vector<int> dupes_pass1;
            de_duplicate(pts_pass1, dupes_pass1);

            std::vector<Triad> triads_pass1;
            // CDT uses robust geometric predicates, so the s_hull-era
            // rotate-and-retry recovery is no longer needed; a failure here is
            // deterministic and the scan is skipped.
            int success = triangulate_CDT(pts_pass1, triads_pass1);

            if (success != 1)
                continue;

            // Collect separation ratios (pre-filter by edge length)
            for (int t = 0; t < triads_pass1.size(); t++) {
                int ID0 = Delaunay_inds_pass1.at(triads_pass1.at(t).a);
                int ID1 = Delaunay_inds_pass1.at(triads_pass1.at(t).b);
                int ID2 = Delaunay_inds_pass1.at(triads_pass1.at(t).c);

                helios::vec3 v0 = getHitXYZ(ID0);
                helios::vec3 v1 = getHitXYZ(ID1);
                helios::vec3 v2 = getHitXYZ(ID2);
                helios::SphericalCoord r0 = getHitRaydir(ID0);
                helios::SphericalCoord r1 = getHitRaydir(ID1);
                helios::SphericalCoord r2 = getHitRaydir(ID2);

                float L0 = (v0 - v1).magnitude();
                float L1 = (v0 - v2).magnitude();
                float L2 = (v1 - v2).magnitude();

                // Skip triangles that fail edge length filter
                if (L0 > Lmax || L1 > Lmax || L2 > Lmax) {
                    continue;
                }

                float ang01 = sqrt(pow(r0.zenith - r1.zenith, 2) + pow(r0.azimuth - r1.azimuth, 2));
                float ang02 = sqrt(pow(r0.zenith - r2.zenith, 2) + pow(r0.azimuth - r2.azimuth, 2));
                float ang12 = sqrt(pow(r1.zenith - r2.zenith, 2) + pow(r1.azimuth - r2.azimuth, 2));

                float ratio01 = L0 / (ang01 + 1e-6);
                float ratio02 = L1 / (ang02 + 1e-6);
                float ratio12 = L2 / (ang12 + 1e-6);
                float max_sep_ratio = max(max(ratio01, ratio02), ratio12);

                all_separation_ratios.push_back(max_sep_ratio);
            }
        }

        // Calculate 25th percentile and set adaptive threshold
        if (!all_separation_ratios.empty()) {
            std::sort(all_separation_ratios.begin(), all_separation_ratios.end());
            size_t idx_25 = all_separation_ratios.size() / 4;
            float percentile_25 = all_separation_ratios[idx_25];
            adaptive_sep_threshold = 8.5f * percentile_25;

            if (printmessages) {
                std::cout << "  25th percentile separation ratio: " << percentile_25 << std::endl;
                std::cout << "  Adaptive threshold: " << adaptive_sep_threshold << std::endl;
            }
        }
    }

    // Second pass: perform triangulation with adaptive filtering
    for (uint s = 0; s < getScanCount(); s++) {

        std::vector<int> Delaunay_inds;

        std::vector<Shx> pts, pts_copy;

        int count = 0;
        for (int r = 0; r < getHitCount(); r++) {

            if (getHitScanID(r) == s && getHitGridCell(r) >= 0) {
                // Auto-filter first returns for multi-return data
                if (use_adaptive_threshold) {
                    if (doesHitDataExist(r, "target_index") && getHitData(r, "target_index") != 0.0) {
                        continue; // Skip non-first returns
                    }
                }

                helios::SphericalCoord direction = getHitRaydir(r);

                helios::vec3 direction_cart = getHitXYZ(r) - getScanOrigin(s);
                direction = cart2sphere(direction_cart);

                Shx pt;
                pt.id = count;
                pt.r = direction.zenith;
                pt.c = direction.azimuth;

                pts.push_back(pt);

                Delaunay_inds.push_back(r);

                count++;
            }
        }

        if (pts.size() == 0) {
            if (printmessages) {
                std::cout << "Scan " << s << " contains no triangles. Skipping this scan..." << std::endl;
            }
            continue;
        }

        float h[2] = {0, 0};
        for (int r = 0; r < pts.size(); r++) {
            if (pts.at(r).c < 0.5 * M_PI) {
                h[0] += 1.f;
            } else if (pts.at(r).c > 1.5 * M_PI) {
                h[1] += 1.f;
            }
        }
        h[0] /= float(pts.size());
        h[1] /= float(pts.size());
        if (h[0] + h[1] > 0.4) {
            if (printmessages) {
                std::cout << "Shifting scan " << s << std::endl;
            }
            for (int r = 0; r < pts.size(); r++) {
                pts.at(r).c += M_PI;
                if (pts.at(r).c > 2.f * M_PI) {
                    pts.at(r).c -= 2.f * M_PI;
                }
            }
        }

        // Snap coordinates to fixed precision for cross-platform consistency.
        // Even with CDT's robust predicates (which make the triangulation
        // deterministic for identical input), the upstream cart2sphere
        // coordinates can differ at the ULP level across architectures
        // (ARM64 vs x86_64); snapping collapses those so de_duplicate and the
        // tessellation stay platform-independent.
        const float COORD_SNAP_PRECISION = 1e-6f;
        for (auto &pt: pts) {
            pt.r = std::round(pt.r / COORD_SNAP_PRECISION) * COORD_SNAP_PRECISION;
            pt.c = std::round(pt.c / COORD_SNAP_PRECISION) * COORD_SNAP_PRECISION;
        }

        std::vector<int> dupes;
        int nx = de_duplicate(pts, dupes);
        pts_copy = pts;

        std::vector<Triad> triads;

        if (printmessages) {
            std::cout << "starting triangulation for scan " << s << "..." << std::endl;
        }

        // CDT uses robust geometric predicates, so the s_hull-era
        // rotate-and-retry recovery is no longer needed; a failure here is
        // deterministic and the scan is skipped.
        int success = triangulate_CDT(pts, triads);

        if (success != 1) {
            if (printmessages) {
                std::cout << "FAILED: could not triangulate scan " << s << ". Skipping this scan." << std::endl;
            }
            continue;
        } else if (printmessages) {
            std::cout << "finished triangulation" << std::endl;
        }

        triangulation_candidate_count += triads.size();

        for (int t = 0; t < triads.size(); t++) {

            int ID0 = Delaunay_inds.at(triads.at(t).a);
            int ID1 = Delaunay_inds.at(triads.at(t).b);
            int ID2 = Delaunay_inds.at(triads.at(t).c);

            helios::vec3 vertex0 = getHitXYZ(ID0);
            helios::SphericalCoord raydir0 = getHitRaydir(ID0);

            helios::vec3 vertex1 = getHitXYZ(ID1);
            helios::SphericalCoord raydir1 = getHitRaydir(ID1);

            helios::vec3 vertex2 = getHitXYZ(ID2);
            helios::SphericalCoord raydir2 = getHitRaydir(ID2);

            helios::vec3 v;
            v = vertex0 - vertex1;
            float L0 = v.magnitude();
            v = vertex0 - vertex2;
            float L1 = v.magnitude();
            v = vertex1 - vertex2;
            float L2 = v.magnitude();

            float aspect_ratio = max(max(L0, L1), L2) / min(min(L0, L1), L2);

            // Apply filtering. Attribute each dropped triangle to ONE primary
            // reason in priority order (Lmax, then aspect/separation) so the
            // diagnostic counts reconcile: candidates == kept + dropped_lmax +
            // dropped_aspect + dropped_degenerate.
            bool dropped_lmax = (L0 > Lmax || L1 > Lmax || L2 > Lmax);
            bool dropped_aspect = false;

            if (use_adaptive_threshold) {
                // Multi-return: use BOTH separation ratio filter AND aspect ratio filter
                float ang01 = sqrt(pow(raydir0.zenith - raydir1.zenith, 2) + pow(raydir0.azimuth - raydir1.azimuth, 2));
                float ang02 = sqrt(pow(raydir0.zenith - raydir2.zenith, 2) + pow(raydir0.azimuth - raydir2.azimuth, 2));
                float ang12 = sqrt(pow(raydir1.zenith - raydir2.zenith, 2) + pow(raydir1.azimuth - raydir2.azimuth, 2));

                float ratio01 = L0 / (ang01 + 1e-6);
                float ratio02 = L1 / (ang02 + 1e-6);
                float ratio12 = L2 / (ang12 + 1e-6);
                float max_sep_ratio = max(max(ratio01, ratio02), ratio12);

                dropped_aspect = (max_sep_ratio > adaptive_sep_threshold) || (aspect_ratio > max_aspect_ratio);
            } else {
                // Single-return: use aspect ratio filter
                dropped_aspect = (aspect_ratio > max_aspect_ratio);
            }

            if (dropped_lmax) {
                triangulation_dropped_lmax++;
                continue;
            }
            if (dropped_aspect) {
                triangulation_dropped_aspect++;
                continue;
            }

            int gridcell = getHitGridCell(ID0);

            if (printmessages && gridcell == -2) {
                cout << "WARNING (triangulateHitPoints): You typically want to define the hit grid cell for all hit points before performing triangulation." << endl;
            }

            RGBcolor color = make_RGBcolor(0, 0, 0);
            color.r = (hits.at(ID0).color.r + hits.at(ID1).color.r + hits.at(ID2).color.r) / 3.f;
            color.g = (hits.at(ID0).color.g + hits.at(ID1).color.g + hits.at(ID2).color.g) / 3.f;
            color.b = (hits.at(ID0).color.b + hits.at(ID1).color.b + hits.at(ID2).color.b) / 3.f;

            Triangulation tri(s, vertex0, vertex1, vertex2, ID0, ID1, ID2, color, gridcell);

            if (tri.area != tri.area) {
                triangulation_dropped_degenerate++;
                continue;
            }

            triangles.push_back(tri);

            Ntriangles++;
        }
    }

    triangulationcomputed = true;

    if (printmessages) {
        cout << "\r                                           ";
        cout << "\rTriangulating...formed " << Ntriangles << " total triangles." << endl;
    }
}

void LiDARcloud::triangulateHitPoints(float Lmax, float max_aspect_ratio, const char *scalar_field, float threshold, const char *comparator) {

    // See the two-argument overload: triangulation requires a fixed theta-phi scan grid that moving-platform scans lack.
    if (anyScanMoving()) {
        helios_runtime_error("ERROR (LiDARcloud::triangulateHitPoints): triangulation is not supported for moving-platform scans (see addScanMoving), which have no fixed theta-phi scan grid to triangulate. For leaf-area inversion of a "
                             "moving scan, call the calculateLeafArea overload that takes a G(theta) argument (it does not require triangulation).");
    }

    if (printmessages && getScanCount() == 0) {
        cout << "WARNING (triangulateHitPoints): No scans have been added to the point cloud.  Skipping triangulation..." << endl;
        return;
    } else if (printmessages && getHitCount() == 0) {
        cout << "WARNING (triangulateHitPoints): No hit points have been added to the point cloud.  Skipping triangulation..." << endl;
        return;
    }

    if (!hitgridcellcomputed) {
        calculateHitGridCell();
    }

    int Ntriangles = 0;

    // Reset triangulation diagnostics for this run (see getTriangulation* getters).
    triangulation_candidate_count = 0;
    triangulation_dropped_lmax = 0;
    triangulation_dropped_aspect = 0;
    triangulation_dropped_degenerate = 0;

    // For multi-return data, calculate adaptive separation ratio threshold
    bool use_adaptive_threshold = isMultiReturnData();
    float adaptive_sep_threshold = 0.0f;

    if (use_adaptive_threshold) {
        if (printmessages) {
            std::cout << "Multi-return data detected - calculating adaptive separation ratio threshold..." << std::endl;
        }

        // First pass: collect separation ratios from all potential triangles
        std::vector<float> all_separation_ratios;

        for (uint s = 0; s < getScanCount(); s++) {
            std::vector<int> Delaunay_inds_pass1;
            std::vector<Shx> pts_pass1, pts_copy_pass1;
            int count_pass1 = 0;

            for (int r = 0; r < getHitCount(); r++) {
                if (getHitScanID(r) == s && getHitGridCell(r) >= 0) {
                    helios::SphericalCoord direction = getHitRaydir(r);
                    helios::vec3 direction_cart = getHitXYZ(r) - getScanOrigin(s);
                    direction = cart2sphere(direction_cart);

                    Shx pt;
                    pt.id = count_pass1;
                    pt.r = direction.zenith;
                    pt.c = direction.azimuth;
                    pts_pass1.push_back(pt);
                    Delaunay_inds_pass1.push_back(r);
                    count_pass1++;
                }
            }

            if (pts_pass1.size() == 0)
                continue;

            // Handle coordinate wrapping
            float h[2] = {0, 0};
            for (int r = 0; r < pts_pass1.size(); r++) {
                if (pts_pass1.at(r).c < 0.5 * M_PI)
                    h[0] += 1.f;
                else if (pts_pass1.at(r).c > 1.5 * M_PI)
                    h[1] += 1.f;
            }
            h[0] /= float(pts_pass1.size());
            h[1] /= float(pts_pass1.size());
            if (h[0] + h[1] > 0.4) {
                for (int r = 0; r < pts_pass1.size(); r++) {
                    pts_pass1.at(r).c += M_PI;
                    if (pts_pass1.at(r).c > 2.f * M_PI)
                        pts_pass1.at(r).c -= 2.f * M_PI;
                }
            }

            std::vector<int> dupes_pass1;
            de_duplicate(pts_pass1, dupes_pass1);

            std::vector<Triad> triads_pass1;
            // CDT uses robust geometric predicates, so the s_hull-era
            // rotate-and-retry recovery is no longer needed; a failure here is
            // deterministic and the scan is skipped.
            int success = triangulate_CDT(pts_pass1, triads_pass1);

            if (success != 1)
                continue;

            // Collect separation ratios (pre-filter by edge length)
            for (int t = 0; t < triads_pass1.size(); t++) {
                int ID0 = Delaunay_inds_pass1.at(triads_pass1.at(t).a);
                int ID1 = Delaunay_inds_pass1.at(triads_pass1.at(t).b);
                int ID2 = Delaunay_inds_pass1.at(triads_pass1.at(t).c);

                helios::vec3 v0 = getHitXYZ(ID0);
                helios::vec3 v1 = getHitXYZ(ID1);
                helios::vec3 v2 = getHitXYZ(ID2);
                helios::SphericalCoord r0 = getHitRaydir(ID0);
                helios::SphericalCoord r1 = getHitRaydir(ID1);
                helios::SphericalCoord r2 = getHitRaydir(ID2);

                float L0 = (v0 - v1).magnitude();
                float L1 = (v0 - v2).magnitude();
                float L2 = (v1 - v2).magnitude();

                // Skip triangles that fail edge length filter
                if (L0 > Lmax || L1 > Lmax || L2 > Lmax) {
                    continue;
                }

                float ang01 = sqrt(pow(r0.zenith - r1.zenith, 2) + pow(r0.azimuth - r1.azimuth, 2));
                float ang02 = sqrt(pow(r0.zenith - r2.zenith, 2) + pow(r0.azimuth - r2.azimuth, 2));
                float ang12 = sqrt(pow(r1.zenith - r2.zenith, 2) + pow(r1.azimuth - r2.azimuth, 2));

                float ratio01 = L0 / (ang01 + 1e-6);
                float ratio02 = L1 / (ang02 + 1e-6);
                float ratio12 = L2 / (ang12 + 1e-6);
                float max_sep_ratio = max(max(ratio01, ratio02), ratio12);

                all_separation_ratios.push_back(max_sep_ratio);
            }
        }

        // Calculate 25th percentile and set adaptive threshold
        if (!all_separation_ratios.empty()) {
            std::sort(all_separation_ratios.begin(), all_separation_ratios.end());
            size_t idx_25 = all_separation_ratios.size() / 4;
            float percentile_25 = all_separation_ratios[idx_25];
            adaptive_sep_threshold = 8.5f * percentile_25;

            if (printmessages) {
                std::cout << "  25th percentile separation ratio: " << percentile_25 << std::endl;
                std::cout << "  Adaptive threshold: " << adaptive_sep_threshold << std::endl;
            }
        }
    }

    // Second pass: perform triangulation with adaptive filtering
    for (uint s = 0; s < getScanCount(); s++) {

        std::vector<int> Delaunay_inds;

        std::vector<Shx> pts, pts_copy;

        std::size_t delete_count = 0;
        int count = 0;

        for (int r = 0; r < getHitCount(); r++) {


            if (getHitScanID(r) == s && getHitGridCell(r) >= 0) {

                if (hits.at(r).data.find(scalar_field) != hits.at(r).data.end()) {
                    double R = getHitData(r, scalar_field);
                    if (strcmp(comparator, "<") == 0) {
                        if (R < threshold) {
                            delete_count++;
                            continue;
                        }
                    } else if (strcmp(comparator, ">") == 0) {
                        if (R > threshold) {
                            delete_count++;
                            continue;
                        }
                    } else if (strcmp(comparator, "=") == 0) {
                        if (R == threshold) {

                            delete_count++;
                            continue;
                        }
                    }
                }

                helios::SphericalCoord direction = getHitRaydir(r);

                helios::vec3 direction_cart = getHitXYZ(r) - getScanOrigin(s);
                direction = cart2sphere(direction_cart);

                Shx pt;
                pt.id = count;
                pt.r = direction.zenith;
                pt.c = direction.azimuth;

                pts.push_back(pt);

                Delaunay_inds.push_back(r);

                count++;
            }
        }

        if (printmessages) {
            std::cout << "Scan " << s << " triangulation: " << count << " points used, " << delete_count << " points filtered out";
            if (strlen(scalar_field) > 0) {
                std::cout << " (filter: " << scalar_field << " " << comparator << " " << threshold << ")";
            }
            std::cout << std::endl;
        }

        if (pts.size() == 0) {
            if (printmessages) {
                std::cout << "Scan " << s << " contains no triangles. Skipping this scan..." << std::endl;
            }
            continue;
        }

        float h[2] = {0, 0};
        for (int r = 0; r < pts.size(); r++) {
            if (pts.at(r).c < 0.5 * M_PI) {
                h[0] += 1.f;
            } else if (pts.at(r).c > 1.5 * M_PI) {
                h[1] += 1.f;
            }
        }
        h[0] /= float(pts.size());
        h[1] /= float(pts.size());
        if (h[0] + h[1] > 0.4) {
            if (printmessages) {
                std::cout << "Shifting scan " << s << std::endl;
            }
            for (int r = 0; r < pts.size(); r++) {
                pts.at(r).c += M_PI;
                if (pts.at(r).c > 2.f * M_PI) {
                    pts.at(r).c -= 2.f * M_PI;
                }
            }
        }

        // Snap coordinates to fixed precision for cross-platform consistency.
        // Even with CDT's robust predicates (which make the triangulation
        // deterministic for identical input), the upstream cart2sphere
        // coordinates can differ at the ULP level across architectures
        // (ARM64 vs x86_64); snapping collapses those so de_duplicate and the
        // tessellation stay platform-independent.
        const float COORD_SNAP_PRECISION = 1e-6f;
        for (auto &pt: pts) {
            pt.r = std::round(pt.r / COORD_SNAP_PRECISION) * COORD_SNAP_PRECISION;
            pt.c = std::round(pt.c / COORD_SNAP_PRECISION) * COORD_SNAP_PRECISION;
        }

        std::vector<int> dupes;
        int nx = de_duplicate(pts, dupes);
        pts_copy = pts;

        std::vector<Triad> triads;

        if (printmessages) {
            std::cout << "starting triangulation for scan " << s << "..." << std::endl;
        }

        // CDT uses robust geometric predicates, so the s_hull-era
        // rotate-and-retry recovery is no longer needed; a failure here is
        // deterministic and the scan is skipped.
        int success = triangulate_CDT(pts, triads);

        if (success != 1) {
            if (printmessages) {
                std::cout << "FAILED: could not triangulate scan " << s << ". Skipping this scan." << std::endl;
            }
            continue;
        } else if (printmessages) {
            std::cout << "finished triangulation" << std::endl;
        }

        triangulation_candidate_count += triads.size();

        for (int t = 0; t < triads.size(); t++) {

            int ID0 = Delaunay_inds.at(triads.at(t).a);
            int ID1 = Delaunay_inds.at(triads.at(t).b);
            int ID2 = Delaunay_inds.at(triads.at(t).c);

            helios::vec3 vertex0 = getHitXYZ(ID0);
            helios::SphericalCoord raydir0 = getHitRaydir(ID0);

            helios::vec3 vertex1 = getHitXYZ(ID1);
            helios::SphericalCoord raydir1 = getHitRaydir(ID1);

            helios::vec3 vertex2 = getHitXYZ(ID2);
            helios::SphericalCoord raydir2 = getHitRaydir(ID2);

            helios::vec3 v;
            v = vertex0 - vertex1;
            float L0 = v.magnitude();
            v = vertex0 - vertex2;
            float L1 = v.magnitude();
            v = vertex1 - vertex2;
            float L2 = v.magnitude();

            float aspect_ratio = max(max(L0, L1), L2) / min(min(L0, L1), L2);

            // Apply filtering. Attribute each dropped triangle to ONE primary
            // reason in priority order (Lmax, then aspect/separation) so the
            // diagnostic counts reconcile: candidates == kept + dropped_lmax +
            // dropped_aspect + dropped_degenerate.
            bool dropped_lmax = (L0 > Lmax || L1 > Lmax || L2 > Lmax);
            bool dropped_aspect = false;

            if (use_adaptive_threshold) {
                // Multi-return: use BOTH separation ratio filter AND aspect ratio filter
                float ang01 = sqrt(pow(raydir0.zenith - raydir1.zenith, 2) + pow(raydir0.azimuth - raydir1.azimuth, 2));
                float ang02 = sqrt(pow(raydir0.zenith - raydir2.zenith, 2) + pow(raydir0.azimuth - raydir2.azimuth, 2));
                float ang12 = sqrt(pow(raydir1.zenith - raydir2.zenith, 2) + pow(raydir1.azimuth - raydir2.azimuth, 2));

                float ratio01 = L0 / (ang01 + 1e-6);
                float ratio02 = L1 / (ang02 + 1e-6);
                float ratio12 = L2 / (ang12 + 1e-6);
                float max_sep_ratio = max(max(ratio01, ratio02), ratio12);

                dropped_aspect = (max_sep_ratio > adaptive_sep_threshold) || (aspect_ratio > max_aspect_ratio);
            } else {
                // Single-return: use aspect ratio filter
                dropped_aspect = (aspect_ratio > max_aspect_ratio);
            }

            if (dropped_lmax) {
                triangulation_dropped_lmax++;
                continue;
            }
            if (dropped_aspect) {
                triangulation_dropped_aspect++;
                continue;
            }

            int gridcell = getHitGridCell(ID0);

            if (printmessages && gridcell == -2) {
                cout << "WARNING (triangulateHitPoints): You typically want to define the hit grid cell for all hit points before performing triangulation." << endl;
            }

            RGBcolor color = make_RGBcolor(0, 0, 0);
            color.r = (hits.at(ID0).color.r + hits.at(ID1).color.r + hits.at(ID2).color.r) / 3.f;
            color.g = (hits.at(ID0).color.g + hits.at(ID1).color.g + hits.at(ID2).color.g) / 3.f;
            color.b = (hits.at(ID0).color.b + hits.at(ID1).color.b + hits.at(ID2).color.b) / 3.f;

            Triangulation tri(s, vertex0, vertex1, vertex2, ID0, ID1, ID2, color, gridcell);

            if (tri.area != tri.area) {
                triangulation_dropped_degenerate++;
                continue;
            }

            triangles.push_back(tri);

            Ntriangles++;
        }
    }

    triangulationcomputed = true;

    if (printmessages) {
        cout << "\r                                           ";
        cout << "\rTriangulating...formed " << Ntriangles << " total triangles." << endl;
    }
}


void LiDARcloud::addTrianglesToContext(Context *context) const {

    if (scans.size() == 0) {
        if (printmessages) {
            std::cout << "WARNING (addTrianglesToContext): There are no scans in the point cloud, and thus there are no triangles to add...skipping." << std::endl;
        }
        return;
    }

    for (std::size_t i = 0; i < getTriangleCount(); i++) {

        Triangulation tri = getTriangle(i);

        context->addTriangle(tri.vertex0, tri.vertex1, tri.vertex2, tri.color);
    }
}

uint LiDARcloud::getGridCellCount(void) const {
    return grid_cells.size();
}

void LiDARcloud::addGridCell(const vec3 &center, const vec3 &size, float rotation) {
    addGridCell(center, center, size, size, rotation, make_int3(1, 1, 1), make_int3(1, 1, 1));
}

void LiDARcloud::addGridCell(const vec3 &center, const vec3 &global_anchor, const vec3 &size, const vec3 &global_size, float rotation, const int3 &global_ijk, const int3 &global_count) {

    GridCell newcell(center, global_anchor, size, global_size, rotation, global_ijk, global_count);

    grid_cells.push_back(newcell);
}

helios::vec3 LiDARcloud::getCellCenter(uint index) const {

    if (index >= getGridCellCount()) {
        helios_runtime_error("ERROR (LiDARcloud::getCellCenter): grid cell index out of range.  Requested center of cell #" + std::to_string(index) + " but there are only " + std::to_string(getGridCellCount()) + " cells in the grid.");
    }

    return grid_cells.at(index).center;
}

helios::vec3 LiDARcloud::getCellGlobalAnchor(uint index) const {

    if (index >= getGridCellCount()) {
        helios_runtime_error("ERROR (LiDARcloud::getCellGlobalAnchor): grid cell index out of range.  Requested anchor of cell #" + std::to_string(index) + " but there are only " + std::to_string(getGridCellCount()) + " cells in the grid.");
    }

    return grid_cells.at(index).global_anchor;
}

helios::vec3 LiDARcloud::getCellSize(uint index) const {

    if (index >= getGridCellCount()) {
        helios_runtime_error("ERROR (LiDARcloud::getCellCenter): grid cell index out of range.  Requested size of cell #" + std::to_string(index) + " but there are only " + std::to_string(getGridCellCount()) + " cells in the grid.");
    }

    return grid_cells.at(index).size;
}

float LiDARcloud::getCellRotation(uint index) const {

    if (index >= getGridCellCount()) {
        helios_runtime_error("ERROR (LiDARcloud::getCellRotation): grid cell index out of range.  Requested rotation of cell #" + std::to_string(index) + " but there are only " + std::to_string(getGridCellCount()) + " cells in the grid.");
    }

    return grid_cells.at(index).azimuthal_rotation;
}

std::vector<float> LiDARcloud::calculateSyntheticGtheta(helios::Context *context) {

    size_t Nprims = context->getPrimitiveCount();

    uint Nscans = getScanCount();

    uint Ncells = getGridCellCount();

    std::vector<float> Gtheta;
    Gtheta.resize(Ncells);

    std::vector<float> area_sum;
    area_sum.resize(Ncells, 0.f);
    std::vector<uint> cell_tri_count;
    cell_tri_count.resize(Ncells, 0);

    std::vector<uint> UUIDs = context->getAllUUIDs();
    for (int p = 0; p < UUIDs.size(); p++) {

        uint UUID = UUIDs.at(p);

        if (context->doesPrimitiveDataExist(UUID, "gridCell")) {

            uint gridCell;
            context->getPrimitiveData(UUID, "gridCell", gridCell);

            std::vector<vec3> vertices = context->getPrimitiveVertices(UUID);
            float area = context->getPrimitiveArea(UUID);
            vec3 normal = context->getPrimitiveNormal(UUID);

            for (int s = 0; s < Nscans; s++) {
                vec3 origin = getScanOrigin(s);
                vec3 raydir = vertices.front() - origin;
                raydir.normalize();

                if (area == area) { // in rare cases you can get area=NaN

                    Gtheta.at(gridCell) += fabs(normal * raydir) * area;

                    area_sum.at(gridCell) += area;
                    cell_tri_count.at(gridCell) += 1;
                }
            }
        }
    }

    for (uint v = 0; v < Ncells; v++) {
        if (cell_tri_count[v] > 0) {
            Gtheta[v] *= float(cell_tri_count[v]) / (area_sum[v]);
        }
    }


    std::vector<float> output_Gtheta;
    output_Gtheta.resize(Ncells, 0.f);

    for (int v = 0; v < Ncells; v++) {
        output_Gtheta.at(v) = Gtheta.at(v);
        if (context->doesPrimitiveDataExist(UUIDs.at(v), "gridCell")) {
            context->setPrimitiveData(UUIDs.at(v), "synthetic_Gtheta", Gtheta.at(v));
        }
    }

    return output_Gtheta;
}

void LiDARcloud::setCellLeafArea(float area, uint index) {

    if (index > getGridCellCount()) {
        helios_runtime_error("ERROR (LiDARcloud::setCellLeafArea): grid cell index out of range.");
    }

    grid_cells.at(index).leaf_area = area;
}

float LiDARcloud::getCellLeafArea(uint index) const {

    if (index >= getGridCellCount()) {
        helios_runtime_error("ERROR (LiDARcloud::getCellLeafArea): grid cell index out of range. Requested leaf area of cell #" + std::to_string(index) + " but there are only " + std::to_string(getGridCellCount()) + " cells in the grid.");
    }

    return grid_cells.at(index).leaf_area;
}

float LiDARcloud::getCellLeafAreaDensity(uint index) const {

    if (index >= getGridCellCount()) {
        helios_runtime_error("ERROR (LiDARcloud::getCellLeafAreaDensity): grid cell index out of range. Requested leaf area density of cell #" + std::to_string(index) + " but there are only " + std::to_string(getGridCellCount()) +
                             " cells in the grid.");
    }

    helios::vec3 gridsize = grid_cells.at(index).size;
    return grid_cells.at(index).leaf_area / (gridsize.x * gridsize.y * gridsize.z);
}

void LiDARcloud::setCellGtheta(float Gtheta, uint index) {

    if (index > getGridCellCount()) {
        helios_runtime_error("ERROR (LiDARcloud::setCellGtheta): grid cell index out of range.");
    }

    grid_cells.at(index).Gtheta = Gtheta;
}

float LiDARcloud::getCellGtheta(uint index) const {

    if (index >= getGridCellCount()) {
        helios_runtime_error("ERROR (LiDARcloud::getCellGtheta): grid cell index out of range. Requested leaf area of cell #" + std::to_string(index) + " but there are only " + std::to_string(getGridCellCount()) + " cells in the grid.");
    }

    return grid_cells.at(index).Gtheta;
}

int LiDARcloud::getCellBeamCount(uint index) const {

    if (index >= getGridCellCount()) {
        helios_runtime_error("ERROR (LiDARcloud::getCellBeamCount): grid cell index out of range. Requested beam count of cell #" + std::to_string(index) + " but there are only " + std::to_string(getGridCellCount()) + " cells in the grid.");
    }

    return grid_cells.at(index).beam_count;
}

float LiDARcloud::getCellRelativeDensityIndex(uint index) const {

    if (index >= getGridCellCount()) {
        helios_runtime_error("ERROR (LiDARcloud::getCellRelativeDensityIndex): grid cell index out of range. Requested RDI of cell #" + std::to_string(index) + " but there are only " + std::to_string(getGridCellCount()) + " cells in the grid.");
    }

    return grid_cells.at(index).I_rdi;
}

float LiDARcloud::getCellMeanPathLength(uint index) const {

    if (index >= getGridCellCount()) {
        helios_runtime_error("ERROR (LiDARcloud::getCellMeanPathLength): grid cell index out of range. Requested mean path length of cell #" + std::to_string(index) + " but there are only " + std::to_string(getGridCellCount()) +
                             " cells in the grid.");
    }

    return grid_cells.at(index).zbar_e;
}

float LiDARcloud::getCellLADVariance(uint index) const {

    if (index >= getGridCellCount()) {
        helios_runtime_error("ERROR (LiDARcloud::getCellLADVariance): grid cell index out of range. Requested LAD variance of cell #" + std::to_string(index) + " but there are only " + std::to_string(getGridCellCount()) + " cells in the grid.");
    }

    return grid_cells.at(index).LAD_variance;
}

bool LiDARcloud::getCellLeafAreaConfidenceInterval(uint index, float confidence_level, float &lower, float &upper) const {

    if (index >= getGridCellCount()) {
        helios_runtime_error("ERROR (LiDARcloud::getCellLeafAreaConfidenceInterval): grid cell index out of range. Requested cell #" + std::to_string(index) + " but there are only " + std::to_string(getGridCellCount()) + " cells in the grid.");
    }
    if (confidence_level <= 0.f || confidence_level >= 1.f) {
        helios_runtime_error("ERROR (LiDARcloud::getCellLeafAreaConfidenceInterval): confidence_level must be strictly between 0 and 1.");
    }

    const GridCell &cell = grid_cells.at(index);
    if (cell.LAD_variance < 0.f || cell.beam_count <= 0) {
        return false; // variance undefined for this voxel
    }

    const float volume = cell.size.x * cell.size.y * cell.size.z;
    const float a = (volume > 0.f) ? cell.leaf_area / volume : 0.f; // LAD point estimate
    const float L = a * cell.Gtheta * cell.zbar_e; // voxel optical depth
    if (!ciValidPimont(L, cell.L1_element, cell.beam_count, confidence_level)) {
        return false; // outside the trustworthy regime -> refuse to emit an interval
    }

    // Two-sided z-multiplier for the requested confidence level.
    const double z = normalQuantile(1.0 - (1.0 - (double) confidence_level) / 2.0);
    const float lad_se = std::sqrt(cell.LAD_variance); // standard error of LAD [1/m]
    const float half_width = (float) z * volume * lad_se; // converted to leaf-area scale [m^2]
    lower = cell.leaf_area - half_width;
    if (lower < 0.f) {
        lower = 0.f; // leaf area is non-negative
    }
    upper = cell.leaf_area + half_width;
    return true;
}

bool LiDARcloud::getGroupLADConfidenceInterval(const std::vector<uint> &indices, float confidence_level, float &mean_lad, float &lower, float &upper) const {

    if (confidence_level <= 0.f || confidence_level >= 1.f) {
        helios_runtime_error("ERROR (LiDARcloud::getGroupLADConfidenceInterval): confidence_level must be strictly between 0 and 1.");
    }

    // Aggregate over the valid voxels in the group (Pimont et al. 2018, Eq. 39): the CI on the mean
    // LAD assumes voxel independence and uses the sum of the per-voxel LAD variances. Voxels outside
    // the Table-3 validity envelope (or with undefined variance) are skipped.
    double sum_lad = 0.0;
    double sum_variance = 0.0;
    uint n_valid = 0;
    for (uint index: indices) {
        if (index >= getGridCellCount()) {
            helios_runtime_error("ERROR (LiDARcloud::getGroupLADConfidenceInterval): grid cell index out of range. Requested cell #" + std::to_string(index) + " but there are only " + std::to_string(getGridCellCount()) + " cells in the grid.");
        }
        const GridCell &cell = grid_cells.at(index);
        if (cell.LAD_variance < 0.f || cell.beam_count <= 0) {
            continue;
        }
        const float volume = cell.size.x * cell.size.y * cell.size.z;
        const float a = (volume > 0.f) ? cell.leaf_area / volume : 0.f;
        const float L = a * cell.Gtheta * cell.zbar_e;
        if (!ciValidPimont(L, cell.L1_element, cell.beam_count, confidence_level)) {
            continue;
        }
        sum_lad += a;
        sum_variance += cell.LAD_variance;
        n_valid++;
    }

    if (n_valid == 0) {
        return false;
    }

    mean_lad = (float) (sum_lad / (double) n_valid);
    const double z = normalQuantile(1.0 - (1.0 - (double) confidence_level) / 2.0);
    const float half_width = (float) (z * std::sqrt(sum_variance) / (double) n_valid);
    lower = mean_lad - half_width;
    if (lower < 0.f) {
        lower = 0.f;
    }
    upper = mean_lad + half_width;
    return true;
}

void LiDARcloud::leafReconstructionFloodfill() {

    size_t group_count = 0;
    int current_group = 0;

    vector<vector<int>> nodes;
    nodes.resize(getHitCount());

    size_t Ntri = 0;
    for (size_t t = 0; t < getTriangleCount(); t++) {

        Triangulation tri = getTriangle(t);

        if (tri.gridcell >= 0) {

            nodes.at(tri.ID0).push_back(t);
            nodes.at(tri.ID1).push_back(t);
            nodes.at(tri.ID2).push_back(t);

            Ntri++;
        }
    }

    std::vector<int> fill_flag;
    fill_flag.resize(Ntri);
    for (size_t t = 0; t < Ntri; t++) {
        fill_flag.at(t) = -1;
    }

    for (size_t t = 0; t < Ntri; t++) { // looping through all triangles

        if (fill_flag.at(t) < 0) {

            floodfill(t, triangles, fill_flag, nodes, current_group, 0, 1e3);

            current_group++;
        }
    }

    for (size_t t = 0; t < Ntri; t++) { // looping through all triangles

        if (fill_flag.at(t) >= 0) {
            int fill_group = fill_flag.at(t);

            if (fill_group >= reconstructed_triangles.size()) {
                reconstructed_triangles.resize(fill_group + 1);
            }

            reconstructed_triangles.at(fill_group).push_back(triangles.at(t));
        }
    }
}

void LiDARcloud::floodfill(size_t t, std::vector<Triangulation> &cloud_triangles, std::vector<int> &fill_flag, std::vector<std::vector<int>> &nodes, int tag, int depth, int maxdepth) {

    Triangulation tri = cloud_triangles.at(t);

    int verts[3] = {tri.ID0, tri.ID1, tri.ID2};

    std::vector<int> connection_list;

    for (int i = 0; i < 3; i++) {
        std::vector<int> connected_tris = nodes.at(verts[i]);
        connection_list.insert(connection_list.begin(), connected_tris.begin(), connected_tris.end());
    }

    std::sort(connection_list.begin(), connection_list.end());

    int count = 0;
    for (int tt = 1; tt < connection_list.size(); tt++) {
        if (connection_list.at(tt - 1) != connection_list.at(tt)) {

            if (count >= 2) {

                int index = connection_list.at(tt - 1);

                if (fill_flag.at(index) == -1 && index != t) {

                    fill_flag.at(index) = tag;

                    if (depth < maxdepth) {
                        floodfill(index, cloud_triangles, fill_flag, nodes, tag, depth + 1, maxdepth);
                    }
                }
            }

            count = 1;
        } else {
            count++;
        }
    }
}

void LiDARcloud::leafReconstructionAlphaMask(float minimum_leaf_group_area, float maximum_leaf_group_area, float leaf_aspect_ratio, const char *mask_file) {
    leafReconstructionAlphaMask(minimum_leaf_group_area, maximum_leaf_group_area, leaf_aspect_ratio, -1.f, mask_file);
}

void LiDARcloud::leafReconstructionAlphaMask(float minimum_leaf_group_area, float maximum_leaf_group_area, float leaf_aspect_ratio, float leaf_length_constant, const char *mask_file) {

    if (printmessages) {
        cout << "Performing alphamask leaf reconstruction..." << flush;
    }

    if (triangles.size() == 0) {
        std::cout << "failed." << std::endl;
        helios_runtime_error("ERROR (LiDARcloud::leafReconstructionAlphamask): There are no triangulated points.  Either the triangulation failed or 'triangulateHitPoints()' was not called.");
    }

    std::string file = mask_file;
    if (file.substr(file.find_last_of(".") + 1) != "png") {
        std::cout << "failed." << std::endl;
        helios_runtime_error("ERROR (LiDARcloud::leafReconstructionAlphaMask): Mask data file " + std::string(mask_file) + " must be PNG image format.");
    }
    std::vector<std::vector<bool>> maskdata = readPNGAlpha(mask_file);
    if (maskdata.size() == 0) {
        std::cout << "failed." << std::endl;
        helios_runtime_error("ERROR (LiDARcloud::leafReconstructionAlphaMask): Could not load mask file " + std::string(mask_file) + ". It contains no data.");
    }
    int ix = maskdata.front().size();
    int jy = maskdata.size();
    int2 masksize = make_int2(ix, jy);
    uint Atotal = 0;
    uint Asolid = 0;
    for (uint j = 0; j < masksize.y; j++) {
        for (uint i = 0; i < masksize.x; i++) {
            Atotal++;
            if (maskdata.at(j).at(i)) {
                Asolid++;
            }
        }
    }

    float solidfraction = float(Asolid) / float(Atotal);

    float total_area = 0.f;

    std::vector<std::vector<float>> group_areas;
    group_areas.resize(getGridCellCount());

    reconstructed_alphamasks_maskfile = mask_file;

    leafReconstructionFloodfill();

    // Filter out small groups by an area threshold

    uint group_count = reconstructed_triangles.size();

    float group_area_max = 0;

    std::vector<bool> group_filter_flag;
    group_filter_flag.resize(reconstructed_triangles.size());

    for (int group = group_count - 1; group >= 0; group--) {

        float garea = 0.f;

        for (size_t t = 0; t < reconstructed_triangles.at(group).size(); t++) {

            float triangle_area = reconstructed_triangles.at(group).at(t).area;

            garea += triangle_area;
        }

        if (garea < minimum_leaf_group_area || garea > maximum_leaf_group_area) {
            group_filter_flag.at(group) = false;
            // reconstructed_triangles.erase( reconstructed_triangles.begin()+group );
        } else {
            group_filter_flag.at(group) = true;
            int cell = reconstructed_triangles.at(group).front().gridcell;
            group_areas.at(cell).push_back(garea);
        }
    }

    vector<float> Lavg;
    Lavg.resize(getGridCellCount(), 0.f);

    int Navg = 20;

    for (int v = 0; v < getGridCellCount(); v++) {

        std::sort(group_areas.at(v).begin(), group_areas.at(v).end());
        // std::partial_sort( group_areas.at(v).begin(), group_areas.at(v).begin()+Navg,group_areas.at(v).end(), std::greater<float>() );

        if (group_areas.at(v).size() > Navg) {
            for (int i = group_areas.at(v).size() - 1; i >= group_areas.at(v).size() - Navg; i--) {
                Lavg.at(v) += sqrtf(group_areas.at(v).at(i)) / float(Navg);
            }
        } else if (group_areas.at(v).size() == 0) {
            Lavg.at(v) = 0.05; // NOTE: hard-coded
        } else {
            for (int i = 0; i < group_areas.at(v).size(); i++) {
                Lavg.at(v) += sqrtf(group_areas.at(v).at(i)) / float(group_areas.at(v).size());
            }
        }

        if (printmessages) {
            std::cout << "Average leaf length for volume #" << v << " : " << Lavg.at(v) << endl;
        }
    }

    // Form alphamasks

    for (int group = 0; group < reconstructed_triangles.size(); group++) {

        if (!group_filter_flag.at(group)) {
            continue;
        }

        int cell = reconstructed_triangles.at(group).front().gridcell;

        helios::vec3 position = make_vec3(0, 0, 0);
        for (int t = 0; t < reconstructed_triangles.at(group).size(); t++) {
            position = position + reconstructed_triangles.at(group).at(t).vertex0 / float(reconstructed_triangles.at(group).size());
        }

        int gind = round(randu() * (reconstructed_triangles.at(group).size() - 1));

        reconstructed_alphamasks_center.push_back(position);
        float l = Lavg.at(reconstructed_triangles.at(group).front().gridcell) * sqrt(leaf_aspect_ratio / solidfraction);
        float w = l / leaf_aspect_ratio;
        reconstructed_alphamasks_size.push_back(helios::make_vec2(w, l));
        helios::vec3 normal = cross(reconstructed_triangles.at(group).at(gind).vertex1 - reconstructed_triangles.at(group).at(gind).vertex0, reconstructed_triangles.at(group).at(gind).vertex2 - reconstructed_triangles.at(group).at(gind).vertex0);
        reconstructed_alphamasks_rotation.push_back(make_SphericalCoord(cart2sphere(normal).zenith, cart2sphere(normal).azimuth));
        reconstructed_alphamasks_gridcell.push_back(reconstructed_triangles.at(group).front().gridcell);
        reconstructed_alphamasks_direct_flag.push_back(1);
    }

    if (printmessages) {
        cout << "done." << endl;
        cout << "Directly reconstructed " << reconstructed_alphamasks_center.size() << " leaf groups." << endl;
    }

    backfillLeavesAlphaMask(Lavg, leaf_aspect_ratio, solidfraction, group_filter_flag);

    for (int group = 0; group < reconstructed_triangles.size(); group++) {

        if (!group_filter_flag.at(group)) {
            std::swap(reconstructed_triangles.at(group), reconstructed_triangles.back());
            reconstructed_triangles.pop_back();
        }
    }

    // reconstructed_triangles.resize(0);
}


void LiDARcloud::backfillLeavesAlphaMask(const vector<float> &leaf_size, float leaf_aspect_ratio, float solidfraction, const vector<bool> &group_filter_flag) {

    if (printmessages) {
        cout << "Backfilling leaves..." << endl;
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::minstd_rand0 generator;
    generator.seed(seed);
    std::normal_distribution<float> randn;

    uint Ngroups = reconstructed_triangles.size();

    uint Ncells = getGridCellCount();

    std::vector<std::vector<uint>> group_gridcell;
    group_gridcell.resize(Ncells);

    // Calculate the current alphamask leaf area for each grid cell
    std::vector<float> leaf_area_current;
    leaf_area_current.resize(Ncells);

    int cell;
    int count = 0;
    for (uint g = 0; g < Ngroups; g++) {
        if (group_filter_flag.at(g)) {
            if (reconstructed_triangles.at(g).size() > 0) {
                cell = reconstructed_triangles.at(g).front().gridcell;
                leaf_area_current.at(cell) += leaf_size.at(cell) * leaf_size.at(cell) * solidfraction;
                group_gridcell.at(cell).push_back(count);
            }
            count++;
        }
    }

    std::vector<int> deleted_groups;
    int backfill_count = 0;

    helios::WarningAggregator backfill_warnings;
    backfill_warnings.setEnabled(printmessages);

    // Get the total theoretical leaf area for each grid cell based on LiDAR scan
    for (uint v = 0; v < Ncells; v++) {

        float leaf_area_total = getCellLeafArea(v);

        float reconstruct_frac = (leaf_area_total - leaf_area_current.at(v)) / leaf_area_total;

        if (leaf_area_total == 0 || reconstructed_alphamasks_size.size() == 0) { // no leaves in gridcell
            backfill_warnings.addWarning("volume_no_measured_leaf_area", "skipping volume #" + std::to_string(v) + " because it has no measured leaf area.");
            continue;
        } else if (getTriangleCount() == 0) {
            backfill_warnings.addWarning("volume_no_triangles", "skipping volume #" + std::to_string(v) + " because it has no triangles.");
            continue;
        } else if (leaf_area_current.at(v) == 0) { // no directly reconstructed leaves in gridcell

            std::vector<SphericalCoord> tri_rots;

            size_t Ntri = 0;
            for (size_t t = 0; t < getTriangleCount(); t++) {
                Triangulation tri = getTriangle(t);
                if (tri.gridcell == v) {
                    helios::vec3 normal = cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0);
                    tri_rots.push_back(make_SphericalCoord(cart2sphere(normal).zenith, cart2sphere(normal).azimuth));
                }
            }

            while (leaf_area_current.at(v) < leaf_area_total) {

                int randi = round(randu() * (tri_rots.size() - 1));

                helios::vec3 cellsize = getCellSize(v);
                helios::vec3 cellcenter = getCellCenter(v);
                float rotation = getCellRotation(v);

                helios::vec3 shift = cellcenter + rotatePoint(helios::make_vec3((randu() - 0.5) * cellsize.x, (randu() - 0.5) * cellsize.y, (randu() - 0.5) * cellsize.z), 0, rotation);

                reconstructed_alphamasks_center.push_back(shift);
                reconstructed_alphamasks_size.push_back(reconstructed_alphamasks_size.front());
                reconstructed_alphamasks_rotation.push_back(tri_rots.at(randi));
                reconstructed_alphamasks_gridcell.push_back(v);
                reconstructed_alphamasks_direct_flag.push_back(0);

                leaf_area_current.at(v) += reconstructed_alphamasks_size.back().x * reconstructed_alphamasks_size.back().y * solidfraction;
            }


        } else if (leaf_area_current.at(v) > leaf_area_total) { // too much leaf area in gridcell

            while (leaf_area_current.at(v) > leaf_area_total) {

                int randi = round(randu() * (group_gridcell.at(v).size() - 1));

                int group_index = group_gridcell.at(v).at(randi);

                deleted_groups.push_back(group_index);

                leaf_area_current.at(v) -= reconstructed_alphamasks_size.at(group_index).x * reconstructed_alphamasks_size.at(group_index).y * solidfraction;
            }

        } else { // not enough leaf area in gridcell

            while (leaf_area_current.at(v) < leaf_area_total) {

                int randi = round(randu() * (group_gridcell.at(v).size() - 1));

                int group_index = group_gridcell.at(v).at(randi);

                helios::vec3 cellsize = getCellSize(v);
                helios::vec3 cellcenter = getCellCenter(v);
                float rotation = getCellRotation(v);
                helios::vec3 cellanchor = getCellGlobalAnchor(v);

                // helios::vec3 shift = reconstructed_alphamasks_center.at(group_index) + helios::make_vec3( 0.45*(randu()-0.5)*cellsize.x, 0.45*(randu()-0.5)*cellsize.y, 0.45*(randu()-0.5)*cellsize.z ); //uniform shift about group
                helios::vec3 shift = reconstructed_alphamasks_center.at(group_index) + helios::make_vec3(0.25 * randn(generator) * cellsize.x, 0.25 * randn(generator) * cellsize.y, 0.25 * randn(generator) * cellsize.z); // Gaussian shift about group
                // helios::vec3 shift = cellcenter + helios::make_vec3( (randu()-0.5)*cellsize.x, (randu()-0.5)*cellsize.y, (randu()-0.5)*cellsize.z ); //uniform shift within voxel
                shift = rotatePointAboutLine(shift, cellanchor, make_vec3(0, 0, 1), rotation);

                if (group_index >= reconstructed_alphamasks_center.size()) {
                    helios_runtime_error("FAILED: " + std::to_string(group_index) + " " + std::to_string(reconstructed_alphamasks_center.size()) + " " + std::to_string(randi));
                } else if (reconstructed_alphamasks_gridcell.at(group_index) != v) {
                    helios_runtime_error("FAILED: selected leaf group is not from this grid cell");
                }

                reconstructed_alphamasks_center.push_back(shift);
                reconstructed_alphamasks_size.push_back(reconstructed_alphamasks_size.at(group_index));
                reconstructed_alphamasks_rotation.push_back(reconstructed_alphamasks_rotation.at(group_index));
                reconstructed_alphamasks_gridcell.push_back(v);
                reconstructed_alphamasks_direct_flag.push_back(0);

                leaf_area_current.at(v) += reconstructed_alphamasks_size.at(group_index).x * reconstructed_alphamasks_size.at(group_index).y * solidfraction;

                backfill_count++;
            }
        }
    }

    for (uint v = 0; v < Ncells; v++) {

        float leaf_area_total = getCellLeafArea(v);

        float current_area = 0;
        for (uint i = 0; i < reconstructed_alphamasks_size.size(); i++) {
            if (reconstructed_alphamasks_gridcell.at(i) == v) {
                current_area += reconstructed_alphamasks_size.at(i).x * reconstructed_alphamasks_size.at(i).y * solidfraction;
            }
        }
    }

    if (printmessages) {
        cout << "Backfilled " << backfill_count << " total leaf groups." << endl;
        cout << "Deleted " << deleted_groups.size() << " total leaf groups." << endl;
    }

    for (int i = deleted_groups.size() - 1; i >= 0; i--) {
        int group_index = deleted_groups.at(i);
        if (group_index >= 0 && group_index < reconstructed_alphamasks_center.size()) {
            // use swap-and-pop method
            std::swap(reconstructed_alphamasks_center.at(group_index), reconstructed_alphamasks_center.back());
            reconstructed_alphamasks_center.pop_back();
            std::swap(reconstructed_alphamasks_size.at(group_index), reconstructed_alphamasks_size.back());
            reconstructed_alphamasks_size.pop_back();
            std::swap(reconstructed_alphamasks_rotation.at(group_index), reconstructed_alphamasks_rotation.back());
            reconstructed_alphamasks_rotation.pop_back();
            std::swap(reconstructed_alphamasks_gridcell.at(group_index), reconstructed_alphamasks_gridcell.back());
            reconstructed_alphamasks_gridcell.pop_back();
            std::swap(reconstructed_alphamasks_direct_flag.at(group_index), reconstructed_alphamasks_direct_flag.back());
            reconstructed_alphamasks_direct_flag.pop_back();
        }
    }

    backfill_warnings.report(std::cerr);

    if (printmessages) {
        cout << "done." << endl;
    }
}

void LiDARcloud::calculateLeafAngleCDF(uint Nbins, std::vector<std::vector<float>> &CDF_theta, std::vector<std::vector<float>> &CDF_phi) {

    uint Ncells = getGridCellCount();

    std::vector<std::vector<float>> PDF_theta, PDF_phi;
    CDF_theta.resize(Ncells);
    PDF_theta.resize(Ncells);
    CDF_phi.resize(Ncells);
    PDF_phi.resize(Ncells);
    for (uint v = 0; v < Ncells; v++) {
        CDF_theta.at(v).resize(Nbins, 0.f);
        PDF_theta.at(v).resize(Nbins, 0.f);
        CDF_phi.at(v).resize(Nbins, 0.f);
        PDF_phi.at(v).resize(Nbins, 0.f);
    }
    float db_theta = 0.5 * M_PI / Nbins;
    float db_phi = 2.f * M_PI / Nbins;

    // calculate PDF from triangulated hit points (not reconstructed triangles)
    for (size_t t = 0; t < triangles.size(); t++) {
        float triangle_area = triangles.at(t).area;
        int gridcell = triangles.at(t).gridcell;

        if (gridcell >= 0 && gridcell < (int) Ncells) { // Valid grid cell
            helios::vec3 normal = cross(triangles.at(t).vertex1 - triangles.at(t).vertex0, triangles.at(t).vertex2 - triangles.at(t).vertex0);
            normal.z = fabs(normal.z); // keep in upper hemisphere

            helios::SphericalCoord normal_dir = cart2sphere(normal);

            int bin_theta = floor(normal_dir.zenith / db_theta);
            if (bin_theta >= Nbins) {
                bin_theta = Nbins - 1;
            }

            int bin_phi = floor(normal_dir.azimuth / db_phi);
            if (bin_phi >= Nbins) {
                bin_phi = Nbins - 1;
            }

            PDF_theta.at(gridcell).at(bin_theta) += triangle_area;
            PDF_phi.at(gridcell).at(bin_phi) += triangle_area;
        }
    }

    // calculate PDF from CDF
    for (uint v = 0; v < Ncells; v++) {
        for (uint i = 0; i < Nbins; i++) {
            for (uint j = 0; j <= i; j++) {
                CDF_theta.at(v).at(i) += PDF_theta.at(v).at(j);
                CDF_phi.at(v).at(i) += PDF_phi.at(v).at(j);
            }
        }
    }

    // char filename[50];
    // std::ofstream file_theta, file_phi;
    // for (uint v = 0; v < Ncells; v++) {
    //     sprintf(filename, "../output/PDF_theta%d.txt", v);
    //     file_theta.open(filename);
    //     sprintf(filename, "../output/PDF_phi%d.txt", v);
    //     file_phi.open(filename);
    //     for (uint i = 0; i < Nbins; i++) {
    //         file_theta << PDF_theta.at(v).at(i) << std::endl;
    //         file_phi << PDF_phi.at(v).at(i) << std::endl;
    //     }
    //     file_theta.close();
    //     file_phi.close();
    // }
}

void LiDARcloud::cropBeamsToGridAngleRange(uint source) {

    // loop through the vertices of the voxel grid.
    std::vector<helios::vec3> grid_vertices;
    helios::vec3 boxmin, boxmax;
    getGridBoundingBox(boxmin, boxmax); // axis aligned bounding box of all grid cells
    grid_vertices.push_back(boxmin);
    grid_vertices.push_back(boxmax);
    grid_vertices.push_back(helios::make_vec3(boxmin.x, boxmin.y, boxmax.z));
    grid_vertices.push_back(helios::make_vec3(boxmax.x, boxmax.y, boxmin.z));
    grid_vertices.push_back(helios::make_vec3(boxmin.x, boxmax.y, boxmin.z));
    grid_vertices.push_back(helios::make_vec3(boxmin.x, boxmax.y, boxmax.z));
    grid_vertices.push_back(helios::make_vec3(boxmax.x, boxmin.y, boxmin.z));
    grid_vertices.push_back(helios::make_vec3(boxmax.x, boxmin.y, boxmax.z));

    float max_theta = 0;
    float min_theta = M_PI;
    float max_phi = 0;
    float min_phi = 2 * M_PI;
    for (uint gg = 0; gg < grid_vertices.size(); gg++) {
        helios::vec3 direction_cart = grid_vertices.at(gg) - getScanOrigin(source);
        helios::SphericalCoord sc = cart2sphere(direction_cart);

        if (sc.azimuth < min_phi) {
            min_phi = sc.azimuth;
        }

        if (sc.azimuth > max_phi) {
            max_phi = sc.azimuth;
        }

        if (sc.zenith < min_theta) {
            min_theta = sc.zenith;
        }

        if (sc.zenith > max_theta) {
            max_theta = sc.zenith;
        }
    }

    vec2 theta_range = helios::make_vec2(min_theta, max_theta);
    vec2 phi_range = helios::make_vec2(min_phi, max_phi);

    for (int r = (getHitCount() - 1); r >= 0; r--) {
        if (getHitScanID(r) == source) {
            helios::SphericalCoord raydir = getHitRaydir(r);
            float this_theta = raydir.zenith;
            float this_phi = raydir.azimuth;
            double this_phi_d = double(this_phi);
            setHitData(r, "beam_azimuth", this_phi_d);
            if (this_phi < phi_range.x || this_phi > phi_range.y || this_theta < theta_range.x || this_theta > theta_range.y) {
                deleteHitPoint(r);
            }
        }
    }
}

// ========== SHARED METHODS FOR GPU AND CD IMPLEMENTATIONS ==========

void LiDARcloud::computeGtheta(uint Ncells, uint Nscans, std::vector<float> &Gtheta, std::vector<float> &Gtheta_bar) {

    // Initialize output vectors
    Gtheta.resize(Ncells, 0.f);
    Gtheta_bar.resize(Ncells, 0.f);

    const size_t Ntri = getTriangleCount();

    std::vector<float> denom_sum;
    denom_sum.resize(Ncells, 0.f);
    std::vector<uint> cell_tri_count;
    cell_tri_count.resize(Ncells, 0);

    // Compute G(theta) for each triangle
    for (size_t t = 0; t < Ntri; t++) {

        Triangulation tri = getTriangle(t);
        int cell = tri.gridcell;

        if (cell >= 0 && cell < Ncells) { // triangle is inside a grid cell

            helios::vec3 t0 = tri.vertex0;
            helios::vec3 t1 = tri.vertex1;
            helios::vec3 t2 = tri.vertex2;

            helios::vec3 v0 = t1 - t0;
            helios::vec3 v1 = t2 - t0;
            helios::vec3 v2 = t2 - t1;

            float L0 = v0.magnitude();
            float L1 = v1.magnitude();
            float L2 = v2.magnitude();

            // Heron's formula for triangle area
            float S = 0.5f * (L0 + L1 + L2);
            float area = sqrt(S * (S - L0) * (S - L1) * (S - L2));

            // Triangle normal from two edges sharing vertex t0 (standard convention). Sign is
            // irrelevant here since only fabs(normal . raydir) is used below.
            helios::vec3 normal = cross(v0, v1);
            normal.normalize();

            helios::vec3 raydir = t0 - getScanOrigin(tri.scanID);
            raydir.normalize();

            float theta = fabs(acos_safe(raydir.z));

            // Skip degenerate triangles: Heron's formula yields NaN (slightly-negative radicand
            // from float error) or zero area for collinear/zero-extent vertices.
            if (std::isfinite(area) && area > 0.f) {
                float normal_dot_ray = fabs(normal * raydir);
                Gtheta.at(cell) += normal_dot_ray * area * fabs(sin(theta));
                denom_sum.at(cell) += fabs(sin(theta)) * area;
                cell_tri_count.at(cell) += 1;
            }
        }
    }

    // Normalize by denominator and average over scans
    for (uint v = 0; v < Ncells; v++) {
        if (cell_tri_count[v] > 0) {
            Gtheta[v] = Gtheta[v] / denom_sum[v];
            Gtheta_bar[v] += Gtheta[v] / float(Nscans);
        }
    }
}

bool LiDARcloud::invertLAD(uint voxel_index, float P, float Gtheta, const std::vector<float> &dr_samples, int min_voxel_hits, const helios::vec3 &gridsize, float &leaf_area, helios::WarningAggregator &warnings) {

    // Validation checks
    if (Gtheta == 0 || Gtheta != Gtheta) { // Check for zero or NaN
        leaf_area = 0.0f;
        return false;
    }

    if (dr_samples.size() < min_voxel_hits) {
        leaf_area = 0.0f;
        return false;
    }

    // Secant method parameters
    float etol = 5e-5f;
    uint maxiter = 100;

    // Relative-error denominator. P is a transmission probability in [0,1]; a fully
    // intercepted voxel (P==0, dense closed canopy) is physically valid but would make
    // the relative error fabs(mean-P)/P divide by zero. Clamp the denominator to a small
    // positive value so the secant iteration stays finite; the analytic fallback below
    // additionally guards -log(P).
    const float P_error_denom = fmax(P, 1e-6f);

    // Initial guesses
    float a = 0.1f;
    float h = 0.01f;

    // Compute initial error
    float mean = 0.f;
    for (size_t j = 0; j < dr_samples.size(); j++) {
        mean += exp(-a * dr_samples[j] * Gtheta);
    }
    mean /= float(dr_samples.size());
    float error = fabs(mean - P) / P_error_denom;

    float tmp = a;
    a = a + h;

    // Secant method iteration
    uint iter = 0;
    float aold, eold;
    while (error > etol && iter < maxiter) {

        aold = tmp;
        eold = error;

        mean = 0.f;
        for (size_t j = 0; j < dr_samples.size(); j++) {
            mean += exp(-a * dr_samples[j] * Gtheta);
        }
        mean /= float(dr_samples.size());
        error = fabs(mean - P) / P_error_denom;

        tmp = a;

        if (error == eold) {
            break; // No progress
        }

        // Secant update
        a = fabs((aold * error - a * eold) / (error - eold));
        iter++;
    }

    // Calculate mean dr
    float dr_bar = 0.0f;
    for (size_t i = 0; i < dr_samples.size(); i++) {
        dr_bar += dr_samples[i];
    }
    dr_bar /= float(dr_samples.size());

    // Check convergence and use fallback if needed. The secant loop can terminate
    // without finding a root (the "no progress" break at error == eold, or hitting
    // maxiter), leaving 'a' near the 0.1 initial guess. Such a stall must NOT be
    // treated as converged: require the achieved error to actually be below the
    // tolerance, not merely that 'a' is finite. Otherwise a stalled solve silently
    // returns leaf_area ~= 0.1 * volume instead of the physically-correct fallback.
    bool converged = (error <= etol && a == a && a <= 100);
    bool used_fallback = false;

    if (!converged) {
        warnings.addWarning("invertLAD_did_not_converge", "LAD inversion failed for volume #" + std::to_string(voxel_index) + ". Using average dr formulation.");
        a = (1.f - P) / (dr_bar * Gtheta);
        used_fallback = true;
    }

    // Additional constraint for high LAD values
    if (a > 5) {
        a = fmin((1.f - P) / dr_bar / Gtheta, -log(P_error_denom) / dr_bar / Gtheta);
    }

    // Compute final leaf area
    leaf_area = a * gridsize.x * gridsize.y * gridsize.z;

    return true;
}

LiDARcloud::LADInversionResult LiDARcloud::invertLADWithVariance(uint voxel_index, float P, float Gtheta, const std::vector<float> &dr_samples, float sum_frac_sq, float element_width, int min_voxel_hits, const helios::vec3 &gridsize,
                                                                 helios::WarningAggregator &warnings) {

    // The point estimate is produced by the existing Beer-Lambert inversion (unchanged). On top of
    // it we compute the per-voxel statistical SAMPLING variance of LAD following Pimont et al.
    // (2018), RSE 215:343-370. The point estimator solves mean(exp(-a*dr*Gtheta)) = P, which for
    // equal path lengths is a = -log(1-I)/(dr_bar*Gtheta), i.e. the Beer-Lambert estimator with
    // a = LAD (the projection coefficient Gtheta is folded into the exponent). We therefore use the
    // Beer-Lambert delta-method variance, consistent with this estimator: with I = 1 - P,
    //   d a / d I = 1 / ((1-I) * dr_bar * Gtheta)
    //   var(a)    = var(I) / ((1-I)^2 * dr_bar^2 * Gtheta^2)
    // and var(I) is the sum of (a) a finite-N sampling term and (b) an N-independent
    // element-position-variability term.

    LADInversionResult result;
    result.beam_count = (int) dr_samples.size();
    result.I_rdi = 1.f - P;

    // Point estimate (unchanged behavior).
    float leaf_area = 0.f;
    bool ok = invertLAD(voxel_index, P, Gtheta, dr_samples, min_voxel_hits, gridsize, leaf_area, warnings);
    result.leaf_area = leaf_area;
    result.converged = ok;

    // Mean and variance of the per-beam path lengths.
    const int N = result.beam_count;
    if (N > 0) {
        float sum = 0.f;
        for (float d: dr_samples) {
            sum += d;
        }
        result.zbar_e = sum / float(N);
        float ss = 0.f;
        for (float d: dr_samples) {
            ss += (d - result.zbar_e) * (d - result.zbar_e);
        }
        result.var_path = ss / float(N);
    }

    // Variance is only defined for a successful inversion with a usable geometry.
    const float dr_bar = result.zbar_e;
    if (!ok || N < min_voxel_hits || Gtheta <= 0.f || Gtheta != Gtheta || dr_bar <= 0.f) {
        result.LAD_variance = -1.f;
        return result;
    }

    const float I = result.I_rdi;
    // Bounded RDI for numerical stability near the fully-intercepted (I -> 1) case (Pimont Eq. C26).
    const float I_b = std::min(I, 1.f - 1.f / (2.f * float(N) + 2.f));

    // --- Term (a): finite-beam sampling variance of the RDI ---
    // Binomial variance I_b(1-I_b)/N is a provable upper bound on the variance of the per-beam
    // transmittance mean (a [0,1]-bounded statistic). For multi-return data the per-beam fraction
    // carries sub-beam information; we guard against model mismatch by taking the larger of the
    // binomial bound and the empirical variance of the per-beam fractions (for single-return data
    // the per-beam fraction is in {0,1}, so the empirical variance equals the binomial bound and
    // the guard is a no-op).
    const float binomial_varI = I_b * (1.f - I_b) / float(N);
    float empirical_varI = (sum_frac_sq / float(N) - P * P) / float(N); // var of the per-beam mean
    if (empirical_varI < 0.f) {
        empirical_varI = 0.f; // floating-point guard
    }
    const float var_I_a = std::max(binomial_varI, empirical_varI);

    // --- Term (b): N-independent element-position-variability variance of the RDI ---
    // Requires the single-element optical depth L1 = lambda1*delta (Pimont Appendix A). For a flat
    // leaf of width w the mean element cross section is S1 = pi*w^2/8, giving L1 = pi*w^2/(8*delta^2)
    // with delta the characteristic voxel size. The asymptotic RDI variance is the empirical fit
    // sigma2_Iinf = 0.23 * L1 * (1-I) * I^(1.9 - 2.3*L1) (valid L1 < 0.3; Pimont Fig. 2).
    float var_I_b = 0.f;
    if (element_width > 0.f) {
        const float volume = gridsize.x * gridsize.y * gridsize.z;
        const float delta = std::cbrt(volume); // characteristic voxel size [m]
        const float L1 = (float) (M_PI) *element_width * element_width / (8.f * delta * delta);
        result.L1_element = L1;
        result.element_size_known = true;
        if (L1 < 0.3f && I > 0.f && I < 1.f) {
            var_I_b = 0.23f * L1 * (1.f - I) * std::pow(I, 1.9f - 2.3f * L1);
            if (var_I_b < 0.f) {
                var_I_b = 0.f;
            }
        }
    }

    // Delta-method propagation to var(a). Use the bounded RDI in the denominator to stay finite.
    const float denom = (1.f - I_b) * (1.f - I_b) * dr_bar * dr_bar * Gtheta * Gtheta;
    result.LAD_variance = (var_I_a + var_I_b) / denom;

    return result;
}

bool LiDARcloud::ciValidPimont(float L, float L1, int N, float confidence_level) const {
    // Range-of-validity envelope for the confidence interval of the bias-corrected estimator
    // (Pimont et al. 2018, Table 3). Outside these ranges the Wald interval is not trustworthy, so
    // the caller refuses to emit one. When the element size was not supplied (L1 < 0) the
    // element-position term was omitted, leaving a sampling-only variance; we gate as if elements
    // were small (L1 = 0), which is the most favorable assumption (documented as sampling-only).
    const float l1 = (L1 < 0.f) ? 0.f : L1;
    if (confidence_level > 0.925f) { // 95% envelope
        return (L <= 2.0f && l1 <= 0.05f && N >= 30) || (L <= 2.5f && l1 <= 0.01f && N >= 150) || (l1 <= 0.05f && N >= 150);
    }
    // 90% envelope (also used as a conservative proxy for confidence levels below 0.90)
    return (L >= 0.5f && L <= 2.0f && l1 <= 0.05f && N >= 40) || (l1 <= 0.01f && N >= 100) || (l1 <= 0.05f && N >= 200);
}

void LiDARcloud::calculateLeafArea(helios::Context *context) {
    calculateLeafArea(context, 1);
}

void LiDARcloud::calculateLeafArea(helios::Context *context, int min_voxel_hits) {
    // Default characteristic element width of 5 cm (matches the leaf-reconstruction fallback). This
    // feeds only the element-position term of the LAD sampling-uncertainty estimate; the leaf-area
    // point estimate is independent of it.
    calculateLeafArea(context, min_voxel_hits, 0.05f);
}

void LiDARcloud::calculateLeafArea(helios::Context *context, int min_voxel_hits, float element_width) {
    // Triangulation-derived G(theta) (the original behavior). Sentinel < 0 => compute G(theta) per voxel.
    calculateLeafArea_inner(context, min_voxel_hits, element_width, -1.f);
}

void LiDARcloud::calculateLeafArea(helios::Context *context, float Gtheta, int min_voxel_hits, float element_width) {
    // Caller-supplied G(theta) for scans that cannot be triangulated (e.g. moving-platform scans).
    if (!(Gtheta > 0.f) || Gtheta > 1.f) {
        helios_runtime_error("ERROR (LiDARcloud::calculateLeafArea): The supplied G(theta) must be in the range (0,1], but " + std::to_string(Gtheta) + " was provided. Use 0.5 for a spherical (random) leaf-angle distribution.");
    }
    calculateLeafArea_inner(context, min_voxel_hits, element_width, Gtheta);
}

void LiDARcloud::calculateLeafArea_inner(helios::Context *context, int min_voxel_hits, float element_width, float supplied_Gtheta) {

    const bool use_supplied_Gtheta = (supplied_Gtheta > 0.f);

    if (printmessages) {
        std::cout << "Calculating leaf area (CollisionDetection)..." << std::endl;
    }

    // Validation checks (same as GPU version). Triangulation is required only to estimate G(theta); when the caller
    // supplies G(theta) directly (e.g. for a moving-platform scan, which cannot be triangulated) it is not needed.
    if (!use_supplied_Gtheta && !triangulationcomputed) {
        helios_runtime_error("ERROR (LiDARcloud::calculateLeafAreaCD): Triangulation must be performed prior to leaf area calculation. See triangulateHitPoints(). For scans that cannot be triangulated (e.g. moving-platform scans), use the "
                             "calculateLeafArea overload that takes a G(theta) argument.");
    }

    if (!hitgridcellcomputed) {
        calculateHitGridCell();
    }

    // Initialize CollisionDetection if needed
    initializeCollisionDetection(context);

    const uint Nscans = getScanCount();
    const uint Ncells = getGridCellCount();

    // Leaf-area inversion requires the full fired-beam population, including pulses
    // that returned nothing (misses / transmitted beams) so that the per-voxel
    // transmission probability has a valid denominator. Misses must be supplied
    // upstream: either the imported scan format retains them, or gapfillMisses() was
    // run to synthesize them. We fail fast (rather than silently producing biased LAD)
    // if no misses are present. This single beam-based equal-weighting algorithm
    // handles both single- and multi-return data: each pulse's returns are grouped
    // into a beam (one return per beam for single-return data) and classified
    // before/inside/after/miss relative to each voxel; P = E_after/(E_inside+E_after).
    if (!hasMisses()) {
        helios_runtime_error(
                "ERROR (LiDARcloud::calculateLeafArea): No miss points found in the point cloud. Leaf area inversion requires fired pulses that returned nothing (misses) in order to count transmitted beams. Provide a scan format that retains misses, or call gapfillMisses() to synthesize them, before calling calculateLeafArea().");
    }

    if (printmessages) {
        if (isMultiReturnData()) {
            std::cout << "Multi-return data detected - using beam-based equal weighting algorithm (CD)" << std::endl;
        } else {
            std::cout << "Single-return data with misses - using beam-based equal weighting algorithm (CD)" << std::endl;
        }
    }

    {
        // ============ BEAM-BASED EQUAL WEIGHTING ALGORITHM (CPU) ============

        // Additional arrays for equal weighting P calculation
        std::vector<std::vector<float>> P_equal_numerator_array(Ncells);
        std::vector<std::vector<float>> P_equal_denominator_array(Ncells);
        // Sum of squared per-beam transmittance fractions, per scan per voxel. Used by the
        // LAD sampling-variance estimate to guard the binomial variance against the empirical
        // spread of multi-return per-beam fractions (see invertLADWithVariance()).
        std::vector<std::vector<float>> P_equal_sumsq_array(Ncells);
        std::vector<std::vector<float>> dr_array(Ncells);

        // Initialize aggregation arrays
        std::vector<std::vector<float>> dr_agg;
        dr_agg.resize(Ncells);
        std::vector<float> Gtheta_bar;
        Gtheta_bar.resize(Ncells, 0.f);

        // Process each scan
        for (uint s = 0; s < Nscans; s++) {

            // Collect hits for this scan
            std::vector<helios::vec3> this_scan_xyz;
            std::vector<uint> this_scan_index;
            // global hit index -> local position in this scan's arrays. A flat vector (rather than a
            // std::map) keeps the per-beam lookups in the classification loop below O(1) and cache-friendly:
            // that loop performs ~Ncells * Nbeams lookups per scan (billions for a dense scan), so a
            // red-black-tree lookup here dominates the whole inversion. Unused entries stay at the sentinel.
            std::vector<uint> global_to_local(getHitCount(), 0);
            for (size_t r = 0; r < getHitCount(); r++) {
                if (getHitScanID(r) == s) {
                    global_to_local[(uint) r] = (uint) this_scan_xyz.size();
                    this_scan_xyz.push_back(getHitXYZ(r));
                    this_scan_index.push_back(r);
                }
            }
            size_t Nhits = this_scan_xyz.size();
            if (Nhits == 0)
                continue;

            // Group hits by timestamp into beams. beam_array holds GLOBAL hit indices;
            // the per-voxel classification arrays (dr, hit_location) are local to this
            // scan, so beam members are mapped back to local positions below.
            BeamGrouping beams = groupHitsByTimestamp(this_scan_index);
            uint Nbeams = beams.Nbeams;

            // Per-hit beam emission origin. For a moving-platform scan each pulse was fired from a different position,
            // so the beam geometry (direction and voxel entry/exit) must be measured from that pulse's own origin -
            // getHitOrigin() returns the per-pulse origin for moving scans and the single scan origin for static scans.
            // Precomputed once here (out of the Ncells x Nhits hot loop, which would otherwise repeat the map lookups).
            std::vector<helios::vec3> this_scan_origin(Nhits);
            for (size_t i = 0; i < Nhits; i++) {
                this_scan_origin[i] = getHitOrigin(this_scan_index[i]);
            }

            // CPU-based voxel intersection with hit_location classification
            std::vector<float> dr(Nhits, 0.0f);
            std::vector<uint> hit_location(Nhits, 0);

            // Process each voxel
            for (uint c = 0; c < Ncells; c++) {

                helios::vec3 center = getCellCenter(c);
                helios::vec3 size = getCellSize(c);
                float rotation = getCellRotation(c);

                // Reset for this voxel
                std::fill(dr.begin(), dr.end(), 0.0f);
                std::fill(hit_location.begin(), hit_location.end(), 0);

// Test each hit against this voxel (CPU/OpenMP)
#pragma omp parallel for
                for (int i = 0; i < static_cast<int>(Nhits); i++) {
                    helios::vec3 hit_xyz = this_scan_xyz[i];
                    helios::vec3 origin = this_scan_origin[i]; // this beam's emission origin (per-pulse for moving scans)

                    // Inverse rotate if needed. The voxel may be rotated about its center; apply the same inverse
                    // rotation to BOTH the hit point and the beam origin so the ray-voxel geometry stays consistent.
                    if (fabs(rotation) > 1e-6f) {
                        helios::vec3 anchor = center;
                        hit_xyz = rotatePointAboutLine(hit_xyz - anchor, helios::make_vec3(0, 0, 0), helios::make_vec3(0, 0, 1), -rotation) + anchor;
                        origin = rotatePointAboutLine(origin - anchor, helios::make_vec3(0, 0, 0), helios::make_vec3(0, 0, 1), -rotation) + anchor;
                    }

                    // Ray from origin to hit
                    helios::vec3 direction = hit_xyz - origin;
                    float hit_distance = direction.magnitude();
                    direction.normalize();

                    // AABB bounds
                    helios::vec3 voxel_min = center - size * 0.5f;
                    helios::vec3 voxel_max = center + size * 0.5f;

                    // Ray-AABB intersection (slab method)
                    float tx_min = (voxel_min.x - origin.x) / direction.x;
                    float tx_max = (voxel_max.x - origin.x) / direction.x;
                    if (tx_min > tx_max)
                        std::swap(tx_min, tx_max);

                    float ty_min = (voxel_min.y - origin.y) / direction.y;
                    float ty_max = (voxel_max.y - origin.y) / direction.y;
                    if (ty_min > ty_max)
                        std::swap(ty_min, ty_max);

                    float tz_min = (voxel_min.z - origin.z) / direction.z;
                    float tz_max = (voxel_max.z - origin.z) / direction.z;
                    if (tz_min > tz_max)
                        std::swap(tz_min, tz_max);

                    float t0 = std::max({tx_min, ty_min, tz_min});
                    float t1 = std::min({tx_max, ty_max, tz_max});

                    // Classify each hit/beam termination by where it lies along the beam
                    // relative to this voxel's entry (t0) and exit (t1):
                    //   1 = before voxel (return stopped short of it)
                    //   2 = inside voxel (return terminated within it)
                    //   3 = after voxel (beam passed through and terminated beyond the exit)
                    // A "miss" (a fired pulse that returned nothing, placed far out along the
                    // beam) is, geometrically, simply a beam that passed through and kept going,
                    // so it is class-3 like any other transmitted beam. Classification is
                    // therefore purely geometric and INDEPENDENT of the absolute placement
                    // distance of the miss point: a miss at 1001 m and a miss at 20000 m both
                    // classify as "after voxel" and contribute identically to the transmission
                    // probability P. This is the correct Beer-Lambert treatment - a transmitted
                    // beam is a transmission event regardless of where (or whether) it eventually
                    // returned.
                    if (t0 < t1 && t1 > 1e-6f) {
                        dr[i] = fabs(t1 - t0);

                        if (hit_distance >= t0 && hit_distance <= t1) {
                            hit_location[i] = 2; // Inside voxel
                        } else if (hit_distance > t1) {
                            hit_location[i] = 3; // After voxel (transmitted through, incl. misses)
                        } else if (hit_distance < t0) {
                            hit_location[i] = 1; // Before voxel
                        }
                    }
                }

                // Beam-level processing
                float P_equal_numerator = 0;
                float P_equal_denominator = 0;
                float P_equal_sumsq = 0; // sum of squared per-beam fractions (for sampling-variance guard)

                for (uint k = 0; k < Nbeams; k++) {
                    float E_before = 0, E_inside = 0, E_after = 0;
                    float drr = 0;
                    int dr_count = 0; // Count returns with dr > 0

                    // Count returns in each location for this beam. Misses (transmitted beams)
                    // are class-3 (after voxel), so they are folded into E_after here.
                    for (uint j = 0; j < beams.beam_array.at(k).size(); j++) {
                        uint i = global_to_local.at(beams.beam_array.at(k).at(j)); // global -> local index

                        if (dr[i] > 0) {
                            drr += dr[i];
                            dr_count++;
                        }

                        if (hit_location[i] == 1)
                            E_before++;
                        else if (hit_location[i] == 2)
                            E_inside++;
                        else if (hit_location[i] == 3)
                            E_after++;
                    }

                    // Equal weighting P calculation - simple average per Eq. 7
                    // P = (1/B_tot) Σ[E_after / (E_inside + E_after)]. A fully-transmitted beam
                    // (E_inside == 0, E_after >= 1) contributes frac = 1; a beam that only
                    // terminated before the voxel (E_before only) contributes nothing.
                    if (E_inside != 0 || E_after != 0) {
                        float frac = E_after / (E_inside + E_after);
                        P_equal_numerator += frac;
                        P_equal_sumsq += frac * frac;
                        P_equal_denominator += 1;
                    }

                    // Average dr over returns that actually intersect voxel
                    if (dr_count > 0) {
                        float drrx = drr / float(dr_count);
                        dr_array.at(c).push_back(drrx);
                    }
                }

                P_equal_numerator_array.at(c).push_back(P_equal_numerator);
                P_equal_denominator_array.at(c).push_back(P_equal_denominator);
                P_equal_sumsq_array.at(c).push_back(P_equal_sumsq);
            }
        }

        // Obtain G(theta) per voxel. Normally computed from triangulation; when the caller supplied a value (e.g. for a
        // moving-platform scan that cannot be triangulated), apply that single value to every voxel instead.
        std::vector<float> Gtheta;
        if (use_supplied_Gtheta) {
            Gtheta.assign(Ncells, supplied_Gtheta);
        } else {
            computeGtheta(Ncells, Nscans, Gtheta, Gtheta_bar);
        }

        // LAD inversion with equal weighting P
        if (printmessages) {
            std::cout << "Inverting to find LAD..." << std::flush;
        }

        helios::WarningAggregator invertLAD_warnings;
        invertLAD_warnings.setEnabled(printmessages);

        for (uint v = 0; v < Ncells; v++) {
            // Calculate P using equal weighting formula
            float P = 0.0f;
            float P_num_sum = 0.0f, P_denom_sum = 0.0f, P_sumsq_sum = 0.0f;
            for (uint s = 0; s < P_equal_numerator_array[v].size(); s++) {
                P_num_sum += P_equal_numerator_array[v][s];
                P_denom_sum += P_equal_denominator_array[v][s];
                P_sumsq_sum += P_equal_sumsq_array[v][s];
            }
            if (P_denom_sum > 0) {
                P = P_num_sum / P_denom_sum;
            }

            // Aggregate dr across all scans
            for (uint s = 0; s < dr_array[v].size(); s++) {
                if (dr_array[v][s] > 0) {
                    dr_agg[v].push_back(dr_array[v][s]);
                }
            }

            // Apply min_voxel_hits filtering
            if (dr_agg[v].size() < min_voxel_hits) {
                setCellLeafArea(0, v);
                setCellGtheta(Gtheta[v], v);
                grid_cells.at(v).beam_count = (int) dr_agg[v].size();
                grid_cells.at(v).LAD_variance = -1.f;
                grid_cells.at(v).ci_valid = false;
                continue;
            }

            // Invert for leaf area AND its sampling variance. The point estimate is unchanged from
            // the shared invertLAD(); the additional output quantifies statistical sampling
            // uncertainty (Pimont et al. 2018), which is stored on the grid cell.
            helios::vec3 gridsize = getCellSize(v);
            LADInversionResult inv = invertLADWithVariance(v, P, Gtheta[v], dr_agg[v], P_sumsq_sum, element_width, min_voxel_hits, gridsize, invertLAD_warnings);

            setCellLeafArea(inv.leaf_area, v);
            setCellGtheta(Gtheta[v], v);

            GridCell &cell = grid_cells.at(v);
            cell.beam_count = inv.beam_count;
            cell.I_rdi = inv.I_rdi;
            cell.zbar_e = inv.zbar_e;
            cell.var_path = inv.var_path;
            cell.L1_element = inv.L1_element;
            cell.LAD_variance = inv.LAD_variance;

            // Pre-evaluate CI validity at the 95% level for export / quick filtering. Per-query
            // accessors re-check validity at the requested confidence level. L = lambda*delta is the
            // voxel optical depth: lambda = a*Gtheta and a = leaf_area/volume, so L = a*Gtheta*zbar_e.
            const float volume = gridsize.x * gridsize.y * gridsize.z;
            const float a = (volume > 0.f) ? inv.leaf_area / volume : 0.f;
            const float L = a * Gtheta[v] * inv.zbar_e;
            cell.ci_valid = (inv.LAD_variance >= 0.f) && ciValidPimont(L, inv.L1_element, inv.beam_count, 0.95f);
        }

        invertLAD_warnings.report(std::cerr);

        if (printmessages) {
            std::cout << "done." << std::endl;
        }
    }
}

// Deprecated wrapper functions for backward compatibility
void LiDARcloud::calculateLeafAreaGPU(helios::Context *context) {
    calculateLeafArea(context);
}

void LiDARcloud::calculateLeafAreaGPU(helios::Context *context, int min_voxel_hits) {
    calculateLeafArea(context, min_voxel_hits);
}

void LiDARcloud::enableGPUAcceleration() {
    if (collision_detection != nullptr) {
        collision_detection->enableGPUAcceleration();
    }
}

void LiDARcloud::disableGPUAcceleration() {
    if (collision_detection != nullptr) {
        collision_detection->disableGPUAcceleration();
    }
}

void LiDARcloud::calculateHitGridCell() {

    if (printmessages) {
        std::cout << "Grouping hit points by grid cell (CPU)..." << std::flush;
    }

    const size_t total_hits = getHitCount();
    const uint Ncells = getGridCellCount();

    if (total_hits == 0) {
        std::cout << "WARNING (calculateHitGridCellCD): There are no hits currently in the point cloud. Skipping grid cell binning calculation." << std::endl;
        return;
    }

    // Hoist the (constant) voxel geometry out of the per-hit loop into flat arrays. The inner loop
    // below runs up to total_hits * Ncells times (billions for a dense scan), so re-fetching each
    // cell's center/anchor/size/rotation through the bounds-checked getters every iteration dominates
    // the cost. Caching them once in contiguous vectors keeps the hot loop reading from cache instead.
    std::vector<helios::vec3> cell_min(Ncells), cell_max(Ncells), cell_anchor(Ncells);
    std::vector<float> cell_rotation(Ncells);
    std::vector<bool> cell_rotated(Ncells);
    for (uint c = 0; c < Ncells; c++) {
        helios::vec3 center = getCellCenter(c);
        helios::vec3 size = getCellSize(c);
        cell_min[c] = center - size * 0.5f;
        cell_max[c] = center + size * 0.5f;
        cell_anchor[c] = getCellGlobalAnchor(c);
        cell_rotation[c] = getCellRotation(c);
        cell_rotated[c] = (fabs(cell_rotation[c]) > 1e-6f);
    }

// Process each hit point (parallelized with OpenMP)
#pragma omp parallel for schedule(dynamic, 1000)
    for (int r = 0; r < static_cast<int>(total_hits); r++) {

        helios::vec3 hit_xyz = getHitXYZ(r);
        int assigned_cell = -1; // Default: not in any cell

        // Test against each voxel. The original ray-from-origin slab test reduces exactly to a
        // point-in-AABB containment test: the ray is cast from the origin through the hit point P,
        // so it passes through P at parameter T = |P|, and "T lies within the box's [t0,t1] entry/exit
        // interval" is true iff P lies within the box bounds. (The old t1 > 1e-6 guard only excluded a
        // box containing the origin, which cannot happen for a hit point away from the origin.)
        for (uint c = 0; c < Ncells; c++) {

            // Inverse rotate hit point into the voxel's local axis-aligned frame if the voxel is rotated.
            helios::vec3 p = hit_xyz;
            if (cell_rotated[c]) {
                p = rotatePointAboutLine(hit_xyz - cell_anchor[c], helios::make_vec3(0, 0, 0), helios::make_vec3(0, 0, 1), -cell_rotation[c]) + cell_anchor[c];
            }

            const helios::vec3 &lo = cell_min[c];
            const helios::vec3 &hi = cell_max[c];
            if (p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y && p.z >= lo.z && p.z <= hi.z) {
                assigned_cell = c;
                break; // Found the cell, stop searching
            }
        }

        // Store result (thread-safe due to unique index per thread)
        setHitGridCell(r, assigned_cell);
    }

    if (printmessages) {
        std::cout << "done." << std::endl;
    }

    hitgridcellcomputed = true;
}

bool LiDARcloud::isMultiReturnData() const {
    // Check if any hit has target_count > 1 (multi-return indicator)
    for (size_t r = 0; r < getHitCount(); r++) {
        if (doesHitDataExist(r, "target_count")) {
            if (getHitData(r, "target_count") > 1) {
                // Multi-return data requires timestamp for beam grouping
                if (!doesHitDataExist(r, "timestamp")) {
                    helios_runtime_error("ERROR (isMultiReturnData): Multi-return data detected (target_count > 1) but 'timestamp' field is missing. Cannot group hits into beams.");
                }
                // Multi-return data requires target_index for triangulation filtering
                if (!doesHitDataExist(r, "target_index")) {
                    helios_runtime_error("ERROR (isMultiReturnData): Multi-return data detected (target_count > 1) but 'target_index' field is missing. Cannot filter first returns for triangulation.");
                }
                return true;
            }
        }
    }
    return false;
}

bool LiDARcloud::isHitMiss(uint index) const {
    // Canonical miss flag (matches the Python `is_miss` LAS extra dimension): a hit is
    // a "miss" (the pulse was fired but returned nothing - transmitted to the sky) when
    // is_miss == 1. This flag is the durable contract and is set by every path that
    // produces misses (the importer, gapfillMisses(), and syntheticScan()), so the
    // distance fallback below is not reached for any data Helios produces. Note that the
    // placement distance of a miss point is path-dependent: gapfillMisses() uses
    // LIDAR_MISS_DISTANCE, while syntheticScan() leaves misses at the ray-tracer no-hit
    // distance LIDAR_RAYTRACE_MISS_T so they classify as transmitted-through beams in the
    // leaf-area inversion. Miss classification must therefore key on the flag, not distance.
    if (doesHitDataExist(index, "is_miss")) {
        return getHitData(index, "is_miss") != 0.0;
    }
    // Interim fallback for legacy data that predates the is_miss flag: treat a return whose
    // range from its scan origin reaches the gapfill miss sentinel distance as a miss. This
    // only catches the gapfillMisses() convention; flagged data never reaches this line.
    helios::vec3 d = getHitXYZ(index) - getScanOrigin(getHitScanID(index));
    return d.magnitude() >= 0.98f * LIDAR_MISS_DISTANCE;
}

bool LiDARcloud::hasMisses() const {
    for (size_t r = 0; r < getHitCount(); r++) {
        if (isHitMiss(r)) {
            return true;
        }
    }
    return false;
}

LiDARcloud::BeamGrouping LiDARcloud::groupHitsByTimestamp(const std::vector<uint> &scan_indices) const {

    BeamGrouping result;

    if (scan_indices.empty()) {
        result.Nbeams = 0;
        return result;
    }

    // Timestamp groups multiple returns of one pulse into a single beam. When the
    // data has no timestamp (e.g. single-return data where each pulse yields at most
    // one return), there is nothing to group: each hit is its own one-return beam.
    bool has_timestamp = true;
    for (uint idx: scan_indices) {
        if (!doesHitDataExist(idx, "timestamp")) {
            has_timestamp = false;
            break;
        }
    }
    if (!has_timestamp) {
        result.Nbeams = scan_indices.size();
        result.beam_array.resize(result.Nbeams);
        for (uint i = 0; i < scan_indices.size(); i++) {
            result.beam_array.at(i).push_back(scan_indices[i]);
        }
        return result;
    }

    // Sort indices by timestamp to group consecutive hits from same beam
    std::vector<uint> sorted_indices = scan_indices;
    std::sort(sorted_indices.begin(), sorted_indices.end(), [this](uint a, uint b) { return this->getHitData(a, "timestamp") < this->getHitData(b, "timestamp"); });

    // Count unique beams by counting timestamp changes (now works correctly with sorted data)
    double previous_time = -1.0;
    result.Nbeams = 0;
    for (uint i = 0; i < sorted_indices.size(); i++) {
        double current_time = getHitData(sorted_indices[i], "timestamp");
        if (current_time != previous_time) {
            result.Nbeams++;
            previous_time = current_time;
        }
    }

    // Group hits by beam (same timestamp = same beam)
    result.beam_array.resize(result.Nbeams);
    double previous_beam = getHitData(sorted_indices[0], "timestamp");
    uint beam_ID = 0;

    for (uint i = 0; i < sorted_indices.size(); i++) {
        double current_beam = getHitData(sorted_indices[i], "timestamp");

        if (current_beam == previous_beam) {
            result.beam_array.at(beam_ID).push_back(sorted_indices[i]);
        } else {
            beam_ID++;
            result.beam_array.at(beam_ID).push_back(sorted_indices[i]);
            previous_beam = current_beam;
        }
    }

    return result;
}

std::vector<float> LiDARcloud::calculateSyntheticLeafArea(helios::Context *context) {

    std::vector<uint> UUIDs_all = context->getAllUUIDs();
    const uint N = UUIDs_all.size();
    const uint Ncells = getGridCellCount();

    // Result: which voxel each primitive belongs to (-1 if none)
    std::vector<int> prim_vol(N, -1);

// CPU/OpenMP version of primitive-to-voxel assignment
#pragma omp parallel for
    for (int p = 0; p < static_cast<int>(N); p++) {
        std::vector<helios::vec3> verts = context->getPrimitiveVertices(UUIDs_all[p]);
        helios::vec3 prim_xyz = verts[0]; // Use first vertex

        // Test against each voxel (same logic as calculateHitGridCellCD)
        for (uint c = 0; c < Ncells; c++) {
            helios::vec3 center = getCellCenter(c);
            helios::vec3 anchor = getCellGlobalAnchor(c);
            helios::vec3 size = getCellSize(c);
            float rotation = getCellRotation(c);

            // Inverse rotate primitive position if voxel is rotated
            helios::vec3 prim_xyz_rot = prim_xyz;
            if (fabs(rotation) > 1e-6f) {
                prim_xyz_rot = rotatePointAboutLine(prim_xyz - anchor, helios::make_vec3(0, 0, 0), helios::make_vec3(0, 0, 1), -rotation) + anchor;
            }

            // Point-in-AABB test (treating as ray from origin for consistency with GPU kernel)
            helios::vec3 origin_pt = helios::make_vec3(0, 0, 0);
            helios::vec3 direction = prim_xyz_rot - origin_pt;
            direction.normalize();

            // AABB bounds
            float x0 = center.x - 0.5f * size.x;
            float x1 = center.x + 0.5f * size.x;
            float y0 = center.y - 0.5f * size.y;
            float y1 = center.y + 0.5f * size.y;
            float z0 = center.z - 0.5f * size.z;
            float z1 = center.z + 0.5f * size.z;

            // Slab method
            float tx_min = (x0 - origin_pt.x) / direction.x;
            float tx_max = (x1 - origin_pt.x) / direction.x;
            if (tx_min > tx_max)
                std::swap(tx_min, tx_max);

            float ty_min = (y0 - origin_pt.y) / direction.y;
            float ty_max = (y1 - origin_pt.y) / direction.y;
            if (ty_min > ty_max)
                std::swap(ty_min, ty_max);

            float tz_min = (z0 - origin_pt.z) / direction.z;
            float tz_max = (z1 - origin_pt.z) / direction.z;
            if (tz_min > tz_max)
                std::swap(tz_min, tz_max);

            float t0 = std::max({tx_min, ty_min, tz_min});
            float t1 = std::min({tx_max, ty_max, tz_max});

            // Check if primitive is inside voxel
            if (t0 < t1 && t1 > 1e-6f) {
                float T = (prim_xyz_rot - origin_pt).magnitude();
                if (T >= t0 && T <= t1) {
                    prim_vol[p] = c;
                    break; // Found the voxel
                }
            }
        }
    }

    // Sum primitive areas per voxel
    std::vector<float> total_area(Ncells, 0.f);
    for (size_t p = 0; p < N; p++) {
        if (prim_vol[p] >= 0) {
            uint gridcell = prim_vol[p];
            total_area[gridcell] += context->getPrimitiveArea(UUIDs_all[p]);
            context->setPrimitiveData(UUIDs_all[p], "gridCell", gridcell);
        }
    }

    // Per-voxel leaf area to return
    std::vector<float> output_LeafArea(Ncells);
    for (uint v = 0; v < Ncells; v++) {
        output_LeafArea[v] = total_area[v];
    }

    // Annotate each primitive with the total leaf area of its containing voxel. Iterate
    // primitives (indexed 0..N), NOT cells: UUIDs_all is sized by primitive count, so
    // indexing it by a cell index is out of bounds whenever Ncells > N and otherwise maps
    // the wrong UUID to a voxel total.
    for (size_t p = 0; p < N; p++) {
        if (prim_vol[p] >= 0) {
            context->setPrimitiveData(UUIDs_all[p], "synthetic_leaf_area", total_area[prim_vol[p]]);
        }
    }

    return output_LeafArea;
}

void LiDARcloud::syntheticScan(helios::Context *context) {
    syntheticScan(context, 1, 0, false, false, true);
}

void LiDARcloud::syntheticScan(helios::Context *context, bool append) {
    syntheticScan(context, 1, 0, false, false, append);
}

void LiDARcloud::syntheticScan(helios::Context *context, bool scan_grid_only, bool record_misses) {
    syntheticScan(context, 1, 0, scan_grid_only, record_misses, true);
}

void LiDARcloud::syntheticScan(helios::Context *context, bool scan_grid_only, bool record_misses, bool append) {
    syntheticScan(context, 1, 0, scan_grid_only, record_misses, append);
}

void LiDARcloud::syntheticScan(helios::Context *context, int rays_per_pulse, float pulse_distance_threshold) {
    syntheticScan(context, rays_per_pulse, pulse_distance_threshold, false, false, true);
}

void LiDARcloud::syntheticScan(helios::Context *context, int rays_per_pulse, float pulse_distance_threshold, bool append) {
    syntheticScan(context, rays_per_pulse, pulse_distance_threshold, false, false, append);
}

void LiDARcloud::syntheticScan(helios::Context *context, int rays_per_pulse, float pulse_distance_threshold, bool scan_grid_only, bool record_misses) {
    syntheticScan(context, rays_per_pulse, pulse_distance_threshold, scan_grid_only, record_misses, true);
}

void LiDARcloud::syntheticScan(helios::Context *context, int rays_per_pulse, float pulse_distance_threshold, bool scan_grid_only, bool record_misses, bool append) {

    // Clear existing hit data if not appending
    if (!append) {
        hits.clear();
        Nhits = 0;
        // Reset hit tables for each scan
        for (auto &hit_table: hit_tables) {
            hit_table.resize(hit_table.Ntheta, hit_table.Nphi, -1);
        }
        hitgridcellcomputed = false;
        triangulationcomputed = false;
    }

    int Npulse;
    if (rays_per_pulse < 1) {
        Npulse = 1;
    } else {
        Npulse = rays_per_pulse;
    }

    if (printmessages) {
        if (Npulse > 1) {
            std::cout << "Performing multi-return synthetic LiDAR scan..." << std::endl;
        } else {
            std::cout << "Performing single-return synthetic LiDAR scan..." << std::endl;
        }
    }

    if (getScanCount() == 0) {
        std::cout << "WARNING (syntheticScan): No scans added to the point cloud. Exiting.." << std::endl;
        return;
    }

    // Ray-tracer no-hit threshold: a traced ray that intersects nothing returns this t value
    // (see performUnifiedRayTracing). Used below to classify whether a beam hit a primitive.
    // This is NOT where miss points are placed in the cloud (that is LIDAR_MISS_DISTANCE).
    float miss_distance = LIDAR_RAYTRACE_MISS_T;

    helios::vec3 bb_center;
    helios::vec3 bb_size;

    if (scan_grid_only == false) {

        // Determine bounding box for Context geometry
        helios::vec2 xbounds, ybounds, zbounds;
        context->getDomainBoundingBox(xbounds, ybounds, zbounds);
        bb_center = helios::make_vec3(xbounds.x + 0.5 * (xbounds.y - xbounds.x), ybounds.x + 0.5 * (ybounds.y - ybounds.x), zbounds.x + 0.5 * (zbounds.y - zbounds.x));
        bb_size = helios::make_vec3(xbounds.y - xbounds.x, ybounds.y - ybounds.x, zbounds.y - zbounds.x);

        // Pad any degenerate (zero-extent) axis so the AABB slab cull does not reject every ray for planar/flat
        // scene geometry (e.g. a single patch or a flat wall with no thickness). A zero-thickness axis forces the
        // slab test's entry and exit parameters to coincide (t0 == t1), failing the strict t0 < t1 test below for
        // any ray actually pointing at the plane. Giving the axis a tiny finite thickness keeps the slab test (and
        // its 1/ray_dir division for axis-aligned rays) well-conditioned without affecting non-degenerate geometry.
        const float bb_pad = 1e-4f;
        if (bb_size.x < bb_pad)
            bb_size.x = bb_pad;
        if (bb_size.y < bb_pad)
            bb_size.y = bb_pad;
        if (bb_size.z < bb_pad)
            bb_size.z = bb_pad;

    } else {

        // Determine bounding box for voxels instead of whole domain
        helios::vec3 boxmin, boxmax;
        getGridBoundingBox(boxmin, boxmax);
        bb_center = helios::make_vec3(boxmin.x + 0.5 * (boxmax.x - boxmin.x), boxmin.y + 0.5 * (boxmax.y - boxmin.y), boxmin.z + 0.5 * (boxmax.z - boxmin.z));
        bb_size = helios::make_vec3(boxmax.x - boxmin.x, boxmax.y - boxmin.y, boxmax.z - boxmin.z);

        // Pad any degenerate (zero-extent) axis so the AABB slab cull does not reject every ray for a single-layer
        // (flat) voxel grid. See the matching note in the domain-bounding-box branch above.
        const float bb_pad = 1e-4f;
        if (bb_size.x < bb_pad)
            bb_size.x = bb_pad;
        if (bb_size.y < bb_pad)
            bb_size.y = bb_pad;
        if (bb_size.z < bb_pad)
            bb_size.z = bb_pad;
    }

    // get geometry information and copy to GPU

    size_t c = 0;

    std::map<std::string, int> textures;
    std::map<std::string, helios::int2> texture_size;
    std::map<std::string, std::vector<std::vector<bool>>> texture_data;
    int tID = 0;

    std::vector<uint> UUIDs_all = context->getAllUUIDs();

    std::vector<uint> ID_mapping;

    //----- PATCHES ----- //

    // figure out how many patches
    size_t Npatches = 0;
    for (int p = 0; p < UUIDs_all.size(); p++) {
        if (context->getPrimitiveType(UUIDs_all.at(p)) == helios::PRIMITIVE_TYPE_PATCH) {
            Npatches++;
        }
    }

    ID_mapping.resize(Npatches);

    helios::vec3 *patch_vertex = (helios::vec3 *) malloc(4 * Npatches * sizeof(helios::vec3)); // allocate host memory
    int *patch_textureID = (int *) malloc(Npatches * sizeof(int)); // allocate host memory
    helios::vec2 *patch_uv = (helios::vec2 *) malloc(2 * Npatches * sizeof(helios::vec2)); // allocate host memory

    c = 0;
    for (int p = 0; p < UUIDs_all.size(); p++) {
        uint UUID = UUIDs_all.at(p);
        if (context->getPrimitiveType(UUID) == helios::PRIMITIVE_TYPE_PATCH) {
            std::vector<helios::vec3> verts = context->getPrimitiveVertices(UUID);
            patch_vertex[4 * c] = verts.at(0);
            patch_vertex[4 * c + 1] = verts.at(1);
            patch_vertex[4 * c + 2] = verts.at(2);
            patch_vertex[4 * c + 3] = verts.at(3);

            ID_mapping.at(c) = UUIDs_all.at(p);

            if (!context->getPrimitiveTextureFile(UUID).empty() && context->primitiveTextureHasTransparencyChannel(UUID)) {
                std::string tex = context->getPrimitiveTextureFile(UUID);
                std::map<std::string, int>::iterator it = textures.find(tex);
                if (it != textures.end()) { // texture already exits
                    patch_textureID[c] = textures.at(tex);
                } else { // new texture
                    patch_textureID[c] = tID;
                    textures[tex] = tID;
                    helios::int2 tsize = context->getPrimitiveTextureSize(UUID);
                    texture_size[tex] = helios::make_int2(tsize.x, tsize.y);
                    texture_data[tex] = *context->getPrimitiveTextureTransparencyData(UUID);
                    tID++;
                }

                std::vector<helios::vec2> uv = context->getPrimitiveTextureUV(UUID);
                if (uv.size() == 4) { // custom uv coordinates
                    patch_uv[2 * c] = uv.at(1);
                    patch_uv[2 * c + 1] = uv.at(3);
                } else { // default uv coordinates
                    patch_uv[2 * c] = helios::make_vec2(0, 0);
                    patch_uv[2 * c + 1] = helios::make_vec2(1, 1);
                }

            } else {
                patch_textureID[c] = -1;
            }

            c++;
        }
    }

    // GPU allocations removed - performUnifiedRayTracing uses CollisionDetection instead

    //----- TRIANGLES ----- //

    // figure out how many triangles
    size_t Ntriangles = 0;
    for (int p = 0; p < UUIDs_all.size(); p++) {
        if (context->getPrimitiveType(UUIDs_all.at(p)) == helios::PRIMITIVE_TYPE_TRIANGLE) {
            Ntriangles++;
        }
    }

    ID_mapping.resize(Npatches + Ntriangles);

    helios::vec3 *tri_vertex = (helios::vec3 *) malloc(3 * Ntriangles * sizeof(helios::vec3)); // allocate host memory
    int *tri_textureID = (int *) malloc(Ntriangles * sizeof(int)); // allocate host memory
    helios::vec2 *tri_uv = (helios::vec2 *) malloc(3 * Ntriangles * sizeof(helios::vec2)); // allocate host memory

    c = 0;
    for (int p = 0; p < UUIDs_all.size(); p++) {
        uint UUID = UUIDs_all.at(p);
        if (context->getPrimitiveType(UUID) == helios::PRIMITIVE_TYPE_TRIANGLE) {
            std::vector<helios::vec3> verts = context->getPrimitiveVertices(UUID);
            tri_vertex[3 * c] = verts.at(0);
            tri_vertex[3 * c + 1] = verts.at(1);
            tri_vertex[3 * c + 2] = verts.at(2);

            ID_mapping.at(Npatches + c) = UUIDs_all.at(p);

            if (!context->getPrimitiveTextureFile(UUID).empty() && context->primitiveTextureHasTransparencyChannel(UUID)) {
                std::string tex = context->getPrimitiveTextureFile(UUID);
                std::map<std::string, int>::iterator it = textures.find(tex);
                if (it != textures.end()) { // texture already exits
                    tri_textureID[c] = textures.at(tex);
                } else { // new texture
                    tri_textureID[c] = tID;
                    textures[tex] = tID;
                    helios::int2 tsize = context->getPrimitiveTextureSize(UUID);
                    texture_size[tex] = helios::make_int2(tsize.x, tsize.y);
                    texture_data[tex] = *context->getPrimitiveTextureTransparencyData(UUID);
                    tID++;
                }

                std::vector<helios::vec2> uv = context->getPrimitiveTextureUV(UUID);
                assert(uv.size() == 3);
                tri_uv[3 * c] = uv.at(0);
                tri_uv[3 * c + 1] = uv.at(1);
                tri_uv[3 * c + 2] = uv.at(2);

            } else {
                tri_textureID[c] = -1;
            }

            c++;
        }
    }

    // GPU allocations removed - performUnifiedRayTracing uses CollisionDetection instead

    // transfer texture data to GPU
    const int Ntextures = textures.size();

    helios::int2 masksize_max = helios::make_int2(0, 0);
    for (std::map<std::string, helios::int2>::iterator it = texture_size.begin(); it != texture_size.end(); ++it) {
        if (it->second.x > masksize_max.x) {
            masksize_max.x = it->second.x;
        }
        if (it->second.y > masksize_max.y) {
            masksize_max.y = it->second.y;
        }
    }

    bool *maskdata = (bool *) malloc(Ntextures * masksize_max.x * masksize_max.y * sizeof(bool)); // allocate host memory
    helios::int2 *masksize = (helios::int2 *) malloc(Ntextures * sizeof(helios::int2)); // allocate host memory

    for (std::map<std::string, helios::int2>::iterator it = texture_size.begin(); it != texture_size.end(); ++it) {
        std::string texture_file = it->first;

        int ID = textures.at(texture_file);

        masksize[ID] = it->second;

        int ind = 0;
        for (int j = 0; j < masksize_max.y; j++) {
            for (int i = 0; i < masksize_max.x; i++) {

                if (i < texture_size.at(texture_file).x && j < texture_size.at(texture_file).y) {
                    maskdata[ID * masksize_max.x * masksize_max.y + ind] = texture_data.at(texture_file).at(j).at(i);
                } else {
                    maskdata[ID * masksize_max.x * masksize_max.y + ind] = false;
                }
                ind++;
            }
        }
    }

    // GPU allocations removed - texture data no longer copied to GPU

    // Per-scan cache of decoded texture RGB pixel maps for sampling hit colors.
    // Row 0 is the top of the texture (matches readPNG/readJPEG storage); UV y=0 is the bottom.
    struct TextureColorMap {
        uint width = 0;
        uint height = 0;
        std::vector<helios::RGBcolor> pixels;
    };
    std::map<std::string, TextureColorMap> texture_color_cache;

    auto load_texture_colors = [&](const std::string &filename) -> const TextureColorMap & {
        auto it = texture_color_cache.find(filename);
        if (it != texture_color_cache.end()) {
            return it->second;
        }
        TextureColorMap entry;
        std::string ext;
        size_t dot = filename.find_last_of('.');
        if (dot != std::string::npos) {
            ext = filename.substr(dot);
            for (char &ch: ext) {
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            }
        }
        if (ext == ".png") {
            std::vector<helios::RGBAcolor> rgba;
            helios::readPNG(filename, entry.width, entry.height, rgba);
            entry.pixels.resize(rgba.size());
            for (size_t i = 0; i < rgba.size(); i++) {
                entry.pixels[i] = helios::make_RGBcolor(rgba[i].r, rgba[i].g, rgba[i].b);
            }
        } else if (ext == ".jpg" || ext == ".jpeg") {
            helios::readJPEG(filename, entry.width, entry.height, entry.pixels);
        }
        return texture_color_cache.emplace(filename, std::move(entry)).first->second;
    };

    auto sample_hit_color = [&](uint UUID, const helios::vec3 &hit_pos) -> helios::RGBcolor {
        const std::string tex_file = context->getPrimitiveTextureFile(UUID);
        if (tex_file.empty() || context->isPrimitiveTextureColorOverridden(UUID)) {
            return context->getPrimitiveColor(UUID);
        }
        const TextureColorMap &tex = load_texture_colors(tex_file);
        if (tex.pixels.empty()) {
            return context->getPrimitiveColor(UUID);
        }

        std::vector<helios::vec3> verts = context->getPrimitiveVertices(UUID);
        std::vector<helios::vec2> uvs = context->getPrimitiveTextureUV(UUID);
        helios::vec2 uv;

        helios::PrimitiveType ptype = context->getPrimitiveType(UUID);
        if (ptype == helios::PRIMITIVE_TYPE_PATCH) {
            // Patch corners are (BL, BR, TR, TL); project the hit onto the (BL->BR, BL->TL) basis.
            helios::vec3 e1 = verts[1] - verts[0];
            helios::vec3 e2 = verts[3] - verts[0];
            helios::vec3 d = hit_pos - verts[0];
            float e1_sq = e1 * e1;
            float e2_sq = e2 * e2;
            float s_param = (e1_sq > 0.f) ? (d * e1) / e1_sq : 0.f;
            float t_param = (e2_sq > 0.f) ? (d * e2) / e2_sq : 0.f;
            if (s_param < 0.f)
                s_param = 0.f;
            else if (s_param > 1.f)
                s_param = 1.f;
            if (t_param < 0.f)
                t_param = 0.f;
            else if (t_param > 1.f)
                t_param = 1.f;
            if (uvs.size() == 4) {
                uv = (1.f - s_param) * (1.f - t_param) * uvs[0] + s_param * (1.f - t_param) * uvs[1] + s_param * t_param * uvs[2] + (1.f - s_param) * t_param * uvs[3];
            } else {
                uv = helios::make_vec2(s_param, t_param);
            }
        } else if (ptype == helios::PRIMITIVE_TYPE_TRIANGLE && uvs.size() == 3) {
            helios::vec3 e1 = verts[1] - verts[0];
            helios::vec3 e2 = verts[2] - verts[0];
            helios::vec3 d = hit_pos - verts[0];
            float dot11 = e1 * e1;
            float dot12 = e1 * e2;
            float dot22 = e2 * e2;
            float dot1d = e1 * d;
            float dot2d = e2 * d;
            float denom = dot11 * dot22 - dot12 * dot12;
            if (std::fabs(denom) < 1e-20f) {
                return context->getPrimitiveColor(UUID);
            }
            float inv_denom = 1.f / denom;
            float beta = (dot22 * dot1d - dot12 * dot2d) * inv_denom;
            float gamma = (dot11 * dot2d - dot12 * dot1d) * inv_denom;
            uv = uvs[0] + beta * (uvs[1] - uvs[0]) + gamma * (uvs[2] - uvs[0]);
        } else {
            return context->getPrimitiveColor(UUID);
        }

        // Wrap UV into [0,1) so repeat-style mappings sample correctly.
        uv.x -= std::floor(uv.x);
        uv.y -= std::floor(uv.y);

        int px = static_cast<int>(uv.x * static_cast<float>(tex.width));
        if (px < 0)
            px = 0;
        if (px >= static_cast<int>(tex.width))
            px = static_cast<int>(tex.width) - 1;
        // Pixel rows: 0 at top, height-1 at bottom; UV y=0 at bottom.
        int py = static_cast<int>((1.f - uv.y) * static_cast<float>(tex.height));
        if (py < 0)
            py = 0;
        if (py >= static_cast<int>(tex.height))
            py = static_cast<int>(tex.height) - 1;
        return tex.pixels[static_cast<size_t>(py) * tex.width + static_cast<size_t>(px)];
    };

    helios::WarningAggregator scan_warnings;
    scan_warnings.setEnabled(printmessages);

    for (int s = 0; s < getScanCount(); s++) {

        helios::vec3 scan_origin = getScanOrigin(s);

        int Ntheta = getScanSizeTheta(s);
        int Nphi = getScanSizePhi(s);

        helios::vec2 thetarange = getScanRangeTheta(s);
        float thetamin = thetarange.x;
        float thetamax = thetarange.y;
        helios::vec2 phirange = getScanRangePhi(s);
        float phimin = phirange.x;
        float phimax = phirange.y;

        std::vector<std::string> column_format = getScanColumnFormat(s);

        // Global scanner orientation. This models the full roll/pitch/yaw pose of a real terrestrial scanner:
        //   - roll and pitch are the residual tilt of the scanner spin axis reported by the dual-axis inclinometer,
        //   - the azimuth offset is the compass heading (yaw) of the instrument about the local vertical.
        // The orientation rotates the entire fan of ray directions about the scanner origin using right-hand-rule
        // rotations, matching the right-handed, Z-up body frame used by commercial scanners (e.g. RIEGL SOCS).
        // The azimuth offset is a right-hand rotation about the world +z axis applied on top of the azimuth sweep; the
        // roll/pitch body axes are defined relative to the scan's azimuth-zero (phiMin) facing direction and rotate with
        // that heading:
        //   - the body "forward" axis is the horizontal projection of the phiMin scan direction, after the azimuth offset (Y_body),
        //   - the body "lateral" (right) axis completes the right-handed frame (X_body = Y_body x Z),
        //   - roll  = right-hand rotation about X_body (the lateral axis),
        //   - pitch = right-hand rotation about Y_body (the forward / azimuth-zero axis).
        // The rotation order is yaw (azimuth) first, then pitch, then roll. A level, north-facing scanner has
        // roll = pitch = azimuth = 0. When phiMin + azimuth = 0 the azimuth-zero direction is +y, so X_body = +x and
        // Y_body = +y (tilt reduces to roll about world-x, pitch about world-y).
        float scanTiltRoll = getScanTiltRoll(s);
        float scanTiltPitch = getScanTiltPitch(s);
        float scanAzimuthOffset = getScanAzimuthOffset(s);
        bool apply_azimuth = (scanAzimuthOffset != 0.f);
        bool apply_tilt = (scanTiltRoll != 0.f || scanTiltPitch != 0.f);
        const helios::vec3 tilt_pivot = helios::make_vec3(0, 0, 0); // rotate directions about the origin (pure rotation of the unit vector)
        const helios::vec3 vertical_axis = helios::make_vec3(0.f, 0.f, 1.f); // world +z: azimuth (yaw) rotation axis
        // Body frame from the azimuth-zero (phiMin) direction, offset by the scanner heading. sphere2cart with zero
        // elevation gives the horizontal heading; adding scanAzimuthOffset rotates that heading about world +z.
        const float heading = phimin + scanAzimuthOffset;
        const helios::vec3 forward_axis = helios::make_vec3(sinf(heading), cosf(heading), 0.f); // Y_body: azimuth-zero heading (after offset)
        const helios::vec3 lateral_axis = helios::make_vec3(cosf(heading), -sinf(heading), 0.f); // X_body = Y_body x (0,0,1)

        // Scan pattern determines how the (theta-index, phi-index) grid maps to zenith angles. For a raster scan the zenith is
        // uniformly spaced over [thetamin,thetamax]; for a spinning multibeam scan each theta-index is a laser channel fired at
        // its own fixed (generally non-uniform) zenith angle. Both patterns share the same azimuth sweep and grid storage.
        const ScanMetadata &scan = scans.at(s);
        const bool spinning_multibeam = (scan.scanPattern == SCAN_PATTERN_SPINNING_MULTIBEAM);

        // Moving-platform support: when the scan carries a 6-DOF trajectory (see addScanMoving), each grid cell's pulse has
        // its own acquisition time, emission origin, and orientation. The pulse time is t = t0 + ordinal*pulse_period, where
        // the pulse ordinal is the cell's position in the firing sequence (ordinal = Ntheta*j + i), matching the value written
        // to data["timestamp"]. For static scans is_moving is false, pulse_period defaults to 1.0 and t0 to 0.0, so the
        // timestamp equals the historical pulse ordinal and the per-cell origin equals the single static scan_origin.
        const bool is_moving = scan.isMoving;
        const double pulse_period = scan.pulse_period;
        const double pulse_t0 = scan.t0;
        // Fixed sensor boresight misalignment (body frame), applied to every beam direction before the platform quaternion.
        const helios::vec4 boresight_quat = quat_from_rpy(scan.boresight_rpy.x, scan.boresight_rpy.y, scan.boresight_rpy.z);

        std::vector<helios::vec3> raydir;
        raydir.resize(Ntheta * Nphi);

        // Per-cell beam emission origin. For static scans every entry is scan_origin (preserving the original single-origin
        // behavior); for moving scans each entry is the platform pose origin at that pulse's time.
        std::vector<helios::vec3> raygrid_origin;
        raygrid_origin.resize(Ntheta * Nphi, scan_origin);

        // Inclusive endpoint sampling: Nphi/Ntheta samples spanning [phimin,phimax] / [thetamin,thetamax].
        // Guard the (N-1) denominator so a single-row/column scan (N==1) samples once at the minimum angle
        // instead of dividing by zero and producing NaN ray directions.
        const float dphi = (Nphi > 1) ? (phimax - phimin) / float(Nphi - 1) : 0.f;
        const float dtheta = (Ntheta > 1) ? (thetamax - thetamin) / float(Ntheta - 1) : 0.f;

        for (uint j = 0; j < Nphi; j++) {
            float phi = phimin + float(j) * dphi;
            for (uint i = 0; i < Ntheta; i++) {
                float theta_z = spinning_multibeam ? scan.beamZenithAngles.at(i) : (thetamin + float(i) * dtheta);
                float theta_elev = 0.5f * M_PI - theta_z;
                helios::vec3 dir = sphere2cart(helios::make_SphericalCoord(1.f, theta_elev, phi));
                if (is_moving) {
                    // Trajectory-driven pose. The static scanTilt is not applied here (addScanMoving requires it to be zero):
                    // attitude is composed as dir_world = R(quat) * R(boresight) * dir_body, and the origin includes the lever arm.
                    const size_t ordinal = size_t(Ntheta) * j + i;
                    const double t = pulse_t0 + double(ordinal) * pulse_period;
                    helios::vec3 pos;
                    helios::vec4 quat;
                    scan.poseAt(t, pos, quat);
                    helios::vec3 dir_body = quat_rotate(boresight_quat, dir);
                    dir = quat_rotate(quat, dir_body);
                    dir.normalize();
                    raygrid_origin.at(Ntheta * j + i) = pos + quat_rotate(quat, scan.lever_arm);
                } else {
                    if (apply_azimuth) {
                        dir = rotatePointAboutLine(dir, tilt_pivot, vertical_axis, scanAzimuthOffset); // yaw about the world +z axis (heading offset)
                    }
                    if (apply_tilt) {
                        dir = rotatePointAboutLine(dir, tilt_pivot, lateral_axis, scanTiltRoll); // roll about the lateral (X_body) axis
                        dir = rotatePointAboutLine(dir, tilt_pivot, forward_axis, scanTiltPitch); // pitch about the forward (Y_body) axis
                    }
                }
                raydir.at(Ntheta * j + i) = dir;
            }
        }

        size_t N = Ntheta * Nphi;

        // Bounding box intersection test (CPU version, no CUDA)
        std::vector<uint> bb_hit(N, 0);

        // Calculate BB bounds once
        helios::vec3 bb_min = bb_center - bb_size * 0.5f;
        helios::vec3 bb_max = bb_center + bb_size * 0.5f;

        // Check if the (static) origin is inside the bounding box. For moving scans each pulse has its own origin, so this
        // fast path is only taken for static scans; moving scans evaluate the per-pulse origin inside the loop below.
        bool origin_inside_bb = !is_moving && (scan_origin.x >= bb_min.x && scan_origin.x <= bb_max.x && scan_origin.y >= bb_min.y && scan_origin.y <= bb_max.y && scan_origin.z >= bb_min.z && scan_origin.z <= bb_max.z);

        for (size_t r = 0; r < N; r++) {
            // If origin inside BB, all rays automatically hit
            if (origin_inside_bb) {
                bb_hit[r] = 1;
                continue;
            }

            // Per-pulse emission origin (equals scan_origin for static scans).
            const helios::vec3 cell_origin = raygrid_origin.at(r);

            // For a moving scan a pulse whose origin is inside the bounding box always interacts with the grid.
            if (is_moving && cell_origin.x >= bb_min.x && cell_origin.x <= bb_max.x && cell_origin.y >= bb_min.y && cell_origin.y <= bb_max.y && cell_origin.z >= bb_min.z && cell_origin.z <= bb_max.z) {
                bb_hit[r] = 1;
                continue;
            }

            helios::vec3 ray_dir = raydir.at(r);

            // AABB ray intersection using slab method
            float tx_min, tx_max, ty_min, ty_max, tz_min, tz_max;

            float a = 1.0f / ray_dir.x;
            if (a >= 0) {
                tx_min = (bb_min.x - cell_origin.x) * a;
                tx_max = (bb_max.x - cell_origin.x) * a;
            } else {
                tx_min = (bb_max.x - cell_origin.x) * a;
                tx_max = (bb_min.x - cell_origin.x) * a;
            }

            float b = 1.0f / ray_dir.y;
            if (b >= 0) {
                ty_min = (bb_min.y - cell_origin.y) * b;
                ty_max = (bb_max.y - cell_origin.y) * b;
            } else {
                ty_min = (bb_max.y - cell_origin.y) * b;
                ty_max = (bb_min.y - cell_origin.y) * b;
            }

            float c = 1.0f / ray_dir.z;
            if (c >= 0) {
                tz_min = (bb_min.z - cell_origin.z) * c;
                tz_max = (bb_max.z - cell_origin.z) * c;
            } else {
                tz_min = (bb_max.z - cell_origin.z) * c;
                tz_max = (bb_min.z - cell_origin.z) * c;
            }

            // Find largest entering t value
            float t0 = tx_min;
            if (ty_min > t0)
                t0 = ty_min;
            if (tz_min > t0)
                t0 = tz_min;

            // Find smallest exiting t value
            float t1 = tx_max;
            if (ty_max < t1)
                t1 = ty_max;
            if (tz_max < t1)
                t1 = tz_max;

            if (t0 < t1 && t1 > 1e-6f) {
                bb_hit[r] = 1;
            }
        }

        // determine how many rays hit the bounding box
        size_t total_scan_rays = Ntheta * Nphi;
        N = 0;
        float hit_out = 0;
        for (int i = 0; i < total_scan_rays; i++) {
            if (bb_hit[i] == 1) {
                N++;
                helios::SphericalCoord dir = cart2sphere(raydir[i]);
                hit_out += sin(dir.zenith);
            }
        }

        // Store base beam directions that hit bounding box
        std::vector<helios::vec3> base_directions;
        base_directions.reserve(N);
        std::vector<helios::int2> pulse_scangrid_ij(N);
        // Per-beam emission origin (equals scan_origin for static scans; the per-pulse platform origin for moving scans).
        std::vector<helios::vec3> pulse_origin(N);

        int count = 0;
        for (int i = 0; i < Ntheta * Nphi; i++) {
            if (bb_hit[i] == 1) {

                base_directions.push_back(raydir.at(i));

                int jj = floor(i / Ntheta);
                int ii = i - jj * Ntheta;
                pulse_scangrid_ij[count] = helios::make_int2(ii, jj);
                pulse_origin[count] = raygrid_origin.at(i);

                count++;
            }
            // NOTE: Miss recording for rays that don't hit BB removed - those rays can't interact with grid.
            // Misses for traced rays are recorded via line 3733 when record_misses=true.
        }

        // Handle case where no rays hit bounding box
        if (N == 0) {
            // If record_misses=true, record all rays as misses
            if (record_misses) {
                // Place miss points just beyond any real target at the ray-tracer no-hit distance.
                // The exact placement distance is not significant: the leaf-area inversion classifies
                // a miss geometrically as a beam transmitted through the voxel ("after voxel"),
                // independent of how far out the point sits. The canonical miss marker is the
                // is_miss flag set below.
                float miss_dist = LIDAR_RAYTRACE_MISS_T;
                for (int i = 0; i < Ntheta * Nphi; i++) {
                    std::map<std::string, double> data;
                    data["target_index"] = 0;
                    data["target_count"] = 1;
                    data["deviation"] = 0.0;
                    // Real per-pulse acquisition time. The grid index i is the pulse ordinal (i = Ntheta*j + row), so this
                    // matches the encoding used on the hit path and unifies miss/hit timestamps at the same scan-grid cell.
                    data["timestamp"] = pulse_t0 + double(i) * pulse_period;
                    data["intensity"] = 1.0; // Full miss
                    data["distance"] = miss_dist;
                    data["nRaysHit"] = Npulse; // All rays in pulse missed together
                    data["is_miss"] = 1.0; // canonical miss flag
                    if (spinning_multibeam) {
                        data["channel"] = double(i % Ntheta); // laser channel index (scan-table row) that fired this beam
                    }
                    if (std::find(column_format.begin(), column_format.end(), "reflectance") != column_format.end()) {
                        data["reflectance"] = 0.0; // 10*log10(1.0): full-miss sentinel intensity maps to 0 dB
                    }

                    helios::vec3 dir = raydir.at(i);
                    helios::vec3 cell_origin = raygrid_origin.at(i);
                    if (is_moving) {
                        data["pulse_id"] = double(i);
                        data["origin_x"] = cell_origin.x;
                        data["origin_y"] = cell_origin.y;
                        data["origin_z"] = cell_origin.z;
                    }
                    helios::vec3 p = cell_origin + dir * miss_dist;
                    addHitPoint(s, p, helios::cart2sphere(dir), helios::RGB::red, data);
                }
            } else {
                scan_warnings.addWarning("synthetic_rays_no_hit", "Synthetic rays did not hit any primitives.");
            }
            continue; // Move to next scan
        }

        // Allocate host memory for results
        float *hit_t = (float *) malloc(N * Npulse * sizeof(float)); // allocate host memory
        float *hit_fnorm = (float *) malloc(N * Npulse * sizeof(float)); // allocate host memory
        int *hit_ID = (int *) malloc(N * Npulse * sizeof(int)); // allocate host memory

        float exit_diameter = getScanBeamExitDiameter(s);
        float beam_divergence = getScanBeamDivergence(s);
        float range_noise_stddev = getScanRangeNoiseStdDev(s);
        float angle_noise_stddev = getScanAngleNoiseStdDev(s);

        // Apply angular (beam-pointing) jitter to the nominal direction of each beam. This is a per-pulse pointing error of
        // the whole beam, distinct from beam divergence (which spreads sub-rays within a beam) and from range noise (which
        // perturbs the distance along the beam). It contributes the across-beam component of the positional error, which
        // grows with range. The jittered direction becomes the new nominal direction, so the divergence cone, finite
        // aperture, and hit-point reconstruction all rotate together with the beam.
        std::vector<helios::vec3> nominal_directions(N);
        for (size_t beam = 0; beam < N; beam++) {
            helios::vec3 base_dir = base_directions[beam];
            if (angle_noise_stddev > 0) {
                // Build an orthonormal tangent basis {u, v} perpendicular to the beam and apply a small-angle tilt with an
                // independent zero-mean Gaussian offset (stddev = angle_noise_stddev) in each tangent direction.
                helios::vec3 reference = (fabs(base_dir.z) < 0.9f) ? helios::make_vec3(0, 0, 1) : helios::make_vec3(1, 0, 0);
                helios::vec3 u = helios::cross(base_dir, reference);
                u.normalize();
                helios::vec3 v = helios::cross(base_dir, u); // orthonormal since base_dir and u are orthonormal
                float a = context->randn(0.f, angle_noise_stddev);
                float b = context->randn(0.f, angle_noise_stddev);
                helios::vec3 jittered = base_dir + u * a + v * b;
                jittered.normalize();
                nominal_directions[beam] = jittered;
            } else {
                nominal_directions[beam] = base_dir;
            }
        }

        // Generate N*Npulse perturbed ray directions using beam parameters
        helios::vec3 *direction = (helios::vec3 *) malloc(N * Npulse * sizeof(helios::vec3));

        for (size_t beam = 0; beam < N; beam++) {
            helios::vec3 base_dir = nominal_directions[beam];

            for (int p = 0; p < Npulse; p++) {
                if (p == 0 || beam_divergence == 0.0f) {
                    // First ray OR zero divergence: use nominal direction
                    // When beam_divergence=0, all rays should be identical to avoid floating-point precision errors
                    direction[beam * Npulse + p] = base_dir;
                } else {
                    // Subsequent rays with non-zero divergence: apply perturbation
                    // Sample random angular offset within divergence cone using Context's RNG
                    float ru = context->randu();
                    float rv = context->randu();

                    float theta_offset = beam_divergence * sqrt(ru); // radial angular distance
                    float phi_offset = 2.0f * M_PI * rv; // azimuthal angle

                    // Apply small angle perturbation in spherical coordinates
                    helios::SphericalCoord base_spherical = helios::cart2sphere(base_dir);

                    // Perturb in elevation space (constructor takes elevation, not zenith)
                    float new_elevation = base_spherical.elevation + theta_offset * cos(phi_offset);
                    float new_azimuth = base_spherical.azimuth + theta_offset * sin(phi_offset) / fmax(cos(base_spherical.elevation), 1e-6f);

                    // Create perturbed direction
                    helios::vec3 perturbed_dir = helios::sphere2cart(helios::SphericalCoord(1.0f, new_elevation, new_azimuth));
                    perturbed_dir.normalize();

                    direction[beam * Npulse + p] = perturbed_dir;
                }
            }
        }

        // Generate ray origins based on exit diameter
        helios::vec3 *ray_origins = (helios::vec3 *) malloc(N * Npulse * sizeof(helios::vec3));

        if (exit_diameter > 0.0f) {
            // Finite aperture: distribute ray origins across disk perpendicular to beam direction
            float radius = exit_diameter * 0.5f;

            for (size_t beam = 0; beam < N; beam++) {
                helios::vec3 base_dir = nominal_directions[beam];

                // Construct orthonormal basis {u, v, base_dir} for disk perpendicular to beam
                helios::vec3 reference = (fabs(base_dir.z) < 0.9f) ? helios::make_vec3(0, 0, 1) : helios::make_vec3(1, 0, 0);
                helios::vec3 u = helios::cross(base_dir, reference);
                u.normalize();
                helios::vec3 v = helios::cross(base_dir, u); // Already normalized since base_dir and u are orthonormal

                for (int p = 0; p < Npulse; p++) {
                    // Uniform sampling on disk using sqrt transform
                    float ru = context->randu();
                    float rv = context->randu();
                    float r_sample = radius * sqrtf(ru); // sqrt for uniform area distribution
                    float theta = 2.0f * M_PI * rv;
                    float x_disk = r_sample * cosf(theta);
                    float y_disk = r_sample * sinf(theta);

                    // Transform disk point to world space (per-beam origin: scan_origin for static, platform pose for moving)
                    helios::vec3 offset = u * x_disk + v * y_disk;
                    ray_origins[beam * Npulse + p] = pulse_origin[beam] + offset;
                }
            }
        } else {
            // Point source: all rays originate from the beam's emission origin (scan_origin for static scans)
            for (size_t beam = 0; beam < N; beam++) {
                for (int p = 0; p < Npulse; p++) {
                    ray_origins[beam * Npulse + p] = pulse_origin[beam];
                }
            }
        }

        // Initialize collision detection for unified ray-tracing
        initializeCollisionDetection(context);

        // Use unified ray-tracing engine
        performUnifiedRayTracing(context, N, Npulse, ray_origins, direction, hit_t, hit_fnorm, hit_ID);

        size_t Nhits = 0;
        size_t beams_with_zero_hits = 0;
        size_t beams_with_one_hit = 0;
        size_t beams_with_multi_hits = 0;

        // looping over beams
        for (size_t r = 0; r < N; r++) {

            std::vector<std::vector<float>> t_pulse;
            std::vector<std::vector<float>> t_hit;

            // looping over rays in each beam
            for (size_t p = 0; p < Npulse; p++) {

                float t = hit_t[r * Npulse + p]; // distance to hit (misses t=1001.0)
                float i = hit_fnorm[r * Npulse + p]; // dot product between beam direction and primitive normal
                float ID = float(hit_ID[r * Npulse + p]); // ID of intersected primitive

                if (record_misses || (!record_misses && t < miss_distance)) {
                    std::vector<float> v{t, i, ID};
                    t_pulse.push_back(v);
                }
            }

            if (t_pulse.size() == 0) {
                // No hits for this beam
                beams_with_zero_hits++;
            } else if (t_pulse.size() == 1) { // this is single-return data, or we only had one hit for this pulse
                beams_with_one_hit++;

                float distance = t_pulse.front().at(0);
                float intensity = t_pulse.front().at(1);
                if (distance >= 0.98f * miss_distance) {
                    intensity = 0.0;
                }
                float nPulseHit = 1;
                float IDmap = t_pulse.front().at(2);

                std::vector<float> v{distance, intensity, nPulseHit, IDmap};
                t_hit.push_back(v);

            } else if (t_pulse.size() > 1) { // more than one hit for this pulse
                beams_with_multi_hits++;

                std::sort(t_pulse.begin(), t_pulse.end(), [](const std::vector<float> &a, const std::vector<float> &b) { return a[0] < b[0]; });

                float t0 = t_pulse.at(0).at(0);
                float d = t_pulse.at(0).at(0);
                float f = t_pulse.at(0).at(1);
                int count = 1;

                // loop over rays in each beam and group into hit points
                for (size_t hit = 1; hit <= t_pulse.size(); hit++) {

                    // if the end has been reached, output the last hitpoint
                    if (hit == t_pulse.size()) {

                        float distance = d / float(count);
                        float intensity = f / float(Npulse);
                        if (distance >= 0.98f * miss_distance) {
                            intensity = 0;
                        }
                        float nPulseHit = float(count);
                        float IDmap = t_pulse.at(hit - 1).at(2);

                        // test for full misses and set those to have intensity = 1
                        if (nPulseHit == Npulse && distance >= 0.98f * miss_distance) {
                            std::vector<float> v{distance, 1.0, nPulseHit, IDmap}; // included the ray count here
                            // Note: the last index of t_pulse (.at(2)) is the object identifier. We don't want object identifiers to be averaged, so we'll assign the hit identifier based on the last ray in the group
                            t_hit.push_back(v);
                        } else {
                            std::vector<float> v{distance, intensity, nPulseHit, IDmap}; // included the ray count here
                            // Note: the last index of t_pulse (.at(2)) is the object identifier. We don't want object identifiers to be averaged, so we'll assign the hit identifier based on the last ray in the group
                            t_hit.push_back(v);
                        }

                        // else if the current ray is more than the pulse threshold distance from t0,  it is part of the next hitpoint so output the previous hitpoint and reset
                    } else if (t_pulse.at(hit).at(0) - t0 > pulse_distance_threshold) {

                        float distance = d / float(count);
                        float intensity = f / float(Npulse);
                        if (distance >= 0.98f * miss_distance) {
                            intensity = 0;
                        }
                        float nPulseHit = float(count);
                        float IDmap = t_pulse.at(hit - 1).at(2);

                        // test for full misses and set those to have intensity = 1
                        if (nPulseHit == Npulse && distance >= 0.98f * miss_distance) {
                            std::vector<float> v{distance, 1.0, nPulseHit, IDmap}; // included the ray count here
                            // Note: the last index of t_pulse (.at(2)) is the object identifier. We don't want object identifiers to be averaged, so we'll assign the hit identifier based on the last ray in the group
                            t_hit.push_back(v);
                        } else {
                            std::vector<float> v{distance, intensity, nPulseHit, IDmap}; // included the ray count here
                            // Note: the last index of t_pulse (.at(2)) is the object identifier. We don't want object identifiers to be averaged, so we'll assign the hit identifier based on the last ray in the group
                            t_hit.push_back(v);
                        }

                        count = 1;
                        d = t_pulse.at(hit).at(0);
                        t0 = t_pulse.at(hit).at(0);
                        f = t_pulse.at(hit).at(1);

                        // or else the current ray is less than pulse threshold and is part of current hitpoint; add it to the current hit point and continue on
                    } else {

                        count++;
                        d += t_pulse.at(hit).at(0);
                        f += t_pulse.at(hit).at(1);
                    }
                }
            }

            float average = 0;
            for (size_t hit = 0; hit < t_hit.size(); hit++) {
                average += t_hit.at(hit).at(0) / float(t_hit.size());
            }

            // Count non-miss returns for proper target_index assignment
            int non_miss_count = 0;
            for (size_t hit = 0; hit < t_hit.size(); hit++) {
                if (t_hit.at(hit).at(0) < 0.98f * miss_distance) {
                    non_miss_count++;
                }
            }

            int real_hit_index = 0;
            for (size_t hit = 0; hit < t_hit.size(); hit++) {

                std::map<std::string, double> data;

                // Check if this is a miss point
                bool is_miss = (t_hit.at(hit).at(0) >= 0.98f * miss_distance);

                // Apply Gaussian range (along-beam) measurement noise to real returns. LiDAR positional error is anisotropic and
                // dominated by an error in the measured range, so the noise is added to the scalar distance and the hit point is
                // reconstructed along the nominal beam direction below, rather than perturbing (x,y,z) isotropically. Misses keep
                // their ray-tracer no-hit distance and are not noise-displaced. Each return draws independently (per-return noise).
                float measured_distance = t_hit.at(hit).at(0);
                if (!is_miss && range_noise_stddev > 0) {
                    measured_distance += context->randn(0.f, range_noise_stddev);
                }

                // Assign target_index: misses get 99, real hits get sequential index (0, 1, 2...)
                if (is_miss) {
                    data["target_index"] = 99; // Special value to exclude from triangulation
                } else {
                    data["target_index"] = real_hit_index;
                    real_hit_index++;
                }

                data["is_miss"] = is_miss ? 1.0 : 0.0; // canonical miss flag
                data["target_count"] = t_hit.size();
                data["deviation"] = fabs(measured_distance - average);
                // Real per-pulse acquisition time. The pulse ordinal is its position in the firing sequence
                // (ordinal = Ntheta*j + i); scaling by pulse_period and offsetting by t0 turns it into seconds. For static
                // scans pulse_period=1 and t0=0, so this equals the historical grid ordinal. All returns of one pulse (this
                // loop over `hit` for a fixed beam r) share the identical time, as required by groupHitsByTimestamp.
                const size_t pulse_ordinal = size_t(pulse_scangrid_ij.at(r).y) * Ntheta + size_t(pulse_scangrid_ij.at(r).x);
                data["timestamp"] = pulse_t0 + double(pulse_ordinal) * pulse_period;
                // Record range-normalized intensity: the range-independent return amplitude rho*cos(theta) with the
                // 1/R^2 range loss of the LiDAR range equation normalized out (see applyRangeIntensityCorrection()).
                data["intensity"] = applyRangeIntensityCorrection(t_hit.at(hit).at(1), measured_distance);
                data["distance"] = measured_distance;
                data["nRaysHit"] = t_hit.at(hit).at(2);
                if (spinning_multibeam) {
                    data["channel"] = double(pulse_scangrid_ij.at(r).x); // laser channel index (scan-table row) that fired this beam
                }

                float UUID = t_hit.at(hit).at(3);

                // Use base direction for this beam (first ray: r*Npulse+0)
                helios::vec3 dir = direction[r * Npulse];
                // Reconstruct the hit point along the beam from its own emission origin (the per-pulse platform origin for
                // moving scans; scan_origin for static scans). For moving scans, record the per-pulse origin and firing index.
                const helios::vec3 beam_origin = pulse_origin[r];
                helios::vec3 p = beam_origin + dir * measured_distance;
                if (is_moving) {
                    data["pulse_id"] = double(pulse_ordinal);
                    data["origin_x"] = beam_origin.x;
                    data["origin_y"] = beam_origin.y;
                    data["origin_z"] = beam_origin.z;
                }

                helios::RGBcolor color = helios::RGB::red;

                if (UUID >= 0 && context->doesPrimitiveExist(uint(UUID))) {

                    color = sample_hit_color(uint(UUID), p);

                    // Sample arbitrary primitive-data fields named in the scan's column format onto
                    // this hit. The column format is the source of truth: add a (non-standard) label
                    // to the scan's column format and the scanner copies that primitive data here.
                    for (const std::string &label: column_format) {
                        if (isStandardColumnToken(label)) {
                            continue;
                        }
                        if (!context->doesPrimitiveDataExist(uint(UUID), label.c_str())) {
                            continue;
                        }

                        double value;
                        switch (context->getPrimitiveDataType(label.c_str())) {
                            case helios::HELIOS_TYPE_FLOAT: {
                                float v;
                                context->getPrimitiveData(uint(UUID), label.c_str(), v);
                                value = double(v);
                                break;
                            }
                            case helios::HELIOS_TYPE_DOUBLE: {
                                double v;
                                context->getPrimitiveData(uint(UUID), label.c_str(), v);
                                value = v;
                                break;
                            }
                            case helios::HELIOS_TYPE_INT: {
                                int v;
                                context->getPrimitiveData(uint(UUID), label.c_str(), v);
                                value = double(v);
                                break;
                            }
                            case helios::HELIOS_TYPE_UINT: {
                                uint v;
                                context->getPrimitiveData(uint(UUID), label.c_str(), v);
                                value = double(v);
                                break;
                            }
                            default:
                                continue; // non-scalar primitive-data types are not transferable to hits
                        }

                        if (label == "reflectivity_lidar") {
                            // Preserve historical semantics: reflectivity modulates intensity.
                            data.at("intensity") *= value;
                        } else if (label == "reflectance") {
                            // "reflectance" is a computed synthetic output (see below), not a primitive-data field
                            // to copy. Skip it here so it is not overwritten by (or silently zeroed from) primitive
                            // data of the same name.
                            continue;
                        } else {
                            data[label] = value;
                        }
                    }
                }

                // If the scan requests "reflectance", record reflectance in decibels. Following the convention used
                // by terrestrial laser scanners (e.g. RIEGL), reflectance is reported relative to a perfect diffuse
                // (Lambertian) reflector viewed at normal incidence, which corresponds to intensity = 1 (0 dB):
                //
                //     reflectance_dB = 10 * log10( |intensity| )
                //
                // where intensity is the range-normalized return amplitude rho*cos(theta) (see
                // applyRangeIntensityCorrection()). Returns with non-positive intensity (misses, fully grazing or
                // back-facing hits) have no detectable signal and are floored at REFLECTANCE_FLOOR_DB rather than
                // -infinity, mirroring a scanner's minimum detectable reflectance.
                if (std::find(column_format.begin(), column_format.end(), "reflectance") != column_format.end()) {
                    constexpr double REFLECTANCE_FLOOR_DB = -999.0;
                    double abs_intensity = fabs(data.at("intensity"));
                    data["reflectance"] = (abs_intensity > 0.0) ? 10.0 * log10(abs_intensity) : REFLECTANCE_FLOOR_DB;
                }

                addHitPoint(s, p, helios::cart2sphere(dir), color, data);

                Nhits++;
            }
        }

        // No device memory to free
        free(ray_origins);
        free(direction);
        free(hit_t);
        free(hit_fnorm);
        free(hit_ID);

        if (printmessages) {
            std::cout << "Created synthetic scan #" << s << " with " << Nhits << " hit points." << std::endl;
        }
    }

    scan_warnings.report(std::cerr);

    // No device memory to free
    free(patch_vertex);
    free(patch_textureID);
    free(patch_uv);
    free(tri_vertex);
    free(tri_textureID);
    free(tri_uv);
    free(maskdata);
    free(masksize);
}
