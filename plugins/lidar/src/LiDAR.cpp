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

#include <random> // per-beam range-noise RNG in the parallelized syntheticScan post-processing
#include <set>

#ifdef _OPENMP
#include <omp.h>
#endif

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

    //! Resolve a scalar (FLOAT/DOUBLE/INT/UINT) data field named `label` for the hit primitive `UUID`, as a double.
    /** Resolution precedence: the hit primitive's own primitive data is checked first; if the label is not present
        there, the data of the primitive's parent compound object is checked. This lets a synthetic-scan column label
        be sampled from either primitive data or object data (e.g. a per-object "branch_order"), with the more specific
        per-primitive value winning when a label exists on both. Returns true and sets `value` on success; returns false
        if neither source carries the label or the stored data is a non-scalar type (vector/string), which cannot be
        transferred to a scalar hit field. */
    bool resolveScalarHitData(helios::Context *context, uint UUID, const std::string &label, double &value) {
        if (context->doesPrimitiveDataExist(UUID, label.c_str())) {
            switch (context->getPrimitiveDataType(label.c_str())) {
                case helios::HELIOS_TYPE_FLOAT: {
                    float v;
                    context->getPrimitiveData(UUID, label.c_str(), v);
                    value = double(v);
                    return true;
                }
                case helios::HELIOS_TYPE_DOUBLE: {
                    context->getPrimitiveData(UUID, label.c_str(), value);
                    return true;
                }
                case helios::HELIOS_TYPE_INT: {
                    int v;
                    context->getPrimitiveData(UUID, label.c_str(), v);
                    value = double(v);
                    return true;
                }
                case helios::HELIOS_TYPE_UINT: {
                    uint v;
                    context->getPrimitiveData(UUID, label.c_str(), v);
                    value = double(v);
                    return true;
                }
                default:
                    return false; // non-scalar primitive-data types are not transferable to hits
            }
        }

        // Fall back to the hit primitive's parent object data. Primitives with no parent object have object id 0;
        // guard with doesObjectExist() because doesObjectDataExist() only checks object existence under HELIOS_DEBUG.
        uint objID = context->getPrimitiveParentObjectID(UUID);
        if (context->doesObjectExist(objID) && context->doesObjectDataExist(objID, label.c_str())) {
            switch (context->getObjectDataType(label.c_str())) {
                case helios::HELIOS_TYPE_FLOAT: {
                    float v;
                    context->getObjectData(objID, label.c_str(), v);
                    value = double(v);
                    return true;
                }
                case helios::HELIOS_TYPE_DOUBLE: {
                    context->getObjectData(objID, label.c_str(), value);
                    return true;
                }
                case helios::HELIOS_TYPE_INT: {
                    int v;
                    context->getObjectData(objID, label.c_str(), v);
                    value = double(v);
                    return true;
                }
                case helios::HELIOS_TYPE_UINT: {
                    uint v;
                    context->getObjectData(objID, label.c_str(), v);
                    value = double(v);
                    return true;
                }
                default:
                    return false; // non-scalar object-data types are not transferable to hits
            }
        }

        return false;
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

    //! Test a caller-owned cancellation flag (the one registered via LiDARcloud::setCancelFlag).
    //! Returns true when the pointed-to int is non-zero, i.e. the caller has requested an abort.
    //! Cheap enough to call from the top of a loop; in hot inner loops it is gated behind a coarse
    //! iteration-count guard so the read happens only every ~128k iterations.
    inline bool triangulationCancelled(volatile int *cancel_flag) {
        return cancel_flag != nullptr && *cancel_flag != 0;
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

namespace {

    //! Minimal double-precision 3-vector for the Risley-prism refraction math (kept in double precision so the rotation phase rotor_rate*t stays accurate over a long acquisition).
    struct dvec3 {
        double x, y, z;
    };
    inline dvec3 operator-(const dvec3 &v) {
        return {-v.x, -v.y, -v.z};
    }
    inline double dot(const dvec3 &a, const dvec3 &b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    inline dvec3 normalize(const dvec3 &v) {
        double m = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        return {v.x / m, v.y / m, v.z / m};
    }

    //! Refract a unit ray at a planar interface using the vector form of Snell's law.
    /**
     * Standard non-paraxial refraction: given a unit incident direction \p incident, the unit surface normal \p normal pointing
     * back toward the medium the ray is leaving, and the indices on the two sides, the transmitted unit direction is
     *   t = r*d + (r*c - sqrt(1 - r^2*(1 - c^2)))*n,   r = n_from/n_to,   c = -(n . d),
     * which is the textbook vector form of Snell's law (e.g. Glassner, "An Introduction to Ray Tracing", 1989). A negative
     * radicand means the critical angle is exceeded; the function returns false (total internal reflection - the ray does not
     * pass) in that case.
     * \param[in] incident  Unit incident ray direction.
     * \param[in] normal  Unit interface normal, oriented toward the incoming medium.
     * \param[in] n_from  Refractive index of the medium the ray is leaving.
     * \param[in] n_to  Refractive index of the medium the ray is entering.
     * \param[out] refracted  Unit transmitted ray direction (unchanged on total internal reflection).
     * \return true if the ray is transmitted, false on total internal reflection.
     */
    bool refractRay(const dvec3 &incident, const dvec3 &normal, double n_from, double n_to, dvec3 &refracted) {
        const double r = n_from / n_to;
        const double c = -dot(normal, incident); // cosine of the incidence angle
        const double radicand = 1.0 - r * r * (1.0 - c * c);
        if (radicand < 0.0) {
            return false; // total internal reflection
        }
        const double k = r * c - std::sqrt(radicand);
        refracted = normalize({r * incident.x + k * normal.x, r * incident.y + k * normal.y, r * incident.z + k * normal.z});
        return true;
    }

    //! Body-frame unit beam direction of a single pulse of a Risley-prism (Livox-style rosette) scan.
    /**
     * Models a stack of rotating wedge prisms by non-paraxial ray tracing: the transmitted beam is propagated through each
     * prism by refracting it at the prism's two faces in turn with the vector form of Snell's law (see \ref refractRay). Each
     * prism is a glass wedge with one face perpendicular to the optical axis and one face tilted by the wedge angle; the prism
     * spins about the optical axis, so at pulse time t its tilted face has rolled to the clocking angle
     * phi = phase + rotor_rate*t. With two (or more) prisms rotating at different, generally incommensurate, rates the exit
     * beam sweeps a non-repetitive rosette inside a circular field of view, denser toward the center - the characteristic
     * Livox pattern. This ray-tracing-through-wedges treatment is the standard non-paraxial Risley-prism model (e.g. Li,
     * "Third-order theory of the Risley-prism-based beam steering system", Appl. Opt. 2011).
     *
     * The optical axis (the un-deflected incident beam) is +y, which is exactly the Helios LiDAR body-forward axis: the
     * azimuth-zero, horizontal direction produced by sphere2cart at phi=0, zenith=pi/2, and the same forward axis the
     * moving-platform path rotates by the boresight and trajectory quaternions. The lateral axis is +x and the up axis is +z,
     * so the tilted-face normal lies on a cone of half-angle (pi/2 - wedge) about +y and rolls in the x-z plane with phi. The
     * pulse time is t = pulse_index * pulse_period, scan-relative: the prism clocking is referenced to the start of the scan,
     * independent of the trajectory t0.
     *
     * The maximum deflection (and hence the circular field of view) is an emergent property of the wedge angles and refractive
     * indices, not a specified parameter: a single ~18.7 deg wedge at n=1.51 deflects the beam ~9.8 deg, so a counter-rotating
     * pair fills a ~38 deg circular FoV.
     *
     * On total internal reflection at any face the beam does not exit; the un-deflected optical axis (+y) is returned (this
     * does not occur for physical Livox-class wedge angles and indices).
     */
    helios::vec3 risleyBodyDirection(const ScanMetadata &scan, size_t pulse_index) {

        const double t = double(pulse_index) * scan.pulse_period;
        const double n_air = scan.risley_refractive_index_air;

        // Incident beam along the optical axis (+y, the Helios body-forward axis). The flat face is perpendicular to this axis.
        dvec3 beam = {0.0, 1.0, 0.0};
        const dvec3 flat_normal = {0.0, -1.0, 0.0}; // entry face normal, oriented back toward the incoming (+y) beam

        for (const RisleyPrism &prism : scan.risley_prisms) {
            const double phi = prism.phase + prism.rotor_rate * t; // clocking angle of this prism's wedge at pulse time t
            const double sinW = std::sin(prism.wedge_angle);
            const double cosW = std::cos(prism.wedge_angle);

            // Tilted (exit) face normal: the face is inclined by the wedge angle from the plane perpendicular to the optical
            // axis, so its outward normal sits at angle wedge off the +y axis and rolls around it with the clocking angle phi.
            // Built directly from the rotation geometry (no external code): axial component cos(wedge) along +y, transverse
            // component sin(wedge) in the x-z plane at angle phi.
            const dvec3 tilted_normal = normalize({sinW * std::cos(phi), cosW, sinW * std::sin(phi)});

            // Refract through the flat entry face (air -> glass), then the tilted exit face (glass -> air). The normal handed to
            // refractRay points back toward the medium the ray is leaving, so the exit-face normal is negated.
            dvec3 in_glass;
            if (!refractRay(beam, flat_normal, n_air, prism.refractive_index, in_glass)) {
                return helios::make_vec3(0.f, 1.f, 0.f); // total internal reflection: return the optical axis (+y)
            }
            dvec3 out_glass;
            if (!refractRay(in_glass, -tilted_normal, prism.refractive_index, n_air, out_glass)) {
                return helios::make_vec3(0.f, 1.f, 0.f);
            }
            beam = out_glass;
        }

        beam = normalize(beam);
        return helios::make_vec3(float(beam.x), float(beam.y), float(beam.z));
    }

} // namespace

helios::SphericalCoord ScanMetadata::rc2direction(uint row, uint column) const {

    if (scanPattern == SCAN_PATTERN_RISLEY_PRISM) {
        // A Risley-prism scan is non-separable: each column is one pulse whose body-frame direction comes from the rotating
        // prism optics at that pulse's time, not from a row x column angular grid. The single source of truth is
        // risleyBodyDirection(); rc2direction returns its spherical form so callers that reason in (zenith,azimuth) stay
        // consistent with the ray generator.
        return cart2sphere(risleyBodyDirection(*this, column));
    }

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

    if (scanPattern == SCAN_PATTERN_RISLEY_PRISM) {
        // A Risley-prism scan has no row x column angular grid (it is stored as a single row, one pulse per column), and it is
        // always trajectory-driven so the direction passed here is in world coordinates and cannot be inverted back to a pulse
        // index. Row/column is meaningless for this pattern; return (0,0). Downstream, moving scans identify points by
        // timestamp / pulse_id / origin rather than (row,column), and triangulation / gap-filling reject moving scans outright.
        return helios::make_int2(0, 0);
    }

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
    // Forward any registered cancellation flag so an in-flight syntheticScan can
    // be aborted mid-trace (the ray loop in castRaysSoA polls it). Re-applied
    // here because the CD object is created lazily, after setCancelFlag().
    collision_detection->setCancelFlag(cancel_flag);
}

void LiDARcloud::setCancelFlag(volatile int *flag) {
    cancel_flag = flag;
    if (collision_detection != nullptr) {
        collision_detection->setCancelFlag(flag);
    }
}

void LiDARcloud::setSyntheticScanProgressPointer(volatile int *ptr) {
    synthetic_scan_progress = ptr;
}

void LiDARcloud::prepareUnifiedRayTracing(helios::Context *context) {
    initializeCollisionDetection(context);
    // Disable automatic BVH rebuilds for the whole batch (geometry is static during a scan) and build the BVH once. When
    // the per-scan beam fan-out is traced in chunks, this prevents a full O(P log P) BVH rebuild on every chunk.
    collision_detection->disableAutomaticBVHRebuilds();
    collision_detection->buildBVH();
}

void LiDARcloud::finishUnifiedRayTracing() {
    collision_detection->enableAutomaticBVHRebuilds();
}

void LiDARcloud::castRaysUnified(size_t total_rays, helios::vec3 *ray_origins, helios::vec3 *direction, float *hit_t, float *hit_fnorm, int *hit_ID, size_t packet_size) {
    const float miss_distance = LIDAR_RAYTRACE_MISS_T;
    constexpr uint MISS_UUID = 0xFFFFFFFFu; // sentinel written by castRaysSoA for a miss

    if (total_rays == 0) {
        return;
    }

    // Low-memory SoA cast: results are written directly into per-ray scratch arrays rather than a full-length
    // RayQuery input vector plus a HitResult output vector (which together cost ~96 bytes/ray of transient storage).
    // The primitive UUID doubles as the hit/miss flag (MISS_UUID == miss); distance/normal are reused below.
    std::vector<uint> uuid(total_rays);
    std::vector<helios::vec3> normal(total_rays);
    // hit_t is reused as the SoA distance output array (float, length total_rays) to avoid a separate allocation.
    // When the rays are grouped into coherent pulses (packet_size > 1) the packet traversal amortizes node/primitive
    // fetches across each pulse's sub-rays; results are identical to the per-ray path.
    if (packet_size > 1) {
        collision_detection->castRaysSoA_packets(ray_origins, direction, total_rays, packet_size, miss_distance, hit_t, normal.data(), uuid.data());
    } else {
        collision_detection->castRaysSoA(ray_origins, direction, total_rays, miss_distance, hit_t, normal.data(), uuid.data());
    }

    // Convert the SoA results to the LiDAR per-ray format expected by the waveform reduction.
    for (size_t i = 0; i < total_rays; i++) {
        if (uuid[i] != MISS_UUID) {
            hit_ID[i] = static_cast<int>(uuid[i]);
            // hit_t[i] already holds the hit distance (written in place by castRaysSoA).
            const helios::vec3 &ray_dir = direction[i];
            hit_fnorm[i] = ray_dir.x * normal[i].x + ray_dir.y * normal[i].y + ray_dir.z * normal[i].z;
        } else {
            hit_t[i] = miss_distance;
            hit_ID[i] = -1;
            hit_fnorm[i] = 1e6;
        }
    }
}

void LiDARcloud::performUnifiedRayTracing(helios::Context *context, size_t N, int Npulse, helios::vec3 *ray_origins, helios::vec3 *direction, float *hit_t, float *hit_fnorm, int *hit_ID) {
    // Standalone single-batch entry point (used directly by tests): prepare the BVH, cast all N*Npulse rays, then restore
    // automatic rebuilds. The chunked syntheticScan path instead calls prepareUnifiedRayTracing() once and
    // castRaysUnified() per chunk to build the BVH only once across the whole scan.
    prepareUnifiedRayTracing(context);
    castRaysUnified(N * size_t(Npulse), ray_origins, direction, hit_t, hit_fnorm, hit_ID, size_t(Npulse));
    finishUnifiedRayTracing();
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

void LiDARcloud::setProgressCallback(std::function<void(float, const std::string &)> callback) {
    progress_callback = std::move(callback);
}

void LiDARcloud::setSyntheticScanMemoryBudget(size_t bytes) {
    if (bytes == 0) {
        helios_runtime_error("ERROR (LiDARcloud::setSyntheticScanMemoryBudget): the memory budget must be greater than zero.");
    }
    synthetic_scan_memory_budget_bytes = bytes;
}

size_t LiDARcloud::getSyntheticScanMemoryBudget() const {
    return synthetic_scan_memory_budget_bytes;
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
    // Only the static raster path has a physically-bounded azimuth sweep where phiMax >> 2pi signals a degrees/radians
    // mistake. A spinning multibeam scan legitimately encodes phiMax = n_revolutions*2pi (many multiples of 2pi), and a
    // moving raster scan may sweep an arbitrary azimuth, so the warning is gated to SCAN_MODE_STATIC_RASTER.
    if (newscan.scanMode == SCAN_MODE_STATIC_RASTER && newscan.phiMax > 4.f * M_PI + epsilon) {
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

uint LiDARcloud::addScanSpinning(const std::vector<float> &beamElevationAngles, float azimuthStep_rad, float pulse_rate_hz, const std::vector<double> &traj_t, const std::vector<vec3> &traj_pos, const std::vector<vec4> &traj_quat,
                                 const vec3 &lever_arm, const vec3 &boresight_rpy, float exitDiameter, float beamDivergence, float rangeNoiseStdDev, float angleNoiseStdDev, const std::vector<std::string> &columnFormat, double t0) {

    // Physical-parameter setup for a continuously-spinning multibeam sensor. The caller supplies the instrument's
    // channel elevations, azimuth resolution, and PRF; the internal Ntheta x Nphi grid, rotation rate, and revolution
    // count are derived here so the caller never hand-flattens the instrument into a raster grid. The heavy lifting
    // (per-pulse time/pose, validation) is delegated to addScanMoving once the grid is built.

    if (beamElevationAngles.empty()) {
        helios_runtime_error("ERROR (LiDARcloud::addScanSpinning): beamElevationAngles is empty. A spinning multibeam sensor requires at least one channel.");
    }
    if (azimuthStep_rad <= 0.f) {
        helios_runtime_error("ERROR (LiDARcloud::addScanSpinning): azimuthStep_rad must be greater than 0, but " + std::to_string(azimuthStep_rad) + " was provided.");
    }
    if (pulse_rate_hz <= 0.f) {
        helios_runtime_error("ERROR (LiDARcloud::addScanSpinning): pulse_rate_hz must be greater than 0, but " + std::to_string(pulse_rate_hz) + " was provided.");
    }
    const size_t M = traj_t.size();
    if (M == 0) {
        helios_runtime_error("ERROR (LiDARcloud::addScanSpinning): the trajectory is empty. At least one pose sample is required (a stationary capture is expressed as two coincident poses separated by the acquisition duration).");
    }
    // Validate the trajectory array lengths here, before traj_pos.front() is used to seed the ScanMetadata origin below.
    // addScanMoving() also validates these, but it runs only after that dereference, so an empty/short traj_pos would be
    // undefined behavior rather than a clear error. Fail fast with an actionable message instead.
    if (traj_pos.size() != M || traj_quat.size() != M) {
        helios_runtime_error("ERROR (LiDARcloud::addScanSpinning): trajectory arrays have inconsistent lengths (traj_t=" + std::to_string(M) + ", traj_pos=" + std::to_string(traj_pos.size()) + ", traj_quat=" + std::to_string(traj_quat.size()) +
                             "). All trajectory arrays must have the same number of samples.");
    }

    // Convert per-channel elevation (above horizon) to the zenith convention used internally (0 = up, pi/2 = horizontal).
    std::vector<float> beamZenithAngles;
    beamZenithAngles.reserve(beamElevationAngles.size());
    for (float elevation: beamElevationAngles) {
        beamZenithAngles.push_back(0.5f * float(M_PI) - elevation);
    }

    const uint channels = uint(beamZenithAngles.size());
    // Round to an integer number of azimuth steps per revolution and use the exact dphi = 2pi/steps_per_rev so the
    // sampling closes the circle perfectly regardless of the requested step.
    const uint steps_per_rev = uint(std::lround(2.0 * M_PI / double(azimuthStep_rad)));
    if (steps_per_rev == 0) {
        helios_runtime_error("ERROR (LiDARcloud::addScanSpinning): azimuthStep_rad=" + std::to_string(azimuthStep_rad) + " is larger than 2pi, which yields zero azimuth steps per revolution. Use a finer azimuth resolution.");
    }

    const double duration = traj_t.back() - traj_t.front();
    if (duration <= 0.0) {
        helios_runtime_error("ERROR (LiDARcloud::addScanSpinning): the trajectory duration (traj_t.back() - traj_t.front() = " + std::to_string(duration) + ") must be greater than 0.");
    }

    const double rotation_rate = double(pulse_rate_hz) / (double(channels) * double(steps_per_rev)); // revolutions per second
    const double n_revolutions = rotation_rate * duration;
    const uint Nphi = uint(std::lround(double(steps_per_rev) * n_revolutions));
    if (Nphi == 0) {
        helios_runtime_error("ERROR (LiDARcloud::addScanSpinning): the derived azimuth-step count is zero (PRF=" + std::to_string(pulse_rate_hz) + " Hz over a " + std::to_string(duration) +
                             " s trajectory yields less than one azimuth step). Increase the PRF, the trajectory duration, or the azimuth resolution.");
    }

    // Build a spinning-multibeam ScanMetadata. phiMax encodes the full multi-revolution sweep (n_revolutions*2pi); the
    // ray generator and rc2direction/direction2rc use the periodic convention (dphi = phiMax/Nphi) so this closes correctly.
    ScanMetadata scan(traj_pos.front(), beamZenithAngles, Nphi, 0.f, float(n_revolutions * 2.0 * M_PI), exitDiameter, beamDivergence, rangeNoiseStdDev, angleNoiseStdDev, columnFormat);
    scan.scanMode = SCAN_MODE_SPINNING;
    scan.steps_per_rev = steps_per_rev;
    scan.rotation_rate = rotation_rate;
    scan.n_revolutions = n_revolutions;

    return addScanMoving(scan, traj_t, traj_pos, traj_quat, lever_arm, boresight_rpy, pulse_rate_hz, t0);
}

uint LiDARcloud::addScanSpinning(const std::vector<float> &beamElevationAngles, float azimuthStep_rad, float pulse_rate_hz, const std::vector<double> &traj_t, const std::vector<vec3> &traj_pos, const std::vector<vec3> &traj_rpy,
                                 const vec3 &lever_arm, const vec3 &boresight_rpy, float exitDiameter, float beamDivergence, float rangeNoiseStdDev, float angleNoiseStdDev, const std::vector<std::string> &columnFormat, double t0) {

    // Convert the per-sample roll/pitch/yaw Euler angles to quaternions, then delegate to the quaternion overload.
    if (traj_rpy.size() != traj_t.size()) {
        helios_runtime_error("ERROR (LiDARcloud::addScanSpinning): trajectory arrays have inconsistent lengths (traj_t=" + std::to_string(traj_t.size()) + ", traj_rpy=" + std::to_string(traj_rpy.size()) + "). All trajectory arrays must have the same number of samples.");
    }

    std::vector<vec4> traj_quat;
    traj_quat.reserve(traj_rpy.size());
    for (const vec3 &rpy: traj_rpy) {
        traj_quat.push_back(quat_from_rpy(rpy.x, rpy.y, rpy.z));
    }

    return addScanSpinning(beamElevationAngles, azimuthStep_rad, pulse_rate_hz, traj_t, traj_pos, traj_quat, lever_arm, boresight_rpy, exitDiameter, beamDivergence, rangeNoiseStdDev, angleNoiseStdDev, columnFormat, t0);
}

uint LiDARcloud::addScanMovingRaster(uint Ntheta, float thetaMin, float thetaMax, uint Nphi, float phiMin, float phiMax, float pulse_rate_hz, const std::vector<double> &traj_t, const std::vector<vec3> &traj_pos,
                                     const std::vector<vec4> &traj_quat, const vec3 &lever_arm, const vec3 &boresight_rpy, float exitDiameter, float beamDivergence, float rangeNoiseStdDev, float angleNoiseStdDev,
                                     const std::vector<std::string> &columnFormat, double t0) {

    // Non-spinning sensor on a moving platform: the caller specifies the per-frame angular fan and the trajectory, and
    // addScanMoving derives the per-pulse time sampling. This is a thin convenience over the low-level addScanMoving so
    // the caller does not pre-build a ScanMetadata; it additionally stamps the SCAN_MODE_MOVING_RASTER descriptor.
    ScanMetadata scan(traj_pos.empty() ? make_vec3(0, 0, 0) : traj_pos.front(), Ntheta, thetaMin, thetaMax, Nphi, phiMin, phiMax, exitDiameter, beamDivergence, rangeNoiseStdDev, angleNoiseStdDev, columnFormat);
    scan.scanMode = SCAN_MODE_MOVING_RASTER;

    return addScanMoving(scan, traj_t, traj_pos, traj_quat, lever_arm, boresight_rpy, pulse_rate_hz, t0);
}

uint LiDARcloud::addScanRisley(const std::vector<RisleyPrism> &prisms, double refractive_index_air, float pulse_rate_hz, const std::vector<double> &traj_t, const std::vector<vec3> &traj_pos, const std::vector<vec4> &traj_quat,
                               const vec3 &lever_arm, const vec3 &boresight_rpy, float exitDiameter, float beamDivergence, float rangeNoiseStdDev, float angleNoiseStdDev, const std::vector<std::string> &columnFormat, double t0) {

    // Physical-parameter setup for a rotating-Risley-prism (Livox-style rosette) sensor. The caller supplies the prism stack
    // and PRF; the pulse count Npulses is derived from the PRF and the trajectory duration, and the scan is stored as an
    // Ntheta=1, Nphi=Npulses table (one beam direction per pulse). The heavy lifting (per-pulse time/pose, trajectory
    // validation) is delegated to addScanMoving once the single-row grid is built.

    if (prisms.empty()) {
        helios_runtime_error("ERROR (LiDARcloud::addScanRisley): the prism stack is empty. A Risley-prism scanner requires at least one rotating wedge prism (a Livox-style sensor uses two counter-rotating prisms).");
    }
    if (pulse_rate_hz <= 0.f) {
        helios_runtime_error("ERROR (LiDARcloud::addScanRisley): pulse_rate_hz must be greater than 0, but " + std::to_string(pulse_rate_hz) + " was provided.");
    }
    if (refractive_index_air <= 0.0) {
        helios_runtime_error("ERROR (LiDARcloud::addScanRisley): refractive_index_air must be greater than 0, but " + std::to_string(refractive_index_air) + " was provided.");
    }
    for (size_t k = 0; k < prisms.size(); k++) {
        if (prisms.at(k).refractive_index <= 0.0) {
            helios_runtime_error("ERROR (LiDARcloud::addScanRisley): prism " + std::to_string(k) + " has a non-positive refractive index (" + std::to_string(prisms.at(k).refractive_index) + ").");
        }
    }
    const size_t M = traj_t.size();
    if (M == 0) {
        helios_runtime_error("ERROR (LiDARcloud::addScanRisley): the trajectory is empty. At least one pose sample is required (a stationary capture is expressed as two coincident poses separated by the acquisition duration).");
    }
    // Validate the trajectory array lengths here, before traj_pos.front() is used to seed the ScanMetadata origin below.
    // addScanMoving() also validates these, but it runs only after that dereference, so an empty/short traj_pos would be
    // undefined behavior rather than a clear error. Fail fast with an actionable message instead.
    if (traj_pos.size() != M || traj_quat.size() != M) {
        helios_runtime_error("ERROR (LiDARcloud::addScanRisley): trajectory arrays have inconsistent lengths (traj_t=" + std::to_string(M) + ", traj_pos=" + std::to_string(traj_pos.size()) + ", traj_quat=" + std::to_string(traj_quat.size()) +
                             "). All trajectory arrays must have the same number of samples.");
    }

    const double duration = traj_t.back() - traj_t.front();
    if (duration <= 0.0) {
        helios_runtime_error("ERROR (LiDARcloud::addScanRisley): the trajectory duration (traj_t.back() - traj_t.front() = " + std::to_string(duration) + ") must be greater than 0.");
    }

    // One pulse per firing of the PRF over the acquisition. Stored as a single-row (Ntheta=1) table so the existing pulse
    // ordinal k = Ntheta*column + row collapses to k = column = pulse index, and every per-pulse quantity (time, origin) is
    // derived exactly as for any other moving scan.
    const uint Npulses = uint(std::lround(double(pulse_rate_hz) * duration));
    if (Npulses == 0) {
        helios_runtime_error("ERROR (LiDARcloud::addScanRisley): the derived pulse count is zero (PRF=" + std::to_string(pulse_rate_hz) + " Hz over a " + std::to_string(duration) +
                             " s trajectory yields less than one pulse). Increase the PRF or the trajectory duration.");
    }

    // Build the single-row Risley-prism ScanMetadata, then attach the prism stack. The angular bounds are not used for ray
    // generation (the Risley branch of the ray generator computes each direction from the prism optics), but are populated
    // with the emergent circular field of view so range queries and bounding logic stay meaningful. The maximum half-angle is
    // estimated by sampling the rosette - the field of view is an emergent property of the optics, not an input.
    ScanMetadata scan(traj_pos.front(), 1u, 0.f, float(M_PI), Npulses, 0.f, float(2.0 * M_PI), exitDiameter, beamDivergence, rangeNoiseStdDev, angleNoiseStdDev, columnFormat);
    scan.scanPattern = SCAN_PATTERN_RISLEY_PRISM;
    scan.scanMode = SCAN_MODE_RISLEY_PRISM;
    scan.risley_prisms = prisms;
    scan.risley_refractive_index_air = refractive_index_air;
    scan.pulse_period = 1.0 / double(pulse_rate_hz); // set early so risleyBodyDirection() below uses the real per-pulse time

    // Estimate the emergent circular field of view by sampling the rosette over a bounded number of pulses, then set the
    // zenith bounds to a cone of that half-angle about the optical axis (+y, zenith pi/2). Clamped to [0, pi].
    const size_t Nsample = std::min<size_t>(Npulses, 20000);
    float max_halfangle = 0.f;
    for (size_t k = 0; k < Nsample; k++) {
        helios::vec3 dir = risleyBodyDirection(scan, k);
        float halfangle = std::acos(std::max(-1.f, std::min(1.f, dir.y))); // angle from the +y optical axis
        if (halfangle > max_halfangle) {
            max_halfangle = halfangle;
        }
    }
    scan.thetaMin = std::max(0.f, 0.5f * float(M_PI) - max_halfangle);
    scan.thetaMax = std::min(float(M_PI), 0.5f * float(M_PI) + max_halfangle);

    return addScanMoving(scan, traj_t, traj_pos, traj_quat, lever_arm, boresight_rpy, pulse_rate_hz, t0);
}

uint LiDARcloud::addScanRisley(const std::vector<RisleyPrism> &prisms, double refractive_index_air, float pulse_rate_hz, const std::vector<double> &traj_t, const std::vector<vec3> &traj_pos, const std::vector<vec3> &traj_rpy,
                               const vec3 &lever_arm, const vec3 &boresight_rpy, float exitDiameter, float beamDivergence, float rangeNoiseStdDev, float angleNoiseStdDev, const std::vector<std::string> &columnFormat, double t0) {

    // Convert the per-sample roll/pitch/yaw Euler angles to quaternions, then delegate to the quaternion overload.
    if (traj_rpy.size() != traj_t.size()) {
        helios_runtime_error("ERROR (LiDARcloud::addScanRisley): trajectory arrays have inconsistent lengths (traj_t=" + std::to_string(traj_t.size()) + ", traj_rpy=" + std::to_string(traj_rpy.size()) + "). All trajectory arrays must have the same number of samples.");
    }

    std::vector<vec4> traj_quat;
    traj_quat.reserve(traj_rpy.size());
    for (const vec3 &rpy: traj_rpy) {
        traj_quat.push_back(quat_from_rpy(rpy.x, rpy.y, rpy.z));
    }

    return addScanRisley(prisms, refractive_index_air, pulse_rate_hz, traj_t, traj_pos, traj_quat, lever_arm, boresight_rpy, exitDiameter, beamDivergence, rangeNoiseStdDev, angleNoiseStdDev, columnFormat, t0);
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

size_t LiDARcloud::getOrCreateHitDataColumn(const std::string &label) {
    auto it = hit_data_label_index.find(label);
    if (it != hit_data_label_index.end()) {
        return it->second;
    }
    // New label: create a column back-filled "absent" for all hits that already exist, so the column
    // stays length-aligned with `hits`. In practice the synthetic scan inserts every standard label on
    // the first hit, so this back-fill is empty; only labels introduced mid-cloud (e.g. gapfill codes)
    // pay an O(N) fill, and there are only a handful of those.
    const size_t slot = hit_data_labels.size();
    hit_data_labels.push_back(label);
    hit_data_label_index[label] = slot;
    hit_data_columns.emplace_back(hits.size(), 0.0);
    hit_data_present.emplace_back(hits.size(), char(0));
    return slot;
}

//! Append one hit's scalar data as a new row across all columns. The hit has already been pushed onto
//! `hits`, so its index is hits.size()-1 and every existing column is currently one element short.
void LiDARcloud::appendHitData(const std::map<std::string, double> &data) {
    const size_t i = hits.size() - 1;

    // Extend every existing column by one absent slot for this new hit.
    for (size_t s = 0; s < hit_data_columns.size(); s++) {
        hit_data_columns[s].push_back(0.0);
        hit_data_present[s].push_back(char(0));
    }

    // Fill in the values this hit actually carries (creating new columns as needed; a column created
    // here is back-filled absent for prior hits AND already extended for this hit by emplace_back above
    // via getOrCreateHitDataColumn, which sizes to hits.size()).
    for (const auto &kv: data) {
        const size_t s = getOrCreateHitDataColumn(kv.first);
        hit_data_columns[s][i] = kv.second;
        hit_data_present[s][i] = char(1);
    }
}

void LiDARcloud::addHitPoint(uint scanID, const vec3 &xyz, const SphericalCoord &direction, const RGBcolor &color, const map<string, double> &data) {

    // error checking
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::addHitPoint): Hit point cannot be added to scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }

    const ScanMetadata &scan = scans.at(scanID); // reference, not a per-hit copy (ScanMetadata holds vectors)
    int2 row_column = scan.direction2rc(direction);

    HitPoint hit(scanID, xyz, direction, row_column, color);

    hits.push_back(hit);
    appendHitData(data);
}

void LiDARcloud::addHitPoint(uint scanID, const vec3 &xyz, const int2 &row_column, const RGBcolor &color, const map<string, double> &data) {

    const ScanMetadata &scan = scans.at(scanID); // reference, not a per-hit copy (ScanMetadata holds vectors)
    SphericalCoord direction = scan.rc2direction(row_column.x, row_column.y);

    HitPoint hit(scanID, xyz, direction, row_column, color);

    hits.push_back(hit);
    appendHitData(data);
}

void LiDARcloud::deleteHitPoint(uint index) {

    if (index >= hits.size()) {
        cerr << "WARNING (deleteHitPoint): Hit point #" << index << " cannot be deleted from the scan because there have only been " << hits.size() << " hit points added." << endl;
        return;
    }

    // erase from vector of hits (use swap-and-pop method). The columnar scalar-data store is indexed by
    // hit position, so it must be swapped-and-popped in exact lockstep or the columns desync from the
    // surviving hits' positions/colors and silently corrupt every subsequent read/export.
    const size_t last = hits.size() - 1;
    for (size_t s = 0; s < hit_data_columns.size(); s++) {
        std::swap(hit_data_columns[s][index], hit_data_columns[s][last]);
        std::swap(hit_data_present[s][index], hit_data_present[s][last]);
        hit_data_columns[s].pop_back();
        hit_data_present[s].pop_back();
    }

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

ReturnMode LiDARcloud::getScanReturnMode(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanReturnMode): Cannot get return mode for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).returnMode;
}

void LiDARcloud::setScanReturnMode(uint scanID, ReturnMode returnMode) {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::setScanReturnMode): Cannot set return mode for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    scans.at(scanID).returnMode = returnMode;
}

SingleReturnSelection LiDARcloud::getScanSingleReturnSelection(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanSingleReturnSelection): Cannot get single-return selection for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).singleReturnSelection;
}

void LiDARcloud::setScanSingleReturnSelection(uint scanID, SingleReturnSelection selection) {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::setScanSingleReturnSelection): Cannot set single-return selection for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    scans.at(scanID).singleReturnSelection = selection;
}

int LiDARcloud::getScanMaxReturns(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanMaxReturns): Cannot get maximum returns for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).maxReturns;
}

void LiDARcloud::setScanMaxReturns(uint scanID, int maxReturns) {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::setScanMaxReturns): Cannot set maximum returns for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    if (maxReturns < 1) {
        helios_runtime_error("ERROR (LiDARcloud::setScanMaxReturns): Maximum returns must be at least 1, but " + std::to_string(maxReturns) + " was given.");
    }
    scans.at(scanID).maxReturns = maxReturns;
}

float LiDARcloud::getScanPulseWidth(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanPulseWidth): Cannot get pulse width for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).pulseWidth;
}

void LiDARcloud::setScanPulseWidth(uint scanID, float pulseWidth) {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::setScanPulseWidth): Cannot set pulse width for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    if (pulseWidth < 0.f) {
        helios_runtime_error("ERROR (LiDARcloud::setScanPulseWidth): Pulse width must be non-negative, but " + std::to_string(pulseWidth) + " was given.");
    }
    scans.at(scanID).pulseWidth = pulseWidth;
}

float LiDARcloud::getScanDetectionThreshold(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanDetectionThreshold): Cannot get detection threshold for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).detectionThreshold;
}

void LiDARcloud::setScanDetectionThreshold(uint scanID, float detectionThreshold) {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::setScanDetectionThreshold): Cannot set detection threshold for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    if (detectionThreshold < 0.f) {
        helios_runtime_error("ERROR (LiDARcloud::setScanDetectionThreshold): Detection threshold must be non-negative, but " + std::to_string(detectionThreshold) + " was given.");
    }
    scans.at(scanID).detectionThreshold = detectionThreshold;
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

ScanMode LiDARcloud::getScanMode(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanMode): Cannot get scan mode for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).scanMode;
}

uint LiDARcloud::getScanStepsPerRev(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanStepsPerRev): Cannot get steps per revolution for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).steps_per_rev;
}

double LiDARcloud::getScanRotationRate(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanRotationRate): Cannot get rotation rate for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).rotation_rate;
}

double LiDARcloud::getScanRevolutions(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanRevolutions): Cannot get revolution count for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).n_revolutions;
}

std::vector<RisleyPrism> LiDARcloud::getScanRisleyPrisms(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanRisleyPrisms): Cannot get Risley prisms for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).risley_prisms;
}

double LiDARcloud::getScanRisleyRefractiveIndexAir(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanRisleyRefractiveIndexAir): Cannot get the Risley air refractive index for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).risley_refractive_index_air;
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

    // Columnar store: resolve (or create, back-filled absent) the label's column, then set this hit's
    // value and mark it present. Observably identical to the old per-hit map[label]=value.
    const size_t slot = getOrCreateHitDataColumn(label);
    hit_data_columns[slot][index] = value;
    hit_data_present[slot][index] = char(1);
}

double LiDARcloud::getHitData(uint index, const char *label) const {

    if (index >= hits.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getHitData): Hit point index out of bounds. Requesting hit #" + std::to_string(index) + " but scan only has " + std::to_string(hits.size()) + " hits.");
    }

    // O(1): one small label->slot hash lookup + one contiguous indexed read, instead of a per-hit
    // red-black-tree descent. Absent (label never set, or not set on this hit) throws the same error.
    auto it = hit_data_label_index.find(label);
    if (it == hit_data_label_index.end() || hit_data_present[it->second][index] == char(0)) {
        helios_runtime_error("ERROR (LiDARcloud::getHitData): Data value ``" + std::string(label) + "'' does not exist.");
    }

    return hit_data_columns[it->second][index];
}

bool LiDARcloud::doesHitDataExist(uint index, const char *label) const {

    if (index >= hits.size()) {
        return false;
    }

    auto it = hit_data_label_index.find(label);
    return it != hit_data_label_index.end() && hit_data_present[it->second][index] != char(0);
}

int LiDARcloud::getHitDataColumnIndex(const char *label) const {
    auto it = hit_data_label_index.find(label);
    if (it == hit_data_label_index.end()) {
        return -1;
    }
    return int(it->second);
}

void LiDARcloud::getHitDataColumn(const char *label, std::vector<double> &data, double absent_value) const {
    const size_t N = hits.size();
    data.resize(N);

    auto it = hit_data_label_index.find(label);
    if (it == hit_data_label_index.end()) {
        // Label never set on any hit: every entry is absent.
        std::fill(data.begin(), data.end(), absent_value);
        return;
    }

    // Single cache-linear pass over the contiguous value and presence columns.
    const std::vector<double> &column = hit_data_columns[it->second];
    const std::vector<char> &present = hit_data_present[it->second];
    for (size_t i = 0; i < N; i++) {
        data[i] = present[i] != char(0) ? column[i] : absent_value;
    }
}

void LiDARcloud::clearHits() {
    hits.clear();
    hit_data_labels.clear();
    hit_data_label_index.clear();
    hit_data_columns.clear();
    hit_data_present.clear();
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

void LiDARcloud::transformHitOrigin(uint index, const std::function<helios::vec3(const helios::vec3 &)> &transform) {
    // Moving-platform hits store their own per-pulse emission origin (labels origin_x/y/z), which must be
    // transformed together with the hit position so the two stay in the same coordinate frame. Static hits
    // carry no such labels and are left unchanged. All-three-or-none semantics preserved.
    if (doesHitDataExist(index, "origin_x") && doesHitDataExist(index, "origin_y") && doesHitDataExist(index, "origin_z")) {
        helios::vec3 o = helios::make_vec3(float(getHitData(index, "origin_x")), float(getHitData(index, "origin_y")), float(getHitData(index, "origin_z")));
        o = transform(o);
        setHitData(index, "origin_x", o.x);
        setHitData(index, "origin_y", o.y);
        setHitData(index, "origin_z", o.z);
    }
}

helios::vec3 LiDARcloud::hitOriginOrFallback(uint index, const helios::vec3 &fallback) const {
    if (doesHitDataExist(index, "origin_x") && doesHitDataExist(index, "origin_y") && doesHitDataExist(index, "origin_z")) {
        return helios::make_vec3(float(getHitData(index, "origin_x")), float(getHitData(index, "origin_y")), float(getHitData(index, "origin_z")));
    }
    return fallback;
}

void LiDARcloud::coordinateShift(const vec3 &shift) {

    for (auto &scan: scans) {
        scan.origin = scan.origin + shift;
    }

    for (size_t i = 0; i < hits.size(); i++) {
        hits[i].position = hits[i].position + shift;
        transformHitOrigin(uint(i), [&](const vec3 &o) { return o + shift; });
    }
}

void LiDARcloud::coordinateShift(uint scanID, const vec3 &shift) {

    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::coordinateShift): Cannot apply coordinate shift to scan " + std::to_string(scanID) + " because it does not exist.");
    }

    scans.at(scanID).origin = scans.at(scanID).origin + shift;

    for (size_t i = 0; i < hits.size(); i++) {
        if (hits[i].scanID == scanID) {
            hits[i].position = hits[i].position + shift;
            transformHitOrigin(uint(i), [&](const vec3 &o) { return o + shift; });
        }
    }
}

void LiDARcloud::coordinateRotation(const SphericalCoord &rotation) {

    for (auto &scan: scans) {
        scan.origin = rotatePoint(scan.origin, rotation);
    }

    for (size_t i = 0; i < hits.size(); i++) {
        hits[i].position = rotatePoint(hits[i].position, rotation);
        transformHitOrigin(uint(i), [&](const vec3 &o) { return rotatePoint(o, rotation); });
        // Recompute the stored ray direction from the hit's own (transformed) origin so it remains correct for moving scans.
        hits[i].direction = cart2sphere(hits[i].position - hitOriginOrFallback(uint(i), scans.at(hits[i].scanID).origin));
    }
}

void LiDARcloud::coordinateRotation(uint scanID, const SphericalCoord &rotation) {

    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::coordinateRotation): Cannot apply rotation to scan " + std::to_string(scanID) + " because it does not exist.");
    }

    scans.at(scanID).origin = rotatePoint(scans.at(scanID).origin, rotation);

    for (size_t i = 0; i < hits.size(); i++) {
        if (hits[i].scanID == scanID) {
            hits[i].position = rotatePoint(hits[i].position, rotation);
            transformHitOrigin(uint(i), [&](const vec3 &o) { return rotatePoint(o, rotation); });
            hits[i].direction = cart2sphere(hits[i].position - hitOriginOrFallback(uint(i), scans.at(scanID).origin));
        }
    }
}

void LiDARcloud::coordinateRotation(float rotation, const vec3 &line_base, const vec3 &line_direction) {

    for (auto &scan: scans) {
        scan.origin = rotatePointAboutLine(scan.origin, line_base, line_direction, rotation);
    }

    for (size_t i = 0; i < hits.size(); i++) {
        hits[i].position = rotatePointAboutLine(hits[i].position, line_base, line_direction, rotation);
        transformHitOrigin(uint(i), [&](const vec3 &o) { return rotatePointAboutLine(o, line_base, line_direction, rotation); });
        hits[i].direction = cart2sphere(hits[i].position - hitOriginOrFallback(uint(i), scans.at(hits[i].scanID).origin));
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
    addGrid(center, size, ndiv, rotation, std::vector<float>());
}

void LiDARcloud::addGrid(const vec3 &center, const vec3 &size, const int3 &ndiv, float rotation, const std::vector<float> &column_z_offsets) {
    if (size.x <= 0 || size.y <= 0 || size.z <= 0) {
        cerr << "failed.\n";
        helios_runtime_error("ERROR (LiDARcloud::addGrid): The grid cell size must be positive.");
    }

    if (ndiv.x <= 0 || ndiv.y <= 0 || ndiv.z <= 0) {
        cerr << "failed.\n";
        helios_runtime_error("ERROR (LiDARcloud::addGrid): The number of grid cells in each direction must be positive.");
    }

    // Optional per-column vertical offset for terrain following. When supplied it must hold one value
    // per (x,y) column, row-major as [j*ndiv.x + i]. Each cell's z is shifted by its column's offset so
    // that vertical voxel columns track an external terrain surface (e.g. a DEM). Empty => no shift, in
    // which case this is byte-for-byte identical to the axis-regular grid built previously.
    const bool terrain_follow = !column_z_offsets.empty();
    if (terrain_follow && column_z_offsets.size() != size_t(ndiv.x) * size_t(ndiv.y)) {
        helios_runtime_error("ERROR (LiDARcloud::addGrid): column_z_offsets must have length ndiv.x*ndiv.y (one value per grid column).");
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

                float zoff = terrain_follow ? column_z_offsets[size_t(j) * size_t(ndiv.x) + size_t(i)] : 0.f;

                vec3 subcenter = make_vec3(x, y, z + zoff);

                vec3 subcenter_rot = rotatePoint(subcenter, make_SphericalCoord(0, rotation * M_PI / 180.f));

                if (printmessages) {
                    cout << "Adding grid cell #" << count << " with center " << subcenter_rot.x + center.x << "," << subcenter_rot.y + center.y << "," << subcenter.z + center.z << " and size " << gsubsize.x << " x " << gsubsize.y << " x "
                         << gsubsize.z << endl;
                }

                addGridCell(subcenter + center, center, gsubsize, size, rotation * M_PI / 180.f, make_int3(i, j, k), ndiv);

                if (terrain_follow) {
                    grid_cells.back().ground_height = zoff;
                }

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
        if (doesHitDataExist(r, "reflectance")) {
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
        if (doesHitDataExist(r, scalar_field)) {
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

        // Cancellation checkpoint between scans: a cancelled run discards any mesh built so far and
        // returns empty (triangulationcomputed stays false) rather than starting the next scan.
        if (triangulationCancelled(cancel_flag)) {
            triangles.clear();
            return;
        }

        std::vector<int> Delaunay_inds;

        std::vector<Shx> pts, pts_copy;

        int count = 0;
        for (int r = 0; r < getHitCount(); r++) {

            // Coarse cancellation poll inside the gather loop (~every 128k hits) so a huge single
            // scan can be aborted before it even reaches the Delaunay call.
            if ((r & 0x1FFFF) == 0 && triangulationCancelled(cancel_flag)) {
                triangles.clear();
                return;
            }

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

        // Cancellation poll immediately before the (uninterruptible) Delaunay call: if the caller
        // already requested an abort, skip this scan's CDT work entirely and return empty.
        if (triangulationCancelled(cancel_flag)) {
            triangles.clear();
            return;
        }

        // CDT uses robust geometric predicates, so the s_hull-era
        // rotate-and-retry recovery is no longer needed; a failure here is
        // deterministic and the scan is skipped.
        int success = triangulate_CDT(pts, triads);

        // Cancellation poll immediately after CDT: bail before building the output mesh so a run
        // cancelled during the (one-scan-bounded) tessellation still aborts promptly.
        if (triangulationCancelled(cancel_flag)) {
            triangles.clear();
            return;
        }

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

            // Coarse cancellation poll inside the triad-build loop (~every 128k triangles).
            if ((t & 0x1FFFF) == 0 && triangulationCancelled(cancel_flag)) {
                triangles.clear();
                return;
            }

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

int LiDARcloud::getContainingGridCell(const helios::vec3 &p) const {

    // Mirrors the per-cell containment test in calculateHitGridCell(): the original ray-from-origin
    // slab test reduces to a point-in-AABB containment test, applied in each cell's local frame
    // (inverse-rotated about the cell anchor when the cell is rotated). Kept here as a self-contained
    // helper using the bounds-checked getters; calculateHitGridCell() caches cell geometry in flat
    // arrays for its OpenMP hot loop, but the external-triangle path is far smaller and does not need
    // that, so the two stay logically identical without sharing the cached buffers.
    const uint Ncells = getGridCellCount();
    for (uint c = 0; c < Ncells; c++) {

        helios::vec3 center = getCellCenter(c);
        helios::vec3 size = getCellSize(c);
        helios::vec3 lo = center - size * 0.5f;
        helios::vec3 hi = center + size * 0.5f;

        helios::vec3 q = p;
        float rotation = getCellRotation(c);
        if (fabs(rotation) > 1e-6f) {
            helios::vec3 anchor = getCellGlobalAnchor(c);
            q = rotatePointAboutLine(p - anchor, helios::make_vec3(0, 0, 0), helios::make_vec3(0, 0, 1), -rotation) + anchor;
        }

        if (q.x >= lo.x && q.x <= hi.x && q.y >= lo.y && q.y <= hi.y && q.z >= lo.z && q.z <= hi.z) {
            return static_cast<int>(c);
        }
    }
    return -1;
}

void LiDARcloud::setExternalTriangulation(const std::vector<helios::vec3> &triangle_vertices, const std::vector<int> &scanIDs) {

    if (triangle_vertices.size() % 3 != 0) {
        helios_runtime_error("ERROR (LiDARcloud::setExternalTriangulation): triangle_vertices size (" + std::to_string(triangle_vertices.size()) + ") must be a multiple of 3 (three vertices per triangle).");
    }

    const size_t Ntri = triangle_vertices.size() / 3;

    if (scanIDs.size() != Ntri) {
        helios_runtime_error("ERROR (LiDARcloud::setExternalTriangulation): scanIDs size (" + std::to_string(scanIDs.size()) + ") must equal the triangle count (" + std::to_string(Ntri) +
                             "). Each triangle requires a source scan for the G(theta) ray direction.");
    }

    if (getGridCellCount() == 0) {
        helios_runtime_error("ERROR (LiDARcloud::setExternalTriangulation): a grid must be defined (see addGrid()) before supplying an external triangulation, so each triangle can be assigned to a grid cell.");
    }

    const uint Nscans = getScanCount();
    for (size_t t = 0; t < Ntri; t++) {
        if (scanIDs.at(t) < 0 || static_cast<uint>(scanIDs.at(t)) >= Nscans) {
            helios_runtime_error("ERROR (LiDARcloud::setExternalTriangulation): triangle " + std::to_string(t) + " has scanID " + std::to_string(scanIDs.at(t)) + ", which is not a valid scan index in [0, " + std::to_string(Nscans) +
                                 "). Per-scan provenance is required; a merged mesh with no scan association is not a valid input.");
        }
    }

    // Discard any previous triangulation and reset diagnostics for this run (see getTriangulation*).
    triangles.clear();
    triangulation_candidate_count = Ntri;
    triangulation_dropped_lmax = 0;
    triangulation_dropped_aspect = 0;
    triangulation_dropped_degenerate = 0;

    for (size_t t = 0; t < Ntri; t++) {

        const helios::vec3 &v0 = triangle_vertices.at(3 * t + 0);
        const helios::vec3 &v1 = triangle_vertices.at(3 * t + 1);
        const helios::vec3 &v2 = triangle_vertices.at(3 * t + 2);

        // Assign by centroid containment so the triangle lands in the same cell its bulk occupies.
        helios::vec3 centroid = (v0 + v1 + v2) / 3.f;
        int gridcell = getContainingGridCell(centroid);

        // ID0/ID1/ID2 are hit-point indices for the internal path; unused by G(theta), so -1 here.
        Triangulation tri(scanIDs.at(t), v0, v1, v2, -1, -1, -1, helios::RGB::green, gridcell);

        // Drop degenerate triangles (collinear/zero-extent vertices give NaN area via Heron's formula),
        // matching triangulateHitPoints().
        if (tri.area != tri.area || tri.area <= 0.f) {
            triangulation_dropped_degenerate++;
            continue;
        }

        triangles.push_back(tri);
    }

    triangulationcomputed = true;

    if (printmessages) {
        std::cout << "Set external triangulation: " << triangles.size() << " triangles (" << triangulation_dropped_degenerate << " degenerate dropped)." << std::endl;
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

        // Cancellation checkpoint between scans: a cancelled run discards any mesh built so far and
        // returns empty (triangulationcomputed stays false) rather than starting the next scan.
        if (triangulationCancelled(cancel_flag)) {
            triangles.clear();
            return;
        }

        std::vector<int> Delaunay_inds;

        std::vector<Shx> pts, pts_copy;

        std::size_t delete_count = 0;
        int count = 0;

        for (int r = 0; r < getHitCount(); r++) {

            // Coarse cancellation poll inside the gather loop (~every 128k hits) so a huge single
            // scan can be aborted before it even reaches the Delaunay call.
            if ((r & 0x1FFFF) == 0 && triangulationCancelled(cancel_flag)) {
                triangles.clear();
                return;
            }

            if (getHitScanID(r) == s && getHitGridCell(r) >= 0) {

                if (doesHitDataExist(r, scalar_field)) {
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

        // Cancellation poll immediately before the (uninterruptible) Delaunay call: if the caller
        // already requested an abort, skip this scan's CDT work entirely and return empty.
        if (triangulationCancelled(cancel_flag)) {
            triangles.clear();
            return;
        }

        // CDT uses robust geometric predicates, so the s_hull-era
        // rotate-and-retry recovery is no longer needed; a failure here is
        // deterministic and the scan is skipped.
        int success = triangulate_CDT(pts, triads);

        // Cancellation poll immediately after CDT: bail before building the output mesh so a run
        // cancelled during the (one-scan-bounded) tessellation still aborts promptly.
        if (triangulationCancelled(cancel_flag)) {
            triangles.clear();
            return;
        }

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

            // Coarse cancellation poll inside the triad-build loop (~every 128k triangles).
            if ((t & 0x1FFFF) == 0 && triangulationCancelled(cancel_flag)) {
                triangles.clear();
                return;
            }

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
    // Triangulation-derived G(theta) (the original behavior): an empty supplied-G(theta) vector tells the inner
    // routine to compute G(theta) per voxel from triangulation.
    calculateLeafArea_inner(context, min_voxel_hits, element_width, std::vector<float>{});
}

void LiDARcloud::calculateLeafArea(helios::Context *context, float Gtheta, int min_voxel_hits, float element_width) {
    // Caller-supplied G(theta) for scans that cannot be triangulated (e.g. moving-platform scans). A single value is
    // broadcast to every voxel by the inner routine.
    if (!(Gtheta > 0.f) || Gtheta > 1.f) {
        helios_runtime_error("ERROR (LiDARcloud::calculateLeafArea): The supplied G(theta) must be in the range (0,1], but " + std::to_string(Gtheta) + " was provided. Use 0.5 for a spherical (random) leaf-angle distribution.");
    }
    calculateLeafArea_inner(context, min_voxel_hits, element_width, std::vector<float>{Gtheta});
}

void LiDARcloud::calculateLeafArea(helios::Context *context, const std::vector<float> &Gtheta_per_cell, int min_voxel_hits, float element_width) {
    // Caller-supplied PER-VOXEL G(theta) (e.g. a vertically-varying leaf-angle distribution). The length must match the
    // grid-cell count and every value must be in (0,1]; the inner routine uses each value for its corresponding voxel.
    const uint Ncells = getGridCellCount();
    if (Gtheta_per_cell.size() != Ncells) {
        helios_runtime_error("ERROR (LiDARcloud::calculateLeafArea): The per-voxel G(theta) vector has " + std::to_string(Gtheta_per_cell.size()) + " entries but the grid has " + std::to_string(Ncells) +
                             " cells. Supply exactly one G(theta) per cell, in grid-cell order.");
    }
    for (size_t v = 0; v < Gtheta_per_cell.size(); v++) {
        const float g = Gtheta_per_cell[v];
        if (!(g > 0.f) || g > 1.f) {
            helios_runtime_error("ERROR (LiDARcloud::calculateLeafArea): Per-voxel G(theta) values must be in the range (0,1], but cell " + std::to_string(v) + " was given " + std::to_string(g) +
                                 ". Use 0.5 for a spherical (random) leaf-angle distribution.");
        }
    }
    calculateLeafArea_inner(context, min_voxel_hits, element_width, Gtheta_per_cell);
}

void LiDARcloud::accumulateBeamCell(const uint *return_indices, size_t Nreturns, const std::vector<float> &dr, const std::vector<uint> &hit_location, float &P_equal_numerator, float &P_equal_denominator, float &P_equal_sumsq,
                                    std::vector<float> &dr_array_cell) {

    float E_before = 0, E_inside = 0, E_after = 0;
    float drr = 0;
    int dr_count = 0; // Count returns with dr > 0

    // Count returns in each location for this beam. Misses (transmitted beams)
    // are class-3 (after voxel), so they are folded into E_after here.
    for (size_t r = 0; r < Nreturns; r++) {
        uint local_index = return_indices[r];
        if (dr[local_index] > 0) {
            drr += dr[local_index];
            dr_count++;
        }

        if (hit_location[local_index] == 1)
            E_before++;
        else if (hit_location[local_index] == 2)
            E_inside++;
        else if (hit_location[local_index] == 3)
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
        dr_array_cell.push_back(drrx);
    }
}

// Ray-AABB slab test producing the entry (t0) and exit (t1) parameters for a single voxel. This uses the IDENTICAL
// per-axis expression as the brute-force inversion path ((min-origin)/dir, swap, max-of-mins / min-of-maxs) so that the
// DDA fast path classifies returns bit-for-bit the same as the brute-force path. Returns true if the ray's slab
// interval is non-empty (t0 < t1). Axis-parallel rays are handled by IEEE infinity arithmetic, as in the original.
static bool cellSlab(const helios::vec3 &origin, const helios::vec3 &direction, const helios::vec3 &voxel_min, const helios::vec3 &voxel_max, float &t0, float &t1) {
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

    t0 = std::max({tx_min, ty_min, tz_min});
    t1 = std::min({tx_max, ty_max, tz_max});
    return t0 < t1;
}

// Ray-AABB slab test against the whole-grid bounding box, used to find where a beam enters/exits the lattice before
// running DDA. Handles axis-parallel rays explicitly (the fixed coordinate must lie within the box on that axis).
static bool rayGridIntersect(const helios::vec3 &origin, const helios::vec3 &direction, const helios::vec3 &grid_min, const helios::vec3 &grid_max, float &t_enter, float &t_exit) {
    const float o[3] = {origin.x, origin.y, origin.z};
    const float d[3] = {direction.x, direction.y, direction.z};
    const float lo[3] = {grid_min.x, grid_min.y, grid_min.z};
    const float hi[3] = {grid_max.x, grid_max.y, grid_max.z};

    float tmin = -std::numeric_limits<float>::max();
    float tmax = std::numeric_limits<float>::max();
    for (int ax = 0; ax < 3; ax++) {
        if (fabs(d[ax]) < 1e-9f) {
            if (o[ax] < lo[ax] || o[ax] > hi[ax])
                return false; // parallel and outside the slab
        } else {
            float t1 = (lo[ax] - o[ax]) / d[ax];
            float t2 = (hi[ax] - o[ax]) / d[ax];
            if (t1 > t2)
                std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            if (tmin > tmax)
                return false;
        }
    }
    t_enter = tmin;
    t_exit = tmax;
    return true;
}

LiDARcloud::VoxelLattice LiDARcloud::detectVoxelLattice() const {

    VoxelLattice lattice;

    const uint Ncells = getGridCellCount();
    if (Ncells == 0) {
        return lattice; // invalid (no cells)
    }

    // Reference values taken from the first cell; every cell must agree for a regular lattice.
    const GridCell &ref = grid_cells.front();
    const helios::int3 count = ref.global_count;
    if (count.x <= 0 || count.y <= 0 || count.z <= 0) {
        return lattice;
    }

    // The grid produced by addGrid() has exactly count.x*count.y*count.z cells. A different total
    // means the cells were assembled some other way and cannot be assumed to tile the lattice.
    const size_t expected_cells = (size_t) count.x * (size_t) count.y * (size_t) count.z;
    if ((size_t) Ncells != expected_cells) {
        return lattice;
    }

    const helios::vec3 cell_extent = make_vec3(ref.global_size.x / float(count.x), ref.global_size.y / float(count.y), ref.global_size.z / float(count.z));
    const helios::vec3 lattice_origin = ref.global_anchor - ref.global_size * 0.5f;

    // Tolerances scaled to the cell size so the checks are robust to float round-off in the stored centers.
    const float pos_tol = 1e-4f * std::max({cell_extent.x, cell_extent.y, cell_extent.z, 1e-6f});
    const float rot_tol = 1e-6f;

    std::vector<int> ijk_to_index(expected_cells, -1);

    for (uint c = 0; c < Ncells; c++) {
        const GridCell &cell = grid_cells.at(c);

        // Shared lattice parameters
        if (cell.global_count.x != count.x || cell.global_count.y != count.y || cell.global_count.z != count.z) {
            return lattice;
        }
        if ((cell.global_anchor - ref.global_anchor).magnitude() > pos_tol || (cell.global_size - ref.global_size).magnitude() > pos_tol || fabs(cell.azimuthal_rotation - ref.azimuthal_rotation) > rot_tol) {
            return lattice;
        }

        // Per-cell size must equal the lattice cell extent
        if (fabs(cell.size.x - cell_extent.x) > pos_tol || fabs(cell.size.y - cell_extent.y) > pos_tol || fabs(cell.size.z - cell_extent.z) > pos_tol) {
            return lattice;
        }

        // global_ijk must be in range and unique
        const helios::int3 ijk = cell.global_ijk;
        if (ijk.x < 0 || ijk.x >= count.x || ijk.y < 0 || ijk.y >= count.y || ijk.z < 0 || ijk.z >= count.z) {
            return lattice;
        }
        const size_t flat = ((size_t) ijk.z * count.y + ijk.y) * count.x + ijk.x;
        if (ijk_to_index[flat] != -1) {
            return lattice; // duplicate ijk
        }
        ijk_to_index[flat] = (int) c;

        // Stored center must match the lattice position for this ijk (un-rotated frame: addGrid stores
        // the un-rotated offset; see addGrid()/calculateHitGridCell()).
        const helios::vec3 expected_center = lattice_origin + make_vec3((float(ijk.x) + 0.5f) * cell_extent.x, (float(ijk.y) + 0.5f) * cell_extent.y, (float(ijk.z) + 0.5f) * cell_extent.z);
        if ((cell.center - expected_center).magnitude() > pos_tol) {
            return lattice;
        }
    }

    lattice.valid = true;
    lattice.origin = lattice_origin;
    lattice.anchor = ref.global_anchor;
    lattice.cell_extent = cell_extent;
    lattice.rotation = ref.azimuthal_rotation;
    lattice.count = count;
    lattice.ijk_to_index = std::move(ijk_to_index);

    return lattice;
}

void LiDARcloud::calculateLeafArea_inner(helios::Context *context, int min_voxel_hits, float element_width, const std::vector<float> &supplied_Gtheta) {

    // An empty vector means "compute G(theta) per voxel from triangulation"; a non-empty vector (size 1 = broadcast, or
    // size == Ncells = per-voxel) means the caller supplied G(theta) and triangulation is not required.
    const bool use_supplied_Gtheta = !supplied_Gtheta.empty();

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

        // When the grid cells form a regular lattice (the common addGrid() case), walk each beam through only the
        // voxels it pierces (3D-DDA) instead of testing every hit against every voxel. This reduces the inversion from
        // O(Nscans * Ncells * Nhits) to O(Nscans * Nbeams * cells_pierced_per_beam). Grids that are not a regular
        // lattice (e.g. assembled cell-by-cell with mixed sizes/rotations) fall back to the brute-force per-cell loop.
        const VoxelLattice lattice = detectVoxelLattice();

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

            // Group hits by timestamp into beams (CSR layout). beam_members holds GLOBAL hit indices; the
            // per-voxel classification arrays (dr, hit_location) are local to this scan, so beam members are
            // mapped back to local positions below.
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

            // Precompute each beam member's local (this-scan) return index once, in the same CSR layout as
            // beams (flat array indexed by beams.beam_offsets). A flat array avoids the per-beam allocation a
            // vector-of-vectors would incur (prohibitive for tens of millions of beams).
            std::vector<uint> beam_members_local(beams.beam_members.size());
            for (size_t m = 0; m < beams.beam_members.size(); m++) {
                beam_members_local[m] = global_to_local[beams.beam_members[m]];
            }

            if (lattice.valid && !force_bruteforce_LAD) {

                // -------- FAST PATH: per-beam 3D-DDA over the voxel lattice --------

                // Walk one beam through the lattice, classifying its returns in each pierced voxel and accumulating
                // into the supplied per-cell scratch. Factored into a lambda so it can run with thread-private scratch.
                // The caller supplies reusable per-thread scratch buffers (ret_dist/dr_cell/hl_cell/local_seq) so the
                // hot per-beam path performs no heap allocation - with millions of beams, per-call allocation would
                // dominate the runtime and serialize the threads through the allocator lock.
                auto process_beam = [&](uint k, std::vector<float> &P_num, std::vector<float> &P_denom, std::vector<float> &P_sumsq, std::vector<std::vector<float>> &dr_cells, std::vector<float> &ret_dist, std::vector<float> &dr_cell,
                                        std::vector<uint> &hl_cell, std::vector<uint> &local_seq) {
                    const uint beam_start = beams.beam_offsets[k];
                    const size_t Nret = beams.beam_offsets[k + 1] - beam_start;
                    if (Nret == 0)
                        return;

                    // All returns of a pulse share one emission origin; use the first return's origin.
                    helios::vec3 origin = this_scan_origin[beam_members_local[beam_start]];

                    // Beam direction: use the farthest return (typically the miss point) so the ray spans the
                    // full beam. Transform origin and hit points into the un-rotated lattice frame (rotation is
                    // shared across all cells and pivots on the lattice anchor - matches calculateHitGridCell()).
                    helios::vec3 origin_L = origin;
                    if (fabs(lattice.rotation) > 1e-6f) {
                        origin_L = rotatePointAboutLine(origin - lattice.anchor, helios::make_vec3(0, 0, 0), helios::make_vec3(0, 0, 1), -lattice.rotation) + lattice.anchor;
                    }

                    helios::vec3 direction(0, 0, 0);
                    float max_hit_distance = -1.f;
                    // Per-return distance from origin (lattice frame), into the reusable buffer.
                    ret_dist.resize(Nret);
                    for (size_t j = 0; j < Nret; j++) {
                        uint i = beam_members_local[beam_start + j];
                        helios::vec3 hit_L = this_scan_xyz[i];
                        if (fabs(lattice.rotation) > 1e-6f) {
                            hit_L = rotatePointAboutLine(hit_L - lattice.anchor, helios::make_vec3(0, 0, 0), helios::make_vec3(0, 0, 1), -lattice.rotation) + lattice.anchor;
                        }
                        helios::vec3 d = hit_L - origin_L;
                        float dist = d.magnitude();
                        ret_dist[j] = dist;
                        if (dist > max_hit_distance) {
                            max_hit_distance = dist;
                            direction = d;
                        }
                    }
                    if (max_hit_distance <= 0.f)
                        return;
                    direction.normalize();

                    // Slab-test the beam against the whole-grid AABB to find the entry/exit parameters.
                    helios::vec3 grid_min = lattice.origin;
                    helios::vec3 grid_max = lattice.origin + make_vec3(lattice.cell_extent.x * lattice.count.x, lattice.cell_extent.y * lattice.count.y, lattice.cell_extent.z * lattice.count.z);

                    float t_enter, t_exit;
                    if (!rayGridIntersect(origin_L, direction, grid_min, grid_max, t_enter, t_exit))
                        return; // beam misses the grid entirely

                    float t_start = std::max(t_enter, 0.f);
                    if (t_exit <= 1e-6f)
                        return; // grid entirely behind the origin

                    // Amanatides-Woo DDA initialization.
                    helios::vec3 entry = origin_L + direction * t_start;
                    int ijk[3];
                    int step[3];
                    float tMax[3];
                    float tDelta[3];
                    const float origin_arr[3] = {origin_L.x, origin_L.y, origin_L.z};
                    const float dir_arr[3] = {direction.x, direction.y, direction.z};
                    const float entry_arr[3] = {entry.x, entry.y, entry.z};
                    const float gmin_arr[3] = {grid_min.x, grid_min.y, grid_min.z};
                    const float extent_arr[3] = {lattice.cell_extent.x, lattice.cell_extent.y, lattice.cell_extent.z};
                    const int count_arr[3] = {lattice.count.x, lattice.count.y, lattice.count.z};
                    for (int ax = 0; ax < 3; ax++) {
                        int idx = (int) std::floor((entry_arr[ax] - gmin_arr[ax]) / extent_arr[ax]);
                        if (idx < 0)
                            idx = 0;
                        if (idx >= count_arr[ax])
                            idx = count_arr[ax] - 1;
                        ijk[ax] = idx;
                        if (fabs(dir_arr[ax]) < 1e-9f) {
                            // Ray parallel to this axis' slabs: never crosses a boundary on this axis.
                            step[ax] = 0;
                            tMax[ax] = std::numeric_limits<float>::max();
                            tDelta[ax] = std::numeric_limits<float>::max();
                        } else {
                            step[ax] = (dir_arr[ax] > 0) ? 1 : -1;
                            float next_boundary = gmin_arr[ax] + float(idx + (step[ax] > 0 ? 1 : 0)) * extent_arr[ax];
                            tMax[ax] = (next_boundary - origin_arr[ax]) / dir_arr[ax];
                            tDelta[ax] = extent_arr[ax] / fabs(dir_arr[ax]);
                        }
                    }

                    // Walk the lattice from entry to exit, classifying the beam's returns in each pierced voxel.
                    // dr_cell/hl_cell/local_seq are the caller's reusable buffers; size them to this beam.
                    dr_cell.assign(Nret, 0.f);
                    hl_cell.assign(Nret, 0);
                    local_seq.resize(Nret);
                    for (uint j = 0; j < Nret; j++)
                        local_seq[j] = j;

                    const size_t max_steps = (size_t) count_arr[0] + count_arr[1] + count_arr[2] + 3;
                    for (size_t stepcount = 0; stepcount <= max_steps; stepcount++) {
                        if (ijk[0] < 0 || ijk[0] >= count_arr[0] || ijk[1] < 0 || ijk[1] >= count_arr[1] || ijk[2] < 0 || ijk[2] >= count_arr[2])
                            break;

                        const size_t flat = ((size_t) ijk[2] * count_arr[1] + ijk[1]) * count_arr[0] + ijk[0];
                        int cell_index = lattice.ijk_to_index[flat];
                        if (cell_index >= 0) {
                            // Compute (t0,t1) from this cell's own corners with the SAME slab expression the
                            // brute-force path uses, so classification is bit-identical (not from running tMax).
                            helios::vec3 cmin = lattice.origin + make_vec3(ijk[0] * lattice.cell_extent.x, ijk[1] * lattice.cell_extent.y, ijk[2] * lattice.cell_extent.z);
                            helios::vec3 cmax = cmin + lattice.cell_extent;

                            float ct0, ct1;
                            if (cellSlab(origin_L, direction, cmin, cmax, ct0, ct1) && ct1 > 1e-6f) {
                                float drval = fabs(ct1 - ct0);
                                // Classify each return of this beam against [ct0,ct1].
                                for (size_t j = 0; j < Nret; j++) {
                                    float hd = ret_dist[j];
                                    dr_cell[j] = drval;
                                    if (hd >= ct0 && hd <= ct1)
                                        hl_cell[j] = 2;
                                    else if (hd > ct1)
                                        hl_cell[j] = 3;
                                    else
                                        hl_cell[j] = 1;
                                }
                                accumulateBeamCell(local_seq.data(), Nret, dr_cell, hl_cell, P_num[cell_index], P_denom[cell_index], P_sumsq[cell_index], dr_cells[cell_index]);
                            }
                        }

                        // Advance to the next voxel along the smallest tMax.
                        int axis = 0;
                        if (tMax[1] < tMax[axis])
                            axis = 1;
                        if (tMax[2] < tMax[axis])
                            axis = 2;
                        if (tMax[axis] > t_exit)
                            break; // exited the grid
                        if (step[axis] == 0)
                            break; // no further progress possible
                        ijk[axis] += step[axis];
                        tMax[axis] += tDelta[axis];
                    }
                };

                // Per-(scan,cell) accumulators. The brute-force path pushes one value per scan into
                // P_equal_*_array[c]; with per-beam nesting we accumulate here and flush once at scan end.
                std::vector<float> P_num_scratch(Ncells, 0.f);
                std::vector<float> P_denom_scratch(Ncells, 0.f);
                std::vector<float> P_sumsq_scratch(Ncells, 0.f);
                std::vector<std::vector<float>> dr_scratch(Ncells);

                // Parallelize over beams (independent). Each thread accumulates into private scratch, then the threads'
                // results are reduced in ascending thread-id order. The dr-sample order still depends on how beams are
                // distributed across threads, so the per-voxel dr mean differs at the FP-rounding level from the serial
                // order - immaterial to the inversion (which uses the mean and count), exactly as the prior
                // OpenMP-over-hits brute-force path was already order-dependent.
#ifdef _OPENMP
                int num_threads = omp_get_max_threads();
#else
                int num_threads = 1;
#endif
                std::vector<std::vector<float>> P_num_thread(num_threads, std::vector<float>(Ncells, 0.f));
                std::vector<std::vector<float>> P_denom_thread(num_threads, std::vector<float>(Ncells, 0.f));
                std::vector<std::vector<float>> P_sumsq_thread(num_threads, std::vector<float>(Ncells, 0.f));
                std::vector<std::vector<std::vector<float>>> dr_thread(num_threads, std::vector<std::vector<float>>(Ncells));

#pragma omp parallel
                {
#ifdef _OPENMP
                    int tid = omp_get_thread_num();
#else
                    int tid = 0;
#endif
                    // Per-thread reusable scratch buffers (no per-beam allocation in the hot loop).
                    std::vector<float> ret_dist, dr_cell;
                    std::vector<uint> hl_cell, local_seq;
#pragma omp for schedule(dynamic, 256)
                    for (int k = 0; k < static_cast<int>(Nbeams); k++) {
                        process_beam((uint) k, P_num_thread[tid], P_denom_thread[tid], P_sumsq_thread[tid], dr_thread[tid], ret_dist, dr_cell, hl_cell, local_seq);
                    }
                }

                // Deterministic reduction across threads (ascending thread id).
                for (int t = 0; t < num_threads; t++) {
                    for (uint c = 0; c < Ncells; c++) {
                        P_num_scratch[c] += P_num_thread[t][c];
                        P_denom_scratch[c] += P_denom_thread[t][c];
                        P_sumsq_scratch[c] += P_sumsq_thread[t][c];
                        for (float v: dr_thread[t][c])
                            dr_scratch[c].push_back(v);
                    }
                }

                // Flush per-(scan,cell) accumulators into the cross-scan arrays.
                for (uint c = 0; c < Ncells; c++) {
                    P_equal_numerator_array.at(c).push_back(P_num_scratch[c]);
                    P_equal_denominator_array.at(c).push_back(P_denom_scratch[c]);
                    P_equal_sumsq_array.at(c).push_back(P_sumsq_scratch[c]);
                    for (float v: dr_scratch[c])
                        dr_array.at(c).push_back(v);
                }

            } else {

                // -------- FALLBACK PATH: brute-force per-cell slab test (non-lattice grids) --------

                // CPU-based voxel intersection with hit_location classification
                std::vector<float> dr(Nhits, 0.0f);
                std::vector<uint> hit_location(Nhits, 0);

                // Process each voxel
                for (uint c = 0; c < Ncells; c++) {

                    helios::vec3 center = getCellCenter(c);
                    helios::vec3 size = getCellSize(c);
                    float rotation = getCellRotation(c);
                    helios::vec3 anchor = getCellGlobalAnchor(c); // rotate about the grid anchor (matches calculateHitGridCell())

                    // Reset for this voxel
                    std::fill(dr.begin(), dr.end(), 0.0f);
                    std::fill(hit_location.begin(), hit_location.end(), 0);

// Test each hit against this voxel (CPU/OpenMP)
#pragma omp parallel for
                    for (int i = 0; i < static_cast<int>(Nhits); i++) {
                        helios::vec3 hit_xyz = this_scan_xyz[i];
                        helios::vec3 origin = this_scan_origin[i]; // this beam's emission origin (per-pulse for moving scans)

                        // Inverse rotate if needed. Apply the same inverse rotation (about the grid anchor) to BOTH
                        // the hit point and the beam origin so the ray-voxel geometry stays consistent with hit binning.
                        if (fabs(rotation) > 1e-6f) {
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
                        accumulateBeamCell(&beam_members_local[beams.beam_offsets[k]], beams.beamSize(k), dr, hit_location, P_equal_numerator, P_equal_denominator, P_equal_sumsq, dr_array.at(c));
                    }

                    P_equal_numerator_array.at(c).push_back(P_equal_numerator);
                    P_equal_denominator_array.at(c).push_back(P_equal_denominator);
                    P_equal_sumsq_array.at(c).push_back(P_equal_sumsq);
                }
            }
        }

        // Obtain G(theta) per voxel. Normally computed from triangulation; when the caller supplied G(theta) (e.g. for a
        // moving-platform scan that cannot be triangulated, or a prescribed leaf-angle distribution) use that instead -
        // a single supplied value is broadcast to every voxel, a per-voxel vector is used as-is. The public overloads
        // validate the length (1 or Ncells) and range (0,1] before calling this; assert the size invariant here.
        std::vector<float> Gtheta;
        if (use_supplied_Gtheta) {
            if (supplied_Gtheta.size() == 1) {
                Gtheta.assign(Ncells, supplied_Gtheta[0]);
            } else {
                assert(supplied_Gtheta.size() == Ncells);
                Gtheta = supplied_Gtheta;
            }
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

bool LiDARcloud::isGPUAvailable() const {
    return CollisionDetection::isGPUAvailable(); // static - valid even before the instance exists
}

bool LiDARcloud::isGPUAccelerationEnabled() const {
    // collision_detection is created lazily; before then the effective state is the
    // constructor default, which is isGPUAvailable().
    return collision_detection != nullptr ? collision_detection->isGPUAccelerationEnabled() : CollisionDetection::isGPUAvailable();
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

    const size_t N = scan_indices.size();

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
        // Each hit is its own beam. CSR layout: members are the scan indices in order, offsets are 0,1,2,...,N.
        result.Nbeams = (uint) N;
        result.beam_members = scan_indices;
        result.beam_offsets.resize(N + 1);
        for (size_t i = 0; i <= N; i++) {
            result.beam_offsets[i] = (uint) i;
        }
        return result;
    }

    // Cache each hit's timestamp once (getHitData is a per-call lookup; the sort below would otherwise
    // call it O(N log N) times). Sort indices by timestamp so returns of the same pulse are contiguous.
    std::vector<double> timestamps(N);
    for (size_t i = 0; i < N; i++) {
        timestamps[i] = getHitData(scan_indices[i], "timestamp");
    }
    std::vector<uint> order(N);
    for (size_t i = 0; i < N; i++) {
        order[i] = (uint) i;
    }
    std::sort(order.begin(), order.end(), [&](uint a, uint b) { return timestamps[a] < timestamps[b]; });

    // Build the CSR layout in one pass: members are the timestamp-sorted scan indices, and a new beam
    // starts wherever the timestamp changes.
    result.beam_members.resize(N);
    result.beam_offsets.clear();
    result.beam_offsets.push_back(0);
    double previous_time = 0.0;
    for (size_t i = 0; i < N; i++) {
        uint si = order[i];
        result.beam_members[i] = scan_indices[si];
        if (i == 0) {
            previous_time = timestamps[si];
        } else if (timestamps[si] != previous_time) {
            result.beam_offsets.push_back((uint) i);
            previous_time = timestamps[si];
        }
    }
    result.beam_offsets.push_back((uint) N);
    result.Nbeams = (uint) result.beam_offsets.size() - 1;

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

namespace {
    //! Detect discrete returns from one pulse's sub-ray hits using an analytic (sparse sum-of-Gaussians) waveform model.
    /**
     * The sorted sub-ray hits are treated as a waveform formed by summing, for each hit, a Gaussian of range-extent
     * \p range_resolution and amplitude (weight * cos). Hits separated by less than \p range_resolution are unresolved and
     * merge into a single return at the energy-weighted range, so two surfaces within one pulse width blend into one
     * "ghost"/"mixed pixel" return at an intermediate range. Each output row is
     * {distance, intensity, nPulseHit, IDmap, echo_width}:
     *   - distance:   energy-weighted mean range of the return's members,
     *   - intensity:  range-normalized echo amplitude = sum(weight*cos) / total_pulse_weight (a fraction of total beam
     *                 energy); an all-miss beam carries the historical sentinel 1.0 and a partial-miss tail 0.0,
     *   - nPulseHit:  integer count of sub-rays in the return,
     *   - IDmap:      primitive identifier (taken from the last member; identifiers are never averaged),
     *   - echo_width: sqrt(range_resolution^2 + weighted range variance), the pulse-width-convolved range spread.
     * \p t_pulse rows are {t, cos, ID, weight} and are sorted in place by range. Returns whose |intensity| is below
     * \p detection_threshold are discarded (miss sentinels are retained so transmitted beams remain available for leaf-area
     * inversion). \p max_returns caps how many returns are reported per pulse: <= 0 means unlimited (report every detected
     * return; the discrete multi-return instrument), while N >= 1 keeps at most N real returns selected by
     * \p single_return_selection (N=1 reproduces classic single-return, N=2 dual-return, etc.). The kept returns are always
     * re-ordered nearest-first. When a pulse produces only misses (a fully transmitted beam) a single miss sentinel is kept
     * regardless of the cap; when real returns exist any miss sentinels are dropped, so a limited-mode pulse reports only
     * real points (matching a real discrete-return instrument).
     */
    std::vector<std::vector<float>> detectReturnsFromSubrays(std::vector<std::vector<float>> &t_pulse, float total_pulse_weight, int Npulse, float range_resolution, float detection_threshold, int max_returns,
                                                             SingleReturnSelection single_return_selection, float miss_distance) {

        std::vector<std::vector<float>> t_hit;
        if (t_pulse.empty()) {
            return t_hit;
        }

        std::sort(t_pulse.begin(), t_pulse.end(), [](const std::vector<float> &a, const std::vector<float> &b) { return a[0] < b[0]; });

        // Total emitted beam energy used to normalize intensity into an energy fraction. Falls back to the sub-ray count
        // (so equal weights reproduce the historical intensity = sum(cos)/Npulse) if the weight sum is degenerate.
        const float denom = (total_pulse_weight > 0.f) ? total_pulse_weight : float(Npulse);

        // Group sorted sub-ray hits into returns. A hit joins the current return if it lies within range_resolution of the
        // return's first (nearest) member; otherwise it opens a new return. This is the peak-resolution of the waveform:
        // members within one pulse range-extent are unresolved and merge into one (blended) return.
        size_t i = 0;
        while (i < t_pulse.size()) {
            const float t0 = t_pulse[i][0];
            double sum_w = 0.0, sum_wt = 0.0, sum_wt2 = 0.0, sum_wcos = 0.0;
            int count = 0;
            float lastID = t_pulse[i][2];
            size_t j = i;
            while (j < t_pulse.size() && (t_pulse[j][0] - t0) <= range_resolution) {
                const float t = t_pulse[j][0];
                const float cosval = t_pulse[j][1];
                const float w = t_pulse[j][3];
                sum_w += w;
                sum_wt += double(w) * t;
                sum_wt2 += double(w) * double(t) * t;
                sum_wcos += double(w) * cosval;
                lastID = t_pulse[j][2];
                count++;
                j++;
            }
            i = j;

            const bool is_miss = (t0 >= 0.98f * miss_distance);
            const float distance = (sum_w > 0.0) ? float(sum_wt / sum_w) : t0;
            float intensity;
            float echo_width;
            if (is_miss) {
                // Pure-miss cluster (misses sit at miss_distance and never group with real hits). Preserve the historical
                // sentinel: a fully transmitted beam (every sub-ray missed) is flagged with intensity 1, a partial-miss tail 0.
                intensity = (count == Npulse) ? 1.0f : 0.0f;
                echo_width = 0.f;
            } else {
                intensity = float(sum_wcos / denom);
                double var = (sum_w > 0.0) ? (sum_wt2 / sum_w - double(distance) * distance) : 0.0;
                if (var < 0.0) {
                    var = 0.0; // guard against round-off for a single-member return
                }
                echo_width = sqrtf(range_resolution * range_resolution + float(var));
            }

            t_hit.push_back({distance, intensity, float(count), lastID, echo_width});
        }

        // Detection threshold (noise floor): discard real returns whose echo amplitude is too weak to be detected. Miss
        // sentinels are always retained so transmitted beams remain recorded for leaf-area inversion.
        if (detection_threshold > 0.f) {
            std::vector<std::vector<float>> kept;
            kept.reserve(t_hit.size());
            for (const auto &h: t_hit) {
                const bool h_is_miss = (h[0] >= 0.98f * miss_distance);
                if (h_is_miss || fabsf(h[1]) >= detection_threshold) {
                    kept.push_back(h);
                }
            }
            t_hit.swap(kept);
        }

        // Limited-return mode (max_returns >= 1): keep at most max_returns real returns per pulse, selected by the policy.
        // max_returns <= 0 is unlimited (discrete multi-return): every detected return is reported, including any miss
        // sentinel alongside real returns, so this path is left untouched. For the limited case the kept real returns are
        // re-ordered nearest-first so the downstream target_index assignment stays in range order. Miss handling: a pulse
        // with no real return keeps a single miss sentinel (a transmitted beam, needed for leaf-area inversion); a pulse
        // with real returns drops the miss sentinel so only real points are reported (matching a real discrete instrument).
        // The SINGLE_RETURN_STRONGEST_PLUS_LAST policy is special: it reports the strongest-plus-last pair (1 or 2 returns)
        // and ignores max_returns entirely.
        if (max_returns > 0) {
            std::vector<size_t> real_idx;
            real_idx.reserve(t_hit.size());
            for (size_t k = 0; k < t_hit.size(); k++) {
                const bool h_is_miss = (t_hit[k][0] >= 0.98f * miss_distance);
                if (!h_is_miss) {
                    real_idx.push_back(k);
                }
            }

            if (real_idx.empty()) {
                // No real return: keep exactly the first (nearest) miss sentinel so the transmitted beam is still recorded.
                if (t_hit.size() > 1) {
                    std::vector<std::vector<float>> only_miss{t_hit[0]};
                    t_hit.swap(only_miss);
                }
            } else if (single_return_selection == SINGLE_RETURN_STRONGEST_PLUS_LAST) {
                // Strongest-plus-last dual return: keep the strongest echo and the last (farthest) return of the pulse,
                // deduplicated to one point when they coincide. This is "pick these two specific returns", not a top-N
                // ranking, so it bypasses the partial_sort path and ignores max_returns (it yields 1 or 2 returns).
                // real_idx is ascending by range, so its last entry is the farthest (last) return.
                const size_t last_idx = real_idx.back();
                size_t strongest_idx = real_idx.front();
                for (const size_t k: real_idx) {
                    if (fabsf(t_hit[k][1]) > fabsf(t_hit[strongest_idx][1])) {
                        strongest_idx = k;
                    }
                }
                std::vector<size_t> ranked;
                if (strongest_idx == last_idx) {
                    ranked = {last_idx}; // strongest IS the last return: report a single point, not a doubled one.
                } else {
                    ranked = {strongest_idx, last_idx};
                    // Re-order the kept pair nearest-first so target_index downstream stays in range order.
                    std::sort(ranked.begin(), ranked.end(), [&](size_t a, size_t b) { return t_hit[a][0] < t_hit[b][0]; });
                }
                std::vector<std::vector<float>> kept;
                kept.reserve(ranked.size());
                for (const size_t k: ranked) {
                    kept.push_back(t_hit[k]);
                }
                t_hit.swap(kept);
            } else {
                std::vector<size_t> ranked = real_idx; // misses are never selectable
                if (int(ranked.size()) > max_returns) {
                    // Rank the real returns by the selection policy and keep the strongest/nearest/farthest max_returns.
                    auto better = [&](size_t a, size_t b) {
                        if (single_return_selection == SINGLE_RETURN_STRONGEST) {
                            return fabsf(t_hit[a][1]) > fabsf(t_hit[b][1]);
                        } else if (single_return_selection == SINGLE_RETURN_FIRST) {
                            return t_hit[a][0] < t_hit[b][0];
                        } else { // SINGLE_RETURN_LAST
                            return t_hit[a][0] > t_hit[b][0];
                        }
                    };
                    std::partial_sort(ranked.begin(), ranked.begin() + max_returns, ranked.end(), better);
                    ranked.resize(size_t(max_returns));
                    // Re-order the kept subset nearest-first so target_index downstream stays in range order.
                    std::sort(ranked.begin(), ranked.end(), [&](size_t a, size_t b) { return t_hit[a][0] < t_hit[b][0]; });
                }
                // ranked now holds the kept real returns in ascending range order (real_idx was already ascending).
                std::vector<std::vector<float>> kept;
                kept.reserve(ranked.size());
                for (const size_t k: ranked) {
                    kept.push_back(t_hit[k]);
                }
                t_hit.swap(kept);
            }
        }

        return t_hit;
    }
} // namespace

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

void LiDARcloud::syntheticScan(helios::Context *context, int rays_per_pulse, float pulse_distance_threshold, ReturnMode return_mode, bool scan_grid_only, bool record_misses, bool append) {
    // Apply the requested return mode to every scan for the duration of this call, then restore the stored per-scan values.
    // The master implementation below reads the return mode (and the other waveform parameters) from each scan's metadata,
    // so this lets a caller select the mode at the call site without permanently mutating the scan configuration.
    std::vector<ReturnMode> saved_modes(scans.size());
    for (size_t s = 0; s < scans.size(); s++) {
        saved_modes[s] = scans[s].returnMode;
        scans[s].returnMode = return_mode;
    }
    syntheticScan(context, rays_per_pulse, pulse_distance_threshold, scan_grid_only, record_misses, append);
    for (size_t s = 0; s < scans.size(); s++) {
        scans[s].returnMode = saved_modes[s];
    }
}

void LiDARcloud::syntheticScan(helios::Context *context, int rays_per_pulse, float pulse_distance_threshold, bool scan_grid_only, bool record_misses, bool append) {

    // Clear existing hit data if not appending
    if (!append) {
        clearHits();
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

    helios::ProgressBar progress_bar(getScanCount(), 50, getScanCount() > 1 && printmessages, "Synthetic scan");
    if (progress_callback) {
        progress_bar.setCallback(progress_callback);
    }

    if (synthetic_scan_progress != nullptr) {
        *synthetic_scan_progress = 0;
    }

    for (int s = 0; s < getScanCount(); s++) {

        // Surface the current scan index on the caller-registered polling counter (if any) at the very top of the loop,
        // before any early-continue, so a host thread watching it advances even for scans whose rays miss the scene.
        if (synthetic_scan_progress != nullptr) {
            *synthetic_scan_progress = s;
        }

        // Report progress at the start of each scan iteration. Using the absolute step (number of scans already
        // completed) keeps the callback firing on every code path through the loop body, including the early
        // "no rays hit the bounding box" continue below; finish() after the loop clamps the bar to 100%.
        progress_bar.update(static_cast<size_t>(s));

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
        // roll = pitch = azimuth = 0. When phiMin - azimuth = 0 the azimuth-zero direction is +y, so X_body = +x and
        // Y_body = +y (tilt reduces to roll about world-x, pitch about world-y).
        float scanTiltRoll = getScanTiltRoll(s);
        float scanTiltPitch = getScanTiltPitch(s);
        float scanAzimuthOffset = getScanAzimuthOffset(s);
        bool apply_azimuth = (scanAzimuthOffset != 0.f);
        bool apply_tilt = (scanTiltRoll != 0.f || scanTiltPitch != 0.f);
        const helios::vec3 tilt_pivot = helios::make_vec3(0, 0, 0); // rotate directions about the origin (pure rotation of the unit vector)
        const helios::vec3 vertical_axis = helios::make_vec3(0.f, 0.f, 1.f); // world +z: azimuth (yaw) rotation axis
        // Body frame from the azimuth-zero (phiMin) direction, offset by the scanner heading. sphere2cart with zero
        // elevation gives the horizontal heading; the azimuth offset then rotates that heading about world +z by the
        // SAME right-hand rotation applied to the ray directions below. Because phi is measured CW-from-+y while the
        // offset is a CCW (right-hand) rotation about +z, advancing the heading by the offset SUBTRACTS it from the
        // phi-angle: rotating (sin phiMin, cos phiMin) CCW by az gives (sin(phiMin-az), cos(phiMin-az)). Using +az here
        // reflected the body frame about the un-offset heading, so any tilt leaned the wrong way once a heading was set.
        const float heading = phimin - scanAzimuthOffset;
        const helios::vec3 forward_axis = helios::make_vec3(sinf(heading), cosf(heading), 0.f); // Y_body: azimuth-zero heading (after offset)
        const helios::vec3 lateral_axis = helios::make_vec3(cosf(heading), -sinf(heading), 0.f); // X_body = Y_body x (0,0,1)

        // Scan pattern determines how the (theta-index, phi-index) grid maps to zenith angles. For a raster scan the zenith is
        // uniformly spaced over [thetamin,thetamax]; for a spinning multibeam scan each theta-index is a laser channel fired at
        // its own fixed (generally non-uniform) zenith angle. Both patterns share the same azimuth sweep and grid storage.
        const ScanMetadata &scan = scans.at(s);
        const bool spinning_multibeam = (scan.scanPattern == SCAN_PATTERN_SPINNING_MULTIBEAM);
        // A Risley-prism (Livox-style rosette) scan is non-separable: each column is one pulse whose body-frame direction comes
        // from the rotating prism optics at that pulse's time. It is stored as a single row (Ntheta=1), so the (i,j) loop below
        // visits one direction per column; the per-pulse direction replaces the raster/spinning theta-phi computation.
        const bool risley = (scan.scanPattern == SCAN_PATTERN_RISLEY_PRISM);

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
        // A continuously-spinning multibeam scan samples a periodic azimuth: the columns are uniformly spaced with no
        // duplicated wrap column, so dphi = (phimax-phimin)/Nphi (exclusive endpoint). This matches the periodic
        // convention already used by ScanMetadata::rc2direction()/direction2rc(). A raster scan samples inclusive
        // endpoints (dphi = (phimax-phimin)/(Nphi-1)); dphi is the column-to-column azimuth step and also sets the
        // continuous-azimuth drift applied within each column below.
        const float dphi = spinning_multibeam ? (phimax - phimin) / float(Nphi) : ((Nphi > 1) ? (phimax - phimin) / float(Nphi - 1) : 0.f);
        const float dtheta = (Ntheta > 1) ? (thetamax - thetamin) / float(Ntheta - 1) : 0.f;

        // Continuous-azimuth (skewed-column) raster sweep. A real terrestrial scanner sweeps the beam vertically with a fast
        // mirror while the entire head rotates continuously in azimuth, so the azimuth advances during each zenith sweep and the
        // zenith columns are slightly skewed (tilted) rather than perfectly vertical. We model this for the raster pattern by
        // drifting the azimuth across the inner (zenith) loop: over one full column (i = 0..Ntheta-1) the azimuth advances by
        // exactly one column step dphi, so row i of column j ends where row 0 of column j+1 begins (seamless continuous rotation,
        // no azimuth gaps or overlap). The per-row increment is therefore dphi/Ntheta. This drift is intentionally NOT reflected
        // in the nominal ScanMetadata::rc2direction()/direction2rc() grid mapping: it is sub-cell (dphi/Ntheta per step) and
        // hit-point binning uses the nominal grid. The skew is disabled for spinning multibeam (all channels in a column fire at
        // one azimuth; azimuth steps between firings) and for Risley (non-separable, no zenith/azimuth grid).
        const float dphi_per_row = (!spinning_multibeam && !risley && Ntheta > 0) ? dphi / float(Ntheta) : 0.f;

        for (uint j = 0; j < Nphi; j++) {
            float phi = phimin + float(j) * dphi;
            for (uint i = 0; i < Ntheta; i++) {
                helios::vec3 dir;
                if (risley) {
                    // Body-frame beam direction from the rotating prism optics at this pulse's time. With Ntheta=1 the column j
                    // is the pulse index; the direction then flows into the same is_moving composition as every other pattern.
                    dir = risleyBodyDirection(scan, j);
                } else {
                    float theta_z = spinning_multibeam ? scan.beamZenithAngles.at(i) : (thetamin + float(i) * dtheta);
                    float theta_elev = 0.5f * M_PI - theta_z;
                    // Skewed azimuth for the raster sweep (dphi_per_row is zero for spinning multibeam, leaving fixed-azimuth columns).
                    float phi_skew = phi + float(i) * dphi_per_row;
                    dir = sphere2cart(helios::make_SphericalCoord(1.f, theta_elev, phi_skew));
                }
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
                    data["echo_width"] = 0.0; // no detectable echo for a full miss
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

        float exit_diameter = getScanBeamExitDiameter(s);
        float beam_divergence = getScanBeamDivergence(s);
        float range_noise_stddev = getScanRangeNoiseStdDev(s);
        float angle_noise_stddev = getScanAngleNoiseStdDev(s);

        // Analytic-waveform return-detection parameters for this scan (see ScanMetadata). The range resolution that merges
        // sub-ray hits into discrete returns is the scan's pulse width when set, otherwise the pulse_distance_threshold
        // argument (preserving the historical behavior of the multi-return overloads).
        const ReturnMode return_mode = scans.at(s).returnMode;
        const SingleReturnSelection single_return_selection = scans.at(s).singleReturnSelection;
        // Effective per-pulse return cap: RETURN_MODE_MULTI reports every detected return (unlimited, signalled by 0);
        // RETURN_MODE_SINGLE limits to maxReturns real returns (1 = single-return, 2 = dual-return, N = N-return).
        const int max_returns = (return_mode == RETURN_MODE_MULTI) ? 0 : scans.at(s).maxReturns;
        const float detection_threshold = scans.at(s).detectionThreshold;
        const float range_resolution = (scans.at(s).pulseWidth > 0.f) ? scans.at(s).pulseWidth : pulse_distance_threshold;

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

        // Determine the beam-chunk size that keeps the live ray-tracing scratch buffers near the configured memory budget.
        // The per-beam fan-out into Npulse sub-rays dominates peak memory: each sub-ray costs BYTES_PER_SUBRAY across the
        // direction/origin/weight/result/SoA buffers held live during the trace. Processing beams in chunks of chunk_beams
        // bounds peak to ~chunk_beams*Npulse*BYTES_PER_SUBRAY regardless of N. The chunk is floored so each trace batch is
        // large enough to be efficient (and to reach the collision-detection GPU path when available) and clamped to N so a
        // small scan runs in a single chunk (preserving the original single-batch behavior and RNG draw order).
        constexpr size_t BYTES_PER_SUBRAY = 40; // direction(12)+origin(12)+weight(4)+hit_t/fnorm/ID(12) + SoA uuid/normal scratch
        constexpr size_t MIN_RAYS_PER_CHUNK = 1050000; // keep batches >= ~1M rays (collision-detection GPU threshold)
        // Resolve the effective budget. When the user has not set one (auto), default to a larger cap on the GPU path than
        // the CPU path, since the GPU ray-tracer handles bigger batches efficiently; users can lower it via
        // setSyntheticScanMemoryBudget(). Initialize the collision engine first so its GPU-enabled state is known here
        // (idempotent; prepareUnifiedRayTracing() below re-uses the same engine).
        initializeCollisionDetection(context);
        size_t effective_budget_bytes = synthetic_scan_memory_budget_bytes;
        if (effective_budget_bytes == 0) {
            effective_budget_bytes = collision_detection->isGPUAccelerationEnabled() ? SYNTHETIC_SCAN_DEFAULT_BUDGET_GPU : SYNTHETIC_SCAN_DEFAULT_BUDGET_CPU;
        }
        size_t target_subrays = effective_budget_bytes / BYTES_PER_SUBRAY;
        size_t chunk_beams = target_subrays / size_t(Npulse);
        size_t min_chunk_beams = (MIN_RAYS_PER_CHUNK + size_t(Npulse) - 1) / size_t(Npulse);
        if (chunk_beams < min_chunk_beams) {
            chunk_beams = min_chunk_beams;
        }
        if (chunk_beams < 1) {
            chunk_beams = 1;
        }
        if (chunk_beams > N) {
            chunk_beams = N;
        }

        // Per-scan running totals (accumulated across all chunks).
        size_t Nhits = 0;
        size_t beams_with_zero_hits = 0;
        size_t beams_with_one_hit = 0;
        size_t beams_with_multi_hits = 0;

        // Prepare the collision-detection engine once for the whole scan so the BVH is built a single time rather than once
        // per chunk (geometry is static during a scan). Paired with finishUnifiedRayTracing() after the chunk loop.
        prepareUnifiedRayTracing(context);

        // Per-chunk scratch buffers, sized for the largest chunk (chunk_beams*Npulse) and reused across chunks. These are the
        // buffers whose N*Npulse sizing previously drove the memory explosion; bounding them to chunk_beams*Npulse is the fix.
        const size_t chunk_capacity = chunk_beams * size_t(Npulse);
        helios::vec3 *direction = (helios::vec3 *) malloc(chunk_capacity * sizeof(helios::vec3));
        helios::vec3 *ray_origins = (helios::vec3 *) malloc(chunk_capacity * sizeof(helios::vec3));
        float *hit_t = (float *) malloc(chunk_capacity * sizeof(float));
        float *hit_fnorm = (float *) malloc(chunk_capacity * sizeof(float));
        int *hit_ID = (int *) malloc(chunk_capacity * sizeof(int));
        std::vector<float> subray_weight(chunk_capacity, 1.0f);
        if (direction == nullptr || ray_origins == nullptr || hit_t == nullptr || hit_fnorm == nullptr || hit_ID == nullptr) {
            helios_runtime_error("ERROR (LiDARcloud::syntheticScan): failed to allocate ray-tracing scratch buffers for a beam chunk of " + std::to_string(chunk_capacity) + " sub-rays. Lower the synthetic-scan memory budget (setSyntheticScanMemoryBudget) or reduce rays_per_pulse.");
        }

        // Pre-warm the texture color cache (serial) so the parallelized per-beam post-processing below performs only
        // read-only lookups into it. load_texture_colors() lazily inserts into a shared std::map, which would be a data
        // race under OpenMP; warming every used texture up front makes the in-loop sample_hit_color() calls race-free.
        // Untextured scenes collect no filenames, so this is a no-op.
        {
            std::set<std::string> warm_tex_files;
            std::vector<uint> all_uuids = context->getAllUUIDs();
            for (uint warm_uuid: all_uuids) {
                std::string tf = context->getPrimitiveTextureFile(warm_uuid);
                if (!tf.empty()) {
                    warm_tex_files.insert(tf);
                }
            }
            for (const std::string &tf: warm_tex_files) {
                load_texture_colors(tf);
            }
        }

        // Salt for the per-beam range-noise RNG seeds (only used when range noise is enabled). Drawn once from the Context
        // RNG so the now thread-parallel range noise stays controllable via Context::seedRandomGenerator and reproducible;
        // each beam's seed = salt + global beam index, making the realization independent of thread count / scheduling.
        const unsigned int range_noise_seed_salt = (range_noise_stddev > 0.f) ? static_cast<unsigned int>(context->randu() * 4294967000.0f) : 0u;

        // Per-beam post-processing outputs, collected in parallel then appended to the point cloud in beam order below.
        struct SyntheticBeamHit {
            helios::vec3 xyz;
            helios::RGBcolor color;
            std::map<std::string, double> data;
        };
        struct SyntheticBeamOutput {
            std::vector<SyntheticBeamHit> hits;
            helios::SphericalCoord dir_sph;
            int npulse_hits = 0; // number of sub-ray hits forming this beam's waveform (drives the zero/one/multi counters)
        };

        // Constants/helpers for the stratified Gaussian footprint sampler (used by the per-sub-ray generation below).
        // GOLDEN_ANGLE = pi*(3 - sqrt(5)): the azimuthal increment that spreads successive sub-rays maximally evenly
        // around the footprint (Vogel/sunflower spiral), shared by the divergence-cone and exit-aperture samplers.
        constexpr float GOLDEN_ANGLE = 2.39996322972865332f; // pi*(3 - sqrt(5))
        // Base-2 radical inverse (van der Corput). The aperture radial strata are driven by this low-discrepancy
        // sequence while the divergence radial strata use linear strata; using two different sequences decorrelates
        // the two independent Gaussian radial dimensions so the joint footprint stays an even disk (no diagonal smear).
        auto radicalInverse2 = [](uint32_t i) -> float {
            i = (i << 16) | (i >> 16);
            i = ((i & 0x55555555u) << 1) | ((i & 0xAAAAAAAAu) >> 1);
            i = ((i & 0x33333333u) << 2) | ((i & 0xCCCCCCCCu) >> 2);
            i = ((i & 0x0F0F0F0Fu) << 4) | ((i & 0xF0F0F0F0u) >> 4);
            i = ((i & 0x00FF00FFu) << 8) | ((i & 0xFF00FF00u) >> 8);
            return float(i) * 2.3283064365386963e-10f; // / 2^32
        };

        for (size_t chunk_begin = 0; chunk_begin < N; chunk_begin += chunk_beams) {
            // Cancellation checkpoint between chunks. castRaysUnified() already short-circuits an in-flight trace to all
            // misses when the flag is set, but without breaking here the loop would still generate rays and run the
            // waveform reduction for every remaining chunk (and, with record_misses=true, record a full grid of miss
            // points). Breaking stops further work so the scan aborts promptly with whatever was recorded so far; the
            // buffer free and finishUnifiedRayTracing() after the loop still run, and the outer scan loop exits below.
            if (cancel_flag != nullptr && *cancel_flag != 0) {
                break;
            }

            const size_t chunk_end = std::min(chunk_begin + chunk_beams, N);
            const size_t chunk_N = chunk_end - chunk_begin; // beams in this chunk

            // Generate this chunk's chunk_N*Npulse sub-ray directions, origins, and (unit) footprint weights. The
            // expensive part is the per-sub-ray trigonometry (cart2sphere/sphere2cart/cos/sin/sqrt/normalize), which is
            // parallelized over beams below.
            //
            // The beam footprint is importance-sampled: rather than drawing sub-rays uniformly over a disk and then
            // re-weighting each by a Gaussian envelope (which wastes the outer, near-zero-weight rays and hard-truncates
            // the beam at its 1/e^2 radius), each radial offset is drawn FROM the Gaussian via the Rayleigh inverse-CDF
            // -- the Gaussian irradiance profile I(r) ~ exp(-2 (r/r0)^2) now lives in the sample DENSITY, so every
            // sub-ray carries unit weight and the beam wings are sampled in proportion to their energy. Coverage is made
            // even and stripe-free with stratification: linear radial strata + golden-angle azimuth for the divergence
            // cone, and a base-2 van der Corput radial sequence + golden-angle azimuth for the exit aperture (the two
            // independent radial dimensions use different sequences so the joint footprint does not develop a diagonal
            // artifact). The per-beam stratum offsets are Cranley-Patterson rotations (one random radial shift + one
            // random azimuth base per beam) so the pattern is decorrelated pulse-to-pulse rather than a fixed lattice.
            //
            // The random per-beam offsets are drawn FIRST, serially, into divergence_rand/aperture_rand because the
            // generation loop below is OpenMP-parallel and Context's mt19937 (context->randu()) is not thread-safe;
            // pre-drawing also keeps a seeded Context (Context::seedRandomGenerator) reproducible independent of thread
            // count/scheduling. NOTE: this sampler intentionally changes the RNG consumption (two draws per BEAM, not per
            // sub-ray) and the sub-ray pattern relative to the previous uniform+weight sampler -- it is NOT bit-identical
            // to the old output by design. subray_weight stays at its initialized value of 1 in every new path; the array
            // and its plumbing are retained as a fallback for future non-Gaussian beam profiles that need explicit weights.
            const bool draw_divergence = (beam_divergence != 0.0f && Npulse > 1);
            const bool draw_aperture = (exit_diameter > 0.0f);

            // Serial RNG pre-draw: two values per BEAM for each enabled footprint dimension -- a radial stratum shift and
            // an azimuth base rotation (the divergence azimuth and aperture azimuth use independent rotations, so only the
            // radial dimension needs the separate van der Corput sequence to decorrelate). Buffers stay empty when no
            // sampling is needed (rays_per_pulse==1, zero divergence, point source).
            std::vector<float> divergence_rand, aperture_rand;
            if (draw_divergence) {
                divergence_rand.resize(chunk_N * 2, 0.f);
                for (size_t local = 0; local < chunk_N; local++) {
                    divergence_rand[local * 2] = context->randu(); // xi_r: radial stratum jitter
                    divergence_rand[local * 2 + 1] = 2.0f * float(M_PI) * context->randu(); // phi0: azimuth base rotation
                }
            }
            if (draw_aperture) {
                aperture_rand.resize(chunk_N * 2, 0.f);
                for (size_t local = 0; local < chunk_N; local++) {
                    aperture_rand[local * 2] = context->randu(); // xi_ap: aperture radial Cranley-Patterson offset
                    aperture_rand[local * 2 + 1] = 2.0f * float(M_PI) * context->randu(); // phi0_ap: aperture azimuth rotation
                }
            }
            // Truncate the Gaussian footprint sampling at the detectability radius. The radial irradiance falls off as
            // exp(-2 r^2/r0^2); beyond the radius where it drops below the detection threshold, no return there could
            // clear detectReturnsFromSubrays' detection_threshold, so sampling further out only spends rays and injects
            // tail shot-noise -- with unit weights a lone outer sub-ray would otherwise create a full 1/Npulse-weight
            // return at a location the real (attenuated) beam could never illuminate enough to detect. The radial CDF is
            // F(r) = 1 - exp(-2 r^2/r0^2), so truncating at that radius means stratifying the uniform CDF variate over
            // [0, 1 - thr] rather than [0, 1) -- this also removes the previous hard 3*r0 clamp and its tail pile-up.
            // When no detection threshold is set, trim only the negligible outer 0.1% of beam energy (~1.86 r0) as a safe
            // default. The 0.5 cap keeps the beam core sampled even if an aggressive (>50%) threshold is requested.
            const float footprint_trim = fminf(0.5f, (detection_threshold > 0.f) ? detection_threshold : 1.0e-3f);
            const float u_max_foot = 1.0f - footprint_trim; // upper bound of the kept CDF interval (1 - exp(-2 R_max^2/r0^2))

            const float aperture_radius = 0.5f * exit_diameter;
#pragma omp parallel for schedule(dynamic, 256)
            for (int local = 0; local < static_cast<int>(chunk_N); local++) {
                const size_t global_r = chunk_begin + local; // length-N array index
                const helios::vec3 base_dir = nominal_directions[global_r];
                const helios::vec3 beam_origin = pulse_origin[global_r];

                // Orthonormal disk basis {u, v} perpendicular to the beam, for finite-aperture origins (constant per beam).
                helios::vec3 u, v;
                if (draw_aperture) {
                    const helios::vec3 reference = (fabs(base_dir.z) < 0.9f) ? helios::make_vec3(0, 0, 1) : helios::make_vec3(1, 0, 0);
                    u = helios::cross(base_dir, reference);
                    u.normalize();
                    v = helios::cross(base_dir, u); // already unit (base_dir and u are orthonormal)
                }
                // Beam-axis spherical coordinates for the divergence perturbation (constant per beam).
                const helios::SphericalCoord base_spherical = helios::cart2sphere(base_dir);

                for (int p = 0; p < Npulse; p++) {
                    const size_t idx = local * size_t(Npulse) + size_t(p);
                    float w = 1.0f;

                    // Sub-ray direction (stratified Gaussian-warped, importance-sampled): the first ray and zero-divergence
                    // beams use the nominal axis (exact center ray, no floating-point spread); other sub-rays are drawn
                    // FROM the divergence-cone Gaussian so each carries unit weight.
                    if (p == 0 || beam_divergence == 0.0f) {
                        direction[idx] = base_dir;
                    } else {
                        const float xi_r = divergence_rand[local * 2]; // per-beam radial stratum jitter
                        const float phi0 = divergence_rand[local * 2 + 1]; // per-beam azimuth base rotation
                        const int j = p - 1; // p==0 is reserved for the center ray
                        const int M = Npulse - 1; // number of stratified sub-rays
                        float uu = (float(j) + 0.5f) / float(M) + xi_r; // Cranley-Patterson-rotated radial stratum
                        uu -= floorf(uu); // wrap into [0,1)
                        uu *= u_max_foot; // map onto [0, u_max_foot): truncate the undetectable Gaussian tail (no pile-up)
                        // Rayleigh inverse-CDF for P(R<=theta)=1-exp(-2 theta^2/D^2), D=beam_divergence (the 1/e^2 half-angle):
                        // density ~ Gaussian irradiance, so the Gaussian lives in the sample density and the weight stays 1.
                        // 1 - uu >= footprint_trim > 0, so logf is always finite (no separate guard needed).
                        float theta_offset = beam_divergence * sqrtf(-0.5f * logf(1.0f - uu));
                        const float phi_offset = phi0 + float(j) * GOLDEN_ANGLE; // golden-angle azimuth
                        // Perturb in elevation space (SphericalCoord takes elevation, not zenith).
                        const float new_elevation = base_spherical.elevation + theta_offset * cosf(phi_offset);
                        const float new_azimuth = base_spherical.azimuth + theta_offset * sinf(phi_offset) / fmaxf(cosf(base_spherical.elevation), 1e-6f);
                        helios::vec3 perturbed_dir = helios::sphere2cart(helios::SphericalCoord(1.0f, new_elevation, new_azimuth));
                        perturbed_dir.normalize();
                        direction[idx] = perturbed_dir;
                    }

                    // Sub-ray origin (stratified Gaussian-warped, importance-sampled over the exit aperture): point source
                    // (exit_diameter==0) emits all rays from beam_origin; otherwise origins are drawn FROM the aperture
                    // Gaussian so each carries unit weight. The radial dimension uses a base-2 van der Corput sequence
                    // (radicalInverse2) so it decorrelates from the divergence radial strata above.
                    if (draw_aperture) {
                        if (p == 0) {
                            ray_origins[idx] = beam_origin; // axial ray: aperture center
                        } else {
                            const float xi_ap = aperture_rand[local * 2]; // per-beam radial CP offset
                            const float phi0_ap = aperture_rand[local * 2 + 1]; // per-beam azimuth rotation
                            const int j = p - 1; // mirror the direction sampler's center-ray reservation
                            float s = radicalInverse2((uint32_t)(j + 1)) + xi_ap;
                            s -= floorf(s); // wrap into [0,1)
                            s *= u_max_foot; // truncate the undetectable aperture tail at the same detectability radius (no pile-up)
                            // Rayleigh inverse-CDF, aperture_radius = the 1/e^2 radius (matches the existing convention).
                            float r_sample = aperture_radius * sqrtf(-0.5f * logf(1.0f - s));
                            const float theta = phi0_ap + float(j) * GOLDEN_ANGLE; // golden-angle azimuth
                            const float x_disk = r_sample * cosf(theta);
                            const float y_disk = r_sample * sinf(theta);
                            const helios::vec3 offset = u * x_disk + v * y_disk;
                            ray_origins[idx] = beam_origin + offset;
                        }
                    } else {
                        ray_origins[idx] = beam_origin;
                    }

                    subray_weight[idx] = w; // w == 1 in every path above; Gaussian is now carried by the sample density
                }
            }

            // Trace this chunk's rays into the scratch result buffers (BVH already built by prepareUnifiedRayTracing).
            // The chunk buffers are laid out pulse-contiguously (sub-ray p of beam local at local*Npulse + p), so passing
            // Npulse as the packet size lets the collision layer traverse each pulse's coherent sub-rays together.
            castRaysUnified(chunk_N * size_t(Npulse), ray_origins, direction, hit_t, hit_fnorm, hit_ID, size_t(Npulse));

            // Post-process beams in this chunk. The per-beam waveform reduction, color/primitive-data sampling, and per-hit
            // data-map construction are independent and run in parallel into per-beam output slots; the results are then
            // merged into the shared point cloud (the merge below pre-sizes the columnar storage and scatters hits into
            // distinct rows, so it is also parallel while reproducing the serial recorded values).
            std::vector<SyntheticBeamOutput> beam_outputs(chunk_N);
#pragma omp parallel for schedule(dynamic, 256)
            for (int local = 0; local < static_cast<int>(chunk_N); local++) {
                const size_t r = local; // index into the per-chunk scratch buffers
                const size_t global_r = chunk_begin + local; // index into the length-N beam arrays
                SyntheticBeamOutput &beam_out = beam_outputs[local];

            // Sub-ray hits for this beam, each row {t, cos, ID, weight}. The Gaussian footprint weight is carried through
            // so the analytic-waveform return detection (detectReturnsFromSubrays) can form energy-weighted ranges and
            // intensities. total_pulse_weight is the total emitted beam energy (sum of weights over ALL fired sub-rays,
            // including those that missed), so intensity is reported as a fraction of the whole pulse.
            std::vector<std::vector<float>> t_pulse;
            float total_pulse_weight = 0.f;

            // looping over rays in each beam
            for (size_t p = 0; p < Npulse; p++) {

                float t = hit_t[r * Npulse + p]; // distance to hit (misses t=1001.0)
                float i = hit_fnorm[r * Npulse + p]; // dot product between beam direction and primitive normal
                float ID = float(hit_ID[r * Npulse + p]); // ID of intersected primitive
                float w = subray_weight[r * Npulse + p]; // Gaussian footprint weight

                total_pulse_weight += w;

                if (record_misses || (!record_misses && t < miss_distance)) {
                    std::vector<float> v{t, i, ID, w};
                    t_pulse.push_back(v);
                }
            }

            // Record this beam's sub-ray-hit class; the shared zero/one/multi counters are tallied in the serial merge.
            beam_out.npulse_hits = int(t_pulse.size());

            // Detect discrete returns from the analytic sum-of-Gaussians waveform. Returns rows
            // {distance, intensity, nPulseHit, IDmap, echo_width}.
            std::vector<std::vector<float>> t_hit = detectReturnsFromSubrays(t_pulse, total_pulse_weight, Npulse, range_resolution, detection_threshold, max_returns, single_return_selection, miss_distance);

            // Count non-miss returns for proper target_index assignment
            int non_miss_count = 0;
            for (size_t hit = 0; hit < t_hit.size(); hit++) {
                if (t_hit.at(hit).at(0) < 0.98f * miss_distance) {
                    non_miss_count++;
                }
            }

            // Per-beam range-noise RNG, constructed only when noise is enabled and seeded deterministically by the global
            // beam index so the (thread-parallel) noise is reproducible and independent of thread count. When
            // range_noise_stddev==0 no random numbers are drawn and the output is bit-identical to the serial path.
            std::mt19937 beam_rng;
            std::normal_distribution<float> range_noise_dist;
            if (range_noise_stddev > 0.f) {
                beam_rng.seed(range_noise_seed_salt + static_cast<unsigned int>(global_r));
                range_noise_dist = std::normal_distribution<float>(0.f, range_noise_stddev);
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
                if (!is_miss && range_noise_stddev > 0.f) {
                    measured_distance += range_noise_dist(beam_rng);
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
                // Pulse-shape deviation: a dimensionless measure of how distorted (broadened) this return's pulse is
                // relative to a clean transmit pulse, analogous to the RIEGL "pulse shape deviation" confidence metric
                // (small = clean single-surface return, large = mixed/sloped/blended return). echo_width is the
                // pulse-width-convolved range spread, echo_width = sqrt(range_resolution^2 + var), so sqrt(var) is the
                // excess broadening beyond the transmit pulse and dividing by range_resolution makes it dimensionless.
                // 0 for misses, clean returns, and the degenerate no-pulse-width case (no reference pulse to deviate from).
                const float echo_width = t_hit.at(hit).at(4);
                float deviation = 0.f;
                if (!is_miss && range_resolution > 0.f) {
                    const float excess_var = echo_width * echo_width - range_resolution * range_resolution;
                    deviation = (excess_var > 0.f) ? sqrtf(excess_var) / range_resolution : 0.f;
                }
                data["deviation"] = deviation;
                // Real per-pulse acquisition time. The pulse ordinal is its position in the firing sequence
                // (ordinal = Ntheta*j + i); scaling by pulse_period and offsetting by t0 turns it into seconds. For static
                // scans pulse_period=1 and t0=0, so this equals the historical grid ordinal. All returns of one pulse (this
                // loop over `hit` for a fixed beam r) share the identical time, as required by groupHitsByTimestamp.
                const size_t pulse_ordinal = size_t(pulse_scangrid_ij.at(global_r).y) * Ntheta + size_t(pulse_scangrid_ij.at(global_r).x);
                data["timestamp"] = pulse_t0 + double(pulse_ordinal) * pulse_period;
                // Record range-normalized intensity: the range-independent return amplitude rho*cos(theta) with the
                // 1/R^2 range loss of the LiDAR range equation normalized out (see applyRangeIntensityCorrection()).
                data["intensity"] = applyRangeIntensityCorrection(t_hit.at(hit).at(1), measured_distance);
                data["distance"] = measured_distance;
                data["nRaysHit"] = t_hit.at(hit).at(2);
                // Pulse-width-convolved range spread of the return (the transmit pulse range-extent combined in quadrature
                // with the range spread of the surfaces that merged into this return). 0 for misses and zero-spread beams.
                data["echo_width"] = t_hit.at(hit).at(4);
                if (spinning_multibeam) {
                    data["channel"] = double(pulse_scangrid_ij.at(global_r).x); // laser channel index (scan-table row) that fired this beam
                }

                float UUID = t_hit.at(hit).at(3);

                // Use base direction for this beam (first ray: r*Npulse+0)
                helios::vec3 dir = direction[r * Npulse];
                // Reconstruct the hit point along the beam from its own emission origin (the per-pulse platform origin for
                // moving scans; scan_origin for static scans). For moving scans, record the per-pulse origin and firing index.
                const helios::vec3 beam_origin = pulse_origin[global_r];
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

                    // Sample arbitrary data fields named in the scan's column format onto this hit. The column
                    // format is the source of truth: add a (non-standard) label to the scan's column format and the
                    // scanner copies that data here. Each label is resolved from the hit primitive's own primitive
                    // data first, then (on a miss) from the primitive's parent-object data, so a label may be sourced
                    // from either primitive or object data.
                    for (const std::string &label: column_format) {
                        if (isStandardColumnToken(label)) {
                            continue;
                        }

                        double value;
                        if (!resolveScalarHitData(context, uint(UUID), label, value)) {
                            continue;
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

                // Stage this hit in the beam's output slot; appended to the cloud in beam order in the serial merge below.
                beam_out.dir_sph = helios::cart2sphere(dir);
                beam_out.hits.push_back(SyntheticBeamHit{p, color, std::move(data)});
            }
            } // end per-beam (parallel) loop for this chunk

            // Merge the staged per-beam hits into the shared columnar point-cloud storage. The serial equivalent is a
            // per-hit addHitPoint()/appendHitData() loop whose cost (string-keyed column lookups + per-hit column
            // extension for millions of hits) dominates a GPU-fast scan. Instead: (1) collect the union of hit-data
            // labels and pre-create every column once, (2) pre-size `hits` and all columns to their final length, then
            // (3) scatter each hit into its final row in parallel. Each hit owns a distinct, precomputed row (beam-order
            // prefix sum) and each label a distinct column slot, so the parallel writes are race-free and the recorded
            // values are identical to the serial path. Column slot order is the sorted label union rather than strict
            // first-appearance order; for a synthetic scan every hit carries the same standard label set, so the first
            // hit would create those columns in (alphabetical) std::map order anyway -> identical order in the common
            // case. The only observable effect of the difference is the column order of a no-explicit-columnFormat ASCII
            // export when sparse per-primitive-data labels appear only on later hits; recorded values are unaffected.
            {
                // (1) Union of hit-data labels across this chunk (parallel collect into per-thread sets, then merge).
#ifdef _OPENMP
                std::vector<std::set<std::string>> thread_label_sets(static_cast<size_t>(omp_get_max_threads()));
#else
                std::vector<std::set<std::string>> thread_label_sets(1);
#endif
#pragma omp parallel for schedule(dynamic, 256)
                for (int local = 0; local < static_cast<int>(chunk_N); local++) {
#ifdef _OPENMP
                    std::set<std::string> &my_labels = thread_label_sets[static_cast<size_t>(omp_get_thread_num())];
#else
                    std::set<std::string> &my_labels = thread_label_sets[0];
#endif
                    for (const SyntheticBeamHit &bh: beam_outputs[local].hits) {
                        for (const auto &kv: bh.data) {
                            my_labels.insert(kv.first);
                        }
                    }
                }
                std::set<std::string> all_labels;
                for (const std::set<std::string> &ts: thread_label_sets) {
                    all_labels.insert(ts.begin(), ts.end());
                }

                // Pre-create all columns (sorted order via std::set) and resolve label -> slot once.
                std::unordered_map<std::string, size_t> slot_of;
                for (const std::string &lbl: all_labels) {
                    slot_of[lbl] = getOrCreateHitDataColumn(lbl);
                }

                // (2) Beam-order row offsets + final sizes; tally the beam-class counters in the same pass.
                std::vector<size_t> row_offset(chunk_N + 1, 0);
                for (size_t local = 0; local < chunk_N; local++) {
                    const SyntheticBeamOutput &beam_out = beam_outputs[local];
                    if (beam_out.npulse_hits == 0) {
                        beams_with_zero_hits++;
                    } else if (beam_out.npulse_hits == 1) {
                        beams_with_one_hit++;
                    } else {
                        beams_with_multi_hits++;
                    }
                    row_offset[local + 1] = row_offset[local] + beam_out.hits.size();
                }
                const size_t chunk_hits = row_offset[chunk_N];
                const size_t old_n = hits.size();
                const size_t new_n = old_n + chunk_hits;

                hits.resize(new_n); // default-constructed HitPoints (gridcell=-2), overwritten below
                for (size_t sl = 0; sl < hit_data_columns.size(); sl++) {
                    hit_data_columns[sl].resize(new_n, 0.0); // every column stays length-aligned with `hits`
                    hit_data_present[sl].resize(new_n, char(0));
                }

                // (3) Scatter each beam's hits into their final rows in parallel (distinct rows/slots => race-free).
                const ScanMetadata &scan = scans.at(s);
#pragma omp parallel for schedule(dynamic, 256)
                for (int local = 0; local < static_cast<int>(chunk_N); local++) {
                    SyntheticBeamOutput &beam_out = beam_outputs[local];
                    if (beam_out.hits.empty()) {
                        continue;
                    }
                    const helios::int2 rc = scan.direction2rc(beam_out.dir_sph);
                    const size_t base = old_n + row_offset[local];
                    for (size_t h = 0; h < beam_out.hits.size(); h++) {
                        SyntheticBeamHit &bh = beam_out.hits[h];
                        const size_t row = base + h;
                        HitPoint &hp = hits[row];
                        hp.position = bh.xyz;
                        hp.direction = beam_out.dir_sph;
                        hp.row_column = rc;
                        hp.color = bh.color;
                        hp.scanID = int(s);
                        for (const auto &kv: bh.data) {
                            const size_t sl = slot_of.at(kv.first); // concurrent read-only lookup (no inserts here)
                            hit_data_columns[sl][row] = kv.second;
                            hit_data_present[sl][row] = char(1);
                        }
                    }
                }
                Nhits += chunk_hits;
            }
        } // end chunk loop

        // Restore automatic BVH rebuilds now that this scan's batched ray tracing is complete.
        finishUnifiedRayTracing();

        // Free the per-chunk scratch buffers (allocated once before the chunk loop, reused across chunks).
        free(ray_origins);
        free(direction);
        free(hit_t);
        free(hit_fnorm);
        free(hit_ID);

        if (printmessages) {
            std::cout << "Created synthetic scan #" << s << " with " << Nhits << " hit points." << std::endl;
        }

        // Cancellation checkpoint between scans: this scan's buffers are freed and ray tracing finished above, so a
        // cancelled run stops here and falls through to progress_bar.finish() rather than starting the next scan.
        if (cancel_flag != nullptr && *cancel_flag != 0) {
            break;
        }
    }

    // Signal completion on the polling counter (host also learns this from the call returning).
    if (synthetic_scan_progress != nullptr) {
        *synthetic_scan_progress = getScanCount();
    }

    progress_bar.finish();

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
