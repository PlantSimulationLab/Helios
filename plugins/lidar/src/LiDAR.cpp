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

using namespace std;
using namespace helios;

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
    columnFormat = {"x", "y", "z"};

    data_file = "";
}

ScanMetadata::ScanMetadata(const vec3 &a_origin, uint a_Ntheta, float a_thetaMin, float a_thetaMax, uint a_Nphi, float a_phiMin, float a_phiMax, float a_exitDiameter, float a_beamDivergence, const vector<string> &a_columnFormat) {

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
    columnFormat = a_columnFormat;

    data_file = "";
}

helios::SphericalCoord ScanMetadata::rc2direction(uint row, uint column) const {

    float zenith = thetaMin + (thetaMax - thetaMin) / float(Ntheta) * float(row);
    float elevation = 0.5f * M_PI - zenith;
    float phi = phiMin - (phiMax - phiMin) / float(Nphi) * float(column);
    return make_SphericalCoord(1, elevation, phi);
};

helios::int2 ScanMetadata::direction2rc(const SphericalCoord &direction) const {

    float theta = direction.zenith;
    float phi = direction.azimuth;

    int row = std::round((theta - thetaMin) / (thetaMax - thetaMin) * float(Ntheta));
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

LiDARcloud::LiDARcloud() {

    Nhits = 0;
    hitgridcellcomputed = false;
    triangulationcomputed = false;
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
    const float miss_distance = 1001.0f;

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

void LiDARcloud::enableMessages() {
    printmessages = true;
}

void LiDARcloud::validateRayDirections() {

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

std::vector<std::string> LiDARcloud::getScanColumnFormat(uint scanID) const {
    if (scanID >= scans.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getScanColumnFormat): Cannot get column format for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
    }
    return scans.at(scanID).columnFormat;
}

helios::vec3 LiDARcloud::getHitXYZ(uint index) const {

    if (index >= hits.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getHitXYZ): Hit point index out of bounds. Requesting hit #" + std::to_string(index) + " but scan only has " + std::to_string(hits.size()) + " hits.");
    }

    return hits.at(index).position;
}

helios::SphericalCoord LiDARcloud::getHitRaydir(uint index) const {

    if (index >= hits.size()) {
        helios_runtime_error("ERROR (LiDARcloud::getHitRaydir): Hit point index out of bounds. Requesting hit #" + std::to_string(index) + " but scan only has " + std::to_string(hits.size()) + " hits.");
    }

    vec3 direction_cart = getHitXYZ(index) - getScanOrigin(getHitScanID(index));
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

void LiDARcloud::coordinateShift(const vec3 &shift) {

    for (auto &scan: scans) {
        scan.origin = scan.origin + shift;
    }

    for (auto &hit: hits) {
        hit.position = hit.position + shift;
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
        }
    }
}

void LiDARcloud::coordinateRotation(const SphericalCoord &rotation) {

    for (auto &scan: scans) {
        scan.origin = rotatePoint(scan.origin, rotation);
    }

    for (auto &hit: hits) {
        hit.position = rotatePoint(hit.position, rotation);
        hit.direction = cart2sphere(hit.position - scans.at(hit.scanID).origin);
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
            hit.direction = cart2sphere(hit.position - scans.at(scanID).origin);
        }
    }
}

void LiDARcloud::coordinateRotation(float rotation, const vec3 &line_base, const vec3 &line_direction) {

    for (auto &scan: scans) {
        scan.origin = rotatePointAboutLine(scan.origin, line_base, line_direction, rotation);
    }

    for (auto &hit: hits) {
        hit.position = rotatePointAboutLine(hit.position, line_base, line_direction, rotation);
        hit.direction = cart2sphere(hit.position - scans.at(hit.scanID).origin);
    }
}

uint LiDARcloud::getTriangleCount() const {
    return triangles.size();
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
        uint scanID = getHitScanID(i);
        vec3 r = xyz - getScanOrigin(scanID);

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

void LiDARcloud::maxPulseFilter(const char *scalar) {

    if (printmessages) {
        std::cout << "Filtering point cloud by maximum " << scalar << " per pulse..." << std::flush;
    }

    std::vector<std::vector<double>> timestamps;
    timestamps.resize(getHitCount());

    std::size_t delete_count = 0;
    for (std::size_t r = 0; r < getHitCount(); r++) {

        if (!doesHitDataExist(r, "timestamp")) {
            std::cerr << "failed\nERROR (LiDARcloud::maxPulseFilter): Hit point " << r
                      << " does not have scalar data "
                         "timestamp"
                         ", which is required for max pulse filtering. No filtering will be performed."
                      << std::endl;
            return;
        } else if (!doesHitDataExist(r, scalar)) {
            std::cerr << "failed\nERROR (LiDARcloud::maxPulseFilter): Hit point " << r
                      << " does not have scalar data "
                         ""
                      << scalar
                      << ""
                         ".  No filtering will be performed."
                      << std::endl;
            return;
        }

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
            std::cerr << "failed\nERROR (LiDARcloud::maxPulseFilter): Hit point " << r
                      << " does not have scalar data "
                         "timestamp"
                         ", which is required for max pulse filtering. No filtering will be performed."
                      << std::endl;
            return;
        } else if (!doesHitDataExist(r, scalar)) {
            std::cerr << "failed\nERROR (LiDARcloud::maxPulseFilter): Hit point " << r
                      << " does not have scalar data "
                         ""
                      << scalar
                      << ""
                         ".  No filtering will be performed."
                      << std::endl;
            return;
        }

        std::vector<double> v{getHitData(r, "timestamp"), getHitData(r, scalar), float(r)};

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
        helios_runtime_error("ERROR (LiDARcloud::gapfillMisses): Invalid scanID " + std::to_string(scanID) +
                             ". Only " + std::to_string(getScanCount()) + " scans exist.");
    }

    if (printmessages) {
        std::cout << "Gap filling complete misses in scan " << scanID << "..." << std::flush;
    }

    float gap_distance = 20000;

    helios::vec3 origin = getScanOrigin(scanID);
    std::vector<helios::vec3> xyz_filled;

    // Populating a hit table for each scan:
    // Column 0 - hit index; Column 1 - timestamp; Column 2 - ray zenith; Column 3 - ray azimuth
    std::vector<std::vector<double>> hit_table;
    for (size_t r = 0; r < getHitCount(); r++) {
        if (getHitScanID(r) == scanID) {

            if (add_flags) {
                // gapfillMisses_code = 0: original points
                setHitData(r, "gapfillMisses_code", 0.0);
            }

            helios::SphericalCoord raydir = getHitRaydir(r);

            if (!doesHitDataExist(r, "timestamp")) {
                helios_runtime_error("ERROR (LiDARcloud::gapfillMisses): Hit " + std::to_string(r) +
                                     " is missing required 'timestamp' data. Cannot perform gap filling.");
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
        return xyz_filled;  // Return empty vector
    }

    // sorting, initial dt and dtheta calculations, and determining minimum target index in the scan

    // sort the hit table by column 1 (timestamp)
    std::sort(hit_table.begin(), hit_table.end(), sortcol1);

    int min_tindex = 1;
    for (size_t r = 0; r < hit_table.size() - 1; r++) {

        // this is to figure out if target indexing uses 0 or 1 offset
        if (min_tindex == 1 && doesHitDataExist(hit_table.at(r).at(0), "target_index") && doesHitDataExist(hit_table.at(r).at(0), "target_count")) {
            if (getHitData(hit_table.at(r).at(0), "target_index") == 0) {
                min_tindex = 0;
            }
        }
    }

    // getting rid of points with target index greater than the minimum

    int ndup_target = 0;
    // create new array without duplicate timestamps
    std::vector<std::vector<double>> hit_table_semiclean;
    for (size_t r = 0; r < hit_table.size() - 1; r++) {

        // only consider first hits
        if (doesHitDataExist(hit_table.at(r).at(0), "target_index") && doesHitDataExist(hit_table.at(r).at(0), "target_count")) {
            if (getHitData(hit_table.at(r).at(0), "target_index") > min_tindex) {
                ndup_target++;
                continue;
            }
        }

        hit_table_semiclean.push_back(hit_table.at(r));
    }
    // Add the last element (loop above stops at size()-1)
    if (!hit_table.empty()) {
        hit_table_semiclean.push_back(hit_table.back());
    }

    //  re-calculating dt

    std::vector<double> dt_semiclean;
    dt_semiclean.resize(hit_table_semiclean.size());
    for (size_t r = 0; r < hit_table_semiclean.size() - 1; r++) {

        dt_semiclean.at(r) = hit_table_semiclean.at(r + 1).at(1) - hit_table_semiclean.at(r).at(1);
        // set the hit index of the new array
        hit_table_semiclean.at(r).at(0) = r;
    }

    //  checking for duplicate timestamps in the remaining data

    int ndup = 0;
    // create new array without duplicate timestamps
    std::vector<std::vector<double>> hit_table_clean;
    for (size_t r = 0; r < hit_table_semiclean.size() - 1; r++) {

        // if there are still rows with duplicate timestamps, it probably means there is no "target_index" column, but multiple hits per timestamp are still included
        // proceed using this assumption, just get rid of the rows where dt = 0 for simplicity (last hits probably are what remain).
        if (dt_semiclean.at(r) == 0) {
            ndup++;
            continue;
        }

        hit_table_clean.push_back(hit_table_semiclean.at(r));
    }

    // recalculate dt and dtheta with only one hit per beam
    // and calculate the minimum dt value
    std::vector<double> dt_clean;
    std::vector<float> dtheta_clean;
    dt_clean.resize(hit_table_clean.size());
    dtheta_clean.resize(hit_table_clean.size());

    double dt_clean_min = 1e6;
    for (size_t r = 0; r < hit_table_clean.size() - 1; r++) {

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
    for (size_t r = 0; r < hit_table_clean.size() - 1; r++) {

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
        return xyz_filled;  // Return empty vector
    }

    dt_avg = dt_avg / float(dt_sum);
    // Calculate the average dtheta to use for extrapolation
    dtheta_avg = dtheta_avg / float(dtheta_sum);

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
            for (int i = 0; i < hit_table2D.at(j).size() - 1; i++) {

                double dt = hit_table2D.at(j).at(i + 1).at(1) - hit_table2D.at(j).at(i).at(1);

                if (dt > 1.5f * dt_clean_min) { // missing hit(s)

                    // calculate number of missing hits
                    int Ngap = round(dt / dt_avg) - 1;

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
                        auto grid_key = std::make_pair(rc.x, rc.y);

                        // Only add if this grid position hasn't been filled yet
                        if (filled_positions.find(grid_key) == filled_positions.end()) {

                            helios::SphericalCoord spherical(gap_distance, 0.5 * M_PI - theta, phi);
                            helios::vec3 xyz = origin + helios::sphere2cart(spherical);
                            xyz_filled.push_back(xyz);

                            std::map<std::string, double> data;
                            data.insert(std::pair<std::string, double>("timestamp", timestep));
                            data.insert(std::pair<std::string, double>("target_index", min_tindex));
                            data.insert(std::pair<std::string, double>("nRaysHit", 500));
                            if (add_flags) {
                                // gapfillMisses_code = 1: gapfilled points
                                data.insert(std::pair<std::string, double>("gapfillMisses_code", 1.0));
                            }
                            addHitPoint(scanID, xyz, spherical, data);
                            filled_positions.insert(grid_key);  // Mark as filled
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
                    if (rc.x >= 0 && rc.x < (int)scans.at(scanID).Ntheta &&
                        rc.y >= 0 && rc.y < (int)scans.at(scanID).Nphi) {

                        // Check if this grid position has already been filled (avoid duplicates)
                        auto grid_key = std::make_pair(rc.x, rc.y);
                        if (filled_positions.find(grid_key) == filled_positions.end()) {

                            helios::SphericalCoord spherical(gap_distance, 0.5 * M_PI - theta, phi);
                            helios::vec3 xyz = origin + helios::sphere2cart(spherical);
                            xyz_filled.push_back(xyz);

                            std::map<std::string, double> data;
                            data.insert(std::pair<std::string, double>("timestamp", timestep));
                            data.insert(std::pair<std::string, double>("target_index", min_tindex));
                            data.insert(std::pair<std::string, double>("nRaysHit", 500));
                            if (add_flags) {
                                // gapfillMisses_code = 3: upward edge points
                                data.insert(std::pair<std::string, double>("gapfillMisses_code", 3.0));
                            }

                            addHitPoint(scanID, xyz, spherical, data);
                            filled_positions.insert(grid_key);  // Mark this position as filled
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
                    if (rc.x >= 0 && rc.x < (int)scans.at(scanID).Ntheta &&
                        rc.y >= 0 && rc.y < (int)scans.at(scanID).Nphi) {

                        // Check if this grid position has already been filled (avoid duplicates)
                        auto grid_key = std::make_pair(rc.x, rc.y);
                        if (filled_positions.find(grid_key) == filled_positions.end()) {

                            helios::SphericalCoord spherical(gap_distance, 0.5 * M_PI - theta, phi);
                            helios::vec3 xyz = origin + helios::sphere2cart(spherical);
                            xyz_filled.push_back(xyz);

                            std::map<std::string, double> data;
                            data.insert(std::pair<std::string, double>("timestamp", timestep));
                            data.insert(std::pair<std::string, double>("target_index", min_tindex));
                            data.insert(std::pair<std::string, double>("nRaysHit", 500));
                            if (add_flags) {
                                // gapfillMisses_code = 2: downward edge points
                                data.insert(std::pair<std::string, double>("gapfillMisses_code", 2.0));
                            }

                            addHitPoint(scanID, xyz, spherical, data);
                            filled_positions.insert(grid_key);  // Mark this position as filled
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
        std::cout << "filled " << xyz_filled.size() << " points (" << npointsfilled << " interior, "
                  << npointsextrapolated << " edge)." << std::endl;
        std::cout << "  Processed " << hit_table2D.size() << " scan columns" << std::endl;
    }
    return xyz_filled;
}

void LiDARcloud::triangulateHitPoints(float Lmax, float max_aspect_ratio) {

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

            if (pts_pass1.size() == 0) continue;

            // Handle coordinate wrapping
            float h[2] = {0, 0};
            for (int r = 0; r < pts_pass1.size(); r++) {
                if (pts_pass1.at(r).c < 0.5 * M_PI) h[0] += 1.f;
                else if (pts_pass1.at(r).c > 1.5 * M_PI) h[1] += 1.f;
            }
            h[0] /= float(pts_pass1.size());
            h[1] /= float(pts_pass1.size());
            if (h[0] + h[1] > 0.4) {
                for (int r = 0; r < pts_pass1.size(); r++) {
                    pts_pass1.at(r).c += M_PI;
                    if (pts_pass1.at(r).c > 2.f * M_PI) pts_pass1.at(r).c -= 2.f * M_PI;
                }
            }

            std::vector<int> dupes_pass1;
            de_duplicate(pts_pass1, dupes_pass1);

            std::vector<Triad> triads_pass1;
            int success = 0, Ntries = 0;
            while (success != 1 && Ntries < 3) {
                Ntries++;
                success = s_hull_pro(pts_pass1, triads_pass1);
                if (success != 1) {
                    for (int r = 0; r < pts_pass1.size(); r++) {
                        pts_pass1.at(r).c += 0.25 * M_PI;
                        if (pts_pass1.at(r).c > 2.f * M_PI) pts_pass1.at(r).c -= 2.f * M_PI;
                    }
                }
            }

            if (success != 1) continue;

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

        // Snap coordinates to fixed precision for cross-platform consistency
        // This eliminates tiny floating-point differences that cause algorithmic divergence
        // Using 1e-6 provides good balance between precision and robustness
        const float COORD_SNAP_PRECISION = 1e-6f;
        for (auto& pt : pts) {
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

        int success = 0;
        int Ntries = 0;
        while (success != 1 && Ntries < 3) {
            Ntries++;

            success = s_hull_pro(pts, triads);

            if (success != 1) {

                // try a 90 degree coordinate shift
                if (printmessages) {
                    std::cout << "Shifting scan " << s << " (try " << Ntries << " of 3)" << std::endl;
                }
                for (int r = 0; r < pts.size(); r++) {
                    pts.at(r).c += 0.25 * M_PI;
                    if (pts.at(r).c > 2.f * M_PI) {
                        pts.at(r).c -= 2.f * M_PI;
                    }
                }
            }
        }

        if (success != 1) {
            if (printmessages) {
                std::cout << "FAILED: could not triangulate scan " << s << ". Skipping this scan." << std::endl;
            }
            continue;
        } else if (printmessages) {
            std::cout << "finished triangulation" << std::endl;
        }

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

            // Apply adaptive filtering for multi-return data
            bool filtered = (L0 > Lmax || L1 > Lmax || L2 > Lmax);

            if (use_adaptive_threshold) {
                // Multi-return: use BOTH separation ratio filter AND aspect ratio filter
                float ang01 = sqrt(pow(raydir0.zenith - raydir1.zenith, 2) + pow(raydir0.azimuth - raydir1.azimuth, 2));
                float ang02 = sqrt(pow(raydir0.zenith - raydir2.zenith, 2) + pow(raydir0.azimuth - raydir2.azimuth, 2));
                float ang12 = sqrt(pow(raydir1.zenith - raydir2.zenith, 2) + pow(raydir1.azimuth - raydir2.azimuth, 2));

                float ratio01 = L0 / (ang01 + 1e-6);
                float ratio02 = L1 / (ang02 + 1e-6);
                float ratio12 = L2 / (ang12 + 1e-6);
                float max_sep_ratio = max(max(ratio01, ratio02), ratio12);

                filtered = filtered || (max_sep_ratio > adaptive_sep_threshold) || (aspect_ratio > max_aspect_ratio);
            } else {
                // Single-return: use aspect ratio filter
                filtered = filtered || (aspect_ratio > max_aspect_ratio);
            }

            if (filtered) {
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

            if (pts_pass1.size() == 0) continue;

            // Handle coordinate wrapping
            float h[2] = {0, 0};
            for (int r = 0; r < pts_pass1.size(); r++) {
                if (pts_pass1.at(r).c < 0.5 * M_PI) h[0] += 1.f;
                else if (pts_pass1.at(r).c > 1.5 * M_PI) h[1] += 1.f;
            }
            h[0] /= float(pts_pass1.size());
            h[1] /= float(pts_pass1.size());
            if (h[0] + h[1] > 0.4) {
                for (int r = 0; r < pts_pass1.size(); r++) {
                    pts_pass1.at(r).c += M_PI;
                    if (pts_pass1.at(r).c > 2.f * M_PI) pts_pass1.at(r).c -= 2.f * M_PI;
                }
            }

            std::vector<int> dupes_pass1;
            de_duplicate(pts_pass1, dupes_pass1);

            std::vector<Triad> triads_pass1;
            int success = 0, Ntries = 0;
            while (success != 1 && Ntries < 3) {
                Ntries++;
                success = s_hull_pro(pts_pass1, triads_pass1);
                if (success != 1) {
                    for (int r = 0; r < pts_pass1.size(); r++) {
                        pts_pass1.at(r).c += 0.25 * M_PI;
                        if (pts_pass1.at(r).c > 2.f * M_PI) pts_pass1.at(r).c -= 2.f * M_PI;
                    }
                }
            }

            if (success != 1) continue;

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
            std::cout << "Scan " << s << " triangulation: " << count << " points used, "
                      << delete_count << " points filtered out";
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

        // Snap coordinates to fixed precision for cross-platform consistency
        // This eliminates tiny floating-point differences that cause algorithmic divergence
        // Using 1e-6 provides good balance between precision and robustness
        const float COORD_SNAP_PRECISION = 1e-6f;
        for (auto& pt : pts) {
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

        int success = 0;
        int Ntries = 0;
        while (success != 1 && Ntries < 3) {
            Ntries++;

            success = s_hull_pro(pts, triads);

            if (success != 1) {

                // try a 90 degree coordinate shift
                if (printmessages) {
                    std::cout << "Shifting scan " << s << " (try " << Ntries << " of 3)" << std::endl;
                }
                for (int r = 0; r < pts.size(); r++) {
                    pts.at(r).c += 0.25 * M_PI;
                    if (pts.at(r).c > 2.f * M_PI) {
                        pts.at(r).c -= 2.f * M_PI;
                    }
                }
            }
        }

        if (success != 1) {
            if (printmessages) {
                std::cout << "FAILED: could not triangulate scan " << s << ". Skipping this scan." << std::endl;
            }
            continue;
        } else if (printmessages) {
            std::cout << "finished triangulation" << std::endl;
        }

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

            // Apply adaptive filtering for multi-return data
            bool filtered = (L0 > Lmax || L1 > Lmax || L2 > Lmax);

            if (use_adaptive_threshold) {
                // Multi-return: use BOTH separation ratio filter AND aspect ratio filter
                float ang01 = sqrt(pow(raydir0.zenith - raydir1.zenith, 2) + pow(raydir0.azimuth - raydir1.azimuth, 2));
                float ang02 = sqrt(pow(raydir0.zenith - raydir2.zenith, 2) + pow(raydir0.azimuth - raydir2.azimuth, 2));
                float ang12 = sqrt(pow(raydir1.zenith - raydir2.zenith, 2) + pow(raydir1.azimuth - raydir2.azimuth, 2));

                float ratio01 = L0 / (ang01 + 1e-6);
                float ratio02 = L1 / (ang02 + 1e-6);
                float ratio12 = L2 / (ang12 + 1e-6);
                float max_sep_ratio = max(max(ratio01, ratio02), ratio12);

                filtered = filtered || (max_sep_ratio > adaptive_sep_threshold) || (aspect_ratio > max_aspect_ratio);
            } else {
                // Single-return: use aspect ratio filter
                filtered = filtered || (aspect_ratio > max_aspect_ratio);
            }

            if (filtered) {
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

    // Get the total theoretical leaf area for each grid cell based on LiDAR scan
    for (uint v = 0; v < Ncells; v++) {

        float leaf_area_total = getCellLeafArea(v);

        float reconstruct_frac = (leaf_area_total - leaf_area_current.at(v)) / leaf_area_total;

        if (leaf_area_total == 0 || reconstructed_alphamasks_size.size() == 0) { // no leaves in gridcell
            if (printmessages) {
                cout << "WARNING: skipping volume #" << v << " because it has no measured leaf area." << endl;
            }
            continue;
        } else if (getTriangleCount() == 0) {
            if (printmessages) {
                cout << "WARNING: skipping volume #" << v << " because it has no triangles." << endl;
            }
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

        std::cout << "azimuth " << sc.azimuth * (180 / M_PI) << ", zenith " << sc.zenith * (180 / M_PI) << std::endl;

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

    std::cout << "theta_range = " << theta_range * (180.0 / M_PI) << std::endl;
    std::cout << "phi_range = " << phi_range * (180.0 / M_PI) << std::endl;

    std::cout << "original # hitpoints = " << getHitCount() << std::endl;

    std::cout << "getHitScanID(getHitCount()-1) = " << getHitScanID(getHitCount() - 1) << " getHitScanID(0) = " << getHitScanID(0) << std::endl;

    for (int r = (getHitCount() - 1); r >= 0; r--) {
        if (getHitScanID(r) == source) {
            helios::SphericalCoord raydir = getHitRaydir(r);
            float this_theta = raydir.zenith;
            float this_phi = raydir.azimuth;
            double this_phi_d = double(this_phi);
            setHitData(r, "beam_azimuth", this_phi_d);
            if (this_phi < phi_range.x || this_phi > phi_range.y || this_phi < phi_range.x || this_theta > theta_range.y) {
                deleteHitPoint(r);
            }
        }
    }

    std::cout << "# hitpoints remaining after crop = " << getHitCount() << std::endl;
}

// ========== SHARED METHODS FOR GPU AND CD IMPLEMENTATIONS ==========

void LiDARcloud::computeGtheta(uint Ncells, uint Nscans,
                               std::vector<float> &Gtheta, std::vector<float> &Gtheta_bar) {

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
            
            helios::vec3 normal = cross(v0, v2);
            normal.normalize();
            
            helios::vec3 raydir = t0 - getScanOrigin(tri.scanID);
            raydir.normalize();
            
            float theta = fabs(acos_safe(raydir.z));
            
            if (area == area) { // Check for NaN
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

bool LiDARcloud::invertLAD(uint voxel_index, float P, float Gtheta,
                           const std::vector<float> &dr_samples, int min_voxel_hits,
                           const helios::vec3 &gridsize, float &leaf_area) {
    
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
    
    // Initial guesses
    float a = 0.1f;
    float h = 0.01f;
    
    // Compute initial error
    float mean = 0.f;
    for (size_t j = 0; j < dr_samples.size(); j++) {
        mean += exp(-a * dr_samples[j] * Gtheta);
    }
    mean /= float(dr_samples.size());
    float error = fabs(mean - P) / P;
    
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
        error = fabs(mean - P) / P;
        
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

    // Check convergence and use fallback if needed
    bool converged = (iter < maxiter - 1 && a == a && a <= 100);
    bool used_fallback = false;
    
    if (!converged) {
        if (printmessages) {
            std::cout << "WARNING: LAD inversion failed for volume #" << voxel_index 
                     << ". Using average dr formulation." << std::endl;
        }
        a = (1.f - P) / (dr_bar * Gtheta);
        used_fallback = true;
    }
    
    // Additional constraint for high LAD values
    if (a > 5) {
        a = fmin((1.f - P) / dr_bar / Gtheta, -log(P) / dr_bar / Gtheta);
    }
    
    // Compute final leaf area
    leaf_area = a * gridsize.x * gridsize.y * gridsize.z;

    return true;
}

uint LiDARcloud::filterRaysByBoundingBox(const helios::vec3 &scan_origin,
                                          const std::vector<helios::vec3> &ray_endpoints,
                                          const helios::vec3 &bb_center,
                                          const helios::vec3 &bb_size,
                                          std::vector<uint> &filtered_indices) {
    
    filtered_indices.clear();
    filtered_indices.reserve(ray_endpoints.size());
    
    // Compute bounding box min/max
    float x0 = bb_center.x - 0.5f * bb_size.x;
    float x1 = bb_center.x + 0.5f * bb_size.x;
    float y0 = bb_center.y - 0.5f * bb_size.y;
    float y1 = bb_center.y + 0.5f * bb_size.y;
    float z0 = bb_center.z - 0.5f * bb_size.z;
    float z1 = bb_center.z + 0.5f * bb_size.z;
    
    // Ray origin
    float ox = scan_origin.x;
    float oy = scan_origin.y;
    float oz = scan_origin.z;
    
    // Test each ray
    for (size_t i = 0; i < ray_endpoints.size(); i++) {
        
        // Check if origin is inside bounding box
        if (ox >= x0 && ox <= x1 && oy >= y0 && oy <= y1 && oz >= z0 && oz <= z1) {
            filtered_indices.push_back(i);
            continue;
        }
        
        // Compute ray direction
        helios::vec3 direction = ray_endpoints[i] - scan_origin;
        direction.normalize();
        
        float dx = direction.x;
        float dy = direction.y;
        float dz = direction.z;
        
        // Slab method for ray-AABB intersection
        float tx_min, ty_min, tz_min;
        float tx_max, ty_max, tz_max;
        
        // X slab
        float a = 1.0f / dx;
        if (a >= 0) {
            tx_min = (x0 - ox) * a;
            tx_max = (x1 - ox) * a;
        } else {
            tx_min = (x1 - ox) * a;
            tx_max = (x0 - ox) * a;
        }
        
        // Y slab
        float b = 1.0f / dy;
        if (b >= 0) {
            ty_min = (y0 - oy) * b;
            ty_max = (y1 - oy) * b;
        } else {
            ty_min = (y1 - oy) * b;
            ty_max = (y0 - oy) * b;
        }
        
        // Z slab
        float c = 1.0f / dz;
        if (c >= 0) {
            tz_min = (z0 - oz) * c;
            tz_max = (z1 - oz) * c;
        } else {
            tz_min = (z1 - oz) * c;
            tz_max = (z0 - oz) * c;
        }
        
        // Find largest entering t value
        float t0 = tx_min;
        if (ty_min > t0) t0 = ty_min;
        if (tz_min > t0) t0 = tz_min;
        
        // Find smallest exiting t value
        float t1 = tx_max;
        if (ty_max < t1) t1 = ty_max;
        if (tz_max < t1) t1 = tz_max;
        
        // Ray intersects if t0 < t1 and t1 > 0
        if (t0 < t1 && t1 > 1e-6f) {
            filtered_indices.push_back(i);
        }
    }
    
    return filtered_indices.size();
}

void LiDARcloud::calculateVoxelPathLengths(const helios::vec3 &scan_origin,
                                              const std::vector<helios::vec3> &ray_directions,
                                              const std::vector<helios::vec3> &voxel_centers,
                                              const std::vector<helios::vec3> &voxel_sizes,
                                              const std::vector<float> &voxel_rotations,
                                              std::vector<std::vector<float>> &dr_agg,
                                              std::vector<float> &hit_before_agg,
                                              std::vector<float> &hit_after_agg) {

    const uint Ncells = voxel_centers.size();

    // Initialize output arrays
    dr_agg.resize(Ncells);
    hit_before_agg.resize(Ncells, 0.0f);
    hit_after_agg.resize(Ncells, 0.0f);

    // Manual ray-voxel intersection (to track ray indices properly)
    // This matches the GPU kernel behavior exactly

    for (uint c = 0; c < Ncells; c++) {
        helios::vec3 center = voxel_centers[c];
        helios::vec3 size = voxel_sizes[c];
        float rotation = voxel_rotations[c];

        // For each ray, test intersection with this voxel
        for (size_t r = 0; r < ray_directions.size(); r++) {

            // Transform ray if voxel is rotated
            helios::vec3 ray_origin = scan_origin;
            helios::vec3 ray_dir = ray_directions[r];

            if (fabs(rotation) > 1e-6f) {
                // Inverse rotate (following GPU kernel pattern)
                helios::vec3 endpoint = scan_origin + ray_directions[r] * 10000.f;
                helios::vec3 anchor = center; // Using voxel center as anchor

                ray_origin = rotatePointAboutLine(scan_origin - anchor, helios::make_vec3(0, 0, 0), helios::make_vec3(0, 0, 1), -rotation) + anchor;
                helios::vec3 transformed_endpoint = rotatePointAboutLine(endpoint - anchor, helios::make_vec3(0, 0, 0), helios::make_vec3(0, 0, 1), -rotation) + anchor;

                ray_dir = transformed_endpoint - ray_origin;
                ray_dir.normalize();
            }

            // Ray-AABB intersection (slab method - matches GPU kernel)
            helios::vec3 voxel_min = center - size * 0.5f;
            helios::vec3 voxel_max = center + size * 0.5f;

            float tx_min = (voxel_min.x - ray_origin.x) / ray_dir.x;
            float tx_max = (voxel_max.x - ray_origin.x) / ray_dir.x;
            if (tx_min > tx_max) std::swap(tx_min, tx_max);

            float ty_min = (voxel_min.y - ray_origin.y) / ray_dir.y;
            float ty_max = (voxel_max.y - ray_origin.y) / ray_dir.y;
            if (ty_min > ty_max) std::swap(ty_min, ty_max);

            float tz_min = (voxel_min.z - ray_origin.z) / ray_dir.z;
            float tz_max = (voxel_max.z - ray_origin.z) / ray_dir.z;
            if (tz_min > tz_max) std::swap(tz_min, tz_max);

            float t0 = std::max({tx_min, ty_min, tz_min});
            float t1 = std::min({tx_max, ty_max, tz_max});

            // Ray intersects voxel if t0 < t1 and t1 > 0
            if (t0 < t1 && t1 > 1e-6f) {
                // Path length through voxel
                float dr = fabs(t1 - t0);
                dr_agg[c].push_back(dr);

                // Weight for radiative transfer
                float weight = 1.0f; // Simplified (Issue 4)
                float zenith_weight = sin(acos_safe(ray_dir.z));

                // All intersecting rays count toward hit_after
                // (GPU kernel only adds to hit_after if hit is within/after voxel,
                // but for empty voxels, all rays that pass through count)
                hit_after_agg[c] += zenith_weight * weight;
            }
        }
    }
}

void LiDARcloud::calculateLeafArea(helios::Context *context) {
    calculateLeafArea(context, 1);
}

void LiDARcloud::calculateLeafArea(helios::Context *context, int min_voxel_hits) {

    if (printmessages) {
        std::cout << "Calculating leaf area (CollisionDetection)..." << std::endl;
    }

    // Validation checks (same as GPU version)
    if (!triangulationcomputed) {
        helios_runtime_error("ERROR (LiDARcloud::calculateLeafAreaCD): Triangulation must be performed prior to leaf area calculation. See triangulateHitPoints().");
    }

    if (!hitgridcellcomputed) {
        calculateHitGridCell();
    }

    // Initialize CollisionDetection if needed
    initializeCollisionDetection(context);

    const uint Nscans = getScanCount();
    const uint Ncells = getGridCellCount();

    // Auto-detect multi-return data and select appropriate algorithm
    const bool use_equal_weighting = isMultiReturnData();

    if (printmessages) {
        if (use_equal_weighting) {
            std::cout << "Multi-return data detected - using beam-based equal weighting algorithm (CD)" << std::endl;

            // Check if gap filling has been applied
            bool has_gapfilled = false;
            for (size_t r = 0; r < getHitCount(); r++) {
                if (doesHitDataExist(r, "gapfillMisses_code")) {
                    has_gapfilled = true;
                    break;
                }
            }

            if (!has_gapfilled) {
                std::cout << "WARNING: Multi-return data detected but gap filling has not been applied." << std::endl;
                std::cout << "         For best results with multi-return data, call gapfillMisses() before calculateLeafArea()." << std::endl;
            }
        } else {
            std::cout << "Single-return data - using standard weighting algorithm (CD)" << std::endl;
        }
    }

    // Branch to appropriate implementation
    if (use_equal_weighting) {
        // ============ MULTI-RETURN PATH (Equal Weighting Algorithm - CPU) ============

        // Additional arrays for equal weighting P calculation
        std::vector<std::vector<float>> P_equal_numerator_array(Ncells);
        std::vector<std::vector<float>> P_equal_denominator_array(Ncells);
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
            for (size_t r = 0; r < getHitCount(); r++) {
                if (getHitScanID(r) == s) {
                    this_scan_xyz.push_back(getHitXYZ(r));
                    this_scan_index.push_back(r);
                }
            }
            size_t Nhits = this_scan_xyz.size();
            if (Nhits == 0) continue;

            // Group hits by timestamp into beams
            BeamGrouping beams = groupHitsByTimestamp(this_scan_index);
            uint Nbeams = beams.Nbeams;

            helios::vec3 origin = getScanOrigin(s);
            float scanner_range = 5000.0f;

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

                    // Inverse rotate if needed
                    if (fabs(rotation) > 1e-6f) {
                        helios::vec3 anchor = center;
                        hit_xyz = rotatePointAboutLine(hit_xyz - anchor, helios::make_vec3(0,0,0),
                                                       helios::make_vec3(0,0,1), -rotation) + anchor;
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
                    if (tx_min > tx_max) std::swap(tx_min, tx_max);

                    float ty_min = (voxel_min.y - origin.y) / direction.y;
                    float ty_max = (voxel_max.y - origin.y) / direction.y;
                    if (ty_min > ty_max) std::swap(ty_min, ty_max);

                    float tz_min = (voxel_min.z - origin.z) / direction.z;
                    float tz_max = (voxel_max.z - origin.z) / direction.z;
                    if (tz_min > tz_max) std::swap(tz_min, tz_max);

                    float t0 = std::max({tx_min, ty_min, tz_min});
                    float t1 = std::min({tx_max, ty_max, tz_max});

                    // Classify hit location relative to voxel
                    if (t0 < t1 && t1 > 1e-6f) {
                        dr[i] = fabs(t1 - t0);

                        if (hit_distance >= t0 && hit_distance <= t1) {
                            hit_location[i] = 2; // Inside voxel
                        } else if (hit_distance > t1 && hit_distance < scanner_range) {
                            hit_location[i] = 3; // After voxel (within range)
                        } else if (hit_distance >= scanner_range) {
                            hit_location[i] = 4; // Miss (beyond range)
                        } else if (hit_distance < t0) {
                            hit_location[i] = 1; // Before voxel
                        }
                    }
                }

                // Beam-level processing
                float P_equal_numerator = 0;
                float P_equal_denominator = 0;

                for (uint k = 0; k < Nbeams; k++) {
                    float E_before = 0, E_inside = 0, E_after = 0, E_miss = 0;
                    float drr = 0;
                    int dr_count = 0;  // Count returns with dr > 0

                    // Count returns in each location for this beam
                    for (uint j = 0; j < beams.beam_array.at(k).size(); j++) {
                        uint i = beams.beam_array.at(k).at(j);

                        if (dr[i] > 0) {
                            drr += dr[i];
                            dr_count++;
                        }

                        if (hit_location[i] == 1) E_before++;
                        else if (hit_location[i] == 2) E_inside++;
                        else if (hit_location[i] == 3) E_after++;
                        else if (hit_location[i] == 4) E_miss++;
                    }

                    // Equal weighting P calculation - simple average per Eq. 7
                    // P = (1/B_tot) Σ[E_after / (E_inside + E_after)]
                    if (E_inside != 0 || E_after != 0) {
                        P_equal_numerator += (E_after / (E_inside + E_after));
                        P_equal_denominator += 1;
                    } else if (E_inside == 0 && E_after == 0 && E_before == 0 && E_miss != 0) {
                        P_equal_numerator += 1;
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
            }
        }

        // Calculate G(theta) using shared method
        std::vector<float> Gtheta;
        computeGtheta(Ncells, Nscans, Gtheta, Gtheta_bar);

        // LAD inversion with equal weighting P
        if (printmessages) {
            std::cout << "Inverting to find LAD..." << std::flush;
        }

        for (uint v = 0; v < Ncells; v++) {
            // Calculate P using equal weighting formula
            float P = 0.0f;
            float P_num_sum = 0.0f, P_denom_sum = 0.0f;
            for (uint s = 0; s < P_equal_numerator_array[v].size(); s++) {
                P_num_sum += P_equal_numerator_array[v][s];
                P_denom_sum += P_equal_denominator_array[v][s];
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
                continue;
            }

            // Use shared invertLAD method
            float leaf_area = 0.0f;
            helios::vec3 gridsize = getCellSize(v);
            invertLAD(v, P, Gtheta[v], dr_agg[v], min_voxel_hits, gridsize, leaf_area);

            setCellLeafArea(leaf_area, v);
            setCellGtheta(Gtheta[v], v);
        }

        if (printmessages) {
            std::cout << "done." << std::endl;
        }

        return; // Exit after multi-return processing
    }

    // ============ SINGLE-RETURN PATH (Standard Algorithm) ============
    // Existing code continues below unchanged
    
    // Aggregation arrays
    std::vector<std::vector<float>> dr_agg;
    dr_agg.resize(Ncells);
    std::vector<float> hit_before_agg;
    hit_before_agg.resize(Ncells, 0);
    std::vector<float> hit_after_agg;
    hit_after_agg.resize(Ncells, 0);
    std::vector<float> hit_inside_agg;
    hit_inside_agg.resize(Ncells, 0);
    std::vector<float> Gtheta_bar;
    Gtheta_bar.resize(Ncells, 0.f);
    
    // Process each scan
    for (uint s = 0; s < Nscans; s++) {
        
        const int Nt = getScanSizeTheta(s);
        const int Np = getScanSizePhi(s);
        const size_t Nmisses = Nt * Np;
        
        const helios::vec3 origin = getScanOrigin(s);
        
        // ----- STAGE B: BOUNDING BOX FILTERING ----- //
        
        // Generate all scan ray endpoints
        std::vector<helios::vec3> scan_endpoints;
        scan_endpoints.reserve(Nmisses);
        
        for (int j = 0; j < Np; j++) {
            for (int i = 0; i < Nt; i++) {
                helios::vec3 direction = sphere2cart(scans.at(s).rc2direction(i, j));
                helios::vec3 endpoint = origin + direction * 10000.f;
                scan_endpoints.push_back(endpoint);
            }
        }
        
        // Get global bounding box
        helios::vec3 gboxmin, gboxmax;
        getGridBoundingBox(gboxmin, gboxmax);
        helios::vec3 bbcenter = gboxmin + 0.5f * (gboxmax - gboxmin);
        helios::vec3 bbsize = gboxmax - gboxmin;
        
        // Filter rays that hit bounding box
        std::vector<uint> bb_hit_indices;
        uint Nmissesbb = filterRaysByBoundingBox(origin, scan_endpoints, bbcenter, bbsize, bb_hit_indices);
        
        if (Nmissesbb == 0) {
            std::cerr << "ERROR (calculateLeafAreaCD): No scan rays passed through grid cells." << std::endl;
            for (uint c = 0; c < Ncells; c++) {
                setCellLeafArea(0, c);
            }
            return;
        }

        // Extract filtered ray directions
        std::vector<helios::vec3> missesbb_directions;
        missesbb_directions.reserve(Nmissesbb);
        for (uint idx : bb_hit_indices) {
            helios::vec3 direction = scan_endpoints[idx] - origin;
            direction.normalize();
            missesbb_directions.push_back(direction);
        }
        
        // ----- STAGE C: VOXEL PATH LENGTHS ----- //

        // Prepare voxel data arrays
        std::vector<helios::vec3> voxel_centers_vec, voxel_sizes_vec;
        std::vector<float> voxel_rotations_vec;
        voxel_centers_vec.reserve(Ncells);
        voxel_sizes_vec.reserve(Ncells);
        voxel_rotations_vec.reserve(Ncells);

        for (uint c = 0; c < Ncells; c++) {
            voxel_centers_vec.push_back(getCellCenter(c));
            voxel_sizes_vec.push_back(getCellSize(c));
            voxel_rotations_vec.push_back(getCellRotation(c));
        }

        // Calculate path lengths using CollisionDetection
        calculateVoxelPathLengths(origin, missesbb_directions,
                                     voxel_centers_vec, voxel_sizes_vec, voxel_rotations_vec,
                                     dr_agg, hit_before_agg, hit_after_agg);

        // FIX: Calculate hit_before for occluded voxels
        // For each hit from this scan, determine which voxels it occurred before
        for (size_t r = 0; r < getHitCount(); r++) {
            if (getHitScanID(r) != s) continue; // Only process hits from this scan

            int hit_voxel = getHitGridCell(r);
            if (hit_voxel < 0) continue; // Hit not in any grid voxel

            helios::vec3 hit_xyz = getHitXYZ(r);
            helios::vec3 ray_dir = (hit_xyz - origin);
            float hit_distance = ray_dir.magnitude();
            ray_dir.normalize();

            // For each voxel, check if this hit occurred BEFORE entering that voxel
            for (uint c = 0; c < Ncells; c++) {
                if (c == hit_voxel) continue; // Hit is in this voxel, not before it

                // Check if ray intersects this voxel
                helios::vec3 voxel_min = voxel_centers_vec[c] - voxel_sizes_vec[c] * 0.5f;
                helios::vec3 voxel_max = voxel_centers_vec[c] + voxel_sizes_vec[c] * 0.5f;

                float tx_min = (voxel_min.x - origin.x) / ray_dir.x;
                float tx_max = (voxel_max.x - origin.x) / ray_dir.x;
                if (tx_min > tx_max) std::swap(tx_min, tx_max);

                float ty_min = (voxel_min.y - origin.y) / ray_dir.y;
                float ty_max = (voxel_max.y - origin.y) / ray_dir.y;
                if (ty_min > ty_max) std::swap(ty_min, ty_max);

                float tz_min = (voxel_min.z - origin.z) / ray_dir.z;
                float tz_max = (voxel_max.z - origin.z) / ray_dir.z;
                if (tz_min > tz_max) std::swap(tz_min, tz_max);

                float t_enter = std::max({tx_min, ty_min, tz_min});
                float t_exit = std::min({tx_max, ty_max, tz_max});

                // Ray intersects this voxel
                if (t_enter < t_exit && t_exit > 1e-6f) {
                    // Check if hit occurred BEFORE entering this voxel
                    if (hit_distance < t_enter) {
                        float zenith_weight = sin(acos_safe(ray_dir.z));
                        hit_before_agg[c] += zenith_weight;
                    }
                }
            }
        }
    }

    // ----- STAGE D: HIT CLASSIFICATION ----- //
    
    // Calculate hit_inside for all scans (same as GPU version)
    for (size_t r = 0; r < getHitCount(); r++) {
        if (getHitGridCell(r) >= 0) {
            helios::vec3 direction = getHitXYZ(r) - getScanOrigin(getHitScanID(r));
            direction.normalize();
            hit_inside_agg.at(getHitGridCell(r)) += sin(acos_safe(direction.z));
        }
    }

    // ----- STAGE E: G(THETA) CALCULATION ----- //
    
    std::vector<float> Gtheta;
    computeGtheta(Ncells, Nscans, Gtheta, Gtheta_bar);
    
    // ----- STAGES F, G, H: TRANSMISSION, LAD INVERSION, FINAL LEAF AREA ----- //
    
    if (printmessages) {
        std::cout << "Inverting to find LAD..." << std::flush;
    }
    
    for (uint v = 0; v < Ncells; v++) {
        
        // Stage F: Calculate transmission probability
        float P = 0.0f;
        if (hit_after_agg[v] - hit_before_agg[v] > 0) {
            P = 1.f - float(hit_inside_agg[v]) / float(hit_after_agg[v] - hit_before_agg[v]);
        }

        // Stages G & H: LAD inversion and final leaf area
        float leaf_area = 0.0f;
        helios::vec3 gridsize = getCellSize(v);

        invertLAD(v, P, Gtheta[v], dr_agg[v], min_voxel_hits, gridsize, leaf_area);

        setCellLeafArea(leaf_area, v);
        setCellGtheta(Gtheta[v], v);
    }
    
    if (printmessages) {
        std::cout << "done." << std::endl;
    }
}

// Helper to find ray index by matching direction (Issue 1 workaround)
size_t findRayIndexByDirection(const helios::vec3 &scan_origin, 
                                 const helios::vec3 &intersection_point,
                                 const std::vector<helios::vec3> &ray_directions) {
    helios::vec3 hit_direction = intersection_point - scan_origin;
    hit_direction.normalize();
    
    for (size_t i = 0; i < ray_directions.size(); i++) {
        if ((hit_direction - ray_directions[i]).magnitude() < 1e-5f) {
            return i;
        }
    }
    return SIZE_MAX; // Not found
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
    
    // Process each hit point (parallelized with OpenMP)
    #pragma omp parallel for schedule(dynamic, 1000)
    for (int r = 0; r < static_cast<int>(total_hits); r++) {
        
        helios::vec3 hit_xyz = getHitXYZ(r);
        int assigned_cell = -1; // Default: not in any cell
        
        // Test against each voxel
        for (uint c = 0; c < Ncells; c++) {
            
            helios::vec3 center = getCellCenter(c);
            helios::vec3 anchor = getCellGlobalAnchor(c);
            helios::vec3 size = getCellSize(c);
            float rotation = getCellRotation(c);
            
            // Inverse rotate hit point if voxel is rotated
            helios::vec3 hit_xyz_rot = hit_xyz;
            if (fabs(rotation) > 1e-6f) {
                hit_xyz_rot = rotatePointAboutLine(hit_xyz - anchor, helios::make_vec3(0, 0, 0), 
                                                   helios::make_vec3(0, 0, 1), -rotation) + anchor;
            }
            
            // Treat hit as a ray from origin for AABB test (matches GPU kernel)
            helios::vec3 origin = helios::make_vec3(0, 0, 0);
            helios::vec3 direction = hit_xyz_rot - origin;
            direction.normalize();
            
            // AABB bounds
            float x0 = center.x - 0.5f * size.x;
            float x1 = center.x + 0.5f * size.x;
            float y0 = center.y - 0.5f * size.y;
            float y1 = center.y + 0.5f * size.y;
            float z0 = center.z - 0.5f * size.z;
            float z1 = center.z + 0.5f * size.z;
            
            // Slab method for ray-AABB intersection
            float tx_min = (x0 - origin.x) / direction.x;
            float tx_max = (x1 - origin.x) / direction.x;
            if (tx_min > tx_max) std::swap(tx_min, tx_max);
            
            float ty_min = (y0 - origin.y) / direction.y;
            float ty_max = (y1 - origin.y) / direction.y;
            if (ty_min > ty_max) std::swap(ty_min, ty_max);
            
            float tz_min = (z0 - origin.z) / direction.z;
            float tz_max = (z1 - origin.z) / direction.z;
            if (tz_min > tz_max) std::swap(tz_min, tz_max);
            
            float t0 = std::max({tx_min, ty_min, tz_min});
            float t1 = std::min({tx_max, ty_max, tz_max});
            
            // Check if hit point is inside voxel
            if (t0 < t1 && t1 > 1e-6f) {
                float T = (hit_xyz_rot - origin).magnitude();
                if (T >= t0 && T <= t1) {
                    assigned_cell = c;
                    break; // Found the cell, stop searching
                }
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

LiDARcloud::BeamGrouping LiDARcloud::groupHitsByTimestamp(const std::vector<uint>& scan_indices) const {

    BeamGrouping result;

    if (scan_indices.empty()) {
        result.Nbeams = 0;
        return result;
    }

    // Sort indices by timestamp to group consecutive hits from same beam
    std::vector<uint> sorted_indices = scan_indices;
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [this](uint a, uint b) {
                  return this->getHitData(a, "timestamp") < this->getHitData(b, "timestamp");
              });

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
                prim_xyz_rot = rotatePointAboutLine(prim_xyz - anchor, helios::make_vec3(0,0,0),
                                                    helios::make_vec3(0,0,1), -rotation) + anchor;
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
            if (tx_min > tx_max) std::swap(tx_min, tx_max);
            
            float ty_min = (y0 - origin_pt.y) / direction.y;
            float ty_max = (y1 - origin_pt.y) / direction.y;
            if (ty_min > ty_max) std::swap(ty_min, ty_max);
            
            float tz_min = (z0 - origin_pt.z) / direction.z;
            float tz_max = (z1 - origin_pt.z) / direction.z;
            if (tz_min > tz_max) std::swap(tz_min, tz_max);
            
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
    
    // Store as primitive data
    std::vector<float> output_LeafArea(Ncells);
    for (int v = 0; v < Ncells; v++) {
        output_LeafArea[v] = total_area[v];
        if (context->doesPrimitiveDataExist(UUIDs_all[v], "gridCell")) {
            context->setPrimitiveData(UUIDs_all[v], "synthetic_leaf_area", total_area[v]);
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
            std::cout << "Performing full-waveform synthetic LiDAR scan..." << std::endl;
        } else {
            std::cout << "Performing discrete return synthetic LiDAR scan..." << std::endl;
        }
    }

    if (getScanCount() == 0) {
        std::cout << "WARNING (syntheticScan): No scans added to the point cloud. Exiting.." << std::endl;
        return;
    }

    float miss_distance = 1001.0; // arbitrary distance from scanner for 'miss' points

    helios::vec3 bb_center;
    helios::vec3 bb_size;

    if (scan_grid_only == false) {

        // Determine bounding box for Context geometry
        helios::vec2 xbounds, ybounds, zbounds;
        context->getDomainBoundingBox(xbounds, ybounds, zbounds);
        bb_center = helios::make_vec3(xbounds.x + 0.5 * (xbounds.y - xbounds.x), ybounds.x + 0.5 * (ybounds.y - ybounds.x), zbounds.x + 0.5 * (zbounds.y - zbounds.x));
        bb_size = helios::make_vec3(xbounds.y - xbounds.x, ybounds.y - ybounds.x, zbounds.y - zbounds.x);

    } else {

        // Determine bounding box for voxels instead of whole domain
        helios::vec3 boxmin, boxmax;
        getGridBoundingBox(boxmin, boxmax);
        bb_center = helios::make_vec3(boxmin.x + 0.5 * (boxmax.x - boxmin.x), boxmin.y + 0.5 * (boxmax.y - boxmin.y), boxmin.z + 0.5 * (boxmax.z - boxmin.z));
        bb_size = helios::make_vec3(boxmax.x - boxmin.x, boxmax.y - boxmin.y, boxmax.z - boxmin.z);
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

        std::vector<helios::vec3> raydir;
        raydir.resize(Ntheta * Nphi);

        for (uint j = 0; j < Nphi; j++) {
            float phi = phimin + float(j) * (phimax - phimin) / float(Nphi - 1);
            for (uint i = 0; i < Ntheta; i++) {
                float theta_z = thetamin + float(i) * (thetamax - thetamin) / float(Ntheta - 1);
                float theta_elev = 0.5f * M_PI - theta_z;
                helios::vec3 dir = sphere2cart(helios::make_SphericalCoord(1.f, theta_elev, phi));
                raydir.at(Ntheta * j + i) = dir;
            }
        }

        size_t N = Ntheta * Nphi;

        // Bounding box intersection test (CPU version, no CUDA)
        std::vector<uint> bb_hit(N, 0);

        // Calculate BB bounds once
        helios::vec3 bb_min = bb_center - bb_size * 0.5f;
        helios::vec3 bb_max = bb_center + bb_size * 0.5f;

        // Check if origin is inside bounding box
        bool origin_inside_bb = (scan_origin.x >= bb_min.x && scan_origin.x <= bb_max.x &&
                                 scan_origin.y >= bb_min.y && scan_origin.y <= bb_max.y &&
                                 scan_origin.z >= bb_min.z && scan_origin.z <= bb_max.z);

        for (size_t r = 0; r < N; r++) {
            // If origin inside BB, all rays automatically hit
            if (origin_inside_bb) {
                bb_hit[r] = 1;
                continue;
            }

            helios::vec3 ray_dir = raydir.at(r);

            // AABB ray intersection using slab method
            float tx_min, tx_max, ty_min, ty_max, tz_min, tz_max;

            float a = 1.0f / ray_dir.x;
            if (a >= 0) {
                tx_min = (bb_min.x - scan_origin.x) * a;
                tx_max = (bb_max.x - scan_origin.x) * a;
            } else {
                tx_min = (bb_max.x - scan_origin.x) * a;
                tx_max = (bb_min.x - scan_origin.x) * a;
            }

            float b = 1.0f / ray_dir.y;
            if (b >= 0) {
                ty_min = (bb_min.y - scan_origin.y) * b;
                ty_max = (bb_max.y - scan_origin.y) * b;
            } else {
                ty_min = (bb_max.y - scan_origin.y) * b;
                ty_max = (bb_min.y - scan_origin.y) * b;
            }

            float c = 1.0f / ray_dir.z;
            if (c >= 0) {
                tz_min = (bb_min.z - scan_origin.z) * c;
                tz_max = (bb_max.z - scan_origin.z) * c;
            } else {
                tz_min = (bb_max.z - scan_origin.z) * c;
                tz_max = (bb_min.z - scan_origin.z) * c;
            }

            // Find largest entering t value
            float t0 = tx_min;
            if (ty_min > t0) t0 = ty_min;
            if (tz_min > t0) t0 = tz_min;

            // Find smallest exiting t value
            float t1 = tx_max;
            if (ty_max < t1) t1 = ty_max;
            if (tz_max < t1) t1 = tz_max;

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

        int count = 0;
        for (int i = 0; i < Ntheta * Nphi; i++) {
            if (bb_hit[i] == 1) {

                base_directions.push_back(raydir.at(i));

                int jj = floor(i / Ntheta);
                int ii = i - jj * Ntheta;
                pulse_scangrid_ij[count] = helios::make_int2(ii, jj);

                count++;
            }
            // NOTE: Miss recording for rays that don't hit BB removed - those rays can't interact with grid.
            // Misses for traced rays are recorded via line 3733 when record_misses=true.
        }

        // Handle case where no rays hit bounding box
        if (N == 0) {
            // If record_misses=true, record all rays as misses
            if (record_misses) {
                float miss_dist = 1001.0f;
                for (int i = 0; i < Ntheta * Nphi; i++) {
                    std::map<std::string, double> data;
                    data["target_index"] = 0;
                    data["target_count"] = 1;
                    data["deviation"] = 0.0;
                    data["timestamp"] = i;
                    data["intensity"] = 1.0;  // Full miss
                    data["distance"] = miss_dist;
                    data["nRaysHit"] = Npulse;  // All rays in pulse missed together

                    helios::vec3 dir = raydir.at(i);
                    helios::vec3 p = scan_origin + dir * miss_dist;
                    addHitPoint(s, p, helios::cart2sphere(dir), helios::RGB::red, data);
                }
            } else if (printmessages) {
                std::cout << "WARNING: Synthetic rays did not hit any primitives." << std::endl;
            }
            continue;  // Move to next scan
        }

        // Allocate host memory for results
        float *hit_t = (float *) malloc(N * Npulse * sizeof(float)); // allocate host memory
        float *hit_fnorm = (float *) malloc(N * Npulse * sizeof(float)); // allocate host memory
        int *hit_ID = (int *) malloc(N * Npulse * sizeof(int)); // allocate host memory

        float exit_diameter = getScanBeamExitDiameter(s);
        float beam_divergence = getScanBeamDivergence(s);

        // Generate N*Npulse perturbed ray directions using beam parameters
        helios::vec3 *direction = (helios::vec3 *) malloc(N * Npulse * sizeof(helios::vec3));

        for (size_t beam = 0; beam < N; beam++) {
            helios::vec3 base_dir = base_directions[beam];

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

                    float theta_offset = beam_divergence * sqrt(ru);  // radial angular distance
                    float phi_offset = 2.0f * M_PI * rv;  // azimuthal angle

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
                helios::vec3 base_dir = base_directions[beam];

                // Construct orthonormal basis {u, v, base_dir} for disk perpendicular to beam
                helios::vec3 reference = (fabs(base_dir.z) < 0.9f) ? helios::make_vec3(0, 0, 1) : helios::make_vec3(1, 0, 0);
                helios::vec3 u = helios::cross(base_dir, reference);
                u.normalize();
                helios::vec3 v = helios::cross(base_dir, u);  // Already normalized since base_dir and u are orthonormal

                for (int p = 0; p < Npulse; p++) {
                    // Uniform sampling on disk using sqrt transform
                    float ru = context->randu();
                    float rv = context->randu();
                    float r_sample = radius * sqrtf(ru);  // sqrt for uniform area distribution
                    float theta = 2.0f * M_PI * rv;
                    float x_disk = r_sample * cosf(theta);
                    float y_disk = r_sample * sinf(theta);

                    // Transform disk point to world space
                    helios::vec3 offset = u * x_disk + v * y_disk;
                    ray_origins[beam * Npulse + p] = scan_origin + offset;
                }
            }
        } else {
            // Point source: all rays originate from scan_origin (backward compatibility)
            for (size_t i = 0; i < N * Npulse; i++) {
                ray_origins[i] = scan_origin;
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
            } else if (t_pulse.size() == 1) { // this is discrete-return data, or we only had one hit for this pulse
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

                std::sort(t_pulse.begin(), t_pulse.end(), [](const std::vector<float> &a, const std::vector<float> &b) {
                    return a[0] < b[0];
                });

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
                        if (nPulseHit == Npulse & distance >= 0.98f * miss_distance) {
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
                        if (nPulseHit == Npulse & distance >= 0.98f * miss_distance) {
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
                if (t_hit.at(hit).at(0) < 0.98f * 1001.0f) {
                    non_miss_count++;
                }
            }

            int real_hit_index = 0;
            for (size_t hit = 0; hit < t_hit.size(); hit++) {

                std::map<std::string, double> data;

                // Check if this is a miss point
                bool is_miss = (t_hit.at(hit).at(0) >= 0.98f * 1001.0f);

                // Assign target_index: misses get 99, real hits get sequential index (0, 1, 2...)
                if (is_miss) {
                    data["target_index"] = 99;  // Special value to exclude from triangulation
                } else {
                    data["target_index"] = real_hit_index;
                    real_hit_index++;
                }

                data["target_count"] = t_hit.size();
                data["deviation"] = fabs(t_hit.at(hit).at(0) - average);
                data["timestamp"] = pulse_scangrid_ij.at(r).y * Ntheta + pulse_scangrid_ij.at(r).x;
                data["intensity"] = t_hit.at(hit).at(1);
                data["distance"] = t_hit.at(hit).at(0);
                data["nRaysHit"] = t_hit.at(hit).at(2);

                float UUID = t_hit.at(hit).at(3);
                if (UUID >= 0 && UUID < ID_mapping.size()) {
                    UUID = ID_mapping.at(int(t_hit.at(hit).at(3)));
                }

                helios::RGBcolor color = helios::RGB::red;

                if (UUID >= 0 && context->doesPrimitiveExist(uint(UUID))) {

                    color = context->getPrimitiveColor(uint(UUID));

                    if (context->doesPrimitiveDataExist(uint(UUID), "object_label") && context->getPrimitiveDataType("object_label") == helios::HELIOS_TYPE_INT) {
                        int label;
                        context->getPrimitiveData(uint(UUID), "object_label", label);
                        data["object_label"] = double(label);
                    }

                    if (context->doesPrimitiveDataExist(uint(UUID), "reflectivity_lidar") && context->getPrimitiveDataType("reflectivity_lidar") == helios::HELIOS_TYPE_FLOAT) {
                        float rho;
                        context->getPrimitiveData(uint(UUID), "reflectivity_lidar", rho);
                        data.at("intensity") *= rho;
                    }
                }

                // Use base direction for this beam (first ray: r*Npulse+0)
                helios::vec3 dir = direction[r * Npulse];
                helios::vec3 p = scan_origin + dir * t_hit.at(hit).at(0);
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
