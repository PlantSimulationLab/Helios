/**
 * \file "Context.cpp" Context declarations.
 *
 * Copyright (C) 2016-2025 Brian Bailey
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */

#include "Context.h"

using namespace helios;

Context::Context() {

    install_out_of_memory_handler();

    //---- ALL DEFAULT VALUES ARE SET HERE ----//

    sim_date = make_Date(1, 6, 2000);

    sim_time = make_Time(12, 0);

    sim_location = make_Location(38.55, 121.76, 8);

    // --- Initialize random number generator ---- //

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed);

    // --- Set Geometry as `Clean' --- //

    currentUUID = 0;

    currentObjectID = 1; // object ID of 0 is reserved for default object
}

void Context::seedRandomGenerator(uint seed) {
    generator.seed(seed);
}

std::minstd_rand0 *Context::getRandomGenerator() {
    return &generator;
}

void Context::addTexture(const char *texture_file) {
    if (textures.find(texture_file) == textures.end()) { // texture has not already been added

        // texture must have type PNG or JPEG
        const std::string &fn = texture_file;
        const std::string &ext = getFileExtension(fn);
        if (ext != ".png" && ext != ".PNG" && ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG") {
            helios_runtime_error("ERROR (Context::addTexture): Texture file " + fn + " is not PNG or JPEG format.");
        } else if (!doesTextureFileExist(texture_file)) {
            helios_runtime_error("ERROR (Context::addTexture): Texture file " + std::string(texture_file) + " does not exist.");
        }

        textures.emplace(texture_file, Texture(texture_file));
    }
}

bool Context::doesTextureFileExist(const char *texture_file) const {
    return std::filesystem::exists(texture_file);
}

bool Context::validateTextureFileExtenstion(const char *texture_file) const {
    const std::string &fn = texture_file;
    const std::string &ext = getFileExtension(fn);
    if (ext != ".png" && ext != ".PNG" && ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG") {
        return false;
    } else {
        return true;
    }
}

Texture::Texture(const char *texture_file) {
    filename = texture_file;

    //------ determine if transparency channel exists ---------//

    // check if texture file has extension ".png"
    const std::string &ext = getFileExtension(filename);
    if (ext != ".png") {
        hastransparencychannel = false;
    } else {
        hastransparencychannel = PNGHasAlpha(filename.c_str());
    }

    //-------- load transparency channel (if exists) ------------//

    if (ext == ".png") {
        transparencydata = readPNGAlpha(filename);
        image_resolution = make_int2(int(transparencydata.front().size()), int(transparencydata.size()));
    } else {
        image_resolution = getImageResolutionJPEG(texture_file);
    }

    //-------- determine solid fraction --------------//

    if (hastransparencychannel) {
        size_t p = 0.f;
        for (auto &j: transparencydata) {
            for (bool transparency: j) {
                if (transparency) {
                    p += 1;
                }
            }
        }
        float sf = float(p) / float(transparencydata.size() * transparencydata.front().size());
        if (std::isnan(sf)) {
            sf = 0.f;
        }
        solidfraction = sf;
    } else {
        solidfraction = 1.f;
    }
}

std::string Texture::getTextureFile() const {
    return filename;
}

helios::int2 Texture::getImageResolution() const {
    return image_resolution;
}

bool Texture::hasTransparencyChannel() const {
    return hastransparencychannel;
}

const std::vector<std::vector<bool>> *Texture::getTransparencyData() const {
    return &transparencydata;
}

float Texture::getSolidFraction(const std::vector<helios::vec2> &uvs) {
    float solidfraction = 1;

    PixelUVKey key;
    key.coords.reserve(2 * uvs.size());
    for (auto &uvc: uvs) {
        key.coords.push_back(int(std::round(uvc.x * (image_resolution.x - 1))));
        key.coords.push_back(int(std::round(uvc.y * (image_resolution.y - 1))));
    }

    if (solidFracCache.find(key) != solidFracCache.end()) {
        return solidFracCache.at(key);
    }

    solidfraction = computeSolidFraction(uvs);
    solidFracCache.emplace(std::move(key), solidfraction);

    return solidfraction;
}

float Texture::computeSolidFraction(const std::vector<helios::vec2> &uvs) const {
    // Early out for opaque textures or degenerate UVs
    if (!hasTransparencyChannel() || uvs.size() < 3)
        return 1.0f;

    // Fetch alpha mask and dimensions
    const auto *alpha2D = getTransparencyData(); // vector<vector<bool>>
    int W = getImageResolution().x;
    int H = getImageResolution().y;

    // Flatten mask to contiguous array
    std::vector<uint8_t> mask(W * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            mask[y * W + x] = (*alpha2D)[H - 1 - y][x];

    // Compute pixel‐space bounding box from UVs
    float minU = uvs[0].x, maxU = uvs[0].x, minV = uvs[0].y, maxV = uvs[0].y;
    for (auto &p: uvs) {
        minU = std::min(minU, p.x);
        maxU = std::max(maxU, p.x);
        minV = std::min(minV, p.y);
        maxV = std::max(maxV, p.y);
    }
    int xmin = std::clamp(int(std::floor(minU * (W - 1))), 0, W - 1);
    int xmax = std::clamp(int(std::ceil(maxU * (W - 1))), 0, W - 1);
    int ymin = std::clamp(int(std::floor(minV * (H - 1))), 0, H - 1);
    int ymax = std::clamp(int(std::ceil(maxV * (H - 1))), 0, H - 1);

    if (xmin > xmax || ymin > ymax)
        return 0.0f;

    // Precompute half‐space coefficients for each edge i→i+1
    int N = int(uvs.size());
    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        int j = (i + 1) % N;
        const auto &a = uvs[i], &b = uvs[j];
        // L(x,y) = (b.x - a.x)*y  - (b.y - a.y)*x  + (a.x*b.y - a.y*b.x)
        A[i] = b.x - a.x;
        B[i] = -(b.y - a.y);
        C[i] = a.x * b.y - a.y * b.x;
    }

    // Raster‐scan, test each pixel center
    int64_t countTotal = 0, countOpaque = 0;
    float invWm1 = 1.0f / float(W - 1);
    float invHm1 = 1.0f / float(H - 1);

    for (int j = ymin; j <= ymax; ++j) {
        float yuv = (j + 0.5f) * invHm1;
        for (int i = xmin; i <= xmax; ++i) {
            float xuv = (i + 0.5f) * invWm1;
            bool inside = true;

            // all edges must satisfy L(xuv,yuv) >= 0
            for (int k = 0; k < N; ++k) {
                float L = A[k] * yuv + B[k] * xuv + C[k];
                if (L < 0.0f) {
                    inside = false;
                    break;
                }
            }

            if (!inside)
                continue;

            ++countTotal;
            countOpaque += mask[j * W + i];
        }
    }

    return countTotal == 0 ? 0.0f : float(countOpaque) / float(countTotal);
}

void Context::markGeometryClean() {
    for (auto &[UUID, primitive]: primitives) {
        primitive->dirty_flag = false;
    }
    dirty_deleted_primitives.clear();
}

void Context::markGeometryDirty() {
    for (auto &[UUID, primitive]: primitives) {
        primitive->dirty_flag = true;
    }
}

bool Context::isGeometryDirty() const {
    if (!dirty_deleted_primitives.empty()) {
        return true;
    }
    for (auto &[UUID, primitive]: primitives) {
        if (primitive->dirty_flag) {
            return true;
        }
    }
    return false;
}

void Context::markPrimitiveDirty(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::markPrimitiveDirty): Primitive with UUID " + std::to_string(UUID) + " does not exist.");
    }
#endif
    primitives.at(UUID)->dirty_flag = true;
}

void Context::markPrimitiveDirty(const std::vector<uint> &UUIDs) const {
    for (uint UUID: UUIDs) {
        markPrimitiveDirty(UUID);
    }
}

void Context::markPrimitiveClean(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::markPrimitiveDirty): Primitive with UUID " + std::to_string(UUID) + " does not exist.");
    }
#endif
    primitives.at(UUID)->dirty_flag = false;
}

void Context::markPrimitiveClean(const std::vector<uint> &UUIDs) const {
    for (uint UUID: UUIDs) {
        markPrimitiveClean(UUID);
    }
}

[[nodiscard]] bool Context::isPrimitiveDirty(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::markPrimitiveDirty): Primitive with UUID " + std::to_string(UUID) + " does not exist.");
    }
#endif
    return primitives.at(UUID)->dirty_flag;
}


void Context::setDate(int day, int month, int year) {
    if (day < 1 || day > 31) {
        helios_runtime_error("ERROR (Context::setDate): Day of month is out of range (day of " + std::to_string(day) + " was given).");
    } else if (month < 1 || month > 12) {
        helios_runtime_error("ERROR (Context::setDate): Month of year is out of range (month of " + std::to_string(month) + " was given).");
    } else if (year < 1000) {
        helios_runtime_error("ERROR (Context::setDate): Year should be specified in YYYY format.");
    }

    sim_date = make_Date(day, month, year);
}

void Context::setDate(const Date &date) {
    if (date.day < 1 || date.day > 31) {
        helios_runtime_error("ERROR (Context::setDate): Day of month is out of range (day of " + std::to_string(date.day) + " was given).");
    } else if (date.month < 1 || date.month > 12) {
        helios_runtime_error("ERROR (Context::setDate): Month of year is out of range (month of " + std::to_string(date.month) + " was given).");
    } else if (date.year < 1000) {
        helios_runtime_error("ERROR (Context::setDate): Year should be specified in YYYY format.");
    }

    sim_date = date;
}

void Context::setDate(int Julian_day, int year) {
    if (Julian_day < 1 || Julian_day > 366) {
        helios_runtime_error("ERROR (Context::setDate): Julian day out of range.");
    } else if (year < 1000) {
        helios_runtime_error("ERROR (Context::setDate): Year should be specified in YYYY format.");
    }

    sim_date = CalendarDay(Julian_day, year);
}

Date Context::getDate() const {
    return sim_date;
}

const char *Context::getMonthString() const {
    if (sim_date.month == 1) {
        return "JAN";
    } else if (sim_date.month == 2) {
        return "FEB";
    } else if (sim_date.month == 3) {
        return "MAR";
    } else if (sim_date.month == 4) {
        return "APR";
    } else if (sim_date.month == 5) {
        return "MAY";
    } else if (sim_date.month == 6) {
        return "JUN";
    } else if (sim_date.month == 7) {
        return "JUL";
    } else if (sim_date.month == 8) {
        return "AUG";
    } else if (sim_date.month == 9) {
        return "SEP";
    } else if (sim_date.month == 10) {
        return "OCT";
    } else if (sim_date.month == 11) {
        return "NOV";
    } else {
        return "DEC";
    }
}

int Context::getJulianDate() const {
    return JulianDay(sim_date.day, sim_date.month, sim_date.year);
}

void Context::setTime(int minute, int hour) {
    setTime(0, minute, hour);
}

void Context::setTime(int second, int minute, int hour) {
    if (second < 0 || second > 59) {
        helios_runtime_error("ERROR (Context::setTime): Second out of range (0-59).");
    } else if (minute < 0 || minute > 59) {
        helios_runtime_error("ERROR (Context::setTime): Minute out of range (0-59).");
    } else if (hour < 0 || hour > 23) {
        helios_runtime_error("ERROR (Context::setTime): Hour out of range (0-23).");
    }

    sim_time = make_Time(hour, minute, second);
}

void Context::setTime(const Time &time) {
    if (time.minute < 0 || time.minute > 59) {
        helios_runtime_error("ERROR (Context::setTime): Minute out of range (0-59).");
    } else if (time.hour < 0 || time.hour > 23) {
        helios_runtime_error("ERROR (Context::setTime): Hour out of range (0-23).");
    }

    sim_time = time;
}

Time Context::getTime() const {
    return sim_time;
}

void Context::setLocation(const helios::Location &location) {
    sim_location = location;
}

helios::Location Context::getLocation() const {
    return sim_location;
}

float Context::randu() {
    return unif_distribution(generator);
}

float Context::randu(float minrange, float maxrange) {
    if (maxrange < minrange) {
        helios_runtime_error("ERROR (Context::randu): Maximum value of range must be greater than minimum value of range.");
        return 0;
    } else if (maxrange == minrange) {
        return minrange;
    } else {
        return minrange + unif_distribution(generator) * (maxrange - minrange);
    }
}

int Context::randu(int minrange, int maxrange) {
    if (maxrange < minrange) {
        helios_runtime_error("ERROR (Context::randu): Maximum value of range must be greater than minimum value of range.");
        return 0;
    } else if (maxrange == minrange) {
        return minrange;
    } else {
        return minrange + (int) lroundf(unif_distribution(generator) * float(maxrange - minrange));
    }
}

float Context::randn() {
    return norm_distribution(generator);
}

float Context::randn(float mean, float stddev) {
    return mean + norm_distribution(generator) * fabs(stddev);
}


std::vector<uint> Context::getAllUUIDs() const {
    // Use cached result if valid
    if (all_uuids_cache_valid) {
        return cached_all_uuids;
    }
    
    // Rebuild cache
    cached_all_uuids.clear();
    cached_all_uuids.reserve(primitives.size());
    for (const auto &[UUID, primitive]: primitives) {
        if (primitive->ishidden) {
            continue;
        }
        cached_all_uuids.push_back(UUID);
    }
    all_uuids_cache_valid = true;
    return cached_all_uuids;
}

std::vector<uint> Context::getDirtyUUIDs(bool include_deleted_UUIDs) const {

    size_t dirty_count = std::count_if(primitives.begin(), primitives.end(), [&](auto const &kv) { return isPrimitiveDirty(kv.first); });

    std::vector<uint> dirty_UUIDs;
    dirty_UUIDs.reserve(dirty_count);
    for (const auto &[UUID, primitive]: primitives) {
        if (!primitive->dirty_flag || primitive->ishidden) {
            continue;
        }
        dirty_UUIDs.push_back(UUID);
    }

    if (include_deleted_UUIDs) {
        dirty_UUIDs.insert(dirty_UUIDs.end(), dirty_deleted_primitives.begin(), dirty_deleted_primitives.end());
    }

    return dirty_UUIDs;
}

std::vector<uint> Context::getDeletedUUIDs() const {
    return dirty_deleted_primitives;
}

void Context::hidePrimitive(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::hidePrimitive): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
#endif
    primitives.at(UUID)->ishidden = true;
    invalidateAllUUIDsCache();
}

void Context::hidePrimitive(const std::vector<uint> &UUIDs) const {
    for (uint UUID: UUIDs) {
        hidePrimitive(UUID);
    }
}

void Context::showPrimitive(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::showPrimitive): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
#endif
    primitives.at(UUID)->ishidden = false;
    invalidateAllUUIDsCache();
}

void Context::showPrimitive(const std::vector<uint> &UUIDs) const {
    for (uint UUID: UUIDs) {
        showPrimitive(UUID);
    }
}

bool Context::isPrimitiveHidden(uint UUID) const {
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::isPrimitiveHidden): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
    return primitives.at(UUID)->ishidden;
}

void Context::cleanDeletedUUIDs(std::vector<uint> &UUIDs) const {
    for (size_t i = UUIDs.size(); i-- > 0;) {
        if (!doesPrimitiveExist(UUIDs.at(i))) {
            UUIDs.erase(UUIDs.begin() + i);
        }
    }
}

void Context::cleanDeletedUUIDs(std::vector<std::vector<uint>> &UUIDs) const {
    for (auto &vec: UUIDs) {
        for (auto it = vec.begin(); it != vec.end();) {
            if (!doesPrimitiveExist(*it)) {
                it = vec.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void Context::cleanDeletedUUIDs(std::vector<std::vector<std::vector<uint>>> &UUIDs) const {
    for (auto &vec2D: UUIDs) {
        for (auto &vec: vec2D) {
            for (auto it = vec.begin(); it != vec.end();) {
                if (!doesPrimitiveExist(*it)) {
                    it = vec.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
}

void Context::addTimeseriesData(const char *label, float value, const Date &date, const Time &time) {
    // floating point value corresponding to date and time
    double date_value = floor(date.year * 366.25) + date.JulianDay();
    date_value += double(time.hour) / 24. + double(time.minute) / 1440. + double(time.second) / 86400.;

    // Check if data label already exists
    if (timeseries_data.find(label) == timeseries_data.end()) { // does not exist
        timeseries_data[label].push_back(value);
        timeseries_datevalue[label].push_back(date_value);
        return;
    } else { // exists

        uint N = getTimeseriesLength(label);

        auto it_data = timeseries_data[label].begin();
        auto it_datevalue = timeseries_datevalue[label].begin();

        if (N == 1) {
            if (date_value < timeseries_datevalue[label].front()) {
                timeseries_data[label].insert(it_data, value);
                timeseries_datevalue[label].insert(it_datevalue, date_value);
                return;
            } else {
                timeseries_data[label].insert(it_data + 1, value);
                timeseries_datevalue[label].insert(it_datevalue + 1, date_value);
                return;
            }
        } else {
            if (date_value < timeseries_datevalue[label].front()) { // check if data should be inserted at beginning of timeseries
                timeseries_data[label].insert(it_data, value);
                timeseries_datevalue[label].insert(it_datevalue, date_value);
                return;
            } else if (date_value > timeseries_datevalue[label].back()) { // check if data should be inserted at end of timeseries
                timeseries_data[label].push_back(value);
                timeseries_datevalue[label].push_back(date_value);
                return;
            }

            // data should be inserted somewhere in the middle of timeseries
            for (uint t = 0; t < N - 1; t++) {
                if (date_value == timeseries_datevalue[label].at(t)) {
                    std::cerr << "WARNING (Context::addTimeseriesData): Skipping duplicate timeseries date/time." << std::endl;
                    continue;
                }
                if (date_value > timeseries_datevalue[label].at(t) && date_value < timeseries_datevalue[label].at(t + 1)) {
                    timeseries_data[label].insert(it_data + t + 1, value);
                    timeseries_datevalue[label].insert(it_datevalue + t + 1, date_value);
                    return;
                }
            }
        }
    }

    helios_runtime_error("ERROR (Context::addTimeseriesData): Failed to insert timeseries data for unknown reason.");
}

void Context::setCurrentTimeseriesPoint(const char *label, uint index) {
    if (timeseries_data.find(label) == timeseries_data.end()) { // does not exist
        helios_runtime_error("ERROR (setCurrentTimeseriesPoint): Timeseries variable `" + std::string(label) + "' does not exist.");
    }
    setDate(queryTimeseriesDate(label, index));
    setTime(queryTimeseriesTime(label, index));
}

float Context::queryTimeseriesData(const char *label, const Date &date, const Time &time) const {
    if (timeseries_data.find(label) == timeseries_data.end()) { // does not exist
        helios_runtime_error("ERROR (setCurrentTimeseriesData): Timeseries variable `" + std::string(label) + "' does not exist.");
    }

    double date_value = floor(date.year * 366.25) + date.JulianDay();
    date_value += double(time.hour) / 24. + double(time.minute) / 1440. + double(time.second) / 86400.;

    double tmin = timeseries_datevalue.at(label).front();
    double tmax = timeseries_datevalue.at(label).back();

    if (date_value < tmin) {
        std::cerr << "WARNING (queryTimeseriesData): Timeseries date and time is outside of the range of the data. Using the earliest data point in the timeseries." << std::endl;
        return timeseries_data.at(label).front();
    } else if (date_value > tmax) {
        std::cerr << "WARNING (queryTimeseriesData): Timeseries date and time is outside of the range of the data. Using the latest data point in the timeseries." << std::endl;
        return timeseries_data.at(label).back();
    }

    if (timeseries_datevalue.at(label).empty()) {
        std::cerr << "WARNING (queryTimeseriesData): timeseries " << label << " does not contain any data." << std::endl;
        return 0;
    } else if (timeseries_datevalue.at(label).size() == 1) {
        return timeseries_data.at(label).front();
    } else {
        int i;
        bool success = false;
        for (i = 0; i < timeseries_data.at(label).size() - 1; i++) {
            if (date_value >= timeseries_datevalue.at(label).at(i) && date_value <= timeseries_datevalue.at(label).at(i + 1)) {
                success = true;
                break;
            }
        }

        if (!success) {
            helios_runtime_error("ERROR (queryTimeseriesData): Failed to query timeseries data for unknown reason.");
        }

        double xminus = timeseries_data.at(label).at(i);
        double xplus = timeseries_data.at(label).at(i + 1);

        double tminus = timeseries_datevalue.at(label).at(i);
        double tplus = timeseries_datevalue.at(label).at(i + 1);

        return float(xminus + (xplus - xminus) * (date_value - tminus) / (tplus - tminus));
    }
}

float Context::queryTimeseriesData(const char *label) const {
    return queryTimeseriesData(label, sim_date, sim_time);
}

float Context::queryTimeseriesData(const char *label, const uint index) const {
    if (timeseries_data.find(label) == timeseries_data.end()) { // does not exist
        helios_runtime_error("ERROR( Context::getTimeseriesData): Timeseries variable " + std::string(label) + " does not exist.");
    }

    return timeseries_data.at(label).at(index);
}

Time Context::queryTimeseriesTime(const char *label, const uint index) const {
    if (timeseries_data.find(label) == timeseries_data.end()) { // does not exist
        helios_runtime_error("ERROR( Context::getTimeseriesTime): Timeseries variable " + std::string(label) + " does not exist.");
    }

    double dateval = timeseries_datevalue.at(label).at(index);

    int year = floor(floor(dateval) / 366.25);
    assert(year > 1000 && year < 10000);

    int JD = floor(dateval - floor(double(year) * 366.25));
    assert(JD > 0 && JD < 367);

    int hour = floor((dateval - floor(dateval)) * 24.);
    int minute = floor(((dateval - floor(dateval)) * 24. - double(hour)) * 60.);
    int second = (int) lround((((dateval - floor(dateval)) * 24. - double(hour)) * 60. - double(minute)) * 60.);

    if (second == 60) {
        second = 0;
        minute++;
    }

    if (minute == 60) {
        minute = 0;
        hour++;
    }

    assert(second >= 0 && second < 60);
    assert(minute >= 0 && minute < 60);
    assert(hour >= 0 && hour < 24);

    return make_Time(hour, minute, second);
}

Date Context::queryTimeseriesDate(const char *label, const uint index) const {
    if (timeseries_data.find(label) == timeseries_data.end()) { // does not exist
        helios_runtime_error("ERROR( Context::getTimeseriesDate): Timeseries variable " + std::string(label) + " does not exist.");
    }

    double dateval = timeseries_datevalue.at(label).at(index);

    int year = floor(floor(dateval) / 366.25);
    assert(year > 1000 && year < 10000);

    int JD = floor(dateval - floor(double(year) * 366.25));
    assert(JD > 0 && JD < 367);

    return Julian2Calendar(JD, year);
}

uint Context::getTimeseriesLength(const char *label) const {
    uint size = 0;
    if (timeseries_data.find(label) == timeseries_data.end()) { // does not exist
        helios_runtime_error("ERROR (Context::getTimeseriesDate): Timeseries variable `" + std::string(label) + "' does not exist.");
    } else {
        size = timeseries_data.at(label).size();
    }

    return size;
}

bool Context::doesTimeseriesVariableExist(const char *label) const {
    if (timeseries_data.find(label) == timeseries_data.end()) { // does not exist
        return false;
    } else {
        return true;
    }
}

std::vector<std::string> Context::listTimeseriesVariables() const {
    std::vector<std::string> labels;
    labels.reserve(timeseries_data.size());
    for (const auto &[timeseries_label, timeseries_data]: timeseries_data) {
        labels.push_back(timeseries_label);
    }
    return labels;
}


void Context::getDomainBoundingBox(vec2 &xbounds, vec2 &ybounds, vec2 &zbounds) const {
    getDomainBoundingBox(getAllUUIDs(), xbounds, ybounds, zbounds);
}

void Context::getDomainBoundingBox(const std::vector<uint> &UUIDs, vec2 &xbounds, vec2 &ybounds, vec2 &zbounds) const {
    // Global bounding box initialization
    xbounds.x = 1e8; // global min x
    xbounds.y = -1e8; // global max x
    ybounds.x = 1e8; // global min y
    ybounds.y = -1e8; // global max y
    zbounds.x = 1e8; // global min z
    zbounds.y = -1e8; // global max z

    // Parallel region over the primitives (UUIDs)
#ifdef USE_OPENMP
#pragma omp parallel
    {
        // Each thread creates its own local bounding box.
        float local_xmin = 1e8, local_xmax = -1e8;
        float local_ymin = 1e8, local_ymax = -1e8;
        float local_zmin = 1e8, local_zmax = -1e8;

// Parallelize the outer loop over primitives. Use "for" inside the parallel region.
#pragma omp for nowait
        for (int i = 0; i < (int)UUIDs.size(); i++) {
            // For each primitive:
            const std::vector<vec3> &verts = getPrimitivePointer_private(UUIDs[i])->getVertices();
            // Update local bounding box for each vertex in this primitive.
            for (const auto &vert: verts) {
                local_xmin = std::min(local_xmin, vert.x);
                local_xmax = std::max(local_xmax, vert.x);
                local_ymin = std::min(local_ymin, vert.y);
                local_ymax = std::max(local_ymax, vert.y);
                local_zmin = std::min(local_zmin, vert.z);
                local_zmax = std::max(local_zmax, vert.z);
            }
        }

// Merge the thread-local bounds into the global bounds.
#pragma omp critical
        {
            xbounds.x = std::min(xbounds.x, local_xmin);
            xbounds.y = std::max(xbounds.y, local_xmax);
            ybounds.x = std::min(ybounds.x, local_ymin);
            ybounds.y = std::max(ybounds.y, local_ymax);
            zbounds.x = std::min(zbounds.x, local_zmin);
            zbounds.y = std::max(zbounds.y, local_zmax);
        }
    } // end parallel region

#else

    for (uint UUID: UUIDs) {
        const std::vector<vec3> &verts = getPrimitivePointer_private(UUID)->getVertices();

        for (auto &vert: verts) {
            if (vert.x < xbounds.x) {
                xbounds.x = vert.x;
            } else if (vert.x > xbounds.y) {
                xbounds.y = vert.x;
            }
            if (vert.y < ybounds.x) {
                ybounds.x = vert.y;
            } else if (vert.y > ybounds.y) {
                ybounds.y = vert.y;
            }
            if (vert.z < zbounds.x) {
                zbounds.x = vert.z;
            } else if (vert.z > zbounds.y) {
                zbounds.y = vert.z;
            }
        }
    }

#endif
}

void Context::getDomainBoundingSphere(vec3 &center, float &radius) const {
    vec2 xbounds, ybounds, zbounds;
    getDomainBoundingBox(xbounds, ybounds, zbounds);

    center.x = xbounds.x + 0.5f * (xbounds.y - xbounds.x);
    center.y = ybounds.x + 0.5f * (ybounds.y - ybounds.x);
    center.z = zbounds.x + 0.5f * (zbounds.y - zbounds.x);

    radius = 0.5f * sqrtf(powf(xbounds.y - xbounds.x, 2) + powf(ybounds.y - ybounds.x, 2) + powf((zbounds.y - zbounds.x), 2));
}

void Context::getDomainBoundingSphere(const std::vector<uint> &UUIDs, vec3 &center, float &radius) const {
    vec2 xbounds, ybounds, zbounds;
    getDomainBoundingBox(UUIDs, xbounds, ybounds, zbounds);

    center.x = xbounds.x + 0.5f * (xbounds.y - xbounds.x);
    center.y = ybounds.x + 0.5f * (ybounds.y - ybounds.x);
    center.z = zbounds.x + 0.5f * (zbounds.y - zbounds.x);

    radius = 0.5f * sqrtf(powf(xbounds.y - xbounds.x, 2) + powf(ybounds.y - ybounds.x, 2) + powf((zbounds.y - zbounds.x), 2));
}

void Context::cropDomainX(const vec2 &xbounds) {
    const std::vector<uint> &UUIDs_all = getAllUUIDs();

    for (uint p: UUIDs_all) {
        const std::vector<vec3> &vertices = getPrimitivePointer_private(p)->getVertices();

        for (auto &vertex: vertices) {
            if (vertex.x < xbounds.x || vertex.x > xbounds.y) {
                deletePrimitive(p);
                break;
            }
        }
    }

    if (getPrimitiveCount() == 0) {
        std::cerr << "WARNING (Context::cropDomainX): No primitives were inside cropped area, and thus all primitives were deleted." << std::endl;
    }
}

void Context::cropDomainY(const vec2 &ybounds) {
    const std::vector<uint> &UUIDs_all = getAllUUIDs();

    for (uint p: UUIDs_all) {
        const std::vector<vec3> &vertices = getPrimitivePointer_private(p)->getVertices();

        for (auto &vertex: vertices) {
            if (vertex.y < ybounds.x || vertex.y > ybounds.y) {
                deletePrimitive(p);
                break;
            }
        }
    }

    if (getPrimitiveCount() == 0) {
        std::cerr << "WARNING (Context::cropDomainY): No primitives were inside cropped area, and thus all primitives were deleted." << std::endl;
    }
}

void Context::cropDomainZ(const vec2 &zbounds) {
    const std::vector<uint> &UUIDs_all = getAllUUIDs();

    for (uint p: UUIDs_all) {
        const std::vector<vec3> &vertices = getPrimitivePointer_private(p)->getVertices();

        for (auto &vertex: vertices) {
            if (vertex.z < zbounds.x || vertex.z > zbounds.y) {
                deletePrimitive(p);
                break;
            }
        }
    }

    if (getPrimitiveCount() == 0) {
        std::cerr << "WARNING (Context::cropDomainZ): No primitives were inside cropped area, and thus all primitives were deleted." << std::endl;
    }
}

void Context::cropDomain(std::vector<uint> &UUIDs, const vec2 &xbounds, const vec2 &ybounds, const vec2 &zbounds) {
    size_t delete_count = 0;
    for (uint UUID: UUIDs) {
        const std::vector<vec3> &vertices = getPrimitivePointer_private(UUID)->getVertices();

        for (auto &vertex: vertices) {
            if (vertex.x < xbounds.x || vertex.x > xbounds.y || vertex.y < ybounds.x || vertex.y > ybounds.y || vertex.z < zbounds.x || vertex.z > zbounds.y) {
                deletePrimitive(UUID);
                delete_count++;
                break;
            }
        }
    }

    if (delete_count == UUIDs.size()) {
        std::cerr << "WARNING (Context::cropDomain): No specified primitives were entirely inside cropped area, and thus all specified primitives were deleted." << std::endl;
    }

    cleanDeletedUUIDs(UUIDs);
}

void Context::cropDomain(const vec2 &xbounds, const vec2 &ybounds, const vec2 &zbounds) {
    std::vector<uint> UUIDs = getAllUUIDs();
    cropDomain(UUIDs, xbounds, ybounds, zbounds);
}


bool Context::areObjectPrimitivesComplete(uint objID) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(objID)) {
        helios_runtime_error("ERROR (Context::areObjectPrimitivesComplete): Object ID of " + std::to_string(objID) + " does not exist in the context.");
    }
#endif
    return getObjectPointer(objID)->arePrimitivesComplete();
}

void Context::cleanDeletedObjectIDs(std::vector<uint> &objIDs) const {
    for (auto it = objIDs.begin(); it != objIDs.end();) {
        if (!doesObjectExist(*it)) {
            it = objIDs.erase(it);
        } else {
            ++it;
        }
    }
}

void Context::cleanDeletedObjectIDs(std::vector<std::vector<uint>> &objIDs) const {
    for (auto &vec: objIDs) {
        for (auto it = vec.begin(); it != vec.end();) {
            if (!doesObjectExist(*it)) {
                it = vec.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void Context::cleanDeletedObjectIDs(std::vector<std::vector<std::vector<uint>>> &objIDs) const {
    for (auto &vec2D: objIDs) {
        for (auto &vec: vec2D) {
            for (auto it = vec.begin(); it != vec.end();) {
                if (!doesObjectExist(*it)) {
                    it = vec.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
}

CompoundObject *Context::getObjectPointer(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    return objects.at(ObjID);
}

uint Context::getObjectCount() const {
    return objects.size();
}

bool Context::doesObjectExist(const uint ObjID) const {
    return objects.find(ObjID) != objects.end();
}

std::vector<uint> Context::getAllObjectIDs() const {
    std::vector<uint> objIDs;
    objIDs.reserve(objects.size());
    size_t i = 0;
    for (auto [objID, object]: objects) {
        if (object->ishidden) {
            continue;
        }
        objIDs.push_back(objID);
        i++;
    }
    return objIDs;
}

void Context::deleteObject(const std::vector<uint> &ObjIDs) {
    for (const uint ObjID: ObjIDs) {
        deleteObject(ObjID);
    }
}

void Context::deleteObject(uint ObjID) {
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::deleteObject): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }

    CompoundObject *obj = objects.at(ObjID);

    for (const auto &[label, type]: obj->object_data_types) {
        decrementObjectDataLabelCounter(label);
    }

    const std::vector<uint> &UUIDs = obj->getPrimitiveUUIDs();


    delete obj;
    objects.erase(ObjID);

    deletePrimitive(UUIDs);
}

std::vector<uint> Context::copyObject(const std::vector<uint> &ObjIDs) {
    std::vector<uint> ObjIDs_copy(ObjIDs.size());
    size_t i = 0;
    for (uint ObjID: ObjIDs) {
        ObjIDs_copy.at(i) = copyObject(ObjID);
        i++;
    }

    return ObjIDs_copy;
}

uint Context::copyObject(uint ObjID) {
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::copyObject): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }

    ObjectType type = objects.at(ObjID)->getObjectType();

    const std::vector<uint> &UUIDs = getObjectPointer(ObjID)->getPrimitiveUUIDs();

    const std::vector<uint> &UUIDs_copy = copyPrimitive(UUIDs);
    for (uint p: UUIDs_copy) {
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
    }

    const std::string &texturefile = objects.at(ObjID)->getTextureFile();

    if (type == OBJECT_TYPE_TILE) {
        Tile *o = getTileObjectPointer(ObjID);

        const int2 &subdiv = o->getSubdivisionCount();

        auto *tile_new = (new Tile(currentObjectID, UUIDs_copy, subdiv, texturefile.c_str(), this));

        objects[currentObjectID] = tile_new;
    } else if (type == OBJECT_TYPE_SPHERE) {
        Sphere *o = getSphereObjectPointer(ObjID);

        uint subdiv = o->getSubdivisionCount();

        auto *sphere_new = (new Sphere(currentObjectID, UUIDs_copy, subdiv, texturefile.c_str(), this));

        objects[currentObjectID] = sphere_new;
    } else if (type == OBJECT_TYPE_TUBE) {
        Tube *o = getTubeObjectPointer(ObjID);

        const std::vector<vec3> &nodes = o->getNodes();
        const std::vector<float> &radius = o->getNodeRadii();
        const std::vector<RGBcolor> &colors = o->getNodeColors();
        const std::vector<std::vector<vec3>> &triangle_vertices = o->getTriangleVertices();
        uint subdiv = o->getSubdivisionCount();

        auto *tube_new = (new Tube(currentObjectID, UUIDs_copy, nodes, radius, colors, triangle_vertices, subdiv, texturefile.c_str(), this));

        objects[currentObjectID] = tube_new;
    } else if (type == OBJECT_TYPE_BOX) {
        Box *o = getBoxObjectPointer(ObjID);

        const int3 &subdiv = o->getSubdivisionCount();

        auto *box_new = (new Box(currentObjectID, UUIDs_copy, subdiv, texturefile.c_str(), this));

        objects[currentObjectID] = box_new;
    } else if (type == OBJECT_TYPE_DISK) {
        Disk *o = getDiskObjectPointer(ObjID);

        const int2 &subdiv = o->getSubdivisionCount();

        auto *disk_new = (new Disk(currentObjectID, UUIDs_copy, subdiv, texturefile.c_str(), this));

        objects[currentObjectID] = disk_new;
    } else if (type == OBJECT_TYPE_POLYMESH) {
        auto *polymesh_new = (new Polymesh(currentObjectID, UUIDs_copy, texturefile.c_str(), this));

        objects[currentObjectID] = polymesh_new;
    } else if (type == OBJECT_TYPE_CONE) {
        Cone *o = getConeObjectPointer(ObjID);

        const std::vector<vec3> &nodes = o->getNodeCoordinates();
        const std::vector<float> &radius = o->getNodeRadii();
        uint subdiv = o->getSubdivisionCount();

        auto *cone_new = (new Cone(currentObjectID, UUIDs_copy, nodes.at(0), nodes.at(1), radius.at(0), radius.at(1), subdiv, texturefile.c_str(), this));

        objects[currentObjectID] = cone_new;
    }

    copyObjectData(ObjID, currentObjectID);

    float T[16];
    getObjectPointer(ObjID)->getTransformationMatrix(T);

    getObjectPointer(currentObjectID)->setTransformationMatrix(T);

    currentObjectID++;
    return currentObjectID - 1;
}

std::vector<uint> Context::filterObjectsByData(const std::vector<uint> &IDs, const char *object_data, float threshold, const char *comparator) const {
    std::vector<uint> output_object_IDs;
    output_object_IDs.resize(IDs.size());
    uint passed_count = 0;

    for (uint i = 0; i < IDs.size(); i++) {
        if (doesObjectDataExist(IDs.at(i), object_data)) {
            HeliosDataType type = getObjectDataType(object_data);
            if (type == HELIOS_TYPE_UINT) {
                uint R;
                getObjectData(IDs.at(i), object_data, R);
                if (strcmp(comparator, "<") == 0) {
                    if (float(R) < threshold) {
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                } else if (strcmp(comparator, ">") == 0) {
                    if (float(R) > threshold) {
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                } else if (strcmp(comparator, "=") == 0) {
                    if (float(R) == threshold) {
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                }
            } else if (type == HELIOS_TYPE_FLOAT) {
                float R;
                getObjectData(IDs.at(i), object_data, R);

                if (strcmp(comparator, "<") == 0) {
                    if (R < threshold) {
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                } else if (strcmp(comparator, ">") == 0) {
                    if (R > threshold) {
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                } else if (strcmp(comparator, "=") == 0) {
                    if (R == threshold) {
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                }
            } else if (type == HELIOS_TYPE_INT) {
                int R;
                getObjectData(IDs.at(i), object_data, R);

                if (strcmp(comparator, "<") == 0) {
                    if (float(R) < threshold) {
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                } else if (strcmp(comparator, ">") == 0) {
                    if (float(R) > threshold) {
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                } else if (strcmp(comparator, "=") == 0) {
                    if (float(R) == threshold) {
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                }
            } else {
                std::cerr << "WARNING: Object data not of type UINT, INT, or FLOAT. Filtering for other types not yet supported." << std::endl;
            }
        }
    }

    output_object_IDs.resize(passed_count);

    return output_object_IDs;
}

void Context::translateObject(uint ObjID, const vec3 &shift) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::translateObject): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    getObjectPointer(ObjID)->translate(shift);
}

void Context::translateObject(const std::vector<uint> &ObjIDs, const vec3 &shift) const {
    for (uint ID: ObjIDs) {
        translateObject(ID, shift);
    }
}

void Context::rotateObject(uint ObjID, float rotation_radians, const char *rotation_axis_xyz) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::rotateObject): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    getObjectPointer(ObjID)->rotate(rotation_radians, rotation_axis_xyz);
}

void Context::rotateObject(const std::vector<uint> &ObjIDs, float rotation_radians, const char *rotation_axis_xyz) const {
    for (uint ID: ObjIDs) {
        rotateObject(ID, rotation_radians, rotation_axis_xyz);
    }
}

void Context::rotateObject(uint ObjID, float rotation_radians, const vec3 &rotation_axis_vector) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::rotateObject): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    getObjectPointer(ObjID)->rotate(rotation_radians, rotation_axis_vector);
}

void Context::rotateObject(const std::vector<uint> &ObjIDs, float rotation_radians, const vec3 &rotation_axis_vector) const {
    for (uint ID: ObjIDs) {
        rotateObject(ID, rotation_radians, rotation_axis_vector);
    }
}

void Context::rotateObject(uint ObjID, float rotation_radians, const vec3 &rotation_origin, const vec3 &rotation_axis_vector) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::rotateObject): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    getObjectPointer(ObjID)->rotate(rotation_radians, rotation_origin, rotation_axis_vector);
}

void Context::rotateObject(const std::vector<uint> &ObjIDs, float rotation_radians, const vec3 &rotation_origin, const vec3 &rotation_axis_vector) const {
    for (uint ID: ObjIDs) {
        rotateObject(ID, rotation_radians, rotation_origin, rotation_axis_vector);
    }
}

void Context::rotateObjectAboutOrigin(uint ObjID, float rotation_radians, const vec3 &rotation_axis_vector) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::rotateObjectAboutOrigin): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    getObjectPointer(ObjID)->rotate(rotation_radians, objects.at(ObjID)->object_origin, rotation_axis_vector);
}

void Context::rotateObjectAboutOrigin(const std::vector<uint> &ObjIDs, float rotation_radians, const vec3 &rotation_axis_vector) const {
    for (uint ID: ObjIDs) {
        rotateObject(ID, rotation_radians, objects.at(ID)->object_origin, rotation_axis_vector);
    }
}

void Context::scaleObject(uint ObjID, const helios::vec3 &scalefact) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::scaleObject): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    getObjectPointer(ObjID)->scale(scalefact);
}

void Context::scaleObject(const std::vector<uint> &ObjIDs, const helios::vec3 &scalefact) const {
    for (uint ID: ObjIDs) {
        scaleObject(ID, scalefact);
    }
}

void Context::scaleObjectAboutCenter(uint ObjID, const helios::vec3 &scalefact) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::scaleObjectAboutCenter): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    getObjectPointer(ObjID)->scaleAboutCenter(scalefact);
}

void Context::scaleObjectAboutCenter(const std::vector<uint> &ObjIDs, const helios::vec3 &scalefact) const {
    for (uint ID: ObjIDs) {
        scaleObjectAboutCenter(ID, scalefact);
    }
}

void Context::scaleObjectAboutPoint(uint ObjID, const helios::vec3 &scalefact, const helios::vec3 &point) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::scaleObjectAboutPoint): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    getObjectPointer(ObjID)->scaleAboutPoint(scalefact, point);
}

void Context::scaleObjectAboutPoint(const std::vector<uint> &ObjIDs, const helios::vec3 &scalefact, const helios::vec3 &point) const {
    for (uint ID: ObjIDs) {
        scaleObjectAboutPoint(ID, scalefact, point);
    }
}

void Context::scaleObjectAboutOrigin(uint ObjID, const helios::vec3 &scalefact) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::scaleObjectAboutOrigin): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    getObjectPointer(ObjID)->scaleAboutPoint(scalefact, objects.at(ObjID)->object_origin);
}

void Context::scaleObjectAboutOrigin(const std::vector<uint> &ObjIDs, const helios::vec3 &scalefact) const {
    for (uint ID: ObjIDs) {
        scaleObjectAboutPoint(ID, scalefact, objects.at(ID)->object_origin);
    }
}

std::vector<uint> Context::getObjectPrimitiveUUIDs(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID) && ObjID != 0) {
        helios_runtime_error("ERROR (Context::getObjectPrimitiveUUIDs): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif

    if (ObjID == 0) {
        // \todo This is inefficient and should be improved by storing the UUIDs for all objID = 0 primitives in the Context.
        std::vector<uint> UUIDs;
        UUIDs.reserve(getPrimitiveCount());
        for (uint UUID: getAllUUIDs()) {
            if (getPrimitiveParentObjectID(UUID) == 0) {
                UUIDs.push_back(UUID);
            }
        }
        return UUIDs;
    }

    return getObjectPointer(ObjID)->getPrimitiveUUIDs();
}

std::vector<uint> Context::getObjectPrimitiveUUIDs(const std::vector<uint> &ObjIDs) const {
    std::vector<uint> output_UUIDs;

    for (uint ObjID: ObjIDs) {
#ifdef HELIOS_DEBUG
        if (!doesObjectExist(ObjID)) {
            helios_runtime_error("ERROR (Context::getObjectPrimitiveUUIDs): Object ID of " + std::to_string(ObjID) + " not found in the context.");
        }
#endif
        const std::vector<uint> &current_UUIDs = getObjectPrimitiveUUIDs(ObjID);
        output_UUIDs.insert(output_UUIDs.end(), current_UUIDs.begin(), current_UUIDs.end());
    }
    return output_UUIDs;
}

std::vector<uint> Context::getObjectPrimitiveUUIDs(const std::vector<std::vector<uint>> &ObjIDs) const {
    std::vector<uint> output_UUIDs;

    for (uint j = 0; j < ObjIDs.size(); j++) {
        for (uint i = 0; i < ObjIDs.at(j).size(); i++) {
#ifdef HELIOS_DEBUG
            if (!doesObjectExist(ObjIDs.at(j).at(i))) {
                helios_runtime_error("ERROR (Context::getObjectPrimitiveUUIDs): Object ID of " + std::to_string(ObjIDs.at(j).at(i)) + " not found in the context.");
            }
#endif

            const std::vector<uint> &current_UUIDs = getObjectPointer(ObjIDs.at(j).at(i))->getPrimitiveUUIDs();
            output_UUIDs.insert(output_UUIDs.end(), current_UUIDs.begin(), current_UUIDs.end());
        }
    }
    return output_UUIDs;
}

helios::ObjectType Context::getObjectType(uint ObjID) const {
    if (ObjID == 0) {
        return OBJECT_TYPE_NONE;
    }
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::getObjectType): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    return getObjectPointer(ObjID)->getObjectType();
}

float Context::getTileObjectAreaRatio(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::getTileObjectAreaRatio): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    if (getObjectPointer(ObjID)->getObjectType() != OBJECT_TYPE_TILE) {
        std::cerr << "WARNING (Context::getTileObjectAreaRatio): ObjectID " << ObjID << " is not a tile object. Skipping..." << std::endl;
        return 0.0;
    }

    if (!(getObjectPointer(ObjID)->arePrimitivesComplete())) {
        std::cerr << "WARNING (Context::getTileObjectAreaRatio): ObjectID " << ObjID << " is missing primitives. Area ratio calculated is area of non-missing subpatches divided by the area of an individual subpatch." << std::endl;
    }

    const int2 &subdiv = getTileObjectPointer(ObjID)->getSubdivisionCount();
    if (subdiv.x == 1 && subdiv.y == 1) {
        return 1.0;
    }

    float area = getTileObjectPointer(ObjID)->getArea();
    const vec2 size = getTileObjectPointer(ObjID)->getSize();

    float subpatch_area = size.x * size.y / scast<float>(subdiv.x * subdiv.y);
    return area / subpatch_area;
}

std::vector<float> Context::getTileObjectAreaRatio(const std::vector<uint> &ObjIDs) const {
    std::vector<float> AreaRatios(ObjIDs.size());
    for (uint i = 0; i < ObjIDs.size(); i++) {
        AreaRatios.at(i) = getTileObjectAreaRatio(ObjIDs.at(i));
    }

    return AreaRatios;
}

void Context::setTileObjectSubdivisionCount(const std::vector<uint> &ObjIDs, const int2 &new_subdiv) {
    // check that all objects are Tile Objects, and get vector of texture files
    std::vector<uint> tile_ObjectIDs;
    std::vector<uint> textured_tile_ObjectIDs;


    std::vector<std::string> tex;

    for (uint ObjID: ObjIDs) {
#ifdef HELIOS_DEBUG
        if (!doesObjectExist(ObjID)) {
            helios_runtime_error("ERROR (Context::setTileObjectSubdivisionCount): Object ID of " + std::to_string(ObjID) + " not found in the context.");
        }
#endif

        // check if the object ID is a tile object and if it is add it the tile_ObjectIDs vector
        if (getObjectPointer(ObjID)->getObjectType() != OBJECT_TYPE_TILE) {
            std::cerr << "WARNING (Context::setTileObjectSubdivisionCount): ObjectID " << ObjID << " is not a tile object. Skipping..." << std::endl;
        } else if (!(getObjectPointer(ObjID)->arePrimitivesComplete())) {
            std::cerr << "WARNING (Context::setTileObjectSubdivisionCount): ObjectID " << ObjID << " is missing primitives. Skipping..." << std::endl;
        } else {
            // test if the tile is textured and push into two different vectors
            Patch *p = getPatchPointer_private(getObjectPointer(ObjID)->getPrimitiveUUIDs().at(0));
            if (!p->hasTexture()) { // no texture
                tile_ObjectIDs.push_back(ObjID);
            } else { // texture
                textured_tile_ObjectIDs.push_back(ObjID);
                tex.push_back(p->getTextureFile());
            }
        }
    }

    // Here just call setSubdivisionCount directly for the non-textured tile objects
    for (unsigned int tile_ObjectID: tile_ObjectIDs) {
        Tile *current_object_pointer = getTileObjectPointer(tile_ObjectID);
        const std::vector<uint> &UUIDs_old = current_object_pointer->getPrimitiveUUIDs();

        vec2 size = current_object_pointer->getSize();
        vec3 center = current_object_pointer->getCenter();
        vec3 normal = current_object_pointer->getNormal();
        SphericalCoord rotation = cart2sphere(normal);
        RGBcolor color = getPrimitiveColor(UUIDs_old.front());

        std::vector<uint> UUIDs_new = addTile(center, size, rotation, new_subdiv, color);

        for (uint UUID: UUIDs_new) {
            getPrimitivePointer_private(UUID)->setParentObjectID(tile_ObjectID);
        }

        current_object_pointer->setPrimitiveUUIDs(UUIDs_new);
        current_object_pointer->setSubdivisionCount(new_subdiv);
        deletePrimitive(UUIDs_old);
    }

    // get a vector of unique texture files that are represented in the input tile objects
    sort(tex.begin(), tex.end());
    std::vector<std::string>::iterator it;
    it = std::unique(tex.begin(), tex.end());
    tex.resize(std::distance(tex.begin(), it));

    // create object templates for all the unique texture files
    std::vector<uint> object_templates;
    std::vector<std::vector<uint>> template_primitives;
    for (uint j = 0; j < tex.size(); j++) {
        // create a template object for the current texture
        uint object_template = addTileObject(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, new_subdiv, tex.at(j).c_str());
        object_templates.emplace_back(object_template);
        std::vector<uint> object_primitives = getTileObjectPointer(object_template)->getPrimitiveUUIDs();
        template_primitives.emplace_back(object_primitives);
    }

    // keep loop over objects on the outside, otherwise need to update textured_tile_ObjectIDs vector all the time
    // for each textured tile object
    for (uint i = 0; i < textured_tile_ObjectIDs.size(); i++) {
        // get info from current object
        Tile *current_object_pointer = getTileObjectPointer(textured_tile_ObjectIDs.at(i));
        std::string current_texture_file = current_object_pointer->getTextureFile();

        std::vector<uint> UUIDs_old = current_object_pointer->getPrimitiveUUIDs();

        vec2 size = current_object_pointer->getSize();
        vec3 center = current_object_pointer->getCenter();
        vec3 normal = current_object_pointer->getNormal();
        SphericalCoord rotation = cart2sphere(normal);

        // for unique textures
        for (uint j = 0; j < tex.size(); j++) {
            // if the current tile object has the same texture file as the current unique texture file
            if (current_texture_file == tex.at(j)) {
                // copy the template primitives and create a new tile with them
                std::vector<uint> new_primitives = copyPrimitive(template_primitives.at(j));

                // change the objectID for the new primitives
                setPrimitiveParentObjectID(new_primitives, textured_tile_ObjectIDs.at(i));
                current_object_pointer->setPrimitiveUUIDs(new_primitives);
                current_object_pointer->setSubdivisionCount(new_subdiv);

                // delete the original object primitives
                deletePrimitive(UUIDs_old);

                float IM[16];
                makeIdentityMatrix(IM);
                current_object_pointer->setTransformationMatrix(IM);

                current_object_pointer->scale(make_vec3(size.x, size.y, 1));

                // transform based on original object data
                if (rotation.elevation != 0) {
                    current_object_pointer->rotate(-rotation.elevation, "x");
                }
                if (rotation.azimuth != 0) {
                    current_object_pointer->rotate(rotation.azimuth, "z");
                }
                current_object_pointer->translate(center);
            }
        }
    }


    // delete the template (objects and primitives)
    deleteObject(object_templates);
}

void Context::setTileObjectSubdivisionCount(const std::vector<uint> &ObjIDs, float area_ratio) {
    // check that all objects are Tile Objects, and get vector of texture files
    std::vector<uint> tile_ObjectIDs;
    std::vector<uint> textured_tile_ObjectIDs;

    std::vector<std::string> tex;
    // for(uint i=1;i<ObjectIDs.size();i++)
    for (uint ObjID: ObjIDs) {
#ifdef HELIOS_DEBUG
        if (!doesObjectExist(ObjID)) {
            helios_runtime_error("ERROR (Context::setTileObjectSubdivisionCount): Object ID of " + std::to_string(ObjID) + " not found in the context.");
        }
#endif

        // check if the object ID is a tile object and if it is add it the tile_ObjectIDs vector
        if (getObjectPointer(ObjID)->getObjectType() != OBJECT_TYPE_TILE) {
            std::cerr << "WARNING (Context::setTileObjectSubdivisionCount): ObjectID " << ObjID << " is not a tile object. Skipping..." << std::endl;
        } else if (!(getObjectPointer(ObjID)->arePrimitivesComplete())) {
            std::cerr << "WARNING (Context::setTileObjectSubdivisionCount): ObjectID " << ObjID << " is missing primitives. Skipping..." << std::endl;
        } else {
            // test if the tile is textured and push into two different vectors
            Patch *p = getPatchPointer_private(getObjectPointer(ObjID)->getPrimitiveUUIDs().at(0));
            if (!p->hasTexture()) { // no texture
                tile_ObjectIDs.push_back(ObjID);
            } else { // texture
                textured_tile_ObjectIDs.push_back(ObjID);
                tex.push_back(p->getTextureFile());
            }
        }
    }

    // Here just call setSubdivisionCount directly for the non-textured tile objects
    for (uint i = 0; i < tile_ObjectIDs.size(); i++) {
        Tile *current_object_pointer = getTileObjectPointer(tile_ObjectIDs.at(i));
        std::vector<uint> UUIDs_old = current_object_pointer->getPrimitiveUUIDs();

        vec2 size = current_object_pointer->getSize();
        vec3 center = current_object_pointer->getCenter();
        vec3 normal = current_object_pointer->getNormal();
        SphericalCoord rotation = cart2sphere(normal);
        RGBcolor color = getPrimitiveColor(UUIDs_old.front());

        float tile_area = current_object_pointer->getArea();

        // subpatch dimensions needed to keep the correct ratio and have the solid fraction area = the input area
        float subpatch_dimension = sqrtf(tile_area / area_ratio);
        float subpatch_per_x = size.x / subpatch_dimension;
        float subpatch_per_y = size.y / subpatch_dimension;

        float option_1_AR = (tile_area / (size.x / ceil(subpatch_per_x) * size.y / floor(subpatch_per_y))) - area_ratio;
        float option_2_AR = (tile_area / (size.x / floor(subpatch_per_x) * size.y / ceil(subpatch_per_y))) - area_ratio;

        int2 new_subdiv;
        if ((int) area_ratio == 1) {
            new_subdiv = make_int2(1, 1);
        } else if (option_1_AR >= option_2_AR) {
            new_subdiv = make_int2(ceil(subpatch_per_x), floor(subpatch_per_y));
        } else {
            new_subdiv = make_int2(floor(subpatch_per_x), ceil(subpatch_per_y));
        }


        std::vector<uint> UUIDs_new = addTile(center, size, rotation, new_subdiv, color);

        for (uint UUID: UUIDs_new) {
            getPrimitivePointer_private(UUID)->setParentObjectID(tile_ObjectIDs.at(i));
        }

        current_object_pointer->setPrimitiveUUIDs(UUIDs_new);
        current_object_pointer->setSubdivisionCount(new_subdiv);
        deletePrimitive(UUIDs_old);
    }

    // get a vector of unique texture files that are represented in the input tile objects
    sort(tex.begin(), tex.end());
    std::vector<std::string>::iterator it;
    it = std::unique(tex.begin(), tex.end());
    tex.resize(std::distance(tex.begin(), it));

    // create object templates for all the unique texture files
    //  the assumption here is that all tile objects with the same texture have the same aspect ratio
    // if this is not true then the copying method won't work well because a new template will need to be created for each texture/aspect ratio combination

    std::vector<uint> object_templates;
    std::vector<std::vector<uint>> template_primitives;
    for (uint j = 0; j < tex.size(); j++) {
        // here we just want to get one tile object with the matching texture
        uint ii;
        for (uint i = 0; i < textured_tile_ObjectIDs.size(); i++) {
            // get info from current object
            Tile *current_object_pointer_b = getTileObjectPointer(textured_tile_ObjectIDs.at(i));
            std::string current_texture_file_b = current_object_pointer_b->getTextureFile();
            // if the current tile object has the same texture file as the current unique texture file
            if (current_texture_file_b == tex.at(j)) {
                ii = i;
                break;
            }
        }

        // get info from current object
        Tile *current_object_pointer = getTileObjectPointer(textured_tile_ObjectIDs.at(ii));
        vec2 tile_size = current_object_pointer->getSize();
        float tile_area = current_object_pointer->getArea();

        // subpatch dimensions needed to keep the correct ratio and have the solid fraction area = the input area
        float subpatch_dimension = sqrtf(tile_area / area_ratio);
        float subpatch_per_x = tile_size.x / subpatch_dimension;
        float subpatch_per_y = tile_size.y / subpatch_dimension;

        float option_1_AR = (tile_area / (tile_size.x / ceil(subpatch_per_x) * tile_size.y / floor(subpatch_per_y))) - area_ratio;
        float option_2_AR = (tile_area / (tile_size.x / floor(subpatch_per_x) * tile_size.y / ceil(subpatch_per_y))) - area_ratio;

        int2 new_subdiv;
        if ((int) area_ratio == 1) {
            new_subdiv = make_int2(1, 1);
        } else if (option_1_AR >= option_2_AR) {
            new_subdiv = make_int2(ceil(subpatch_per_x), floor(subpatch_per_y));
        } else {
            new_subdiv = make_int2(floor(subpatch_per_x), ceil(subpatch_per_y));
        }

        // create a template object for the current texture
        uint object_template = addTileObject(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, new_subdiv, tex.at(j).c_str());
        object_templates.emplace_back(object_template);
        std::vector<uint> object_primitives = getTileObjectPointer(object_template)->getPrimitiveUUIDs();
        template_primitives.emplace_back(object_primitives);
    }

    // keep loop over objects on the outside, otherwise need to update textured_tile_ObjectIDs vector all the time
    // for each textured tile object
    for (uint i = 0; i < textured_tile_ObjectIDs.size(); i++) {
        // get info from current object
        Tile *current_object_pointer = getTileObjectPointer(textured_tile_ObjectIDs.at(i));
        // std::string current_texture_file = getPrimitivePointer_private(current_object_pointer->getPrimitiveUUIDs().at(0))->getTextureFile();
        std::string current_texture_file = current_object_pointer->getTextureFile();
        // std::cout << "current_texture_file for ObjID " << textured_tile_ObjectIDs.at(i) << " = " << current_texture_file << std::endl;
        std::vector<uint> UUIDs_old = current_object_pointer->getPrimitiveUUIDs();

        vec2 size = current_object_pointer->getSize();
        vec3 center = current_object_pointer->getCenter();
        vec3 normal = current_object_pointer->getNormal();
        SphericalCoord rotation = cart2sphere(normal);

        // for unique textures
        for (uint j = 0; j < tex.size(); j++) {
            // if the current tile object has the same texture file as the current unique texture file
            if (current_texture_file == tex.at(j)) {
                // copy the template primitives and create a new tile with them
                std::vector<uint> new_primitives = copyPrimitive(template_primitives.at(j));

                // change the objectID for the new primitives
                setPrimitiveParentObjectID(new_primitives, textured_tile_ObjectIDs.at(i));

                int2 new_subdiv = getTileObjectPointer(object_templates.at(j))->getSubdivisionCount();
                current_object_pointer->setPrimitiveUUIDs(new_primitives);
                current_object_pointer->setSubdivisionCount(new_subdiv);

                // delete the original object primitives
                deletePrimitive(UUIDs_old);

                float IM[16];
                makeIdentityMatrix(IM);
                current_object_pointer->setTransformationMatrix(IM);

                current_object_pointer->scale(make_vec3(size.x, size.y, 1));

                if (rotation.elevation != 0) {
                    current_object_pointer->rotate(-rotation.elevation, "x");
                }
                if (rotation.azimuth != 0) {
                    current_object_pointer->rotate(rotation.azimuth, "z");
                }
                current_object_pointer->translate(center);
            }
        }
    }

    // delete the template (objects and primitives)
    deleteObject(object_templates);
}


std::vector<uint> Context::addSphere(uint Ndivs, const vec3 &center, float radius) {
    RGBcolor color = make_RGBcolor(0.f, 0.75f, 0.f); // Default color is green

    return addSphere(Ndivs, center, radius, color);
}

std::vector<uint> Context::addSphere(uint Ndivs, const vec3 &center, float radius, const RGBcolor &color) {
    std::vector<uint> UUID;

    float dtheta = PI_F / float(Ndivs);
    float dphi = 2.0f * PI_F / float(Ndivs);

    // bottom cap
    for (int j = 0; j < Ndivs; j++) {
        vec3 v0 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F, 0));
        vec3 v1 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + dtheta, float(j) * dphi));
        vec3 v2 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + dtheta, float(j + 1) * dphi));

        UUID.push_back(addTriangle(v0, v1, v2, color));
    }

    // top cap
    for (int j = 0; j < Ndivs; j++) {
        vec3 v0 = center + sphere2cart(make_SphericalCoord(radius, 0.5f * PI_F, 0));
        vec3 v1 = center + sphere2cart(make_SphericalCoord(radius, 0.5f * PI_F - dtheta, float(j) * dphi));
        vec3 v2 = center + sphere2cart(make_SphericalCoord(radius, 0.5f * PI_F - dtheta, float(j + 1) * dphi));

        UUID.push_back(addTriangle(v2, v1, v0, color));
    }

    // middle
    for (int j = 0; j < Ndivs; j++) {
        for (int i = 1; i < Ndivs - 1; i++) {
            vec3 v0 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + float(i) * dtheta, float(j) * dphi));
            vec3 v1 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + float(i + 1) * dtheta, float(j) * dphi));
            vec3 v2 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + float(i + 1) * dtheta, float(j + 1) * dphi));
            vec3 v3 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + float(i) * dtheta, float(j + 1) * dphi));

            UUID.push_back(addTriangle(v0, v1, v2, color));
            UUID.push_back(addTriangle(v0, v2, v3, color));
        }
    }

    return UUID;
}

std::vector<uint> Context::addSphere(uint Ndivs, const vec3 &center, float radius, const char *texturefile) {
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addSphere): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addSphere): Texture file " + std::string(texturefile) + " does not exist.");
    }

    std::vector<uint> UUID;

    float dtheta = PI_F / float(Ndivs);
    float dphi = 2.0f * PI_F / float(Ndivs);

    // bottom cap
    for (int j = 0; j < Ndivs; j++) {
        vec3 v0 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F, 0));
        vec3 v1 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + dtheta, float(j) * dphi));
        vec3 v2 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + dtheta, float(j + 1) * dphi));

        vec3 n0 = v0 - center;
        n0.normalize();
        vec3 n1 = v1 - center;
        n1.normalize();
        vec3 n2 = v2 - center;
        n2.normalize();

        vec2 uv0 = make_vec2(1.f - atan2f(sin((float(j) + 0.5f) * dphi), -cos((float(j) + 0.5f) * dphi)) / (2.f * PI_F) - 0.5f, 1.f - n0.z * 0.5f - 0.5f);
        vec2 uv1 = make_vec2(1.f - atan2f(n1.x, -n1.y) / (2.f * PI_F) - 0.5f, 1.f - n1.z * 0.5f - 0.5f);
        vec2 uv2 = make_vec2(1.f - atan2f(n2.x, -n2.y) / (2.f * PI_F) - 0.5f, 1.f - n2.z * 0.5f - 0.5f);

        if (j == Ndivs - 1) {
            uv2.x = 1;
        }

        uint triangle_uuid = addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2);
        if (getPrimitiveArea(triangle_uuid) > 0) {
            UUID.push_back(triangle_uuid);
        } else {
            deletePrimitive(triangle_uuid);
        }
    }

    // top cap
    for (int j = 0; j < Ndivs; j++) {
        vec3 v0 = center + sphere2cart(make_SphericalCoord(radius, 0.5f * PI_F, 0));
        vec3 v1 = center + sphere2cart(make_SphericalCoord(radius, 0.5f * PI_F - dtheta, float(j + 1) * dphi));
        vec3 v2 = center + sphere2cart(make_SphericalCoord(radius, 0.5f * PI_F - dtheta, float(j) * dphi));

        vec3 n0 = v0 - center;
        n0.normalize();
        vec3 n1 = v1 - center;
        n1.normalize();
        vec3 n2 = v2 - center;
        n2.normalize();

        vec2 uv0 = make_vec2(1.f - atan2f(sinf((float(j) + 0.5f) * dphi), -cosf((float(j) + 0.5f) * dphi)) / (2.f * PI_F) - 0.5f, 1.f - n0.z * 0.5f - 0.5f);
        vec2 uv1 = make_vec2(1.f - atan2f(n1.x, -n1.y) / (2.f * PI_F) - 0.5f, 1.f - n1.z * 0.5f - 0.5f);
        vec2 uv2 = make_vec2(1.f - atan2f(n2.x, -n2.y) / (2.f * PI_F) - 0.5f, 1.f - n2.z * 0.5f - 0.5f);

        if (j == Ndivs - 1) {
            uv2.x = 1;
        }

        uint triangle_uuid = addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2);
        if (getPrimitiveArea(triangle_uuid) > 0) {
            UUID.push_back(triangle_uuid);
        } else {
            deletePrimitive(triangle_uuid);
        }
    }

    // middle
    for (int j = 0; j < Ndivs; j++) {
        for (int i = 1; i < Ndivs - 1; i++) {
            vec3 v0 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + float(i) * dtheta, float(j) * dphi));
            vec3 v1 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + float(i + 1) * dtheta, float(j) * dphi));
            vec3 v2 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + float(i + 1) * dtheta, float(j + 1) * dphi));
            vec3 v3 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + float(i) * dtheta, float(j + 1) * dphi));

            vec3 n0 = v0 - center;
            n0.normalize();
            vec3 n1 = v1 - center;
            n1.normalize();
            vec3 n2 = v2 - center;
            n2.normalize();
            vec3 n3 = v3 - center;
            n3.normalize();

            vec2 uv0 = make_vec2(1.f - atan2f(n0.x, -n0.y) / (2.f * PI_F) - 0.5f, 1.f - n0.z * 0.5f - 0.5f);
            vec2 uv1 = make_vec2(1.f - atan2f(n1.x, -n1.y) / (2.f * PI_F) - 0.5f, 1.f - n1.z * 0.5f - 0.5f);
            vec2 uv2 = make_vec2(1.f - atan2f(n2.x, -n2.y) / (2.f * PI_F) - 0.5f, 1.f - n2.z * 0.5f - 0.5f);
            vec2 uv3 = make_vec2(1.f - atan2f(n3.x, -n3.y) / (2.f * PI_F) - 0.5f, 1.f - n3.z * 0.5f - 0.5f);

            if (j == Ndivs - 1) {
                uv2.x = 1;
                uv3.x = 1;
            }

            uint triangle_uuid1 = addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2);
            if (getPrimitiveArea(triangle_uuid1) > 0) {
                UUID.push_back(triangle_uuid1);
            } else {
                deletePrimitive(triangle_uuid1);
            }
            uint triangle_uuid2 = addTriangle(v0, v2, v3, texturefile, uv0, uv2, uv3);
            if (getPrimitiveArea(triangle_uuid2) > 0) {
                UUID.push_back(triangle_uuid2);
            } else {
                deletePrimitive(triangle_uuid2);
            }
        }
    }

    return UUID;
}

std::vector<uint> Context::addTile(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv) {
    RGBcolor color(0.f, 0.75f, 0.f); // Default color is green

    return addTile(center, size, rotation, subdiv, color);
}

std::vector<uint> Context::addTile(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv, const RGBcolor &color) {
    vec2 subsize;
    subsize.x = size.x / float(subdiv.x);
    subsize.y = size.y / float(subdiv.y);

    std::vector<uint> UUID(subdiv.x * subdiv.y);

    size_t t = 0;
    for (uint j = 0; j < subdiv.y; j++) {
        for (uint i = 0; i < subdiv.x; i++) {
            vec3 subcenter = make_vec3(-0.5f * size.x + (float(i) + 0.5f) * subsize.x, -0.5f * size.y + (float(j) + 0.5f) * subsize.y, 0);

            UUID[t] = addPatch(subcenter, subsize, make_SphericalCoord(0, 0), color);

            if (rotation.elevation != 0.f) {
                getPrimitivePointer_private(UUID[t])->rotate(-rotation.elevation, "x");
            }
            if (rotation.azimuth != 0.f) {
                getPrimitivePointer_private(UUID[t])->rotate(-rotation.azimuth, "z");
            }
            getPrimitivePointer_private(UUID[t])->translate(center);

            t++;
        }
    }

    return UUID;
}

std::vector<uint> Context::addTile(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv, const char *texturefile) {
    return addTile(center, size, rotation, subdiv, texturefile, make_int2(1, 1));
}

std::vector<uint> Context::addTile(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv, const char *texturefile, const int2 &texture_repeat) {
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addTile): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addTile): Texture file " + std::string(texturefile) + " does not exist.");
    } else if (texture_repeat.x < 1 || texture_repeat.y < 1) {
        helios_runtime_error("ERROR (Context::addTile): Number of texture repeats must be greater than 0.");
    }

    // Automatically resize the repeat count so that it evenly divides the subdivisions.
    int2 repeat = texture_repeat;
    repeat.x = std::min(subdiv.x, repeat.x);
    repeat.y = std::min(subdiv.y, repeat.y);
    while (subdiv.x % repeat.x != 0) {
        repeat.x--;
    }
    while (subdiv.y % repeat.y != 0) {
        repeat.y--;
    }

    std::vector<uint> UUID;

    vec2 subsize;
    subsize.x = size.x / float(subdiv.x);
    subsize.y = size.y / float(subdiv.y);

    std::vector<helios::vec2> uv(4);
    int2 sub_per_repeat;
    sub_per_repeat.x = subdiv.x / repeat.x;
    sub_per_repeat.y = subdiv.y / repeat.y;
    vec2 uv_sub;
    uv_sub.x = 1.f / float(sub_per_repeat.x);
    uv_sub.y = 1.f / float(sub_per_repeat.y);

    addTexture(texturefile);

    const int2 &sz = textures.at(texturefile).getImageResolution();
    if (subdiv.x >= repeat.x * sz.x || subdiv.y >= repeat.y * sz.y) {
        helios_runtime_error("ERROR (Context::addTile): The resolution of the texture image '" + std::string(texturefile) + "' is lower than the number of tile subdivisions. Increase resolution of the texture image.");
    }

    for (uint j = 0; j < subdiv.y; j++) {
        for (uint i = 0; i < subdiv.x; i++) {
            vec3 subcenter = make_vec3(-0.5f * size.x + (float(i) + 0.5f) * subsize.x, -0.5f * size.y + (float(j) + 0.5f) * subsize.y, 0.f);

            uint i_local = i % sub_per_repeat.x;
            uint j_local = j % sub_per_repeat.y;
            uv.at(0) = make_vec2(float(i_local) * uv_sub.x, float(j_local) * uv_sub.y);
            uv.at(1) = make_vec2(float(i_local + 1) * uv_sub.x, float(j_local) * uv_sub.y);
            uv.at(2) = make_vec2(float(i_local + 1) * uv_sub.x, float(j_local + 1) * uv_sub.y);
            uv.at(3) = make_vec2(float(i_local) * uv_sub.x, float(j_local + 1) * uv_sub.y);

            auto *patch_new = (new Patch(texturefile, uv, textures, 0, currentUUID));

            if (patch_new->getSolidFraction() == 0) {
                delete patch_new;
                continue;
            }

            assert(size.x > 0.f && size.y > 0.f);
            patch_new->scale(make_vec3(subsize.x, subsize.y, 1));

            patch_new->translate(subcenter);

            if (rotation.elevation != 0) {
                patch_new->rotate(-rotation.elevation, "x");
            }
            if (rotation.azimuth != 0) {
                patch_new->rotate(-rotation.azimuth, "z");
            }

            patch_new->translate(center);

            primitives[currentUUID] = patch_new;
            currentUUID++;
            UUID.push_back(currentUUID - 1);
        }
    }

    return UUID;
}

std::vector<uint> Context::addTube(uint Ndivs, const std::vector<vec3> &nodes, const std::vector<float> &radius) {
    std::vector<RGBcolor> color(nodes.size(), make_RGBcolor(0.f, 0.75f, 0.f));

    return addTube(Ndivs, nodes, radius, color);
}

std::vector<uint> Context::addTube(uint radial_subdivisions, const std::vector<vec3> &nodes, const std::vector<float> &radius, const std::vector<RGBcolor> &color) {
    const uint node_count = nodes.size();

    if (node_count == 0) {
        helios_runtime_error("ERROR (Context::addTube): Node and radius arrays are empty.");
    } else if (node_count != radius.size()) {
        helios_runtime_error("ERROR (Context::addTube): Size of `nodes' and `radius' arguments must agree.");
    } else if (node_count != color.size()) {
        helios_runtime_error("ERROR (Context::addTube): Size of `nodes' and `color' arguments must agree.");
    }

    vec3 vec, convec;
    std::vector<float> cfact(radial_subdivisions + 1);
    std::vector<float> sfact(radial_subdivisions + 1);
    std::vector<std::vector<vec3>> xyz;
    resize_vector(xyz, node_count, radial_subdivisions + 1);

    vec3 nvec(0.1817f, 0.6198f, 0.7634f); // random vector to get things going

    for (int j = 0; j < radial_subdivisions + 1; j++) {
        cfact[j] = cosf(2.f * PI_F * float(j) / float(radial_subdivisions));
        sfact[j] = sinf(2.f * PI_F * float(j) / float(radial_subdivisions));
    }

    for (int i = 0; i < node_count; i++) { // looping over tube segments

        if (radius.at(i) < 0) {
            helios_runtime_error("ERROR (Context::addTube): Radius of tube must be positive.");
        }

        if (i == 0) {
            vec.x = nodes[i + 1].x - nodes[i].x;
            vec.y = nodes[i + 1].y - nodes[i].y;
            vec.z = nodes[i + 1].z - nodes[i].z;
        } else if (i == node_count - 1) {
            vec.x = nodes[i].x - nodes[i - 1].x;
            vec.y = nodes[i].y - nodes[i - 1].y;
            vec.z = nodes[i].z - nodes[i - 1].z;
        } else {
            vec.x = 0.5f * ((nodes[i].x - nodes[i - 1].x) + (nodes[i + 1].x - nodes[i].x));
            vec.y = 0.5f * ((nodes[i].y - nodes[i - 1].y) + (nodes[i + 1].y - nodes[i].y));
            vec.z = 0.5f * ((nodes[i].z - nodes[i - 1].z) + (nodes[i + 1].z - nodes[i].z));
        }

        // Ensure nvec is not parallel to vec to avoid degenerate cross products
        vec.normalize();
        if (fabs(nvec * vec) > 0.95f) {
            nvec = vec3(0.1817f, 0.6198f, 0.7634f); // Reset to original random vector
            if (fabs(nvec * vec) > 0.95f) {
                nvec = vec3(1.0f, 0.0f, 0.0f); // Use x-axis if still parallel
            }
        }
        // Also handle nearly vertical axes
        if (fabs(vec.z) > 0.95f) {
            nvec = vec3(1.0f, 0.0f, 0.0f); // Use horizontal direction for vertical axes
        }
        
        convec = cross(nvec, vec);
        convec.normalize();
        nvec = cross(vec, convec);
        nvec.normalize();

        for (int j = 0; j < radial_subdivisions + 1; j++) {
            vec3 normal;
            normal.x = cfact[j] * radius[i] * nvec.x + sfact[j] * radius[i] * convec.x;
            normal.y = cfact[j] * radius[i] * nvec.y + sfact[j] * radius[i] * convec.y;
            normal.z = cfact[j] * radius[i] * nvec.z + sfact[j] * radius[i] * convec.z;

            xyz[j][i].x = nodes[i].x + normal.x;
            xyz[j][i].y = nodes[i].y + normal.y;
            xyz[j][i].z = nodes[i].z + normal.z;
        }
    }

    vec3 v0, v1, v2;
    std::vector<uint> UUIDs(2 * (node_count - 1) * radial_subdivisions);

    int ii = 0;
    for (int i = 0; i < node_count - 1; i++) {
        for (int j = 0; j < radial_subdivisions; j++) {
            v0 = xyz[j][i];
            v1 = xyz[j + 1][i + 1];
            v2 = xyz[j + 1][i];

            UUIDs.at(ii) = addTriangle(v0, v1, v2, color.at(i));

            v0 = xyz[j][i];
            v1 = xyz[j][i + 1];
            v2 = xyz[j + 1][i + 1];

            UUIDs.at(ii + 1) = addTriangle(v0, v1, v2, color.at(i));

            ii += 2;
        }
    }

    return UUIDs;
}

std::vector<uint> Context::addTube(uint radial_subdivisions, const std::vector<vec3> &nodes, const std::vector<float> &radius, const char *texturefile) {
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addTube): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addTube): Texture file " + std::string(texturefile) + " does not exist.");
    }

    const uint node_count = nodes.size();

    if (node_count == 0) {
        helios_runtime_error("ERROR (Context::addTube): Node and radius arrays are empty.");
    } else if (node_count != radius.size()) {
        helios_runtime_error("ERROR (Context::addTube): Size of `nodes' and `radius' arguments must agree.");
    }

    vec3 vec, convec;
    std::vector<float> cfact(radial_subdivisions + 1);
    std::vector<float> sfact(radial_subdivisions + 1);
    std::vector<std::vector<vec3>> xyz, normal;
    std::vector<std::vector<vec2>> uv;
    resize_vector(xyz, node_count, radial_subdivisions + 1);
    resize_vector(normal, node_count, radial_subdivisions + 1);
    resize_vector(uv, node_count, radial_subdivisions + 1);

    vec3 nvec(0.1817f, 0.6198f, 0.7634f); // random vector to get things going

    for (int j = 0; j < radial_subdivisions + 1; j++) {
        cfact[j] = cosf(2.f * PI_F * float(j) / float(radial_subdivisions));
        sfact[j] = sinf(2.f * PI_F * float(j) / float(radial_subdivisions));
    }

    for (int i = 0; i < node_count; i++) { // looping over tube segments

        if (radius.at(i) < 0) {
            helios_runtime_error("ERROR (Context::addTube): Radius of tube must be positive.");
        }

        if (i == 0) {
            vec.x = nodes[i + 1].x - nodes[i].x;
            vec.y = nodes[i + 1].y - nodes[i].y;
            vec.z = nodes[i + 1].z - nodes[i].z;
        } else if (i == node_count - 1) {
            vec.x = nodes[i].x - nodes[i - 1].x;
            vec.y = nodes[i].y - nodes[i - 1].y;
            vec.z = nodes[i].z - nodes[i - 1].z;
        } else {
            vec.x = 0.5f * ((nodes[i].x - nodes[i - 1].x) + (nodes[i + 1].x - nodes[i].x));
            vec.y = 0.5f * ((nodes[i].y - nodes[i - 1].y) + (nodes[i + 1].y - nodes[i].y));
            vec.z = 0.5f * ((nodes[i].z - nodes[i - 1].z) + (nodes[i + 1].z - nodes[i].z));
        }

        // Ensure nvec is not parallel to vec to avoid degenerate cross products
        vec.normalize();
        if (fabs(nvec * vec) > 0.95f) {
            nvec = vec3(0.1817f, 0.6198f, 0.7634f); // Reset to original random vector
            if (fabs(nvec * vec) > 0.95f) {
                nvec = vec3(1.0f, 0.0f, 0.0f); // Use x-axis if still parallel
            }
        }
        // Also handle nearly vertical axes
        if (fabs(vec.z) > 0.95f) {
            nvec = vec3(1.0f, 0.0f, 0.0f); // Use horizontal direction for vertical axes
        }
        
        convec = cross(nvec, vec);
        convec.normalize();
        nvec = cross(vec, convec);
        nvec.normalize();

        for (int j = 0; j < radial_subdivisions + 1; j++) {
            normal[j][i].x = cfact[j] * radius[i] * nvec.x + sfact[j] * radius[i] * convec.x;
            normal[j][i].y = cfact[j] * radius[i] * nvec.y + sfact[j] * radius[i] * convec.y;
            normal[j][i].z = cfact[j] * radius[i] * nvec.z + sfact[j] * radius[i] * convec.z;

            xyz[j][i].x = nodes[i].x + normal[j][i].x;
            xyz[j][i].y = nodes[i].y + normal[j][i].y;
            xyz[j][i].z = nodes[i].z + normal[j][i].z;

            uv[j][i].x = float(i) / float(node_count - 1);
            uv[j][i].y = float(j) / float(radial_subdivisions);

            normal[j][i] = normal[j][i] / radius[i];
        }
    }

    vec3 v0, v1, v2;
    vec2 uv0, uv1, uv2;
    std::vector<uint> UUIDs(2 * (node_count - 1) * radial_subdivisions);

    int ii = 0;
    for (int i = 0; i < node_count - 1; i++) {
        for (int j = 0; j < radial_subdivisions; j++) {
            v0 = xyz[j][i];
            v1 = xyz[j + 1][i + 1];
            v2 = xyz[j + 1][i];

            uv0 = uv[j][i];
            uv1 = uv[j + 1][i + 1];
            uv2 = uv[j + 1][i];

            uint triangle_uuid = addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2);
            if (getPrimitiveArea(triangle_uuid) > 0) {
                UUIDs.at(ii) = triangle_uuid;
            } else {
                deletePrimitive(triangle_uuid);
                UUIDs.at(ii) = 0; // Mark as invalid
            }

            v0 = xyz[j][i];
            v1 = xyz[j][i + 1];
            v2 = xyz[j + 1][i + 1];

            uv0 = uv[j][i];
            uv1 = uv[j][i + 1];
            uv2 = uv[j + 1][i + 1];

            uint triangle_uuid2 = addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2);
            if (getPrimitiveArea(triangle_uuid2) > 0) {
                UUIDs.at(ii + 1) = triangle_uuid2;
            } else {
                deletePrimitive(triangle_uuid2);
                UUIDs.at(ii + 1) = 0; // Mark as invalid
            }

            ii += 2;
        }
    }

    // Remove invalid UUIDs (zeros) from the vector
    UUIDs.erase(std::remove(UUIDs.begin(), UUIDs.end(), 0), UUIDs.end());
    
    return UUIDs;
}

std::vector<uint> Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv) {
    RGBcolor color = make_RGBcolor(0.f, 0.75f, 0.f); // Default color is green

    return addBox(center, size, subdiv, color, false);
}

std::vector<uint> Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const RGBcolor &color) {
    return addBox(center, size, subdiv, color, false);
}

std::vector<uint> Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const char *texturefile) {
    return addBox(center, size, subdiv, texturefile, false);
}

std::vector<uint> Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const RGBcolor &color, bool reverse_normals) {
    std::vector<uint> UUID;

    vec3 subsize;
    subsize.x = size.x / float(subdiv.x);
    subsize.y = size.y / float(subdiv.y);
    subsize.z = size.z / float(subdiv.z);

    vec3 subcenter;
    std::vector<uint> U;

    if (reverse_normals) { // normals point inward

        // x-z faces (vertical)

        // right
        subcenter = center + make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, PI_F), make_int2(subdiv.x, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // left
        subcenter = center - make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, 0), make_int2(subdiv.x, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // y-z faces (vertical)

        // front
        subcenter = center + make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 1.5 * PI_F), make_int2(subdiv.y, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // back
        subcenter = center - make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 0.5 * PI_F), make_int2(subdiv.y, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // x-y faces (horizontal)

        // top
        subcenter = center + make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(PI_F, 0), make_int2(subdiv.x, subdiv.y), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // bottom
        subcenter = center - make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(0, 0), make_int2(subdiv.x, subdiv.y), color);
        UUID.insert(UUID.end(), U.begin(), U.end());
    } else { // normals point outward

        // x-z faces (vertical)

        // right
        subcenter = center + make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, 0), make_int2(subdiv.x, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // left
        subcenter = center - make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, PI_F), make_int2(subdiv.x, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // y-z faces (vertical)

        // front
        subcenter = center + make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 0.5 * PI_F), make_int2(subdiv.y, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // back
        subcenter = center - make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 1.5 * PI_F), make_int2(subdiv.y, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // x-y faces (horizontal)

        // top
        subcenter = center + make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(0, 0), make_int2(subdiv.x, subdiv.y), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // bottom
        subcenter = center - make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(PI_F, 0), make_int2(subdiv.x, subdiv.y), color);
        UUID.insert(UUID.end(), U.begin(), U.end());
    }

    return UUID;
}

std::vector<uint> Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const char *texturefile, bool reverse_normals) {
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addBox): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addBox): Texture file " + std::string(texturefile) + " does not exist.");
    }

    std::vector<uint> UUID;

    vec3 subsize;
    subsize.x = size.x / float(subdiv.x);
    subsize.y = size.y / float(subdiv.y);
    subsize.z = size.z / float(subdiv.z);

    vec3 subcenter;
    std::vector<uint> U;

    if (reverse_normals) { // normals point inward

        // x-z faces (vertical)

        // right
        subcenter = center + make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, PI_F), make_int2(subdiv.x, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // left
        subcenter = center - make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, 0), make_int2(subdiv.x, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // y-z faces (vertical)

        // front
        subcenter = center + make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 1.5 * PI_F), make_int2(subdiv.y, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // back
        subcenter = center - make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 0.5 * PI_F), make_int2(subdiv.y, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // x-y faces (horizontal)

        // top
        subcenter = center + make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(PI_F, 0), make_int2(subdiv.x, subdiv.y), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // bottom
        subcenter = center - make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(0, 0), make_int2(subdiv.x, subdiv.y), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());
    } else { // normals point outward

        // x-z faces (vertical)

        // right
        subcenter = center + make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, 0), make_int2(subdiv.x, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // left
        subcenter = center - make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, PI_F), make_int2(subdiv.x, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // y-z faces (vertical)

        // front
        subcenter = center + make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 0.5 * PI_F), make_int2(subdiv.y, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // back
        subcenter = center - make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 1.5 * PI_F), make_int2(subdiv.y, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // x-y faces (horizontal)

        // top
        subcenter = center + make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(0, 0), make_int2(subdiv.x, subdiv.y), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // bottom
        subcenter = center - make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(PI_F, 0), make_int2(subdiv.x, subdiv.y), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());
    }

    return UUID;
}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size) {
    return addDisk(make_int2(Ndivs, 1), center, size, make_SphericalCoord(0, 0), make_RGBAcolor(1, 0, 0, 1));
}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation) {
    return addDisk(make_int2(Ndivs, 1), center, size, rotation, make_RGBAcolor(1, 0, 0, 1));
}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBcolor &color) {
    return addDisk(make_int2(Ndivs, 1), center, size, rotation, make_RGBAcolor(color, 1));
}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBAcolor &color) {
    return addDisk(make_int2(Ndivs, 1), center, size, rotation, color);
}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const char *texture_file) {
    return addDisk(make_int2(Ndivs, 1), center, size, rotation, texture_file);
}

std::vector<uint> Context::addDisk(const int2 &Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBcolor &color) {
    return addDisk(Ndivs, center, size, rotation, make_RGBAcolor(color, 1));
}

std::vector<uint> Context::addDisk(const int2 &Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBAcolor &color) {
    std::vector<uint> UUID(Ndivs.x + Ndivs.x * (Ndivs.y - 1) * 2);
    int i = 0;
    for (int r = 0; r < Ndivs.y; r++) {
        for (int t = 0; t < Ndivs.x; t++) {
            float dtheta = 2.f * PI_F / float(Ndivs.x);
            float theta = dtheta * float(t);
            float theta_plus = dtheta * float(t + 1);

            float rx = size.x / float(Ndivs.y) * float(r);
            float ry = size.y / float(Ndivs.y) * float(r);

            float rx_plus = size.x / float(Ndivs.y) * float(r + 1);
            float ry_plus = size.y / float(Ndivs.y) * float(r + 1);

            if (r == 0) {
                UUID.at(i) = addTriangle(make_vec3(0, 0, 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), color);
            } else {
                UUID.at(i) = addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx * cosf(theta), ry * sinf(theta), 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), color);
                i++;
                UUID.at(i) = addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), color);
            }
            getPrimitivePointer_private(UUID.at(i))->rotate(rotation.elevation, "y");
            getPrimitivePointer_private(UUID.at(i))->rotate(rotation.azimuth, "z");
            getPrimitivePointer_private(UUID.at(i))->translate(center);

            i++;
        }
    }

    return UUID;
}

std::vector<uint> Context::addDisk(const int2 &Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const char *texturefile) {
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addDisk): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addDisk): Texture file " + std::string(texturefile) + " does not exist.");
    }

    std::vector<uint> UUID;
    UUID.reserve(Ndivs.x + Ndivs.x * (Ndivs.y - 1) * 2); // Reserve expected capacity
    for (int r = 0; r < Ndivs.y; r++) {
        for (int t = 0; t < Ndivs.x; t++) {
            float dtheta = 2.f * PI_F / float(Ndivs.x);
            float theta = dtheta * float(t);
            float theta_plus = dtheta * float(t + 1);

            float rx = size.x / float(Ndivs.y) * float(r);
            float ry = size.y / float(Ndivs.y) * float(r);
            float rx_plus = size.x / float(Ndivs.y) * float(r + 1);
            float ry_plus = size.y / float(Ndivs.y) * float(r + 1);

            if (r == 0) {
                uint triangle_uuid = addTriangle(make_vec3(0, 0, 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), texturefile, make_vec2(0.5, 0.5),
                                         make_vec2(0.5f * (1.f + cosf(theta) * rx_plus / size.x), 0.5f * (1.f + sinf(theta) * ry_plus / size.y)),
                                         make_vec2(0.5f * (1.f + cosf(theta_plus) * rx_plus / size.x), 0.5f * (1.f + sinf(theta_plus) * ry_plus / size.y)));
                if (getPrimitiveArea(triangle_uuid) > 0) {
                    UUID.push_back(triangle_uuid);
                } else {
                    deletePrimitive(triangle_uuid);
                    continue;
                }
            } else {
                uint triangle_uuid1 = addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx * cosf(theta), ry * sinf(theta), 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), texturefile,
                                         make_vec2(0.5f * (1.f + cosf(theta_plus) * rx / size.x), 0.5f * (1.f + sinf(theta_plus) * ry / size.y)), make_vec2(0.5f * (1.f + cosf(theta) * rx / size.x), 0.5f * (1.f + sinf(theta) * ry / size.y)),
                                         make_vec2(0.5f * (1.f + cosf(theta) * rx_plus / size.x), 0.5f * (1.f + sinf(theta) * ry_plus / size.y)));
                if (getPrimitiveArea(triangle_uuid1) > 0) {
                    UUID.push_back(triangle_uuid1);
                } else {
                    deletePrimitive(triangle_uuid1);
                }
                
                uint triangle_uuid2 = addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), texturefile,
                                         make_vec2(0.5f * (1.f + cosf(theta_plus) * rx / size.x), 0.5f * (1.f + sinf(theta_plus) * ry / size.y)), make_vec2(0.5f * (1.f + cosf(theta) * rx_plus / size.x), 0.5f * (1.f + sinf(theta) * ry_plus / size.y)),
                                         make_vec2(0.5f * (1.f + cosf(theta_plus) * rx_plus / size.x), 0.5f * (1.f + sinf(theta_plus) * ry_plus / size.y)));
                if (getPrimitiveArea(triangle_uuid2) > 0) {
                    UUID.push_back(triangle_uuid2);
                } else {
                    deletePrimitive(triangle_uuid2);
                    continue;
                }
            }
            // Apply transformations to all valid triangles added in this iteration
            size_t start_idx = UUID.size() - (r == 0 ? 1 : 2);
            for (size_t uuid_idx = start_idx; uuid_idx < UUID.size(); uuid_idx++) {
                getPrimitivePointer_private(UUID.at(uuid_idx))->rotate(rotation.elevation, "y");
                getPrimitivePointer_private(UUID.at(uuid_idx))->rotate(rotation.azimuth, "z");
                getPrimitivePointer_private(UUID.at(uuid_idx))->translate(center);
            }
        }
    }

    return UUID;
}

std::vector<uint> Context::addCone(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1) {
    RGBcolor color;
    color = make_RGBcolor(0.f, 0.75f, 0.f); // Default color is green

    return addCone(Ndivs, node0, node1, radius0, radius1, color);
}

std::vector<uint> Context::addCone(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1, RGBcolor &color) {
    std::vector<helios::vec3> nodes{node0, node1};
    std::vector<float> radii{radius0, radius1};

    vec3 vec, convec;
    std::vector<float> cfact(Ndivs + 1);
    std::vector<float> sfact(Ndivs + 1);
    std::vector<std::vector<vec3>> xyz, normal;
    xyz.resize(Ndivs + 1);
    normal.resize(Ndivs + 1);
    for (uint j = 0; j < Ndivs + 1; j++) {
        xyz.at(j).resize(2);
        normal.at(j).resize(2);
    }
    vec3 nvec(0.1817f, 0.6198f, 0.7634f); // random vector to get things going

    for (int j = 0; j < Ndivs + 1; j++) {
        cfact[j] = cosf(2.f * PI_F * float(j) / float(Ndivs));
        sfact[j] = sinf(2.f * PI_F * float(j) / float(Ndivs));
    }

    for (int i = 0; i < 2; i++) { // looping over cone segments

        if (i == 0) {
            vec.x = nodes[i + 1].x - nodes[i].x;
            vec.y = nodes[i + 1].y - nodes[i].y;
            vec.z = nodes[i + 1].z - nodes[i].z;
        } else if (i == 1) {
            vec.x = nodes[i].x - nodes[i - 1].x;
            vec.y = nodes[i].y - nodes[i - 1].y;
            vec.z = nodes[i].z - nodes[i - 1].z;
        }

        float norm;
        convec = cross(nvec, vec);
        norm = convec.magnitude();
        convec.x = convec.x / norm;
        convec.y = convec.y / norm;
        convec.z = convec.z / norm;
        nvec = cross(vec, convec);
        norm = nvec.magnitude();
        nvec.x = nvec.x / norm;
        nvec.y = nvec.y / norm;
        nvec.z = nvec.z / norm;


        for (int j = 0; j < Ndivs + 1; j++) {
            normal[j][i].x = cfact[j] * radii[i] * nvec.x + sfact[j] * radii[i] * convec.x;
            normal[j][i].y = cfact[j] * radii[i] * nvec.y + sfact[j] * radii[i] * convec.y;
            normal[j][i].z = cfact[j] * radii[i] * nvec.z + sfact[j] * radii[i] * convec.z;

            xyz[j][i].x = nodes[i].x + normal[j][i].x;
            xyz[j][i].y = nodes[i].y + normal[j][i].y;
            xyz[j][i].z = nodes[i].z + normal[j][i].z;

            normal[j][i] = normal[j][i] / radii[i];
        }
    }

    vec3 v0, v1, v2;
    std::vector<uint> UUID;

    for (int i = 0; i < 2 - 1; i++) {
        for (int j = 0; j < Ndivs; j++) {
            v0 = xyz[j][i];
            v1 = xyz[j + 1][i + 1];
            v2 = xyz[j + 1][i];

            UUID.push_back(addTriangle(v0, v1, v2, color));

            v0 = xyz[j][i];
            v1 = xyz[j][i + 1];
            v2 = xyz[j + 1][i + 1];

            UUID.push_back(addTriangle(v0, v1, v2, color));
        }
    }

    return UUID;
}

std::vector<uint> Context::addCone(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1, const char *texturefile) {
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addCone): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addCone): Texture file " + std::string(texturefile) + " does not exist.");
    }

    std::vector<helios::vec3> nodes{node0, node1};
    std::vector<float> radii{radius0, radius1};

    vec3 vec, convec;
    std::vector<float> cfact(Ndivs + 1);
    std::vector<float> sfact(Ndivs + 1);
    std::vector<std::vector<vec3>> xyz, normal;
    std::vector<std::vector<vec2>> uv;
    xyz.resize(Ndivs + 1);
    normal.resize(Ndivs + 1);
    uv.resize(Ndivs + 1);
    for (uint j = 0; j < Ndivs + 1; j++) {
        xyz.at(j).resize(2);
        normal.at(j).resize(2);
        uv.at(j).resize(2);
    }
    vec3 nvec(0.f, 1.f, 0.f);

    for (int j = 0; j < Ndivs + 1; j++) {
        cfact[j] = cosf(2.f * PI_F * float(j) / float(Ndivs));
        sfact[j] = sinf(2.f * PI_F * float(j) / float(Ndivs));
    }

    for (int i = 0; i < 2; i++) { // looping over cone segments

        if (i == 0) {
            vec.x = nodes[i + 1].x - nodes[i].x;
            vec.y = nodes[i + 1].y - nodes[i].y;
            vec.z = nodes[i + 1].z - nodes[i].z;
        } else if (i == 1) {
            vec.x = nodes[i].x - nodes[i - 1].x;
            vec.y = nodes[i].y - nodes[i - 1].y;
            vec.z = nodes[i].z - nodes[i - 1].z;
        }

        float norm;
        convec = cross(nvec, vec);
        norm = convec.magnitude();
        convec.x = convec.x / norm;
        convec.y = convec.y / norm;
        convec.z = convec.z / norm;
        nvec = cross(vec, convec);
        norm = nvec.magnitude();
        nvec.x = nvec.x / norm;
        nvec.y = nvec.y / norm;
        nvec.z = nvec.z / norm;

        for (int j = 0; j < Ndivs + 1; j++) {
            normal[j][i].x = cfact[j] * radii[i] * nvec.x + sfact[j] * radii[i] * convec.x;
            normal[j][i].y = cfact[j] * radii[i] * nvec.y + sfact[j] * radii[i] * convec.y;
            normal[j][i].z = cfact[j] * radii[i] * nvec.z + sfact[j] * radii[i] * convec.z;

            xyz[j][i].x = nodes[i].x + normal[j][i].x;
            xyz[j][i].y = nodes[i].y + normal[j][i].y;
            xyz[j][i].z = nodes[i].z + normal[j][i].z;

            uv[j][i].x = float(i) / float(2 - 1);
            uv[j][i].y = float(j) / float(Ndivs);

            normal[j][i] = normal[j][i] / radii[i];
        }
    }

    vec3 v0, v1, v2;
    vec2 uv0, uv1, uv2;
    std::vector<uint> UUID;

    for (int i = 0; i < 2 - 1; i++) {
        for (int j = 0; j < Ndivs; j++) {
            v0 = xyz[j][i];
            v1 = xyz[j + 1][i + 1];
            v2 = xyz[j + 1][i];

            uv0 = uv[j][i];
            uv1 = uv[j + 1][i + 1];
            uv2 = uv[j + 1][i];

            if ((v1 - v0).magnitude() > 1e-6 && (v2 - v0).magnitude() > 1e-6 && (v2 - v1).magnitude() > 1e-6) {
                uint triangle_uuid = addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2);
                if (getPrimitiveArea(triangle_uuid) > 0) {
                    UUID.push_back(triangle_uuid);
                } else {
                    deletePrimitive(triangle_uuid);
                }
            }

            v0 = xyz[j][i];
            v1 = xyz[j][i + 1];
            v2 = xyz[j + 1][i + 1];

            uv0 = uv[j][i];
            uv1 = uv[j][i + 1];
            uv2 = uv[j + 1][i + 1];

            if ((v1 - v0).magnitude() > 1e-6 && (v2 - v0).magnitude() > 1e-6 && (v2 - v1).magnitude() > 1e-6) {
                uint triangle_uuid = addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2);
                if (getPrimitiveArea(triangle_uuid) > 0) {
                    UUID.push_back(triangle_uuid);
                } else {
                    deletePrimitive(triangle_uuid);
                }
            }
        }
    }

    return UUID;
}

void Context::colorPrimitiveByDataPseudocolor(const std::vector<uint> &UUIDs, const std::string &primitive_data, const std::string &colormap, uint Ncolors) {
    colorPrimitiveByDataPseudocolor(UUIDs, primitive_data, colormap, Ncolors, 9999999, -9999999);
}

void Context::colorPrimitiveByDataPseudocolor(const std::vector<uint> &UUIDs, const std::string &primitive_data, const std::string &colormap, uint Ncolors, float data_min, float data_max) {
    std::map<uint, float> pcolor_data;

    float data_min_new = 9999999;
    float data_max_new = -9999999;
    for (uint UUID: UUIDs) {
        if (!doesPrimitiveExist(UUID)) {
            std::cerr << "WARNING (Context::colorPrimitiveDataPseudocolor): primitive for UUID " << std::to_string(UUID) << " does not exist. Skipping this primitive." << std::endl;
            continue;
        }

        float dataf = 0;
        if (doesPrimitiveDataExist(UUID, primitive_data.c_str())) {
            if (getPrimitiveDataType(primitive_data.c_str()) != HELIOS_TYPE_FLOAT && getPrimitiveDataType(primitive_data.c_str()) != HELIOS_TYPE_INT && getPrimitiveDataType(primitive_data.c_str()) != HELIOS_TYPE_UINT &&
                getPrimitiveDataType(primitive_data.c_str()) != HELIOS_TYPE_DOUBLE) {
                std::cerr << "WARNING (Context::colorPrimitiveDataPseudocolor): Only primitive data types of int, uint, float, and double are supported for this function. Skipping this primitive." << std::endl;
                continue;
            }

            if (getPrimitiveDataType(primitive_data.c_str()) == HELIOS_TYPE_FLOAT) {
                float data;
                getPrimitiveData(UUID, primitive_data.c_str(), data);
                dataf = data;
            } else if (getPrimitiveDataType(primitive_data.c_str()) == HELIOS_TYPE_DOUBLE) {
                double data;
                getPrimitiveData(UUID, primitive_data.c_str(), data);
                dataf = float(data);
            } else if (getPrimitiveDataType(primitive_data.c_str()) == HELIOS_TYPE_INT) {
                int data;
                getPrimitiveData(UUID, primitive_data.c_str(), data);
                dataf = float(data);
            } else if (getPrimitiveDataType(primitive_data.c_str()) == HELIOS_TYPE_UINT) {
                uint data;
                getPrimitiveData(UUID, primitive_data.c_str(), data);
                dataf = float(data);
            }
        }

        if (data_min == 9999999 && data_max == -9999999) {
            if (dataf < data_min_new) {
                data_min_new = dataf;
            }
            if (dataf > data_max_new) {
                data_max_new = dataf;
            }
        }

        pcolor_data[UUID] = dataf;
    }

    if (data_min == 9999999 && data_max == -9999999) {
        data_min = data_min_new;
        data_max = data_max_new;
    }

    std::vector<RGBcolor> colormap_data = generateColormap(colormap, Ncolors);

    std::map<std::string, std::vector<std::string>> cmap_texture_filenames;

    for (auto &[UUID, pdata]: pcolor_data) {
        std::string texturefile = getPrimitiveTextureFile(UUID);

        int cmap_ind = std::round((pdata - data_min) / (data_max - data_min) * float(Ncolors - 1));

        if (cmap_ind < 0) {
            cmap_ind = 0;
        } else if (cmap_ind >= Ncolors) {
            cmap_ind = Ncolors - 1;
        }

        if (!texturefile.empty() && primitiveTextureHasTransparencyChannel(UUID)) { // primitive has texture with transparency channel

            overridePrimitiveTextureColor(UUID);
            setPrimitiveColor(UUID, colormap_data.at(cmap_ind));
        } else { // primitive does not have texture with transparency channel - assign constant color

            if (!getPrimitiveTextureFile(UUID).empty()) {
                overridePrimitiveTextureColor(UUID);
            }

            setPrimitiveColor(UUID, colormap_data.at(cmap_ind));
        }
    }
}

std::vector<RGBcolor> Context::generateColormap(const std::vector<helios::RGBcolor> &ctable, const std::vector<float> &cfrac, uint Ncolors) {
    if (Ncolors > 9999) {
        std::cerr << "WARNING (Context::generateColormap): Truncating number of color map textures to maximum value of 9999." << std::endl;
    }

    if (ctable.size() != cfrac.size()) {
        helios_runtime_error("ERROR (Context::generateColormap): The length of arguments 'ctable' and 'cfrac' must match.");
    }
    if (ctable.empty()) {
        helios_runtime_error("ERROR (Context::generateColormap): 'ctable' and 'cfrac' arguments contain empty vectors.");
    }

    std::vector<RGBcolor> color_table(Ncolors);

    for (int i = 0; i < Ncolors; i++) {
        float frac = float(i) / float(Ncolors - 1) * cfrac.back();

        int j;
        for (j = 0; j < cfrac.size() - 1; j++) {
            if (frac >= cfrac.at(j) && frac <= cfrac.at(j + 1)) {
                break;
            }
        }

        float cminus = std::fmaxf(0.f, cfrac.at(j));
        float cplus = std::fminf(1.f, cfrac.at(j + 1));

        float jfrac = (frac - cminus) / (cplus - cminus);

        RGBcolor color;
        color.r = ctable.at(j).r + jfrac * (ctable.at(j + 1).r - ctable.at(j).r);
        color.g = ctable.at(j).g + jfrac * (ctable.at(j + 1).g - ctable.at(j).g);
        color.b = ctable.at(j).b + jfrac * (ctable.at(j + 1).b - ctable.at(j).b);

        color_table.at(i) = color;
    }

    return color_table;
}

std::vector<RGBcolor> Context::generateColormap(const std::string &colormap, uint Ncolors) {
    std::vector<RGBcolor> ctable_c;
    std::vector<float> clocs_c;

    if (colormap == "hot") {
        ctable_c.resize(5);
        ctable_c.at(0) = make_RGBcolor(0.f, 0.f, 0.f);
        ctable_c.at(1) = make_RGBcolor(0.5f, 0.f, 0.5f);
        ctable_c.at(2) = make_RGBcolor(1.f, 0.f, 0.f);
        ctable_c.at(3) = make_RGBcolor(1.f, 0.5f, 0.f);
        ctable_c.at(4) = make_RGBcolor(1.f, 1.f, 0.f);

        clocs_c.resize(5);
        clocs_c.at(0) = 0.f;
        clocs_c.at(1) = 0.25f;
        clocs_c.at(2) = 0.5f;
        clocs_c.at(3) = 0.75f;
        clocs_c.at(4) = 1.f;
    } else if (colormap == "cool") {
        ctable_c.resize(2);
        ctable_c.at(0) = RGB::cyan;
        ctable_c.at(1) = RGB::magenta;

        clocs_c.resize(2);
        clocs_c.at(0) = 0.f;
        clocs_c.at(1) = 1.f;
    } else if (colormap == "lava") {
        ctable_c.resize(5);
        ctable_c.at(0) = make_RGBcolor(0.f, 0.05f, 0.05f);
        ctable_c.at(1) = make_RGBcolor(0.f, 0.6f, 0.6f);
        ctable_c.at(2) = make_RGBcolor(1.f, 1.f, 1.f);
        ctable_c.at(3) = make_RGBcolor(1.f, 0.f, 0.f);
        ctable_c.at(4) = make_RGBcolor(0.5f, 0.f, 0.f);

        clocs_c.resize(5);
        clocs_c.at(0) = 0.f;
        clocs_c.at(1) = 0.4f;
        clocs_c.at(2) = 0.5f;
        clocs_c.at(3) = 0.6f;
        clocs_c.at(4) = 1.f;
    } else if (colormap == "rainbow") {
        ctable_c.resize(4);
        ctable_c.at(0) = RGB::navy;
        ctable_c.at(1) = RGB::cyan;
        ctable_c.at(2) = RGB::yellow;
        ctable_c.at(3) = make_RGBcolor(0.75f, 0.f, 0.f);

        clocs_c.resize(4);
        clocs_c.at(0) = 0.f;
        clocs_c.at(1) = 0.3f;
        clocs_c.at(2) = 0.7f;
        clocs_c.at(3) = 1.f;
    } else if (colormap == "parula") {
        ctable_c.resize(4);
        ctable_c.at(0) = RGB::navy;
        ctable_c.at(1) = make_RGBcolor(0, 0.6, 0.6);
        ctable_c.at(2) = RGB::goldenrod;
        ctable_c.at(3) = RGB::yellow;

        clocs_c.resize(4);
        clocs_c.at(0) = 0.f;
        clocs_c.at(1) = 0.4f;
        clocs_c.at(2) = 0.7f;
        clocs_c.at(3) = 1.f;
    } else if (colormap == "gray") {
        ctable_c.resize(2);
        ctable_c.at(0) = RGB::black;
        ctable_c.at(1) = RGB::white;

        clocs_c.resize(2);
        clocs_c.at(0) = 0.f;
        clocs_c.at(1) = 1.f;
    } else if (colormap == "green") {
        ctable_c.resize(2);
        ctable_c.at(0) = RGB::black;
        ctable_c.at(1) = RGB::green;

        clocs_c.resize(2);
        clocs_c.at(0) = 0.f;
        clocs_c.at(1) = 1.f;
    } else {
        helios_runtime_error("ERROR (Context::generateColormapTextures): Unknown colormap " + colormap + ".");
    }

    return generateColormap(ctable_c, clocs_c, Ncolors);
}

std::vector<std::string> Context::generateTexturesFromColormap(const std::string &texturefile, const std::vector<RGBcolor> &colormap_data) {
    uint Ncolors = colormap_data.size();

    // check that texture file exists
    std::ifstream tfile(texturefile);
    if (!tfile) {
        helios_runtime_error("ERROR (Context::generateTexturesFromColormap): Texture file " + texturefile + " does not exist, or you do not have permission to read it.");
    }
    tfile.close();

    // get file extension
    std::string file_ext = getFileExtension(texturefile);

    // get file base/stem
    std::string file_base = getFileStem(texturefile);

    std::vector<RGBcolor> color_table(Ncolors);

    std::vector<std::string> texture_filenames(Ncolors);

    if (file_ext == "png" || file_ext == "PNG") {
        std::vector<RGBAcolor> pixel_data;
        uint width, height;
        readPNG(texturefile, width, height, pixel_data);

        for (int i = 0; i < Ncolors; i++) {
            std::ostringstream filename;
            filename << "lib/images/colormap_" << file_base << "_" << std::setw(4) << std::setfill('0') << std::to_string(i) << ".png";

            texture_filenames.at(i) = filename.str();

            RGBcolor color = colormap_data.at(i);

            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
                    pixel_data.at(row * width + col) = make_RGBAcolor(color, pixel_data.at(row * width + col).a);
                }
            }

            writePNG(filename.str(), width, height, pixel_data);
        }
    }

    return texture_filenames;
}

void Context::out_of_memory_handler() {
    helios_runtime_error("ERROR: Out of host memory. The program has run out of memory and cannot continue.");
}

void Context::install_out_of_memory_handler() {
    std::set_new_handler(out_of_memory_handler);
}

Context::~Context() {
    for (auto &[UUID, primitive]: primitives) {
        delete getPrimitivePointer_private(UUID);
    }

    for (auto &[UUID, object]: objects) {
        delete getObjectPointer(UUID);
    }
}

PrimitiveType Context::getPrimitiveType(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::getPrimitiveType): Primitive with UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
#endif
    return getPrimitivePointer_private(UUID)->getType();
}

void Context::setPrimitiveParentObjectID(uint UUID, uint objID) {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::setPrimitiveParentObjectID): Primitive with UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
#endif

    uint current_objID = getPrimitivePointer_private(UUID)->getParentObjectID();
    getPrimitivePointer_private(UUID)->setParentObjectID(objID);

    if (current_objID != 0u && current_objID != objID) {
        if (doesObjectExist(current_objID)) {
            objects.at(current_objID)->deleteChildPrimitive(UUID);

            if (getObjectPointer_private(current_objID)->getPrimitiveUUIDs().empty()) {
                CompoundObject *obj = objects.at(current_objID);
                delete obj;
                objects.erase(current_objID);
            }
        }
    }
}

void Context::setPrimitiveParentObjectID(const std::vector<uint> &UUIDs, uint objID) {
    for (uint UUID: UUIDs) {
        setPrimitiveParentObjectID(UUID, objID);
    }
}

uint Context::getPrimitiveParentObjectID(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::getPrimitiveParentObjectID): Primitive with UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
#endif
    return getPrimitivePointer_private(UUID)->getParentObjectID();
}

std::vector<uint> Context::getPrimitiveParentObjectID(const std::vector<uint> &UUIDs) const {
    std::vector<uint> objIDs(UUIDs.size());
    for (uint i = 0; i < UUIDs.size(); i++) {
#ifdef HELIOS_DEBUG
        if (!doesPrimitiveExist(UUIDs[i])) {
            helios_runtime_error("ERROR (Context::getPrimitiveParentObjectID): Primitive with UUID of " + std::to_string(UUIDs[i]) + " does not exist in the Context.");
        }
#endif
        objIDs[i] = getPrimitivePointer_private(UUIDs[i])->getParentObjectID();
    }
    return objIDs;
}


std::vector<uint> Context::getUniquePrimitiveParentObjectIDs(const std::vector<uint> &UUIDs) const {
    return getUniquePrimitiveParentObjectIDs(UUIDs, false);
}


std::vector<uint> Context::getUniquePrimitiveParentObjectIDs(const std::vector<uint> &UUIDs, bool include_ObjID_zero) const {
    std::vector<uint> primitiveObjIDs;
    if (UUIDs.empty()) {
        return primitiveObjIDs;
    }

    // vector of parent object ID for each primitive
    primitiveObjIDs.resize(UUIDs.size());
    for (uint i = 0; i < UUIDs.size(); i++) {
#ifdef HELIOS_DEBUG
        if (!doesPrimitiveExist(UUIDs.at(i))) {
            helios_runtime_error("ERROR (Context::getUniquePrimitiveParentObjectIDs): Primitive with UUID of " + std::to_string(UUIDs.at(i)) + " does not exist in the Context.");
        }
#endif
        primitiveObjIDs.at(i) = getPrimitivePointer_private(UUIDs.at(i))->getParentObjectID();
    }

    // sort
    std::sort(primitiveObjIDs.begin(), primitiveObjIDs.end());

    // unique
    auto it = unique(primitiveObjIDs.begin(), primitiveObjIDs.end());
    primitiveObjIDs.resize(distance(primitiveObjIDs.begin(), it));

    // remove object ID = 0 from the output if desired and it exists
    if (include_ObjID_zero == false & primitiveObjIDs.front() == uint(0)) {
        primitiveObjIDs.erase(primitiveObjIDs.begin());
    }

    return primitiveObjIDs;
}

float Context::getPrimitiveArea(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::getPrimitiveArea): Primitive with UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
#endif
    return getPrimitivePointer_private(UUID)->getArea();
}

void Context::getPrimitiveBoundingBox(uint UUID, vec3 &min_corner, vec3 &max_corner) const {
    const std::vector UUIDs = {UUID};
    getPrimitiveBoundingBox(UUIDs, min_corner, max_corner);
}

void Context::getPrimitiveBoundingBox(const std::vector<uint> &UUIDs, vec3 &min_corner, vec3 &max_corner) const {
    uint p = 0;
    for (uint UUID: UUIDs) {
        if (!doesPrimitiveExist(UUID)) {
            helios_runtime_error("ERROR (Context::getPrimitiveBoundingBox): Primitive with UUID of " + std::to_string(UUID) + " does not exist in the Context.");
        }

        const std::vector<vec3> &vertices = getPrimitiveVertices(UUID);

        if (p == 0) {
            min_corner = vertices.front();
            max_corner = min_corner;
        }

        for (const vec3 &vert: vertices) {
            if (vert.x < min_corner.x) {
                min_corner.x = vert.x;
            }
            if (vert.y < min_corner.y) {
                min_corner.y = vert.y;
            }
            if (vert.z < min_corner.z) {
                min_corner.z = vert.z;
            }
            if (vert.x > max_corner.x) {
                max_corner.x = vert.x;
            }
            if (vert.y > max_corner.y) {
                max_corner.y = vert.y;
            }
            if (vert.z > max_corner.z) {
                max_corner.z = vert.z;
            }
        }

        p++;
    }
}

helios::vec3 Context::getPrimitiveNormal(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getNormal();
}

void Context::getPrimitiveTransformationMatrix(uint UUID, float (&T)[16]) const {
    getPrimitivePointer_private(UUID)->getTransformationMatrix(T);
}

void Context::setPrimitiveTransformationMatrix(uint UUID, float (&T)[16]) {
    getPrimitivePointer_private(UUID)->setTransformationMatrix(T);
}

void Context::setPrimitiveTransformationMatrix(const std::vector<uint> &UUIDs, float (&T)[16]) {
    for (uint UUID: UUIDs) {
        getPrimitivePointer_private(UUID)->setTransformationMatrix(T);
    }
}

std::vector<helios::vec3> Context::getPrimitiveVertices(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getVertices();
}


helios::RGBcolor Context::getPrimitiveColor(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getColor();
}

helios::RGBcolor Context::getPrimitiveColorRGB(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getColorRGB();
}

helios::RGBAcolor Context::getPrimitiveColorRGBA(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getColorRGBA();
}

void Context::setPrimitiveColor(uint UUID, const RGBcolor &color) const {
    getPrimitivePointer_private(UUID)->setColor(color);
}

void Context::setPrimitiveColor(const std::vector<uint> &UUIDs, const RGBcolor &color) const {
    for (uint UUID: UUIDs) {
        getPrimitivePointer_private(UUID)->setColor(color);
    }
}

void Context::setPrimitiveColor(uint UUID, const RGBAcolor &color) const {
    getPrimitivePointer_private(UUID)->setColor(color);
}

void Context::setPrimitiveColor(const std::vector<uint> &UUIDs, const RGBAcolor &color) const {
    for (uint UUID: UUIDs) {
        getPrimitivePointer_private(UUID)->setColor(color);
    }
}

std::string Context::getPrimitiveTextureFile(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getTextureFile();
}

void Context::setPrimitiveTextureFile(uint UUID, const std::string &texturefile) const {
    getPrimitivePointer_private(UUID)->setTextureFile(texturefile.c_str());
}

helios::int2 Context::getPrimitiveTextureSize(uint UUID) const {
    std::string texturefile = getPrimitivePointer_private(UUID)->getTextureFile();
    if (!texturefile.empty() && textures.find(texturefile) != textures.end()) {
        return textures.at(texturefile).getImageResolution();
    }
    return {0, 0};
}

std::vector<helios::vec2> Context::getPrimitiveTextureUV(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getTextureUV();
}

bool Context::primitiveTextureHasTransparencyChannel(uint UUID) const {
    std::string texturefile = getPrimitivePointer_private(UUID)->getTextureFile();
    if (!texturefile.empty() && textures.find(texturefile) != textures.end()) {
        return textures.at(texturefile).hasTransparencyChannel();
    }
    return false;
}

const std::vector<std::vector<bool>> *Context::getPrimitiveTextureTransparencyData(uint UUID) const {
    if (primitiveTextureHasTransparencyChannel(UUID)) {
        const std::vector<std::vector<bool>> *data = textures.at(getPrimitivePointer_private(UUID)->getTextureFile()).getTransparencyData();
        return data;
    }

    helios_runtime_error("ERROR (Context::getPrimitiveTransparencyData): Texture transparency data does not exist for primitive " + std::to_string(UUID) + ".");
    return nullptr;
}

void Context::overridePrimitiveTextureColor(uint UUID) const {
    getPrimitivePointer_private(UUID)->overrideTextureColor();
}

void Context::overridePrimitiveTextureColor(const std::vector<uint> &UUIDs) const {
    for (uint UUID: UUIDs) {
        getPrimitivePointer_private(UUID)->overrideTextureColor();
    }
}

void Context::usePrimitiveTextureColor(uint UUID) const {
    getPrimitivePointer_private(UUID)->useTextureColor();
}

void Context::usePrimitiveTextureColor(const std::vector<uint> &UUIDs) const {
    for (uint UUID: UUIDs) {
        getPrimitivePointer_private(UUID)->useTextureColor();
    }
}

bool Context::isPrimitiveTextureColorOverridden(uint UUID) const {
    return getPrimitivePointer_private(UUID)->isTextureColorOverridden();
}

float Context::getPrimitiveSolidFraction(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getSolidFraction();
}

void Context::printPrimitiveInfo(uint UUID) const {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Info for UUID " << UUID << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    PrimitiveType type = getPrimitiveType(UUID);
    std::string stype;
    if (type == 0) {
        stype = "PRIMITIVE_TYPE_PATCH";
    } else if (type == 1) {
        stype = "PRIMITIVE_TYPE_TRIANGLE";
    } else if (type == 2) {
        stype = "PRIMITIVE_TYPE_VOXEL";
    }

    std::cout << "Type: " << stype << std::endl;
    std::cout << "Parent ObjID: " << getPrimitiveParentObjectID(UUID) << std::endl;
    std::cout << "Surface Area: " << getPrimitiveArea(UUID) << std::endl;
    std::cout << "Normal Vector: " << getPrimitiveNormal(UUID) << std::endl;

    if (type == PRIMITIVE_TYPE_PATCH) {
        std::cout << "Patch Center: " << getPatchCenter(UUID) << std::endl;
        std::cout << "Patch Size: " << getPatchSize(UUID) << std::endl;
    } else if (type == PRIMITIVE_TYPE_VOXEL) {
        std::cout << "Voxel Center: " << getVoxelCenter(UUID) << std::endl;
        std::cout << "Voxel Size: " << getVoxelSize(UUID) << std::endl;
    }

    std::vector<vec3> primitive_vertices = getPrimitiveVertices(UUID);
    std::cout << "Vertices: " << std::endl;
    for (uint i = 0; i < primitive_vertices.size(); i++) {
        std::cout << "   " << primitive_vertices.at(i) << std::endl;
    }

    float T[16];
    getPrimitiveTransformationMatrix(UUID, T);
    std::cout << "Transform: " << std::endl;
    std::cout << "   " << T[0] << "      " << T[1] << "      " << T[2] << "      " << T[3] << std::endl;
    std::cout << "   " << T[4] << "      " << T[5] << "      " << T[6] << "      " << T[7] << std::endl;
    std::cout << "   " << T[8] << "      " << T[9] << "      " << T[10] << "      " << T[11] << std::endl;
    std::cout << "   " << T[12] << "      " << T[13] << "      " << T[14] << "      " << T[15] << std::endl;

    std::cout << "Color: " << getPrimitiveColor(UUID) << std::endl;
    std::cout << "Texture File: " << getPrimitiveTextureFile(UUID) << std::endl;
    std::cout << "Texture Size: " << getPrimitiveTextureSize(UUID) << std::endl;
    std::cout << "Texture UV: " << std::endl;
    std::vector<vec2> uv = getPrimitiveTextureUV(UUID);
    for (uint i = 0; i < uv.size(); i++) {
        std::cout << "   " << uv.at(i) << std::endl;
    }

    std::cout << "Texture Transparency: " << primitiveTextureHasTransparencyChannel(UUID) << std::endl;
    std::cout << "Color Overridden: " << isPrimitiveTextureColorOverridden(UUID) << std::endl;
    std::cout << "Solid Fraction: " << getPrimitiveSolidFraction(UUID) << std::endl;


    std::cout << "Primitive Data: " << std::endl;
    // Primitive* pointer = getPrimitivePointer_private(UUID);
    std::vector<std::string> pd = listPrimitiveData(UUID);
    for (uint i = 0; i < pd.size(); i++) {
        uint dsize = getPrimitiveDataSize(UUID, pd.at(i).c_str());
        HeliosDataType dtype = getPrimitiveDataType(pd.at(i).c_str());
        std::string dstype;

        if (dtype == HELIOS_TYPE_INT) {
            dstype = "HELIOS_TYPE_INT";
        } else if (dtype == HELIOS_TYPE_UINT) {
            dstype = "HELIOS_TYPE_UINT";
        } else if (dtype == HELIOS_TYPE_FLOAT) {
            dstype = "HELIOS_TYPE_FLOAT";
        } else if (dtype == HELIOS_TYPE_DOUBLE) {
            dstype = "HELIOS_TYPE_DOUBLE";
        } else if (dtype == HELIOS_TYPE_VEC2) {
            dstype = "HELIOS_TYPE_VEC2";
        } else if (dtype == HELIOS_TYPE_VEC3) {
            dstype = "HELIOS_TYPE_VEC3";
        } else if (dtype == HELIOS_TYPE_VEC4) {
            dstype = "HELIOS_TYPE_VEC4";
        } else if (dtype == HELIOS_TYPE_INT2) {
            dstype = "HELIOS_TYPE_INT2";
        } else if (dtype == HELIOS_TYPE_INT3) {
            dstype = "HELIOS_TYPE_INT3";
        } else if (dtype == HELIOS_TYPE_INT4) {
            dstype = "HELIOS_TYPE_INT4";
        } else if (dtype == HELIOS_TYPE_STRING) {
            dstype = "HELIOS_TYPE_STRING";
        } else {
            assert(false);
        }


        std::cout << "   " << "[name: " << pd.at(i) << ", type: " << dstype << ", size: " << dsize << "]:" << std::endl;


        if (dtype == HELIOS_TYPE_INT) {
            std::vector<int> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_UINT) {
            std::vector<uint> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_FLOAT) {
            std::vector<float> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_DOUBLE) {
            std::vector<double> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_VEC2) {
            std::vector<vec2> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_VEC3) {
            std::vector<vec3> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_VEC4) {
            std::vector<vec4> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_INT2) {
            std::vector<int2> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_INT3) {
            std::vector<int3> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_INT4) {
            std::vector<int4> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_STRING) {
            std::vector<std::string> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else {
            assert(false);
        }
    }
    std::cout << "-------------------------------------------" << std::endl;
}

void Context::printObjectInfo(uint ObjID) const {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Info for ObjID " << ObjID << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    ObjectType otype = getObjectType(ObjID);
    std::string ostype;
    if (otype == 0) {
        ostype = "OBJECT_TYPE_TILE";
    } else if (otype == 1) {
        ostype = "OBJECT_TYPE_SPHERE";
    } else if (otype == 2) {
        ostype = "OBJECT_TYPE_TUBE";
    } else if (otype == 3) {
        ostype = "OBJECT_TYPE_BOX";
    } else if (otype == 4) {
        ostype = "OBJECT_TYPE_DISK";
    } else if (otype == 5) {
        ostype = "OBJECT_TYPE_POLYMESH";
    } else if (otype == 6) {
        ostype = "OBJECT_TYPE_CONE";
    }

    std::cout << "Type: " << ostype << std::endl;
    std::cout << "Object Bounding Box Center: " << getObjectCenter(ObjID) << std::endl;
    std::cout << "One-sided Surface Area: " << getObjectArea(ObjID) << std::endl;

    std::cout << "Primitive Count: " << getObjectPrimitiveCount(ObjID) << std::endl;

    if (areObjectPrimitivesComplete(ObjID)) {
        std::cout << "Object Primitives Complete" << std::endl;
    } else {
        std::cout << "Object Primitives Incomplete" << std::endl;
    }

    std::cout << "Primitive UUIDs: " << std::endl;
    std::vector<uint> primitive_UUIDs = getObjectPrimitiveUUIDs(ObjID);
    for (uint i = 0; i < primitive_UUIDs.size(); i++) {
        if (i < 5) {
            PrimitiveType ptype = getPrimitiveType(primitive_UUIDs.at(i));
            std::string pstype;
            if (ptype == 0) {
                pstype = "PRIMITIVE_TYPE_PATCH";
            } else if (ptype == 1) {
                pstype = "PRIMITIVE_TYPE_TRIANGLE";
            }
            std::cout << "   " << primitive_UUIDs.at(i) << " (" << pstype << ")" << std::endl;
        } else {
            std::cout << "   ..." << std::endl;
            PrimitiveType ptype = getPrimitiveType(primitive_UUIDs.at(primitive_UUIDs.size() - 2));
            std::string pstype;
            if (ptype == 0) {
                pstype = "PRIMITIVE_TYPE_PATCH";
            } else if (ptype == 1) {
                pstype = "PRIMITIVE_TYPE_TRIANGLE";
            }
            std::cout << "   " << primitive_UUIDs.at(primitive_UUIDs.size() - 2) << " (" << pstype << ")" << std::endl;
            ptype = getPrimitiveType(primitive_UUIDs.at(primitive_UUIDs.size() - 1));
            if (ptype == 0) {
                pstype = "PRIMITIVE_TYPE_PATCH";
            } else if (ptype == 1) {
                pstype = "PRIMITIVE_TYPE_TRIANGLE";
            }
            std::cout << "   " << primitive_UUIDs.at(primitive_UUIDs.size() - 1) << " (" << pstype << ")" << std::endl;
            break;
        }
    }

    if (otype == OBJECT_TYPE_TILE) {
        std::cout << "Tile Center: " << getTileObjectCenter(ObjID) << std::endl;
        std::cout << "Tile Size: " << getTileObjectSize(ObjID) << std::endl;
        std::cout << "Tile Subdivision Count: " << getTileObjectSubdivisionCount(ObjID) << std::endl;
        std::cout << "Tile Normal: " << getTileObjectNormal(ObjID) << std::endl;

        std::cout << "Tile Texture UV: " << std::endl;
        std::vector<vec2> uv = getTileObjectTextureUV(ObjID);
        for (uint i = 0; i < uv.size(); i++) {
            std::cout << "   " << uv.at(i) << std::endl;
        }

        std::cout << "Tile Vertices: " << std::endl;
        std::vector<vec3> primitive_vertices = getTileObjectVertices(ObjID);
        for (uint i = 0; i < primitive_vertices.size(); i++) {
            std::cout << "   " << primitive_vertices.at(i) << std::endl;
        }
    } else if (otype == OBJECT_TYPE_SPHERE) {
        std::cout << "Sphere Center: " << getSphereObjectCenter(ObjID) << std::endl;
        std::cout << "Sphere Radius: " << getSphereObjectRadius(ObjID) << std::endl;
        std::cout << "Sphere Subdivision Count: " << getSphereObjectSubdivisionCount(ObjID) << std::endl;
    } else if (otype == OBJECT_TYPE_TUBE) {
        std::cout << "Tube Subdivision Count: " << getTubeObjectSubdivisionCount(ObjID) << std::endl;
        std::cout << "Tube Nodes: " << std::endl;
        std::vector<vec3> nodes = getTubeObjectNodes(ObjID);
        for (uint i = 0; i < nodes.size(); i++) {
            if (i < 10) {
                std::cout << "   " << nodes.at(i) << std::endl;
            } else {
                std::cout << "      ..." << std::endl;
                std::cout << "   " << nodes.at(nodes.size() - 2) << std::endl;
                std::cout << "   " << nodes.at(nodes.size() - 1) << std::endl;
                break;
            }
        }
        std::cout << "Tube Node Radii: " << std::endl;
        std::vector<float> noderadii = getTubeObjectNodeRadii(ObjID);
        for (uint i = 0; i < noderadii.size(); i++) {
            if (i < 10) {
                std::cout << "   " << noderadii.at(i) << std::endl;
            } else {
                std::cout << "      ..." << std::endl;
                std::cout << "   " << noderadii.at(noderadii.size() - 2) << std::endl;
                std::cout << "   " << noderadii.at(noderadii.size() - 1) << std::endl;
                break;
            }
        }
        std::cout << "Tube Node Colors: " << std::endl;
        std::vector<helios::RGBcolor> nodecolors = getTubeObjectNodeColors(ObjID);
        for (uint i = 0; i < nodecolors.size(); i++) {
            if (i < 10) {
                std::cout << "   " << nodecolors.at(i) << std::endl;
            } else {
                std::cout << "      ..." << std::endl;
                std::cout << "      " << nodecolors.at(nodecolors.size() - 2) << std::endl;
                std::cout << "      " << nodecolors.at(nodecolors.size() - 1) << std::endl;
                break;
            }
        }
    } else if (otype == OBJECT_TYPE_BOX) {
        std::cout << "Box Center: " << getBoxObjectCenter(ObjID) << std::endl;
        std::cout << "Box Size: " << getBoxObjectSize(ObjID) << std::endl;
        std::cout << "Box Subdivision Count: " << getBoxObjectSubdivisionCount(ObjID) << std::endl;
    } else if (otype == OBJECT_TYPE_DISK) {
        std::cout << "Disk Center: " << getDiskObjectCenter(ObjID) << std::endl;
        std::cout << "Disk Size: " << getDiskObjectSize(ObjID) << std::endl;
        std::cout << "Disk Subdivision Count: " << getDiskObjectSubdivisionCount(ObjID) << std::endl;

        // }else if(type == OBJECT_TYPE_POLYMESH){
        // nothing for now
    } else if (otype == OBJECT_TYPE_CONE) {
        std::cout << "Cone Length: " << getConeObjectLength(ObjID) << std::endl;
        std::cout << "Cone Axis Unit Vector: " << getConeObjectAxisUnitVector(ObjID) << std::endl;
        std::cout << "Cone Subdivision Count: " << getConeObjectSubdivisionCount(ObjID) << std::endl;
        std::cout << "Cone Nodes: " << std::endl;
        std::vector<vec3> nodes = getConeObjectNodes(ObjID);
        for (uint i = 0; i < nodes.size(); i++) {
            std::cout << "   " << nodes.at(i) << std::endl;
        }
        std::cout << "Cone Node Radii: " << std::endl;
        std::vector<float> noderadii = getConeObjectNodeRadii(ObjID);
        for (uint i = 0; i < noderadii.size(); i++) {
            std::cout << "   " << noderadii.at(i) << std::endl;
        }
    }


    float T[16];
    getObjectTransformationMatrix(ObjID, T);
    std::cout << "Transform: " << std::endl;
    std::cout << "   " << T[0] << "      " << T[1] << "      " << T[2] << "      " << T[3] << std::endl;
    std::cout << "   " << T[4] << "      " << T[5] << "      " << T[6] << "      " << T[7] << std::endl;
    std::cout << "   " << T[8] << "      " << T[9] << "      " << T[10] << "      " << T[11] << std::endl;
    std::cout << "   " << T[12] << "      " << T[13] << "      " << T[14] << "      " << T[15] << std::endl;

    std::cout << "Texture File: " << getObjectTextureFile(ObjID) << std::endl;

    std::cout << "Object Data: " << std::endl;
    // Primitive* pointer = getPrimitivePointer_private(ObjID);
    std::vector<std::string> pd = listObjectData(ObjID);
    for (uint i = 0; i < pd.size(); i++) {
        uint dsize = getObjectDataSize(ObjID, pd.at(i).c_str());
        HeliosDataType dtype = getObjectDataType(pd.at(i).c_str());
        std::string dstype;

        if (dtype == HELIOS_TYPE_INT) {
            dstype = "HELIOS_TYPE_INT";
        } else if (dtype == HELIOS_TYPE_UINT) {
            dstype = "HELIOS_TYPE_UINT";
        } else if (dtype == HELIOS_TYPE_FLOAT) {
            dstype = "HELIOS_TYPE_FLOAT";
        } else if (dtype == HELIOS_TYPE_DOUBLE) {
            dstype = "HELIOS_TYPE_DOUBLE";
        } else if (dtype == HELIOS_TYPE_VEC2) {
            dstype = "HELIOS_TYPE_VEC2";
        } else if (dtype == HELIOS_TYPE_VEC3) {
            dstype = "HELIOS_TYPE_VEC3";
        } else if (dtype == HELIOS_TYPE_VEC4) {
            dstype = "HELIOS_TYPE_VEC4";
        } else if (dtype == HELIOS_TYPE_INT2) {
            dstype = "HELIOS_TYPE_INT2";
        } else if (dtype == HELIOS_TYPE_INT3) {
            dstype = "HELIOS_TYPE_INT3";
        } else if (dtype == HELIOS_TYPE_INT4) {
            dstype = "HELIOS_TYPE_INT4";
        } else if (dtype == HELIOS_TYPE_STRING) {
            dstype = "HELIOS_TYPE_STRING";
        } else {
            assert(false);
        }


        std::cout << "   " << "[name: " << pd.at(i) << ", type: " << dstype << ", size: " << dsize << "]:" << std::endl;


        if (dtype == HELIOS_TYPE_INT) {
            std::vector<int> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_UINT) {
            std::vector<uint> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_FLOAT) {
            std::vector<float> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_DOUBLE) {
            std::vector<double> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_VEC2) {
            std::vector<vec2> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_VEC3) {
            std::vector<vec3> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_VEC4) {
            std::vector<vec4> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_INT2) {
            std::vector<int2> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_INT3) {
            std::vector<int3> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_INT4) {
            std::vector<int4> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_STRING) {
            std::vector<std::string> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    break;
                }
            }
        } else {
            assert(false);
        }
    }
    std::cout << "-------------------------------------------" << std::endl;
}

CompoundObject *Context::getObjectPointer_private(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    return objects.at(ObjID);
}

void Context::hideObject(uint ObjID) {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::hideObject): Object ID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    objects.at(ObjID)->ishidden = true;
    for (uint UUID: objects.at(ObjID)->getPrimitiveUUIDs()) {
#ifdef HELIOS_DEBUG
        if (!doesPrimitiveExist(UUID)) {
            helios_runtime_error("ERROR (Context::hideObject): Primitive UUID of " + std::to_string(UUID) + " does not exist in the Context.");
        }
#endif
        primitives.at(UUID)->ishidden = true;
    }
}

void Context::hideObject(const std::vector<uint> &ObjIDs) {
    for (uint ObjID: ObjIDs) {
        hideObject(ObjID);
    }
}

void Context::showObject(uint ObjID) {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::showObject): Object ID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    objects.at(ObjID)->ishidden = false;
    for (uint UUID: objects.at(ObjID)->getPrimitiveUUIDs()) {
#ifdef HELIOS_DEBUG
        if (!doesPrimitiveExist(UUID)) {
            helios_runtime_error("ERROR (Context::showObject): Primitive UUID of " + std::to_string(UUID) + " does not exist in the Context.");
        }
#endif
        primitives.at(UUID)->ishidden = false;
    }
}

void Context::showObject(const std::vector<uint> &ObjIDs) {
    for (uint ObjID: ObjIDs) {
        showObject(ObjID);
    }
}

bool Context::isObjectHidden(uint ObjID) const {
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::isObjectHidden): Object ID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
    return objects.at(ObjID)->ishidden;
}

float Context::getObjectArea(uint ObjID) const {
    return getObjectPointer_private(ObjID)->getArea();
}

helios::vec3 Context::getObjectAverageNormal(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getObjectAverageNormal): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif

    const std::vector<uint> &UUIDs = objects.at(ObjID)->getPrimitiveUUIDs();

    vec3 norm_avg;
    for (uint UUID: UUIDs) {
        norm_avg += getPrimitiveNormal(UUID);
    }
    norm_avg.normalize();

    return norm_avg;
}

uint Context::getObjectPrimitiveCount(uint ObjID) const {
    return getObjectPointer_private(ObjID)->getPrimitiveCount();
}

helios::vec3 Context::getObjectCenter(uint ObjID) const {
    return getObjectPointer_private(ObjID)->getObjectCenter();
}

std::string Context::getObjectTextureFile(uint ObjID) const {
    return getObjectPointer_private(ObjID)->getTextureFile();
}

void Context::getObjectTransformationMatrix(uint ObjID, float (&T)[16]) const {
    getObjectPointer_private(ObjID)->getTransformationMatrix(T);
}

void Context::setObjectTransformationMatrix(uint ObjID, float (&T)[16]) const {
    getObjectPointer_private(ObjID)->setTransformationMatrix(T);
}

void Context::setObjectTransformationMatrix(const std::vector<uint> &ObjIDs, float (&T)[16]) const {
    for (uint ObjID: ObjIDs) {
        getObjectPointer_private(ObjID)->setTransformationMatrix(T);
    }
}

void Context::setObjectAverageNormal(uint ObjID, const vec3 &origin, const vec3 &new_normal) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("setObjectAverageNormal: invalid objectID");
    }
#endif

    // 1) Compute unit old & new normals
    vec3 oldN = normalize(getObjectAverageNormal(ObjID));
    vec3 newN = normalize(new_normal);

    // 2) Minimal‐angle axis & angle
    float d = std::clamp(oldN * newN, -1.f, 1.f);
    float angle = acosf(d);
    vec3 axis = cross(oldN, newN);
    if (axis.magnitude() < 1e-6f) {
        // pick any vector ⟂ oldN
        axis = (std::abs(oldN.x) < std::abs(oldN.z)) ? cross(oldN, {1, 0, 0}) : cross(oldN, {0, 0, 1});
    }
    axis = axis.normalize();

    // 3) Apply that minimal‐angle rotation to the compound (no pizza‐spin yet)
    //    NOTE: correct argument order is (objectID, angle, origin, axis)
    rotateObject(ObjID, angle, origin, axis);

    // 4) Fetch the updated transform and extract the world‐space “forward” (local +X)
    float M_mid[16];
    getObjectPointer_private(ObjID)->getTransformationMatrix(M_mid);

    vec3 localX{1, 0, 0};
    vec3 t1;
    // vecmult multiplies the 4×4 M_mid by v3 (w=0), writing into t1
    vecmult(M_mid, localX, t1);
    t1 = normalize(t1);

    // 5) Compute desired forward = world‐X projected into the new plane
    vec3 worldX{1, 0, 0};
    vec3 targ = worldX - newN * (newN * worldX);
    targ = normalize(targ);

    // 6) Compute signed twist about newN that carries t1→targ
    float twist = atan2f(newN * cross(t1, targ), // dot(newN, t1×targ)
                         t1 * targ // dot(t1, targ)
    );

    // 7) Apply that compensating twist about the same origin
    rotateObject(ObjID, twist, origin, newN);
}

void Context::setObjectOrigin(uint ObjID, const vec3 &origin) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::setObjectOrigin): invalid objectID");
    }
#endif
    objects.at(ObjID)->object_origin = origin;
}

bool Context::objectHasTexture(uint ObjID) const {
    return getObjectPointer_private(ObjID)->hasTexture();
}

void Context::setObjectColor(uint ObjID, const RGBcolor &color) const {
    getObjectPointer_private(ObjID)->setColor(color);
}

void Context::setObjectColor(const std::vector<uint> &ObjIDs, const RGBcolor &color) const {
    for (const uint ObjID: ObjIDs) {
        getObjectPointer_private(ObjID)->setColor(color);
    }
}

void Context::setObjectColor(uint ObjID, const RGBAcolor &color) const {
    getObjectPointer_private(ObjID)->setColor(color);
}

void Context::setObjectColor(const std::vector<uint> &ObjIDs, const RGBAcolor &color) const {
    for (const uint ObjID: ObjIDs) {
        getObjectPointer_private(ObjID)->setColor(color);
    }
}

bool Context::doesObjectContainPrimitive(uint ObjID, uint UUID) const {
    return getObjectPointer_private(ObjID)->doesObjectContainPrimitive(UUID);
}

void Context::overrideObjectTextureColor(uint ObjID) const {
    getObjectPointer_private(ObjID)->overrideTextureColor();
}

void Context::overrideObjectTextureColor(const std::vector<uint> &ObjIDs) const {
    for (uint ObjID: ObjIDs) {
        getObjectPointer_private(ObjID)->overrideTextureColor();
    }
}

void Context::useObjectTextureColor(uint ObjID) const {
    getObjectPointer_private(ObjID)->useTextureColor();
}

void Context::useObjectTextureColor(const std::vector<uint> &ObjIDs) {
    for (uint ObjID: ObjIDs) {
        getObjectPointer_private(ObjID)->useTextureColor();
    }
}

void Context::getObjectBoundingBox(uint ObjID, vec3 &min_corner, vec3 &max_corner) const {
    const std::vector ObjIDs{ObjID};
    getObjectBoundingBox(ObjIDs, min_corner, max_corner);
}

void Context::getObjectBoundingBox(const std::vector<uint> &ObjIDs, vec3 &min_corner, vec3 &max_corner) const {
    uint o = 0;
    for (uint ObjID: ObjIDs) {
        if (objects.find(ObjID) == objects.end()) {
            helios_runtime_error("ERROR (Context::getObjectBoundingBox): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
        }

        const std::vector<uint> &UUIDs = objects.at(ObjID)->getPrimitiveUUIDs();

        uint p = 0;
        for (const uint UUID: UUIDs) {
            const std::vector<vec3> &vertices = getPrimitiveVertices(UUID);

            if (p == 0 && o == 0) {
                min_corner = vertices.front();
                max_corner = min_corner;
                p++;
                continue;
            }

            for (const vec3 &vert: vertices) {
                if (vert.x < min_corner.x) {
                    min_corner.x = vert.x;
                }
                if (vert.y < min_corner.y) {
                    min_corner.y = vert.y;
                }
                if (vert.z < min_corner.z) {
                    min_corner.z = vert.z;
                }
                if (vert.x > max_corner.x) {
                    max_corner.x = vert.x;
                }
                if (vert.y > max_corner.y) {
                    max_corner.y = vert.y;
                }
                if (vert.z > max_corner.z) {
                    max_corner.z = vert.z;
                }
            }
        }

        o++;
    }
}

Tile *Context::getTileObjectPointer_private(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getTileObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    } else if (objects.at(ObjID)->getObjectType() != OBJECT_TYPE_TILE) {
        helios_runtime_error("ERROR (Context::getTileObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Tile Object.");
    }
#endif
    return dynamic_cast<Tile *>(objects.at(ObjID));
}

Sphere *Context::getSphereObjectPointer_private(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getSphereObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    } else if (objects.at(ObjID)->getObjectType() != OBJECT_TYPE_SPHERE) {
        helios_runtime_error("ERROR (Context::getSphereObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Sphere Object.");
    }
#endif
    return dynamic_cast<Sphere *>(objects.at(ObjID));
}

Tube *Context::getTubeObjectPointer_private(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getTubeObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    } else if (objects.at(ObjID)->getObjectType() != OBJECT_TYPE_TUBE) {
        helios_runtime_error("ERROR (Context::getTubeObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Tube Object.");
    }
#endif
    return dynamic_cast<Tube *>(objects.at(ObjID));
}

Box *Context::getBoxObjectPointer_private(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getBoxObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    } else if (objects.at(ObjID)->getObjectType() != OBJECT_TYPE_BOX) {
        helios_runtime_error("ERROR (Context::getBoxObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Box Object.");
    }
#endif
    return dynamic_cast<Box *>(objects.at(ObjID));
}

Disk *Context::getDiskObjectPointer_private(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getDiskObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    } else if (objects.at(ObjID)->getObjectType() != OBJECT_TYPE_DISK) {
        helios_runtime_error("ERROR (Context::getDiskObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Disk Object.");
    }
#endif
    return dynamic_cast<Disk *>(objects.at(ObjID));
}

Polymesh *Context::getPolymeshObjectPointer_private(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getPolymeshObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    } else if (objects.at(ObjID)->getObjectType() != OBJECT_TYPE_POLYMESH) {
        helios_runtime_error("ERROR (Context::getPolymeshObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Polymesh Object.");
    }
#endif
    return dynamic_cast<Polymesh *>(objects.at(ObjID));
}

Cone *Context::getConeObjectPointer_private(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getConeObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    } else if (objects.at(ObjID)->getObjectType() != OBJECT_TYPE_CONE) {
        helios_runtime_error("ERROR (Context::getConeObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Cone Object.");
    }
#endif
    return dynamic_cast<Cone *>(objects.at(ObjID));
}

helios::vec3 Context::getTileObjectCenter(uint ObjID) const {
    return getTileObjectPointer_private(ObjID)->getCenter();
}

helios::vec2 Context::getTileObjectSize(uint ObjID) const {
    return getTileObjectPointer_private(ObjID)->getSize();
}

helios::int2 Context::getTileObjectSubdivisionCount(uint ObjID) const {
    return getTileObjectPointer_private(ObjID)->getSubdivisionCount();
}

helios::vec3 Context::getTileObjectNormal(uint ObjID) const {
    return getTileObjectPointer_private(ObjID)->getNormal();
}

std::vector<helios::vec2> Context::getTileObjectTextureUV(uint ObjID) const {
    return getTileObjectPointer_private(ObjID)->getTextureUV();
}

std::vector<helios::vec3> Context::getTileObjectVertices(uint ObjID) const {
    return getTileObjectPointer_private(ObjID)->getVertices();
}

helios::vec3 Context::getSphereObjectCenter(uint ObjID) const {
    return getSphereObjectPointer_private(ObjID)->getCenter();
}

helios::vec3 Context::getSphereObjectRadius(uint ObjID) const {
    return getSphereObjectPointer_private(ObjID)->getRadius();
}

uint Context::getSphereObjectSubdivisionCount(uint ObjID) const {
    return getSphereObjectPointer_private(ObjID)->getSubdivisionCount();
}

float Context::getSphereObjectVolume(uint ObjID) const {
    return getSphereObjectPointer_private(ObjID)->getVolume();
}

uint Context::getTubeObjectSubdivisionCount(uint ObjID) const {
    return getTubeObjectPointer_private(ObjID)->getSubdivisionCount();
}

std::vector<helios::vec3> Context::getTubeObjectNodes(uint ObjID) const {
    return getTubeObjectPointer_private(ObjID)->getNodes();
}

uint Context::getTubeObjectNodeCount(uint ObjID) const {
    return getTubeObjectPointer_private(ObjID)->getNodeCount();
}

std::vector<float> Context::getTubeObjectNodeRadii(uint ObjID) const {
    return getTubeObjectPointer_private(ObjID)->getNodeRadii();
}

std::vector<RGBcolor> Context::getTubeObjectNodeColors(uint ObjID) const {
    return getTubeObjectPointer_private(ObjID)->getNodeColors();
}

float Context::getTubeObjectVolume(uint ObjID) const {
    return getTubeObjectPointer_private(ObjID)->getVolume();
}

float Context::getTubeObjectSegmentVolume(uint ObjID, uint segment_index) const {
    return getTubeObjectPointer_private(ObjID)->getSegmentVolume(segment_index);
}

void Context::appendTubeSegment(uint ObjID, const helios::vec3 &node_position, float node_radius, const RGBcolor &node_color) {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::appendTubeSegment): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    dynamic_cast<Tube *>(objects.at(ObjID))->appendTubeSegment(node_position, node_radius, node_color);
}

void Context::appendTubeSegment(uint ObjID, const helios::vec3 &node_position, float node_radius, const char *texturefile, const helios::vec2 &textureuv_ufrac) {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::appendTubeSegment): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    dynamic_cast<Tube *>(objects.at(ObjID))->appendTubeSegment(node_position, node_radius, texturefile, textureuv_ufrac);
}

void Context::scaleTubeGirth(uint ObjID, float scale_factor) {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::scaleTubeGirth): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    dynamic_cast<Tube *>(objects.at(ObjID))->scaleTubeGirth(scale_factor);
}

void Context::setTubeRadii(uint ObjID, const std::vector<float> &node_radii) {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::setTubeRadii): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    dynamic_cast<Tube *>(objects.at(ObjID))->setTubeRadii(node_radii);
}

void Context::scaleTubeLength(uint ObjID, float scale_factor) {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::scaleTubeLength): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    dynamic_cast<Tube *>(objects.at(ObjID))->scaleTubeLength(scale_factor);
}

void Context::pruneTubeNodes(uint ObjID, uint node_index) {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::pruneTubeNodes): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    dynamic_cast<Tube *>(objects.at(ObjID))->pruneTubeNodes(node_index);
}

void Context::setTubeNodes(uint ObjID, const std::vector<helios::vec3> &node_xyz) {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::setTubeNodes): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    dynamic_cast<Tube *>(objects.at(ObjID))->setTubeNodes(node_xyz);
}

helios::vec3 Context::getBoxObjectCenter(uint ObjID) const {
    return getBoxObjectPointer_private(ObjID)->getCenter();
}

helios::vec3 Context::getBoxObjectSize(uint ObjID) const {
    return getBoxObjectPointer_private(ObjID)->getSize();
}

helios::int3 Context::getBoxObjectSubdivisionCount(uint ObjID) const {
    return getBoxObjectPointer_private(ObjID)->getSubdivisionCount();
}

float Context::getBoxObjectVolume(uint ObjID) const {
    return getBoxObjectPointer_private(ObjID)->getVolume();
}

helios::vec3 Context::getDiskObjectCenter(uint ObjID) const {
    return getDiskObjectPointer_private(ObjID)->getCenter();
}

helios::vec2 Context::getDiskObjectSize(uint ObjID) const {
    return getDiskObjectPointer_private(ObjID)->getSize();
}

uint Context::getDiskObjectSubdivisionCount(uint ObjID) const {
    return getDiskObjectPointer_private(ObjID)->getSubdivisionCount().x;
}

uint Context::getConeObjectSubdivisionCount(uint ObjID) const {
    return getConeObjectPointer_private(ObjID)->getSubdivisionCount();
}

std::vector<helios::vec3> Context::getConeObjectNodes(uint ObjID) const {
    return getConeObjectPointer_private(ObjID)->getNodeCoordinates();
}

std::vector<float> Context::getConeObjectNodeRadii(uint ObjID) const {
    return getConeObjectPointer_private(ObjID)->getNodeRadii();
}

helios::vec3 Context::getConeObjectNode(uint ObjID, int number) const {
    return getConeObjectPointer_private(ObjID)->getNodeCoordinate(number);
}

float Context::getConeObjectNodeRadius(uint ObjID, int number) const {
    return getConeObjectPointer_private(ObjID)->getNodeRadius(number);
}

helios::vec3 Context::getConeObjectAxisUnitVector(uint ObjID) const {
    return getConeObjectPointer_private(ObjID)->getAxisUnitVector();
}

float Context::getConeObjectLength(uint ObjID) const {
    return getConeObjectPointer_private(ObjID)->getLength();
}

float Context::getConeObjectVolume(uint ObjID) const {
    return getConeObjectPointer_private(ObjID)->getVolume();
}

float Context::getPolymeshObjectVolume(uint ObjID) const {
    return getPolymeshObjectPointer_private(ObjID)->getVolume();
}
