/**
 * \file "RadiationCameraEXIF.cpp" Helpers that convert a RadiationCamera + Context state
 *                                  into a populated helios::ImageEXIFData for embedding in
 *                                  the JPEG output of writeCameraImage().
 *
 * Copyright (C) 2016-2026 Brian Bailey
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

#include "RadiationModel.h"

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>

#include "global.h"

using namespace helios;

namespace {

    //! Format a Date + Time as "YYYY:MM:DD HH:MM:SS" (EXIF datetime convention).
    std::string formatExifDateTime(const helios::Date &date, const helios::Time &time) {
        std::ostringstream oss;
        oss << std::setw(4) << std::setfill('0') << date.year << ":"
            << std::setw(2) << std::setfill('0') << date.month << ":"
            << std::setw(2) << std::setfill('0') << date.day << " "
            << std::setw(2) << std::setfill('0') << time.hour << ":"
            << std::setw(2) << std::setfill('0') << time.minute << ":"
            << std::setw(2) << std::setfill('0') << time.second;
        return oss.str();
    }

    //! Convert Helios local-time + UTC_offset (Helios's +West convention) into UTC date + time.
    /**
     * Helios `Location::UTC_offset` is positive for time zones West of UTC, so `UTC = local + offset`.
     * Day rollover (forward or backward, including year boundaries) is handled via Julian-day
     * arithmetic so leap years are correct.
     */
    void toUTC(const helios::Date &local_date, const helios::Time &local_time, float utc_offset_helios_west_positive,
               helios::Date &utc_date, helios::Time &utc_time) {

        const int hour_offset_int = static_cast<int>(std::floor(utc_offset_helios_west_positive));
        const float fractional_hour = utc_offset_helios_west_positive - static_cast<float>(hour_offset_int);
        const int minute_offset = static_cast<int>(std::round(fractional_hour * 60.0f));

        int total_minutes = static_cast<int>(local_time.hour) * 60 + static_cast<int>(local_time.minute) + hour_offset_int * 60 + minute_offset;
        const int total_seconds = static_cast<int>(local_time.second);

        int day_shift = 0;
        while (total_minutes < 0) {
            total_minutes += 24 * 60;
            day_shift -= 1;
        }
        while (total_minutes >= 24 * 60) {
            total_minutes -= 24 * 60;
            day_shift += 1;
        }

        // Resolve day_shift via Julian-day arithmetic with year wraparound.
        int year = local_date.year;
        int jd = local_date.JulianDay() + day_shift;
        while (jd < 1) {
            year -= 1;
            const int prev_year_days = (((year % 4 == 0) && (year % 100 != 0)) || (year % 400 == 0)) ? 366 : 365;
            jd += prev_year_days;
        }
        while (true) {
            const int year_days = (((year % 4 == 0) && (year % 100 != 0)) || (year % 400 == 0)) ? 366 : 365;
            if (jd <= year_days) break;
            jd -= year_days;
            year += 1;
        }

        utc_date = helios::Julian2Calendar(jd, year);
        utc_time.hour = total_minutes / 60;
        utc_time.minute = total_minutes % 60;
        utc_time.second = total_seconds;
    }

    //! Format the standard +E UTC offset as "+HH:MM" / "-HH:MM" given the Helios +W offset.
    std::string formatStandardOffset(float utc_offset_helios_west_positive) {
        // Standard convention is positive for East of UTC, so we negate.
        float std_offset = -utc_offset_helios_west_positive;
        char sign = (std_offset >= 0.f) ? '+' : '-';
        float abs_off = std::fabs(std_offset);
        int hh = static_cast<int>(std::floor(abs_off));
        int mm = static_cast<int>(std::round((abs_off - static_cast<float>(hh)) * 60.0f));
        // Handle minute rollover from rounding.
        if (mm >= 60) {
            mm -= 60;
            hh += 1;
        }
        char buf[8];
        std::snprintf(buf, sizeof(buf), "%c%02d:%02d", sign, hh, mm);
        return std::string(buf);
    }

} // namespace

void RadiationModel::populateImageEXIF(const std::string &camera_label, helios::ImageEXIFData &exif) const {

    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::populateImageEXIF): Camera '" + camera_label + "' does not exist.");
    }

    const auto &cam = cameras.at(camera_label);

    // ---- IFD0: identification + datetime ----

    // Make: use the explicit manufacturer field if set (typical for library cameras); otherwise
    // fall back to "Helios". For library-loaded cameras the `model` field carries the legacy
    // "<manufacturer> <model>" concatenation, so we strip the manufacturer prefix when writing
    // the Model tag — photogrammetry sensor databases key on Make and Model separately.
    if (!cam.manufacturer.empty()) {
        exif.make = cam.manufacturer;
        const std::string prefix = cam.manufacturer + " ";
        if (cam.model.size() > prefix.size() && cam.model.compare(0, prefix.size(), prefix) == 0) {
            exif.model = cam.model.substr(prefix.size());
        } else {
            exif.model = cam.model;
        }
    } else {
        exif.make = "Helios";
        exif.model = cam.model.empty() ? std::string("HeliosCamera") : cam.model;
    }
    exif.software = "Helios";
    exif.orientation = 1;

    const helios::Date local_date = context->getDate();
    const helios::Time local_time = context->getTime();
    const helios::Location loc = context->getLocation();

    const std::string local_dt = formatExifDateTime(local_date, local_time);
    exif.datetime = local_dt;
    exif.datetime_original = local_dt;
    exif.datetime_digitized = local_dt;
    exif.offset_time = formatStandardOffset(loc.UTC_offset);

    // ---- Exif SubIFD: intrinsics + exposure ----

    // Back-compute optical focal length in mm from HFOV and physical sensor width so the EXIF
    // value matches the actual rendering geometry (rather than `cam.lens_focal_length`, which
    // may have been set independently). This mirrors what populateCameraMetadata() exports
    // to the JSON sidecar.
    if (cam.sensor_width_mm > 0.f && cam.HFOV_degrees > 0.f) {
        const float HFOV_rad = cam.HFOV_degrees * static_cast<float>(M_PI) / 180.f;
        const float optical_focal_length_mm = cam.sensor_width_mm / (2.0f * std::tan(HFOV_rad / 2.0f));
        exif.focal_length_mm = optical_focal_length_mm;

        // FocalPlaneXResolution / YResolution: pixels per cm (unit = 3).
        const float sensor_width_cm = cam.sensor_width_mm / 10.0f;
        const float sensor_height_mm = (cam.FOV_aspect_ratio > 0.f) ? (cam.sensor_width_mm / cam.FOV_aspect_ratio) : cam.sensor_width_mm;
        const float sensor_height_cm = sensor_height_mm / 10.0f;
        if (sensor_width_cm > 0.f && cam.resolution.x > 0) {
            exif.focal_plane_x_resolution = static_cast<float>(cam.resolution.x) / sensor_width_cm;
        }
        if (sensor_height_cm > 0.f && cam.resolution.y > 0) {
            exif.focal_plane_y_resolution = static_cast<float>(cam.resolution.y) / sensor_height_cm;
        }
        exif.focal_plane_resolution_unit = 3; // cm
    }

    exif.pixel_x_dimension = cam.resolution.x;
    exif.pixel_y_dimension = cam.resolution.y;
    exif.exposure_time_s = cam.shutter_speed;

    // FNumber + MaxApertureValue (APEX). The radiation camera has a single fixed aperture
    // determined by lens_diameter; if the aperture is closed (pinhole, lens_diameter == 0)
    // both tags are omitted.
    if (cam.lens_diameter > 0.f && exif.focal_length_mm > 0.f) {
        const float lens_diameter_mm = cam.lens_diameter * 1000.0f; // m -> mm
        const float f_number = exif.focal_length_mm / lens_diameter_mm;
        exif.f_number = f_number;
        // APEX: Av = 2 * log2(N).
        exif.max_aperture_value_apex = 2.0f * std::log2(f_number);
    }

    // SubjectDistance: working distance from camera to focal plane (`cam.focal_length` in meters).
    if (cam.focal_length > 0.f) {
        exif.subject_distance_m = cam.focal_length;
    }

    // FocalLengthIn35mmFilm: scale the optical focal length by 36 mm / sensor_width_mm.
    if (exif.focal_length_mm > 0.f && cam.sensor_width_mm > 0.f) {
        const float fl35 = exif.focal_length_mm * (36.0f / cam.sensor_width_mm);
        if (fl35 > 0.f) {
            exif.focal_length_in_35mm = static_cast<unsigned int>(std::round(fl35));
        }
    }

    // ExposureMode: 0 = auto, 1 = manual. "ISOxxx" specifies ISO directly so the camera is
    // in manual exposure control.
    if (cam.exposure == "auto") {
        exif.exposure_mode = 0;
    } else if (cam.exposure == "manual" ||
               (cam.exposure.size() > 3 && cam.exposure.compare(0, 3, "ISO") == 0)) {
        exif.exposure_mode = 1;
    }

    // WhiteBalance: 0 = auto, 1 = manual (here used for "off" / fixed gains).
    if (cam.white_balance == "auto") {
        exif.white_balance = 0;
    } else if (!cam.white_balance.empty()) {
        exif.white_balance = 1;
    }

    // DigitalZoomRatio: written as-is from cam.camera_zoom (1.0 means no zoom).
    if (cam.camera_zoom > 0.f) {
        exif.digital_zoom_ratio = cam.camera_zoom;
    }

    // ExposureBiasValue: the applied auto-exposure gain expressed in EV stops.
    // `applied_exposure_gain` is a linear multiplier; EV = log2(gain). gain==1 => 0 EV.
    if (cam.applied_exposure_gain > 0.f) {
        exif.exposure_bias_ev = std::log2(cam.applied_exposure_gain);
        exif.has_exposure_bias = true;
    }

    // Parse "ISOxxx" prefix from the exposure mode string (e.g. "ISO100").
    if (cam.exposure.size() > 3 && cam.exposure.compare(0, 3, "ISO") == 0) {
        try {
            int iso_int = std::stoi(cam.exposure.substr(3));
            if (iso_int > 0) {
                exif.iso = static_cast<unsigned int>(iso_int);
            }
        } catch (...) {
            // Not an integer ISO; leave exif.iso at 0 so the tag is omitted.
        }
    }

    exif.lens_make = cam.lens_make;
    exif.lens_model = cam.lens_model;
    exif.lens_specification = cam.lens_specification;

    // ---- GPS IFD: lat/lon/alt + bearing + UTC date/time ----

    // Flat-earth conversion centered at the Location origin. Convention: +Y = North, +X = East,
    // +Z = up. Helios's longitude convention is +W; standard EXIF is +E, so we negate.
    const double lat0_deg = static_cast<double>(loc.latitude_deg);
    const double lon0_deg_std = -static_cast<double>(loc.longitude_deg); // +W -> +E
    const double lat0_rad = lat0_deg * M_PI / 180.0;
    const double meters_per_deg_lat = 111320.0;
    const double meters_per_deg_lon = 111320.0 * std::cos(lat0_rad);

    exif.latitude_deg = lat0_deg + (static_cast<double>(cam.position.y) / meters_per_deg_lat);
    exif.longitude_deg = lon0_deg_std;
    if (meters_per_deg_lon > 0.0) {
        exif.longitude_deg += (static_cast<double>(cam.position.x) / meters_per_deg_lon);
    }
    exif.altitude_m = static_cast<double>(loc.altitude_m) + static_cast<double>(cam.position.z);

    // Bearing of the optical axis projected onto the horizontal plane (0 = North, 90 = East).
    helios::vec3 dir = cam.lookat - cam.position;
    dir.normalize();
    double yaw_deg = std::atan2(static_cast<double>(dir.x), static_cast<double>(dir.y)) * 180.0 / M_PI;
    if (yaw_deg < 0.0) yaw_deg += 360.0;
    exif.img_direction_deg = yaw_deg;

    // GPS UTC date/time.
    helios::Date utc_date;
    helios::Time utc_time;
    toUTC(local_date, local_time, loc.UTC_offset, utc_date, utc_time);

    {
        std::ostringstream ds;
        ds << std::setw(4) << std::setfill('0') << utc_date.year << ":"
           << std::setw(2) << std::setfill('0') << utc_date.month << ":"
           << std::setw(2) << std::setfill('0') << utc_date.day;
        exif.gps_datestamp = ds.str();
    }
    {
        std::ostringstream ts;
        ts << std::setw(2) << std::setfill('0') << utc_time.hour << ":"
           << std::setw(2) << std::setfill('0') << utc_time.minute << ":"
           << std::setw(2) << std::setfill('0') << utc_time.second;
        exif.gps_timestamp_hms = ts.str();
    }

    exif.gps_valid = true;

    // ---- XMP: Pix4D Camera namespace yaw/pitch/roll ----
    //
    // pitch is "+down" (Pix4D convention). roll is 0 in v1 because RadiationCamera has no
    // up-vector field; this is documented as a known limitation.
    exif.yaw_deg = yaw_deg;
    exif.pitch_deg = -static_cast<double>(std::asin(static_cast<double>(dir.z))) * 180.0 / M_PI;
    exif.roll_deg = 0.0;
    exif.xmp_valid = true;
}
