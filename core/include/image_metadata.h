/**
 * \file "image_metadata.h" Image metadata structures for EXIF/XMP embedding.
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

#ifndef HELIOS_IMAGE_METADATA_H
#define HELIOS_IMAGE_METADATA_H

#include <string>

namespace helios {

    //! Image metadata fields for EXIF (APP1 / TIFF IFDs) and XMP embedding in JPEG files.
    /**
     * Populated by callers (e.g., the radiation plugin) and consumed by the metadata-aware
     * `helios::writeJPEG` overload. Fields left at their default-empty/zero values are
     * omitted from the written EXIF/XMP segments. All angles are in degrees and all
     * geographic coordinates use the standard convention (+N latitude, +E longitude) —
     * callers are responsible for converting from any internal convention (e.g., Helios's
     * non-standard +W longitude in `helios::Location`) before populating this struct.
     */
    struct ImageEXIFData {

        // --- IFD0 (primary image) ---

        //! Camera manufacturer string (EXIF tag 0x010F).
        std::string make = "Helios";
        //! Camera model string (EXIF tag 0x0110).
        std::string model;
        //! Software / creator-tool string (EXIF tag 0x0131).
        std::string software = "Helios";
        //! File modification timestamp, formatted "YYYY:MM:DD HH:MM:SS" (local time) (EXIF tag 0x0132).
        std::string datetime;
        //! Image orientation; 1 = normal (EXIF tag 0x0112).
        unsigned int orientation = 1;

        // --- Exif SubIFD ---

        //! Original capture timestamp, formatted "YYYY:MM:DD HH:MM:SS" (local time) (EXIF tag 0x9003).
        std::string datetime_original;
        //! Digitization timestamp, formatted "YYYY:MM:DD HH:MM:SS" (local time) (EXIF tag 0x9004).
        std::string datetime_digitized;
        //! UTC offset for the datetime fields, formatted "+HH:MM" or "-HH:MM" (standard +E convention) (EXIF tag 0x9011).
        std::string offset_time;
        //! Optical focal length in mm. 0 => tag omitted (EXIF tag 0x920A).
        float focal_length_mm = 0.f;
        //! Pixels-per-unit along the X (horizontal) sensor axis (EXIF tag 0xA20E).
        float focal_plane_x_resolution = 0.f;
        //! Pixels-per-unit along the Y (vertical) sensor axis (EXIF tag 0xA20F).
        float focal_plane_y_resolution = 0.f;
        //! Resolution unit for focal-plane resolution. 2 = inch, 3 = cm. Default 3 (EXIF tag 0xA210).
        unsigned int focal_plane_resolution_unit = 3;
        //! Exposure time in seconds. 0 => tag omitted (EXIF tag 0x829A).
        float exposure_time_s = 0.f;
        //! Lens f-number (aperture). 0 => tag omitted (EXIF tag 0x829D).
        float f_number = 0.f;
        //! ISO speed rating. 0 => tag omitted (EXIF tag 0x8827).
        unsigned int iso = 0;
        //! Exposure bias / gain in EV stops. Set `has_exposure_bias` to true to emit; defaults to 0 EV (EXIF tag 0x9204).
        float exposure_bias_ev = 0.f;
        //! Whether `exposure_bias_ev` should be emitted (a 0 EV bias is meaningful, so a sentinel flag is needed).
        bool has_exposure_bias = false;
        //! Maximum lens aperture in APEX units. 0 => tag omitted (EXIF tag 0x9205).
        float max_aperture_value_apex = 0.f;
        //! Distance from camera to subject (working distance) in meters. 0 => tag omitted (EXIF tag 0x9206).
        float subject_distance_m = 0.f;
        //! Image width in pixels (EXIF tag 0xA002).
        unsigned int pixel_x_dimension = 0;
        //! Image height in pixels (EXIF tag 0xA003).
        unsigned int pixel_y_dimension = 0;
        //! Lens manufacturer string (EXIF tag 0xA433).
        std::string lens_make;
        //! Lens model string (EXIF tag 0xA434).
        std::string lens_model;
        //! Lens specification string (EXIF tag 0xA432, written as ASCII fallback).
        std::string lens_specification;
        //! Exposure mode: 0=Auto, 1=Manual, 2=Auto bracket. UINT_MAX => tag omitted (EXIF tag 0xA402).
        unsigned int exposure_mode = (unsigned int) -1;
        //! White balance: 0=Auto, 1=Manual. UINT_MAX => tag omitted (EXIF tag 0xA403).
        unsigned int white_balance = (unsigned int) -1;
        //! Digital zoom ratio. 0 => tag omitted; 1.0 indicates no zoom (EXIF tag 0xA404).
        float digital_zoom_ratio = 0.f;
        //! Focal length in mm scaled to 35 mm-film equivalent. 0 => tag omitted (EXIF tag 0xA405).
        unsigned int focal_length_in_35mm = 0;

        // --- GPS IFD ---

        //! Whether GPS fields are populated and should be written.
        bool gps_valid = false;
        //! Latitude in decimal degrees, +N / -S (standard convention).
        double latitude_deg = 0.0;
        //! Longitude in decimal degrees, +E / -W (standard convention; flipped from Helios's internal +W).
        double longitude_deg = 0.0;
        //! Altitude in meters above sea level (or above the chosen reference if not WGS84).
        double altitude_m = 0.0;
        //! Image direction (bearing of optical axis projected onto horizontal plane), 0..360 from true North.
        double img_direction_deg = 0.0;
        //! GPS date stamp formatted "YYYY:MM:DD" in UTC.
        std::string gps_datestamp;
        //! GPS time stamp formatted "HH:MM:SS" in UTC.
        std::string gps_timestamp_hms;

        // --- XMP (Pix4D Camera namespace http://pix4d.com/camera/1.0) ---

        //! Whether XMP orientation fields are populated and should be written.
        bool xmp_valid = false;
        //! Camera yaw in degrees, 0=North, 90=East (Camera:Yaw).
        double yaw_deg = 0.0;
        //! Camera pitch in degrees, +down convention (Camera:Pitch).
        double pitch_deg = 0.0;
        //! Camera roll in degrees about the optical axis (Camera:Roll). v1: always 0 — no up-vector in RadiationCamera.
        double roll_deg = 0.0;
    };

}

#endif
