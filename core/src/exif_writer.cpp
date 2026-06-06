/**
 * \file "exif_writer.cpp" Hand-rolled EXIF (TIFF/APP1) and XMP (APP1) segment builders for JPEG embedding.
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

#include "exif_writer.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "global.h"
#include "pugixml.hpp"

namespace helios::detail {

    namespace {

        // EXIF/TIFF data types (TIFF 6.0 / EXIF 2.32)
        constexpr uint16_t TYPE_BYTE = 1;
        constexpr uint16_t TYPE_ASCII = 2;
        constexpr uint16_t TYPE_SHORT = 3;
        constexpr uint16_t TYPE_LONG = 4;
        constexpr uint16_t TYPE_RATIONAL = 5;
        constexpr uint16_t TYPE_UNDEFINED = 7;
        constexpr uint16_t TYPE_SRATIONAL = 10;

        // JPEG APP1 segment payload hard limit (65535 - 2 length bytes - 0)
        constexpr size_t APP1_MAX_PAYLOAD = 65533;

        // Number of bytes occupied by an IFD entry's "value/offset" field
        constexpr size_t IFD_ENTRY_VALUE_BYTES = 4;

        // ---- Little-endian raw writers ----

        void putU16(std::vector<unsigned char> &b, uint16_t v) {
            b.push_back(static_cast<unsigned char>(v & 0xFF));
            b.push_back(static_cast<unsigned char>((v >> 8) & 0xFF));
        }
        void putU32(std::vector<unsigned char> &b, uint32_t v) {
            b.push_back(static_cast<unsigned char>(v & 0xFF));
            b.push_back(static_cast<unsigned char>((v >> 8) & 0xFF));
            b.push_back(static_cast<unsigned char>((v >> 16) & 0xFF));
            b.push_back(static_cast<unsigned char>((v >> 24) & 0xFF));
        }
        void patchU32(std::vector<unsigned char> &b, size_t pos, uint32_t v) {
            b[pos] = static_cast<unsigned char>(v & 0xFF);
            b[pos + 1] = static_cast<unsigned char>((v >> 8) & 0xFF);
            b[pos + 2] = static_cast<unsigned char>((v >> 16) & 0xFF);
            b[pos + 3] = static_cast<unsigned char>((v >> 24) & 0xFF);
        }

        // ---- IFD entry representation ----
        //
        // Each entry has: tag, type, count, and either an inline value (<=4 bytes) or a pointer
        // to data appended to a trailing data blob. We collect entries first, then emit the IFD
        // in two passes: pass 1 finalizes data-blob offsets given a base offset; pass 2 emits the
        // 12-byte entry records and the trailing data blob.
        struct IFDEntry {
            uint16_t tag = 0;
            uint16_t type = 0;
            uint32_t count = 0;
            // If the value fits in 4 bytes, it is stored inline here (padded with zeros).
            std::array<unsigned char, 4> inline_value{};
            // If it does not fit, the data is stored here and a pointer is written instead.
            std::vector<unsigned char> external_data;
            bool inline_only = true;
            // For sub-IFD pointer tags (ExifSubIFD, GPSSubIFD): inline_value carries an offset
            // patched in later. The patch site within the entry record itself is at byte offset 8
            // (tag=2 + type=2 + count=4 = 8).
            bool is_pointer_placeholder = false;
            // Bookkeeping for pointer placeholders to know which sub-IFD they point to.
            int pointer_target_index = -1; // 0 = ExifSubIFD, 1 = GPS IFD
        };

        size_t typeSize(uint16_t t) {
            switch (t) {
                case TYPE_BYTE:
                case TYPE_ASCII:
                case TYPE_UNDEFINED:
                    return 1;
                case TYPE_SHORT:
                    return 2;
                case TYPE_LONG:
                    return 4;
                case TYPE_RATIONAL:
                case TYPE_SRATIONAL:
                    return 8;
                default:
                    return 1;
            }
        }

        // Pack a value into IFDEntry, choosing inline vs external based on byte length.
        void finalizeEntry(IFDEntry &e, const std::vector<unsigned char> &raw) {
            const size_t bytes = raw.size();
            if (bytes <= IFD_ENTRY_VALUE_BYTES) {
                e.inline_only = true;
                e.inline_value.fill(0);
                std::memcpy(e.inline_value.data(), raw.data(), bytes);
            } else {
                e.inline_only = false;
                e.external_data = raw;
            }
        }

        IFDEntry makeAscii(uint16_t tag, const std::string &s) {
            // EXIF ASCII strings are NUL-terminated; count includes the NUL byte.
            IFDEntry e;
            e.tag = tag;
            e.type = TYPE_ASCII;
            e.count = static_cast<uint32_t>(s.size() + 1);
            std::vector<unsigned char> raw(s.size() + 1, 0);
            std::memcpy(raw.data(), s.data(), s.size());
            finalizeEntry(e, raw);
            return e;
        }

        IFDEntry makeUndefined(uint16_t tag, const std::string &s) {
            // UNDEFINED is raw bytes; not NUL-terminated.
            IFDEntry e;
            e.tag = tag;
            e.type = TYPE_UNDEFINED;
            e.count = static_cast<uint32_t>(s.size());
            std::vector<unsigned char> raw(s.size());
            if (!s.empty()) {
                std::memcpy(raw.data(), s.data(), s.size());
            }
            finalizeEntry(e, raw);
            return e;
        }

        IFDEntry makeShort(uint16_t tag, uint16_t v) {
            IFDEntry e;
            e.tag = tag;
            e.type = TYPE_SHORT;
            e.count = 1;
            // SHORT values <=4 bytes always inline. EXIF requires SHORT inline values to be
            // written in the low half of the 4-byte value field (little-endian).
            std::vector<unsigned char> raw(2);
            raw[0] = static_cast<unsigned char>(v & 0xFF);
            raw[1] = static_cast<unsigned char>((v >> 8) & 0xFF);
            finalizeEntry(e, raw);
            return e;
        }

        IFDEntry makeLong(uint16_t tag, uint32_t v) {
            IFDEntry e;
            e.tag = tag;
            e.type = TYPE_LONG;
            e.count = 1;
            std::vector<unsigned char> raw(4);
            raw[0] = static_cast<unsigned char>(v & 0xFF);
            raw[1] = static_cast<unsigned char>((v >> 8) & 0xFF);
            raw[2] = static_cast<unsigned char>((v >> 16) & 0xFF);
            raw[3] = static_cast<unsigned char>((v >> 24) & 0xFF);
            finalizeEntry(e, raw);
            return e;
        }

        IFDEntry makeSRational(uint16_t tag, int32_t numerator, int32_t denominator) {
            IFDEntry e;
            e.tag = tag;
            e.type = TYPE_SRATIONAL;
            e.count = 1;
            std::vector<unsigned char> raw(8);
            const uint32_t n = static_cast<uint32_t>(numerator);
            const uint32_t d = static_cast<uint32_t>(denominator);
            raw[0] = static_cast<unsigned char>(n & 0xFF);
            raw[1] = static_cast<unsigned char>((n >> 8) & 0xFF);
            raw[2] = static_cast<unsigned char>((n >> 16) & 0xFF);
            raw[3] = static_cast<unsigned char>((n >> 24) & 0xFF);
            raw[4] = static_cast<unsigned char>(d & 0xFF);
            raw[5] = static_cast<unsigned char>((d >> 8) & 0xFF);
            raw[6] = static_cast<unsigned char>((d >> 16) & 0xFF);
            raw[7] = static_cast<unsigned char>((d >> 24) & 0xFF);
            finalizeEntry(e, raw);
            return e;
        }

        IFDEntry makeRational(uint16_t tag, uint32_t numerator, uint32_t denominator) {
            IFDEntry e;
            e.tag = tag;
            e.type = TYPE_RATIONAL;
            e.count = 1;
            std::vector<unsigned char> raw(8);
            raw[0] = static_cast<unsigned char>(numerator & 0xFF);
            raw[1] = static_cast<unsigned char>((numerator >> 8) & 0xFF);
            raw[2] = static_cast<unsigned char>((numerator >> 16) & 0xFF);
            raw[3] = static_cast<unsigned char>((numerator >> 24) & 0xFF);
            raw[4] = static_cast<unsigned char>(denominator & 0xFF);
            raw[5] = static_cast<unsigned char>((denominator >> 8) & 0xFF);
            raw[6] = static_cast<unsigned char>((denominator >> 16) & 0xFF);
            raw[7] = static_cast<unsigned char>((denominator >> 24) & 0xFF);
            finalizeEntry(e, raw);
            return e;
        }

        IFDEntry makeRationalArray(uint16_t tag, const std::vector<std::pair<uint32_t, uint32_t>> &rats) {
            IFDEntry e;
            e.tag = tag;
            e.type = TYPE_RATIONAL;
            e.count = static_cast<uint32_t>(rats.size());
            std::vector<unsigned char> raw(rats.size() * 8);
            for (size_t i = 0; i < rats.size(); ++i) {
                const uint32_t n = rats[i].first;
                const uint32_t d = rats[i].second;
                raw[i * 8 + 0] = static_cast<unsigned char>(n & 0xFF);
                raw[i * 8 + 1] = static_cast<unsigned char>((n >> 8) & 0xFF);
                raw[i * 8 + 2] = static_cast<unsigned char>((n >> 16) & 0xFF);
                raw[i * 8 + 3] = static_cast<unsigned char>((n >> 24) & 0xFF);
                raw[i * 8 + 4] = static_cast<unsigned char>(d & 0xFF);
                raw[i * 8 + 5] = static_cast<unsigned char>((d >> 8) & 0xFF);
                raw[i * 8 + 6] = static_cast<unsigned char>((d >> 16) & 0xFF);
                raw[i * 8 + 7] = static_cast<unsigned char>((d >> 24) & 0xFF);
            }
            finalizeEntry(e, raw);
            return e;
        }

        // Convert decimal degrees (absolute value) to (degrees/1, minutes/1, seconds*1e6/1e6) rationals.
        std::vector<std::pair<uint32_t, uint32_t>> decimalDegToDMSRationals(double deg) {
            double abs_deg = std::fabs(deg);
            uint32_t d = static_cast<uint32_t>(std::floor(abs_deg));
            double rem_min = (abs_deg - static_cast<double>(d)) * 60.0;
            uint32_t m = static_cast<uint32_t>(std::floor(rem_min));
            double rem_sec = (rem_min - static_cast<double>(m)) * 60.0;
            // Encode seconds with microsecond precision.
            constexpr uint32_t SEC_DENOM = 1000000;
            uint32_t s_num = static_cast<uint32_t>(std::round(rem_sec * static_cast<double>(SEC_DENOM)));
            return {{d, 1u}, {m, 1u}, {s_num, SEC_DENOM}};
        }

        // Encode shutter speed as a rational. Prefer 1/N for sub-second exposures.
        std::pair<uint32_t, uint32_t> shutterSpeedRational(float seconds) {
            if (seconds <= 0.f) {
                return {0u, 1u};
            }
            if (seconds < 1.0f) {
                // 1/N form
                double n = std::round(1.0 / static_cast<double>(seconds));
                if (n < 1.0) n = 1.0;
                if (n > 4294967294.0) n = 4294967294.0;
                return {1u, static_cast<uint32_t>(n)};
            }
            // Whole / fractional second form with 1000 denominator.
            return {static_cast<uint32_t>(std::round(seconds * 1000.0)), 1000u};
        }

        // ---- IFD assembly ----
        //
        // Emit an IFD at base_offset within the TIFF stream. Returns the byte offset of the next
        // free location (where the next IFD or trailing data can be placed).
        //
        // `tiff_buf` is the buffer the IFD is being written into; the TIFF header occupies bytes
        // 0..7 of this buffer.
        //
        // `entries` is consumed: pointer placeholders are patched with their target offsets
        // (provided via subifd_offsets keyed by pointer_target_index).
        size_t emitIFD(std::vector<unsigned char> &tiff_buf,
                       size_t base_offset,
                       std::vector<IFDEntry> &entries,
                       uint32_t next_ifd_offset,
                       const std::array<uint32_t, 2> &subifd_offsets) {
            // IFD layout: 2-byte count, N*12-byte entries, 4-byte next-IFD pointer, then the
            // trailing data blob for entries whose value didn't fit inline.
            const size_t n = entries.size();
            const size_t record_bytes = 2 + n * 12 + 4;
            const size_t data_blob_offset = base_offset + record_bytes;

            // Pad buffer to base_offset if needed (the caller has typically already done this).
            if (tiff_buf.size() < base_offset) {
                tiff_buf.resize(base_offset, 0);
            }

            // Compute external-data offsets first.
            std::vector<uint32_t> entry_external_offset(n, 0);
            size_t running = data_blob_offset;
            for (size_t i = 0; i < n; ++i) {
                if (!entries[i].inline_only) {
                    // EXIF/TIFF requires WORD alignment of offsets per the spec; we use 2-byte
                    // alignment for safety with RATIONAL/SHORT arrays.
                    if (running % 2 != 0) {
                        running += 1;
                    }
                    entry_external_offset[i] = static_cast<uint32_t>(running);
                    running += entries[i].external_data.size();
                }
            }

            // Now write the IFD record itself.
            const size_t pre_size = tiff_buf.size();
            (void) pre_size;
            // Resize buffer to hold the record + data blob.
            tiff_buf.resize(running, 0);

            size_t cursor = base_offset;
            // Entry count
            tiff_buf[cursor++] = static_cast<unsigned char>(n & 0xFF);
            tiff_buf[cursor++] = static_cast<unsigned char>((n >> 8) & 0xFF);

            // Entries
            for (size_t i = 0; i < n; ++i) {
                const IFDEntry &e = entries[i];
                // tag (2)
                tiff_buf[cursor + 0] = static_cast<unsigned char>(e.tag & 0xFF);
                tiff_buf[cursor + 1] = static_cast<unsigned char>((e.tag >> 8) & 0xFF);
                // type (2)
                tiff_buf[cursor + 2] = static_cast<unsigned char>(e.type & 0xFF);
                tiff_buf[cursor + 3] = static_cast<unsigned char>((e.type >> 8) & 0xFF);
                // count (4)
                tiff_buf[cursor + 4] = static_cast<unsigned char>(e.count & 0xFF);
                tiff_buf[cursor + 5] = static_cast<unsigned char>((e.count >> 8) & 0xFF);
                tiff_buf[cursor + 6] = static_cast<unsigned char>((e.count >> 16) & 0xFF);
                tiff_buf[cursor + 7] = static_cast<unsigned char>((e.count >> 24) & 0xFF);

                // value / offset (4)
                if (e.is_pointer_placeholder) {
                    const uint32_t target = (e.pointer_target_index >= 0)
                                                    ? subifd_offsets[static_cast<size_t>(e.pointer_target_index)]
                                                    : 0u;
                    patchU32(tiff_buf, cursor + 8, target);
                } else if (e.inline_only) {
                    tiff_buf[cursor + 8] = e.inline_value[0];
                    tiff_buf[cursor + 9] = e.inline_value[1];
                    tiff_buf[cursor + 10] = e.inline_value[2];
                    tiff_buf[cursor + 11] = e.inline_value[3];
                } else {
                    patchU32(tiff_buf, cursor + 8, entry_external_offset[i]);
                }
                cursor += 12;
            }

            // Next IFD offset
            patchU32(tiff_buf, cursor, next_ifd_offset);
            cursor += 4;

            // Trailing data blob
            for (size_t i = 0; i < n; ++i) {
                if (!entries[i].inline_only) {
                    const uint32_t off = entry_external_offset[i];
                    std::memcpy(&tiff_buf[off], entries[i].external_data.data(), entries[i].external_data.size());
                }
            }

            return running;
        }

        // ---- Build the EXIF SubIFD entries from m ----
        std::vector<IFDEntry> buildExifSubIFD(const ImageEXIFData &m) {
            std::vector<IFDEntry> e;
            if (m.exposure_time_s > 0.f) {
                auto r = shutterSpeedRational(m.exposure_time_s);
                e.push_back(makeRational(0x829A, r.first, r.second));
            }
            if (m.f_number > 0.f) {
                // FNumber: RATIONAL, 1/100 precision.
                uint32_t num = static_cast<uint32_t>(std::round(static_cast<double>(m.f_number) * 100.0));
                e.push_back(makeRational(0x829D, num, 100u));
            }
            if (!m.datetime_original.empty()) {
                e.push_back(makeAscii(0x9003, m.datetime_original));
            }
            if (!m.datetime_digitized.empty()) {
                e.push_back(makeAscii(0x9004, m.datetime_digitized));
            }
            if (!m.offset_time.empty()) {
                // OffsetTime / OffsetTimeOriginal / OffsetTimeDigitized share the same value here.
                e.push_back(makeAscii(0x9010, m.offset_time));
                e.push_back(makeAscii(0x9011, m.offset_time));
                e.push_back(makeAscii(0x9012, m.offset_time));
            }
            if (m.iso > 0) {
                e.push_back(makeShort(0x8827, static_cast<uint16_t>(std::min<unsigned int>(m.iso, 65535u))));
            }
            if (m.has_exposure_bias) {
                // ExposureBiasValue: SRATIONAL, 1/100 EV precision.
                int32_t num = static_cast<int32_t>(std::round(static_cast<double>(m.exposure_bias_ev) * 100.0));
                e.push_back(makeSRational(0x9204, num, 100));
            }
            if (m.max_aperture_value_apex > 0.f) {
                // MaxApertureValue: RATIONAL (APEX units), 1/100 precision.
                uint32_t num = static_cast<uint32_t>(std::round(static_cast<double>(m.max_aperture_value_apex) * 100.0));
                e.push_back(makeRational(0x9205, num, 100u));
            }
            if (m.subject_distance_m > 0.f) {
                // SubjectDistance: RATIONAL, mm precision.
                uint32_t num = static_cast<uint32_t>(std::round(static_cast<double>(m.subject_distance_m) * 1000.0));
                e.push_back(makeRational(0x9206, num, 1000u));
            }
            if (m.focal_length_mm > 0.f) {
                // 1/1000 mm precision.
                uint32_t num = static_cast<uint32_t>(std::round(static_cast<double>(m.focal_length_mm) * 1000.0));
                e.push_back(makeRational(0x920A, num, 1000u));
            }
            if (m.pixel_x_dimension > 0) {
                e.push_back(makeLong(0xA002, m.pixel_x_dimension));
            }
            if (m.pixel_y_dimension > 0) {
                e.push_back(makeLong(0xA003, m.pixel_y_dimension));
            }
            if (m.focal_plane_x_resolution > 0.f) {
                uint32_t num = static_cast<uint32_t>(std::round(static_cast<double>(m.focal_plane_x_resolution) * 1000.0));
                e.push_back(makeRational(0xA20E, num, 1000u));
            }
            if (m.focal_plane_y_resolution > 0.f) {
                uint32_t num = static_cast<uint32_t>(std::round(static_cast<double>(m.focal_plane_y_resolution) * 1000.0));
                e.push_back(makeRational(0xA20F, num, 1000u));
            }
            if (m.focal_plane_x_resolution > 0.f || m.focal_plane_y_resolution > 0.f) {
                e.push_back(makeShort(0xA210, static_cast<uint16_t>(m.focal_plane_resolution_unit)));
            }
            if (m.exposure_mode != (unsigned int) -1) {
                e.push_back(makeShort(0xA402, static_cast<uint16_t>(m.exposure_mode)));
            }
            if (m.white_balance != (unsigned int) -1) {
                e.push_back(makeShort(0xA403, static_cast<uint16_t>(m.white_balance)));
            }
            if (m.digital_zoom_ratio > 0.f) {
                // DigitalZoomRatio: RATIONAL, 1/100 precision.
                uint32_t num = static_cast<uint32_t>(std::round(static_cast<double>(m.digital_zoom_ratio) * 100.0));
                e.push_back(makeRational(0xA404, num, 100u));
            }
            if (m.focal_length_in_35mm > 0) {
                e.push_back(makeShort(0xA405, static_cast<uint16_t>(std::min<unsigned int>(m.focal_length_in_35mm, 65535u))));
            }
            if (!m.lens_make.empty()) {
                e.push_back(makeAscii(0xA433, m.lens_make));
            }
            if (!m.lens_model.empty()) {
                e.push_back(makeAscii(0xA434, m.lens_model));
            }
            if (!m.lens_specification.empty()) {
                // LensSpecification (0xA432) is officially 4 RATIONALs (min focal, max focal, min F, max F).
                // We don't have those decomposed; emit as ASCII LensSpecificationDescription instead.
                e.push_back(makeAscii(0xA435, m.lens_specification));
            }

            // ExifVersion (UNDEFINED 4) = "0232"
            e.push_back(makeUndefined(0x9000, "0232"));

            // IFD entries must be sorted by tag.
            std::sort(e.begin(), e.end(), [](const IFDEntry &a, const IFDEntry &b) { return a.tag < b.tag; });
            return e;
        }

        // ---- Build GPS IFD entries from m ----
        std::vector<IFDEntry> buildGPSIFD(const ImageEXIFData &m) {
            std::vector<IFDEntry> e;

            // GPSVersionID (BYTE x4) = 2,3,0,0
            {
                IFDEntry v;
                v.tag = 0x0000;
                v.type = TYPE_BYTE;
                v.count = 4;
                std::vector<unsigned char> raw = {2, 3, 0, 0};
                finalizeEntry(v, raw);
                e.push_back(v);
            }
            // GPSLatitudeRef
            e.push_back(makeAscii(0x0001, (m.latitude_deg >= 0.0) ? "N" : "S"));
            // GPSLatitude
            e.push_back(makeRationalArray(0x0002, decimalDegToDMSRationals(m.latitude_deg)));
            // GPSLongitudeRef
            e.push_back(makeAscii(0x0003, (m.longitude_deg >= 0.0) ? "E" : "W"));
            // GPSLongitude
            e.push_back(makeRationalArray(0x0004, decimalDegToDMSRationals(m.longitude_deg)));
            // GPSAltitudeRef (BYTE: 0 = above sea level, 1 = below)
            {
                IFDEntry v;
                v.tag = 0x0005;
                v.type = TYPE_BYTE;
                v.count = 1;
                std::vector<unsigned char> raw(1);
                raw[0] = (m.altitude_m >= 0.0) ? 0 : 1;
                finalizeEntry(v, raw);
                e.push_back(v);
            }
            // GPSAltitude (always non-negative; sign carried by AltitudeRef)
            {
                const double abs_alt = std::fabs(m.altitude_m);
                const uint32_t num = static_cast<uint32_t>(std::round(abs_alt * 1000.0));
                e.push_back(makeRational(0x0006, num, 1000u));
            }
            // GPSTimeStamp (RATIONAL x3: H, M, S)
            if (!m.gps_timestamp_hms.empty()) {
                int hh = 0, mm = 0, ss = 0;
                if (std::sscanf(m.gps_timestamp_hms.c_str(), "%d:%d:%d", &hh, &mm, &ss) == 3) {
                    std::vector<std::pair<uint32_t, uint32_t>> ts = {
                            {static_cast<uint32_t>(hh), 1u},
                            {static_cast<uint32_t>(mm), 1u},
                            {static_cast<uint32_t>(ss), 1u}};
                    e.push_back(makeRationalArray(0x0007, ts));
                }
            }
            // GPSImgDirectionRef = 'T' (true north)
            e.push_back(makeAscii(0x0010, "T"));
            // GPSImgDirection (RATIONAL, 1/100 deg precision)
            {
                double dir = m.img_direction_deg;
                if (dir < 0.0) dir += 360.0;
                if (dir >= 360.0) dir -= 360.0;
                uint32_t num = static_cast<uint32_t>(std::round(dir * 100.0));
                e.push_back(makeRational(0x0011, num, 100u));
            }
            // GPSDateStamp (ASCII, 11 bytes: "YYYY:MM:DD\0")
            if (!m.gps_datestamp.empty()) {
                e.push_back(makeAscii(0x001D, m.gps_datestamp));
            }

            std::sort(e.begin(), e.end(), [](const IFDEntry &a, const IFDEntry &b) { return a.tag < b.tag; });
            return e;
        }

    } // namespace

    std::vector<unsigned char> buildEXIFAppSegment(const ImageEXIFData &m) {
        // Payload layout:
        //   "Exif\0\0"                                              (6 bytes)
        //   TIFF header: "II" 0x002A 0x00000008                     (8 bytes)
        //   IFD0 (record + data blob)
        //   ExifSubIFD (record + data blob)   -- pointed to by IFD0 tag 0x8769
        //   GPS IFD    (record + data blob)   -- pointed to by IFD0 tag 0x8825   (if m.gps_valid)

        // ---- Build IFD0 entries ----
        std::vector<IFDEntry> ifd0;
        if (!m.make.empty()) ifd0.push_back(makeAscii(0x010F, m.make));
        if (!m.model.empty()) ifd0.push_back(makeAscii(0x0110, m.model));
        ifd0.push_back(makeShort(0x0112, static_cast<uint16_t>(m.orientation)));
        if (!m.software.empty()) ifd0.push_back(makeAscii(0x0131, m.software));
        if (!m.datetime.empty()) ifd0.push_back(makeAscii(0x0132, m.datetime));

        // ExifSubIFD pointer (LONG, tag 0x8769) -- patched later.
        IFDEntry exif_ptr;
        exif_ptr.tag = 0x8769;
        exif_ptr.type = TYPE_LONG;
        exif_ptr.count = 1;
        exif_ptr.is_pointer_placeholder = true;
        exif_ptr.pointer_target_index = 0;
        exif_ptr.inline_only = true;
        ifd0.push_back(exif_ptr);

        // GPS IFD pointer (LONG, tag 0x8825) -- patched later, only if GPS valid.
        if (m.gps_valid) {
            IFDEntry gps_ptr;
            gps_ptr.tag = 0x8825;
            gps_ptr.type = TYPE_LONG;
            gps_ptr.count = 1;
            gps_ptr.is_pointer_placeholder = true;
            gps_ptr.pointer_target_index = 1;
            gps_ptr.inline_only = true;
            ifd0.push_back(gps_ptr);
        }

        std::sort(ifd0.begin(), ifd0.end(), [](const IFDEntry &a, const IFDEntry &b) { return a.tag < b.tag; });

        std::vector<IFDEntry> exif_sub = buildExifSubIFD(m);
        std::vector<IFDEntry> gps_ifd;
        if (m.gps_valid) {
            gps_ifd = buildGPSIFD(m);
        }

        // ---- Two-pass layout: compute offsets, then emit ----
        //
        // TIFF stream begins right after "Exif\0\0". Inside the TIFF stream:
        //   bytes 0..7  : TIFF header (II 002A 00000008)
        //   byte 8       : IFD0 record start
        //   then        : IFD0 data blob
        //   then        : ExifSubIFD record + data blob
        //   then        : GPS IFD record + data blob

        std::vector<unsigned char> tiff;
        tiff.reserve(1024);
        // TIFF header
        tiff.push_back('I');
        tiff.push_back('I');
        putU16(tiff, 0x002A);
        putU32(tiff, 8); // offset to IFD0

        // Compute IFD0 size to know where ExifSubIFD will start.
        auto ifdRecordSize = [](const std::vector<IFDEntry> &v) {
            size_t s = 2 + v.size() * 12 + 4;
            for (const auto &e: v) {
                if (!e.inline_only) {
                    if (s % 2 != 0) s += 1;
                    s += e.external_data.size();
                }
            }
            return s;
        };

        const size_t ifd0_base = 8;
        const size_t ifd0_size = ifdRecordSize(ifd0);
        const size_t exif_base = ifd0_base + ifd0_size;
        const size_t exif_size = ifdRecordSize(exif_sub);
        const size_t gps_base = exif_base + exif_size;

        std::array<uint32_t, 2> subifd_offsets = {static_cast<uint32_t>(exif_base),
                                                  m.gps_valid ? static_cast<uint32_t>(gps_base) : 0u};

        // Emit IFD0
        emitIFD(tiff, ifd0_base, ifd0, /*next_ifd=*/0u, subifd_offsets);
        // Emit ExifSubIFD
        emitIFD(tiff, exif_base, exif_sub, /*next_ifd=*/0u, subifd_offsets);
        // Emit GPS IFD
        if (m.gps_valid) {
            emitIFD(tiff, gps_base, gps_ifd, /*next_ifd=*/0u, subifd_offsets);
        }

        // Compose final payload: "Exif\0\0" + tiff
        std::vector<unsigned char> payload;
        payload.reserve(6 + tiff.size());
        const unsigned char header[] = {'E', 'x', 'i', 'f', 0, 0};
        payload.insert(payload.end(), header, header + 6);
        payload.insert(payload.end(), tiff.begin(), tiff.end());

        if (payload.size() > APP1_MAX_PAYLOAD) {
            helios_runtime_error("ERROR (buildEXIFAppSegment): EXIF payload (" + std::to_string(payload.size()) +
                                 " bytes) exceeds JPEG APP1 size limit (" + std::to_string(APP1_MAX_PAYLOAD) + ").");
        }
        return payload;
    }

    std::vector<unsigned char> buildXMPAppSegment(const ImageEXIFData &m) {
        // RDF/XML packet using pugixml.
        pugi::xml_document doc;

        pugi::xml_node xpacket_begin = doc.append_child(pugi::node_pi);
        xpacket_begin.set_name("xpacket");
        xpacket_begin.set_value("begin=\"\xEF\xBB\xBF\" id=\"W5M0MpCehiHzreSzNTczkc9d\"");

        pugi::xml_node x_xmpmeta = doc.append_child("x:xmpmeta");
        x_xmpmeta.append_attribute("xmlns:x") = "adobe:ns:meta/";
        x_xmpmeta.append_attribute("x:xmptk") = "Helios";

        pugi::xml_node rdf = x_xmpmeta.append_child("rdf:RDF");
        rdf.append_attribute("xmlns:rdf") = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

        pugi::xml_node desc = rdf.append_child("rdf:Description");
        desc.append_attribute("rdf:about") = "";
        desc.append_attribute("xmlns:tiff") = "http://ns.adobe.com/tiff/1.0/";
        desc.append_attribute("xmlns:xmp") = "http://ns.adobe.com/xap/1.0/";
        desc.append_attribute("xmlns:Camera") = "http://pix4d.com/camera/1.0";

        if (!m.make.empty()) desc.append_attribute("tiff:Make") = m.make.c_str();
        if (!m.model.empty()) desc.append_attribute("tiff:Model") = m.model.c_str();
        if (!m.software.empty()) desc.append_attribute("xmp:CreatorTool") = m.software.c_str();

        auto fmt = [](double v) {
            std::ostringstream oss;
            oss.precision(6);
            oss << std::fixed << v;
            return oss.str();
        };

        if (m.xmp_valid) {
            desc.append_attribute("Camera:Yaw") = fmt(m.yaw_deg).c_str();
            desc.append_attribute("Camera:Pitch") = fmt(m.pitch_deg).c_str();
            desc.append_attribute("Camera:Roll") = fmt(m.roll_deg).c_str();
        }

        pugi::xml_node xpacket_end = doc.append_child(pugi::node_pi);
        xpacket_end.set_name("xpacket");
        xpacket_end.set_value("end=\"w\"");

        std::ostringstream xml_oss;
        doc.save(xml_oss, "", pugi::format_raw | pugi::format_no_declaration);
        const std::string xml = xml_oss.str();

        // Assemble payload: "http://ns.adobe.com/xap/1.0/\0" + RDF/XML
        static const char NS_ID[] = "http://ns.adobe.com/xap/1.0/";
        const size_t ns_len = std::strlen(NS_ID) + 1; // include NUL
        std::vector<unsigned char> payload;
        payload.reserve(ns_len + xml.size());
        payload.insert(payload.end(), NS_ID, NS_ID + ns_len);
        payload.insert(payload.end(), xml.begin(), xml.end());

        if (payload.size() > APP1_MAX_PAYLOAD) {
            helios_runtime_error("ERROR (buildXMPAppSegment): XMP payload (" + std::to_string(payload.size()) +
                                 " bytes) exceeds JPEG APP1 size limit (" + std::to_string(APP1_MAX_PAYLOAD) + ").");
        }
        return payload;
    }

} // namespace helios::detail
