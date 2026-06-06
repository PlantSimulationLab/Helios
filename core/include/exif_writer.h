/**
 * \file "exif_writer.h" Hand-rolled EXIF (TIFF/APP1) and XMP (APP1) segment builders for JPEG embedding.
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

#ifndef HELIOS_EXIF_WRITER_H
#define HELIOS_EXIF_WRITER_H

#include <vector>

#include "image_metadata.h"

namespace helios::detail {

    //! Build the payload of a JPEG APP1 EXIF segment (excludes the 0xFFE1 marker and 2-byte length, which libjpeg writes).
    /**
     * The returned buffer begins with the EXIF identifier "Exif\0\0" followed by a little-endian TIFF
     * header, IFD0, ExifSubIFD, and (when m.gps_valid) GPS IFD. Tags whose corresponding fields in
     * `m` are unset are omitted.
     *
     * \param[in] m Populated metadata struct.
     * \return Byte payload suitable for `jpeg_write_marker(&cinfo, JPEG_APP0+1, data, size)`. Throws via
     *         `helios_runtime_error` if the resulting payload would exceed the JPEG APP1 size limit (65533).
     */
    std::vector<unsigned char> buildEXIFAppSegment(const ImageEXIFData &m);

    //! Build the payload of a JPEG APP1 XMP segment (excludes the 0xFFE1 marker and 2-byte length).
    /**
     * The returned buffer begins with the XMP namespace identifier "http://ns.adobe.com/xap/1.0/\0"
     * followed by an RDF/XML packet declaring `tiff:` and Pix4D `Camera:` namespaces with Yaw / Pitch / Roll
     * values from `m`.
     *
     * \param[in] m Populated metadata struct (must have xmp_valid==true to produce a non-empty payload).
     * \return Byte payload suitable for `jpeg_write_marker(&cinfo, JPEG_APP0+1, data, size)`. Throws via
     *         `helios_runtime_error` if the payload exceeds the JPEG APP1 size limit (65533).
     */
    std::vector<unsigned char> buildXMPAppSegment(const ImageEXIFData &m);

}

#endif
