/** \file "annotation_io.h" Writing of machine-learning image annotation files (YOLO and COCO).

    Copyright (C) 2016-2026 Brian Bailey

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    SPDX-License-Identifier: LGPL-2.1-or-later

*/

#ifndef HELIOS_ANNOTATION_IO
#define HELIOS_ANNOTATION_IO

#include "helios_vector_types.h"
#include "json.hpp"
#include <map>
#include <string>
#include <utility>
#include <vector>

//! Writing of image annotation files in the standard formats used to train machine-learning models
/**
 * These routines are deliberately independent of how the image was produced. Every entry point
 * takes plain pixel data -- binary masks, bounding boxes -- rather than a camera or a renderer, so
 * that any plug-in that can produce a per-pixel object label can write annotations in the same
 * formats. The radiation plug-in supplies its masks from ray-traced camera pixel labels; the
 * synthetic annotation plug-in supplies them from a rasterized ID rendering.
 *
 * The image coordinate convention throughout is the one the annotation formats require: the origin
 * is the TOP-LEFT corner of the image, with y increasing downward. Callers whose pixel buffer is
 * stored in a different orientation must apply the flip when building their masks, not here.
 */
namespace helios::annotation {

    //! A single bounding box in the normalized form used by the YOLO annotation format
    struct YOLOBox {
        //! Zero-based class index of the object
        uint class_ID = 0;
        //! Center of the box, normalized to [0,1] by the image dimensions, origin at the top-left
        helios::vec2 center;
        //! Width and height of the box, normalized to [0,1] by the image dimensions
        helios::vec2 size;
    };

    //! Find a pixel of a binary mask that lies on the region boundary, to start a boundary trace from
    /**
     * \param[in] mask Binary mask, indexed [row][column].
     * \param[in] resolution Image dimensions in pixels.
     * \return Coordinates of a boundary pixel, or (-1,-1) if the mask holds no region.
     */
    [[nodiscard]] std::pair<int, int> findStartingBoundaryPixel(const std::vector<std::vector<bool>> &mask, const helios::int2 &resolution);

    //! Isolate one connected component into its own full-resolution mask
    /**
     * Tracing against this isolated mask rather than the whole label mask keeps the trace from
     * wandering into a different component that happens to touch this one diagonally.
     *
     * \param[in] component_pixels Pixels making up the component.
     * \param[in] resolution Image dimensions in pixels.
     * \return Mask holding only the given component.
     */
    [[nodiscard]] std::vector<std::vector<bool>> buildComponentMask(const std::vector<std::pair<int, int>> &component_pixels, const helios::int2 &resolution);

    //! Trace a region outline using the Moore neighborhood algorithm
    /**
     * \param[in] mask Binary mask holding the region to trace, indexed [row][column].
     * \param[in] start_x Column of the boundary pixel to start from.
     * \param[in] start_y Row of the boundary pixel to start from.
     * \param[in] resolution Image dimensions in pixels.
     * \return Ordered outline of the region.
     */
    [[nodiscard]] std::vector<std::pair<int, int>> traceBoundaryMoore(const std::vector<std::vector<bool>> &mask, int start_x, int start_y, const helios::int2 &resolution);

    //! Collect the boundary pixels of a region by breadth-first search
    /**
     * Used as a fallback when the Moore trace returns too few points to form a usable outline. The
     * points come back in queue order rather than as an ordered walk around the region.
     *
     * \param[in] mask Binary mask holding the region, indexed [row][column].
     * \param[in] start_x Column of the boundary pixel to start from.
     * \param[in] start_y Row of the boundary pixel to start from.
     * \param[in] resolution Image dimensions in pixels.
     * \return Boundary pixels of the region.
     */
    [[nodiscard]] std::vector<std::pair<int, int>> traceBoundarySimple(const std::vector<std::vector<bool>> &mask, int start_x, int start_y, const helios::int2 &resolution);

    //! Trace the outline of every connected region in a set of binary masks and convert them to COCO annotations
    /**
     * Each mask is scanned for 8-connected components, and the outline of each component is traced
     * to a polygon. A component that yields fewer than three boundary points is discarded, since it
     * cannot form a polygon.
     *
     * \param[in] label_masks Binary mask per label value. Each mask is indexed [row][column] with row 0 at the top of the image.
     * \param[in] object_class_ID Class index recorded on every annotation produced.
     * \param[in] resolution Image dimensions in pixels.
     * \param[in] image_id Identifier of the image these annotations belong to, matching an entry in the COCO "images" array.
     * \return One annotation per connected component, with keys "id", "image_id", "category_id", "bbox", "area", "iscrowd" and "segmentation".
     */
    [[nodiscard]] std::vector<std::map<std::string, std::vector<float>>> maskToAnnotations(const std::map<int, std::vector<std::vector<bool>>> &label_masks, uint object_class_ID, const helios::int2 &resolution, int image_id);

    //! Load an existing COCO JSON file or create a new one, and get the image ID to annotate against
    /**
     * If the image is already present in the file its existing ID is returned, so that repeated
     * calls for the same image accumulate annotations rather than duplicating the image entry.
     *
     * \param[in] filename Path of the COCO JSON file.
     * \param[in] append If true, an existing file at this path is loaded and added to. If false, a new document is started.
     * \param[in] resolution Image dimensions in pixels.
     * \param[in] image_file Path of the image being annotated. Only its file name is recorded.
     * \return The COCO document and the image ID to use for annotations of this image.
     */
    [[nodiscard]] std::pair<nlohmann::json, int> initializeCOCOJson(const std::string &filename, bool append, const helios::int2 &resolution, const std::string &image_file);

    //! Add class definitions to a COCO document, ignoring any that are already defined
    /**
     * \param[inout] coco_json COCO document to add the categories to.
     * \param[in] class_IDs Class index of each category.
     * \param[in] class_names Name of each category, in the same order as class_IDs.
     */
    void addCOCOCategory(nlohmann::json &coco_json, const std::vector<uint> &class_IDs, const std::vector<std::string> &class_names);

    //! Write a COCO document to file
    /**
     * \param[in] coco_json COCO document to write.
     * \param[in] filename Path of the file to write.
     */
    void writeCOCOJson(const nlohmann::json &coco_json, const std::string &filename);

    //! Write bounding boxes to a file in the YOLO annotation format
    /**
     * Each box is written as "class_ID x_center y_center width height", with the four geometric
     * values normalized to [0,1]. Boxes of zero width or height are skipped, since they describe no
     * region and are rejected by readers.
     *
     * \param[in] boxes Bounding boxes to write.
     * \param[in] filename Path of the file to write.
     */
    void writeYOLOBoxes(const std::vector<YOLOBox> &boxes, const std::string &filename);

    //! Write a class name file listing one class name per line, in class index order
    /**
     * This is the form expected by most detection training pipelines, in which a class's index is
     * its line number counting from zero. The class indices supplied must therefore be contiguous
     * and start at zero.
     *
     * \param[in] class_names Class name of each class index.
     * \param[in] filename Path of the file to write.
     */
    void writeYOLOClassNames(const std::map<uint, std::string> &class_names, const std::string &filename);

} // namespace helios::annotation

#endif
