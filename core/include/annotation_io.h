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

namespace helios {
    class Context;
}

//! Writing of image annotation files in the standard formats used to train machine-learning models
/**
 * These routines are deliberately independent of how the image was produced. Every entry point
 * takes plain pixel data -- binary masks, bounding boxes -- rather than a camera or a renderer, so
 * that any plug-in that can produce a per-pixel object label can write annotations in the same
 * formats. The radiation plug-in supplies its masks from ray-traced camera pixel labels; the
 * visualizer and synthetic annotation plug-ins supply them from a rasterized ID rendering.
 *
 * The routines that take a "pixel-to-primitive map" turn a per-pixel record of which primitive is
 * visible, together with labels stored as primitive or object data in the Context, into finished
 * annotation files. Two renderers that produce the same map for the same labeled Context therefore
 * write identical annotations. The map is a row-major vector of length width*height in the image
 * convention below, in which each element is the UUID of the primitive visible in that pixel plus
 * one, and zero means no primitive is visible.
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

    //! Compute a bounding box for each labeled object visible in a pixel-to-primitive map
    /**
     * A label is a primitive or object data value of type `uint` or `int`: every primitive carrying the same value of the same data label belongs to one object, and one box is produced per distinct value. Primitives
     * that do not carry the data label are not annotated.
     *
     * A box encloses whole every pixel in which the object is visible: it runs from the left edge of the first pixel column to the right edge of the last, and from the top edge of the first pixel row to the bottom
     * edge of the last. An object visible in a single pixel therefore has a box one pixel in size.
     *
     * \param[in] context Context holding the primitives named by the map and their label data.
     * \param[in] pixel_UUIDs Pixel-to-primitive map (UUID+1 per pixel, 0 for none), row-major with row 0 at the top of the image.
     * \param[in] resolution Image dimensions in pixels.
     * \param[in] data_labels Names of the primitive or object data labels to annotate.
     * \param[in] class_IDs Class index written for the objects of each data label, in the same order as data_labels.
     * \param[in] use_object_data If true, labels are read from the data of each primitive's parent object. If false, they are read from primitive data.
     * \return One box per distinct pair of class index and label value, ordered by class index and then label value.
     */
    [[nodiscard]] std::vector<YOLOBox> labelBoundingBoxes(const helios::Context *context, const std::vector<uint> &pixel_UUIDs, const helios::int2 &resolution, const std::vector<std::string> &data_labels, const std::vector<uint> &class_IDs,
                                                          bool use_object_data);

    //! Write YOLO bounding boxes and a class file for the labeled objects visible in a pixel-to-primitive map
    /**
     * The boxes are written to a file named after the image, with its extension replaced by ".txt", in the directory `image_path`. The class file holds one "class_ID data_label" pair per line.
     *
     * \param[in] context Context holding the primitives named by the map and their label data.
     * \param[in] pixel_UUIDs Pixel-to-primitive map (UUID+1 per pixel, 0 for none), row-major with row 0 at the top of the image.
     * \param[in] resolution Image dimensions in pixels.
     * \param[in] data_labels Names of the primitive or object data labels to annotate.
     * \param[in] class_IDs Class index written for the objects of each data label, in the same order as data_labels.
     * \param[in] image_file Name of the image these boxes annotate. Only its stem is used, to name the output file.
     * \param[in] classes_txt_file Name of the class file to write within `image_path`.
     * \param[in] image_path Directory to write the output files to.
     * \param[in] use_object_data If true, labels are read from the data of each primitive's parent object. If false, they are read from primitive data.
     * \param[in] caller Name of the calling method, used to attribute error messages (e.g. "Visualizer::writeImageBoundingBoxes").
     */
    void writeLabelBoundingBoxes(const helios::Context *context, const std::vector<uint> &pixel_UUIDs, const helios::int2 &resolution, const std::vector<std::string> &data_labels, const std::vector<uint> &class_IDs, const std::string &image_file,
                                 const std::string &classes_txt_file, const std::string &image_path, bool use_object_data, const std::string &caller);

    //! Build a binary mask for each distinct value of a data label visible in a pixel-to-primitive map
    /**
     * \param[in] context Context holding the primitives named by the map and their label data.
     * \param[in] pixel_UUIDs Pixel-to-primitive map (UUID+1 per pixel, 0 for none), row-major with row 0 at the top of the image.
     * \param[in] resolution Image dimensions in pixels.
     * \param[in] data_label Name of the primitive or object data label, which must have type `uint` or `int`.
     * \param[in] use_object_data If true, the label is read from the data of each primitive's parent object. If false, it is read from primitive data.
     * \return Binary mask per label value, indexed [row][column] with row 0 at the top of the image.
     */
    [[nodiscard]] std::map<int, std::vector<std::vector<bool>>> labelMasks(const helios::Context *context, const std::vector<uint> &pixel_UUIDs, const helios::int2 &resolution, const std::string &data_label, bool use_object_data);

    //! Write COCO JSON segmentation masks for the labeled objects visible in a pixel-to-primitive map
    /**
     * Each connected region of pixels sharing one value of a data label becomes one annotation, so an object split in two by an occluder yields two annotations. Pixels touching only at a corner are connected.
     * The "bbox" of an annotation is (x, y, width, height) in pixels, enclosing whole every pixel of the region.
     * A region of only one or two pixels is too small to outline and is not written.
     *
     * \param[in] context Context holding the primitives named by the map and their label data.
     * \param[in] pixel_UUIDs Pixel-to-primitive map (UUID+1 per pixel, 0 for none), row-major with row 0 at the top of the image.
     * \param[in] resolution Image dimensions in pixels.
     * \param[in] data_labels Names of the primitive or object data labels to annotate. Each becomes a COCO category.
     * \param[in] class_IDs Class index (COCO category ID) of each data label, in the same order as data_labels.
     * \param[in] json_filename Path of the COCO JSON file to write. ".json" is appended if it is missing.
     * \param[in] image_file Path of the image these masks annotate, which must exist.
     * \param[in] data_attribute_labels Primitive or object data labels of type `int`, `uint`, `float` or `double` whose mean over the pixels of each mask is written to that annotation's "attributes". A label that
     * exists as both is read from primitive data. Labels that do not exist are ignored.
     * \param[in] append_file If true, annotations are added to the file at `json_filename` if it already exists. If false, a new file is written.
     * \param[in] use_object_data If true, labels are read from the data of each primitive's parent object. If false, they are read from primitive data.
     * \param[in] caller Name of the calling method, used to attribute error messages (e.g. "Visualizer::writeImageSegmentationMasks").
     */
    void writeLabelSegmentationMasks(const helios::Context *context, const std::vector<uint> &pixel_UUIDs, const helios::int2 &resolution, const std::vector<std::string> &data_labels, const std::vector<uint> &class_IDs,
                                     const std::string &json_filename, const std::string &image_file, const std::vector<std::string> &data_attribute_labels, bool append_file, bool use_object_data, const std::string &caller);

} // namespace helios::annotation

#endif
