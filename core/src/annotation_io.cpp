/** \file "annotation_io.cpp" Writing of machine-learning image annotation files (YOLO and COCO).

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

#include "annotation_io.h"
#include "Context.h"
#include "global.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <queue>
#include <set>
#include <stack>

using namespace helios;

namespace {

    //! Read the integer label that a primitive carries under one data label
    /**
     * \param[in] context Context holding the primitive.
     * \param[in] UUID Primitive to read the label of.
     * \param[in] data_label Name of the primitive or object data label.
     * \param[in] use_object_data If true, the label is read from the primitive's parent object. If false, it is read from the primitive.
     * \param[out] label_value Value of the label, if the primitive carries one.
     * \return True if the primitive (or its parent object) has this data label with type `uint` or `int`.
     */
    bool readLabelValue(const Context *context, uint UUID, const std::string &data_label, bool use_object_data, uint &label_value) {
        if (use_object_data) {
            const uint objID = context->getPrimitiveParentObjectID(UUID);
            if (objID == 0 || !context->doesObjectExist(objID) || !context->doesObjectDataExist(objID, data_label.c_str())) {
                return false;
            }
            const HeliosDataType datatype = context->getObjectDataType(data_label.c_str());
            if (datatype == HELIOS_TYPE_UINT) {
                context->getObjectData(objID, data_label.c_str(), label_value);
                return true;
            } else if (datatype == HELIOS_TYPE_INT) {
                int label_value_int;
                context->getObjectData(objID, data_label.c_str(), label_value_int);
                label_value = (uint) label_value_int;
                return true;
            }
            return false;
        }

        if (!context->doesPrimitiveDataExist(UUID, data_label.c_str())) {
            return false;
        }
        const HeliosDataType datatype = context->getPrimitiveDataType(data_label.c_str());
        if (datatype == HELIOS_TYPE_UINT) {
            context->getPrimitiveData(UUID, data_label.c_str(), label_value);
            return true;
        } else if (datatype == HELIOS_TYPE_INT) {
            int label_value_int;
            context->getPrimitiveData(UUID, data_label.c_str(), label_value_int);
            label_value = (uint) label_value_int;
            return true;
        }
        return false;
    }

    //! Read a numeric primitive or object data value of a primitive as a double
    /**
     * \param[in] context Context holding the primitive.
     * \param[in] UUID Primitive to read the value of.
     * \param[in] data_label Name of the primitive or object data label.
     * \param[in] is_primitive_data If true, the value is read from the primitive. If false, it is read from the primitive's parent object.
     * \param[out] value Value of the data, if the primitive carries it.
     * \return True if the primitive (or its parent object) has this data label with type `int`, `uint`, `float` or `double`.
     */
    bool readNumericData(const Context *context, uint UUID, const std::string &data_label, bool is_primitive_data, double &value) {
        if (is_primitive_data) {
            if (!context->doesPrimitiveDataExist(UUID, data_label.c_str())) {
                return false;
            }
            const HeliosDataType datatype = context->getPrimitiveDataType(data_label.c_str());
            if (datatype == HELIOS_TYPE_INT) {
                int val;
                context->getPrimitiveData(UUID, data_label.c_str(), val);
                value = static_cast<double>(val);
                return true;
            } else if (datatype == HELIOS_TYPE_UINT) {
                uint val;
                context->getPrimitiveData(UUID, data_label.c_str(), val);
                value = static_cast<double>(val);
                return true;
            } else if (datatype == HELIOS_TYPE_FLOAT) {
                float val;
                context->getPrimitiveData(UUID, data_label.c_str(), val);
                value = static_cast<double>(val);
                return true;
            } else if (datatype == HELIOS_TYPE_DOUBLE) {
                context->getPrimitiveData(UUID, data_label.c_str(), value);
                return true;
            }
            return false;
        }

        const uint objID = context->getPrimitiveParentObjectID(UUID);
        if (objID == 0 || !context->doesObjectDataExist(objID, data_label.c_str())) {
            return false;
        }
        const HeliosDataType datatype = context->getObjectDataType(data_label.c_str());
        if (datatype == HELIOS_TYPE_INT) {
            int val;
            context->getObjectData(objID, data_label.c_str(), val);
            value = static_cast<double>(val);
            return true;
        } else if (datatype == HELIOS_TYPE_UINT) {
            uint val;
            context->getObjectData(objID, data_label.c_str(), val);
            value = static_cast<double>(val);
            return true;
        } else if (datatype == HELIOS_TYPE_FLOAT) {
            float val;
            context->getObjectData(objID, data_label.c_str(), val);
            value = static_cast<double>(val);
            return true;
        } else if (datatype == HELIOS_TYPE_DOUBLE) {
            context->getObjectData(objID, data_label.c_str(), value);
            return true;
        }
        return false;
    }

    //! Check that a pixel-to-primitive map has one element per pixel of the image it describes
    void validatePixelMapSize(const std::vector<uint> &pixel_UUIDs, const helios::int2 &resolution, const std::string &caller) {
        if (resolution.x <= 0 || resolution.y <= 0 || pixel_UUIDs.size() != size_t(resolution.x) * size_t(resolution.y)) {
            helios_runtime_error("ERROR (" + caller + "): The pixel-to-primitive map has " + std::to_string(pixel_UUIDs.size()) + " elements, which does not match the image resolution of " + std::to_string(resolution.x) + " x " +
                                 std::to_string(resolution.y) + " pixels.");
        }
    }

} // namespace

std::pair<int, int> helios::annotation::findStartingBoundaryPixel(const std::vector<std::vector<bool>> &mask, const helios::int2 &resolution) {
    for (int j = 0; j < resolution.y; j++) {
        for (int i = 0; i < resolution.x; i++) {
            if (mask[j][i]) {
                // Check if this pixel is on the boundary
                for (int di = -1; di <= 1; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        if (di == 0 && dj == 0)
                            continue;
                        int ni = i + di;
                        int nj = j + dj;
                        if (ni < 0 || ni >= resolution.x || nj < 0 || nj >= resolution.y || !mask[nj][ni]) {
                            return {i, j}; // Found boundary pixel
                        }
                    }
                }
            }
        }
    }
    return {-1, -1}; // No boundary found
}

std::vector<std::vector<bool>> helios::annotation::buildComponentMask(const std::vector<std::pair<int, int>> &component_pixels, const helios::int2 &resolution) {
    std::vector<std::vector<bool>> component_mask(resolution.y, std::vector<bool>(resolution.x, false));
    for (const auto &pixel: component_pixels) {
        component_mask[pixel.second][pixel.first] = true;
    }
    return component_mask;
}

std::vector<std::pair<int, int>> helios::annotation::traceBoundaryMoore(const std::vector<std::vector<bool>> &mask, int start_x, int start_y, const helios::int2 &resolution) {
    std::vector<std::pair<int, int>> contour;

    // 8-connected neighbors in clockwise order starting from East
    int dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
    int dy[] = {0, 1, 1, 1, 0, -1, -1, -1};

    int x = start_x, y = start_y;
    int dir = 6; // Start looking West (opposite of East)

    do {
        contour.push_back({x, y});

        // Look for the next boundary pixel
        bool found = false;
        for (int i = 0; i < 8; i++) {
            int new_dir = (dir + i) % 8;
            int nx = x + dx[new_dir];
            int ny = y + dy[new_dir];

            if (nx >= 0 && nx < resolution.x && ny >= 0 && ny < resolution.y && mask[ny][nx]) {
                x = nx;
                y = ny;
                dir = (new_dir + 6) % 8; // Backtrack direction
                found = true;
                break;
            }
        }

        if (!found)
            break;

    } while (!(x == start_x && y == start_y) && contour.size() < resolution.x * resolution.y);

    return contour;
}

std::vector<std::pair<int, int>> helios::annotation::traceBoundarySimple(const std::vector<std::vector<bool>> &mask, int start_x, int start_y, const helios::int2 &resolution) {
    std::vector<std::pair<int, int>> contour;
    std::set<std::pair<int, int>> visited_boundary;

    // Use a simple approach: walk along the boundary
    std::queue<std::pair<int, int>> boundary_queue;
    boundary_queue.push({start_x, start_y});
    visited_boundary.insert({start_x, start_y});

    while (!boundary_queue.empty()) {
        auto [x, y] = boundary_queue.front();
        boundary_queue.pop();
        contour.push_back({x, y});

        // 8-connected neighbors
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                if (di == 0 && dj == 0)
                    continue;
                int nx = x + di;
                int ny = y + dj;

                if (nx >= 0 && nx < resolution.x && ny >= 0 && ny < resolution.y && mask[ny][nx] && visited_boundary.find({nx, ny}) == visited_boundary.end()) {

                    // Check if this pixel is on the boundary
                    bool is_boundary = false;
                    for (int ddi = -1; ddi <= 1; ddi++) {
                        for (int ddj = -1; ddj <= 1; ddj++) {
                            if (ddi == 0 && ddj == 0)
                                continue;
                            int nnx = nx + ddi;
                            int nny = ny + ddj;
                            if (nnx < 0 || nnx >= resolution.x || nny < 0 || nny >= resolution.y || !mask[nny][nnx]) {
                                is_boundary = true;
                                break;
                            }
                        }
                        if (is_boundary)
                            break;
                    }

                    if (is_boundary) {
                        boundary_queue.push({nx, ny});
                        visited_boundary.insert({nx, ny});
                    }
                }
            }
        }
    }

    return contour;
}

std::vector<std::map<std::string, std::vector<float>>> helios::annotation::maskToAnnotations(const std::map<int, std::vector<std::vector<bool>>> &label_masks, uint object_class_ID, const helios::int2 &resolution, int image_id) {
    std::vector<std::map<std::string, std::vector<float>>> annotations;
    int annotation_id = 0;

    for (const auto &label_pair: label_masks) {
        const auto &mask = label_pair.second;

        // Create a visited mask for connected components
        std::vector<std::vector<bool>> visited(resolution.y, std::vector<bool>(resolution.x, false));

        // Find all connected components for this label
        for (int j = 0; j < resolution.y; j++) {
            for (int i = 0; i < resolution.x; i++) {
                if (mask[j][i] && !visited[j][i]) {
                    // Check whether this pixel is on the boundary of the region
                    bool is_boundary = false;

                    for (int di = -1; di <= 1; di++) {
                        for (int dj = -1; dj <= 1; dj++) {
                            int ni = i + di;
                            int nj = j + dj;
                            if (ni < 0 || ni >= resolution.x || nj < 0 || nj >= resolution.y || !mask[nj][ni]) {
                                is_boundary = true;
                                break;
                            }
                        }
                        if (is_boundary)
                            break;
                    }

                    if (is_boundary) {
                        // First, mark all pixels in this connected component using flood fill
                        std::stack<std::pair<int, int>> stack;
                        std::vector<std::pair<int, int>> component_pixels;
                        stack.push({i, j});
                        visited[j][i] = true;

                        int min_x = i, max_x = i, min_y = j, max_y = j;
                        int area = 0;

                        while (!stack.empty()) {
                            auto [ci, cj] = stack.top();
                            stack.pop();
                            component_pixels.push_back({ci, cj});
                            area++;

                            min_x = std::min(min_x, ci);
                            max_x = std::max(max_x, ci);
                            min_y = std::min(min_y, cj);
                            max_y = std::max(max_y, cj);

                            // Check 8-connected neighbors. This must match the connectivity of the boundary tracers below, which are 8-connected: a 4-connected fill would split a diagonal chain of pixels into a
                            // separate component per pixel, while the tracer would walk the whole chain as one object.
                            for (int di = -1; di <= 1; di++) {
                                for (int dj = -1; dj <= 1; dj++) {
                                    if (di == 0 && dj == 0)
                                        continue;
                                    int ni = ci + di;
                                    int nj = cj + dj;
                                    if (ni >= 0 && ni < resolution.x && nj >= 0 && nj < resolution.y && mask[nj][ni] && !visited[nj][ni]) {
                                        stack.push({ni, nj});
                                        visited[nj][ni] = true;
                                    }
                                }
                            }
                        }

                        // Isolate this component so that the trace cannot leak into a diagonally touching neighbor
                        std::vector<std::vector<bool>> component_mask = buildComponentMask(component_pixels, resolution);

                        std::pair<int, int> start_pixel = findStartingBoundaryPixel(component_mask, resolution);

                        if (start_pixel.first >= 0) {
                            auto contour = traceBoundaryMoore(component_mask, start_pixel.first, start_pixel.second, resolution);

                            // The Moore trace can stall on very small or awkwardly shaped regions, so fall back to collecting the boundary pixels directly
                            if (contour.size() < 10) {
                                contour = traceBoundarySimple(component_mask, start_pixel.first, start_pixel.second, resolution);
                            }

                            if (contour.size() >= 3) {
                                std::map<std::string, std::vector<float>> annotation;
                                annotation["id"] = {(float) annotation_id++};
                                annotation["image_id"] = {(float) image_id};
                                annotation["category_id"] = {(float) object_class_ID};
                                annotation["bbox"] = {(float) min_x, (float) min_y, (float) (max_x - min_x), (float) (max_y - min_y)};
                                annotation["area"] = {(float) area};
                                annotation["iscrowd"] = {0.0f};

                                // Convert contour to segmentation format (flatten coordinates)
                                std::vector<float> segmentation;
                                for (const auto &point: contour) {
                                    segmentation.push_back((float) point.first); // x coordinate
                                    segmentation.push_back((float) point.second); // y coordinate
                                }
                                annotation["segmentation"] = segmentation;

                                annotations.push_back(annotation);
                            }
                        }
                    }
                }
            }
        }
    }

    return annotations;
}

std::pair<nlohmann::json, int> helios::annotation::initializeCOCOJson(const std::string &filename, bool append, const helios::int2 &resolution, const std::string &image_file) {
    nlohmann::json coco_json;
    int image_id = 0;

    if (append) {
        std::ifstream existing_file(filename);
        if (existing_file.is_open()) {
            try {
                existing_file >> coco_json;
            } catch (const std::exception &e) {
                coco_json.clear();
            }
            existing_file.close();
        }
    }

    // Initialize JSON structure if empty
    if (coco_json.empty()) {
        coco_json["categories"] = nlohmann::json::array();
        coco_json["images"] = nlohmann::json::array();
        coco_json["annotations"] = nlohmann::json::array();
    }

    // Extract just the filename (no path) from the image file
    std::filesystem::path image_path_obj(image_file);
    std::string filename_only = image_path_obj.filename().string();

    // Check if this image already exists in the JSON
    bool image_exists = false;
    for (const auto &img: coco_json["images"]) {
        if (img["file_name"] == filename_only) {
            image_id = img["id"];
            image_exists = true;
            break;
        }
    }

    // If image doesn't exist, add it with a new unique ID
    if (!image_exists) {
        // Find the next available image ID
        int max_image_id = -1;
        for (const auto &img: coco_json["images"]) {
            if (img["id"] > max_image_id) {
                max_image_id = img["id"];
            }
        }
        image_id = max_image_id + 1;

        // Add the new image entry
        nlohmann::json image_entry;
        image_entry["id"] = image_id;
        image_entry["file_name"] = filename_only;
        image_entry["height"] = resolution.y;
        image_entry["width"] = resolution.x;
        coco_json["images"].push_back(image_entry);
    }

    return std::make_pair(coco_json, image_id);
}

void helios::annotation::addCOCOCategory(nlohmann::json &coco_json, const std::vector<uint> &class_IDs, const std::vector<std::string> &class_names) {
    if (class_IDs.size() != class_names.size()) {
        helios_runtime_error("ERROR (annotation::addCOCOCategory): The lengths of class_IDs and class_names vectors must be the same.");
    }

    for (size_t i = 0; i < class_IDs.size(); ++i) {
        bool category_exists = false;
        for (auto &cat: coco_json["categories"]) {
            if (cat["id"] == class_IDs[i]) {
                category_exists = true;
                break;
            }
        }
        if (!category_exists) {
            nlohmann::json category;
            category["id"] = class_IDs[i];
            category["name"] = class_names[i];
            category["supercategory"] = "none";
            coco_json["categories"].push_back(category);
        }
    }
}

void helios::annotation::writeCOCOJson(const nlohmann::json &coco_json, const std::string &filename) {
    std::ofstream json_file(filename);
    if (!json_file.is_open()) {
        helios_runtime_error("ERROR (annotation::writeCOCOJson): Could not open file '" + filename + "'.");
    }

    json_file << coco_json.dump(2) << std::endl;
    json_file.close();
}

void helios::annotation::writeYOLOBoxes(const std::vector<YOLOBox> &boxes, const std::string &filename) {
    std::ofstream label_file(filename);
    if (!label_file.is_open()) {
        helios_runtime_error("ERROR (annotation::writeYOLOBoxes): Could not open output bounding box file '" + filename + "'.");
    }

    // The precision is set once, before any value is written, so that all four geometric values are
    // formatted the same way.
    label_file << std::setprecision(6) << std::fixed;

    for (const YOLOBox &box: boxes) {
        if (box.size.x <= 0.f || box.size.y <= 0.f) { // filter boxes describing no region
            continue;
        }
        label_file << box.class_ID << " " << box.center.x << " " << box.center.y << " " << box.size.x << " " << box.size.y << std::endl;
    }

    label_file.close();
}

void helios::annotation::writeYOLOClassNames(const std::map<uint, std::string> &class_names, const std::string &filename) {
    std::ofstream classes_file(filename);
    if (!classes_file.is_open()) {
        helios_runtime_error("ERROR (annotation::writeYOLOClassNames): Could not open output class name file '" + filename + "'.");
    }

    // The class index is the line number counting from zero, so the indices must be contiguous
    // starting at zero for the file to describe them correctly.
    uint expected_class_ID = 0;
    for (const auto &class_name: class_names) {
        if (class_name.first != expected_class_ID) {
            helios_runtime_error("ERROR (annotation::writeYOLOClassNames): Class indices must be contiguous and start at zero, because a class name file identifies each class by its line number. Expected class index " +
                                 std::to_string(expected_class_ID) + " but found " + std::to_string(class_name.first) + ".");
        }
        classes_file << class_name.second << std::endl;
        expected_class_ID++;
    }

    classes_file.close();
}

std::vector<helios::annotation::YOLOBox> helios::annotation::labelBoundingBoxes(const helios::Context *context, const std::vector<uint> &pixel_UUIDs, const helios::int2 &resolution, const std::vector<std::string> &data_labels,
                                                                                const std::vector<uint> &class_IDs, bool use_object_data) {

    if (data_labels.size() != class_IDs.size()) {
        helios_runtime_error("ERROR (annotation::labelBoundingBoxes): The lengths of data_labels and class_IDs vectors must be the same.");
    }
    validatePixelMapSize(pixel_UUIDs, resolution, "annotation::labelBoundingBoxes");

    // Bounds (xmin, xmax, ymin, ymax) in pixels of each object, keyed by (class_ID, label value)
    std::map<std::pair<uint, uint>, vec4> label_bounds;

    for (int j = 0; j < resolution.y; j++) {
        for (int i = 0; i < resolution.x; i++) {

            const uint UUID_plus_one = pixel_UUIDs.at(size_t(j) * size_t(resolution.x) + size_t(i));
            if (UUID_plus_one == 0) { // no primitive visible in this pixel
                continue;
            }
            const uint UUID = UUID_plus_one - 1;
            if (!context->doesPrimitiveExist(UUID)) {
                continue;
            }

            for (size_t label_idx = 0; label_idx < data_labels.size(); label_idx++) {

                uint label_value;
                if (!readLabelValue(context, UUID, data_labels[label_idx], use_object_data, label_value)) {
                    continue;
                }

                const std::pair<uint, uint> key = std::make_pair(class_IDs[label_idx], label_value);

                if (label_bounds.find(key) == label_bounds.end()) {
                    label_bounds[key] = make_vec4(1e6, -1, 1e6, -1);
                }

                vec4 &bounds = label_bounds[key];
                if (i < bounds.x) {
                    bounds.x = i;
                }
                if (i > bounds.y) {
                    bounds.y = i;
                }
                if (j < bounds.z) {
                    bounds.z = j;
                }
                if (j > bounds.w) {
                    bounds.w = j;
                }
            }
        }
    }

    std::vector<YOLOBox> yolo_boxes;
    yolo_boxes.reserve(label_bounds.size());
    for (const auto &box: label_bounds) {
        const vec4 bbox = box.second;
        // The bounds are the indices of the first and last pixels of the object. Pixel i covers the interval from i to i+1, so the box that encloses those pixels whole ends at the far edge of the last one.
        const vec2 box_min = make_vec2(bbox.x, bbox.z);
        const vec2 box_max = make_vec2(bbox.y + 1.f, bbox.w + 1.f);
        YOLOBox yolo_box;
        yolo_box.class_ID = box.first.first;
        yolo_box.center = make_vec2(0.5f * (box_min.x + box_max.x) / float(resolution.x), 0.5f * (box_min.y + box_max.y) / float(resolution.y));
        yolo_box.size = make_vec2((box_max.x - box_min.x) / float(resolution.x), (box_max.y - box_min.y) / float(resolution.y));
        yolo_boxes.push_back(yolo_box);
    }

    return yolo_boxes;
}

void helios::annotation::writeLabelBoundingBoxes(const helios::Context *context, const std::vector<uint> &pixel_UUIDs, const helios::int2 &resolution, const std::vector<std::string> &data_labels, const std::vector<uint> &class_IDs,
                                                 const std::string &image_file, const std::string &classes_txt_file, const std::string &image_path, bool use_object_data, const std::string &caller) {

    if (data_labels.size() != class_IDs.size()) {
        helios_runtime_error("ERROR (" + caller + "): The lengths of the data label and object_class_ID vectors must be the same.");
    }
    validatePixelMapSize(pixel_UUIDs, resolution, caller);

    std::string output_path = image_path;
    if (!image_path.empty() && !validateOutputPath(output_path)) {
        helios_runtime_error("ERROR (" + caller + "): Invalid image output directory '" + image_path + "'. Check that the path exists and that you have write permission.");
    } else if (!isDirectoryPath(output_path)) {
        helios_runtime_error("ERROR (" + caller + "): Expected a directory path but got a file path for argument 'image_path'.");
    }

    const std::string outfile_txt = output_path + std::filesystem::path(image_file).stem().string() + ".txt";

    writeYOLOBoxes(labelBoundingBoxes(context, pixel_UUIDs, resolution, data_labels, class_IDs, use_object_data), outfile_txt);

    std::ofstream classes_txt_stream(output_path + classes_txt_file);
    if (!classes_txt_stream.is_open()) {
        helios_runtime_error("ERROR (" + caller + "): Could not open output classes file '" + output_path + classes_txt_file + "'.");
    }
    for (size_t i = 0; i < class_IDs.size(); i++) {
        classes_txt_stream << class_IDs.at(i) << " " << data_labels.at(i) << std::endl;
    }
    classes_txt_stream.close();
}

std::map<int, std::vector<std::vector<bool>>> helios::annotation::labelMasks(const helios::Context *context, const std::vector<uint> &pixel_UUIDs, const helios::int2 &resolution, const std::string &data_label, bool use_object_data) {

    validatePixelMapSize(pixel_UUIDs, resolution, "annotation::labelMasks");

    std::map<int, std::vector<std::vector<bool>>> label_masks;

    for (int j = 0; j < resolution.y; j++) {
        for (int i = 0; i < resolution.x; i++) {

            const uint UUID_plus_one = pixel_UUIDs.at(size_t(j) * size_t(resolution.x) + size_t(i));
            if (UUID_plus_one == 0) { // no primitive visible in this pixel
                continue;
            }
            const uint UUID = UUID_plus_one - 1;
            if (!context->doesPrimitiveExist(UUID)) {
                continue;
            }

            uint label_value;
            if (!readLabelValue(context, UUID, data_label, use_object_data, label_value)) {
                continue;
            }

            if (label_masks.find(label_value) == label_masks.end()) {
                label_masks[label_value] = std::vector<std::vector<bool>>(resolution.y, std::vector<bool>(resolution.x, false));
            }
            label_masks[label_value][j][i] = true;
        }
    }

    return label_masks;
}

void helios::annotation::writeLabelSegmentationMasks(const helios::Context *context, const std::vector<uint> &pixel_UUIDs, const helios::int2 &resolution, const std::vector<std::string> &data_labels, const std::vector<uint> &class_IDs,
                                                     const std::string &json_filename, const std::string &image_file, const std::vector<std::string> &data_attribute_labels, bool append_file, bool use_object_data, const std::string &caller) {

    if (data_labels.size() != class_IDs.size()) {
        helios_runtime_error("ERROR (" + caller + "): The lengths of the data label and object_class_ID vectors must be the same.");
    }
    validatePixelMapSize(pixel_UUIDs, resolution, caller);

    // Check that all data labels exist
    const std::vector<std::string> all_primitive_data = context->listAllPrimitiveDataLabels();
    const std::vector<std::string> all_object_data = context->listAllObjectDataLabels();
    const std::vector<std::string> &all_label_data = use_object_data ? all_object_data : all_primitive_data;
    helios::WarningAggregator missing_label_warnings;
    for (const auto &data_label: data_labels) {
        if (std::find(all_label_data.begin(), all_label_data.end(), data_label) == all_label_data.end()) {
            if (use_object_data) {
                missing_label_warnings.addWarning("missing_object_data_label", "Object data label '" + data_label + "' does not exist in the context.");
            } else {
                missing_label_warnings.addWarning("missing_primitive_data_label", "Primitive data label '" + data_label + "' does not exist in the context.");
            }
        }
    }
    missing_label_warnings.report(std::cerr);

    if (!std::filesystem::exists(image_file)) {
        helios_runtime_error("ERROR (" + caller + "): Image file '" + image_file + "' does not exist.");
    }

    // Ensure the JSON filename has a .json extension
    std::string outfile = json_filename;
    if (outfile.length() < 5 || outfile.substr(outfile.length() - 5) != ".json") {
        outfile += ".json";
    }

    auto coco_json_pair = initializeCOCOJson(outfile, append_file, resolution, image_file);
    nlohmann::json coco_json = coco_json_pair.first;
    const int image_id = coco_json_pair.second;
    addCOCOCategory(coco_json, class_IDs, data_labels);

    // Attributes are looked up in primitive data first and object data second; labels found in neither are ignored
    struct AttributeInfo {
        std::string label;
        bool is_primitive_data;
    };
    std::vector<AttributeInfo> attribute_info;
    for (const auto &attribute_label: data_attribute_labels) {
        if (std::find(all_primitive_data.begin(), all_primitive_data.end(), attribute_label) != all_primitive_data.end()) {
            attribute_info.push_back({attribute_label, true});
        } else if (std::find(all_object_data.begin(), all_object_data.end(), attribute_label) != all_object_data.end()) {
            attribute_info.push_back({attribute_label, false});
        }
    }
    const bool use_attributes = !attribute_info.empty();

    for (size_t label_idx = 0; label_idx < data_labels.size(); ++label_idx) {

        const std::map<int, std::vector<std::vector<bool>>> label_masks = labelMasks(context, pixel_UUIDs, resolution, data_labels[label_idx], use_object_data);

        // Find the highest existing annotation ID to avoid conflicts
        int max_annotation_id = -1;
        for (const auto &existing_annotation: coco_json["annotations"]) {
            if (existing_annotation["id"] > max_annotation_id) {
                max_annotation_id = existing_annotation["id"];
            }
        }

        // The annotation and its attributes are built together for each connected component, so that the attribute values written to an annotation are always those of its own pixels.
        for (const auto &label_pair: label_masks) {
            const auto &mask = label_pair.second;

            std::vector<std::vector<bool>> visited(resolution.y, std::vector<bool>(resolution.x, false));

            for (int j = 0; j < resolution.y; j++) {
                for (int i = 0; i < resolution.x; i++) {
                    if (!mask[j][i] || visited[j][i]) {
                        continue;
                    }

                    // Flood fill the connected component that starts at this pixel
                    std::stack<std::pair<int, int>> stack;
                    std::vector<std::pair<int, int>> component_pixels;
                    stack.push({i, j});
                    visited[j][i] = true;

                    int min_x = i, max_x = i, min_y = j, max_y = j;
                    int area = 0;

                    while (!stack.empty()) {
                        auto [ci, cj] = stack.top();
                        stack.pop();
                        area++;
                        component_pixels.push_back({ci, cj});

                        min_x = std::min(min_x, ci);
                        max_x = std::max(max_x, ci);
                        min_y = std::min(min_y, cj);
                        max_y = std::max(max_y, cj);

                        // Check 8-connected neighbors. This must match the connectivity of the boundary tracers below, which are 8-connected: a 4-connected fill would split a diagonal chain of pixels into a
                        // separate component per pixel, while the tracer would walk the whole chain as one object.
                        for (int di = -1; di <= 1; di++) {
                            for (int dj = -1; dj <= 1; dj++) {
                                if (di == 0 && dj == 0)
                                    continue;
                                int ni = ci + di;
                                int nj = cj + dj;
                                if (ni >= 0 && ni < resolution.x && nj >= 0 && nj < resolution.y && mask[nj][ni] && !visited[nj][ni]) {
                                    stack.push({ni, nj});
                                    visited[nj][ni] = true;
                                }
                            }
                        }
                    }

                    // Trace the boundary over a mask holding only this component, so that the start pixel is guaranteed to belong to it and the walk cannot cross into a different component that touches it diagonally.
                    const std::vector<std::vector<bool>> component_mask = buildComponentMask(component_pixels, resolution);
                    const std::pair<int, int> start_pixel = findStartingBoundaryPixel(component_mask, resolution);
                    if (start_pixel.first < 0) {
                        continue;
                    }

                    auto contour = traceBoundaryMoore(component_mask, start_pixel.first, start_pixel.second, resolution);

                    // The Moore trace can stall on very small or awkwardly shaped regions, so fall back to collecting the boundary pixels directly
                    if (contour.size() < 10) {
                        contour = traceBoundarySimple(component_mask, start_pixel.first, start_pixel.second, resolution);
                    }

                    if (contour.size() < 3) { // too few points to form a polygon
                        continue;
                    }

                    nlohmann::json json_annotation;
                    json_annotation["id"] = max_annotation_id + 1;
                    json_annotation["image_id"] = image_id;
                    json_annotation["category_id"] = (int) class_IDs[label_idx];
                    // The width and height count whole pixels, so the box reaches the far edge of the last pixel in each direction
                    json_annotation["bbox"] = {min_x, min_y, max_x - min_x + 1, max_y - min_y + 1};
                    json_annotation["area"] = area;
                    json_annotation["iscrowd"] = 0;

                    // Convert contour to segmentation format (flatten coordinates)
                    std::vector<int> segmentation_coords;
                    for (const auto &point: contour) {
                        segmentation_coords.push_back(point.first); // x coordinate
                        segmentation_coords.push_back(point.second); // y coordinate
                    }
                    json_annotation["segmentation"] = {segmentation_coords};

                    if (use_attributes) {
                        std::map<std::string, double> component_attributes;
                        for (const auto &attribute: attribute_info) {
                            double sum = 0.0;
                            int count = 0;
                            for (const auto &[pixel_i, pixel_j]: component_pixels) {
                                const uint UUID_plus_one = pixel_UUIDs.at(size_t(pixel_j) * size_t(resolution.x) + size_t(pixel_i));
                                if (UUID_plus_one == 0 || !context->doesPrimitiveExist(UUID_plus_one - 1)) {
                                    continue;
                                }
                                double value = 0.0;
                                if (readNumericData(context, UUID_plus_one - 1, attribute.label, attribute.is_primitive_data, value)) {
                                    sum += value;
                                    count++;
                                }
                            }
                            // An attribute carried by none of the mask's primitives is reported as zero
                            component_attributes[attribute.label] = (count > 0) ? sum / count : 0.0;
                        }
                        json_annotation["attributes"] = component_attributes;
                    }

                    coco_json["annotations"].push_back(json_annotation);
                    max_annotation_id++;
                }
            }
        }
    }

    writeCOCOJson(coco_json, outfile);
}
