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
#include "global.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <queue>
#include <set>
#include <stack>

using namespace helios;

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
