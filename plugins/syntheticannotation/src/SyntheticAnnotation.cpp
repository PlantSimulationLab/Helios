/** \file "SyntheticAnnotation.cpp" Primary source file for synthetic image annotation plug-in.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
*/

#include "SyntheticAnnotation.h"
#include "annotation_io.h"
#include <iomanip>
#include "Visualizer.h"

using namespace std;
using namespace helios;

namespace {

    //! Restores the primitive appearance that render() overwrites, whatever path leaves the function
    /**
     * render() recolors every primitive with its label's ID code so that the IDs can be read back
     * out of the rendered image. Those colors are not the user's, so they must be put back before
     * returning -- including when a helios_runtime_error is thrown partway through, which would
     * otherwise leave the caller's entire scene painted in ID codes.
     */
    struct PrimitiveAppearanceGuard {
        PrimitiveAppearanceGuard(helios::Context *a_context, std::vector<uint> a_UUIDs) : context(a_context), UUIDs(std::move(a_UUIDs)) {
            colors.reserve(UUIDs.size());
            texture_overridden.reserve(UUIDs.size());
            for (uint UUID: UUIDs) {
                colors.push_back(context->getPrimitiveColorRGBA(UUID));
                texture_overridden.push_back(context->isPrimitiveTextureColorOverridden(UUID));
            }
        }

        ~PrimitiveAppearanceGuard() {
            for (size_t p = 0; p < UUIDs.size(); p++) {
                if (!context->doesPrimitiveExist(UUIDs.at(p))) {
                    continue;
                }
                context->setPrimitiveColor(UUIDs.at(p), colors.at(p));
                if (!texture_overridden.at(p)) {
                    context->usePrimitiveTextureColor(UUIDs.at(p));
                }
                // "object_label" is set by labelPrimitives() purely to drive this rendering pass.
                // Leaving it behind would let a later render() -- or any user code inspecting
                // primitive data -- see labels from a previous run.
                if (context->doesPrimitiveDataExist(UUIDs.at(p), "object_label")) {
                    context->clearPrimitiveData(UUIDs.at(p), "object_label");
                }
            }
        }

        PrimitiveAppearanceGuard(const PrimitiveAppearanceGuard &) = delete;
        PrimitiveAppearanceGuard &operator=(const PrimitiveAppearanceGuard &) = delete;

        helios::Context *context;
        std::vector<uint> UUIDs;
        std::vector<helios::RGBAcolor> colors;
        std::vector<bool> texture_overridden;
    };

} // namespace

SyntheticAnnotation::SyntheticAnnotation(helios::Context *__context) {
    context = __context;
    objectdetection_enabled = true;
    semanticsegmentation_enabled = true;
    instancesegmentation_enabled = true;
    currentLabelID = 1;
    printmessages = true;
    background_color = make_RGBcolor(0.9, 0.9, 0.9);
    skydome_texture_file = "plugins/visualizer/textures/SkyDome_clouds.jpg";
    labelminpixels = 10;
    window_width = 1000;
    window_height = 800;
    camera_position.push_back(make_vec3(1, 0, 1));
    camera_lookat.push_back(make_vec3(0, 0, 1));
}

void SyntheticAnnotation::labelPrimitives(const char *label) {
    labelPrimitives(context->getAllUUIDs(), label);
}

void SyntheticAnnotation::labelPrimitives(const uint UUIDs, const char *label) {
    std::vector<uint> UUID_vect = {UUIDs};
    labelPrimitives(UUID_vect, label);
}

void SyntheticAnnotation::labelPrimitives(const std::vector<uint> &UUIDs, const char *label) {
    std::vector<std::vector<uint>> UUIDs_vect;
    UUIDs_vect.push_back(UUIDs);
    labelPrimitives(UUIDs_vect, label);
}

void SyntheticAnnotation::labelPrimitives(const std::vector<std::vector<uint>> &UUIDs, const char *label) {

    if (UUIDs.size() == 0) {
        return;
    }

    std::vector<uint> IDs;
    IDs.resize(UUIDs.size());
    for (size_t group = 0; group < UUIDs.size(); group++) { // looping over label groups, which is the outer index of the UUIDs vector
        for (size_t p = 0; p < UUIDs.at(group).size(); p++) { // looping over primitives in label group, which is hte inner index of the UUIDs vector
            if (context->doesPrimitiveExist(UUIDs.at(group).at(p))) {
                // set object_label primitive data for this group
                context->setPrimitiveData(UUIDs.at(group).at(p), "object_label", currentLabelID);
            }
        }
        IDs.at(group) = currentLabelID;
        currentLabelID++;
    }

    if (labelUUIDs.find(label) == labelUUIDs.end()) { // this is the first time we've seen this label group
        labelUUIDs[label] = UUIDs;
        labelIDs[label] = IDs;
    } else {
        for (size_t group = 0; group < UUIDs.size(); group++) {
            labelUUIDs.at(label).push_back(UUIDs.at(group));
        }
        labelIDs.at(label).insert(labelIDs.at(label).end(), IDs.begin(), IDs.end());
    }

    assert(labelUUIDs.at(label).size() == labelIDs.at(label).size());
}

void SyntheticAnnotation::labelUnlabeledPrimitives(const char *label) {

    // Each remaining primitive becomes its own object group, matching how labelPrimitives()
    // interprets the outer index of a vector of UUID groups.
    std::vector<std::vector<uint>> unlabeled_groups;
    for (uint UUID: context->getAllUUIDs()) {
        if (!context->doesPrimitiveDataExist(UUID, "object_label")) {
            unlabeled_groups.push_back({UUID});
        }
    }

    if (unlabeled_groups.empty()) {
        return;
    }

    labelPrimitives(unlabeled_groups, label);
}

void SyntheticAnnotation::setBackgroundColor(const helios::RGBcolor &color) {
    background_color = color;
}

void SyntheticAnnotation::addSkyDome(const char *filename) {
    // An empty filename removes the sky dome, leaving the background colour showing. The existence check is skipped for it because there is no file to find, not because a missing file is being tolerated.
    if (std::string(filename).empty()) {
        skydome_texture_file.clear();
        return;
    }
    if (!std::filesystem::exists(filename)) {
        helios_runtime_error("ERROR (SyntheticAnnotation::addSkyDome): Sky dome texture file " + std::string(filename) + " does not exist.");
    }
    skydome_texture_file = filename;
}

void SyntheticAnnotation::setWindowSize(const uint __window_width, const uint __window_height) {
    window_width = __window_width;
    window_height = __window_height;
}

void SyntheticAnnotation::setMinimumLabelPixels(int a_labelminpixels) {
    // An outline needs at least three points to be a polygon, so an object covering fewer than
    // three pixels cannot be written to the segmentation masks at all. Allowing a threshold below
    // that would let such objects through the filter only to be dropped later with no diagnostic.
    if (a_labelminpixels < 3) {
        helios_runtime_error("ERROR (SyntheticAnnotation::setMinimumLabelPixels): Minimum label pixel count must be at least 3, but " + std::to_string(a_labelminpixels) +
                             " was given. An object covering fewer than three pixels has no traceable outline and cannot be written to the segmentation masks.");
    }
    labelminpixels = a_labelminpixels;
}

void SyntheticAnnotation::disableMessages() {
    printmessages = false;
}

void SyntheticAnnotation::enableMessages() {
    printmessages = true;
}

void SyntheticAnnotation::setCameraPosition(const helios::vec3 &a_camera_position, const helios::vec3 &a_camera_lookat) {
    std::vector<vec3> position = {a_camera_position};
    std::vector<vec3> lookat = {a_camera_lookat};
    setCameraPosition(position, lookat);
}

void SyntheticAnnotation::enableObjectDetection() {
    objectdetection_enabled = true;
}

void SyntheticAnnotation::disableObjectDetection() {
    objectdetection_enabled = false;
}

void SyntheticAnnotation::enableSemanticSegmentation() {
    semanticsegmentation_enabled = true;
}

void SyntheticAnnotation::disableSemanticSegmentation() {
    semanticsegmentation_enabled = false;
}

void SyntheticAnnotation::enableInstanceSegmentation() {
    instancesegmentation_enabled = true;
}

void SyntheticAnnotation::disableInstanceSegmentation() {
    instancesegmentation_enabled = false;
}

void SyntheticAnnotation::setCameraPosition(const std::vector<helios::vec3> &a_camera_position, const std::vector<helios::vec3> &a_camera_lookat) {

    if (a_camera_position.size() != a_camera_lookat.size()) {
        helios_runtime_error("ERROR (SyntheticAnnotation::setCameraPosition): the number of camera lookat coordinates specified is less than that of camera positions.");
    }
    camera_position = a_camera_position;
    camera_lookat = a_camera_lookat;
}

void SyntheticAnnotation::render(const char *outputdir) {

    if (labelIDs.empty()) {
        std::cerr << "WARNING (SyntheticAnnotation::render): No primitives have been labeled. You must call labelPrimitives() before generating rendered images. Exiting..." << std::endl;
        return;
    }

    std::vector<uint> UUIDs_all = context->getAllUUIDs();

    // get camera settings from global data if they were specified in a loaded XML file
    if (context->doesGlobalDataExist("camera_position")) {
        if (context->getGlobalDataType("camera_position") == helios::HELIOS_TYPE_VEC3) {
            context->getGlobalData("camera_position", camera_position);
        } else {
            std::cerr << "WARNING (SyntheticAnnotation::render): Camera position was specified in XML file but does not have type vec3. Ignoring.." << std::endl;
        }
    }
    if (context->doesGlobalDataExist("camera_lookat")) {
        if (context->getGlobalDataType("camera_lookat") == helios::HELIOS_TYPE_VEC3) {
            context->getGlobalData("camera_lookat", camera_lookat);
        } else {
            std::cerr << "WARNING (SyntheticAnnotation::render): Camera lookat coordinate was specified in XML file but does not have type vec3. Ignoring.." << std::endl;
        }
    }

    if (camera_position.size() != camera_lookat.size()) {
        helios_runtime_error("ERROR (SyntheticAnnotation::render): the number of camera lookat coordinates specified in XML file is less than that of camera positions.");
    }

    // get window size from global data if they were specified in a loaded XML
    if (context->doesGlobalDataExist("image_resolution")) {
        if (context->getGlobalDataType("image_resolution") == helios::HELIOS_TYPE_INT2) {
            int2 resolution;
            context->getGlobalData("image_resolution", resolution);
            window_width = resolution.x;
            window_height = resolution.y;
        } else {
            std::cerr << "WARNING (SyntheticAnnotation::render): Image resolution was specified in XML file, but does not have type int2. Ignoring..." << std::endl;
        }
    }

    // get output flags from global data if they were specified in a loaded XML
    if (context->doesGlobalDataExist("object_detection")) {
        if (context->getGlobalDataType("object_detection") == helios::HELIOS_TYPE_STRING) {
            std::string objectdetection;
            context->getGlobalData("object_detection", objectdetection);
            if (objectdetection == "enabled") {
                objectdetection_enabled = true;
            } else if (objectdetection == "disabled") {
                objectdetection_enabled = false;
            } else {
                helios_runtime_error("ERROR (SyntheticAnnotation::render): Object detection flag specified in XML file has unrecognized value '" + objectdetection + "'. Valid values are 'enabled' or 'disabled'.");
            }
        } else {
            std::cerr << "WARNING (SyntheticAnnotation::render): Object detection flag was specified in XML file, but does not have type string. Ignoring..." << std::endl;
        }
    }
    if (context->doesGlobalDataExist("semantic_segmentation")) {
        if (context->getGlobalDataType("semantic_segmentation") == helios::HELIOS_TYPE_STRING) {
            std::string semanticsegmentation;
            context->getGlobalData("semantic_segmentation", semanticsegmentation);
            if (semanticsegmentation == "enabled") {
                semanticsegmentation_enabled = true;
            } else if (semanticsegmentation == "disabled") {
                semanticsegmentation_enabled = false;
            } else {
                helios_runtime_error("ERROR (SyntheticAnnotation::render): Semantic segmentation flag specified in XML file has unrecognized value '" + semanticsegmentation + "'. Valid values are 'enabled' or 'disabled'.");
            }
        } else {
            std::cerr << "WARNING (SyntheticAnnotation::render): Semantic segmentation flag was specified in XML file, but does not have type string. Ignoring..." << std::endl;
        }
    }
    if (context->doesGlobalDataExist("instance_segmentation")) {
        if (context->getGlobalDataType("instance_segmentation") == helios::HELIOS_TYPE_STRING) {
            std::string instancesegmentation;
            context->getGlobalData("instance_segmentation", instancesegmentation);
            if (instancesegmentation == "enabled") {
                instancesegmentation_enabled = true;
            } else if (instancesegmentation == "disabled") {
                instancesegmentation_enabled = false;
            } else {
                helios_runtime_error("ERROR (SyntheticAnnotation::render): Instance segmentation flag specified in XML file has unrecognized value '" + instancesegmentation + "'. Valid values are 'enabled' or 'disabled'.");
            }
        } else {
            std::cerr << "WARNING (SyntheticAnnotation::render): Instance segmentation flag was specified in XML file, but does not have type string. Ignoring..." << std::endl;
        }
    }

    // check whether the output directory was supplied with a trailing '/' - if not, add it
    std::string odir = outputdir;
    if (odir.back() != '/') {
        odir += '/';
    }

    std::string slash = "/";
#ifdef _WIN32
    std::replace(odir.begin(), odir.end(), '/', '\\');
    slash = "\\";
#endif
    bool dir = std::filesystem::create_directory(odir);
    if (!dir && !std::filesystem::exists(odir)) {
        helios_runtime_error("ERROR (SyntheticAnnotation::render): output directory " + std::string(outputdir) + " could not be created. Exiting...");
    }
    // create sub-directory structure for each view
    for (int d = 0; d < camera_position.size(); d++) {
        std::stringstream viewdir;
        viewdir << odir << "view" << std::setfill('0') << std::setw(5) << d << slash;
        dir = std::filesystem::create_directory(viewdir.str());
        if (!dir && !std::filesystem::exists(viewdir.str())) {
            helios_runtime_error("ERROR (SyntheticAnnotation::render): view sub-directory could not be created. Exiting...");
        }
    }

    uint framebufferW, framebufferH;
    std::stringstream outfile;

    //------ RGB rendering with no labels --------//

    if (printmessages) {
        std::cout << "Rendering RGB image containing " << UUIDs_all.size() / 1000.f << "K primitives..." << std::flush;
    }

    Visualizer vis_RGB(window_width, window_height, 8, false, false);
    vis_RGB.disableMessages();

    vis_RGB.getFramebufferSize(framebufferW, framebufferH);

    vis_RGB.buildContextGeometry(context);
    vis_RGB.hideWatermark();
    vis_RGB.setBackgroundColor(background_color);
    // The sky dome is drawn over the background colour, so a caller that has set a flat colour would otherwise never see it. Clearing the sky texture is how that caller asks for the plain colour instead,
    // which is what matching a photographed backdrop needs.
    if (!skydome_texture_file.empty()) {
        vis_RGB.setBackgroundSkyTexture(skydome_texture_file.c_str(), 30);
    }
    vis_RGB.setLightDirection(sphere2cart(make_SphericalCoord(30 * M_PI / 180.f, 205 * M_PI / 180.f)));
    vis_RGB.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);

    for (int view = 0; view < camera_position.size(); view++) {

        vis_RGB.setCameraPosition(camera_position.at(view), camera_lookat.at(view));

        vis_RGB.plotUpdate(true);

        outfile.clear();
        outfile.str("");
        outfile << odir << "view" << std::setfill('0') << std::setw(5) << view << "/RGB_rendering.jpeg";
        // std::snprintf(outfile, odir.size()+48, "%sview%05d/RGB_rendering.jpeg", odir.c_str(),view);
        vis_RGB.printWindow(outfile.str().c_str());
    }

    vis_RGB.closeWindow();

    if (printmessages) {
        std::cout << "done." << std::endl;
    }

    // Record the original color and texture-override flag of every primitive, and restore them --
    // along with clearing the "object_label" data -- when this function returns by any path.
    PrimitiveAppearanceGuard appearance_guard(context, UUIDs_all);

    //------ Combined image labeled by RGB color code --------//

    if (printmessages) {
        std::cout << "Generating labeled image containing " << labelIDs.size() << " label groups..." << std::endl;
    }

    // The ID pass encodes label IDs as RGB colors and decodes them back out of the rendered
    // pixels, so the framebuffer must reproduce primitive colors exactly: no color boost (see
    // Visualizer::enableExactColorMode()), no lighting, and no anti-aliasing (0 samples below,
    // since blended edge pixels decode to meaningless IDs).
    //
    // It is also rendered headless, which draws to an offscreen framebuffer that is read back
    // directly. The windowed path instead guesses whether the front or back buffer holds the
    // current frame by sampling nine pixels and taking whichever has more non-black content;
    // against the white background used here both buffers score identically, the tie is broken
    // toward the back buffer, and the result is that getWindowPixelsRGB() returns the *previous*
    // frame. That makes every object's mask a copy of the object rendered before it.
    Visualizer vis(window_width, window_height, 0, false, true);
    vis.disableMessages();
    vis.enableExactColorMode();
    vis.setLightingModel(Visualizer::LIGHTING_NONE);

    vis.getFramebufferSize(framebufferW, framebufferH);

    // Map each object's ID code to the label it belongs to, so that the integer IDs appearing in
    // pixelID_combined.txt and the instance masks can be interpreted. Also assign each label a
    // zero-based class index, which is the class written into the bounding-box annotations.
    outfile.clear();
    outfile.str("");
    outfile << odir << "ID_mapping.txt";
    std::ofstream mapping_file(outfile.str());
    mapping_file << "object_ID label class_ID" << std::endl;

    std::map<std::string, int> label_class_IDs;
    //! Class index of each object ID, used to label that object's mask in the COCO output
    std::map<int, uint> object_class_of_ID;
    int next_class_ID = 0;
    for (auto g = labelIDs.begin(); g != labelIDs.end(); ++g) {
        label_class_IDs[g->first] = next_class_ID;
        next_class_ID++;
    }

    // A class name file in the standard one-name-per-line order, indexed by class ID.
    outfile.clear();
    outfile.str("");
    outfile << odir << "classes.txt";
    std::ofstream classes_file(outfile.str());
    for (auto g = labelIDs.begin(); g != labelIDs.end(); ++g) {
        classes_file << g->first << std::endl;
    }
    classes_file.close();

    int gID = 0;
    for (auto g = labelIDs.begin(); g != labelIDs.end(); ++g) { // looping over labels

        std::vector<uint> label_group_IDs = g->second;

        std::string label = g->first;

        assert(labelIDs.at(label).size() == labelUUIDs.at(label).size());

        for (size_t group = 0; group < label_group_IDs.size(); group++) { // looping over objects within each label

            gID = label_group_IDs.at(group);
            mapping_file << gID << " " << label << " " << label_class_IDs.at(label) << std::endl;
            object_class_of_ID[int(gID)] = uint(label_class_IDs.at(label));
            RGBcolor code = int2rgb(gID);
            std::vector<uint> UUIDs_group = labelUUIDs.at(label).at(group);
            for (int p = 0; p < UUIDs_group.size(); p++) { // looping over primitives in group

                // labelUUIDs only ever contains primitives that were labeled, so every UUID
                // reaching here belongs to this group and gets the group's ID color code. The
                // "object_label" data is re-established here rather than assumed to still be
                // present: it is cleared when render() returns, so that a second render() call
                // does not see labels left behind by the first.
                if (!context->doesPrimitiveExist(UUIDs_group.at(p))) {
                    continue;
                }
                context->setPrimitiveData(UUIDs_group.at(p), "object_label", uint(gID));
                context->setPrimitiveColor(UUIDs_group.at(p), code);
                context->overridePrimitiveTextureColor(UUIDs_group.at(p));
            }
        }
    }

    // make all unlabeled primitives white
    for (size_t p = 0; p < UUIDs_all.size(); p++) {

        if (!context->doesPrimitiveDataExist(UUIDs_all.at(p), "object_label")) { // primitive has NOT been labeled
            context->setPrimitiveColor(UUIDs_all.at(p), make_RGBcolor(1, 1, 1));
        }
        context->overridePrimitiveTextureColor(UUIDs_all.at(p));
    }

    mapping_file.close();

    vis.setBackgroundColor(make_RGBcolor(1, 1, 1));


    std::vector<uint> pixels;
    pixels.resize(framebufferH * framebufferW * 3);

    vis.buildContextGeometry(context);
    vis.hideWatermark();

    for (int view = 0; view < camera_position.size(); view++) {

        vis.setCameraPosition(camera_position.at(view), camera_lookat.at(view));

        vis.plotUpdate(true);

        vis.getWindowPixelsRGB(&pixels[0]);

        outfile.clear();
        outfile.str("");
        outfile << odir << "view" << std::setfill('0') << std::setw(5) << view << "/pixelID_combined.txt";
        // std::snprintf(outfile, odir.size()+48, "%sview%05d/pixelID_combined.txt", odir.c_str(), view);
        std::ofstream file(outfile.str());
        // Rows are emitted top-down to match RGB_rendering.jpeg and the segmentation masks.
        // getWindowPixelsRGB() returns the framebuffer bottom-up, so source row
        // (framebufferH-1-j) supplies output row j.
        for (int j = 0; j < framebufferH; j++) {
            for (int i = 0; i < framebufferW; i++) {

                int t_row = 3 * ((framebufferH - 1 - j) * framebufferW + i);
                uint ID = rgb2int(make_RGBcolor(pixels[t_row] / 255.f, pixels[t_row + 1] / 255.f, pixels[t_row + 2] / 255.f));
                file << ID << " ";
            }
            file << std::endl;
        }
        file.close();
        //------ Generate labels for objects with occlusion --------//

        if (objectdetection_enabled) {

            if (printmessages) {
                std::cout << "Generating rectangular labels for view " << view << "..." << std::flush;
            }

            int4 bbox;

            // All of the view's boxes go into a single file named after the image, which is the
            // layout the YOLO format expects and what Visualizer::displayImageWithBoundingBoxes()
            // looks for. The class of each box is carried in its first column rather than by being
            // in a per-label file.
            std::vector<annotation::YOLOBox> yolo_boxes;

            for (auto g = labelIDs.begin(); g != labelIDs.end(); ++g) { // looping over labels
                for (size_t group = 0; group < g->second.size(); group++) { // looping over objects within each label

                    gID = g->second.at(group);

                    uint pixelcount = getGroupRectangularBBox(gID, pixels, framebufferW, framebufferH, bbox);

                    if (pixelcount >= labelminpixels) {
                        annotation::YOLOBox yolo_box;
                        yolo_box.class_ID = label_class_IDs.at(g->first);
                        yolo_box.center = make_vec2((bbox.x + 0.5f * (bbox.y - bbox.x)) / float(framebufferW), (bbox.z + 0.5f * (bbox.w - bbox.z)) / float(framebufferH));
                        yolo_box.size = make_vec2((bbox.y - bbox.x) / float(framebufferW), (bbox.w - bbox.z) / float(framebufferH));
                        yolo_boxes.push_back(yolo_box);
                    }
                }
            }

            outfile.clear();
            outfile.str("");
            outfile << odir << "view" << std::setfill('0') << std::setw(5) << view << "/RGB_rendering.txt";
            annotation::writeYOLOBoxes(yolo_boxes, outfile.str());

            // The class name file has to sit beside the annotation file: that is where the
            // visualizer looks for it when no class file is named explicitly.
            std::map<uint, std::string> class_names;
            for (const auto &label_class: label_class_IDs) {
                class_names[uint(label_class.second)] = label_class.first;
            }
            outfile.clear();
            outfile.str("");
            outfile << odir << "view" << std::setfill('0') << std::setw(5) << view << "/classes.txt";
            annotation::writeYOLOClassNames(class_names, outfile.str());


            if (printmessages) {
                std::cout << "done." << std::endl;
            }
        }

        if (semanticsegmentation_enabled) {

            if (printmessages) {
                std::cout << "Performing semantic segmentation for view " << view << "..." << std::flush;
            }

            // The semantic mask is a single multi-class image: one pixel per framebuffer pixel,
            // holding the class index of whichever label occupies it. Each label contributes its
            // own class index to the same mask, so the mask is accumulated across all labels here
            // and written out once after the loop. The class index for each label name is
            // recorded in semantic_segmentation_ID_mapping.txt.
            //
            // Unlabeled pixels carry the ID code of white, which is what an unlabeled primitive
            // and the background are both colored with in the ID pass above.
            const size_t mask_size = size_t(framebufferW) * size_t(framebufferH);
            std::vector<int> semantic_mask(mask_size, rgb2int(make_RGBcolor(1, 1, 1))); // background/unlabeled

            outfile.clear();
            outfile.str("");
            outfile << odir << "view" << std::setfill('0') << std::setw(5) << view << "/semantic_segmentation_ID_mapping.txt";
            std::ofstream SemanticSegmentationID(outfile.str());
            SemanticSegmentationID << "label" << " " << "class_ID" << std::endl;

            int new_label = 0;
            for (auto g = labelIDs.begin(); g != labelIDs.end(); ++g) { // looping over labels
                new_label += 1;
                SemanticSegmentationID << g->first << " " << new_label << std::endl;

                // Collect the object ID codes belonging to this label
                std::vector<uint> groupID;
                groupID.reserve(g->second.size());
                for (size_t group = 0; group < g->second.size(); group++) { // looping over objects within each label
                    groupID.push_back(g->second.at(group));
                }

                // Stamp this label's class index into every pixel occupied by one of its objects.
                // Rows are emitted top-down, but getWindowPixelsRGB() returns the framebuffer
                // bottom-up, so source row (framebufferH-1-j) supplies output row j.
                size_t element_position = 0;
                for (int j = 0; j < framebufferH; j++) {
                    for (int i = 0; i < framebufferW; i++) {

                        const int t_row = 3 * ((framebufferH - 1 - j) * framebufferW + i);
                        if (std::count(groupID.begin(), groupID.end(), rgb2int(make_RGBcolor(pixels[t_row] / 255.f, pixels[t_row + 1] / 255.f, pixels[t_row + 2] / 255.f)))) {
                            semantic_mask.at(element_position) = new_label;
                        }

                        element_position++;
                    }
                }
            }
            SemanticSegmentationID.close();

            // Write the completed multi-class mask
            outfile.clear();
            outfile.str("");
            outfile << odir << "view" << std::setfill('0') << std::setw(5) << view << "/semantic_segmentation.txt";
            std::ofstream SemanticSegmentation(outfile.str());
            for (int j = 0; j < framebufferH; j++) {
                for (int i = 0; i < framebufferW; i++) {
                    SemanticSegmentation << semantic_mask.at(size_t(j) * size_t(framebufferW) + size_t(i)) << " ";
                }
                SemanticSegmentation << std::endl;
            }
            SemanticSegmentation.close();

            if (printmessages) {
                std::cout << "Semantic segmentation ... done." << std::endl;
            }
        }

        //------ Instance segmentation masks in COCO format --------//

        if (instancesegmentation_enabled) {

            if (printmessages) {
                std::cout << "Performing instance segmentation for view " << view << "..." << std::flush;
            }

            // The masks come from the whole-scene rendering, so they describe each object as it
            // actually appears: parts hidden behind other objects are not included. This is the
            // convention the COCO format and the radiation plug-in's camera annotations both use.
            const std::map<int, std::vector<std::vector<bool>>> object_masks = buildObjectMasks(pixels, framebufferW, framebufferH);

            std::stringstream imagefile;
            imagefile << odir << "view" << std::setfill('0') << std::setw(5) << view << "/RGB_rendering.jpeg";
            outfile.clear();
            outfile.str("");
            outfile << odir << "view" << std::setfill('0') << std::setw(5) << view << "/instances.json";

            const int2 image_resolution = make_int2(int(framebufferW), int(framebufferH));
            auto coco = annotation::initializeCOCOJson(outfile.str(), false, image_resolution, imagefile.str());
            nlohmann::json coco_json = coco.first;
            const int image_id = coco.second;

            // Declare every label as a category, so that a viewer can name each mask.
            std::vector<uint> category_IDs;
            std::vector<std::string> category_names;
            for (const auto &label_class: label_class_IDs) {
                category_IDs.push_back(uint(label_class.second));
                category_names.push_back(label_class.first);
            }
            annotation::addCOCOCategory(coco_json, category_IDs, category_names);

            // Each object is traced on its own, so that two objects of the same class stay
            // separate annotations rather than merging into one region.
            int annotation_id = 0;
            for (const auto &object_mask: object_masks) {

                const int object_ID = object_mask.first;
                if (object_class_of_ID.find(object_ID) == object_class_of_ID.end()) {
                    continue; // a decoded ID that belongs to no labeled object
                }

                std::map<int, std::vector<std::vector<bool>>> single_object;
                single_object[object_ID] = object_mask.second;

                std::vector<std::map<std::string, std::vector<float>>> annotations = annotation::maskToAnnotations(single_object, object_class_of_ID.at(object_ID), image_resolution, image_id);

                for (const auto &ann: annotations) {
                    nlohmann::json json_annotation;
                    json_annotation["id"] = annotation_id++;
                    json_annotation["image_id"] = image_id;
                    json_annotation["category_id"] = (int) ann.at("category_id")[0];
                    const auto &bbox_values = ann.at("bbox");
                    json_annotation["bbox"] = {(int) bbox_values[0], (int) bbox_values[1], (int) bbox_values[2], (int) bbox_values[3]};
                    json_annotation["area"] = (int) ann.at("area")[0];
                    json_annotation["iscrowd"] = 0;
                    json_annotation["segmentation"] = {ann.at("segmentation")};
                    coco_json["annotations"].push_back(json_annotation);
                }
            }

            annotation::writeCOCOJson(coco_json, outfile.str());

            if (printmessages) {
                std::cout << "done." << std::endl;
            }
        }
    }

    vis.closeWindow();

    vis.clearGeometry();

    // Primitive colors and "object_label" data are restored by appearance_guard's destructor.
}

uint SyntheticAnnotation::getGroupRectangularBBox(const uint ID, const std::vector<uint> &pixels, const uint framebuffer_width, const uint framebuffer_height, helios::int4 &bbox) const {

    int t = 0;
    int xmin = framebuffer_width;
    int xmax = 0;
    int ymin = framebuffer_height;
    int ymax = 0;
    int pixelcount = 0;

    // The bounding box is reported in image coordinates with the origin at the TOP-left, which is
    // what the YOLO annotation format requires. getWindowPixelsRGB() returns the framebuffer
    // bottom-up, so source row (framebuffer_height-1-j) supplies image row j.
    for (int j = 0; j < framebuffer_height; j++) {
        for (int i = 0; i < framebuffer_width; i++) {

            t = 3 * ((framebuffer_height - 1 - j) * framebuffer_width + i);

            if (rgb2int(make_RGBcolor(pixels[t] / 255.f, pixels[t + 1] / 255.f, pixels[t + 2] / 255.f)) != ID) {
                continue;
            }

            if (i < xmin) {
                xmin = i;
            }
            if (i > xmax) {
                xmax = i;
            }
            if (j < ymin) {
                ymin = j;
            }
            if (j > ymax) {
                ymax = j;
            }

            pixelcount++;
        }
    }

    bbox = make_int4(xmin, xmax, ymin, ymax);

    if (xmin == framebuffer_width || xmax == 0 || ymin == framebuffer_height || ymax == 0) {
        bbox = make_int4(0, 0, 0, 0);
        return 0;
    } else {
        return pixelcount;
    }
}

std::map<int, std::vector<std::vector<bool>>> SyntheticAnnotation::buildObjectMasks(const std::vector<uint> &pixels, const uint framebuffer_width, const uint framebuffer_height) const {

    const int background_ID = rgb2int(make_RGBcolor(1, 1, 1));

    std::map<int, std::vector<std::vector<bool>>> object_masks;
    std::map<int, uint> pixel_counts;

    // Rows are built top-down to match the rendered image, but getWindowPixelsRGB() returns the
    // framebuffer bottom-up, so source row (framebuffer_height-1-j) supplies image row j.
    for (uint j = 0; j < framebuffer_height; j++) {
        for (uint i = 0; i < framebuffer_width; i++) {

            const size_t t = 3 * (size_t(framebuffer_height - 1 - j) * size_t(framebuffer_width) + size_t(i));
            const int ID = rgb2int(make_RGBcolor(pixels[t] / 255.f, pixels[t + 1] / 255.f, pixels[t + 2] / 255.f));

            if (ID == background_ID) { // background and unlabeled primitives
                continue;
            }

            if (object_masks.find(ID) == object_masks.end()) {
                object_masks[ID] = std::vector<std::vector<bool>>(framebuffer_height, std::vector<bool>(framebuffer_width, false));
                pixel_counts[ID] = 0;
            }
            object_masks.at(ID).at(j).at(i) = true;
            pixel_counts.at(ID)++;
        }
    }

    // Anti-aliasing along an object edge can decode to an ID that belongs to no object, and an
    // object that is almost entirely occluded carries too few pixels to annotate usefully. Both are
    // removed by the same minimum-pixel test applied to the bounding boxes.
    for (auto it = object_masks.begin(); it != object_masks.end();) {
        if (pixel_counts.at(it->first) < uint(labelminpixels)) {
            it = object_masks.erase(it);
        } else {
            ++it;
        }
    }

    return object_masks;
}

helios::RGBcolor SyntheticAnnotation::int2rgb(const int ID) const {

    float R, G, B;
    int r, g, b;
    int rem;

    b = floor(float(ID) / 256.f / 256.f);
    rem = ID - b * 256 * 256;
    g = floor(float(rem) / 256.f);
    rem = rem - g * 256;
    r = rem;

    R = float(r) / 255.f;
    G = float(g) / 255.f;
    B = float(b) / 255.f;

    return helios::make_RGBcolor(R, G, B);
}

int SyntheticAnnotation::rgb2int(const helios::RGBcolor color) const {

    int ID = color.r * 255 + color.g * 255 * 256 + color.b * 255 * 256 * 256;

    return ID;
}
