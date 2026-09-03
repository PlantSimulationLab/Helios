/** \file "SyntheticAnnotation.h" Primary header file for synthetic image annotation plug-in.
    \author Brian Bailey

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef __SYNTHETICANNOTATION__
#define __SYNTHETICANNOTATION__

#include "Context.h"
#include "Visualizer.h"

class SyntheticAnnotation {
public:
    //! Synthetic image annotation plug-in default constructor
    /** \param[in] context Pointer to the Helios context
     */
    explicit SyntheticAnnotation(helios::Context *context);

    //! Function to perform a self-test of plug-in functions
    static int selfTest(int argc, char **argv);

    //! Assign a label to every primitive in the Context, as a single object
    /**
     * \param[in] label Label to assign
     */
    void labelPrimitives(const char *label);

    //! Assign a label to a single primitive, which becomes one object
    /**
     * \param[in] UUIDs UUID of the primitive to label
     * \param[in] label Label to assign
     */
    void labelPrimitives(uint UUIDs, const char *label);

    //! Assign a label to a group of primitives, which together form one object
    /**
     * \param[in] UUIDs UUIDs of the primitives making up the object
     * \param[in] label Label to assign
     */
    void labelPrimitives(const std::vector<uint> &UUIDs, const char *label);

    //! Assign a label to several groups of primitives, each group forming a separate object
    /**
     * This is normally the overload wanted, since a scene usually contains many objects of the
     * same class (many leaves, many fruit) that must be kept distinct from one another.
     *
     * \param[in] UUIDs Outer index is the object, inner index is the primitives making it up
     * \param[in] label Label to assign to all of the objects
     */
    void labelPrimitives(const std::vector<std::vector<uint>> &UUIDs, const char *label);

    //! Assign a label to every primitive that has not already been labeled
    /**
     * Each unlabeled primitive is placed in its own object group, in the same way that
     * \ref labelPrimitives(const std::vector<std::vector<uint>> &UUIDs, const char *label) "labelPrimitives()" treats a vector of UUID groups. Call this after all other
     * labelPrimitives() calls, since it acts on whatever remains unlabeled at the time it runs.
     *
     * \param[in] label Label to assign to the remaining primitives (e.g. "background")
     */
    void labelUnlabeledPrimitives(const char *label);

    //! Set the background color of the RGB rendering
    /**
     * \param[in] color Background color
     */
    void setBackgroundColor(const helios::RGBcolor &color);

    //! Set the sky dome texture image used as the background of the RGB rendering
    /**
     * The sky dome is drawn over the background colour, so it is what a viewer sees unless it is removed. Pass an empty filename to remove it and leave the flat colour set by
     * \ref setBackgroundColor(const helios::RGBcolor &color) "setBackgroundColor()" showing, which is what reproducing a photographed backdrop calls for.
     *
     * \param[in] filename Path to the sky dome texture image (JPEG or PNG), or an empty string to render against the background colour alone.
     */
    void addSkyDome(const char *filename);

    //! Set the resolution of the rendered images
    /**
     * \param[in] window_width Image width in pixels
     * \param[in] window_height Image height in pixels
     */
    void setWindowSize(uint window_width, uint window_height);

    //! Set the minimum number of pixels an object must cover to be written to the annotations
    /**
     * Objects smaller than this are omitted from the bounding-box and instance segmentation
     * output, which keeps objects that are almost entirely occluded out of the training data.
     *
     * \param[in] labelminpixels Minimum pixel count. Must be at least 3, since an object covering fewer pixels has no traceable outline. Default is 10.
     */
    void setMinimumLabelPixels(int labelminpixels);

    //! Disable standard output messages (default is enabled)
    void disableMessages();

    //! Enable standard output messages
    void enableMessages();

    //! Set the camera position and view direction for a single view
    /**
     * \param[in] camera_position Cartesian position of the camera
     * \param[in] camera_lookat Cartesian coordinate the camera points toward
     */
    void setCameraPosition(const helios::vec3 &camera_position, const helios::vec3 &camera_lookat);

    //! Set the camera positions and view directions for multiple views
    /**
     * The scene is rendered once per view, each into its own output sub-directory, which is the
     * usual way to generate a dataset of many images from one scene.
     *
     * \param[in] camera_position Cartesian position of the camera for each view
     * \param[in] camera_lookat Cartesian coordinate the camera points toward for each view
     */
    void setCameraPosition(const std::vector<helios::vec3> &camera_position, const std::vector<helios::vec3> &camera_lookat);

    //! Enable calculation and writing of rectangular bounding boxes for object detection when render() function is called
    void enableObjectDetection();

    //! Disable calculation and writing of rectangular bounding boxes for object detection when render() function is called
    void disableObjectDetection();

    //! Enable calculation and writing of object mask (full image) for semantic segmentation
    void enableSemanticSegmentation();

    //! Disable calculation and writing of object mask (full image) for semantic segmentation
    void disableSemanticSegmentation();

    //! Enable calculation and writing of un-occluded object masks for each object (instance segmentation)
    void enableInstanceSegmentation();

    //! Disable calculation and writing of un-occluded object masks for each object (instance segmentation)
    void disableInstanceSegmentation();

    //! Render the RGB image and generate all enabled annotations
    /**
     * The output directory is created if it does not already exist, with one sub-directory per
     * camera view.
     *
     * \param[in] outputdir Base directory to save output files
     */
    void render(const char *outputdir);

private:
    helios::Context *context;

    bool printmessages;

    bool objectdetection_enabled;

    bool instancesegmentation_enabled;

    bool semanticsegmentation_enabled;

    helios::RGBcolor background_color;

    //! Sky dome texture used as the background of the RGB rendering
    std::string skydome_texture_file;

    //! Minimum number of pixels an object must cover to be written to the annotations
    int labelminpixels;

    uint window_width, window_height;

    std::vector<helios::vec3> camera_position;

    std::vector<helios::vec3> camera_lookat;

    uint currentLabelID;

    std::map<std::string, std::vector<std::vector<uint>>> labelUUIDs;

    std::map<std::string, std::vector<uint>> labelIDs;

    helios::RGBcolor int2rgb(int ID) const;

    int rgb2int(helios::RGBcolor color) const;

    uint getGroupRectangularBBox(uint ID, const std::vector<uint> &pixels, uint framebuffer_width, uint framebuffer_height, helios::int4 &bbox) const;

    //! Build a binary mask for each labeled object visible in the rendered ID image
    /**
     * The masks are in top-down image order, which is the orientation the annotation formats
     * require. getWindowPixelsRGB() returns the framebuffer bottom-up, so the flip is applied here.
     * Objects covering fewer than the minimum number of pixels are omitted.
     *
     * \param[in] pixels Framebuffer contents from the object ID rendering pass.
     * \param[in] framebuffer_width Framebuffer width in pixels.
     * \param[in] framebuffer_height Framebuffer height in pixels.
     * \return Binary mask per object ID, indexed [row][column] with row 0 at the top of the image.
     */
    std::map<int, std::vector<std::vector<bool>>> buildObjectMasks(const std::vector<uint> &pixels, uint framebuffer_width, uint framebuffer_height) const;

    // void getGroupPolygonMask( const std::vector<uint>& pixels, const uint framebuffer_width, const uint framebuffer_height ) const;
};

#endif
