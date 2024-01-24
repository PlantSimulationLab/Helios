/** \file "SyntheticAnnotation.h" Primary header file for synthetic image annotation plug-in.
    \author Brian Bailey

    Copyright (C) 2016-2024 Brian Bailey

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

class SyntheticAnnotation{
public:

    //! Synthetic image annotation plug-in default constructor
    /** \param[in] "context" Pointer to the Helios context
    */
    explicit SyntheticAnnotation( helios::Context* context );

    //! Function to perform a self-test of plug-in functions
    int selfTest( ) const;

    void labelPrimitives( const char* label );

    void labelPrimitives( uint UUIDs, const char* label );

    void labelPrimitives( std::vector<uint> UUIDs, const char* label );

    void labelPrimitives( std::vector<std::vector<uint> > UUIDs, const char* label );

    void labelUnlabeledPrimitives( const char* label );

    void setBackgroundColor( const helios::RGBcolor &color );

    void addSkyDome( const char* filename );

    void setWindowSize( uint window_width, uint window_height );

    void setCameraPosition( const helios::vec3 &camera_position, const helios::vec3 &camera_lookat );

    void setCameraPosition( const std::vector<helios::vec3> &camera_position, const std::vector<helios::vec3> &camera_lookat );

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

    //! Render RGB image and generate annotations
    /* \param[in] "outputdir" Base directory to save output files.
     */
    void render( const char* outputdir );

private:

    helios::Context* context;

    bool printmessages;

    bool objectdetection_enabled;

    bool instancesegmentation_enabled;

    bool semanticsegmentation_enabled;

    helios::RGBcolor background_color;

    uint window_width, window_height;

    std::vector<helios::vec3> camera_position;

    std::vector<helios::vec3> camera_lookat;

    uint currentLabelID;

    std::map< std::string, std::vector<std::vector<uint> > > labelUUIDs;

    std::map< std::string, std::vector<uint> > labelIDs;

    helios::RGBcolor int2rgb( int ID ) const;

    int rgb2int( helios::RGBcolor color ) const;

    uint getGroupRectangularBBox( uint ID, const std::vector<uint>& pixels, uint framebuffer_width, uint framebuffer_height, helios::int4& bbox ) const;

    //void getGroupPolygonMask( const std::vector<uint>& pixels, const uint framebuffer_width, const uint framebuffer_height ) const;

    void writePixelID( const char* filename, int labelminpixels, Visualizer* vis ) const;

};

#endif
