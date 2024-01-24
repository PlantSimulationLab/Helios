/** \file "SyntheticAnnotation.cpp" Primary source file for synthetic image annotation plug-in.

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/ 

#include "SyntheticAnnotation.h"
#include "Visualizer.h"
#include <iomanip>

using namespace std;
using namespace helios;

SyntheticAnnotation::SyntheticAnnotation( helios::Context* __context ){
  context = __context;
  objectdetection_enabled = true;
  semanticsegmentation_enabled = true;
  instancesegmentation_enabled = true;
  currentLabelID = 1;
  printmessages = true;
  background_color = make_RGBcolor(0.9,0.9,0.9);
  window_width = 1000;
  window_height = 800;
  camera_position.push_back( make_vec3(1,0,1) );
  camera_lookat.push_back( make_vec3(0, 0, 1) );
}

int SyntheticAnnotation::selfTest() const{

    if( printmessages ) {
        std::cout << "Running synthetic image annotation self-test..." << std::flush;
    }
    int error_count = 0;

    //testing the r-g-b encoding scheme here
    int ID1 = 77830;
    int ID2 = rgb2int(int2rgb(ID1));

    if( ID1!=ID2 ){
        if( printmessages ) {
            std::cerr << "failed. RGB encoding scheme incorrect." << std::endl;
        }
        error_count++;
    }

    if( error_count==0 ){
        if( printmessages ) {
            std::cout << "passed." << std::endl;
        }
        return 0;
    }else{
        if( printmessages ) {
            std::cout << "failed self-test with " << error_count << " errors." << std::endl;
        }
        return 1;
    }

}

void SyntheticAnnotation::labelPrimitives( const char* label ){
    labelPrimitives( context->getAllUUIDs(), label );
}

void SyntheticAnnotation::labelPrimitives( const uint UUIDs, const char* label ){
    std::vector<uint> UUID_vect = {UUIDs};
    labelPrimitives( UUID_vect, label );
}

void SyntheticAnnotation::labelPrimitives( const std::vector<uint> UUIDs, const char* label  ) {
    std::vector<std::vector<uint> > UUIDs_vect;
    UUIDs_vect.push_back(UUIDs);
    labelPrimitives( UUIDs_vect, label );
}

void SyntheticAnnotation::labelPrimitives( const std::vector<std::vector<uint> > UUIDs, const char* label  ){

    if( UUIDs.size()==0 ){
        return;
    }

    std::vector<uint> IDs;
    IDs.resize(UUIDs.size());
    for( size_t group=0; group<UUIDs.size(); group++ ) { //looping over label groups, which is the outer index of the UUIDs vector
        for (size_t p = 0; p < UUIDs.at(group).size(); p++) { //looping over primitives in label group, which is hte inner index of the UUIDs vector
            if (context->doesPrimitiveExist(UUIDs.at(group).at(p))) {
                //set object_label primitive data for this group
                context->setPrimitiveData(UUIDs.at(group).at(p), "object_label", currentLabelID);
            }
        }
        IDs.at(group) = currentLabelID;
        currentLabelID++;
    }

    if( labelUUIDs.find(label)==labelUUIDs.end() ) { //this is the first time we've seen this label group
        labelUUIDs[label] = UUIDs;
        labelIDs[label] = IDs;
    }else {
        for( size_t group=0; group<UUIDs.size(); group++ ) {
            labelUUIDs.at(label).push_back(UUIDs.at(group));
        }
        labelIDs.at( label ).insert( labelIDs.at( label ).end(), IDs.begin(), IDs.end() );
    }

    assert(labelUUIDs.at(label).size()==labelIDs.at(label).size());

}

void SyntheticAnnotation::labelUnlabeledPrimitives(const char *label) {

    std::vector<uint> UUIDs_all = context->getAllUUIDs();
    for( uint p=0; p<UUIDs_all.size(); p++ ){

    }

}

void SyntheticAnnotation::setBackgroundColor(const helios::RGBcolor &color ) {
    background_color = color;
}

void SyntheticAnnotation::addSkyDome(const char *filename) {


}

void SyntheticAnnotation::setWindowSize( const uint __window_width, const uint __window_height) {
    window_width = __window_width;
    window_height = __window_height;
}

void SyntheticAnnotation::setCameraPosition(const helios::vec3 &a_camera_position, const helios::vec3 &a_camera_lookat) {
    std::vector<vec3> position = {a_camera_position};
    std::vector<vec3> lookat = {a_camera_lookat};
    setCameraPosition( position, lookat );
}

void SyntheticAnnotation::enableObjectDetection(){
    objectdetection_enabled = true;
}

void SyntheticAnnotation::disableObjectDetection(){
    objectdetection_enabled = false;
}

void SyntheticAnnotation::enableSemanticSegmentation(){
    semanticsegmentation_enabled = true;
}

void SyntheticAnnotation::disableSemanticSegmentation(){
    semanticsegmentation_enabled = false;
}

void SyntheticAnnotation::enableInstanceSegmentation(){
    instancesegmentation_enabled = true;
}

void SyntheticAnnotation::disableInstanceSegmentation(){
    instancesegmentation_enabled = false;
}

void SyntheticAnnotation::setCameraPosition(const std::vector<helios::vec3> &a_camera_position, const std::vector<helios::vec3> &a_camera_lookat) {

    if( a_camera_position.size()!=a_camera_lookat.size() ){
        helios_runtime_error("ERROR (SyntheticAnnotation::setCameraPosition): the number of camera lookat coordinates specified is less than that of camera positions.");
    }
    camera_position = __camera_position;
    camera_lookat = __camera_lookat;

}

void SyntheticAnnotation::render( const char* outputdir ) {

    //todo: need to implement method for setting this
    int labelminpixels = 10;

    if (labelIDs.empty()) {
        std::cerr << "WARNING (SyntheticAnnotation::render): No primitives have been labeled. You must call labelPrimitives() before generating rendered images. Exiting..." << std::endl;
        return;
    }

    std::vector<uint> UUIDs_all = context->getAllUUIDs();

    //get camera settings from global data if they were specified in a loaded XML file
    if (context->doesGlobalDataExist("camera_position") ){
        if( context->getGlobalDataType("camera_position")==helios::HELIOS_TYPE_VEC3 ) {
            context->getGlobalData("camera_position", camera_position);
        }else{
            std::cerr << "WARNING (SyntheticAnnotation::render): Camera position was specified in XML file but does not have type vec3. Ignoring.." << std::endl;
        }
    }
    if (context->doesGlobalDataExist("camera_lookat") ){
        if( context->getGlobalDataType("camera_lookat")==helios::HELIOS_TYPE_VEC3 ) {
            context->getGlobalData("camera_lookat", camera_lookat);
        }else{
            std::cerr << "WARNING (SyntheticAnnotation::render): Camera lookat coordinate was specified in XML file but does not have type vec3. Ignoring.." << std::endl;
        }
    }

    if( camera_position.size()!=camera_lookat.size() ){
        helios_runtime_error("ERROR (SyntheticAnnotation::render): the number of camera lookat coordinates specified in XML file is less than that of camera positions.");
    }

    //get window size from global data if they were specified in a loaded XML
    if( context->doesGlobalDataExist( "image_resolution") ){
        if( context->getGlobalDataType("image_resolution")==helios::HELIOS_TYPE_INT2 ) {
            int2 resolution;
            context->getGlobalData("image_resolution", resolution);
            window_width = resolution.x;
            window_height = resolution.y;
        }else{
            std::cerr << "WARNING (SyntheticAnnotation::render): Image resolution was specified in XML file, but does not have type int2. Ignoring..." << std::endl;
        }
    }

    //get output flags from global data if they were specified in a loaded XML
    if( context->doesGlobalDataExist( "object_detection") ){
        if( context->getGlobalDataType("object_detection")==helios::HELIOS_TYPE_STRING ) {
            std::string objectdetection;
            context->getGlobalData("object_detection", objectdetection);
            if( objectdetection.compare("enabled") ) {
                objectdetection_enabled = true;
            }else if( objectdetection.compare("disabled") ) {
                objectdetection_enabled = false;
            }
        }else{
            std::cerr << "WARNING (SyntheticAnnotation::render): Object detection flag was specified in XML file, but does not have type string. Ignoring..." << std::endl;
        }
    }
    if( context->doesGlobalDataExist( "semantic_segmentation") ){
        if( context->getGlobalDataType("semantic_segmentation")==helios::HELIOS_TYPE_STRING ) {
            std::string semanticsegmentation;
            context->getGlobalData("semantic_segmentation", semanticsegmentation);
            if( semanticsegmentation.compare("enabled") ) {
                semanticsegmentation_enabled = true;
            }else if( semanticsegmentation.compare("disabled") ) {
                semanticsegmentation_enabled = false;
            }
        }else{
            std::cerr << "WARNING (SyntheticAnnotation::render): Semantic segmentation flag was specified in XML file, but does not have type string. Ignoring..." << std::endl;
        }
    }
    if( context->doesGlobalDataExist( "instance_segmentation") ){
        if( context->getGlobalDataType("instance_segmentation")==helios::HELIOS_TYPE_STRING ) {
            std::string instancesegmentation;
            context->getGlobalData("instance_segmentation", instancesegmentation);
            if( instancesegmentation.compare("enabled") ) {
                instancesegmentation_enabled = true;
            }else if( instancesegmentation.compare("disabled") ) {
                instancesegmentation_enabled = false;
            }
        }else{
            std::cerr << "WARNING (SyntheticAnnotation::render): Instance segmentation flag was specified in XML file, but does not have type string. Ignoring..." << std::endl;
        }
    }

    //check whether the output directory was supplied with a trailing '/' - if not, add it
    std::string odir = outputdir;
    if( odir.back()!='/' ){
        odir += '/';
    }

    //check that output directory exists, if not create it
    std::string createdir = "mkdir -p ";
    createdir += odir;
    int dir = system(createdir.c_str());
    if (dir < 0) {
        helios_runtime_error("ERROR (SyntheticAnnotation::render): output directory " + outputdir + " could not be created. Exiting...");
    }
    //create sub-directory structure for each view
    //std::string viewdir;
    for( int d=0; d<camera_position.size(); d++ ){
      std::stringstream viewdir;
      viewdir << createdir << "view" << std::setfill('0') << std::setw(5) << d << "/";
      std::cout << "viewdir: " << viewdir.str() << std::endl;
      //std::snprintf(viewdir,createdir.size()+24,"%sview%05d/",createdir.c_str(),d);
      int dir = system( viewdir.str().c_str() );
      if (dir < 0) {
        helios_runtime_error("ERROR (SyntheticAnnotation::render): view sub-directory could not be created. Exiting...");
      }
    }

    uint framebufferW, framebufferH;
    std::stringstream outfile;

    //------ RGB rendering with no labels --------//

    if (printmessages) {
        std::cout << "Rendering RGB image containing " << UUIDs_all.size() / 1000.f << "K primitives..." << std::flush;
    }

    Visualizer vis_RGB(window_width, window_height, 8, false);
    vis_RGB.disableMessages();

    vis_RGB.getFramebufferSize(framebufferW, framebufferH);

    vis_RGB.buildContextGeometry(context);
    vis_RGB.hideWatermark();
    vis_RGB.setBackgroundColor(background_color);
    vis_RGB.addSkyDomeByCenter( 20, make_vec3(0,0,0), 30, "plugins/visualizer/textures/SkyDome_clouds.jpg" );
    vis_RGB.setLightDirection(sphere2cart(make_SphericalCoord(30 * M_PI / 180.f, 205 * M_PI / 180.f)));
    vis_RGB.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);

    //todo: need to add option to tell which sky dome image file to use
    //vis_RGB.addSkyDomeByCenter( 50, make_vec3(0,0,0), 30, "plugins/visualizer/textures/SkyDome_clouds.jpg" );

    for( int view=0; view<camera_position.size(); view++ ) {

        vis_RGB.setCameraPosition(camera_position.at(view), camera_lookat.at(view));

        vis_RGB.plotUpdate( true );

        wait(5);

        outfile.clear();
        outfile.str("");
        outfile << odir << "view" << std::setfill('0') << std::setw(5) << view << "/RGB_rendering.jpeg";
        //std::snprintf(outfile, odir.size()+48, "%sview%05d/RGB_rendering.jpeg", odir.c_str(),view);
        vis_RGB.printWindow(outfile.str().c_str());

    }

    vis_RGB.closeWindow();

    if (printmessages) {
        std::cout << "done." << std::endl;
    }

    //keep track of the original color and texture override flag for each primitive so we can change it back at the end
    std::vector<RGBAcolor> color_original;
    std::vector<bool> textureoverride_original;
    color_original.resize(UUIDs_all.size());
    textureoverride_original.resize(UUIDs_all.size());

    //set color of all primitives based on label RGB color code
    for (int p = 0; p < UUIDs_all.size(); p++) {
        color_original.at(p) = context->getPrimitiveColorRGBA(UUIDs_all.at(p));
        textureoverride_original.at(p) = context->isPrimitiveTextureColorOverridden(UUIDs_all.at(p));
    }

    //------ Combined image labeled by RGB color code --------//

    if (printmessages) {
        std::cout << "Generating labeled image containing " << labelIDs.size() << " label groups..." << std::endl;
    }

    Visualizer vis(window_width, window_height, 0, false);
    vis.disableMessages();

    vis.getFramebufferSize(framebufferW, framebufferH);

    outfile.clear();
    outfile.str("");
    outfile << odir << "/ID_mapping.txt";
    //std::snprintf(outfile, odir.size()+24, "%s/ID_mapping.txt", odir.c_str());
    std::ofstream mapping_file(outfile.str());

    int gID = 0;
    for ( auto g = labelIDs.begin(); g != labelIDs.end(); ++g) { //looping over labels

        //todo I think some additional modifications will be needed to make this work again
//        mapping_file << g->first << " " << g->second << std::endl;
        std::vector<uint> label_group_IDs = g->second;

        std::string label = g->first;

        assert( labelIDs.at(label).size()==labelUUIDs.at(label).size() );

        for (size_t group = 0; group < label_group_IDs.size(); group++) {//looping over objects within each label

            gID = label_group_IDs.at(group);
            RGBcolor code = int2rgb(gID);
            std::vector<uint> UUIDs_group = labelUUIDs.at(label).at(group);
            for (int p = 0; p < UUIDs_group.size(); p++) { //looping over primitives in group

                if (context->doesPrimitiveDataExist(UUIDs_group.at(p), "object_label")) { //primitive has been labeled

                    context->setPrimitiveColor(UUIDs_group.at(p),code);

                } else { //primitive was not labeled, assign default color of white
                    context->setPrimitiveColor( UUIDs_group.at(p), make_RGBcolor(1, 1, 1));
                }
                context->overridePrimitiveTextureColor(UUIDs_group.at(p));
            }

        }
    }

    //make all unlabeled primitives white
    for( size_t p=0; p<UUIDs_all.size(); p++ ){

        if ( !context->doesPrimitiveDataExist(UUIDs_all.at(p), "object_label")) { //primitive has NOT been labeled
            context->setPrimitiveColor(UUIDs_all.at(p),make_RGBcolor(1, 1, 1));
        }
        context->overridePrimitiveTextureColor(UUIDs_all.at(p));

    }

    mapping_file.close();

    vis.setBackgroundColor(make_RGBcolor(1, 1, 1));


    std::vector<uint> pixels;
    pixels.resize(framebufferH * framebufferW * 3);

    vis.buildContextGeometry(context);
    vis.hideWatermark();

    for( int view=0; view<camera_position.size(); view++ ) {

        vis.setCameraPosition(camera_position.at(view), camera_lookat.at(view));

        vis.plotUpdate( true );

        vis.getWindowPixelsRGB(&pixels[0]);

        outfile.clear();
        outfile.str("");
        outfile << odir << "view" << std::setfill('0') << std::setw(5) << view << "/pixelID_combined.txt";
        //std::snprintf(outfile, odir.size()+48, "%sview%05d/pixelID_combined.txt", odir.c_str(), view);
        std::ofstream file(outfile.str());
        int t = 0;
        for (int j = framebufferH; j > 0; j--) {
            for (int i = 0; i < framebufferW; i++) {

                uint ID = rgb2int(make_RGBcolor(pixels[t] / 255.f, pixels[t + 1] / 255.f, pixels[t + 2] / 255.f));
                file << ID << " " << std::flush;

                t += 3;

            }
            file << std::endl;
            
        }
        file.close();
        int t_max = t;
        //------ Generate labels for objects with occlusion --------//

        if( objectdetection_enabled ){

            if( printmessages ){
                std::cout << "Generating rectangular labels for view " << view << "..." << std::flush;
            }

            int4 bbox;

            for ( auto g = labelIDs.begin(); g != labelIDs.end(); ++g) { //looping over labels
              outfile.clear();
              outfile.str("");
              outfile << odir << "view" << std::setfill('0') << std::setw(5) << view << "/rectangular_labels_" << g->first.c_str() << ".txt";
              //std::snprintf(outfile, outputdir.size()+48, "%s/view%05d/rectangular_labels_%s.txt", outputdir,view,g->first.c_str());
              std::ofstream labelfile_rectoccluded(outfile.str());
                for (size_t group = 0; group < g->second.size(); group++) {//looping over objects within each label

                    gID = g->second.at(group);

                    uint pixelcount = getGroupRectangularBBox( gID, pixels, framebufferW, framebufferH, bbox );

                    if( pixelcount>=labelminpixels  ){

                        //todo: This doesn't actually take into account different label groups. Need to write in different format.
                        labelfile_rectoccluded << 0 << " " << (bbox.x + 0.5 * (bbox.y - bbox.x)) / float(framebufferW) << " "
                                               << (bbox.z + 0.5 * (bbox.w - bbox.z)) / float(framebufferH) << " " << std::setprecision(6)
                                               << std::fixed << (bbox.y - bbox.x) / float(framebufferW) << " "
                                               << (bbox.w - bbox.z) / float(framebufferH) << std::endl;

                    }

                }
                labelfile_rectoccluded.close();
            }

            

            if( printmessages ){
                std::cout << "done." << std::endl;
            }

            }
            
            if( semanticsegmentation_enabled ){

                    if (printmessages) {
                        std::cout << "Performing semantic segmentation for view " << view << "... and element: " << std::flush << endl;
                    }

                    int cont = 1;
                    int new_label = 0;
                    std::vector<int> ID2;
                    std::vector<uint> groupIDall;
                    int element_position;
                    //Extract masks and write to file for each Label group
                    outfile.clear();
                    outfile.str("");
                    outfile << odir << "view" << std::setfill('0') << std::setw(5) << view << "/semantic_segmentation_ID_mapping.txt";
                    //std::snprintf(outfile, odir.size()+48, "%sview%05d/semantic_segmentation_ID_mapping.txt", odir.c_str(), view);
                    std::ofstream SemanticSegmentationID(outfile.str());
                    SemanticSegmentationID << "Element" << " " << "Label" << std::endl;
                    for ( auto g = labelIDs.begin(); g != labelIDs.end(); ++g) { //looping over labels
                        new_label += 1;
                        cout << g->first << endl;
                        SemanticSegmentationID << g->first << " " << new_label << std::endl;
                        outfile.clear();
                        outfile.str("");
                        outfile << odir << "view" << std::setfill('0') << std::setw(5) << view << "/semantic_segmentation.txt";
                        //std::snprintf(outfile, odir.size()+48, "%sview%05d/semantic_segmentation.txt", odir.c_str(), view);
                        std::ofstream SemanticSegmentation(outfile.str());
                        std::vector<uint> groupID;
                        for (size_t group = 0; group < g->second.size(); group++) {//looping over objects within each label
                            gID = g->second.at(group);
                            groupID.push_back(gID);
                        }
                            groupIDall.insert(end(groupIDall),begin(groupID),end(groupID));

                            if (cont == 1) {t = t_max; }
                            else if (cont == 0) {t = 0; }
                            
                            // Create blank image
                            if (new_label == 1) {
                                for( int j=0; j<framebufferH; j++ ){
                                    for( int i=0; i<framebufferW; i++ ){

                                        ID2.push_back(rgb2int( make_RGBcolor(1, 1, 1)));

                                    }
                                }
                            }
                            element_position = 0;
                            //Add labels of objects to image
                            for( int j=framebufferH; j>0; j-- ){
                                for( int i=0; i<framebufferW; i++ ){
                                    
                                    
                                    t = 3*((framebufferW-1)*j + i + j);
                                    if (std::count(groupID.begin(),groupID.end(),rgb2int(make_RGBcolor(pixels[t]/255.f,pixels[t+1]/255.f,pixels[t+2]/255.f)))){
                                        ID2.at(element_position) = new_label;
                                    }
                                    
                                    SemanticSegmentation << ID2.at(element_position) << " " << std::flush;
                                    
                                    element_position+=1;


                                }
                                //labelfile_semanticsegmentation << std::endl; //Use this line if you want separete files for the Semantic Segmentation
                                SemanticSegmentation << std::endl;
                            }
                        SemanticSegmentation.close();
             
                    }
                        SemanticSegmentationID.close();

                    if (printmessages) {
                        std::cout << "Semantic segmentation ... done." << std::endl;
                    }

            }

        

    }

    vis.closeWindow();

    vis.clearGeometry();

/*     if (printmessages) {
        std::cout << "Occluded objets ... done." << std::endl;
    } */

    //------ Generate labels for objects without occlusion --------//

    if( instancesegmentation_enabled ){

        int4 bbox;

        for( int view=0; view<camera_position.size(); view++ ) {

        if( printmessages ){
            std::cout << "Performing instance segmentation for view " << view << "... and element: " << std::flush << endl;
        }

            for (std::map<std::string, std::vector<uint> >::iterator g = labelIDs.begin();
                 g != labelIDs.end(); ++g) { //looping over labels
                 int counter_object = 0;
                 cout << g->first << endl;
                for (size_t group = 0; group < g->second.size(); group++) {//looping over objects within each label
                    counter_object += 1;
                    gID = g->second.at(group);

                    vis.buildContextGeometry(context, labelUUIDs.at(g->first).at(group));

                    vis.setCameraPosition(camera_position.at(view), camera_lookat.at(view));

                    vis.plotUpdate( true );

                    outfile.clear();
                    outfile.str("");
                    outfile << odir << "view" << std::setfill('0') << std::setw(5) << view << "/instance_segmentation_" << g->first.c_str() << "_" << std::setfill('0') << std::setw(7) << ".txt";
                    //std::snprintf(outfile, odir.size()+96, "%sview%05d/instance_segmentation_%s_%07d.txt",odir.c_str(), view,g->first.c_str(),counter_object);

                    //Extract masks and write to file

                    vis.getWindowPixelsRGB(&pixels[0]);

                    writePixelID(outfile.str().c_str(),labelminpixels,&vis);

                    vis.clearGeometry();

                }
            }

        }

        if( printmessages ){
            std::cout << "done." << std::endl;
        }

    }

    //------- Clean up ---------//

    //set primitive colors back to how they were before
    for (int p = 0; p < UUIDs_all.size(); p++) {
        context->setPrimitiveColor(UUIDs_all.at(p),color_original.at(p));
        if (!textureoverride_original.at(p)) {
            context->usePrimitiveTextureColor(UUIDs_all.at(p));
        }
    }


}

uint SyntheticAnnotation::getGroupRectangularBBox( const uint ID, const std::vector<uint> &pixels, const uint framebuffer_width, const uint framebuffer_height, helios::int4& bbox ) const {

    int t=0;
    int xmin = framebuffer_width;
    int xmax = 0;
    int ymin = framebuffer_height;
    int ymax = 0;
    int pixelcount = 0;

    for( int j=0; j<framebuffer_height; j++ ){
        for( int i=0; i<framebuffer_width; i++ ){

            if( rgb2int(make_RGBcolor(pixels[t]/255.f,pixels[t+1]/255.f,pixels[t+2]/255.f)) != ID ){
                t+=3;
                continue;
            }

            if( i<xmin ){
                xmin=i;
            }
            if( i>xmax ){
                xmax=i;
            }
            if( j<ymin ){
                ymin=j;
            }
            if( j>ymax ){
                ymax=j;
            }

            t+=3;
            pixelcount++;

        }
    }

    bbox = make_int4( xmin, xmax, ymin, ymax );

    if( xmin==framebuffer_width || xmax==0 || ymin==framebuffer_height || ymax==0 ){
        bbox = make_int4(0,0,0,0);
        return 0;
    }else{
        return pixelcount;
    }

}

helios::RGBcolor SyntheticAnnotation::int2rgb( const int ID ) const{

    float R, G, B;
    int r, g, b;
    int rem;

    b = floor(float(ID)/256.f/256.f);
    rem = ID-b*256*256;
    g = floor(float(rem)/256.f);
    rem = rem-g*256;
    r = rem;

    R = float(r)/255.f;
    G = float(g)/255.f;
    B = float(b)/255.f;

    return helios::make_RGBcolor(R,G,B);

}

int SyntheticAnnotation::rgb2int( const helios::RGBcolor color ) const{

    int ID = color.r*255+color.g*255*256+color.b*255*256*256;

    return ID;

}

void SyntheticAnnotation::writePixelID( const char* filename, const int labelminpixels, Visualizer* vis ) const{

    uint framebufferH, framebufferW;

    vis->getFramebufferSize(framebufferW,framebufferH);

    std::vector<uint> pixels;
    pixels.resize(framebufferH*framebufferW*3);

    vis->getWindowPixelsRGB( &pixels[0] );

    int t=0;
    int xmin = framebufferW;
    int xmax = 0;
    int ymin = framebufferH;
    int ymax = 0;
    int pixelcount = 0;

    for( int j=0; j<framebufferH; j++ ){
        for( int i=0; i<framebufferW; i++ ){

            if( pixels[t]==255 && pixels[t+1]==255 && pixels[t+2]==255 ){
                t+=3;
                continue;
            }

            if( i<xmin ){
                xmin=i;
            }
            if( i>xmax ){
                xmax=i;
            }
            if( j<ymin ){
                ymin=j;
            }
            if( j>ymax ){
                ymax=j;
            }

            t+=3;
            pixelcount++;

        }
    }

    if( xmin==framebufferW || xmax==0 || ymin==framebufferH || ymax==0 || pixelcount<labelminpixels ){
        return;
    }

    std::ofstream file(filename);

    file << xmin << " " << xmax << " " << ymin << " " << ymax << std::endl;

    // t=0;
    for( int j=framebufferH; j>0; j--  ){
        for( int i=0; i<framebufferW; i++ ){

            if( i>=xmin && i<=xmax && j>=ymin && j<=ymax ){

                int ID = rgb2int( make_RGBcolor(pixels[t]/255.f,pixels[t+1]/255.f,pixels[t+2]/255.f) );
                file << ID << " " <<  std::flush;

            }

            // t+=3;
            t = 3*((framebufferW-1)*j + i + j);

        }
        if( j>=ymin && j<=ymax ){
            file << std::endl;
        }
    }
    file.close();
}
