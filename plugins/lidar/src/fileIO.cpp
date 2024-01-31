/** \file "fileIO.cpp" Declarations for LiDAR plug-in related to file input/output.

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "LiDAR.h"
#include "pugixml.hpp"

using namespace helios;
using namespace std;

void LiDARcloud::loadXML( const char* filename ){
    loadXML( filename, false );
}

void LiDARcloud::loadXML( const char* filename, bool load_grid_only ){

    if( printmessages ){
        cout << "Reading XML file: " << filename << "..." << flush;
    }

    //Check if file exists
    ifstream f(filename);
    if( !f.good() ){
        cerr << "failed.\n";
        helios_runtime_error("ERROR (LiDARcloud::loadXML): XML file does not exist.");
    }

    // Using "pugixml" parser.  See pugixml.org
    pugi::xml_document xmldoc;

    //load file
    pugi::xml_parse_result result = xmldoc.load_file(filename);

    //error checking
    if (!result){
        cout << "failed." << endl;
        cerr << "XML  file " << filename << " parsed with errors, attribute value: [" << xmldoc.child("node").attribute("attr").value() << "]\n";
        helios_runtime_error("ERROR (LiDARcloud::loadXML): Errors were found while parsing XML file. Error description: " + std::string(result.description()) );
    }

    pugi::xml_node helios = xmldoc.child("helios");

    if( helios.empty() ){
        std::cout << "failed." << std::endl;
        helios_runtime_error("ERROR (LiDARcloud::loadXML): XML file must have tag '<helios> ... </helios>' bounding all other tags.");
    }

    //-------------- Scans ---------------//

    uint scan_count = 0; //counter variable for scans
    size_t total_hits = 0;

    if(load_grid_only == false){

        //looping over any scans specified in XML file
        for (pugi::xml_node s = helios.child("scan"); s; s = s.next_sibling("scan")){

            // ----- scan origin ------//
            const char* origin_str = s.child_value("origin");

            if( strlen(origin_str)==0 ){
                cerr << "failed.\n";
                helios_runtime_error("ERROR (LiDARcloud::loadXML): An origin was not specified for scan #" + std::to_string(scan_count));
            }

            vec3 origin = string2vec3( origin_str ); //note: pugi loads xml data as a character.  need to separate it into 3 floats

            // ----- scan size (resolution) ------//
            const char* size_str = s.child_value("size");

            helios::int2 size;
            if( strlen(size_str)==0 ){
                cerr << "failed.\n";
                helios_runtime_error("ERROR (LiDARcloud::loadXML): A size was not specified for scan #" + std::to_string(scan_count));
            }else{
                size = string2int2( size_str ); //note: pugi loads xml data as a character.  need to separate it into 2 ints
            }
            if( size.x<=0 || size.y<=0 ){
                cerr << "failed.\n";
                helios_runtime_error("ERROR (LiDARcloud::loadXML): The scan size must be positive (check scan #" + std::to_string(scan_count) + ").");
            }

            // ----- scan translation ------//
            const char* offset_str = s.child_value("translation");

            vec3 translation = make_vec3(0,0,0);
            if( strlen(offset_str)>0 ){
                translation = string2vec3( offset_str ); //note: pugi loads xml data as a character.  need to separate it into 3 floats
            }

            // ----- scan rotation ------//
            const char* rotation_str = s.child_value("rotation");

            SphericalCoord rotation_sphere(0,0,0);
            if( strlen(rotation_str)>0 ){
                vec2 rotation = string2vec2( rotation_str ); //note: pugi loads xml data as a character.  need to separate it into 2 floats
                rotation = rotation*M_PI/180.f;
                rotation_sphere = make_SphericalCoord(rotation.x,rotation.y);
            }

            // ----- thetaMin ------//
            const char* thetaMin_str = s.child_value("thetaMin");

            float thetaMin;
            if( strlen(thetaMin_str)==0 ){
                //cerr << "WARNING (loadXML): A minimum zenithal scan angle was not specified for scan #" << scan_count << "...assuming thetaMin = 0." << flush;
                thetaMin=0.f;
            }else{
                thetaMin = atof(thetaMin_str)*M_PI/180.f;
            }

            if( thetaMin<0 ){
                helios_runtime_error("ERROR (LiDARcloud::loadXML): thetaMin cannot be less than 0.");
            }

            // ----- thetaMax ------//
            const char* thetaMax_str = s.child_value("thetaMax");

            float thetaMax;
            if( strlen(thetaMax_str)==0 ){
                thetaMax=M_PI;
            }else{
                thetaMax = atof(thetaMax_str)*M_PI/180.f;
            }

            if( thetaMax-1e-5>M_PI ){
                helios_runtime_error("ERROR (LiDARcloud::loadXML): thetaMax cannot be greater than 180 degrees.");
            }

            // ----- phiMin ------//
            const char* phiMin_str = s.child_value("phiMin");

            float phiMin;
            if( strlen(phiMin_str)==0 ){
                phiMin=0.f;
            }else{
                phiMin = atof(phiMin_str)*M_PI/180.f;
            }

            if( phiMin<0 ){
                helios_runtime_error("ERROR (LiDARcloud::loadXML): phiMin cannot be less than 0.");
            }

            // ----- phiMax ------//
            const char* phiMax_str = s.child_value("phiMax");

            float phiMax;
            if( strlen(phiMax_str)==0 ){
                phiMax=2.f*M_PI;
            }else{
                phiMax = atof(phiMax_str)*M_PI/180.f;
            }

            if( phiMax-1e-5>4.f*M_PI ){
                helios_runtime_error("ERROR (LiDARcloud::loadXML): phiMax cannot be greater than 720 degrees.");
            }

            // ----- exitDiameter ------//
            const char* exitDiameter_str_uc = s.child_value("exitDiameter");
            const char* exitDiameter_str_lc = s.child_value("exitdiameter");

            float exitDiameter;
            if( strlen(exitDiameter_str_uc)==0 && strlen(exitDiameter_str_lc)==0 ){
                exitDiameter=0;
            }else if( strlen(exitDiameter_str_uc)>0 ){
                exitDiameter = fmax(0,atof(exitDiameter_str_uc));
            }else{
                exitDiameter = fmax(0,atof(exitDiameter_str_lc));
            }

            // ----- beamDivergence ------//
            const char* beamDivergence_str_uc = s.child_value("beamDivergence");
            const char* beamDivergence_str_lc = s.child_value("beamdivergence");

            float beamDivergence;
            if( strlen(beamDivergence_str_uc)==0 && strlen(beamDivergence_str_lc)==0 ){
                beamDivergence=0;
            }else if( strlen(beamDivergence_str_uc)>0 ){
                beamDivergence = fmax(0,atof(beamDivergence_str_uc));
            }else{
                beamDivergence = fmax(0,atof(beamDivergence_str_lc));
            }

            // ----- distanceFilter ------//
            const char* dFilter_str = s.child_value("distanceFilter");

            float distanceFilter = -1;
            if( strlen(dFilter_str)>0 ){
                distanceFilter = atof(dFilter_str);
            }

            // ------ ASCII data file format ------- //

            const char* data_format = s.child_value("ASCII_format");

            std::vector<std::string> column_format;
            if( strlen(data_format)!=0 ){

                std::string tmp;

                std::istringstream stream(data_format);
                while( stream >> tmp ){
                    column_format.push_back(tmp);
                }

            }

            //create a temporary scan object
            ScanMetadata scan(origin, size.x, thetaMin, thetaMax, size.y, phiMin, phiMax, exitDiameter, beamDivergence, column_format );

            addScan( scan );

            uint scanID = getScanCount() -1;

            // ----- ASCII data file name ------//
            std::string data_filename = deblank(s.child_value("filename"));

            if( !data_filename.empty() ){

                char str[100];
                strcpy(str,"input/"); //first look in the input directory
                strcat(str,data_filename.c_str());
                ifstream f(str);
                if( !f.good() ){

                    //if not in input directory, try absolute path
                    strcpy(str,data_filename.c_str());
                    f.open(str);

                    if( !f.good() ){
                        cout << "failed.\n";
                        helios_runtime_error("ERROR (LiDARcloud::loadXML): Data file `" + std::string(str) + "' given for scan #" + std::to_string(scan_count) + " does not exist.");
                    }
                    f.close();
                }

                scan.data_file = str; //set the data file for the scan

                //add hit points to scan if data file was given

                total_hits += loadASCIIFile( scanID, scan );

                if( translation.magnitude()>0.f ){
                    coordinateShift( scanID, translation );
                }
                if( rotation_sphere.elevation!=0 || rotation_sphere.azimuth!=0 ){
                    coordinateRotation( scanID, rotation_sphere );
                }

            }

            scan_count ++;

        }
    }

    //------------ Grids ------------//

    uint cell_count = 0; //counter variable for scans

    //looping over any grids specified in XML file
    for (pugi::xml_node s = helios.child("grid"); s; s = s.next_sibling("grid")){

        // ----- grid center ------//
        const char* center_str = s.child_value("center");

        if( strlen(center_str)==0 ){
            cerr << "failed.\n";
            helios_runtime_error("ERROR (LiDARcloud::loadXML): A center was not specified for grid #" + std::to_string(cell_count));
        }

        vec3 center = string2vec3( center_str ); //note: pugi loads xml data as a character.  need to separate it into 3 floats

        // ----- grid size ------//
        const char* gsize_str = s.child_value("size");

        if( strlen(gsize_str)==0 ){
            cerr << "failed.\n";
            helios_runtime_error("ERROR (LiDARcloud::loadXML): A size was not specified for grid cell #" + std::to_string(cell_count));
        }

        vec3 gsize = string2vec3( gsize_str ); //note: pugi loads xml data as a character.  need to separate it into 3 floats

        if( gsize.x<=0 || gsize.y<=0 || gsize.z<=0 ){
            cerr << "failed.\n";
            helios_runtime_error("ERROR (LiDARcloud::loadXML): The grid cell size must be positive.");
        }

        // ----- grid rotation ------//
        float rotation;
        const char* grot_str = s.child_value("rotation");

        if( strlen(grot_str)==0 ){
            rotation = 0; //if no rotation specified, assume = 0
        }else{
            rotation = atof(grot_str);
        }

        // ----- grid cells ------//
        uint Nx, Ny, Nz;

        const char* Nx_str = s.child_value("Nx");

        if( strlen(Nx_str)==0 ){ //If no Nx specified, assume Nx=1;
            Nx=1;
        }else{
            Nx = atof(Nx_str);
        }
        if( Nx<=0 ){
            cerr << "failed.\n";
            helios_runtime_error("ERROR (LiDARcloud::loadXML): The number of grid cells must be positive.");
        }

        const char* Ny_str = s.child_value("Ny");

        if( strlen(Ny_str)==0 ){ //If no Ny specified, assume Ny=1;
            Ny=1;
        }else{
            Ny = atof(Ny_str);
        }
        if( Ny<=0 ){
            cerr << "failed.\n";
            helios_runtime_error("ERROR (LiDARcloud::loadXML): The number of grid cells must be positive.");
        }

        const char* Nz_str = s.child_value("Nz");

        if( strlen(Nz_str)==0 ){ //If no Nz specified, assume Nz=1;
            Nz=1;
        }else{
            Nz = atof(Nz_str);
        }
        if( Nz<=0 ){
            cerr << "failed.\n";
            helios_runtime_error("ERROR (LiDARcloud::loadXML): The number of grid cells must be positive.");
        }

        int3 gridDivisions = helios::make_int3(Nx, Ny, Nz);

        //add cells to grid

        vec3 gsubsize = make_vec3(float(gsize.x)/float(Nx),float(gsize.y)/float(Ny),float(gsize.z)/float(Nz));

        float x, y, z;
        uint count = 0;
        for( int k=0; k<Nz; k++ ){
            z = -0.5f*float(gsize.z) + (float(k)+0.5f)*float(gsubsize.z);
            for( int j=0; j<Ny; j++ ){
                y = -0.5f*float(gsize.y) + (float(j)+0.5f)*float(gsubsize.y);
                for( int i=0; i<Nx; i++ ){
                    x = -0.5f*float(gsize.x) + (float(i)+0.5f)*float(gsubsize.x);

                    vec3 subcenter = make_vec3(x,y,z);

                    vec3 subcenter_rot = rotatePoint(subcenter, make_SphericalCoord(0,rotation*M_PI/180.f) );

                    if( printmessages ){
                        cout << "Adding grid cell #" << count << " with center " << subcenter_rot.x+center.x << "," << subcenter_rot.y+center.y << "," << subcenter.z+center.z << " and size " << gsubsize.x << " x " << gsubsize.y << " x " << gsubsize.z << endl;
                    }

                    addGridCell( subcenter+center, center, gsubsize, gsize, rotation*M_PI/180.f, make_int3(i,j,k), make_int3(Nx,Ny,Nz) );

                    count++;

                }
            }
        }


    }

    if( printmessages ){

        cout << "done." << endl;

        cout << "Successfully read " << getScanCount() << " scan(s), which contain " << total_hits << " total hit points." << endl;

    }

}

size_t LiDARcloud::loadASCIIFile( uint scanID, ScanMetadata &scandata ){

    ifstream datafile(scandata.data_file); //open the file

    if(!datafile.is_open()){ //check that file exists
        throw( std::runtime_error("ERROR (LiDARcloud::loadASCIIFile): ASCII data file '" + scandata.data_file + "' does not exist.") );
    }

    vec3 temp_xyz;
    SphericalCoord temp_direction;
    RGBcolor temp_rgb;
    float temp_row, temp_column;
    double temp_data;
    std::map<std::string, double> data;

    vector<unsigned int> row, column;
    std::size_t hit_count = 0;
    while ( datafile.good() ){ //loop through file to read scan data

        temp_xyz = make_vec3(-9999,-9999,-9999);
        temp_rgb = make_RGBcolor(1,0,0); //default color: red
        temp_row = -1;
        temp_column = -1;
        temp_direction = make_SphericalCoord(-9999,-9999);

        for( uint i=0; i<scandata.columnFormat.size(); i++ ){
            if( scandata.columnFormat.at(i) == "row" ){
                datafile >> temp_row;
            }else if( scandata.columnFormat.at(i) == "column" ){
                datafile >> temp_column;
            }else if( scandata.columnFormat.at(i) == "zenith" ){
                datafile >> temp_direction.zenith;
            }else if( scandata.columnFormat.at(i) == "azimuth" ){
                datafile >> temp_direction.azimuth;
            }else if( scandata.columnFormat.at(i) == "zenith_rad" ){
                datafile >> temp_direction.zenith;
                temp_direction.zenith = deg2rad(temp_direction.zenith);
            }else if( scandata.columnFormat.at(i) == "azimuth_rad" ){
                datafile >> temp_direction.azimuth;
                temp_direction.azimuth = deg2rad(temp_direction.azimuth);
            }else if( scandata.columnFormat.at(i) == "x" ){
                datafile >> temp_xyz.x;
            }else if( scandata.columnFormat.at(i) == "y" ){
                datafile >> temp_xyz.y;
            }else if( scandata.columnFormat.at(i) == "z" ){
                datafile >> temp_xyz.z;
            }else if( scandata.columnFormat.at(i) == "r" ){
                datafile >> temp_rgb.r;
            }else if( scandata.columnFormat.at(i) == "g" ){
                datafile >> temp_rgb.g;
            }else if( scandata.columnFormat.at(i) == "b" ){
                datafile >> temp_rgb.b;
            }else if( scandata.columnFormat.at(i) == "r255" ){
                datafile >> temp_rgb.r;
                temp_rgb.r/=255.f;
            }else if( scandata.columnFormat.at(i) == "g255" ){
                datafile >> temp_rgb.g;
                temp_rgb.g/=255.f;
            }else if( scandata.columnFormat.at(i) == "b255" ){
                datafile >> temp_rgb.b;
                temp_rgb.b/=255.f;
            }else{ //assume that rest is data
                datafile >> temp_data;
                data[ scandata.columnFormat.at(i) ] = temp_data;
            }
        }

        if( !datafile.good() ){//if the whole line was not read successfully, stop
            if( hit_count==0 ){
                std::cerr << "WARNING: Something is likely wrong with the data file " << scandata.data_file << ". Check that the format is consistent with that specified in the XML metadata file." << std::endl;
            }
            break;
        }

        // -- Checks to make sure everything was specified correctly -- //

        //hit point
        if( temp_xyz.x==-9999 ){
            throw( std::runtime_error("ERROR (LiDARcloud::loadASCIIFile): x-coordinate not specified for hit point #" + std::to_string(hit_count) + " of scan #" + std::to_string(scanID) ) );
        }else if( temp_xyz.y==-9999 ){
            throw( std::runtime_error("ERROR (LiDARcloud::loadASCIIFile): y-coordinate not specified for hit point #" + std::to_string(hit_count) + " of scan #" + std::to_string(scanID) ) );
        }else if( temp_xyz.z==-9999 ){
            throw( std::runtime_error("ERROR (LiDARcloud::loadASCIIFile): z-coordinate not specified for hit point #" + std::to_string(hit_count) + " of scan #" + std::to_string(scanID) ) );
        }

        //direction
        if( temp_direction.elevation==-9999 || temp_direction.azimuth==-9999 ){
            temp_direction = cart2sphere( temp_xyz - scandata.origin );
        }

        //add hit point to the scan
        addHitPoint( scanID, temp_xyz, temp_direction, temp_rgb, data );

        hit_count++;

    }

    datafile.close();

    return hit_count;


}

void LiDARcloud::exportTriangleNormals( const char* filename ){

    ofstream file;

    file.open(filename);

    if( !file.is_open() ) {
        throw (std::runtime_error("ERROR (LiDARcloud::exportTriangleNormals): Could not open file '" + std::string(filename) + "' for writing."));
    }

    for( std::size_t t=0; t<triangles.size(); t++ ){

        Triangulation tri = triangles.at(t);

        vec3 v0 = tri.vertex0;
        vec3 v1 = tri.vertex1;
        vec3 v2 = tri.vertex2;

        vec3 normal = cross( v1-v0, v2-v0 );
        normal.normalize();

        file << normal.x << " " << normal.y << " " << normal.z << std::endl;

    }

    file.close();

}

void LiDARcloud::exportTriangleNormals( const char* filename, int gridcell ){

    ofstream file;

    file.open(filename);

    if( !file.is_open() ) {
        throw (std::runtime_error("ERROR (LiDARcloud::exportTriangleNormals): Could not open file '" + std::string(filename) + "' for writing."));
    }

    for( std::size_t t=0; t<triangles.size(); t++ ){

        Triangulation tri = triangles.at(t);

        if( tri.gridcell == gridcell ){

            vec3 v0 = tri.vertex0;
            vec3 v1 = tri.vertex1;
            vec3 v2 = tri.vertex2;

            vec3 normal = cross( v1-v0, v2-v0 );
            normal.normalize();

            file << normal.x << " " << normal.y << " " << normal.z << std::endl;

        }

    }

    file.close();


}

void LiDARcloud::exportTriangleAreas( const char* filename ){

    ofstream file;

    file.open(filename);

    if( !file.is_open() ) {
        throw (std::runtime_error("ERROR (LiDARcloud::exportTriangleAreas): Could not open file '" + std::string(filename) + "' for writing."));
    }

    for( std::size_t t=0; t<triangles.size(); t++ ){

        Triangulation tri = triangles.at(t);

        file << tri.area << std::endl;

    }

    file.close();

}

void LiDARcloud::exportTriangleAreas( const char* filename, int gridcell ){

    ofstream file;

    file.open(filename);

    if( !file.is_open() ) {
        throw (std::runtime_error("ERROR (LiDARcloud::exportTriangleAreas): Could not open file '" + std::string(filename) + "' for writing."));
    }

    for( std::size_t t=0; t<triangles.size(); t++ ){

        Triangulation tri = triangles.at(t);

        if( tri.gridcell == gridcell ){

            file << tri.area << std::endl;

        }

    }

    file.close();

}

void LiDARcloud::exportTriangleInclinationDistribution( const char* filename, uint Nbins ){

    std::vector<std::vector<float>> inclinations( getGridCellCount() );
    for( int i=0; i<getGridCellCount(); i++ ){
        inclinations.at(i).resize( Nbins );
    }

    float db = 0.5f*M_PI/float(Nbins); //bin width

    for( std::size_t t=0; t<triangles.size(); t++ ){

        Triangulation tri = triangles.at(t);

        int cell = tri.gridcell;

        if( cell<0 ){
            continue;
        }

        vec3 v0 = tri.vertex0;
        vec3 v1 = tri.vertex1;
        vec3 v2 = tri.vertex2;

        vec3 normal = cross( v1-v0, v2-v0 );
        normal.normalize();

        float angle = acos_safe(fabs(normal.z));

        float area = tri.area;

        uint bin = floor(angle/db);
        if( bin>=Nbins ){
            bin = Nbins-1;
        }

        inclinations.at(cell).at(bin) += area;

    }

    ofstream file;

    file.open(filename);

    if( !file.is_open() ){
        throw( std::runtime_error("ERROR (LiDARcloud::exportTriangleInclinationDistribution): Could not open file '" + std::string(filename) + "' for writing.") );
    }

    for( int cell=0; cell<getGridCellCount(); cell++ ){
        for( int bin=0; bin<Nbins; bin++ ) {
            file << inclinations.at(cell).at(bin) << " ";
        }
        file << std::endl;
    }

    file.close();

}

void LiDARcloud::exportLeafAreas( const char* filename ){

    ofstream file;

    file.open(filename);

    if( !file.is_open() ) {
        throw (std::runtime_error("ERROR (LiDARcloud::exportLeafAreas): Could not open file '" + std::string(filename) + "' for writing."));
    }

    for( uint i=0; i<getGridCellCount(); i++ ){

        file << getCellLeafArea(i) << std::endl;

    }

    file.close();

}

void LiDARcloud::exportLeafAreaDensities( const char* filename ){

    ofstream file;

    file.open(filename);

    if( !file.is_open() ) {
        throw (std::runtime_error("ERROR (LiDARcloud::exportLeafAreaDensities): Could not open file '" + std::string(filename) + "' for writing."));
    }

    for( uint i=0; i<getGridCellCount(); i++ ){

        file << getCellLeafAreaDensity(i) << std::endl;

    }

    file.close();

}

void LiDARcloud::exportGtheta( const char* filename ){

    ofstream file;

    file.open(filename);

    if( !file.is_open() ) {
        throw (std::runtime_error("ERROR (LiDARcloud::exportGtheta): Could not open file '" + std::string(filename) + "' for writing."));
    }

    for( uint i=0; i<getGridCellCount(); i++ ){

        file << getCellGtheta(i) << std::endl;

    }

    file.close();

}

void LiDARcloud::exportPointCloud( const char* filename ){

    if(getScanCount() ==1 ){
        exportPointCloud( filename, 0 );
    }else{

        for( int i=0; i< getScanCount(); i++ ){

            std::string filename_a = filename;
            char scan[20];
            sprintf(scan,"%d",i);

            size_t dotindex = filename_a.find_last_of(".");
            if( dotindex == filename_a.size()-1 || filename_a.size()-1-dotindex>4  ){//no file extension was provided
                filename_a = filename_a + "_" + scan;
            }else{ //has file extension
                std::string ext = filename_a.substr(dotindex,filename_a.size()-1);
                filename_a = filename_a.substr(0,dotindex) + "_" + scan + ext;
            }

            exportPointCloud( filename_a.c_str(), i );

        }

    }
}


void LiDARcloud::exportPointCloud( const char* filename, uint scanID ){

    if( scanID> getScanCount()){
        std::cerr << "ERROR (LiDARcloud::exportPointCloud): Cannot export scan " << scanID << " because this scan does not exist." << std::endl;
        throw 1;
    }

    ofstream file;

    file.open(filename);

    if( !file.is_open() ) {
        throw (std::runtime_error("ERROR (LiDARcloud::exportPointCloud): Could not open file '" + std::string(filename) + "' for writing."));
    }

    std::vector<std::string> hit_data;
    for( int r=0; r< getHitCount(); r++ ){
        std::map<std::string,double> data = hits.at(r).data;
        for( std::map<std::string,double>::iterator iter=data.begin(); iter!=data.end(); ++iter ){
            std::vector<std::string>::iterator it = find(hit_data.begin(),hit_data.end(),iter->first);
            if( it==hit_data.end() ){
                hit_data.push_back(iter->first);
            }
        }
    }

    std::vector<std::string> ASCII_format = getScanColumnFormat(scanID);

    if( ASCII_format.size()==0 ){
        ASCII_format.push_back("x");
        ASCII_format.push_back("y");
        ASCII_format.push_back("z");
    }

    for( int r=0; r< getHitCount(); r++ ){

        if( getHitScanID(r) != scanID ){
            continue;
        }

        vec3 xyz = getHitXYZ(r);
        RGBcolor color = getHitColor(r);

        for( int c=0; c<ASCII_format.size(); c++ ){

            if( ASCII_format.at(c).compare("x")==0 ){
                file << xyz.x;
            }else if( ASCII_format.at(c).compare("y")==0 ){
                file << xyz.y;
            }else if( ASCII_format.at(c).compare("z")==0 ){
                file << xyz.z;
            }else if( ASCII_format.at(c).compare("r")==0 ){
                file << color.r;
            }else if( ASCII_format.at(c).compare("g")==0 ){
                file << color.g;
            }else if( ASCII_format.at(c).compare("b")==0 ){
                file << color.b;
            }else if( ASCII_format.at(c).compare("r255")==0 ){
                file << round(color.r*255);
            }else if( ASCII_format.at(c).compare("g255")==0 ){
                file << round(color.g*255);
            }else if( ASCII_format.at(c).compare("b255")==0 ){
                file << round(color.b*255);
            }else if( ASCII_format.at(c).compare("zenith")==0 ){
                file << getHitRaydir(r).zenith;
            }else if( ASCII_format.at(c).compare("azimuth")==0 ){
                file << getHitRaydir(r).azimuth;
            }else if( hits.at(r).data.find(ASCII_format.at(c))!=hits.at(r).data.end() ){ //hit scalar data
                file << getHitData(r,ASCII_format.at(c).c_str());
            }else{
                file << -9999;
            }

            if( c<ASCII_format.size()-1 ){
                file << " ";
            }

        }

        file << std::endl;

    }

    file.close();

}

void LiDARcloud::exportPointCloudPTX( const char* filename, uint scanID ){

    if( scanID> getScanCount()){
        std::cerr << "ERROR (LiDARcloud::exportPointCloudPTX): Cannot export scan " << scanID << " because this scan does not exist." << std::endl;
        throw 1;
    }

    ofstream file;

    file.open(filename);

    if( !file.is_open() ) {
        throw (std::runtime_error("ERROR (LiDARcloud::exportPointCloudPTX): Could not open file '" + std::string(filename) + "' for writing."));
    }

    std::vector<std::string> ASCII_format = getScanColumnFormat(scanID);

    uint Nx = getScanSizeTheta(scanID);
    uint Ny = getScanSizePhi(scanID);

    file << Nx << std::endl;
    file << Ny << std::endl;
    file << "0 0 0" << std::endl;
    file << "1 0 0" << std::endl;
    file << "0 1 0" << std::endl;
    file << "0 0 1" << std::endl;
    file << "1 0 0 0" << std::endl;
    file << "0 1 0 0" << std::endl;
    file << "0 0 1 0" << std::endl;
    file << "0 0 0 1" << std::endl;

    std::vector<std::vector<vec4> > xyzi(Ny);
    for( int j=0; j<Ny; j++ ){
        xyzi.at(j).resize(Nx);
        for( int i=0; i<Nx; i++ ){
            xyzi.at(j).at(i) = make_vec4(0,0,0,1);
        }
    }

    vec3 origin = getScanOrigin(scanID);

    for( int r=0; r<getHitCount(); r++ ){

        if( getHitScanID(r) != scanID ){
            continue;
        }

        SphericalCoord raydir = getHitRaydir(r);

        int2 row_column = scans.at(scanID).direction2rc(raydir);

        assert( row_column.x>=0 && row_column.x<Nx && row_column.y>=0 && row_column.y<Ny );

        vec3 xyz = getHitXYZ(r);

        if( (xyz-origin).magnitude()>=1e4 ){
            continue;
        }

        float intensity = 1.f;
        if( hits.at(r).data.find("intensity")!=hits.at(r).data.end() ) {
            intensity = getHitData(r, "intensity");
        }

        xyzi.at(row_column.y).at(row_column.x) = make_vec4( xyz.x, xyz.y, xyz.z, intensity );

    }

    for( int j=0; j<Ny; j++ ){
        for( int i=0; i<Nx; i++ ){
            file << xyzi.at(j).at(i).x << " " << xyzi.at(j).at(i).y << " " << xyzi.at(j).at(i).z << " " << xyzi.at(j).at(i).w << std::endl;
        }
    }

    file.close();

}

