/** \file "fileIO.cpp" Declarations for Aerial LiDAR plug-in related to file input/output.

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "AerialLiDAR.h"
#include "pugixml.hpp"

using namespace helios;
using namespace std;

void AerialLiDARcloud::loadXML( const char* filename ){

  if( printmessages ){
    cout << "Reading XML file: " << filename << "..." << flush;
  }
    
  //Check if file exists
  ifstream f(filename);
  if( !f.good() ){
    cerr << "failed.\n XML file does not exist." << endl;
    exit(EXIT_FAILURE);
  }

  // Using "pugixml" parser.  See pugixml.org
  pugi::xml_document xmldoc;

  //load file
  pugi::xml_parse_result result = xmldoc.load_file(filename);

  //error checking
  if (!result){
    cout << "failed." << endl;
    cerr << "XML  file " << filename << " parsed with errors, attribute value: [" << xmldoc.child("node").attribute("attr").value() << "]\n";
    cerr << "Error description: " << result.description() << "\n";
    //cerr << "Error offset: " << result.offset << " (error at [..." << (filename + result.offset) << "]\n\n";
    exit(EXIT_FAILURE);
  }

  pugi::xml_node helios = xmldoc.child("helios");

  if( helios.empty() ){
    std::cout << "failed." << std::endl;
    std::cerr << "ERROR (AerialLiDARcloud::loadXML): XML file must have tag '<helios> ... </helios>' bounding all other tags." << std::endl;
    exit(EXIT_FAILURE);
  }

  //-------------- Scans ---------------//

  uint scan_count = 0; //counter variable for scans
  uint total_hits = 0;

  //looping over any scans specified in XML file
  for (pugi::xml_node s = helios.child("scan"); s; s = s.next_sibling("scan")){

    // ----- scan center ------//
    const char* center_str = s.child_value("center");

    vec3 center;
    if( strlen(center_str)==0 ){
      center = helios::make_vec3(0,0,0);
    }else{
      center = string2vec3( center_str ); //note: pugi loads xml data as a character.  need to separate it into 3 floats
    }
      
    // ----- scan extent ------//
    const char* extent_str = s.child_value("extent");

    helios::vec2 extent;
    if( strlen(extent_str)==0 ){
      extent = helios::make_vec2(0,0);
    }else{
      extent = string2vec2( extent_str ); //note: pugi loads xml data as a character.  need to separate it into 2 floats
    }
    if( extent.x<0 || extent.y<0 ){
      cerr << "failed.\nERROR (AerialLiDARcloud::loadXML): The scan extent must be positive (check scan #" << scan_count << ")." << endl;
      exit(EXIT_FAILURE);
    }

    // ----- cone angle ------//
    const char* cone_str = s.child_value("coneangle");

    float coneAngle;
    if( strlen(cone_str)==0 ){
      coneAngle = 0;
    }else{
      coneAngle = atof(cone_str)*M_PI/180.f;
    }

    if( coneAngle<0 ){
      cerr << "ERROR (AerialLiDARcloud::loadXML): cone angle cannot be less than 0." << endl;
      exit(EXIT_FAILURE);
    }else if( coneAngle>90 ){
      cerr << "ERROR (AerialLiDARcloud::loadXML): cone angle cannot be greater than 90 degrees." << endl;
      exit(EXIT_FAILURE);
    }

    // ----- scan density ------//
    const char* density_str = s.child_value("scandensity");

    float scanDensity;
    if( strlen(density_str)==0 ){
      scanDensity = 0;
    }else{
      scanDensity = atof(density_str);

      if( scanDensity<=0 ){
	cerr << "ERROR (AerialLiDARcloud::loadXML): scan density should be greater than 0." << endl;
	exit(EXIT_FAILURE);
      }
      
    }

    // ----- exitDiameter ------//
    const char* exitDiameter_str = s.child_value("exitdiameter");

    float exitDiameter;
    if( strlen(exitDiameter_str)==0 ){
      exitDiameter=0;
    }else{
      exitDiameter = fmax(0,atof(exitDiameter_str));
    }

    // ----- beamDivergence ------//
    const char* beamDivergence_str = s.child_value("beamdivergence");

    float beamDivergence;
    if( strlen(beamDivergence_str)==0 ){
      beamDivergence=0;
    }else{
      beamDivergence = fmax(0,atof(beamDivergence_str));
    }
       
    //create a temporary scan object
    AerialScanMetadata scan(center,extent,coneAngle,scanDensity,exitDiameter,beamDivergence);

    addScan( scan );

    uint scanID = getScanCount()-1;

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
	  cout << "failed.\n ERROR (AerialLiDARcloud::loadXML): Data file `" << str << "' given for scan #" << scan_count << " does not exist." << endl;
	  exit(EXIT_FAILURE);
	}
	f.close();
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

      //add hit points to scan if data file was given
    
      ifstream datafile(str); //open the file
  
      if(!datafile.is_open()){ //check that file exists
	cout << "failed." << endl;
	cerr << "ERROR (loadXML): data file does not exist." << endl;
	exit(EXIT_FAILURE);
      }
      
      vec3 temp_xyz;
      SphericalCoord temp_direction;
      RGBcolor temp_rgb;
      float temp_data;
      std::map<std::string, float> data;
      int direction_flag = 0;

      std::size_t hit_count = 0;
      while ( datafile.good() ){ //loop through file to read scan data

	hit_count++;

	temp_xyz = make_vec3(-9999,-9999,-9999);
	temp_rgb = make_RGBcolor(1,0,0); //default color: red
//	temp_direction = make_SphericalCoord(-0.5*M_PI,0); //default direction: vertical (downward)
    float temp_zenith = M_PI;
    float temp_azimuth = 0;

	for( uint i=0; i<column_format.size(); i++ ){
	  if( column_format.at(i).compare("zenith")==0 ){
	    datafile >> temp_zenith;
        temp_zenith = deg2rad(temp_zenith);
	  }else if( column_format.at(i).compare("azimuth")==0 ){
	    datafile >> temp_azimuth;
        temp_azimuth = deg2rad(temp_azimuth);
	  }else if( column_format.at(i).compare("zenith_rad")==0 ){
	    datafile >> temp_zenith;
	  }else if( column_format.at(i).compare("azimuth_rad")==0 ){
	    datafile >> temp_azimuth;
	  }else if( column_format.at(i).compare("x")==0 ){
	    datafile >> temp_xyz.x;
	  }else if( column_format.at(i).compare("y")==0 ){
	    datafile >> temp_xyz.y;
	  }else if( column_format.at(i).compare("z")==0 ){
	    datafile >> temp_xyz.z;
	  }else if( column_format.at(i).compare("r")==0 ){
	    datafile >> temp_rgb.r;
	  }else if( column_format.at(i).compare("g")==0 ){
	    datafile >> temp_rgb.g;
	  }else if( column_format.at(i).compare("b")==0 ){
	    datafile >> temp_rgb.b;
	  }else if( column_format.at(i).compare("r255")==0 ){
	    datafile >> temp_rgb.r;
	    temp_rgb.r/=255.f;
	  }else if( column_format.at(i).compare("g255")==0 ){
	    datafile >> temp_rgb.g;
	    temp_rgb.g/=255.f;
	  }else if( column_format.at(i).compare("b255")==0 ){
	    datafile >> temp_rgb.b;
	    temp_rgb.b/=255.f;
	  }else{ //assume that rest is floating point data
	    datafile >> temp_data;
	    data[ column_format.at(i) ] = temp_data; 
	  }
	}
	
	if( !datafile.good() ){//if the whole line was not read successfully, stop
	  if( hit_count==1 ){
	    std::cerr << "WARNING: Something is likely wrong with the data file " << filename << ". Check that the format is consistent with that specified in the XML metadata file." << std::endl;
	  }
	  break;
	}
      	
	// -- Checks to make sure everything was specified correctly -- //

	//hit point
	if( temp_xyz.x==-9999 ){
	  std::cerr << "ERROR (AerialLiDARcloud::loadXML): x-coordinate not specified for hit point #" << hit_count-1 << " of scan #" << scan_count << std::endl;
	  exit(EXIT_FAILURE);
	}else if( temp_xyz.y==-9999 ){
	  std::cerr << "ERROR (AerialLiDARcloud::loadXML): t-coordinate not specified for hit point #" << hit_count-1 << " of scan #" << scan_count << std::endl;
	  exit(EXIT_FAILURE);
	}else if( temp_xyz.z==-9999 ){
	  std::cerr << "ERROR (AerialLiDARcloud::loadXML): z-coordinate not specified for hit point #" << hit_count-1 << " of scan #" << scan_count << std::endl;
	  exit(EXIT_FAILURE);
	}

    SphericalCoord temp_direction(1.f, 0.5f*M_PI - temp_zenith, temp_azimuth);

    //add hit point to the scan
	addHitPoint( scanID, temp_xyz, temp_direction, temp_rgb, data );
	
	total_hits ++;
      
      }
    
      datafile.close();

    }
      
    //coordinateShift( scanID, translation );
    //scan.coordinateRotation( rotation_sphere );

    scan_count ++;
    
  }

  //------------ Grid ------------//

  uint cell_count = 0; //counter variable for scans

  //looping over any grids specified in XML file
  for (pugi::xml_node s = helios.child("grid"); s; s = s.next_sibling("grid")){

    // ----- grid center ------//
    const char* center_str = s.child_value("center");
    
    if( strlen(center_str)==0 ){
      cerr << "failed.\nERROR (AerialLiDARcloud::loadXML): A center was not specified for grid #" << cell_count << endl;
      exit(EXIT_FAILURE);
    }
    
    gridcenter = string2vec3( center_str ); //note: pugi loads xml data as a character.  need to separate it into 3 floats

    // ----- grid size ------//
    const char* gsize_str = s.child_value("size");
    
    if( strlen(gsize_str)==0 ){
      cerr << "failed.\nERROR (AerialLiDARcloud::loadXML): A size was not specified for grid cell #" << cell_count << endl;
      exit(EXIT_FAILURE);
    }

    gridextent = string2vec3( gsize_str ); //note: pugi loads xml data as a character.  need to separate it into 3 floats

    if( gridextent.x<=0 || gridextent.y<=0 || gridextent.z<=0 ){
      cerr << "failed.\nERROR (AerialLiDARcloud::loadXML): The grid size/extent must be positive." << endl;
      exit(EXIT_FAILURE);
    }
    
    // ----- grid rotation ------//
    const char* grot_str = s.child_value("rotation");
    
    if( strlen(grot_str)==0 ){
      gridrotation = 0; //if no rotation specified, assume = 0
    }else{
      gridrotation = atof(grot_str);
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
      cerr << "failed.\nERROR (AerialLiDARcloud::loadXML): The number of gridcells must be positive." << endl;
      exit(EXIT_FAILURE);
    }
    
    const char* Ny_str = s.child_value("Ny");
    
    if( strlen(Ny_str)==0 ){ //If no Ny specified, assume Ny=1;
      Ny=1;
    }else{
      Ny = atof(Ny_str);
    }
    if( Ny<=0 ){
      cerr << "failed.\nERROR (loadXML): The number of gridcells must be positive." << endl;
      exit(EXIT_FAILURE);
    }

    const char* Nz_str = s.child_value("Nz");
    
    if( strlen(Nz_str)==0 ){ //If no Nz specified, assume Nz=1;
      Nz=1;
    }else{
      Nz = atof(Nz_str);
    }
    if( Nz<=0 ){
      cerr << "failed.\nERROR (AerialLiDARcloud::loadXML): The number of gridcells must be positive." << endl;
      exit(EXIT_FAILURE);
    }

    gridresolution = make_int3(Nx,Ny,Nz);


  }

  if( printmessages ){
    
    cout << "done." << endl;
  
    cout << "Successfully read " << getScanCount() << " scan(s), which contain " << total_hits << " total hit points." << endl;

  }
    
}
