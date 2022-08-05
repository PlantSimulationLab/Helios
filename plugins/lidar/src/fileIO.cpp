/** \file "fileIO.cpp" Declarations for LiDAR plug-in related to file input/output. 
    \author Brian Bailey

    Copyright (C) 2016-2022 Brian Bailey

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

void LiDARcloud::loadXML( const char* filename, const bool load_grid_only ){

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
    std::cerr << "ERROR (loadXML): XML file must have tag '<helios> ... </helios>' bounding all other tags." << std::endl;
    exit(EXIT_FAILURE);
  }

  //-------------- Scans ---------------//

  uint scan_count = 0; //counter variable for scans
  uint total_hits = 0;

  if(load_grid_only == false){

    //looping over any scans specified in XML file
    for (pugi::xml_node s = helios.child("scan"); s; s = s.next_sibling("scan")){

      // ----- scan origin ------//
      const char* origin_str = s.child_value("origin");
    
      if( strlen(origin_str)==0 ){
	cerr << "failed.\nERROR (loadXML): An origin was not specified for scan #" << scan_count << endl;
	exit(EXIT_FAILURE);
      }
    
      vec3 origin = string2vec3( origin_str ); //note: pugi loads xml data as a character.  need to separate it into 3 floats
    
      // ----- scan size (resolution) ------//
      const char* size_str = s.child_value("size");

      helios::int2 size;
      if( strlen(size_str)==0 ){
	cerr << "failed.\nERROR (loadXML): A size was not specified for scan #" << scan_count << endl;
	exit(EXIT_FAILURE);
      }else{
	size = string2int2( size_str ); //note: pugi loads xml data as a character.  need to separate it into 2 ints
      }
      if( size.x<=0 || size.y<=0 ){
	cerr << "failed.\nERROR (loadXML): The scan size must be positive (check scan #" << scan_count << ")." << endl;
	exit(EXIT_FAILURE);
      }

      // ----- scan translation ------//
      const char* offset_str = s.child_value("translation");
    
      vec3 translation = make_vec3(0,0,0);
      if( strlen(offset_str)>0 ){
	translation = string2vec3( offset_str ); //note: pugi loads xml data as a character.  need to separate it into 3 floats
      }
      
      // ----- scan rotation ------//
      const char* rotation_str = s.child_value("rotation");
      
      SphericalCoord rotation_sphere;
      if( strlen(rotation_str)>0 ){
	vec3 rotation = string2vec3( rotation_str ); //note: pugi loads xml data as a character.  need to separate it into 3 floats
	rotation = rotation*M_PI/180.f;
	rotation_sphere = cart2sphere(rotation);
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
	cerr << "ERROR (loadXML): thetaMin cannot be less than 0." << endl;
	exit(EXIT_FAILURE);
      }
      
      // ----- thetaMax ------//
      const char* thetaMax_str = s.child_value("thetaMax");
      
      float thetaMax;
      if( strlen(thetaMax_str)==0 ){
	//cerr << "WARNING (loadXML): A maximum zenithal scan angle was not specified for scan #" << scan_count << "...assuming thetaMax = PI rad." << flush;
	thetaMax=M_PI;
      }else{
	thetaMax = atof(thetaMax_str)*M_PI/180.f;
      }
      
      if( thetaMax-1e-5>M_PI ){
	cerr << "ERROR (loadXML): thetaMax cannot be greater than 180 degrees." << endl;
	exit(EXIT_FAILURE);
      }
      
      // ----- phiMin ------//
      const char* phiMin_str = s.child_value("phiMin");
      
      float phiMin;
      if( strlen(phiMin_str)==0 ){
	//cerr << "WARNING (loadXML): A minimum azimuthal scan angle was not specified for scan #" << scan_count << "...assuming phiMin = 0." << flush;
	phiMin=0.f;
      }else{
	phiMin = atof(phiMin_str)*M_PI/180.f;
      }
      
      if( phiMin<0 ){
	cerr << "ERROR (loadXML): phiMin cannot be less than 0." << endl;
	exit(EXIT_FAILURE);
      }
      
      // ----- phiMax ------//
      const char* phiMax_str = s.child_value("phiMax");
      
      float phiMax;
      if( strlen(phiMax_str)==0 ){
	//cerr << "WARNING (loadXML): A maximum azimuthal scan angle was not specified for scan #" << scan_count << "...assuming phiMax = 2PI rad." << flush;
	phiMax=2.f*M_PI;
      }else{
	phiMax = atof(phiMax_str)*M_PI/180.f;
      }
      
      if( phiMax-1e-5>4.f*M_PI ){
	cerr << "ERROR (loadXML): phiMax cannot be greater than 720 degrees." << endl;
	exit(EXIT_FAILURE);
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
	    cout << "failed.\n ERROR (loadXML): Data file `" << str << "' given for scan #" << scan_count << " does not exist." << endl;
	    exit(EXIT_FAILURE);
	  }
	  f.close();
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
	float temp_row, temp_column;
	double temp_data;
	std::map<std::string, double> data;
	int direction_flag = 0;

	vector<unsigned int> row, column;
	std::size_t hit_count = 0;
	while ( datafile.good() ){ //loop through file to read scan data
	  
	  hit_count++;
	  
	  temp_xyz = make_vec3(-9999,-9999,-9999);
	  temp_rgb = make_RGBcolor(1,0,0); //default color: red
	  temp_row = -1;
	  temp_column = -1;
	  temp_direction = make_SphericalCoord(-9999,-9999);
	  
	  for( uint i=0; i<column_format.size(); i++ ){
	    if( column_format.at(i).compare("row")==0 ){
	      datafile >> temp_row;
	    }else if( column_format.at(i).compare("column")==0 ){
	      datafile >> temp_column;
	    }else if( column_format.at(i).compare("zenith")==0 ){
	      datafile >> temp_direction.zenith;
	    }else if( column_format.at(i).compare("azimuth")==0 ){
	      datafile >> temp_direction.azimuth;
	    }else if( column_format.at(i).compare("zenith_rad")==0 ){
	      datafile >> temp_direction.zenith;
	      temp_direction.zenith *= 180.f/M_PI;
	    }else if( column_format.at(i).compare("azimuth_rad")==0 ){
	      datafile >> temp_direction.azimuth;
	      temp_direction.azimuth *= 180.f/M_PI;
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
	    }else{ //assume that rest is data
	      datafile >> temp_data;
	      data[ column_format.at(i) ] = temp_data;
	    }
	  }
	  
	  if( !datafile.good() ){//if the whole line was not read successfully, stop
	    if( hit_count==1 ){
	      std::cerr << "WARNING: Something is likely wrong with the data file " << filename << ". Check that the format is consisten with that specified in the XML metadata file." << std::endl;
	    }
	    break;
	  }
	
	  // -- Checks to make sure everything was specified correctly -- //

	  //hit point
	  if( temp_xyz.x==-9999 ){
	    std::cerr << "ERROR (loadXML): x-coordinate not specified for hit point #" << hit_count-1 << " of scan #" << scan_count << std::endl;
	    exit(EXIT_FAILURE);
	  }else if( temp_xyz.y==-9999 ){
	    std::cerr << "ERROR (loadXML): t-coordinate not specified for hit point #" << hit_count-1 << " of scan #" << scan_count << std::endl;
	    exit(EXIT_FAILURE);
	  }else if( temp_xyz.z==-9999 ){
	    std::cerr << "ERROR (loadXML): z-coordinate not specified for hit point #" << hit_count-1 << " of scan #" << scan_count << std::endl;
	    exit(EXIT_FAILURE);
	  }

	  //add hit point to the scan
	  addHitPoint( scanID, temp_xyz, temp_direction, temp_rgb, data );

	  total_hits ++;
      
	}
    
	datafile.close();

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
      cerr << "failed.\nERROR (loadXML): A center was not specified for grid #" << cell_count << endl;
      exit(EXIT_FAILURE);
    }
    
    vec3 center = string2vec3( center_str ); //note: pugi loads xml data as a character.  need to separate it into 3 floats

    // ----- grid size ------//
    const char* gsize_str = s.child_value("size");
    
    if( strlen(gsize_str)==0 ){
      cerr << "failed.\nERROR (loadXML): A size was not specified for grid cell #" << cell_count << endl;
      exit(EXIT_FAILURE);
    }

    vec3 gsize = string2vec3( gsize_str ); //note: pugi loads xml data as a character.  need to separate it into 3 floats

    if( gsize.x<=0 || gsize.y<=0 || gsize.z<=0 ){
      cerr << "failed.\nERROR (loadXML): The gridcell size must be positive." << endl;
      exit(EXIT_FAILURE);
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
      cerr << "failed.\nERROR (loadXML): The number of gridcells must be positive." << endl;
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
      cerr << "failed.\nERROR (loadXML): The number of gridcells must be positive." << endl;
      exit(EXIT_FAILURE);
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

void LiDARcloud::exportTriangleNormals( const char* filename ){

  ofstream file;

  file.open(filename);
  
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

void LiDARcloud::exportTriangleNormals( const char* filename, const int gridcell ){

  ofstream file;

  file.open(filename);
  
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
  
  for( std::size_t t=0; t<triangles.size(); t++ ){

    Triangulation tri = triangles.at(t);

    file << tri.area << std::endl;
    
  }

  file.close();
  
}

void LiDARcloud::exportTriangleAreas( const char* filename, const int gridcell ){

  ofstream file;

  file.open(filename);
  
  for( std::size_t t=0; t<triangles.size(); t++ ){

    Triangulation tri = triangles.at(t);

    if( tri.gridcell == gridcell ){

      file << tri.area << std::endl;

    }
    
  }

  file.close();
  
}

void LiDARcloud::exportLeafAreas( const char* filename ){

  ofstream file;

  file.open(filename);
  
  for( uint i=0; i<getGridCellCount(); i++ ){

    file << getCellLeafArea(i) << std::endl;

  }

  file.close();
  
}

void LiDARcloud::exportLeafAreaDensities( const char* filename ){

  ofstream file;

  file.open(filename);
  
  for( uint i=0; i<getGridCellCount(); i++ ){

    file << getCellLeafAreaDensity(i) << std::endl;

  }

  file.close();
  
}

void LiDARcloud::exportGtheta( const char* filename ){

  ofstream file;

  file.open(filename);
  
  for( uint i=0; i<getGridCellCount(); i++ ){

    file << getCellGtheta(i) << std::endl;

  }

  file.close();
  
}

void LiDARcloud::exportPointCloud( const char* filename ){
  
  if( getScanCount()==1 ){
    exportPointCloud( filename, 0 );
  }else{

    for( int i=0; i<getScanCount(); i++ ){

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


void LiDARcloud::exportPointCloud( const char* filename, const uint scanID ){

  if( scanID>getScanCount() ){
    std::cerr << "ERROR (LiDARcloud::exportPointCloud): Cannot export scan " << scanID << " because this scan does not exist." << std::endl;
    throw 1;
  }

  ofstream file;

  file.open(filename);

  std::vector<std::string> hit_data;
  for( int r=0; r<getHitCount(); r++ ){
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

  for( int r=0; r<getHitCount(); r++ ){

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
	file << -9999 << std::endl;
      }

      if( c<ASCII_format.size()-1 ){
	file << " ";
      }

    }

    file << std::endl;
    
  }

  file.close();
  
}

