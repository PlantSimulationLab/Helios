/** \file "RadiationModel.cpp" Primary source file for radiation transport model.

    Copyright (C) 2016-2023  Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "RadiationModel.h"

using namespace helios;

RadiationModel::RadiationModel( helios::Context* context_a ){

    context = context_a;

    // All default values set here

    message_flag = true;

    rho_default = 0.f;
    tau_default = 0.f;
    eps_default = 1.f;

    kappa_default = 1.f;
    sigmas_default = 0.f;

    temperature_default = 300;

    periodic_flag = make_vec2(0,0);

    initializeOptiX();

}

RadiationModel::~RadiationModel(){
    RT_CHECK_ERROR( rtContextDestroy( OptiX_Context ) );
}

void RadiationModel::disableMessages() {
    message_flag=false;
}

void RadiationModel::enableMessages() {
    message_flag=true;
}

void RadiationModel::setDirectRayCount( const std::string &label, size_t N ){
    if( !doesBandExist(label) ){
        helios_runtime_error("ERROR (RadiationModel::setDirectRayCount): Cannot set ray count for band " + label + " because it is not a valid band.");
    }
    radiation_bands.at(label).directRayCount = N;
}

void RadiationModel::setDiffuseRayCount( const std::string &label, size_t N ){
    if( !doesBandExist(label) ){
        helios_runtime_error("ERROR (RadiationModel::setDiffuseRayCount): Cannot set ray count for band " + label + " because it is not a valid band." );
    }
    radiation_bands.at(label).diffuseRayCount = N;
}

void RadiationModel::setDiffuseRadiationFlux( const std::string &label, float flux ){
    if( !doesBandExist(label) ){
        helios_runtime_error("ERROR (RadiationModel::setDiffuseRadiationFlux): Cannot set flux value for band " + label + " because it is not a valid band." );
    }
    radiation_bands.at(label).diffuseFlux = flux;
}

void RadiationModel::setDiffuseRadiationExtinctionCoeff(const std::string &label, float K, const SphericalCoord &peak_dir ){
    setDiffuseRadiationExtinctionCoeff( label,K,sphere2cart(peak_dir) );
}

void RadiationModel::setDiffuseRadiationExtinctionCoeff( const std::string &label, float K, const vec3 &peak_dir ){
    if( !doesBandExist(label) ){
        helios_runtime_error("ERROR (RadiationModel::setDiffuseRadiationExtinctionCoeff): Cannot set diffuse extinction value for band " + label + " because it is not a valid band.");
    }

    vec3 dir = peak_dir;
    dir.normalize();

    int N=100;
    float norm = 0.f;
    for( int j=0; j<N; j++ ){
        for( int i=0; i<N; i++ ){
            float theta = 0.5f*M_PI/float(N)*(0.5f+float(i));
            float phi = 2.f*M_PI/float(N)*(0.5f+float(j));
            vec3 n = sphere2cart( make_SphericalCoord(0.5f*M_PI-theta,phi) );

            float psi = acos_safe( n*dir );
            float fd;
            if( psi<M_PI/180.f ){
                fd = powf(M_PI/180.f,-K);
            }else{
                fd = powf(psi,-K);
            }

            norm += fd*cosf(theta)*sinf(theta)*M_PI/float(N*N);
            //note: the multipication factors are dtheta*dphi/pi = (0.5*pi/N)*(2*pi/N)/pi = pi/N^2
        }
    }

    radiation_bands.at(label).diffuseExtinction = K;
    radiation_bands.at(label).diffusePeakDir = dir;
    radiation_bands.at(label).diffuseDistNorm = 1.f/norm;

}

void RadiationModel::addRadiationBand(const std::string &label ){

    if( radiation_bands.find(label)!=radiation_bands.end() ){
        std::cerr << "ERROR (RadiationModel::addRadiationBand): Radiation band " << label << " has already been added. Skipping this call to addRadiationBand()." << std::endl;
        return;
    }

    RadiationBand band(label);

    radiation_bands.emplace(label,band);

    //Initialize all radiation source fluxes to zero
    for( auto &source : radiation_sources ) {
        source.source_fluxes[label] = 0.f;
    }

}

void RadiationModel::addRadiationBand( const std::string &label, float wavelength1, float wavelength2 ){

    if( radiation_bands.find(label)!=radiation_bands.end() ){
        std::cerr << "ERROR (RadiationModel::addRadiationBand): Radiation band " << label << " has already been added. Skipping this call to addRadiationBand()." << std::endl;
        return;
    }else if( wavelength1>wavelength2 ){
        helios_runtime_error("ERROR (RadiationModel::addRadiationBand): The upper wavelength bound for a band must be greater than the lower bound.");
    }else if( wavelength2-wavelength1<1 ){
        helios_runtime_error("ERROR (RadiationModel::addRadiationBand): The waveband range of a radiation band must be at least 1 nm.");

    }

    RadiationBand band(label);

    band.wavebandBounds = make_vec2(wavelength1,wavelength2);

    radiation_bands.emplace(label,band);

    //Initialize all radiation source fluxes to zero
    for( auto &source : radiation_sources ) {
        source.source_fluxes[label] = 0.f;
    }





}

void RadiationModel::copyRadiationBand( const std::string &old_label, const std::string &new_label ) {

    if( !doesBandExist(old_label) ){
        helios_runtime_error("ERROR (RadiationModel::copyRadiationBand): Cannot copy band " + old_label + " because it does not exist.");
    }

    vec2 waveBounds = radiation_bands.at(old_label).wavebandBounds;

    copyRadiationBand( old_label, new_label, waveBounds.x, waveBounds.y );
}

void RadiationModel::copyRadiationBand(const std::string &old_label, const std::string &new_label, float wavelength_min, float wavelength_max ){

    if( !doesBandExist(old_label) ){
        helios_runtime_error("ERROR (RadiationModel::copyRadiationBand): Cannot copy band " + old_label + " because it does not exist.");
    }

    RadiationBand band = radiation_bands.at(old_label);
    band.label = new_label;
    band.wavebandBounds = make_vec2(wavelength_min,wavelength_max);

    radiation_bands.emplace(new_label,band);

    //copy source fluxes
    for( auto &source : radiation_sources ) {
        source.source_fluxes[new_label] = source.source_fluxes.at(old_label);
    }

}

bool RadiationModel::doesBandExist( const std::string &label ) const{
    if( radiation_bands.find(label)==radiation_bands.end() ){
        return false;
    }else{
        return true;
    }
}

void RadiationModel::disableEmission( const std::string &label ){

    if( !doesBandExist(label) ){
        helios_runtime_error("ERROR (RadiationModel::disableEmission): Cannot disable emission for band " + label + " because it is not a valid band." );
    }

    radiation_bands.at(label).emissionFlag = false;

}

void RadiationModel::enableEmission( const std::string &label ){

    if( !doesBandExist(label) ){
        helios_runtime_error("ERROR (RadiationModel::enableEmission): Cannot disable emission for band " + label + " because it is not a valid band." );
    }

    radiation_bands.at(label).emissionFlag = true;

}

uint RadiationModel::addCollimatedRadiationSource() {

    return addCollimatedRadiationSource( make_vec3(0,0,1) );

}

uint RadiationModel::addCollimatedRadiationSource(const SphericalCoord &direction ){
    return addCollimatedRadiationSource( sphere2cart(direction) );
}

uint RadiationModel::addCollimatedRadiationSource(const vec3 &direction ){

    if( direction.magnitude()==0 ){
        helios_runtime_error("ERROR (RadiationModel::addCollimatedRadiationSource): Invalid collimated source direction. Direction vector should not have length of zero.");
    }

    uint Nsources = radiation_sources.size()+1;
    if( Nsources>256 ){
        helios_runtime_error("ERROR (RadiationModel::addCollimatedRadiationSource): A maximum of 256 radiation sources are allowed.");
    }

    RadiationSource collimated_source( direction );

    //initialize fluxes to 0
    for( const auto &band : radiation_bands ) {
        collimated_source.source_fluxes[band.first] = 0.f;
    }

    radiation_sources.emplace_back(collimated_source);

    return Nsources-1;

}

uint RadiationModel::addSphereRadiationSource( const vec3 &position, float radius ){

    if( radius<=0 ){
        helios_runtime_error("ERROR (RadiationModel::addSphereRadiationSource): Spherical radiation source radius must be positive.");
    }

    uint Nsources = radiation_sources.size()+1;
    if( Nsources>256 ){
        helios_runtime_error("ERROR (RadiationModel::addSphereRadiationSource): A maximum of 256 radiation sources are allowed.");
    }

    RadiationSource sphere_source( position, 2.f*fabsf(radius) );

    //initialize fluxes to 0
    for( const auto &band : radiation_bands ) {
        sphere_source.source_fluxes[band.first] = 0.f;
    }

    radiation_sources.emplace_back( sphere_source );

    return Nsources-1;

}

void RadiationModel::addSphereRadiationSource(const uint &sourceID, const helios::vec3 &position, float radius){
    // Replace the existing radiation source with ID of "sourceID"
    if( radius<=0 ){
        helios_runtime_error("ERROR (RadiationModel::addSphereRadiationSource): Spherical radiation source radius must be positive.");
    }

    if( sourceID>256 ){
        helios_runtime_error("ERROR (RadiationModel::addSphereRadiationSource): A maximum of 256 radiation sources are allowed.");
    }

    RadiationSource sphere_source( position, 2.f*fabsf(radius) );

    //initialize fluxes to 0
    for( const auto &band : radiation_bands ) {
        sphere_source.source_fluxes[band.first] = 0.f;
    }

    radiation_sources.at(sourceID) = sphere_source;
}

uint RadiationModel::addSunSphereRadiationSource() {
    return addSunSphereRadiationSource( make_vec3(0,0,1) );
}

uint RadiationModel::addSunSphereRadiationSource(const SphericalCoord &sun_direction ){
    return addSunSphereRadiationSource( sphere2cart(sun_direction) );
}

uint RadiationModel::addSunSphereRadiationSource(const vec3 &sun_direction ){

    uint Nsources = radiation_sources.size()+1;
    if( Nsources>256 ){
        helios_runtime_error("ERROR (RadiationModel::addSunSphereRadiationSource): A maximum of 256 radiation sources are allowed.");
    }

    RadiationSource sphere_source( 150e9*sun_direction/sun_direction.magnitude(), 150e9, 2.f*695.5e6, sigma*powf(5700,4)/1288.437f );

    //initialize fluxes to 0
    for( const auto &band : radiation_bands ) {
        sphere_source.source_fluxes[band.first] = 0.f;
    }

    radiation_sources.emplace_back( sphere_source );

    return Nsources-1;

}

uint RadiationModel::addRectangleRadiationSource( const vec3 &position, const vec2 &size, const vec3& rotation ){

    if( size.x<=0 || size.y<=0 ){
        helios_runtime_error("ERROR (RadiationModel::addRectangleRadiationSource): Radiation source size must be positive.");
    }

    uint Nsources = radiation_sources.size()+1;
    if( Nsources>256 ){
        helios_runtime_error("ERROR (RadiationModel::addRectangleRadiationSource): A maximum of 256 radiation sources are allowed.");
    }

    RadiationSource rectangle_source( position, size, rotation );

    //initialize fluxes to 0
    for( const auto &band : radiation_bands ) {
        rectangle_source.source_fluxes[band.first] = 0.f;
    }

    radiation_sources.emplace_back( rectangle_source );

    return Nsources-1;

}

uint RadiationModel::addDiskRadiationSource( const vec3 &position, float radius, const vec3& rotation ){

    if( radius<=0 ){
        helios_runtime_error("ERROR (RadiationModel::addDiskRadiationSource): Disk radiation source radius must be positive.");
    }

    uint Nsources = radiation_sources.size()+1;
    if( Nsources>256 ){
        helios_runtime_error("ERROR (RadiationModel::addDiskRadiationSource): A maximum of 256 radiation sources are allowed.");
    }

    RadiationSource disk_source( position, radius, rotation );

    //initialize fluxes to 0
    for( const auto &band : radiation_bands ) {
        disk_source.source_fluxes[band.first] = 0.f;
    }

    radiation_sources.emplace_back( disk_source );

    return Nsources-1;


}

void RadiationModel::setSourceSpectrumIntegral(uint source_ID, float source_integral){

    if( source_ID >= radiation_sources.size() ){
        helios_runtime_error("ERROR (RadiationModel::setSourceSpectrumIntegral): Source ID out of bounds. Only " + std::to_string(radiation_sources.size()-1) + " radiation sources have been created." );
    }else if( source_integral<0 ){
        helios_runtime_error("ERROR (RadiationModel::setSourceIntegral): Source integral must be non-negative." );
    }

    radiation_sources.at(source_ID).source_integral = source_integral;

}

void RadiationModel::setSourceSpectrumIntegral(uint source_ID, float source_integral, float wavelength1, float wavelength2){

    if( source_ID >= radiation_sources.size() ){
        helios_runtime_error("ERROR (RadiationModel::setSourceSpectrumIntegral): Source ID out of bounds. Only " + std::to_string(radiation_sources.size()-1) + " radiation sources have been created." );
    }else if( source_integral<0 ){
        helios_runtime_error("ERROR (RadiationModel::setSourceSpectrumIntegral): Source integral must be non-negative." );
    }else if( radiation_sources.at(source_ID).source_spectrum.empty() ){
        std::cout << "WARNING (RadiationModel::setSourceSpectrumIntegral): Spectral distribution has not been set for radiation source. Cannot set its integral." << std::endl;
        return;
    }

    RadiationSource &source = radiation_sources.at(source_ID);

    float old_integral = integrateSpectrum( source.source_spectrum, wavelength1, wavelength2 );

    for( vec2 &wavelength : source.source_spectrum ){
        wavelength.y *= source_integral/old_integral;
    }

    radiation_sources.at(source_ID).source_integral = source_integral;

}

void RadiationModel::setSourceFlux( uint source_ID, const std::string &label, float flux ){

    if( !doesBandExist(label) ){
        helios_runtime_error( "ERROR (RadiationModel::setSourceFlux): Cannot add radiation source for band " + label + " because it is not a valid band.");
    }else if( source_ID >= radiation_sources.size() ){
        helios_runtime_error("ERROR (RadiationModel::setSourceFlux): Source ID out of bounds. Only " + std::to_string(radiation_sources.size()-1) + " radiation sources have been created." );
    }else if( flux<0 ){
        helios_runtime_error("ERROR (RadiationModel::setSourceFlux): Source flux must be non-negative." );
    }

    radiation_sources.at(source_ID).source_fluxes[label] = flux * radiation_sources.at(source_ID).source_flux_scaling_factor;

}

void RadiationModel::setSourceFlux(const std::vector<uint> &source_ID, const std::string &band_label, float flux ){
    for( auto ID : source_ID ){
        setSourceFlux( ID, band_label, flux );
    }
}

float RadiationModel::getSourceFlux( uint source_ID, const std::string &label )const{
    if( !doesBandExist(label) ){
        helios_runtime_error( "ERROR (RadiationModel::getSourceFlux): Cannot add radiation source for band " + label + " because it is not a valid band.");
    }else if( source_ID >= radiation_sources.size() ){
        helios_runtime_error("ERROR (RadiationModel::setSourceFlux): Source ID out of bounds. Only " + std::to_string(radiation_sources.size()-1) + " radiation sources have been created." );
    }else if( radiation_sources.at(source_ID).source_fluxes.find(label) == radiation_sources.at(source_ID).source_fluxes.end() ) {
        helios_runtime_error("ERROR (RadiationModel::getSourceFlux): Cannot get flux for source #" + std::to_string(source_ID) + " because radiative band " + label + " does not exist.");
    }

    if( radiation_sources.at(source_ID).source_fluxes.find(label) == radiation_sources.at(source_ID).source_fluxes.end() ) {
        helios_runtime_error("ERROR (RadiationModel::getSourceFlux): Cannot get flux for source #" + std::to_string(source_ID) +
                " because radiative band " + label + " does not exist.");
    }else {
        return radiation_sources.at(source_ID).source_fluxes.at(label);
    }
    return 0;
}

void RadiationModel::setSourceSpectrum( uint source_ID, const std::vector<helios::vec2> &spectrum ){

    if( source_ID >= radiation_sources.size() ){
        helios_runtime_error( "ERROR (RadiationModel::setSourceSpectrum): Cannot add radiation spectra for this source because it is not a valid radiation source ID.\n");
    }

    //validate spectrum
    for( auto s=0; s<spectrum.size(); s++ ) {
        //check that wavelengths are monotonic
        if( s>0 ) {
            if( spectrum.at(s).x<=spectrum.at(s-1).x ) {
                helios_runtime_error("ERROR (RadiationModel::setSourceSpectrum): Source spectral data validation failed. Wavelengths must increase monotonically.");
            }
        }
        //check that wavelength is within a reasonable range
        if( spectrum.at(s).x<0 || spectrum.at(s).x>100000 ) {
            helios_runtime_error("ERROR (RadiationModel::setSourceSpectrum): Source spectral data validation failed. Wavelength value of " + std::to_string(spectrum.at(s).x) + " appears to be erroneous.");
        }
        //check that flux is non-negative
        if( spectrum.at(s).y<0 ) {
            helios_runtime_error("ERROR (RadiationModel::setSourceSpectrum): Source spectral data validation failed. Flux value at wavelength of " + std::to_string(spectrum.at(s).x) + " appears is negative.");
        }
    }

    RadiationSource &source = radiation_sources.at(source_ID);

    source.source_spectrum = spectrum;

    //set the flux for all current bands based on the spectrum's integral



}

void RadiationModel::setSourceSpectrum(const std::vector<uint> &source_ID, const std::vector<helios::vec2> &spectrum ) {
    for (auto ID: source_ID) {
        setSourceSpectrum(ID, spectrum);
    }
}

void RadiationModel::setSourceSpectrum(uint source_ID, const std::string &spectrum_label ){

    if( source_ID >= radiation_sources.size() ){
        helios_runtime_error( "ERROR (RadiationModel::setSourceSpectrum): Cannot add radiation spectra for this source because it is not a valid radiation source ID.\n");
    }else if( spectrum_label == "ASTMG173" || spectrum_label == "solar_spectrum_ASTMG173" ) {
        context->loadXML("plugins/radiation/spectral_data/solar_spectrum_ASTMG173.xml", true);
        if( !context->doesGlobalDataExist("solar_spectrum_ASTMG173") ){
            helios_runtime_error( "ERROR (RadiationModel::setSourceSpectrum): The file ""plugins/radiation/spectral_data/solar_spectrum_ASTMG173.xml"" could not be loaded.");
        }else {
            std::vector<vec2> spectrum;
            context->getGlobalData("solar_spectrum_ASTMG173",spectrum);
            radiation_sources.at(source_ID).source_spectrum = spectrum;
            radiation_sources.at(source_ID).source_spectrum_label = spectrum_label;
        }
    }else{
        if( !context->doesGlobalDataExist(spectrum_label.c_str()) ) {
            helios_runtime_error( "ERROR (RadiationModel::setSourceSpectrum): Global data " + std::string(spectrum_label) + " does not exist and thus source spectrum cannot be set.");
        }else{
            std::vector<vec2> spectrum;
            context->getGlobalData(spectrum_label.c_str(),spectrum);
            radiation_sources.at(source_ID).source_spectrum = spectrum;
            radiation_sources.at(source_ID).source_spectrum_label = spectrum_label;
        }
    }

}

void RadiationModel::setSourceSpectrum(const std::vector<uint> &source_ID, const std::string &spectrum_label ){
    for (auto ID: source_ID) {
        setSourceSpectrum(ID, spectrum_label);
    }
}

float RadiationModel::integrateSpectrum( uint source_ID, const std::vector<helios::vec2> &object_spectrum, float wavelength1, float wavelength2 ) const{

    if( source_ID >= radiation_sources.size() ){
        helios_runtime_error( "ERROR (RadiationModel::integrateSpectrum): Radiation spectrum was not set for source ID. Make sure to set its spectrum using setSourceSpectrum() function.");
    }else if( object_spectrum.size()<2 ){
        helios_runtime_error( "ERROR (RadiationModel::integrateSpectrum): Radiation spectrum must have at least 2 wavelengths.");
    }else if( wavelength1>wavelength2 || wavelength1==wavelength2 ){
        helios_runtime_error( "ERROR (RadiationModel::integrateSpectrum): Lower wavelength bound must be less than the upper wavelength bound." );
    }

    std::vector<helios::vec2> source_spectrum = radiation_sources.at(source_ID).source_spectrum;

    int istart = 0;
    int iend = (int)object_spectrum.size()-1;
    for( auto i=0; i<object_spectrum.size()-1; i++ ){

        if( object_spectrum.at(i).x<=wavelength1 && object_spectrum.at(i+1).x>wavelength1 ){
            istart = i;
        }
        if( object_spectrum.at(i).x<=wavelength2 && object_spectrum.at(i+1).x>wavelength2 ){
            iend = i+1;
            break;
        }

    }

    float E = 0;
    float Etot=0;
    if (radiation_sources.at(source_ID).source_integral!=0){
        Etot = radiation_sources.at(source_ID).source_integral;
    }
    for( auto i=istart; i<iend; i++ ){

        float x0 = object_spectrum.at(i).x;
        float Esource0 = interp1( source_spectrum, object_spectrum.at(i).x);
        float Eobject0 = object_spectrum.at(i).y;

        float x1 = object_spectrum.at(i+1).x;
        float Eobject1 = object_spectrum.at(i+1).y;
        float Esource1 = interp1( source_spectrum, object_spectrum.at(i+1).x);

        E += 0.5f*(Eobject0*Esource0+Eobject1*Esource1)*(x1-x0);
        if (radiation_sources.at(source_ID).source_integral==0){
            Etot += 0.5f*(Esource1+Esource0)*(x1-x0);
        }
    }

    return E/Etot;

}

float RadiationModel::integrateSpectrum( const std::vector<helios::vec2> &object_spectrum, float wavelength1, float wavelength2 ) const{

    if( object_spectrum.size()<2 ){
        helios_runtime_error( "ERROR (RadiationModel::integrateSpectrum): Radiation spectrum must have at least 2 wavelengths.");
    }else if( wavelength1>wavelength2 || wavelength1==wavelength2 ){
        helios_runtime_error( "ERROR (RadiationModel::integrateSpectrum): Lower wavelength bound must be less than the upper wavelength bound." );
    }

    int istart = 1;
    int iend = (int)object_spectrum.size()-1;
    for( auto i=0; i<object_spectrum.size()-1; i++ ){

        if( object_spectrum.at(i).x<=wavelength1 && object_spectrum.at(i+1).x>wavelength1 ){
            istart = i;
        }
        if( object_spectrum.at(i).x<=wavelength2 && object_spectrum.at(i+1).x>wavelength2 ){
            iend = i+1;
            break;
        }

    }

    float E = 0;
    for( auto i=istart; i<iend; i++ ){
        float E0 = object_spectrum.at(i).y;
        float x0 = object_spectrum.at(i).x;
        float E1 = object_spectrum.at(i+1).y;
        float x1 = object_spectrum.at(i+1).x;
        E += (E0+E1)*(x1-x0)*0.5f;
    }

    return E;

}

float RadiationModel::integrateSpectrum( const std::vector<helios::vec2> &object_spectrum) const{
    float wavelength1 = object_spectrum.at(0).x;
    float wavelength2 = object_spectrum.at(object_spectrum.size()-1).x;
    float E = RadiationModel::integrateSpectrum(object_spectrum, wavelength1, wavelength2 );
    return E;
}

float RadiationModel::integrateSpectrum(uint source_ID, const std::vector<helios::vec2> &object_spectrum, const std::vector<helios::vec2> &camera_spectrum ) const{

    if( source_ID >= radiation_sources.size() ){
        helios_runtime_error( "ERROR (RadiationModel::integrateSpectrum): Radiation spectrum was not set for source ID. Make sure to set its spectrum using setSourceSpectrum() function.");
    }else if( object_spectrum.size()<2 ) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Radiation spectrum must have at least 2 wavelengths.");
    }

    std::vector<helios::vec2> source_spectrum = radiation_sources.at(source_ID).source_spectrum;

    float E = 0;
    float Etot=0;
    if (radiation_sources.at(source_ID).source_integral!=0){
        Etot = radiation_sources.at(source_ID).source_integral;
    }

    for( auto i=1; i<object_spectrum.size(); i++ ){

        if( object_spectrum.at(i).x <= source_spectrum.front().x || object_spectrum.at(i).x <= camera_spectrum.front().x ){
            continue;
        }
        if( object_spectrum.at(i).x > source_spectrum.back().x || object_spectrum.at(i).x > camera_spectrum.back().x ){
            break;
        }
        float x1 = object_spectrum.at(i).x;
        float Eobject1 = object_spectrum.at(i).y;
        float Esource1 = interp1( source_spectrum, x1);
        float Ecamera1 = interp1( camera_spectrum, x1);


        float x0 = object_spectrum.at(i-1).x;
        float Eobject0 = object_spectrum.at(i-1).y;
        float Esource0 = interp1( source_spectrum, x0);
        float Ecamera0 = interp1( camera_spectrum, x0);

        E += 0.5f*((Eobject1*Esource1*Ecamera1)+(Eobject0*Ecamera0*Esource0))*(x1-x0);
        if (radiation_sources.at(source_ID).source_integral==0){
            Etot += 0.5f*(Esource1+Esource0)*(x1-x0);
        }
    }


    return E/Etot;

}

float RadiationModel::integrateSpectrum( const std::vector<helios::vec2> &object_spectrum, const std::vector<helios::vec2> &camera_spectrum ) const{

    if( object_spectrum.size()<2 ) {
        helios_runtime_error("ERROR (RadiationModel::integrateSpectrum): Radiation spectrum must have at least 2 wavelengths.");
    }

    float E = 0;
    float Etot = 0;
    for( auto i=0; i<object_spectrum.size(); i++ ){

        if( object_spectrum.at(i).x <= camera_spectrum.front().x ){
            continue;
        }
        if( object_spectrum.at(i).x > camera_spectrum.back().x ){
            break;
        }

        float x1 = object_spectrum.at(i).x;
        float Eobject1 = object_spectrum.at(i).y;
        float Ecamera1 = interp1( camera_spectrum, x1);


        float x0 = object_spectrum.at(i-1).x;
        float Eobject0 = object_spectrum.at(i-1).y;
        float Ecamera0 = interp1( camera_spectrum, x0);

        E += 0.5f*((Eobject1*Ecamera1)+(Eobject0*Ecamera0))*(x1-x0);
        Etot += 0.5f*(Ecamera1+Ecamera0)*(x1-x0);

    }

    return E/Etot;

}

void RadiationModel::setSourcePosition( uint source_ID, const vec3 &position ){

    if( source_ID >= radiation_sources.size() ){
        helios_runtime_error("ERROR (RadiationModel::setSourcePosition): Source ID out of bounds. Only " + std::to_string(radiation_sources.size()-1) + " radiation sources.");
    }

    if( radiation_sources.at(source_ID).source_type == RADIATION_SOURCE_TYPE_SPHERE ) {
        radiation_sources.at(source_ID).source_position = position * radiation_sources.at(source_ID).source_position_scaling_factor;
    }else {
        radiation_sources.at(source_ID).source_position = position/position.magnitude();
    }

}

helios::vec3 RadiationModel::getSourcePosition( uint source_ID )const {
    if( source_ID >= radiation_sources.size() ){
        helios_runtime_error("ERROR (RadiationModel::getSourcePosition): Source ID does not exist.");
    }
    return radiation_sources.at(source_ID).source_position;
}

void RadiationModel::setScatteringDepth( const std::string &label, uint depth ){

    if( !doesBandExist(label) ){
        helios_runtime_error( "ERROR (RadiationModel::setScatteringDepth): Cannot set scattering depth for band " + label + " because it is not a valid band.");
    }
    radiation_bands.at(label).scatteringDepth = depth;

}

void RadiationModel::setMinScatterEnergy(const std::string &label, uint energy ){

    if( !doesBandExist(label) ){
        helios_runtime_error( "ERROR (setMinScatterEnergy): Cannot set minimum scattering energy for band " + label + " because it is not a valid band.");
    }
    radiation_bands.at(label).minScatterEnergy = energy;

}

void RadiationModel::enforcePeriodicBoundary(const std::string &boundary ){

    if( boundary=="x" ){

        periodic_flag.x = 1;

    }else if( boundary=="y" ){

        periodic_flag.y = 1;

    }else if( boundary=="xy" ){

        periodic_flag.x = 1;
        periodic_flag.y = 1;

    }else{

        std::cout << "WARNING (RadiationModel::enforcePeriodicBoundary()): unknown boundary of """ << boundary << """. Possible choices are ""x"", ""y"", or ""xy""." << std::endl;

    }

}

void RadiationModel::addRadiationCamera(const std::string &camera_label, const std::vector<std::string> &band_label, const helios::vec3 &position, const helios::vec3 &lookat, float lens_diameter, const helios::vec2 &sensor_size, float focal_plane_distance, float HFOV_degrees, uint antialiasing_samples, const helios::int2 &resolution) {

    if( sensor_size.x<=0 || sensor_size.y<=0 ){
        helios_runtime_error("ERROR (RadiationModel::addRadiationCamera): Sensor size must be greater than 0.");
    }else if( antialiasing_samples==0 ){
        helios_runtime_error("ERROR (RadiationModel::addRadiationCamera): The model requires at least 1 antialiasing sample to run.");
    }else if( resolution.x<=0 || resolution.y<=0 ){
        helios_runtime_error("ERROR (RadiationModel::addRadiationCamera): Camera resolution must be at least 1x1.");
    }else if( HFOV_degrees<0 || HFOV_degrees>180.f ){
        helios_runtime_error("ERROR (RadiationModel::addRadiationCamera): Camera horzontal field of view must be between 0 and 180 degrees.");
    }

    RadiationCamera camera(camera_label, band_label, position, lookat, resolution, lens_diameter, sensor_size, focal_plane_distance, HFOV_degrees, antialiasing_samples );
    if(cameras.find(camera_label)==cameras.end()){
        cameras.emplace(camera_label,camera);
    }
    else{
        if( message_flag ){
            std::cout<<"Camera with label "<< camera_label <<"already exists. Existing properties will be replaced by new inputs."<<std::endl;
        }
        cameras.erase(camera_label);
        cameras.emplace(camera_label,camera);
    }

}

void RadiationModel::addRadiationCamera(const std::string &camera_label, const std::vector<std::string> &band_label, const helios::vec3 &position, const helios::vec3 &lookat, const CameraProperties &camera_properties, uint antialiasing_samples) {

    if( camera_properties.sensor_size.x<=0 || camera_properties.sensor_size.y<=0 ){
        helios_runtime_error("ERROR (RadiationModel::addRadiationCamera): Sensor size must be greater than 0.");
    }else if( antialiasing_samples==0 ){
        helios_runtime_error("ERROR (RadiationModel::addRadiationCamera): The model requires at least 1 antialiasing sample to run.");
    }else if( camera_properties.camera_resolution.x<=0 || camera_properties.camera_resolution.y<=0 ){
        helios_runtime_error("ERROR (RadiationModel::addRadiationCamera): Camera resolution must be at least 1x1.");
    }else if( camera_properties.HFOV<0 || camera_properties.HFOV>180.f ){
        helios_runtime_error("ERROR (RadiationModel::addRadiationCamera): Camera horizontal field of view must be between 0 and 180 degrees.");
    }

    RadiationCamera camera(camera_label, band_label, position, lookat, camera_properties.camera_resolution, camera_properties.lens_diameter, camera_properties.sensor_size, camera_properties.focal_plane_distance, camera_properties.HFOV, antialiasing_samples );
    if(cameras.find(camera_label)==cameras.end()){
        cameras.emplace(camera_label,camera);
    }
    else{
        if( message_flag ){
            std::cout<<"Camera with label "<< camera_label <<"already exists. Existing properties will be replaced by new inputs."<<std::endl;
        }
        cameras.erase(camera_label);
        cameras.emplace(camera_label,camera);
    }
}

void RadiationModel::addRadiationCamera(const std::string &camera_label, const std::vector<std::string> &band_label, const helios::vec3 &position, const helios::SphericalCoord &viewing_direction, const CameraProperties &camera_properties, uint antialiasing_samples) {

    vec3 lookat = position + sphere2cart(viewing_direction);
    addRadiationCamera( camera_label, band_label, position, lookat, camera_properties, antialiasing_samples );

}

void RadiationModel::setCameraSpectralResponse( const std::string &camera_label, const std::string &band_label, const std::string &global_data ){
    if(cameras.find(camera_label) == cameras.end() ){
        helios_runtime_error("ERROR (setCameraSpectralResponse): Camera """ + camera_label + """ does not exist.");
    }

    cameras.at(camera_label).band_spectral_response[band_label] = global_data;

}

void RadiationModel::setCameraPosition( const std::string &camera_label, const helios::vec3& position ){
    if(cameras.find(camera_label) == cameras.end() ){
        helios_runtime_error("ERROR (RadiationModel::setCameraPosition): Camera """ + camera_label + """ does not exist.");
    }

    cameras.at(camera_label).position = position;
}

void RadiationModel::setCameraLookat( const std::string &camera_label, const helios::vec3& lookat ){
    if(cameras.find(camera_label) == cameras.end() ){
        helios_runtime_error("ERROR (RadiationModel::setCameraLookat): Camera """ + camera_label + """ does not exist.");
    }

    cameras.at(camera_label).lookat = lookat;
}

void RadiationModel::setCameraOrientation( const std::string &camera_label, const helios::vec3& direction ){
    if(cameras.find(camera_label) == cameras.end() ){
        helios_runtime_error("ERROR (RadiationModel::setCameraOrientation): Camera """ + camera_label + """ does not exist.");
    }

    cameras.at(camera_label).lookat = cameras.at(camera_label).position + direction;
}

void RadiationModel::setCameraOrientation( const std::string &camera_label, const helios::SphericalCoord& direction ){
    if(cameras.find(camera_label) == cameras.end() ){
        helios_runtime_error("ERROR (RadiationModel::setCameraOrientation): Camera """ + camera_label + """ does not exist.");
    }

    cameras.at(camera_label).lookat = cameras.at(camera_label).position + sphere2cart(direction);
}

std::vector<std::string> RadiationModel::getAllCameraLabels(){
    std::vector<std::string> labels(cameras.size());
    uint cam=0;
    for( const auto &camera : cameras ){
        labels.at(cam) = camera.second.label;
        cam++;
    }
    return labels;
}

void RadiationModel::writeCameraImage(const std::string &camera, const std::vector<std::string> &bands, const std::string &imagefile_base, const std::string &image_path, int frame, float flux_to_pixel_conversion) {

    //check if camera exists
    if( cameras.find(camera)==cameras.end() ){
        std::cout << "ERROR (RadiationModel::writeCameraImage): camera with label " << camera << " does not exist. Skipping image write for this camera." << std::endl;
        return;
    }

    if( bands.size()!=1 && bands.size()!=3 ){
        std::cout << "ERROR (RadiationModel::writeCameraImage): input vector of band labels (""bands"") should either have length of 1 (grayscale image) or length of 3 (RGB image). Skipping image write for this camera." << std::endl;
        return;
    }

    std::vector<std::vector<float> > camera_data(bands.size());

    for (uint b = 0; b < bands.size(); b++) {

        //check if band exists
        if (std::find(cameras.at(camera).band_labels.begin(), cameras.at(camera).band_labels.end(), bands.at(b)) == cameras.at(camera).band_labels.end()) {
            std::cout << "ERROR (RadiationModel::writeCameraImage): camera " << camera << " band with label " << bands.at(b) << " does not exist. Skipping image write for this camera." << std::endl;
            return;
        }

        std::string global_data_label = "camera_" + camera + "_" + bands.at(b);

        if (!context->doesGlobalDataExist(global_data_label.c_str())) {
            std::cout << "ERROR (RadiationModel::writeCameraImage): image data for camera " << camera << ", band " << bands.at(b) << " has not been created. Did you run the radiation model? Skipping image write for this camera." << std::endl;
            return;
        }

        context->getGlobalData(global_data_label.c_str(), camera_data.at(b));

    }

    std::string frame_str;
    if( frame>=0 ) {
        frame_str = std::to_string(frame);
    }

    std::ostringstream outfile;

    if( image_path.find_last_of('/')==image_path.length()-1 ) {
        outfile << image_path;
    }else {
        outfile << image_path << "/";
    }

    if( frame>=0 ) {
        outfile << "camera_" << camera << "_" << imagefile_base << "_" << std::setw(5) << std::setfill('0') << frame_str << ".jpeg";
    }else{
        outfile << "camera_" << camera << "_" << imagefile_base << ".jpeg";
    }
    std::ofstream testfile(outfile.str());

    if( !testfile.is_open() ){
        std::cout << "ERROR (RadiationMode::writeCameraImage): image file " << outfile.str() << " could not be opened. Check that the path exists and that you have write permission. Skipping image write for this camera." << std::endl;
        return;
    }
    testfile.close();

    int2 camera_resolution = cameras.at(camera).resolution;

    std::vector<RGBcolor> pixel_data(camera_resolution.x*camera_resolution.y);

    RGBcolor pixel_color;
    for (uint j = 0; j < camera_resolution.y; j++) {
        for (uint i = 0; i < camera_resolution.x; i++) {
            if( camera_data.size()==1 ){
                float c = camera_data.front().at(j * camera_resolution.x + i);
                pixel_color = make_RGBcolor(c,c,c);
            }else{
                pixel_color = make_RGBcolor(camera_data.at(0).at(j * camera_resolution.x + i),camera_data.at(1).at(j * camera_resolution.x + i),camera_data.at(2).at(j * camera_resolution.x + i));
            }
            pixel_data.at(j * camera_resolution.x + i) = pixel_color;
        }
    }

    writeJPEG( outfile.str(), camera_resolution.x, camera_resolution.y, pixel_data );

}

void RadiationModel::writeNormCameraImage(const std::string &camera, const std::vector<std::string> &bands, const std::string &imagefile_base, const std::string &image_path, int frame){
    float maxval = 0;
    // Find maximum mean value over all bands
    for (const std::string& band: bands){
        float maxval_band = 0;
        std::string global_data_label = "camera_" + camera + "_" + band;
        std::vector<float> cameradata;
        context->getGlobalData(global_data_label.c_str(), cameradata);
        for (float val: cameradata){
            if (val > maxval){
                maxval_band = val;
            }
        }
        maxval += maxval_band/bands.size();
    }
    // Normalize all bands
    for (const std::string& band: bands) {
        std::string global_data_label = "camera_" + camera + "_" + band;
        std::vector<float> cameradata;
        context->getGlobalData(global_data_label.c_str(), cameradata);
        for (float &val: cameradata) {
            val = val / maxval;
        }
        context->setGlobalData(global_data_label.c_str(), HELIOS_TYPE_FLOAT, cameradata.size(), &cameradata[0]);
    }

    RadiationModel::writeCameraImage(camera, bands, imagefile_base, image_path, frame);
}

void RadiationModel::initializeOptiX() {

    /* Context */
    RT_CHECK_ERROR( rtContextCreate( &OptiX_Context ) );
    RT_CHECK_ERROR( rtContextSetPrintEnabled( OptiX_Context, 1 ) );

    RT_CHECK_ERROR( rtContextSetRayTypeCount( OptiX_Context, 4 ) );
    //ray types are:
    // 0: direct_ray_type
    // 1: diffuse_ray_type
    // 2: camera_ray_type
    // 3: pixel_label_ray_type

    RT_CHECK_ERROR( rtContextSetEntryPointCount( OptiX_Context, 4 ) );
    //ray entery points are
    // 0: direct_raygen
    // 1: diffuse_raygen
    // 2: camera_raygen
    // 3: pixel_label_raygen

    /* Ray Types */
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "direct_ray_type", &direct_ray_type_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet1ui( direct_ray_type_RTvariable, RAYTYPE_DIRECT ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "diffuse_ray_type", &diffuse_ray_type_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet1ui( diffuse_ray_type_RTvariable, RAYTYPE_DIFFUSE ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "camera_ray_type", &camera_ray_type_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet1ui( camera_ray_type_RTvariable, RAYTYPE_CAMERA ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "pixel_label_ray_type", &pixel_label_ray_type_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet1ui( pixel_label_ray_type_RTvariable, RAYTYPE_PIXEL_LABEL ) );

    /* Ray Generation Program */

    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayGeneration.cu.ptx", "direct_raygen", &direct_raygen ) );
    RT_CHECK_ERROR( rtContextSetRayGenerationProgram( OptiX_Context, RAYTYPE_DIRECT, direct_raygen ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayGeneration.cu.ptx", "diffuse_raygen", &diffuse_raygen ) );
    RT_CHECK_ERROR( rtContextSetRayGenerationProgram( OptiX_Context, RAYTYPE_DIFFUSE, diffuse_raygen ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayGeneration.cu.ptx", "camera_raygen", &camera_raygen ) );
    RT_CHECK_ERROR( rtContextSetRayGenerationProgram( OptiX_Context, RAYTYPE_CAMERA, camera_raygen ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayGeneration.cu.ptx", "pixel_label_raygen", &pixel_label_raygen ) );
    RT_CHECK_ERROR( rtContextSetRayGenerationProgram( OptiX_Context, RAYTYPE_PIXEL_LABEL, pixel_label_raygen ) );

    /* Declare Buffers and Variables */

    //primitive reflectivity buffer
    addBuffer( "rho", rho_RTbuffer, rho_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1 );
    //primitive transmissivity buffer
    addBuffer( "tau", tau_RTbuffer, tau_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1 );

    //primitive reflectivity buffer
    addBuffer( "rho_cam", rho_cam_RTbuffer, rho_cam_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1 );
    //primitive transmissivity buffer
    addBuffer( "tau_cam", tau_cam_RTbuffer, tau_cam_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1 );

    //primitive transformation matrix buffer
    addBuffer( "transform_matrix", transform_matrix_RTbuffer, transform_matrix_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 2 );

    //primitive type buffer
    addBuffer( "primitive_type", primitive_type_RTbuffer, primitive_type_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1 );

    //primitive area buffer
    addBuffer( "primitive_area", primitive_area_RTbuffer, primitive_area_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1 );

    //primitive UUID buffers
    addBuffer( "patch_UUID", patch_UUID_RTbuffer, patch_UUID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1 );
    addBuffer( "triangle_UUID", triangle_UUID_RTbuffer, triangle_UUID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1 );
    addBuffer( "disk_UUID", disk_UUID_RTbuffer, disk_UUID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1 );
    addBuffer( "tile_UUID", tile_UUID_RTbuffer, tile_UUID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1 );
    addBuffer( "voxel_UUID", voxel_UUID_RTbuffer, voxel_UUID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1 );

    //Object ID Buffer
    addBuffer( "objectID", objectID_RTbuffer, objectID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1 );

    //Primitive ID Buffer
    addBuffer( "primitiveID", primitiveID_RTbuffer, primitiveID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1 );

    //primitive two-sided flag buffer
    addBuffer( "twosided_flag", twosided_flag_RTbuffer, twosided_flag_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_BYTE, 1 );

    //patch buffers
    addBuffer( "patch_vertices", patch_vertices_RTbuffer, patch_vertices_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 2 );

    //triangle buffers
    addBuffer( "triangle_vertices", triangle_vertices_RTbuffer, triangle_vertices_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 2 );

    //disk buffers
    addBuffer( "disk_centers", disk_centers_RTbuffer, disk_centers_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 1 );
    addBuffer( "disk_radii", disk_radii_RTbuffer, disk_radii_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1 );
    addBuffer( "disk_normals", disk_normals_RTbuffer, disk_normals_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 1 );

    //tile buffers
    addBuffer( "tile_vertices", tile_vertices_RTbuffer, tile_vertices_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 2 );

    //voxel buffers
    addBuffer( "voxel_vertices", voxel_vertices_RTbuffer, voxel_vertices_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 2 );

    //object buffers
    addBuffer( "object_subdivisions", object_subdivisions_RTbuffer, object_subdivisions_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_INT2, 1 );

    //radiation energy rate data buffers
    // - in - //
    addBuffer( "radiation_in", radiation_in_RTbuffer, radiation_in_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1 );
    // - out,top - //
    addBuffer( "radiation_out_top", radiation_out_top_RTbuffer, radiation_out_top_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1 );
    // - out,bottom - //
    addBuffer( "radiation_out_bottom", radiation_out_bottom_RTbuffer, radiation_out_bottom_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1 );
    // - camera - //
    addBuffer( "radiation_in_camera", radiation_in_camera_RTbuffer, radiation_in_camera_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1 );
    addBuffer( "camera_pixel_label", camera_pixel_label_RTbuffer, camera_pixel_label_RTvariable, RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, 1 );

    //primitive scattering buffers
    // - top - //
    addBuffer( "scatter_buff_top", scatter_buff_top_RTbuffer, scatter_buff_top_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1 );
    // - bottom - //
    addBuffer( "scatter_buff_bottom", scatter_buff_bottom_RTbuffer, scatter_buff_bottom_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1 );

    //Energy absorbed by "sky"
    addBuffer( "Rsky", Rsky_RTbuffer, Rsky_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1 );

    //number of external radiation sources
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "Nsources", &Nsources_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet1ui( Nsources_RTvariable,0 ) );

    //External radiation source positions
    addBuffer( "source_positions", source_positions_RTbuffer, source_positions_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 1 );

    //External radiation source widths
    addBuffer( "source_widths", source_widths_RTbuffer, source_widths_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, 1 );

    //External radiation source rotations
    addBuffer( "source_rotations", source_rotations_RTbuffer, source_rotations_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 1 );

    //External radiation source types
    addBuffer( "source_types", source_types_RTbuffer, source_types_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1 );

    //External radiation source fluxes
    addBuffer( "source_fluxes", source_fluxes_RTbuffer, source_fluxes_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1 );

    //number of radiation bands
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "Nbands", &Nbands_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet1ui( Nbands_RTvariable,0 ) );

    //flag to disable launches for certain bands
    addBuffer( "band_launch_flag", band_launch_flag_RTbuffer, band_launch_flag_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_BYTE, 1 );

    //number of Context primitives
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "Nprimitives", &Nprimitives_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet1ui( Nprimitives_RTvariable,0 ) );

    //Flux of diffuse radiation
    addBuffer( "diffuse_flux", diffuse_flux_RTbuffer, diffuse_flux_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1 );

    //Diffuse distribution extinction coefficient of ambient diffuse radiation
    addBuffer( "diffuse_extinction", diffuse_extinction_RTbuffer, diffuse_extinction_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1 );

    //Direction of peak diffuse radiation
    addBuffer( "diffuse_peak_dir", diffuse_peak_dir_RTbuffer, diffuse_peak_dir_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 1 );

    //Diffuse distribution normalization factor
    addBuffer( "diffuse_dist_norm", diffuse_dist_norm_RTbuffer, diffuse_dist_norm_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1 );

    //Bounding sphere radius and center
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "bound_sphere_radius", &bound_sphere_radius_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet1f( bound_sphere_radius_RTvariable, 0.f ));
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "bound_sphere_center", &bound_sphere_center_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet3f( bound_sphere_center_RTvariable, 0.f, 0.f, 0.f ));

    //Bounding Box
    addBuffer( "bbox_UUID", bbox_UUID_RTbuffer, bbox_UUID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1 );
    addBuffer( "bbox_vertices", bbox_vertices_RTbuffer, bbox_vertices_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 2 );

    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "periodic_flag", &periodic_flag_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet2f( periodic_flag_RTvariable, 0.f, 0.f ));

    //Texture mask data
    RT_CHECK_ERROR( rtBufferCreate( OptiX_Context, RT_BUFFER_INPUT, &maskdata_RTbuffer ) );
    RT_CHECK_ERROR( rtBufferSetFormat( maskdata_RTbuffer, RT_FORMAT_BYTE ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "maskdata", &maskdata_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSetObject( maskdata_RTvariable, maskdata_RTbuffer ) );
    std::vector<std::vector<std::vector<bool> > > dummydata;
    initializeBuffer3D( maskdata_RTbuffer, dummydata );

    //Texture mask size
    addBuffer( "masksize", masksize_RTbuffer, masksize_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_INT2, 1 );

    //Texture mask ID
    addBuffer( "maskID", maskID_RTbuffer, maskID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_INT, 1 );

    //Texture u,v data
    addBuffer( "uvdata", uvdata_RTbuffer, uvdata_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, 2 );

    //Texture u,v ID
    addBuffer( "uvID", uvID_RTbuffer, uvID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_INT, 1 );

    // Radiation Camera Variables
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "camera_position", &camera_position_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet3f( camera_position_RTvariable, 0.f, 0.f, 0.f ));
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "camera_direction", &camera_direction_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet2f( camera_direction_RTvariable, 0.f, 0.f ));
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "camera_lens_diameter", &camera_lens_diameter_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet1f( camera_lens_diameter_RTvariable, 0.f ));
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "sensor_size", &sensor_size_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet2f( sensor_size_RTvariable, 0.f, 0.f ));
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "camera_focal_length", &camera_focal_length_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet1f( camera_focal_length_RTvariable, 0.f ));
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "camera_viewplane_length", &camera_viewplane_length_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet1f( camera_viewplane_length_RTvariable, 0.f ));
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "Ncameras", &Ncameras_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet1ui( Ncameras_RTvariable, 0 ));
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "camera_ID", &camera_ID_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet1ui( camera_ID_RTvariable, 0 ));

    // primitive scattering buffers (cameras)
    addBuffer( "scatter_buff_top_cam", scatter_buff_top_cam_RTbuffer, scatter_buff_top_cam_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1 );
    addBuffer( "scatter_buff_bottom_cam", scatter_buff_bottom_cam_RTbuffer, scatter_buff_bottom_cam_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1 );

    /* Hit Programs */
    RTprogram closest_hit_direct;
    RTprogram closest_hit_diffuse;
    RTprogram closest_hit_camera;
    RTprogram closest_hit_pixel_label;
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayHit.cu.ptx", "closest_hit_direct", &closest_hit_direct ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayHit.cu.ptx", "closest_hit_diffuse", &closest_hit_diffuse ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayHit.cu.ptx", "closest_hit_camera", &closest_hit_camera ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayHit.cu.ptx", "closest_hit_pixel_label", &closest_hit_pixel_label ) );

    /* Initialize Patch Geometry */

    RTprogram  patch_intersection_program;
    RTprogram  patch_bounding_box_program;

    RT_CHECK_ERROR( rtGeometryCreate( OptiX_Context, &patch ) );

    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "rectangle_bounds", &patch_bounding_box_program ) );
    RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( patch, patch_bounding_box_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "rectangle_intersect", &patch_intersection_program ) );
    RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( patch, patch_intersection_program ) );

    /* Create Patch Material */

    RT_CHECK_ERROR( rtMaterialCreate( OptiX_Context, &patch_material ) );

    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( patch_material, RAYTYPE_DIRECT, closest_hit_direct ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( patch_material, RAYTYPE_DIFFUSE, closest_hit_diffuse ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( patch_material, RAYTYPE_CAMERA, closest_hit_camera ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( patch_material, RAYTYPE_PIXEL_LABEL, closest_hit_pixel_label ) );

    /* Initialize Triangle Geometry */

    RTprogram  triangle_intersection_program;
    RTprogram  triangle_bounding_box_program;

    RT_CHECK_ERROR( rtGeometryCreate( OptiX_Context, &triangle ) );

    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "triangle_bounds", &triangle_bounding_box_program ) );
    RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( triangle, triangle_bounding_box_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "triangle_intersect", &triangle_intersection_program ) );
    RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( triangle, triangle_intersection_program ) );

    /* Create Triangle Material */

    RT_CHECK_ERROR( rtMaterialCreate( OptiX_Context, &triangle_material ) );

    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( triangle_material, RAYTYPE_DIRECT, closest_hit_direct ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( triangle_material, RAYTYPE_DIFFUSE, closest_hit_diffuse ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( triangle_material, RAYTYPE_CAMERA, closest_hit_camera ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( triangle_material, RAYTYPE_PIXEL_LABEL, closest_hit_pixel_label ) );

    /* Initialize Disk Geometry */

    RTprogram  disk_intersection_program;
    RTprogram  disk_bounding_box_program;

    RT_CHECK_ERROR( rtGeometryCreate( OptiX_Context, &disk ) );

    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "disk_bounds", &disk_bounding_box_program ) );
    RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( disk, disk_bounding_box_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "disk_intersect", &disk_intersection_program ) );
    RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( disk, disk_intersection_program ) );

    /* Create Disk Material */

    RT_CHECK_ERROR( rtMaterialCreate( OptiX_Context, &disk_material ) );

    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( disk_material, RAYTYPE_DIRECT, closest_hit_direct ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( disk_material, RAYTYPE_DIFFUSE, closest_hit_diffuse ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( disk_material, RAYTYPE_CAMERA, closest_hit_camera ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( disk_material, RAYTYPE_PIXEL_LABEL, closest_hit_pixel_label ) );

    /* Initialize Tile Geometry */

    RTprogram  tile_intersection_program;
    RTprogram  tile_bounding_box_program;

    RT_CHECK_ERROR( rtGeometryCreate( OptiX_Context, &tile ) );

    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "tile_bounds", &tile_bounding_box_program ) );
    RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( tile, tile_bounding_box_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "tile_intersect", &tile_intersection_program ) );
    RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( tile, tile_intersection_program ) );

    /* Create Tile Material */

    RT_CHECK_ERROR( rtMaterialCreate( OptiX_Context, &tile_material ) );

    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( tile_material, RAYTYPE_DIRECT, closest_hit_direct ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( tile_material, RAYTYPE_DIFFUSE, closest_hit_diffuse ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( tile_material, RAYTYPE_CAMERA, closest_hit_camera) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( tile_material, RAYTYPE_PIXEL_LABEL, closest_hit_pixel_label ) );

    /* Initialize Voxel Geometry */

    RTprogram  voxel_intersection_program;
    RTprogram  voxel_bounding_box_program;

    RT_CHECK_ERROR( rtGeometryCreate( OptiX_Context, &voxel ) );

    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "voxel_bounds", &voxel_bounding_box_program ) );
    RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( voxel, voxel_bounding_box_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "voxel_intersect", &voxel_intersection_program ) );
    RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( voxel, voxel_intersection_program ) );

    /* Create Voxel Material */

    RT_CHECK_ERROR( rtMaterialCreate( OptiX_Context, &voxel_material ) );

    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( voxel_material, RAYTYPE_DIRECT, closest_hit_direct ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( voxel_material, RAYTYPE_DIFFUSE, closest_hit_diffuse ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( voxel_material, RAYTYPE_CAMERA, closest_hit_camera ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( voxel_material, RAYTYPE_PIXEL_LABEL, closest_hit_pixel_label ) );

    /* Initialize Bounding Box Geometry */

    RTprogram  bbox_intersection_program;
    RTprogram  bbox_bounding_box_program;

    RT_CHECK_ERROR( rtGeometryCreate( OptiX_Context, &bbox ) );

    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "bbox_bounds", &bbox_bounding_box_program ) );
    RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( bbox, bbox_bounding_box_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx", "bbox_intersect", &bbox_intersection_program ) );
    RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( bbox, bbox_intersection_program ) );

    /* Create Bounding Box Material */

    RT_CHECK_ERROR( rtMaterialCreate( OptiX_Context, &bbox_material ) );

    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( bbox_material, RAYTYPE_DIRECT, closest_hit_direct ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( bbox_material, RAYTYPE_DIFFUSE, closest_hit_diffuse ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( bbox_material, RAYTYPE_CAMERA, closest_hit_camera ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( bbox_material, RAYTYPE_PIXEL_LABEL, closest_hit_pixel_label ) );

    /* Miss Program */
    RTprogram miss_program_direct;
    RTprogram miss_program_diffuse;
    RTprogram miss_program_camera;
    RTprogram miss_program_pixel_label;
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayHit.cu.ptx", "miss_direct", &miss_program_direct ) );
    RT_CHECK_ERROR( rtContextSetMissProgram( OptiX_Context, RAYTYPE_DIRECT, miss_program_direct ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayHit.cu.ptx", "miss_diffuse", &miss_program_diffuse ) );
    RT_CHECK_ERROR( rtContextSetMissProgram( OptiX_Context, RAYTYPE_DIFFUSE, miss_program_diffuse ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayHit.cu.ptx", "miss_camera", &miss_program_camera ) );
    RT_CHECK_ERROR( rtContextSetMissProgram( OptiX_Context, RAYTYPE_CAMERA, miss_program_camera ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( OptiX_Context, "plugins/radiation/cuda_compile_ptx_generated_rayHit.cu.ptx", "miss_pixel_label", &miss_program_pixel_label ) );
    RT_CHECK_ERROR( rtContextSetMissProgram( OptiX_Context, RAYTYPE_PIXEL_LABEL, miss_program_pixel_label ) );

    /* Create OptiX Geometry Structures */

    RTtransform transform;

    RTgeometrygroup geometry_group;

    RTgeometryinstance patch_instance;
    RTgeometryinstance triangle_instance;
    RTgeometryinstance disk_instance;
    RTgeometryinstance tile_instance;
    RTgeometryinstance voxel_instance;
    RTgeometryinstance bbox_instance;

    /* Create top level group and associated (dummy) acceleration */
    RT_CHECK_ERROR( rtGroupCreate( OptiX_Context, &top_level_group ) );
    RT_CHECK_ERROR( rtGroupSetChildCount( top_level_group, 1 ) );

    RT_CHECK_ERROR( rtAccelerationCreate( OptiX_Context, &top_level_acceleration ) );
    RT_CHECK_ERROR( rtAccelerationSetBuilder(top_level_acceleration,"NoAccel") );
    RT_CHECK_ERROR( rtAccelerationSetTraverser(top_level_acceleration,"NoAccel") );
    RT_CHECK_ERROR( rtGroupSetAcceleration( top_level_group, top_level_acceleration) );

    /* mark acceleration as dirty */
    RT_CHECK_ERROR( rtAccelerationMarkDirty( top_level_acceleration ) );

    /* Create transform node */
    RT_CHECK_ERROR( rtTransformCreate( OptiX_Context, &transform ) );
    float m[16];
    m[0] = 1.f; m[1] = 0; m[2] = 0; m[3] = 0;
    m[4] = 0.f; m[5] = 1.f; m[6] = 0; m[7] = 0;
    m[8] = 0.f; m[9] = 0; m[10] = 1.f; m[11] = 0;
    m[12] = 0.f; m[13] = 0; m[14] = 0; m[15] = 1.f;
    RT_CHECK_ERROR( rtTransformSetMatrix( transform, 0, m, nullptr ) );
    RT_CHECK_ERROR( rtGroupSetChild( top_level_group, 0, transform ) );

    /* Create geometry group and associated acceleration*/
    RT_CHECK_ERROR( rtGeometryGroupCreate( OptiX_Context, &geometry_group ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChildCount( geometry_group, 6 ) );
    RT_CHECK_ERROR( rtTransformSetChild( transform, geometry_group ) );

    //create acceleration object for group and specify some build hints
    RT_CHECK_ERROR( rtAccelerationCreate(OptiX_Context,&geometry_acceleration) );
    RT_CHECK_ERROR( rtAccelerationSetBuilder(geometry_acceleration,"Trbvh") );
    RT_CHECK_ERROR( rtAccelerationSetTraverser(geometry_acceleration,"Bvh") );
    RT_CHECK_ERROR( rtGeometryGroupSetAcceleration( geometry_group, geometry_acceleration) );
    RT_CHECK_ERROR( rtAccelerationMarkDirty( geometry_acceleration ) );

    /* Create geometry instances */
    //patches
    RT_CHECK_ERROR( rtGeometryInstanceCreate( OptiX_Context, &patch_instance ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( patch_instance, patch ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( patch_instance, 1 ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( patch_instance, 0, patch_material ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChild( geometry_group, 0, patch_instance ) );
    //triangles
    RT_CHECK_ERROR( rtGeometryInstanceCreate( OptiX_Context, &triangle_instance ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( triangle_instance, triangle ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( triangle_instance, 1 ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( triangle_instance, 0, triangle_material ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChild( geometry_group, 1, triangle_instance ) );
    //disks
    RT_CHECK_ERROR( rtGeometryInstanceCreate( OptiX_Context, &disk_instance ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( disk_instance, disk ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( disk_instance, 1 ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( disk_instance, 0, disk_material ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChild( geometry_group, 2, disk_instance ) );
    //tiles
    RT_CHECK_ERROR( rtGeometryInstanceCreate( OptiX_Context, &tile_instance ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( tile_instance, tile ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( tile_instance, 1 ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( tile_instance, 0, tile_material ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChild( geometry_group, 3, tile_instance ) );
    //voxels
    RT_CHECK_ERROR( rtGeometryInstanceCreate( OptiX_Context, &voxel_instance ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( voxel_instance, voxel ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( voxel_instance, 1 ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( voxel_instance, 0, voxel_material ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChild( geometry_group, 4, voxel_instance ) );

    //voxels
    RT_CHECK_ERROR( rtGeometryInstanceCreate( OptiX_Context, &bbox_instance ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( bbox_instance, bbox ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( bbox_instance, 1 ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( bbox_instance, 0, bbox_material ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChild( geometry_group, 5, bbox_instance ) );

    /* Set the top_object variable */
    //NOTE: Not sure exactly where this has to be set
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "top_object", &top_object ) );
    RT_CHECK_ERROR( rtVariableSetObject( top_object, top_level_group ) );

    //random number seed
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "random_seed", &random_seed_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet1ui( random_seed_RTvariable, std::chrono::system_clock::now().time_since_epoch().count() ) );

    //launch offset
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "launch_offset", &launch_offset_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSet1ui( launch_offset_RTvariable, 0 ) );

    //maximum scattering depth
    RT_CHECK_ERROR( rtBufferCreate( OptiX_Context, RT_BUFFER_INPUT, &max_scatters_RTbuffer ) );
    RT_CHECK_ERROR( rtBufferSetFormat( max_scatters_RTbuffer, RT_FORMAT_UNSIGNED_INT ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, "max_scatters", &max_scatters_RTvariable ) );
    RT_CHECK_ERROR( rtVariableSetObject( max_scatters_RTvariable, max_scatters_RTbuffer ) );
    zeroBuffer1D( max_scatters_RTbuffer, 1 );

    // RTsize device_memory;
    // RT_CHECK_ERROR( rtContextGetAttribute( OptiX_Context, RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY, sizeof(RTsize), &device_memory ) );

    // device_memory *= 1e-6;
    // if( device_memory < 1000 ){
    //   printf("available device memory at end of OptiX initialization: %6.3f MB\n",device_memory);
    // }else{
    //   printf("available device memory at end of OptiX initialization: %6.3f GB\n",device_memory*1e-3);
    // }

}

void RadiationModel::updateGeometry() {
    updateGeometry( context->getAllUUIDs() );
}

void RadiationModel::updateGeometry( const std::vector<uint>& UUIDs ){

    if( message_flag ){
        std::cout << "Updating geometry in radiation transport model..." << std::flush;
    }

    for ( auto &band : radiation_bands ) {
        band.second.radiativepropertiesinitialized = false;
    }

    context_UUIDs = UUIDs;

    for( int u=context_UUIDs.size()-1; u>=0; u-- ){
        if( !context->doesPrimitiveExist( context_UUIDs.at(u) ) ){
            context_UUIDs.erase( context_UUIDs.begin()+u );
        }else if( context->getPrimitiveArea(context_UUIDs.at(u))==0 ){
            context_UUIDs.erase( context_UUIDs.begin()+u );
        }
    }

    //--- Make Bounding Patches ---//

    //determine domain bounding sphere
    float sphere_radius;
    vec3 sphere_center;
    context->getDomainBoundingSphere( sphere_center, sphere_radius );

    rtVariableSet1f( bound_sphere_radius_RTvariable, sphere_radius );
    rtVariableSet3f( bound_sphere_center_RTvariable, sphere_center.x, sphere_center.y, sphere_center.z );

    //--- Populate Primitive Geometry Buffers ---//

    size_t Nprimitives = context_UUIDs.size(); //Number of primitives

    //transformation matrix buffer - size=Nobjects
    std::vector<std::vector<float> > m_global;

    //primitive type buffer - size=Nobjects
    std::vector<uint> ptype_global;

    //primitive area buffer - size=Nobjects
    std::vector<float> area_global;

    //primitive UUID buffers - total size of all combined is Nobjects
    std::vector<uint> patch_UUID;
    std::vector<uint> triangle_UUID;
    std::vector<uint> disk_UUID;
    std::vector<uint> tile_UUID;
    std::vector<uint> voxel_UUID;

    //twosided flag buffer - size=Nobjects
    std::vector<bool> twosided_flag_global;

    //primitive geometry specification buffers
    std::vector<std::vector<optix::float3> > patch_vertices;
    std::vector<std::vector<optix::float3> > triangle_vertices;
    std::vector<std::vector<optix::float3> > disk_vertices;
    std::vector<std::vector<optix::float3> > tile_vertices;
    std::vector<std::vector<optix::float3> > voxel_vertices;

    //number of patch subdivisions for each tile - size is same as tile_vertices
    std::vector<optix::int2> object_subdivisions;

    std::vector<uint> objID_all = context->getAllObjectIDs();

    //ID of object corresponding to each primitive - size Nprimitives
    std::vector<uint> objectID;
    objectID.resize(Nprimitives);

    std::size_t patch_count = 0;
    std::size_t triangle_count = 0;
    std::size_t disk_count = 0;
    std::size_t tile_count = 0;
    std::size_t voxel_count = 0;

    area_global.resize(Nprimitives);

    primitiveID.resize(0);

    //Create a vector of primitive pointers 'primitives' (note: only add one pointer for compound objects)
    uint objID = 0;
    uint ID = 99999;
    for( std::size_t u=0; u<Nprimitives; u++ ){

        uint p = context_UUIDs.at(u);

        //primitve area
        area_global.at(u) = context->getPrimitiveArea(p);

        uint parentID = context->getPrimitiveParentObjectID(p);

        if( ID!=parentID || parentID==0 || context->getObjectPointer(parentID)->getObjectType()!=helios::OBJECT_TYPE_TILE  ){//if this is a new object, or primitive does not belong to an object
            primitiveID.push_back( u );
            ID = parentID;
            objID++;
        }else{
            ID = parentID;
        }

        assert(objID>0);

        objectID.at(u) = objID-1;

    }

    //Nobjects is the number of isolated primitives plus the number of compound objects (all primitives inside and object combined only counts as one element)
    size_t Nobjects = primitiveID.size();

    m_global.resize(Nobjects);
    ptype_global.resize(Nobjects);
    twosided_flag_global.resize(Nobjects);//initialize to be two-sided
    for( size_t i=0; i<Nobjects; i++ ){
        twosided_flag_global.at(i) = true;
    }

    //Populate attributes for each primitive in the pointer vector 'primitives'
    for( std::size_t u=0; u<Nobjects; u++ ){

        uint p = context_UUIDs.at(primitiveID.at(u));

        //transformation matrix
        float m[16];

        //primitive type
        helios::PrimitiveType type = context->getPrimitiveType(p);
        ptype_global.at(u) =  type;

        assert( ptype_global.at(u)>=0 && ptype_global.at(u)<=4 );

        //primitive twosided flag
        if( context->doesPrimitiveDataExist( p,"twosided_flag") ){
            uint flag;
            context->getPrimitiveData( p,"twosided_flag",flag);
            if( flag ){
                twosided_flag_global.at(u) = true;
            }else{
                twosided_flag_global.at(u) = false;
            }
        }

        uint parentID = context->getPrimitiveParentObjectID(p);

        if( parentID>0 && context->getObjectPointer(parentID)->getObjectType()==helios::OBJECT_TYPE_TILE ){//tile objects

            ptype_global.at(u) = 3;

            context->getObjectPointer(parentID)->getTransformationMatrix(m);

            m_global.at(u).resize(16);
            for( uint i=0; i<16; i++ ){
                m_global.at(u).at(i) = m[i];
            }

            std::vector<vec3> vertices = context->getTileObjectPointer(parentID)->getVertices();
            std::vector<optix::float3> v;
            v.push_back( optix::make_float3(vertices.at(0).x,vertices.at(0).y,vertices.at(0).z) );
            v.push_back( optix::make_float3(vertices.at(1).x,vertices.at(1).y,vertices.at(1).z) );
            v.push_back( optix::make_float3(vertices.at(2).x,vertices.at(2).y,vertices.at(2).z) );
            v.push_back( optix::make_float3(vertices.at(3).x,vertices.at(3).y,vertices.at(3).z) );
            tile_vertices.push_back(v);

            helios::int2 subdiv = context->getTileObjectPointer(parentID)->getSubdivisionCount();

            object_subdivisions.push_back( optix::make_int2(subdiv.x,subdiv.y) );

            tile_UUID.push_back( primitiveID.at(u) );
            tile_count ++;

        }else if( type == helios::PRIMITIVE_TYPE_PATCH ){ //patches

            context->getPrimitiveTransformationMatrix( p,m);

            m_global.at(u).resize(16);
            for( uint i=0; i<16; i++ ){
                m_global.at(u).at(i) = m[i];
            }

            std::vector<vec3> vertices = context->getPrimitiveVertices(p);
            std::vector<optix::float3> v;
            v.push_back( optix::make_float3(vertices.at(0).x,vertices.at(0).y,vertices.at(0).z) );
            v.push_back( optix::make_float3(vertices.at(1).x,vertices.at(1).y,vertices.at(1).z) );
            v.push_back( optix::make_float3(vertices.at(2).x,vertices.at(2).y,vertices.at(2).z) );
            v.push_back( optix::make_float3(vertices.at(3).x,vertices.at(3).y,vertices.at(3).z) );
            patch_vertices.push_back(v);
            object_subdivisions.push_back( optix::make_int2(1,1) );
            patch_UUID.push_back( primitiveID.at(u) );
            patch_count ++;
        }else if( type == helios::PRIMITIVE_TYPE_TRIANGLE ){ //triangles

            context->getPrimitiveTransformationMatrix(p,m);

            m_global.at(u).resize(16);
            for( uint i=0; i<16; i++ ){
                m_global.at(u).at(i) = m[i];
            }

            std::vector<vec3> vertices = context->getPrimitiveVertices(p);
            std::vector<optix::float3> v;
            v.push_back( optix::make_float3(vertices.at(0).x,vertices.at(0).y,vertices.at(0).z) );
            v.push_back( optix::make_float3(vertices.at(1).x,vertices.at(1).y,vertices.at(1).z) );
            v.push_back( optix::make_float3(vertices.at(2).x,vertices.at(2).y,vertices.at(2).z) );
            triangle_vertices.push_back(v);
            object_subdivisions.push_back( optix::make_int2(1,1) );
            triangle_UUID.push_back( primitiveID.at(u) );
            triangle_count ++;
        }else if( type == helios::PRIMITIVE_TYPE_VOXEL ){ //voxels

            context->getPrimitiveTransformationMatrix(p,m);

            m_global.at(u).resize(16);
            for( uint i=0; i<16; i++ ){
                m_global.at(u).at(i) = m[i];
            }

            helios::vec3 center = context->getVoxelCenter(p);
            helios::vec3 size = context->getVoxelSize(p);
            std::vector<optix::float3> v;
            v.push_back( optix::make_float3(center.x-0.5f*size.x, center.y-0.5f*size.y, center.z-0.5f*size.z ) );
            v.push_back( optix::make_float3(center.x+0.5f*size.x, center.y+0.5f*size.y, center.z+0.5f*size.z ) );
            voxel_vertices.push_back(v);
            object_subdivisions.push_back( optix::make_int2(1,1) );
            voxel_UUID.push_back( primitiveID.at(u) );
            voxel_count ++;
        }

    }

    //Texture mask data
    std::vector<std::vector<std::vector<bool> > > maskdata;
    std::map<std::string,uint> maskname;
    std::vector<optix::int2> masksize;
    std::vector<int> maskID;
    std::vector<std::vector<optix::float2> > uvdata;
    std::vector<int> uvID;
    maskID.resize(Nobjects);
    uvID.resize(Nobjects);

    for( size_t u=0; u<Nobjects; u++ ){

        uint p = context_UUIDs.at(primitiveID.at(u));

        std::string maskfile = context->getPrimitiveTextureFile(p);

        uint parentID = context->getPrimitiveParentObjectID(p);

        if( context->getPrimitiveType(p)==PRIMITIVE_TYPE_VOXEL || maskfile.size()==0 || !context->primitiveTextureHasTransparencyChannel(p) ){ //does not have texture transparency

            maskID.at(u) = -1;
            uvID.at(u) = -1;

        }else{

            // texture mask data //

            //Check if this mask has already been added
            if( maskname.find(maskfile) != maskname.end() ){ //has already been added
                uint ID = maskname.at(maskfile);
                maskID.at(u) = ID;
            }else{ //mask has not been added

                uint ID = maskdata.size();
                maskID.at(u) = ID;
                maskname[maskfile] = maskdata.size();
                maskdata.push_back( *context->getPrimitiveTextureTransparencyData(p) );
                auto sy = maskdata.back().size();
                auto sx = maskdata.back().front().size();
                masksize.push_back( optix::make_int2(sx,sy) );
            }

            // uv coordinates //
            std::vector<vec2> uv;

            if( parentID==0 || context->getObjectPointer(parentID)->getObjectType()!=helios::OBJECT_TYPE_TILE ){ //primitives
                uv = context->getPrimitiveTextureUV(p);
            }

            if( !uv.empty() ){ //has custom (u,v) coordinates
                std::vector<optix::float2> uvf2;
                uvf2.resize(4);
                //first index if uvf2 is the minimum (u,v) coordinate, second index is the size of the (u,v) rectangle in x- and y-directions.

                for( int i=0; i<uv.size(); i++ ){
                    uvf2.at(i) = optix::make_float2(uv.at(i).x, uv.at(i).y);
                }
                if( uv.size()==3 ){
                    uvf2.at(3) = optix::make_float2(0,0);
                }
                uvdata.push_back( uvf2 );
                uvID.at(u) = uvdata.size()-1;
            }else{ //DOES NOT have custom (u,v) coordinates
                uvID.at(u) = -1;
            }

        }
    }

    int2 size_max(0,0);
    for( int t=0; t<maskdata.size(); t++ ){
        int2 sz( maskdata.at(t).front().size(), maskdata.at(t).size() );
        if( sz.x>size_max.x ){
            size_max.x = sz.x;
        }
        if( sz.y>size_max.y ){
            size_max.y = sz.y;
        }
    }

    for( int t=0; t<maskdata.size(); t++ ){
        maskdata.at(t).resize(size_max.y);
        for( int j=0; j<size_max.y; j++ ){
            maskdata.at(t).at(j).resize(size_max.x);
        }
    }

    initializeBuffer3D( maskdata_RTbuffer, maskdata );
    initializeBuffer1Dint2( masksize_RTbuffer, masksize );
    initializeBuffer1Di( maskID_RTbuffer, maskID );

    initializeBuffer2Dfloat2( uvdata_RTbuffer, uvdata );
    initializeBuffer1Di( uvID_RTbuffer, uvID );

    //Bounding box
    helios::vec2 xbounds, ybounds, zbounds;
    context->getDomainBoundingBox( xbounds, ybounds, zbounds );

    xbounds.x -= 1e-5;
    xbounds.y += 1e-5;
    ybounds.x -= 1e-5;
    ybounds.y += 1e-5;

    std::vector<uint> bbox_UUID;
    int bbox_face_count = 0;

    std::vector<std::vector<optix::float3> > bbox_vertices;

    //primitive type

    std::vector<optix::float3> v;
    v.resize(4);

    if( periodic_flag.x==1 ){

        // -x facing
        v.at(0) = optix::make_float3(xbounds.x,ybounds.x,zbounds.x);
        v.at(1) = optix::make_float3(xbounds.x,ybounds.y,zbounds.x);
        v.at(2) = optix::make_float3(xbounds.x,ybounds.y,zbounds.y);
        v.at(3) = optix::make_float3(xbounds.x,ybounds.x,zbounds.y);
        bbox_vertices.push_back(v);
        bbox_UUID.push_back(Nprimitives+bbox_face_count);
        objectID.push_back( Nobjects+bbox_face_count );
        ptype_global.push_back(5);
        bbox_face_count++;

        // +x facing
        v.at(0) = optix::make_float3(xbounds.y,ybounds.x,zbounds.x);
        v.at(1) = optix::make_float3(xbounds.y,ybounds.y,zbounds.x);
        v.at(2) = optix::make_float3(xbounds.y,ybounds.y,zbounds.y);
        v.at(3) = optix::make_float3(xbounds.y,ybounds.x,zbounds.y);
        bbox_vertices.push_back(v);
        bbox_UUID.push_back(Nprimitives+bbox_face_count);
        objectID.push_back( Nobjects+bbox_face_count );
        ptype_global.push_back(5);
        bbox_face_count++;

    }
    if( periodic_flag.y==1 ){

        // -y facing
        v.at(0) = optix::make_float3(xbounds.x,ybounds.x,zbounds.x);
        v.at(1) = optix::make_float3(xbounds.y,ybounds.x,zbounds.x);
        v.at(2) = optix::make_float3(xbounds.y,ybounds.x,zbounds.y);
        v.at(3) = optix::make_float3(xbounds.x,ybounds.x,zbounds.y);
        bbox_vertices.push_back(v);
        bbox_UUID.push_back(Nprimitives+bbox_face_count);
        objectID.push_back( Nobjects+bbox_face_count );
        ptype_global.push_back(5);
        bbox_face_count++;

        // +y facing
        v.at(0) = optix::make_float3(xbounds.x,ybounds.y,zbounds.x);
        v.at(1) = optix::make_float3(xbounds.y,ybounds.y,zbounds.x);
        v.at(2) = optix::make_float3(xbounds.y,ybounds.y,zbounds.y);
        v.at(3) = optix::make_float3(xbounds.x,ybounds.y,zbounds.y);
        bbox_vertices.push_back(v);
        bbox_UUID.push_back(Nprimitives+bbox_face_count);
        objectID.push_back( Nobjects+bbox_face_count );
        ptype_global.push_back(5);
        bbox_face_count++;

    }

    initializeBuffer2Df( transform_matrix_RTbuffer, m_global );
    initializeBuffer1Dui( primitive_type_RTbuffer, ptype_global );
    initializeBuffer1Df( primitive_area_RTbuffer, area_global );
    initializeBuffer1Dbool( twosided_flag_RTbuffer, twosided_flag_global );
    initializeBuffer2Dfloat3( patch_vertices_RTbuffer, patch_vertices );
    initializeBuffer2Dfloat3( triangle_vertices_RTbuffer, triangle_vertices );
    initializeBuffer2Dfloat3( tile_vertices_RTbuffer, tile_vertices );
    initializeBuffer2Dfloat3( voxel_vertices_RTbuffer, voxel_vertices );
    initializeBuffer2Dfloat3( bbox_vertices_RTbuffer, bbox_vertices );

    initializeBuffer1Dint2( object_subdivisions_RTbuffer, object_subdivisions );

    initializeBuffer1Dui( patch_UUID_RTbuffer, patch_UUID );
    initializeBuffer1Dui( triangle_UUID_RTbuffer, triangle_UUID );
    initializeBuffer1Dui( disk_UUID_RTbuffer, disk_UUID );
    initializeBuffer1Dui( tile_UUID_RTbuffer, tile_UUID );
    initializeBuffer1Dui( voxel_UUID_RTbuffer, voxel_UUID );
    initializeBuffer1Dui( bbox_UUID_RTbuffer, bbox_UUID );

    initializeBuffer1Dui( objectID_RTbuffer, objectID );
    initializeBuffer1Dui( primitiveID_RTbuffer, primitiveID );

    RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( patch, patch_count ) );
    RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( triangle, triangle_count ) );
    RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( disk, disk_count ) );
    RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( tile, tile_count ) );
    RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( voxel, voxel_count ) );
    RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( bbox, bbox_face_count ) );

    RT_CHECK_ERROR( rtAccelerationMarkDirty( geometry_acceleration ) );

    /* Set the top_object variable */
    //NOTE: not sure if this has to be set again or not..
    RT_CHECK_ERROR( rtVariableSetObject( top_object, top_level_group ) );

    RTsize device_memory;
    RT_CHECK_ERROR( rtContextGetAttribute( OptiX_Context, RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY, sizeof(RTsize), &device_memory ) );

    device_memory *= 1e-6;

    if( device_memory<500 ){
        std::cout << "WARNING (RadiationModel): device memory is very low (" << device_memory << " MB)" << std::endl;
    }

    /* Validate/Compile OptiX Context */
    RT_CHECK_ERROR( rtContextValidate( OptiX_Context ) );
    RT_CHECK_ERROR( rtContextCompile( OptiX_Context ) );

    // device_memory;
    // RT_CHECK_ERROR( rtContextGetAttribute( OptiX_Context, RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY, sizeof(RTsize), &device_memory ) );

    // device_memory *= 1e-6;
    // if( device_memory < 1000 ){
    //   printf("available device memory at end of OptiX context compile: %6.3f MB\n",device_memory);
    // }else{
    //   printf("available device memory at end of OptiX context compile: %6.3f GB\n",device_memory*1e-3);
    // }

    isgeometryinitialized = true;

    // device_memory;
    // RT_CHECK_ERROR( rtContextGetAttribute( OptiX_Context, RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY, sizeof(RTsize), &device_memory ) );

    // device_memory *= 1e-6;
    // if( device_memory < 1000 ){
    //   printf("available device memory before acceleration build: %6.3f MB\n",device_memory);
    // }else{
    //   printf("available device memory before acceleration build: %6.3f GB\n",device_memory*1e-3);
    // }

    optix::int3 launch_dim_dummy = optix::make_int3( 1, 1, 1 );
    RT_CHECK_ERROR( rtContextLaunch3D( OptiX_Context, RAYTYPE_DIRECT , launch_dim_dummy.x, launch_dim_dummy.y, launch_dim_dummy.z ) );

    RT_CHECK_ERROR( rtContextGetAttribute( OptiX_Context, RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY, sizeof(RTsize), &device_memory ) );

    // device_memory;
    // RT_CHECK_ERROR( rtContextGetAttribute( OptiX_Context, RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY, sizeof(RTsize), &device_memory ) );

    // device_memory *= 1e-6;
    // if( device_memory < 1000 ){
    //   printf("available device memory at end of acceleration build: %6.3f MB\n",device_memory);
    // }else{
    //   printf("available device memory at end of acceleration build: %6.3f GB\n",device_memory*1e-3);
    // }

    if( message_flag ){
        std::cout << "done." << std::endl;
    }

}

void RadiationModel::updateRadiativeProperties( const std::vector<std::string> &labels ) {

    // Possible scenarios for specifying a primitive's radiative properties
    // 1. If primitive data of form reflectivity_band/transmissivity_band is given, this value is used and overrides any other option.
    // 2. If primitive data of form reflectivity_spectrum/transmissivity_spectrum is given that references global data containing spectral reflectivity/transmissivity:
    //    2a. If radiation source spectrum was not given, assume source spectral intensity is constant over band and calculate using primitive spectrum
    //    2b. If radiation source spectrum was given, calculate using both source and primitive spectrum.

    if( message_flag ) {
        std::cout << "Updating radiative properties..." << std::flush;
    }

    uint Nbands = labels.size();//number of radiative bands
    uint Nsources = radiation_sources.size();
    uint Ncameras = cameras.size();
    size_t Nobjects = primitiveID.size();
    size_t Nprimitives = context_UUIDs.size();

    scattering_iterations_needed.resize(Nbands);
    for( int b=0; b<Nbands; b++ ){
        scattering_iterations_needed.at(b) = false;
    }

    std::vector<std::vector<std::vector<float> > > rho, tau; //first index is the source, second index is the primitive, third index is the band
    std::vector<std::vector<std::vector<std::vector<float> > > > rho_cam, tau_cam; //Fourth index is the camera
    float eps;

    std::string prop;

    for( auto &label : radiation_bands){
        label.second.radiativepropertiesinitialized = false;
    }
    for ( const auto &label: labels) {
        radiation_bands.at(label).radiativepropertiesinitialized = true;
    }

    rho.resize(Nsources);
    tau.resize(Nsources);
    for( size_t s=0; s<Nsources; s++ ){
        rho.at(s).resize(Nprimitives);
        tau.at(s).resize(Nprimitives);
        for ( size_t p=0; p<Nprimitives; p++ ) {
            rho.at(s).at(p).resize(Nbands);
            tau.at(s).at(p).resize(Nbands);
        }
    }
    if ( Ncameras ) {
        rho_cam.resize(Nsources);
        tau_cam.resize(Nsources);
        for( size_t s=0; s<Nsources; s++ ) {
            rho_cam.at(s).resize(Nprimitives);
            tau_cam.at(s).resize(Nprimitives);
            for (size_t p = 0; p < Nprimitives; p++) {
                rho_cam.at(s).at(p).resize(Nbands);
                tau_cam.at(s).at(p).resize(Nbands);
                for( size_t b=0; b<Nbands; b++ ){
                    rho_cam.at(s).at(p).at(b).resize(Ncameras);
                    tau_cam.at(s).at(p).at(b).resize(Ncameras);
                }
            }
        }
    }

    //Pre-calculate all unique camera spectral responses for all cameras and bands
    std::vector<std::vector<std::vector<helios::vec2> > > camera_response_unique;
    camera_response_unique.resize(Ncameras);
    if (Ncameras > 0) {
        uint cam = 0;
        for (const auto &camera: cameras) {

            camera_response_unique.at(cam).resize(Nbands);

            for (uint b = 0; b < Nbands; b++) {

                if( camera.second.band_spectral_response.find(labels.at(b)) == camera.second.band_spectral_response.end() ){
                    continue;
                }

                std::string camera_response = camera.second.band_spectral_response.at(labels.at(b));

                if (!camera_response.empty()) {

                    if( !context->doesGlobalDataExist(camera_response.c_str()) ){
                        if( camera_response!="uniform" ) {
                            std::cerr << "WARNING (RadiationModel::updateRadiativeProperties): Camera spectral response \"" << camera_response << "\" does not exist. Assuming a uniform spectral response..." << std::flush;
                        }
                    }else if ( context->getGlobalDataType(camera_response.c_str()) == helios::HELIOS_TYPE_VEC2) {

                        std::vector<helios::vec2> data;
                        context->getGlobalData(camera_response.c_str(), data);

                        camera_response_unique.at(cam).at(b) = data;

                    }else if ( context->getGlobalDataType(camera_response.c_str()) != helios::HELIOS_TYPE_VEC2 && context->getGlobalDataType(camera_response.c_str()) != helios::HELIOS_TYPE_STRING ) {
                        camera_response.clear();
                        std::cout << "WARNING (RadiationModel::runBand): Camera spectral response \"" << camera_response << "\" is not of type HELIOS_TYPE_VEC2 or HELIOS_TYPE_STRING. Assuming a uniform spectral response..." << std::flush;
                    }

                }

            }
            cam++;
        }
    }

    //Pre-calculate all unique primitive reflectivity and transmissivity values before assigning to primitives

    //first, figure out all of the spectra referenced by all primitives and store it in "surface_spectra" to avoid having to load it again
    std::map<std::string,std::vector<helios::vec2> > surface_spectra_rho;
    std::map<std::string,std::vector<helios::vec2> > surface_spectra_tau;
    for (size_t u = 0; u < Nprimitives; u++) {

        uint UUID = context_UUIDs.at(u);

        if (context->doesPrimitiveDataExist(UUID, "reflectivity_spectrum") ) {
            if (context->getPrimitiveDataType(UUID, "reflectivity_spectrum") == HELIOS_TYPE_STRING) {
                std::string spectrum_label;
                context->getPrimitiveData(UUID, "reflectivity_spectrum", spectrum_label);

                //get the spectral reflectivity data and store it in surface_spectra to avoid having to load it again
                if (surface_spectra_rho.find(spectrum_label) == surface_spectra_rho.end() ){
                    if( !context->doesGlobalDataExist(spectrum_label.c_str()) ){
                        if (message_flag) {
                            std::cerr << "WARNING (RadiationModel::runBand): Primitive spectral reflectivity \"" << spectrum_label << "\" does not exist. Using default reflectivity of 0..." << std::flush;
                        }
                        std::vector<helios::vec2> data;
                        surface_spectra_rho.emplace(spectrum_label, data);
                    }else if( context->getGlobalDataType(spectrum_label.c_str() ) == HELIOS_TYPE_VEC2 ) {

                        std::vector<helios::vec2> data;
                        context->getGlobalData(spectrum_label.c_str(), data);
                        surface_spectra_rho.emplace(spectrum_label, data);

                    }else if ( context->getGlobalDataType(spectrum_label.c_str()) != helios::HELIOS_TYPE_VEC2 && context->getGlobalDataType(spectrum_label.c_str()) != helios::HELIOS_TYPE_STRING ) {
                        spectrum_label.clear();
                        std::cout << "WARNING (RadiationModel::runBand): Object spectral reflectivity \"" << spectrum_label << "\" is not of type HELIOS_TYPE_VEC2 or HELIOS_TYPE_STRING. Assuming a uniform spectral distribution..." << std::flush;
                    }

                }

            }
        }

        if (context->doesPrimitiveDataExist(UUID, "transmissivity_spectrum") ) {
            if (context->getPrimitiveDataType(UUID, "transmissivity_spectrum") == HELIOS_TYPE_STRING) {
                std::string spectrum_label;
                context->getPrimitiveData(UUID, "transmissivity_spectrum", spectrum_label);

                //get the spectral transmissivity data and store it in surface_spectra to avoid having to load it again
                if (surface_spectra_tau.find(spectrum_label) == surface_spectra_tau.end() ){
                    if( !context->doesGlobalDataExist(spectrum_label.c_str()) ) {
                        if (message_flag){
                            std::cerr << "WARNING (RadiationModel::runBand): Primitive spectral transmissivity \"" << spectrum_label << "\" does not exist. Using default transmissivity of 0..." << std::flush;
                        }
                        std::vector<helios::vec2> data;
                        surface_spectra_tau.emplace(spectrum_label, data);
                    }else if( context->getGlobalDataType(spectrum_label.c_str() ) == HELIOS_TYPE_VEC2 ) {

                        std::vector<helios::vec2> data;
                        context->getGlobalData(spectrum_label.c_str(), data);
                        surface_spectra_tau.emplace(spectrum_label, data);

                    }else if ( context->getGlobalDataType(spectrum_label.c_str()) != helios::HELIOS_TYPE_VEC2 && context->getGlobalDataType(spectrum_label.c_str()) != helios::HELIOS_TYPE_STRING ) {
                        spectrum_label.clear();
                        std::cout << "WARNING (RadiationModel::runBand): Object spectral transmissivity \"" << spectrum_label << "\" is not of type HELIOS_TYPE_VEC2 or HELIOS_TYPE_STRING. Assuming a uniform spectral distribution..." << std::flush;
                    }
                }

            }
        }
    }

//    printf("%d (rho) %d (tau) unique surface spectra loaded\n", surface_spectra_rho.size(),surface_spectra_tau.size());

    //second, calculate unique values of rho and tau for all sources and bands
    std::map<std::string,std::vector<std::vector<float> > > rho_unique;
    std::map<std::string,std::vector<std::vector<float> > > tau_unique;

    std::map<std::string,std::vector<std::vector<std::vector<float> > > > rho_cam_unique;
    std::map<std::string,std::vector<std::vector<std::vector<float> > > > tau_cam_unique;

    std::vector<std::vector<float> > empty;
    empty.resize(Nbands);
    for( uint b=0; b<Nbands; b++ ) {
        empty.at(b).resize(Nsources,0);
    }
    std::vector<std::vector<std::vector<float> > > empty_cam;
    if (Ncameras > 0) {
        empty_cam.resize(Nbands);
        for (uint b = 0; b < Nbands; b++) {
            empty_cam.at(b).resize(Nsources);
            for (uint s = 0; s < Nsources; s++) {
                empty_cam.at(b).at(s).resize(Ncameras, 0);
            }
        }
    }

    for( const auto &spectrum : surface_spectra_rho ) { //reflectivity

        rho_unique.emplace(spectrum.first, empty);
        if (Ncameras > 0) {
            rho_cam_unique.emplace(spectrum.first, empty_cam);
        }

        for (uint b = 0; b < Nbands; b++) {
            std::string band = labels.at(b);

            for (uint s = 0; s < Nsources; s++) {

                //integrate
                if( radiation_bands.at(band).wavebandBounds.x != 0 && radiation_bands.at(band).wavebandBounds.y != 0 && !spectrum.second.empty() ) {
                    if (!radiation_sources.at(s).source_spectrum.empty()) {
                        rho_unique.at(spectrum.first).at(b).at(s) = integrateSpectrum(s, spectrum.second, radiation_bands.at(band).wavebandBounds.x, radiation_bands.at(band).wavebandBounds.y);
                    } else {
                        //source spectrum not provided, assume source intensity is constant over the band
                        rho_unique.at(spectrum.first).at(b).at(s) = integrateSpectrum(spectrum.second, radiation_bands.at(band).wavebandBounds.x, radiation_bands.at(band).wavebandBounds.y)/(radiation_bands.at(band).wavebandBounds.y-radiation_bands.at(band).wavebandBounds.x);
                    }
                }else{
                    rho_unique.at(spectrum.first).at(b).at(s) = rho_default;
                }

                //cameras
                if (Ncameras > 0) {
                    uint cam=0;
                    for (const auto &camera: cameras) {

                        if( camera_response_unique.at(cam).at(b).empty() ){
                            rho_cam_unique.at(spectrum.first).at(b).at(s).at(cam) = rho_unique.at(spectrum.first).at(b).at(s);

                        }else {

                            //integrate
                            if( !spectrum.second.empty() ) {
                                if (!radiation_sources.at(s).source_spectrum.empty()) {
                                    rho_cam_unique.at(spectrum.first).at(b).at(s).at(cam) = integrateSpectrum(s, spectrum.second, camera_response_unique.at(cam).at(b));
                                } else{
                                    rho_cam_unique.at(spectrum.first).at(b).at(s).at(cam) = integrateSpectrum(spectrum.second, camera_response_unique.at(cam).at(b));
                                }
                            }else{
                                rho_cam_unique.at(spectrum.first).at(b).at(s).at(cam) = rho_default;
                            }

                        }

                        cam++;
                    }
                }

            }

        }

    }


    for( const auto &spectrum : surface_spectra_tau ) {//transmissivity

        tau_unique.emplace(spectrum.first, empty);
        if (Ncameras > 0) {
            tau_cam_unique.emplace(spectrum.first, empty_cam);
        }

        for( uint b=0; b<Nbands; b++ ) {
            std::string band = labels.at(b);

            for (uint s = 0; s < Nsources; s++) {

                //integrate
                if( radiation_bands.at(band).wavebandBounds.x!=0 && radiation_bands.at(band).wavebandBounds.y!=0 && !spectrum.second.empty()) {
                    if (!radiation_sources.at(s).source_spectrum.empty()) {
                        tau_unique.at(spectrum.first).at(b).at(s) = integrateSpectrum(s, spectrum.second, radiation_bands.at(band).wavebandBounds.x, radiation_bands.at(band).wavebandBounds.y);
                    } else{
                        tau_unique.at(spectrum.first).at(b).at(s) = integrateSpectrum(spectrum.second, radiation_bands.at(band).wavebandBounds.x, radiation_bands.at(band).wavebandBounds.y)/(radiation_bands.at(band).wavebandBounds.y-radiation_bands.at(band).wavebandBounds.x);
                    }
                }else{
                    tau_unique.at(spectrum.first).at(b).at(s) = tau_default;
                }

                //cameras
                if (Ncameras > 0) {
                    uint cam=0;
                    for (const auto &camera: cameras) {

                        if( camera_response_unique.at(cam).at(b).empty() ){

                            tau_cam_unique.at(spectrum.first).at(b).at(s).at(cam) = tau_unique.at(spectrum.first).at(b).at(s);

                        }else {

                            //integrate
                            if( !spectrum.second.empty() ) {
                                if (!radiation_sources.at(s).source_spectrum.empty()) {
                                    tau_cam_unique.at(spectrum.first).at(b).at(s).at(cam) = integrateSpectrum(s, spectrum.second, camera_response_unique.at(cam).at(b));
                                } else{
                                    tau_cam_unique.at(spectrum.first).at(b).at(s).at(cam) = integrateSpectrum(spectrum.second, camera_response_unique.at(cam).at(b));
                                }
                            }else{
                                tau_cam_unique.at(spectrum.first).at(b).at(s).at(cam) = tau_default;
                            }

                        }

                        cam++;
                    }
                }

            }

        }

    }

    for (size_t u = 0; u < Nprimitives; u++) {

        uint UUID = context_UUIDs.at(u);

        helios::PrimitiveType type = context->getPrimitiveType(UUID);

        if (type == helios::PRIMITIVE_TYPE_VOXEL) {

//                // NOTE: This is a little confusing - for volumes of participating media, we're going to use the "rho" variable to store the absorption coefficient and use the "tau" variable to store the scattering coefficient.  This is to save on memory so we don't have to define separate arrays.
//
//                // Absorption coefficient
//
//                prop = "attenuation_coefficient_" + band;
//
//                if (context->doesPrimitiveDataExist(UUID, prop.c_str())) {
//                    context->getPrimitiveData(UUID, prop.c_str(), rho.at(u).at(b));
//                } else {
//                    rho.at(u).at(b) = kappa_default;
//                    context->setPrimitiveData(UUID, prop.c_str(), helios::HELIOS_TYPE_FLOAT, 1, &kappa_default);
//                }
//
//                if (rho.at(u).at(b) < 0) {
//                    rho.at(u).at(b) = 0.f;
//                    if (message_flag) {
//                        std::cout
//                                << "WARNING (RadiationModel): absorption coefficient cannot be less than 0.  Clamping to 0 for band "
//                                << band << "." << std::endl;
//                    }
//                } else if (rho.at(u).at(b) > 1.f) {
//                    rho.at(u).at(b) = 1.f;
//                    if (message_flag) {
//                        std::cout
//                                << "WARNING (RadiationModel): absorption coefficient cannot be greater than 1.  Clamping to 1 for band "
//                                << band << "." << std::endl;
//                    }
//                }
//
//                // Scattering coefficient
//
//                prop = "scattering_coefficient_" + band;
//
//                if (context->doesPrimitiveDataExist(UUID, prop.c_str())) {
//                    context->getPrimitiveData(UUID, prop.c_str(), tau[u]);
//                } else {
//                    tau.at(u).at(b) = sigmas_default;
//                    context->setPrimitiveData(UUID, prop.c_str(), helios::HELIOS_TYPE_FLOAT, 1, &sigmas_default);
//                }
//
//                if (tau.at(u).at(b) < 0) {
//                    tau.at(u).at(b) = 0.f;
//                    if (message_flag) {
//                        std::cout
//                                << "WARNING (RadiationModel): scattering coefficient cannot be less than 0.  Clamping to 0 for band "
//                                << band << "." << std::endl;
//                    }
//                } else if (tau.at(u).at(b) > 1.f) {
//                    tau.at(u).at(b) = 1.f;
//                    if (message_flag) {
//                        std::cout
//                                << "WARNING (RadiationModel): scattering coefficient cannot be greater than 1.  Clamping to 1 for band "
//                                << band << "." << std::endl;
//                    }
//                }

        } else { //other than voxels

            // Reflectivity

            //check for primitive data of form "reflectivity_spectrum" that can be used to calculate reflectivity
            std::string spectrum_label;
            if (context->doesPrimitiveDataExist(UUID, "reflectivity_spectrum")) {
                if (context->getPrimitiveDataType(UUID, "reflectivity_spectrum") == HELIOS_TYPE_STRING) {
                    context->getPrimitiveData(UUID, "reflectivity_spectrum", spectrum_label);
                }
            }

            uint b = 0;
            for (const auto &band: labels) {

                //check for primitive data of form "reflectivity_bandname"
                prop = "reflectivity_" + band;

                float rho_s = rho_default;
                if (context->doesPrimitiveDataExist(UUID, prop.c_str())) {
                    context->getPrimitiveData(UUID, prop.c_str(), rho_s);
                }

                for (uint s = 0; s < Nsources; s++) {
                    if (!spectrum_label.empty() && context->doesGlobalDataExist(spectrum_label.c_str())) {

                        rho.at(s).at(u).at(b) = rho_unique.at(spectrum_label).at(b).at(s);

                        //cameras
                        for( uint cam=0; cam<Ncameras; cam++ ){
                            rho_cam.at(s).at(u).at(b).at(cam) = rho_cam_unique.at(spectrum_label).at(b).at(s).at(cam);
                        }

                        //assign default value if there is no primitive data or spectral data
                    } else {
                        rho.at(s).at(u).at(b) = rho_s;

                        //cameras
                        for( uint cam=0; cam<Ncameras; cam++ ){
                            rho_cam.at(s).at(u).at(b).at(cam) = rho_s;
                        }
                    }

                    //error checking
                    if (rho.at(s).at(u).at(b) < 0) {
                        rho.at(s).at(u).at(b) = 0.f;
                        if (message_flag) {
                            std::cout << "WARNING (RadiationModel): reflectivity cannot be less than 0.  Clamping to 0 for band " << band << "." << std::flush;
                        }
                    } else if (rho.at(s).at(u).at(b) > 1.f) {
                        rho.at(s).at(u).at(b) = 1.f;
                        if (message_flag) {
                            std::cout << "WARNING (RadiationModel): reflectivity cannot be greater than 1.  Clamping to 1 for band " << band << "." << std::flush;
                        }
                    }
                    if( rho.at(s).at(u).at(b)!=0 ){
                        scattering_iterations_needed.at(b) = true;
                    }

                }
                b++;
            }

            // Transmissivity

            //check for primitive data of form "transmissivity_spectrum" that can be used to calculate transmissivity
            spectrum_label.resize(0);
            if (context->doesPrimitiveDataExist(UUID, "transmissivity_spectrum")) {
                if (context->getPrimitiveDataType(UUID, "transmissivity_spectrum") == HELIOS_TYPE_STRING) {
                    context->getPrimitiveData(UUID, "transmissivity_spectrum", spectrum_label);
                }
            }

            b = 0;
            for ( const auto &band: labels ) {

                //check for primitive data of form "transmissivity_bandname"
                prop = "transmissivity_" + band;

                float tau_s = tau_default;
                if (context->doesPrimitiveDataExist(UUID, prop.c_str())) {
                    context->getPrimitiveData(UUID, prop.c_str(), tau_s);
                }

                for (uint s = 0; s < Nsources; s++) {
                    if (!spectrum_label.empty() && context->doesGlobalDataExist(spectrum_label.c_str())) {
                        tau.at(s).at(u).at(b) = tau_unique.at(spectrum_label).at(b).at(s);

                        //cameras
                        for( uint cam=0; cam<Ncameras; cam++ ){
                            tau_cam.at(s).at(u).at(b).at(cam) = tau_cam_unique.at(spectrum_label).at(b).at(s).at(cam);
                        }

                        //assign default value if there is no primitive data or spectral data
                    } else {
                        tau.at(s).at(u).at(b) = tau_s;

                        //cameras
                        for( uint cam=0; cam<Ncameras; cam++ ){
                            tau_cam.at(s).at(u).at(b).at(cam) = tau_s;
                        }
                    }

                    //error checking
                    if (tau.at(s).at(u).at(b) < 0) {
                        tau.at(s).at(u).at(b) = 0.f;
                        if (message_flag) {
                            std::cout << "WARNING (RadiationModel): transmissivity cannot be less than 0.  Clamping to 0 for band " << band << "." << std::endl;
                        }
                    } else if (tau.at(s).at(u).at(b) > 1.f) {
                        tau.at(s).at(u).at(b) = 1.f;
                        if (message_flag) {
                            std::cout << "WARNING (RadiationModel): transmissivity cannot be greater than 1.  Clamping to 1 for band " << band << "." << std::endl;
                        }
                    }
                    if( tau.at(s).at(u).at(b)!=0 ){
                        scattering_iterations_needed.at(b) = true;
                    }
                }
                b++;
            }

            // Emissivity (only for error checking)

            b = 0;
            for (const auto &band: labels) {

                prop = "emissivity_" + band;

                if (context->doesPrimitiveDataExist(UUID, prop.c_str())) {
                    context->getPrimitiveData(UUID, prop.c_str(), eps);
                } else {
                    eps = eps_default;
                }

                if (eps < 0) {
                    eps = 0.f;
                    if (message_flag) {
                        std::cout << "WARNING (RadiationModel): emissivity cannot be less than 0.  Clamping to 0 for band " << band << "." << std::endl;
                    }
                } else if (eps > 1.f) {
                    eps = 1.f;
                    if (message_flag) {
                        std::cout << "WARNING (RadiationModel): emissivity cannot be greater than 1.  Clamping to 1 for band " << band << "." << std::endl;
                    }
                }
                if( eps!=1 ){
                    scattering_iterations_needed.at(b) = true;
                }

                assert(doesBandExist(band));

                for (uint s = 0; s < Nsources; s++) {
                    if (radiation_bands.at(band).emissionFlag) { //emission enabled
                        if (eps != 1.f && rho.at(s).at(u).at(b) == 0 && tau.at(s).at(u).at(b) == 0) {
                            rho.at(s).at(u).at(b) = 1.f - eps;
                        } else if (eps + tau.at(s).at(u).at(b) + rho.at(s).at(u).at(b) != 1.f && eps > 0.f) {
                            helios_runtime_error(
                                    "ERROR (RadiationModel): emissivity, transmissivity, and reflectivity must sum to 1 to ensure energy conservation. Band " +
                                    band + ", Primitive #" + std::to_string(UUID) + ": eps=" +
                                    std::to_string(eps) + ", tau=" + std::to_string(tau.at(s).at(u).at(b)) +
                                    ", rho=" + std::to_string(rho.at(s).at(u).at(b)) +
                                    ". It is also possible that you forgot to disable emission for this band.");
                        } else if (radiation_bands.at(band).scatteringDepth == 0 && eps != 1.f) {
                            eps = 1.f;
                            rho.at(s).at(u).at(b) = 0.f;
                            tau.at(s).at(u).at(b) = 0.f;
                        }
                    } else if (tau.at(s).at(u).at(b) + rho.at(s).at(u).at(b) > 1.f) {
                        helios_runtime_error(
                                "ERROR (RadiationModel): transmissivity and reflectivity cannot sum to greater than 1 ensure energy conservation. Band " +
                                band + ", Primitive #" + std::to_string(UUID) + ": eps=" +
                                std::to_string(eps) + ", tau=" + std::to_string(tau.at(s).at(u).at(b)) +
                                ", rho=" + std::to_string(rho.at(s).at(u).at(b)) +
                                ". It is also possible that you forgot to disable emission for this band.");
                    }
                }
                b++;
            }
        }
    }

    initializeBuffer1Df(rho_RTbuffer, flatten(rho));
    initializeBuffer1Df(tau_RTbuffer, flatten(tau));

    initializeBuffer1Df(rho_cam_RTbuffer, flatten(rho_cam));
    initializeBuffer1Df(tau_cam_RTbuffer, flatten(tau_cam));

    if( message_flag ) {
        std::cout << "done\n";
    }

}

void RadiationModel::runBand( const std::string &label ) {
    std::vector<std::string> labels{label};
    runBand(labels);
}

void RadiationModel::runBand( const std::vector<std::string> &label ) {

    //----- VERIFICATIONS -----//

    //Check to make sure some geometry was added to the context
    if( context->getPrimitiveCount()==0 ){
        std::cerr << "WARNING (RadiationModel::runBand): No geometry was added to the context. There is nothing to simulate...exiting." << std::endl;
        return;
    }

    //Check to make sure geometry was built in OptiX
    if( !isgeometryinitialized ){
        updateGeometry();
    }

    for( const std::string &band : label ){
        if( !doesBandExist(band) ){
            helios_runtime_error("ERROR (RadiationModel::runBand): Cannot run band " + band + " because it is not a valid band. Use addRadiationBand() function to add the band.");
        }
    }

    //if there are no radiation sources in the simulation, add at least one but with zero fluxes
    if( radiation_sources.empty() ){
        addCollimatedRadiationSource();
    }

    for( const auto &band : label ) {
        if (!radiation_bands.at(band).radiativepropertiesinitialized) {
            updateRadiativeProperties(label);
            break;
        }
    }

    //Set the number of radiative bands
    size_t Nbands = label.size();

    RT_CHECK_ERROR( rtVariableSet1ui( Nbands_RTvariable, Nbands ) );

    //Run all bands by default
    std::vector<bool> band_launch_flag(Nbands);
    for( uint b=0; b<Nbands; b++ ){
        band_launch_flag.at(b) = 1;
    }
    initializeBuffer1Dbool( band_launch_flag_RTbuffer, band_launch_flag );

    //Set the number of Context primitives
    size_t Nobjects = primitiveID.size();
    size_t Nprimitives = context_UUIDs.size();
    RT_CHECK_ERROR( rtVariableSet1ui( Nprimitives_RTvariable, Nprimitives ) );

    //Set the random number seed
    RT_CHECK_ERROR( rtVariableSet1ui( random_seed_RTvariable, std::chrono::system_clock::now().time_since_epoch().count() ) );

    //Number of external radiation sources
    uint Nsources = radiation_sources.size();
    RT_CHECK_ERROR( rtVariableSet1ui( Nsources_RTvariable, Nsources ) );

    //Set periodic boundary condition (if applicable)
    RT_CHECK_ERROR( rtVariableSet2f( periodic_flag_RTvariable, periodic_flag.x, periodic_flag.y ));

    //Number of radiation cameras
    uint Ncameras = cameras.size();
    RT_CHECK_ERROR( rtVariableSet1ui( Ncameras_RTvariable, Ncameras ) );

    //Set scattering depth for each band
    std::vector<uint> scattering_depth(Nbands);
    bool scatteringenabled = false;
    for( auto b=0; b<Nbands; b++ ){
        scattering_depth.at(b) = radiation_bands.at(label.at(b)).scatteringDepth;
        if( scattering_depth.at(b)>0 ){
            scatteringenabled=true;
        }
    }
    initializeBuffer1Dui( max_scatters_RTbuffer, scattering_depth );

    //Issue warning if rho>0, tau>0, or eps<1
    for( int b=0; b<Nbands; b++ ){
        if( scattering_depth.at(b)==0 && scattering_iterations_needed.at(b) ){
            std::cout << "WARNING (RadiationModel::runBand): Surface radiative properties for band " << label.at(b) << " are set to non-default values, but scattering iterations are disabled. Surface radiative properties will be ignored unless scattering depth is non-zero." << std::endl;
        }
    }

    //Set diffuse flux for each band
    std::vector<float> diffuse_flux(Nbands);
    bool diffuseenabled = false;
    for( auto b=0; b<Nbands; b++ ){
        diffuse_flux.at(b) = radiation_bands.at(label.at(b)).diffuseFlux;
        if( diffuse_flux.at(b)>0.f ){
            diffuseenabled=true;
        }
    }
    initializeBuffer1Df( diffuse_flux_RTbuffer, diffuse_flux );

    //Set diffuse extinction coefficient for each band
    std::vector<float> diffuse_extinction(Nbands,0);
    if( diffuseenabled ) {
        for (auto b = 0; b < Nbands; b++) {
            diffuse_extinction.at(b) = radiation_bands.at(label.at(b)).diffuseExtinction;
        }
    }
    initializeBuffer1Df(diffuse_extinction_RTbuffer, diffuse_extinction);

    //Set diffuse distribution normalization factor for each band
    std::vector<float> diffuse_dist_norm(Nbands,0);
    if( diffuseenabled ) {
        for (auto b = 0; b < Nbands; b++) {
            diffuse_dist_norm.at(b) = radiation_bands.at(label.at(b)).diffuseDistNorm;
        }
        initializeBuffer1Df(diffuse_dist_norm_RTbuffer, diffuse_dist_norm);
    }

    //Set diffuse distribution peak direction for each band
    std::vector<optix::float3> diffuse_peak_dir(Nbands);
    if( diffuseenabled ) {
        for (auto b = 0; b < Nbands; b++) {
            helios::vec3 peak_dir = radiation_bands.at(label.at(b)).diffusePeakDir;
            diffuse_peak_dir.at(b) = optix::make_float3(peak_dir.x,peak_dir.y,peak_dir.z);
        }
        initializeBuffer1Dfloat3(diffuse_peak_dir_RTbuffer, diffuse_peak_dir);
    }

    //Determine whether emission is enabled for any band
    bool emissionenabled = false;
    for( auto b=0; b<Nbands; b++ ){
        if( radiation_bands.at(label.at(b)).emissionFlag ){
            emissionenabled=true;
        }
    }

    //Figure out the maximum direct ray count for all bands in this run and use this as the launch size
    size_t directRayCount = 0;
    for( const auto &band : label ) {
        if( radiation_bands.at(band).directRayCount>directRayCount ) {
            directRayCount = radiation_bands.at(band).directRayCount;
        }
    }

    //Figure out the maximum diffuse ray count for all bands in this run and use this as the launch size
    size_t diffuseRayCount = 0;
    for( const auto &band : label ) {
        if( radiation_bands.at(band).diffuseRayCount>diffuseRayCount ) {
            diffuseRayCount = radiation_bands.at(band).diffuseRayCount;
        }
    }

    //Figure out the maximum diffuse ray count for all bands in this run and use this as the launch size
    size_t scatteringDepth = 0;
    for( const auto &band : label ) {
        if( radiation_bands.at(band).scatteringDepth>scatteringDepth ) {
            scatteringDepth = radiation_bands.at(band).scatteringDepth;
        }
    }

    // Zero buffers
    zeroBuffer1D( radiation_in_RTbuffer, Nbands*Nprimitives );
    zeroBuffer1D( scatter_buff_top_RTbuffer, Nbands*Nprimitives );
    zeroBuffer1D( scatter_buff_bottom_RTbuffer, Nbands*Nprimitives );
    zeroBuffer1D( Rsky_RTbuffer, Nbands*Nprimitives );

    if( Ncameras>0 ){
        zeroBuffer1D( scatter_buff_top_cam_RTbuffer, Nbands*Nprimitives );
        zeroBuffer1D( scatter_buff_bottom_cam_RTbuffer, Nbands*Nprimitives );
    }

    std::vector<float> TBS_top, TBS_bottom;
    TBS_top.resize(Nbands*Nprimitives, 0);
    TBS_bottom = TBS_top;

    std::map<std::string,std::vector<std::vector<float> > > radiation_in_camera;

    size_t maxRays = 1024*1024*1024; //maximum number of total rays in a launch

    // ***** DIRECT LAUNCH FROM ALL RADIATION SOURCES ***** //

    optix::int3 launch_dim_dir;

    bool rundirect = false;
    for( uint s=0; s<Nsources; s++ ){
        for( uint b=0; b<Nbands; b++ ){
            if( radiation_sources.at(s).source_fluxes.at(label.at(b))>0.f){
                rundirect = true;
                break;
            }
        }
    }

    if( Nsources>0 && rundirect ){

        //update radiation source buffers

        std::vector<std::vector<float> > fluxes; //first index is the source, second index is the band (only those passed to runBand() function)
        fluxes.resize(Nsources);
        std::vector<optix::float3> positions(Nsources);
        std::vector<optix::float2> widths(Nsources);
        std::vector<optix::float3> rotations(Nsources);
        std::vector<uint> types(Nsources);

        size_t s=0;
        for( const auto &source : radiation_sources ) {

            fluxes.at(s).resize(Nbands);


            for( auto b=0; b<label.size(); b++ ) {
                if (source.source_fluxes.find(label.at(b)) == source.source_fluxes.end()) {
                    fluxes.at(s).at(b) = 0;
                } else {
                    fluxes.at(s).at(b) = source.source_fluxes.at(label.at(b));
                }
            }

            positions.at(s) = optix::make_float3( source.source_position.x, source.source_position.y, source.source_position.z );
            widths.at(s) = optix::make_float2(source.source_width.x, source.source_width.y);
            rotations.at(s) = optix::make_float3( source.source_rotation.x,source.source_rotation.y,source.source_rotation.z );
            types.at(s) = source.source_type;

            s++;
        }

        initializeBuffer1Df( source_fluxes_RTbuffer, flatten(fluxes) );
        initializeBuffer1Dfloat3( source_positions_RTbuffer, positions );
        initializeBuffer1Dfloat2( source_widths_RTbuffer, widths );
        initializeBuffer1Dfloat3( source_rotations_RTbuffer, rotations );
        initializeBuffer1Dui( source_types_RTbuffer, types );

        // -- Ray Trace -- //

        // Compute direct launch dimension
        size_t n = ceil(sqrt(double(directRayCount)));

        size_t maxPrims = floor( float(maxRays)/float(n*n) );

        int Nlaunches = ceil( n*n*Nobjects/float(maxRays) );

        size_t prims_per_launch = fmin( Nobjects, maxPrims );

        for( uint launch=0; launch<Nlaunches; launch++ ){

            size_t prims_this_launch;
            if( (launch+1)*prims_per_launch > Nobjects ){
                prims_this_launch = Nobjects-launch*prims_per_launch;
            }else{
                prims_this_launch = prims_per_launch;
            }

            RT_CHECK_ERROR( rtVariableSet1ui( launch_offset_RTvariable, launch*prims_per_launch ) );

            launch_dim_dir = optix::make_int3( round(n), round(n), prims_this_launch );

            if( message_flag ){
                std::cout << "Performing primary direct radiation ray trace for bands ";
                for( const auto &band : label ) {
                    std::cout << band << ", ";
                }
                std::cout << " (batch " << launch+1 << " of " << Nlaunches << ")..." << std::flush;
            }
            RT_CHECK_ERROR( rtContextLaunch3D( OptiX_Context, RAYTYPE_DIRECT , launch_dim_dir.x, launch_dim_dir.y, launch_dim_dir.z ) );

            if( message_flag ){
                std::cout << "\r                                                                                                                               \r" << std::flush;
            }

        }

        if( message_flag ){
            std::cout << "Performing primary direct radiation ray trace for bands ";
            for( const auto &band : label ) {
                std::cout << band << ", ";
            }
            std::cout << "...done." << std::endl;
        }

    }//end direct source launch

    // --- Diffuse/Emission launch ---- //

    if( emissionenabled || diffuseenabled ){

        std::vector<float> flux_top, flux_bottom;
        flux_top.resize(Nbands*Nprimitives,0);
        flux_bottom = flux_top;

        //If we are doing a diffuse/emission ray trace anyway and we have direct scattered energy, we get a "free" scattering trace here
        if( scatteringenabled && rundirect ){
            flux_top=getOptiXbufferData( scatter_buff_top_RTbuffer );
            flux_bottom=getOptiXbufferData( scatter_buff_bottom_RTbuffer );
            zeroBuffer1D( scatter_buff_top_RTbuffer, Nbands*Nprimitives );
            zeroBuffer1D( scatter_buff_bottom_RTbuffer, Nbands*Nprimitives );
        }

        //add any emitted energy to the outgoing energy buffer
        if( emissionenabled ) {
            // Update primitive outgoing emission
            float eps, temperature;

            void *ptr;
            float *scatter_buff_top_cam_data, *scatter_buff_bottom_cam_data;
            if( Ncameras>0 ){
                //add emitted flux to camera scattered energy buffer
                RT_CHECK_ERROR( rtBufferMap( scatter_buff_top_cam_RTbuffer, &ptr ) );
                scatter_buff_top_cam_data = (float*)ptr;
                RT_CHECK_ERROR( rtBufferMap( scatter_buff_bottom_cam_RTbuffer, &ptr ) );
                scatter_buff_bottom_cam_data = (float*)ptr;
            }

            for (auto b = 0; b < Nbands; b++) {
                if (radiation_bands.at(label.at(b)).emissionFlag) {
                    std::string prop = "emissivity_" + label.at(b);
                    for (size_t u = 0; u < Nprimitives; u++) {
                        size_t ind = u * Nbands + b;
                        uint p = context_UUIDs.at(u);
                        if (context->doesPrimitiveDataExist(p, prop.c_str())) {
                            context->getPrimitiveData(p, prop.c_str(), eps);
                        } else {
                            eps = eps_default;
                        }
                        if (scattering_depth.at(b) == 0 && eps != 1.f) {
                            eps = 1.f;
                        }
                        if (context->doesPrimitiveDataExist(p, "temperature")) {
                            context->getPrimitiveData(p, "temperature", temperature);
                            if (temperature < 0) {
                                temperature = temperature_default;
                            }
                        } else {
                            temperature = temperature_default;
                        }
                        float out_top = sigma * eps * pow(temperature, 4);
                        flux_top.at(ind) += out_top;
                        if( Ncameras>0 ) {
                            scatter_buff_top_cam_data[ind] += out_top;
                        }
                        if (!context->doesPrimitiveDataExist(p, "twosided_flag")) {
                            flux_bottom.at(ind) += flux_top.at(ind);
                            if( Ncameras>0 ) {
                                scatter_buff_bottom_cam_data[ind] += out_top;
                            }
                        } else {
                            uint flag;
                            context->getPrimitiveData(p, "twosided_flag", flag);
                            if (flag) {
                                flux_bottom.at(ind) += flux_top.at(ind);
                                if( Ncameras>0 ) {
                                    scatter_buff_bottom_cam_data[ind] += out_top;
                                }
                            }
                        }
                    }
                }
            }
            if( Ncameras>0 ){
                RT_CHECK_ERROR( rtBufferUnmap( scatter_buff_top_cam_RTbuffer ) );
                RT_CHECK_ERROR( rtBufferUnmap( scatter_buff_bottom_cam_RTbuffer ) );
            }
        }

        initializeBuffer1Df( radiation_out_top_RTbuffer, flux_top );
        initializeBuffer1Df( radiation_out_bottom_RTbuffer, flux_bottom );

        // Compute diffuse launch dimension
        size_t n = ceil(sqrt(double(diffuseRayCount)));

        size_t maxPrims = floor( float(maxRays)/float(n*n) );

        int Nlaunches = ceil( n*n*Nobjects/float(maxRays) );

        size_t prims_per_launch = fmin( Nobjects, maxPrims );

        for( uint launch=0; launch<Nlaunches; launch++ ){

            size_t prims_this_launch;
            if( (launch+1)*prims_per_launch > Nobjects ){
                prims_this_launch = Nobjects-launch*prims_per_launch;
            }else{
                prims_this_launch = prims_per_launch;
            }

            RT_CHECK_ERROR( rtVariableSet1ui( launch_offset_RTvariable, launch*prims_per_launch ) );

            optix::int3 launch_dim_diff = optix::make_int3( round(n), round(n), prims_this_launch );
            assert( launch_dim_diff.x>0 && launch_dim_diff.y>0 );

            if( message_flag ){
                std::cout << "Performing primary diffuse radiation ray trace for bands ";
                for( const auto &band : label ) {
                    std::cout << band << " ";
                }
                std::cout << " (batch " << launch+1 << " of " << Nlaunches << ")..." << std::flush;
            }
            RT_CHECK_ERROR( rtContextLaunch3D( OptiX_Context, RAYTYPE_DIFFUSE , launch_dim_diff.x, launch_dim_diff.y, launch_dim_diff.z ) );
            if( message_flag ){
                std::cout << "\r                                                                                                                               \r" << std::flush;
            }
        }

        if( message_flag ){
            std::cout << "Performing primary diffuse radiation ray trace for bands ";
            for( const auto &band : label ) {
                std::cout << band << ", ";
            }
            std::cout << "...done." << std::endl;
        }

    }

    if( scatteringenabled ){

        for( auto b=0; b<Nbands; b++ ) {
            diffuse_flux.at(b) = 0.f;
        }
        initializeBuffer1Df( diffuse_flux_RTbuffer, diffuse_flux );

        size_t n = ceil(sqrt(double(diffuseRayCount)));

        size_t maxPrims = floor( float(maxRays)/float(n*n) );

        int Nlaunches = ceil( n*n*Nobjects/float(maxRays) );

        size_t prims_per_launch = fmin( Nobjects, maxPrims );

        uint s;
        for( s=0; s<scatteringDepth; s++ ){
            if( message_flag ){
                std::cout << "Performing scattering ray trace (iteration " << s+1 << " of " << scatteringDepth << ")..." << std::flush;
            }

            for( uint b=0; b<Nbands; b++ ){
                uint depth = radiation_bands.at(label.at(b)).scatteringDepth;
                if( s+1>depth ) {
                    if( message_flag ) {
                        std::cout << "Skipping band " << label.at(b) << " for scattering launch " << s + 1 << std::flush;
                    }
                    band_launch_flag.at(b) = false;
                }else{
                    band_launch_flag.at(b) = true;
                }
            }
            initializeBuffer1Dbool( band_launch_flag_RTbuffer, band_launch_flag );

//            TBS_top=getOptiXbufferData( scatter_buff_top_RTbuffer );
//            TBS_bottom=getOptiXbufferData( scatter_buff_bottom_RTbuffer );
//            float TBS_max = 0;
//            for( size_t u=0; u<Nprimitives*Nbands; u++ ){
//                if( TBS_top.at(u)+TBS_bottom.at(u)>TBS_max ){
//                    TBS_max = TBS_top.at(u)+TBS_bottom.at(u);
//                }
//            }

            copyBuffer1D( scatter_buff_top_RTbuffer, radiation_out_top_RTbuffer );
            zeroBuffer1D( scatter_buff_top_RTbuffer, Nbands*Nprimitives );
            copyBuffer1D( scatter_buff_bottom_RTbuffer, radiation_out_bottom_RTbuffer );
            zeroBuffer1D( scatter_buff_bottom_RTbuffer, Nbands*Nprimitives );

            for( uint launch=0; launch<Nlaunches; launch++ ){

                size_t prims_this_launch;
                if( (launch+1)*prims_per_launch > Nobjects ){
                    prims_this_launch = Nobjects-launch*prims_per_launch;
                }else{
                    prims_this_launch = prims_per_launch;
                }
                optix::int3 launch_dim_diff = optix::make_int3( round(n), round(n), prims_this_launch );

                RT_CHECK_ERROR( rtVariableSet1ui( launch_offset_RTvariable, launch*prims_per_launch ) );

                RT_CHECK_ERROR( rtContextLaunch3D( OptiX_Context, RAYTYPE_DIFFUSE , launch_dim_diff.x, launch_dim_diff.y, launch_dim_diff.z ) );

            }
            if( message_flag ){
                std::cout << "\r                                                                                                                           \r" << std::flush;
            }

        }

        if( message_flag ){
            std::cout << "Performing scattering ray trace...done." << std::endl;
        }

    }

    // **** CAMERA RAY TRACE **** //
    if( Ncameras>0 ){

        //re-set outgoing radiation buffers
        copyBuffer1D( scatter_buff_top_cam_RTbuffer, radiation_out_top_RTbuffer );
        copyBuffer1D( scatter_buff_bottom_cam_RTbuffer, radiation_out_bottom_RTbuffer );

        //re-set diffuse radiation fluxes
        if( diffuseenabled ){
            for( auto b=0; b<Nbands; b++ ){
                diffuse_flux.at(b) = radiation_bands.at(label.at(b)).diffuseFlux;
            }
            initializeBuffer1Df( diffuse_flux_RTbuffer, diffuse_flux );
        }


        size_t n = ceil(sqrt(double(diffuseRayCount)));

        uint cam=0;
        for( auto & camera : cameras ){

            //set variable values
            RT_CHECK_ERROR( rtVariableSet3f( camera_position_RTvariable, camera.second.position.x, camera.second.position.y, camera.second.position.z ) );
            helios::SphericalCoord dir = cart2sphere( camera.second.lookat - camera.second.position );
            RT_CHECK_ERROR( rtVariableSet2f( camera_direction_RTvariable, dir.zenith, dir.azimuth ) );
            RT_CHECK_ERROR( rtVariableSet1f( camera_lens_diameter_RTvariable, camera.second.lens_diameter ) );
            RT_CHECK_ERROR( rtVariableSet2f( sensor_size_RTvariable, camera.second.sensor_size.x, camera.second.sensor_size.y ) );
            RT_CHECK_ERROR( rtVariableSet1f( camera_focal_length_RTvariable, camera.second.focal_length ) );
            RT_CHECK_ERROR( rtVariableSet1f( camera_viewplane_length_RTvariable, 0.5f*camera.second.sensor_size.x/tanf(camera.second.HFOV_degrees*M_PI/180.f) ) );
            RT_CHECK_ERROR( rtVariableSet1ui( camera_ID_RTvariable, cam ));

            zeroBuffer1D( radiation_in_camera_RTbuffer, camera.second.resolution.x*camera.second.resolution.y*Nbands );

            optix::int3 launch_dim_camera = optix::make_int3( camera.second.antialiasing_samples, camera.second.resolution.x, camera.second.resolution.y );

            if( message_flag ){
                std::cout << "Performing scattering radiation camera ray trace for camera " << camera.second.label << "..." << std::flush;
            }
            RT_CHECK_ERROR( rtContextLaunch3D( OptiX_Context, RAYTYPE_CAMERA , launch_dim_camera.x, launch_dim_camera.y, launch_dim_camera.z ) );
            if( message_flag ){
                std::cout << "done." << std::endl;
            }

            std::vector<float> radiation_camera = getOptiXbufferData(radiation_in_camera_RTbuffer);

            std::string camera_label = camera.second.label;

            for( auto b=0; b<Nbands; b++ ) {

                camera.second.pixel_data[label.at(b)].resize(camera.second.resolution.x*camera.second.resolution.y);

                std::string data_label = "camera_" + camera_label + "_" + label.at(b);

                for( auto p=0; p<camera.second.resolution.x*camera.second.resolution.y; p++ ){
                    camera.second.pixel_data.at(label.at(b)).at(p) = radiation_camera.at(p*Nbands+b);
                }

                context->setGlobalData(data_label.c_str(), HELIOS_TYPE_FLOAT,camera.second.resolution.x*camera.second.resolution.y,&camera.second.pixel_data.at(label.at(b))[0]);

            }

            //--- Pixel Labeling Trace ---//

            zeroBuffer1D(camera_pixel_label_RTbuffer,camera.second.resolution.x*camera.second.resolution.y);

            if( message_flag ){
                std::cout << "Performing camera pixel labeling ray trace for camera " << camera.second.label << "..." << std::flush;
            }
            RT_CHECK_ERROR( rtContextLaunch3D( OptiX_Context, RAYTYPE_PIXEL_LABEL , 1, launch_dim_camera.y, launch_dim_camera.z ) );
            if( message_flag ){
                std::cout << "done." << std::endl;
            }

            camera.second.pixel_label_UUID = getOptiXbufferData_ui(camera_pixel_label_RTbuffer);

            //the IDs from the ray trace do not necessarily correspond to the actual primitive UUIDs, so look them up.
            for( uint ID=0; ID<camera.second.pixel_label_UUID.size(); ID++ ){
                if( camera.second.pixel_label_UUID.at(ID)>0 ){
                    camera.second.pixel_label_UUID.at(ID) = context_UUIDs.at(camera.second.pixel_label_UUID.at(ID) - 1) + 1;
                }
            }

            std::string data_label = "camera_" + camera_label + "_pixel_UUID";

            context->setGlobalData(data_label.c_str(), HELIOS_TYPE_UINT,camera.second.resolution.x*camera.second.resolution.y,&camera.second.pixel_label_UUID[0]);

            cam++;
        }
    }

    //deposit any energy that is left to make sure we satisfy conservation of energy
    TBS_top=getOptiXbufferData( scatter_buff_top_RTbuffer );
    TBS_bottom=getOptiXbufferData( scatter_buff_bottom_RTbuffer );

    // Set variables in geometric objects

    std::vector<float> radiation_flux_data;
    radiation_flux_data=getOptiXbufferData( radiation_in_RTbuffer );

    std::vector<uint> UUIDs_context_all = context->getAllUUIDs();

    for( auto b=0; b<Nbands; b++ ) {

        std::string prop = "radiation_flux_" + label.at(b);
        for (size_t u = 0; u < Nprimitives; u++) {
            size_t ind = u * Nbands + b;
            float R = radiation_flux_data.at(ind) + TBS_top.at(ind) + TBS_bottom.at(ind);
            context->setPrimitiveData(context_UUIDs.at(u), prop.c_str(), R);
            if (radiation_flux_data.at(ind) != radiation_flux_data.at(ind)) {
                std::cout << "NaN here " << ind << std::endl;
            }
        }

        if( UUIDs_context_all.size()!=Nprimitives ){
            for( uint UUID : UUIDs_context_all ){
                if( context->doesPrimitiveExist(UUID) && !context->doesPrimitiveDataExist(UUID,prop.c_str()) ){
                    context->setPrimitiveData(UUID,prop.c_str(),0.f);
                }
            }
        }

    }

}

float RadiationModel::getSkyEnergy() {

    std::vector<float> Rsky_SW;
    Rsky_SW=getOptiXbufferData( Rsky_RTbuffer );
    float Rsky=0.f;
    for( size_t i=0; i<Rsky_SW.size(); i++ ){
        Rsky += Rsky_SW.at(i);
    }
    return Rsky;

}

std::vector<float> RadiationModel::getTotalAbsorbedFlux() {

    std::vector<float> total_flux;
    total_flux.resize(context->getPrimitiveCount(),0.f);

    for( const auto &band : radiation_bands ){

        std::string label = band.first;

        for( size_t u=0; u<context_UUIDs.size(); u++ ){

            uint p = context_UUIDs.at(u);

            std::string str = "radiation_flux_" + label;

            float R;
            context->getPrimitiveData( p, str.c_str(), R );
            total_flux.at(u) += R;

        }

    }

    return total_flux;

}

std::vector<float> RadiationModel::getOptiXbufferData( RTbuffer buffer ){

    void* _data_;
    RT_CHECK_ERROR( rtBufferMap( buffer, &_data_ ) );
    float* data_ptr = (float*)_data_;

    RTsize size;
    RT_CHECK_ERROR( rtBufferGetSize1D( buffer, &size ) );

    std::vector<float> data_vec;
    data_vec.resize(size);
    for( int i=0; i<size; i++ ){
        data_vec.at(i)=data_ptr[i];
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

    return data_vec;

}

std::vector<double> RadiationModel::getOptiXbufferData_d( RTbuffer buffer ){

    void* _data_;
    RT_CHECK_ERROR( rtBufferMap( buffer, &_data_ ) );
    double* data_ptr = (double*)_data_;

    RTsize size;
    RT_CHECK_ERROR( rtBufferGetSize1D( buffer, &size ) );

    std::vector<double> data_vec;
    data_vec.resize(size);
    for( int i=0; i<size; i++ ){
        data_vec.at(i)=data_ptr[i];
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

    return data_vec;

}

std::vector<uint> RadiationModel::getOptiXbufferData_ui( RTbuffer buffer ){

    void* _data_;
    RT_CHECK_ERROR( rtBufferMap( buffer, &_data_ ) );
    uint* data_ptr = (uint*)_data_;

    RTsize size;
    RT_CHECK_ERROR( rtBufferGetSize1D( buffer, &size ) );

    std::vector<uint> data_vec;
    data_vec.resize(size);
    for( int i=0; i<size; i++ ){
        data_vec.at(i)=data_ptr[i];
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

    return data_vec;

}

void RadiationModel::addBuffer( const char* name, RTbuffer& buffer, RTvariable& variable, RTbuffertype type, RTformat format, size_t dimension ){

    RT_CHECK_ERROR( rtBufferCreate( OptiX_Context, type, &buffer ) );
    RT_CHECK_ERROR( rtBufferSetFormat( buffer, format ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( OptiX_Context, name, &variable ) );
    RT_CHECK_ERROR( rtVariableSetObject( variable, buffer ) );
    if( dimension==1 ){
        zeroBuffer1D( buffer, 1 );
    }else if( dimension==2 ){
        zeroBuffer2D( buffer, optix::make_int2(1,1) );
    }else{
        helios_runtime_error("ERROR (RadiationModel::addBuffer): invalid buffer dimension of " + std::to_string(dimension) + ", must be 1 or 2.");
    }

}

void RadiationModel::zeroBuffer1D(RTbuffer &buffer, size_t bsize ){

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    if( format==RT_FORMAT_USER ){//Note: for now, assume user format means it's a double

        std::vector<double> array;
        array.resize(bsize);
        for( int i=0; i<bsize; i++ ){
            array.at(i)=0.f;
        }

        initializeBuffer1Dd( buffer, array );

    }else if( format==RT_FORMAT_FLOAT ){

        std::vector<float> array;
        array.resize(bsize);
        for( int i=0; i<bsize; i++ ){
            array.at(i)=0.f;
        }

        initializeBuffer1Df( buffer, array );

    }else if( format==RT_FORMAT_FLOAT2 ){

        std::vector<optix::float2> array;
        array.resize(bsize);
        for( int i=0; i<bsize; i++ ){
            array.at(i)=optix::make_float2(0,0);
        }

        initializeBuffer1Dfloat2( buffer, array );

    }else if( format==RT_FORMAT_FLOAT3 ){

        std::vector<optix::float3> array;
        array.resize(bsize);
        for( int i=0; i<bsize; i++ ){
            array.at(i)=optix::make_float3(0,0,0);
        }

        initializeBuffer1Dfloat3( buffer, array );

    }else if( format==RT_FORMAT_INT ){

        std::vector<int> array;
        array.resize(bsize);
        for( int i=0; i<bsize; i++ ){
            array.at(i)=0;
        }

        initializeBuffer1Di( buffer, array );

    }else if( format==RT_FORMAT_INT2 ){

        std::vector<optix::int2> array;
        array.resize(bsize);
        for( int i=0; i<bsize; i++ ){
            array.at(i)=optix::make_int2(0,0);
        }

        initializeBuffer1Dint2( buffer, array );

    }else if( format==RT_FORMAT_INT3 ){

        std::vector<optix::int3> array;
        array.resize(bsize);
        for( int i=0; i<bsize; i++ ){
            array.at(i)=optix::make_int3(0,0,0);
        }

        initializeBuffer1Dint3( buffer, array );

    }else if( format==RT_FORMAT_UNSIGNED_INT ){

        std::vector<uint> array;
        array.resize(bsize);
        for( int i=0; i<bsize; i++ ){
            array.at(i)=0;
        }

        initializeBuffer1Dui( buffer, array );

    }else if( format==RT_FORMAT_BYTE ){

        std::vector<bool> array;
        array.resize(bsize);
        for( int i=0; i<bsize; i++ ){
            array.at(i)=false;
        }

        initializeBuffer1Dbool( buffer, array );
    }else{
        helios_runtime_error("ERROR (RadiationModel::zeroBuffer1D): Buffer type not supported.");
    }

}

void RadiationModel::copyBuffer1D( RTbuffer &buffer, RTbuffer &buffer_copy ){

  /* \todo Add support for all data types (currently only works for float and float3)*/

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    //get buffer size
    RTsize bsize;
    rtBufferGetSize1D( buffer, &bsize );

    rtBufferSetSize1D( buffer_copy, bsize );

    if( format==RT_FORMAT_FLOAT ){

        void* ptr;
        RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );
        float* data = (float*)ptr;

        void* ptr_copy;
        RT_CHECK_ERROR( rtBufferMap( buffer_copy, &ptr_copy ) );
        float* data_copy = (float*)ptr_copy;

        for( size_t i = 0; i <bsize; i++ ) {
            data_copy[i] = data[i];
        }

    }else if( format==RT_FORMAT_FLOAT3 ){

        void* ptr;
        RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );
        optix::float3* data = (optix::float3*)ptr;

        void* ptr_copy;
        RT_CHECK_ERROR( rtBufferMap( buffer_copy, &ptr_copy ) );
        optix::float3* data_copy = (optix::float3*)ptr_copy;

        for( size_t i = 0; i <bsize; i++ ) {
            data_copy[i] = data[i];
        }

    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );
    RT_CHECK_ERROR( rtBufferUnmap( buffer_copy ) );

}

void RadiationModel::initializeBuffer1Dd(RTbuffer &buffer, const std::vector<double> &array ){

    size_t bsize = array.size();

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    if( format!=RT_FORMAT_USER ){
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer1Dd): Buffer must have type double.");
    }

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    double* data = (double*)ptr;

    for( size_t i = 0; i <bsize; i++ ) {
        data[i] = array[i];
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer1Df(RTbuffer &buffer, const std::vector<float> &array ){

    size_t bsize = array.size();

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    if( format!=RT_FORMAT_FLOAT ){
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer1Df): Buffer must have type float.");
    }

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    float* data = (float*)ptr;

    for( size_t i = 0; i <bsize; i++ ) {
        data[i] = array[i];
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer1Dfloat2(RTbuffer &buffer, const std::vector<optix::float2> &array ){

    size_t bsize = array.size();

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    if( format!=RT_FORMAT_FLOAT2 ){
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer1Dfloat2): Buffer must have type float2.");
    }

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    optix::float2* data = (optix::float2*)ptr;

    for( size_t i = 0; i <bsize; i++ ) {
        data[i].x = array[i].x;
        data[i].y = array[i].y;
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer1Dfloat3(RTbuffer &buffer, const std::vector<optix::float3> &array ){

    size_t bsize = array.size();

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    if( format!=RT_FORMAT_FLOAT3 ){
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer1Dfloat3): Buffer must have type float3.");
    }

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    optix::float3* data = (optix::float3*)ptr;

    for( size_t i = 0; i <bsize; i++ ) {
        data[i].x = array[i].x;
        data[i].y = array[i].y;
        data[i].z = array[i].z;
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer1Dfloat4(RTbuffer &buffer, const std::vector<optix::float4> &array ){

    size_t bsize = array.size();

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    if( format!=RT_FORMAT_FLOAT4 ){
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer1Dfloat4): Buffer must have type float4.");
    }

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    optix::float4* data = (optix::float4*)ptr;

    for( size_t i = 0; i <bsize; i++ ) {
        data[i].x = array[i].x;
        data[i].y = array[i].y;
        data[i].z = array[i].z;
        data[i].w = array[i].w;
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer1Di(RTbuffer &buffer, const std::vector<int> &array ){

    size_t bsize = array.size();

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    if( format!=RT_FORMAT_INT ){
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer1Di): Buffer must have type int.");
    }

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    int* data = (int*)ptr;

    for( size_t i = 0; i <bsize; i++ ) {
        data[i] = array[i];
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer1Dui(RTbuffer &buffer, const std::vector<uint> &array ){

    size_t bsize = array.size();

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    if( format!=RT_FORMAT_UNSIGNED_INT ){
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer1Dui): Buffer must have type unsigned int.");
    }

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    uint* data = (uint*)ptr;

    for( size_t i = 0; i <bsize; i++ ) {
        data[i] = array[i];
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer1Dint2(RTbuffer &buffer, const std::vector<optix::int2> &array ){

    size_t bsize = array.size();

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    if( format!=RT_FORMAT_INT2 ){
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer1Dint2): Buffer must have type int2.");
    }

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    optix::int2* data = (optix::int2*)ptr;

    for( size_t i = 0; i <bsize; i++ ) {
        data[i] = array[i];
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer1Dint3(RTbuffer &buffer, const std::vector<optix::int3> &array ){

    size_t bsize = array.size();

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    if( format!=RT_FORMAT_INT3 ){
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer1Dint3): Buffer must have type int3.");
    }

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    optix::int3* data = (optix::int3*)ptr;

    for( size_t i = 0; i <bsize; i++ ) {
        data[i] = array[i];
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer1Dbool(RTbuffer &buffer, const std::vector<bool> &array ){

    size_t bsize = array.size();

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize1D( buffer, bsize ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    if( format!=RT_FORMAT_BYTE ){
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer1Dbool): Buffer must have type bool.");
    }

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    bool* data = (bool*)ptr;

    for( size_t i = 0; i <bsize; i++ ) {
        data[i] = array[i];
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::zeroBuffer2D(RTbuffer &buffer, optix::int2 bsize ){

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    if( format==RT_FORMAT_USER ){//Note: for now we'll assume this means it's a double
        std::vector<std::vector<double> > array;
        array.resize(bsize.y);
        for( int j=0; j<bsize.y; j++ ){
            array.at(j).resize(bsize.x);
            for( int i=0; i<bsize.x; i++ ){
                array.at(j).at(i)=0.f;
            }
        }
        initializeBuffer2Dd( buffer, array );
    }else if( format==RT_FORMAT_FLOAT ){
        std::vector<std::vector<float> > array;
        array.resize(bsize.y);
        for( int j=0; j<bsize.y; j++ ){
            array.at(j).resize(bsize.x);
            for( int i=0; i<bsize.x; i++ ){
                array.at(j).at(i)=0.f;
            }
        }
        initializeBuffer2Df( buffer, array );
    }else if( format==RT_FORMAT_FLOAT2 ){
        std::vector<std::vector<optix::float2> > array;
        array.resize(bsize.y);
        for( int j=0; j<bsize.y; j++ ){
            array.at(j).resize(bsize.x);
            for( int i=0; i<bsize.x; i++ ){
                array.at(j).at(i).x=0.f;
                array.at(j).at(i).y=0.f;
            }
        }
        initializeBuffer2Dfloat2( buffer, array );
    }else if( format==RT_FORMAT_FLOAT3 ){
        std::vector<std::vector<optix::float3> > array;
        array.resize(bsize.y);
        for( int j=0; j<bsize.y; j++ ){
            array.at(j).resize(bsize.x);
            for( int i=0; i<bsize.x; i++ ){
                array.at(j).at(i).x=0.f;
                array.at(j).at(i).y=0.f;
                array.at(j).at(i).z=0.f;
            }
        }
        initializeBuffer2Dfloat3( buffer, array );
    }else if( format==RT_FORMAT_FLOAT4 ){
        std::vector<std::vector<optix::float4> > array;
        array.resize(bsize.y);
        for( int j=0; j<bsize.y; j++ ){
            array.at(j).resize(bsize.x);
            for( int i=0; i<bsize.x; i++ ){
                array.at(j).at(i).x=0.f;
                array.at(j).at(i).y=0.f;
                array.at(j).at(i).z=0.f;
                array.at(j).at(i).w=0.f;
            }
        }
        initializeBuffer2Dfloat4( buffer, array );
    }else if( format==RT_FORMAT_INT ){
        std::vector<std::vector<int> > array;
        array.resize(bsize.y);
        for( int j=0; j<bsize.y; j++ ){
            array.at(j).resize(bsize.x);
            for( int i=0; i<bsize.x; i++ ){
                array.at(j).at(i)=0;
            }
        }
        initializeBuffer2Di( buffer, array );
    }else if( format==RT_FORMAT_UNSIGNED_INT ){
        std::vector<std::vector<uint> > array;
        array.resize(bsize.y);
        for( int j=0; j<bsize.y; j++ ){
            array.at(j).resize(bsize.x);
            for( int i=0; i<bsize.x; i++ ){
                array.at(j).at(i)=0;
            }
        }
        initializeBuffer2Dui( buffer, array );
    }else if( format==RT_FORMAT_INT2 ){
        std::vector<std::vector<optix::int2> > array;
        array.resize(bsize.y);
        for( int j=0; j<bsize.y; j++ ){
            array.at(j).resize(bsize.x);
            for( int i=0; i<bsize.x; i++ ){
                array.at(j).at(i).x=0;
                array.at(j).at(i).y=0;
            }
        }
        initializeBuffer2Dint2( buffer, array );
    }else if( format==RT_FORMAT_INT3 ){
        std::vector<std::vector<optix::int3> > array;
        array.resize(bsize.y);
        for( int j=0; j<bsize.y; j++ ){
            array.at(j).resize(bsize.x);
            for( int i=0; i<bsize.x; i++ ){
                array.at(j).at(i).x=0;
                array.at(j).at(i).y=0;
                array.at(j).at(i).z=0;
            }
        }
        initializeBuffer2Dint3( buffer, array );
    }else if( format==RT_FORMAT_BYTE ){
        std::vector<std::vector<bool> > array;
        array.resize(bsize.y);
        for( int j=0; j<bsize.y; j++ ){
            array.at(j).resize(bsize.x);
            for( int i=0; i<bsize.x; i++ ){
                array.at(j).at(i)=false;
            }
        }
        initializeBuffer2Dbool( buffer, array );
    }else{
        helios_runtime_error("ERROR (RadiationModel::zeroBuffer2D): unknown buffer format.");
    }

}

void RadiationModel::initializeBuffer2Dd(RTbuffer &buffer, const std::vector<std::vector<double>> &array ){

    optix::int2 bsize;
    bsize.y = array.size();
    if( bsize.y==0 ){
        bsize.x = 0;
    }else{
        bsize.x = array.front().size();
    }

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

    //get buffer formatsyn
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    if( format==RT_FORMAT_USER ){
        double* data = (double*)ptr;
        for( size_t j = 0; j<bsize.y; j++ ) {
            for( size_t i = 0; i<bsize.x; i++ ) {
                data[i+j*bsize.x] = array[j][i];
            }
        }
    }else{
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer2Dd): Buffer does not have format 'RT_FORMAT_USER'.");
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer2Df(RTbuffer &buffer, const std::vector<std::vector<float>> &array ){

    optix::int2 bsize;
    bsize.y = array.size();
    if( bsize.y==0 ){
        bsize.x = 0;
    }else{
        bsize.x = array.front().size();
    }

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    if( format==RT_FORMAT_FLOAT ){
        float* data = (float*)ptr;
        for( size_t j = 0; j<bsize.y; j++ ) {
            for( size_t i = 0; i<bsize.x; i++ ) {
                data[i+j*bsize.x] = array[j][i];
            }
        }
    }else{
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer2Df): Buffer does not have format 'RT_FORMAT_FLOAT'.");
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer2Dfloat2(RTbuffer &buffer, const std::vector<std::vector<optix::float2>> &array ){

    optix::int2 bsize;
    bsize.y = array.size();
    if( bsize.y==0 ){
        bsize.x = 0;
    }else{
        bsize.x = array.front().size();
    }

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    if( format==RT_FORMAT_FLOAT2 ){
        optix::float2* data = (optix::float2*)ptr;
        for( size_t j = 0; j<bsize.y; j++ ) {
            for( size_t i = 0; i<bsize.x; i++ ) {
                data[i+j*bsize.x].x = array[j][i].x;
                data[i+j*bsize.x].y = array[j][i].y;
            }
        }
    }else{
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer2Dfloat2): Buffer does not have format 'RT_FORMAT_FLOAT2'.");
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer2Dfloat3(RTbuffer &buffer, const std::vector<std::vector<optix::float3>> &array ){

    optix::int2 bsize;
    bsize.y = array.size();
    if( bsize.y==0 ){
        bsize.x = 0;
    }else{
        bsize.x = array.front().size();
    }

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    if( format==RT_FORMAT_FLOAT3 ){
        optix::float3* data = (optix::float3*)ptr;
        for( size_t j = 0; j<bsize.y; j++ ) {
            for( size_t i = 0; i<bsize.x; i++ ) {
                data[i+j*bsize.x].x = array.at(j).at(i).x;
                data[i+j*bsize.x].y = array.at(j).at(i).y;
                data[i+j*bsize.x].z = array.at(j).at(i).z;
            }
        }
    }else{
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer2Dfloat3): Buffer does not have format 'RT_FORMAT_FLOAT3'.");
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer2Dfloat4(RTbuffer &buffer, const std::vector<std::vector<optix::float4>> &array ){

    optix::int2 bsize;
    bsize.y = array.size();
    if( bsize.y==0 ){
        bsize.x = 0;
    }else{
        bsize.x = array.front().size();
    }

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    if( format==RT_FORMAT_FLOAT4 ){
        optix::float4* data = (optix::float4*)ptr;
        for( size_t j = 0; j<bsize.y; j++ ) {
            for( size_t i = 0; i<bsize.x; i++ ) {
                data[i+j*bsize.x].x = array[j][i].x;
                data[i+j*bsize.x].y = array[j][i].y;
                data[i+j*bsize.x].z = array[j][i].z;
                data[i+j*bsize.x].w = array[j][i].w;
            }
        }
    }else{
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer2Dfloat4): Buffer does not have format 'RT_FORMAT_FLOAT4'.");
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer2Di(RTbuffer &buffer, const std::vector<std::vector<int>> &array ){

    optix::int2 bsize;
    bsize.y = array.size();
    if( bsize.y==0 ){
        bsize.x = 0;
    }else{
        bsize.x = array.front().size();
    }

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    if( format==RT_FORMAT_INT ){
        int* data = (int*)ptr;
        for( size_t j = 0; j<bsize.y; j++ ) {
            for( size_t i = 0; i<bsize.x; i++ ) {
                data[i+j*bsize.x] = array[j][i];
            }
        }
    }else{
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer2Di): Buffer does not have format 'RT_FORMAT_INT'.");
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer2Dui(RTbuffer &buffer, const std::vector<std::vector<uint>> &array ){

    optix::int2 bsize;
    bsize.y = array.size();
    if( bsize.y==0 ){
        bsize.x = 0;
    }else{
        bsize.x = array.front().size();
    }

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    if( format==RT_FORMAT_UNSIGNED_INT ){
        uint* data = (uint*)ptr;
        for( size_t j = 0; j<bsize.y; j++ ) {
            for( size_t i = 0; i<bsize.x; i++ ) {
                data[i+j*bsize.x] = array[j][i];
            }
        }
    }else{
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer2Dui): Buffer does not have format 'RT_FORMAT_UNSIGNED_INT'.");
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer2Dint2(RTbuffer &buffer, const std::vector<std::vector<optix::int2>> &array ){

    optix::int2 bsize;
    bsize.y = array.size();
    if( bsize.y==0 ){
        bsize.x = 0;
    }else{
        bsize.x = array.front().size();
    }

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    if( format==RT_FORMAT_INT2 ){
        optix::int2* data = (optix::int2*)ptr;
        for( size_t j = 0; j<bsize.y; j++ ) {
            for( size_t i = 0; i<bsize.x; i++ ) {
                data[i+j*bsize.x].x = array[j][i].x;
                data[i+j*bsize.x].y = array[j][i].y;
            }
        }
    }else{
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer2Dint2): Buffer does not have format 'RT_FORMAT_INT2'.");
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer2Dint3(RTbuffer &buffer, const std::vector<std::vector<optix::int3>> &array ){

    optix::int2 bsize;
    bsize.y = array.size();
    if( bsize.y==0 ){
        bsize.x = 0;
    }else{
        bsize.x = array.front().size();
    }

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    if( format==RT_FORMAT_INT3 ){
        optix::int3* data = (optix::int3*)ptr;
        for( size_t j = 0; j<bsize.y; j++ ) {
            for( size_t i = 0; i<bsize.x; i++ ) {
                data[i+j*bsize.x].x = array[j][i].x;
                data[i+j*bsize.x].y = array[j][i].y;
                data[i+j*bsize.x].z = array[j][i].z;
            }
        }
    }else{
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer2Dint3): Buffer does not have format 'RT_FORMAT_INT3'.");
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

void RadiationModel::initializeBuffer2Dbool(RTbuffer &buffer, const std::vector<std::vector<bool>> &array ){

    optix::int2 bsize;
    bsize.y = array.size();
    if( bsize.y==0 ){
        bsize.x = 0;
    }else{
        bsize.x = array.front().size();
    }

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize2D( buffer, bsize.x, bsize.y ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    if( format==RT_FORMAT_BYTE ){
        bool* data = (bool*)ptr;
        for( size_t j = 0; j<bsize.y; j++ ) {
            for( size_t i = 0; i<bsize.x; i++ ) {
                data[i+j*bsize.x] = array[j][i];
            }
        }
    }else{
        helios_runtime_error("ERROR (RadiationModel::initializeBuffer2Dbool): Buffer does not have format 'RT_FORMAT_BYTE'.");
    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );

}

template <typename anytype>
void RadiationModel::initializeBuffer3D(RTbuffer &buffer, const std::vector<std::vector<std::vector<anytype>>> &array ){

    optix::int3 bsize;
    bsize.z = array.size();
    if( bsize.z==0 ){
        bsize.y=0;
        bsize.x=0;
    }else{
        bsize.y=array.front().size();
        if( bsize.y==0 ){
            bsize.x=0;
        }else{
            bsize.x = array.front().front().size();
        }
    }

    //set buffer size
    RT_CHECK_ERROR( rtBufferSetSize3D( buffer, bsize.x, bsize.y, bsize.z ) );

    //get buffer format
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat( buffer, &format ) );

    //zero out buffer
    void* ptr;
    RT_CHECK_ERROR( rtBufferMap( buffer, &ptr ) );

    if( format==RT_FORMAT_FLOAT ){
        float* data = (float*)ptr;
        for( size_t k = 0; k<bsize.z; k++ ) {
            for( size_t j = 0; j<bsize.y; j++ ) {
                for( size_t i = 0; i<bsize.x; i++ ) {
                    data[i+j*bsize.x+k*bsize.y*bsize.x] = array[k][j][i];
                }
            }
        }
    }else if( format==RT_FORMAT_INT ){
        int* data = (int*)ptr;
        for( size_t k = 0; k<bsize.z; k++ ) {
            for( size_t j = 0; j<bsize.y; j++ ) {
                for( size_t i = 0; i<bsize.x; i++ ) {
                    data[i+j*bsize.x+k*bsize.y*bsize.x] = array[k][j][i];
                }
            }
        }
    }else if( format==RT_FORMAT_UNSIGNED_INT ){
        uint* data = (uint*)ptr;
        for( size_t k = 0; k<bsize.z; k++ ) {
            for( size_t j = 0; j<bsize.y; j++ ) {
                for( size_t i = 0; i<bsize.x; i++ ) {
                    data[i+j*bsize.x+k*bsize.y*bsize.x] = array[k][j][i];
                }
            }
        }
    }else if( format==RT_FORMAT_BYTE ){
        bool* data = (bool*)ptr;
        for( size_t k = 0; k<bsize.z; k++ ) {
            for( size_t j = 0; j<bsize.y; j++ ) {
                for( size_t i = 0; i<bsize.x; i++ ) {
                    data[i+j*bsize.x+k*bsize.y*bsize.x] = array[k][j][i];
                }
            }
        }
    }else{
        std::cerr<< "ERROR (RadiationModel::initializeBuffer3D): unsupported buffer format." << std::endl;

    }

    RT_CHECK_ERROR( rtBufferUnmap( buffer ) );


}

float RadiationModel::calculateGtheta(helios::Context* context, vec3 view_direction ){

    vec3 dir = view_direction;
    dir.normalize();

    float Gtheta = 0;
    float total_area = 0;
    for( std::size_t u=0; u<primitiveID.size(); u++ ){

        uint UUID = context_UUIDs.at(primitiveID.at(u));

        vec3 normal = context->getPrimitiveNormal(UUID);
        float area = context->getPrimitiveArea(UUID);

        Gtheta += fabsf(normal*dir)*area;

        total_area += area;

    }

    return Gtheta/total_area;

}

void RadiationModel::setCameraCalibration(CameraCalibration *CameraCalibration){
    cameracalibration = CameraCalibration;
    calibration_flag = true;
}

void RadiationModel::updateCameraResponse(const std::string &orginalcameralabel, const std::vector<std::string> &sourcelabels_raw,
                                          const std::vector<std::string>& cameraresponselabels, vec2 &wavelengthrange,
                                          const std::vector<std::vector<float>> &truevalues, const std::string &calibratedmark) {

    std::vector<std::string> objectlabels;
    vec2 wavelengthrange_c = wavelengthrange;
    cameracalibration->preprocessSpectra(sourcelabels_raw, cameraresponselabels, objectlabels, wavelengthrange_c);

    RadiationCamera calibratecamera = cameras.at(orginalcameralabel);
    CameraProperties cameraproperties;
    cameraproperties.HFOV = calibratecamera.HFOV_degrees;
    cameraproperties.camera_resolution = calibratecamera.resolution;
    cameraproperties.focal_plane_distance = calibratecamera.focal_length;
    cameraproperties.lens_diameter = calibratecamera.lens_diameter;
    cameraproperties.sensor_size = calibratecamera.sensor_size;

    std::vector<uint> UUIDs_target = cameracalibration->getColorBoardUUIDs();
    std::string cameralabel = "calibration";
    std::map<uint,std::vector<vec2>> simulatedcolorboardspectra;
    for (uint UUID:UUIDs_target){
        simulatedcolorboardspectra.emplace(UUID, NULL);
    }

    for (uint ID = 0; ID<radiation_sources.size();ID++) {
        RadiationModel::setSourceSpectrumIntegral(ID, 1);
    }

    std::vector<float> wavelengths;
    context->getGlobalData("wavelengths", wavelengths);
    int numberwavelengths= wavelengths.size();

    for (int iw=0; iw < numberwavelengths; iw++) {
        std::string wavelengthlabel = std::to_string(wavelengths.at(iw));

        std::vector<std::string> sourcelabels;
        for (std::string sourcelabel_raw:sourcelabels_raw){
            std::vector<vec2> icalsource;
            icalsource.push_back(cameracalibration->processedspectra.at("source").at(sourcelabel_raw).at(iw));
            icalsource.push_back(cameracalibration->processedspectra.at("source").at(sourcelabel_raw).at(iw));
            icalsource.at(1).x+=1;
            std::string sourcelable = "Cal_source_"+sourcelabel_raw;
            sourcelabels.push_back(sourcelable);
            context->setGlobalData(sourcelable.c_str(),HELIOS_TYPE_VEC2,2,&icalsource[0]);
        }

        std::vector<vec2> icalcamera(2);
        icalcamera.at(0).y=1;
        icalcamera.at(1).y=1;
        icalcamera.at(0).x=wavelengths.at(iw);
        icalcamera.at(1).x=wavelengths.at(iw)+1;
        std::string camlable = "Cal_cameraresponse";
        context->setGlobalData(camlable.c_str(),HELIOS_TYPE_VEC2,2,&icalcamera[0]);

        for (auto objectpair : cameracalibration->processedspectra.at("object")){
            std::vector<vec2> spectrum_obj;
            spectrum_obj.push_back(objectpair.second.at(iw));
            spectrum_obj.push_back(objectpair.second.at(iw));
            spectrum_obj.at(1).x += 1;
            context->setGlobalData(objectpair.first.c_str(), HELIOS_TYPE_VEC2, 2, &spectrum_obj[0]);
        }

        RadiationModel::addRadiationBand(wavelengthlabel, std::stof(wavelengthlabel), std::stof(wavelengthlabel) + 1);
        RadiationModel::disableEmission(wavelengthlabel);

        uint ID = 0;
        for (std::string sourcelabel_raw:sourcelabels_raw) {
            RadiationModel::setSourceSpectrum(ID, sourcelabels.at(ID).c_str());
            RadiationModel::setSourceFlux(ID, wavelengthlabel, 1);
            ID ++;
        }
        RadiationModel::setScatteringDepth(wavelengthlabel, 1);
        RadiationModel::setDiffuseRadiationFlux(wavelengthlabel, 0);
        RadiationModel::setDiffuseRadiationExtinctionCoeff(wavelengthlabel, 0.f, make_vec3(-0.5, 0.5, 1));

        RadiationModel::addRadiationCamera(cameralabel, {wavelengthlabel}, calibratecamera.position, calibratecamera.lookat, cameraproperties, 10);
        RadiationModel::setCameraSpectralResponse(cameralabel, wavelengthlabel, camlable);
        RadiationModel::updateGeometry();
        RadiationModel::runBand({wavelengthlabel});

        std::vector<float> camera_data;
        //RadiationModel::writeCameraImage(cameralabel,{wavelengthlabel},wavelengthlabel,"",1000);
        std::string global_data_label = "camera_" + cameralabel + "_" + wavelengthlabel;
        context->getGlobalData(global_data_label.c_str(), camera_data);

        std::vector<uint> pixel_labels;
        std::string global_data_label_UUID = "camera_" + cameralabel + "_pixel_UUID";
        context->getGlobalData(global_data_label_UUID.c_str(), pixel_labels);

        for (uint j = 0; j < calibratecamera.resolution.y; j++) {
            for (uint i = 0; i < calibratecamera.resolution.x; i++) {
                float icdata = camera_data.at(j * calibratecamera.resolution.x + i);

                uint UUID = pixel_labels.at(j * calibratecamera.resolution.x + i)-1;
                if (find(UUIDs_target.begin(), UUIDs_target.end(), UUID) != UUIDs_target.end()){
                    if ( simulatedcolorboardspectra.at(UUID).empty()){
                        simulatedcolorboardspectra.at(UUID).push_back(make_vec2(wavelengths.at(iw), icdata / float(numberwavelengths)));
                    }
                    else if (simulatedcolorboardspectra.at(UUID).back().x == wavelengths.at(iw)){
                        simulatedcolorboardspectra.at(UUID).back().y += icdata / float(numberwavelengths);
                    }
                    else if(simulatedcolorboardspectra.at(UUID).back().x !=wavelengths.at(iw)){
                        simulatedcolorboardspectra.at(UUID).push_back(make_vec2(wavelengths.at(iw), icdata / float(numberwavelengths)));
                    }
                }
            }
        }
    }
    // Update camera response spectra
    cameracalibration->updateCameraResponseSpectra(cameraresponselabels, calibratedmark ,simulatedcolorboardspectra, truevalues);
    // Reset color board spectra
    std::vector<uint> UUIDs_colorbd = cameracalibration->getColorBoardUUIDs();
    for (uint UUID:UUIDs_colorbd){
        std::string colorboardspectra;
        context->getPrimitiveData(UUID, "reflectivity_spectrum", colorboardspectra);
        context->setPrimitiveData(UUID, "reflectivity_spectrum", colorboardspectra+"_raw");
    }
}

void RadiationModel::runRadiationImaging(const std::string& cameralabel, const std::vector<std::string>& sourcelabels, const std::vector<std::string>& bandlabels,
                                         const std::vector<std::string>& cameraresponselabels, helios::vec2 wavelengthrange,
                                         float fluxscale, float diffusefactor, uint scatteringdepth){

    float sources_fluxsum = 0;
    std::vector<float> sources_fluxes;
    for (uint ID = 0; ID < sourcelabels.size(); ID++){
        std::vector<vec2> Source_spectrum;
        context->getGlobalData(sourcelabels.at(ID).c_str(), Source_spectrum);
        sources_fluxes.push_back(RadiationModel::integrateSpectrum(Source_spectrum, wavelengthrange.x, wavelengthrange.y));
        RadiationModel::setSourceSpectrum(ID, sourcelabels.at(ID).c_str());
        RadiationModel::setSourceSpectrumIntegral(ID, sources_fluxes.at(ID));
        sources_fluxsum += sources_fluxes.at(ID);
    }

    RadiationModel::addRadiationBand(bandlabels.at(0), wavelengthrange.x, wavelengthrange.y);
    RadiationModel::disableEmission(bandlabels.at(0));
    for( uint ID =0; ID<radiation_sources.size();ID++ ) {
        RadiationModel::setSourceFlux(ID, bandlabels.at(0), (1 - diffusefactor) * sources_fluxes.at(ID) * fluxscale);
    }
    RadiationModel::setScatteringDepth(bandlabels.at(0), scatteringdepth);
    RadiationModel::setDiffuseRadiationFlux(bandlabels.at(0), diffusefactor * sources_fluxsum );
    RadiationModel::setDiffuseRadiationExtinctionCoeff(bandlabels.at(0), 1.f, make_vec3(-0.5, 0.5, 1) );

    if (bandlabels.size()>1){
        for (int iband=1;iband<bandlabels.size();iband++){
            RadiationModel::copyRadiationBand(bandlabels.at(iband-1), bandlabels.at(iband), wavelengthrange.x, wavelengthrange.y);
            for( uint ID =0; ID<radiation_sources.size();ID++ ) {
                RadiationModel::setSourceFlux(ID, bandlabels.at(iband), (1 - diffusefactor) * sources_fluxes.at(ID) * fluxscale);
            }
            RadiationModel::setDiffuseRadiationFlux(bandlabels.at(iband), diffusefactor * sources_fluxsum );
        }
    }

    for (int iband=0;iband<bandlabels.size();iband++){
        RadiationModel::setCameraSpectralResponse(cameralabel, bandlabels.at(iband), cameraresponselabels.at(iband));
    }

    RadiationModel::updateGeometry();
    RadiationModel::runBand(bandlabels);
}

void RadiationModel::runRadiationImaging(const std::vector<std::string>& cameralabels, const std::vector<std::string>& sourcelabels, const std::vector<std::string>& bandlabels,
                                         const std::vector<std::string>& cameraresponselabels, helios::vec2 wavelengthrange,
                                         float fluxscale, float diffusefactor, uint scatteringdepth){

    float sources_fluxsum = 0;
    std::vector<float> sources_fluxes;
    for (uint ID = 0; ID < sourcelabels.size(); ID++){
        std::vector<vec2> Source_spectrum;
        context->getGlobalData(sourcelabels.at(ID).c_str(), Source_spectrum);
        sources_fluxes.push_back(RadiationModel::integrateSpectrum(Source_spectrum, wavelengthrange.x, wavelengthrange.y));
        RadiationModel::setSourceSpectrum(ID, sourcelabels.at(ID).c_str());
        RadiationModel::setSourceSpectrumIntegral(ID, sources_fluxes.at(ID));
        sources_fluxsum += sources_fluxes.at(ID);
    }

    RadiationModel::addRadiationBand(bandlabels.at(0), wavelengthrange.x, wavelengthrange.y);
    RadiationModel::disableEmission(bandlabels.at(0));
    for( uint ID =0; ID<radiation_sources.size();ID++ ) {
        RadiationModel::setSourceFlux(ID, bandlabels.at(0), (1 - diffusefactor) * sources_fluxes.at(ID) * fluxscale);
    }
    RadiationModel::setScatteringDepth(bandlabels.at(0), scatteringdepth);
    RadiationModel::setDiffuseRadiationFlux(bandlabels.at(0), diffusefactor * sources_fluxsum );
    RadiationModel::setDiffuseRadiationExtinctionCoeff(bandlabels.at(0), 1.f, make_vec3(-0.5, 0.5, 1) );

    if (bandlabels.size()>1){
        for (int iband=1;iband<bandlabels.size();iband++){
            RadiationModel::copyRadiationBand(bandlabels.at(iband-1), bandlabels.at(iband), wavelengthrange.x, wavelengthrange.y);
            for( uint ID =0; ID<radiation_sources.size();ID++ ) {
                RadiationModel::setSourceFlux(ID, bandlabels.at(iband), (1 - diffusefactor) * sources_fluxes.at(ID) * fluxscale);
            }
            RadiationModel::setDiffuseRadiationFlux(bandlabels.at(iband), diffusefactor * sources_fluxsum );
        }
    }

    for (int ic=0;ic<cameralabels.size();ic++){
        for (int iband=0;iband<bandlabels.size();iband++){
            RadiationModel::setCameraSpectralResponse(cameralabels.at(ic), bandlabels.at(iband), cameraresponselabels.at(iband));
        }
    }


    RadiationModel::updateGeometry();
    RadiationModel::runBand(bandlabels);
}

float RadiationModel::getCameraResponseScale(const std::string &orginalcameralabel, const std::vector<std::string>& cameraresponselabels,
                                             const std::vector<std::string>& bandlabels, const std::vector<std::string> &sourcelabels,
                                             vec2 &wavelengthrange, const std::vector<std::vector<float>> &truevalues){


    RadiationCamera calibratecamera = cameras.at(orginalcameralabel);
    CameraProperties cameraproperties;
    cameraproperties.HFOV = calibratecamera.HFOV_degrees;
    cameraproperties.camera_resolution = calibratecamera.resolution;
    cameraproperties.focal_plane_distance = calibratecamera.focal_length;
    cameraproperties.lens_diameter = calibratecamera.lens_diameter;
    cameraproperties.sensor_size = calibratecamera.sensor_size;

    std::string cameralabel = orginalcameralabel+"Scale";
    RadiationModel::addRadiationCamera(cameralabel, bandlabels, calibratecamera.position, calibratecamera.lookat, cameraproperties,20);
    RadiationModel::runRadiationImaging(cameralabel, sourcelabels, bandlabels, cameraresponselabels, wavelengthrange, 1, 0);

    // Get camera spectral response scale based on comparing true values and calibrated image
    float camerascale = cameracalibration->getCameraResponseScale(cameralabel, cameraproperties.camera_resolution, bandlabels, truevalues);
    return camerascale;
}


void RadiationModel::writePrimitiveDataLabelMap(const std::string &cameralabel, const std::string &filename, const std::string &labelname, float padvalue){

    if( cameras.find(cameralabel)==cameras.end() ){
        helios_runtime_error( "ERROR (RadiationModel::writePrimitiveDataLabelMap): Camera '" + cameralabel + "' does not exist." );
    }

    //Get image UUID labels
    std::vector<uint> camera_UUIDs;
    std::string global_data_label = "camera_" + cameralabel + "_pixel_UUID";
    if( !context->doesGlobalDataExist(global_data_label.c_str() ) ){
        helios_runtime_error( "ERROR (RadiationModel::writePrimitiveDataLabelMap): Pixel labels for camera '" + cameralabel + "' do not exist. Was the radiation model run to generate labels?" );
    }
    context->getGlobalData(global_data_label.c_str(), camera_UUIDs);
    std::vector<uint> pixel_UUIDs = camera_UUIDs;
    int2 camera_resolution = cameras.at(cameralabel).resolution;

    //Output label image in ".txt" format
    std::ofstream pixel_data(filename);

    for (uint j = 0; j < camera_resolution.y; j++) {
        for (uint i = 0; i < camera_resolution.x; i++) {
            uint UUID =pixel_UUIDs.at(j * camera_resolution.x + i)-1;
            if (context->doesPrimitiveExist(UUID) && context->doesPrimitiveDataExist(UUID,labelname.c_str())){
                HeliosDataType datatype = context->getPrimitiveDataType(UUID,labelname.c_str());
                if( datatype == HELIOS_TYPE_FLOAT ){
                    float labeldata;
                    context->getPrimitiveData(UUID,labelname.c_str(),labeldata);
                    pixel_data << labeldata << " ";
                }
                else if (datatype == HELIOS_TYPE_UINT){
                    uint labeldata;
                    context->getPrimitiveData(UUID,labelname.c_str(),labeldata);
                    pixel_data << labeldata << " ";
                }
                else if (datatype == HELIOS_TYPE_INT){
                    int labeldata;
                    context->getPrimitiveData(UUID,labelname.c_str(),labeldata);
                    pixel_data << labeldata << " ";
                }
                else if (datatype == HELIOS_TYPE_DOUBLE){
                    double labeldata;
                    context->getPrimitiveData(UUID,labelname.c_str(),labeldata);
                    pixel_data << labeldata << " ";
                }
            }
            else{
                pixel_data << padvalue << " ";
            }
        }
        pixel_data << "\n";
    }
    pixel_data.close();
}

void RadiationModel::writeDepthImage(const std::string &cameralabel, const std::string &filename) {

    //Get image UUID labels
    std::vector<uint> camera_UUIDs;
    std::string global_data_label = "camera_" + cameralabel + "_pixel_UUID";
    context->getGlobalData(global_data_label.c_str(), camera_UUIDs);
    std::vector<uint> pixel_UUIDs = camera_UUIDs;
    helios::vec3 camera_position = cameras.at(cameralabel).position;
    helios::vec3 camera_lookat = cameras.at(cameralabel).lookat;

    // Set depth for all primitives
    float maxdepth = 0;
    float mindepth = -1;
    for( uint UUIDr : pixel_UUIDs ){
        uint UUID = UUIDr-1;
        if (context->doesPrimitiveExist(UUID)) {
            std::vector<vec3> vertices = context->getPrimitiveVertices(UUID);//Centre
            vec3 vertex = vertices.at(0) - camera_position;
            vec3 cameraplane_direct = camera_lookat - camera_position;
            float dotproductvc = vertex * cameraplane_direct;
            float cam_norm = cameraplane_direct.magnitude();
            float depth = dotproductvc / cam_norm;
            if (depth > maxdepth) {
                maxdepth = depth;
            }
            if (mindepth == -1) {
                mindepth = depth;
            } else if (depth < mindepth) {
                mindepth = depth;
            }
            context->setPrimitiveData(UUID, "depth", depth);
        }
    }

    for (uint UUIDr : pixel_UUIDs ){
        uint UUID = UUIDr-1;
        if (context->doesPrimitiveExist(UUID)){
            float depth;
            context->getPrimitiveData(UUID,"depth",depth);
            float depth_norm = (depth-mindepth)/(maxdepth-mindepth);
            context->setPrimitiveData(UUID,"depth_norm",depth_norm);
        }
    }

    // Output depth image in ".txt" format
    RadiationModel::writePrimitiveDataLabelMap(cameralabel, filename, "depth_norm",1);
}

void RadiationModel::setPadValue(const std::string &cameralabel, const std::vector<std::string> &bandlabels, const std::vector<float> &padvalues) {
    for (uint b = 0; b < bandlabels.size(); b++) {
        std::string bandlabel = bandlabels.at(b);

        std::string image_value_label = "camera_" + cameralabel + "_" + bandlabel;
        std::vector<float> cameradata;
        context->getGlobalData(image_value_label.c_str(), cameradata);

        std::vector<uint> camera_UUIDs;
        std::string image_UUID_label = "camera_" + cameralabel + "_pixel_UUID";
        context->getGlobalData(image_UUID_label.c_str(), camera_UUIDs);

        for (uint i = 0; i < cameradata.size(); i++) {
            uint UUID = camera_UUIDs.at(i)-1;
            if (!context->doesPrimitiveExist(UUID)) {
                cameradata.at(i) = padvalues.at(b);
            }
        }
        context->setGlobalData(image_value_label.c_str(), HELIOS_TYPE_FLOAT, cameradata.size(), &cameradata[0]);
    }
}

void RadiationModel::calibrateCamera(const std::string &originalcameralabel, const std::vector<std::string> &sourcelabels,
                                     const std::vector<std::string>& cameraresplabels_raw, const std::vector<std::string> &bandlabels, const float scalefactor,
                                     const std::vector<std::vector<float>> &truevalues, const std::string &calibratedmark) {

    if( cameras.find(originalcameralabel) == cameras.end() ){
        helios_runtime_error("ERROR (RadiationModel::calibrateCamera): Camera " + originalcameralabel + " does not exist.");
    }else if( radiation_sources.empty() ){
        helios_runtime_error("ERROR (RadiationModel::calibrateCamera): No radiation sources were added to the radiation model. Cannot perform calibration.");
    }

    CameraCalibration cameracalibration_(context);
    if (!calibration_flag){
        std::cout<< "No color board added, use default color calibration." <<std::endl;
        cameracalibration = &cameracalibration_;
        vec3 centrelocation =make_vec3(0,0,0.2); // Location of color board
        vec3 rotationrad =make_vec3(0,0,1.5705); // Rotation angle of color board
        cameracalibration->addDefaultColorboard(centrelocation, rotationrad,0.1);
    }
    vec2 wavelengthrange = make_vec2(-10000,10000);

    // Calibrated camera response labels
    std::vector<std::string> cameraresplabels_cal(cameraresplabels_raw.size());

    for (int iband=0;iband<bandlabels.size();iband++){
        cameraresplabels_cal.at(iband) = calibratedmark + "_" + cameraresplabels_raw.at(iband);
    }

    RadiationModel::runRadiationImaging(originalcameralabel, sourcelabels, bandlabels, cameraresplabels_raw, wavelengthrange, 1, 0);
    // Update camera responses
    RadiationModel::updateCameraResponse(originalcameralabel, sourcelabels, cameraresplabels_raw, wavelengthrange, truevalues, calibratedmark);

    float camerascale = RadiationModel::getCameraResponseScale(originalcameralabel, cameraresplabels_cal, bandlabels,sourcelabels, wavelengthrange, truevalues);

    std::cout << "Camera response scale: " << camerascale << std::endl;
    // Scale and write calibrated camera responses
    cameracalibration->writeCalibratedCameraResponses(cameraresplabels_raw, calibratedmark, camerascale*scalefactor);

}

void RadiationModel::calibrateCamera(const std::string &originalcameralabel, const float scalefactor, const std::vector<std::vector<float>> &truevalues,
                                     const std::string &calibratedmark) {

    if( cameras.find(originalcameralabel) == cameras.end() ){
        helios_runtime_error("ERROR (RadiationModel::calibrateCamera): Camera " + originalcameralabel + " does not exist.");
    }else if( radiation_sources.empty() ){
        helios_runtime_error("ERROR (RadiationModel::calibrateCamera): No radiation sources were added to the radiation model. Cannot perform calibration.");
    }

    CameraCalibration cameracalibration_(context);
    if (!calibration_flag){
        std::cout<< "No color board added, use default color calibration." <<std::endl;
        vec3 centrelocation =make_vec3(0,0,0.2); // Location of color board
        vec3 rotationrad =make_vec3(0,0,1.5705); // Rotation angle of color board
        cameracalibration_.addDefaultColorboard(centrelocation, rotationrad,0.1);
        RadiationModel::setCameraCalibration(&cameracalibration_);
    }

    vec2 wavelengthrange = make_vec2(-10000,10000);

    std::vector<std::string> bandlabels = cameras.at(originalcameralabel).band_labels;

    // Get camera response spectra labels from camera
    std::vector<std::string> cameraresplabels_cal(cameras.at(originalcameralabel).band_spectral_response.size() );
    std::vector<std::string> cameraresplabels_raw = cameraresplabels_cal;

    int iband = 0;
    for( auto &band : cameras.at(originalcameralabel).band_spectral_response ){
        cameraresplabels_raw.at(iband) = band.second;
        cameraresplabels_cal.at(iband) = calibratedmark + "_" + band.second;
        iband++;
    }

    // Get labels of radiation sources from camera
    std::vector<std::string> sourcelabels(radiation_sources.size());
    int isource = 0;
    for( auto &source : radiation_sources ){
        if( source.source_spectrum.empty() ){
            helios_runtime_error("ERROR (RadiationModel::calibrateCamera): A spectral distribution was not specified for source " + source.source_spectrum_label + ". Cannot perform camera calibration.");
        }
        sourcelabels.at(isource) = source.source_spectrum_label;
        isource++;
    }

    RadiationModel::updateGeometry();
    RadiationModel::runBand(bandlabels);
    // Update camera responses
    RadiationModel::updateCameraResponse(originalcameralabel, sourcelabels, cameraresplabels_raw, wavelengthrange, truevalues, calibratedmark);

    float camerascale = RadiationModel::getCameraResponseScale(originalcameralabel, cameraresplabels_cal, bandlabels,sourcelabels, wavelengthrange, truevalues);

    std::cout << "Camera response scale: " << camerascale << std::endl;
    // Scale and write calibrated camera responses
    cameracalibration->writeCalibratedCameraResponses(cameraresplabels_raw, calibratedmark, camerascale*scalefactor);

}

std::vector<helios::vec2> RadiationModel::generateGaussianCameraResponse(float FWHM, float mu, float centrawavelength, const helios::int2 &wavebanrange){

    // Convert FWHM to sigma
    float sigma = FWHM / (2 * std::sqrt(2 * std::log(2)));

    size_t lenspectra = wavebanrange.y - wavebanrange.x;
    std::vector<helios::vec2> cameraresponse(lenspectra);


    for (int i = 0; i < lenspectra; ++i) {
        cameraresponse.at(i).x = float(wavebanrange.x + i);
    }

    // Gaussian function
    for (size_t i = 0; i < lenspectra; ++i) {
        cameraresponse.at(i).y = centrawavelength * std::exp(-std::pow((cameraresponse.at(i).x - mu), 2) / (2 * std::pow(sigma, 2)));
    }


    return cameraresponse;
}

void sutilHandleError(RTcontext context, RTresult code, const char* file, int line)
{
    const char* message;
    char s[2048];
    rtContextGetErrorString(context, code, &message);
    sprintf(s, "%s\n(%s:%d)", message, file, line);
    sutilReportError( s );
    exit(1);
}

void sutilReportError(const char* message)
{
    fprintf( stderr, "OptiX Error: %s\n", message );
#if defined(_WIN32) && defined(RELEASE_PUBLIC)
    {
    char s[2048];
    sprintf( s, "OptiX Error: %s", message );
    MessageBox( 0, s, "OptiX Error", MB_OK|MB_ICONWARNING|MB_SYSTEMMODAL );
  }
#endif
}
