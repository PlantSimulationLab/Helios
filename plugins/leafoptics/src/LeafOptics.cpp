/** \file "LeafOptics.cpp" Implementation of PROSPECT-PRO leaf optical model.

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "LeafOptics.h"
using namespace std;
using namespace helios;

LeafOptics::LeafOptics( helios::Context* a_context ){
    if( message_flag ){
        std::cout << "Initializing LeafOptics model..." << std::flush;
    }

    context = a_context; //just copying the pointer to the context

    // Load leaf refraction index and specific absorption coefficients  (400,401,...2500 nm)
    context->loadXML("plugins/leafoptics/spectral_data/prospect_spectral_library.xml", true);

    // Load leaf refraction index -  refractiveindex (n)
    std::vector<helios::vec2> data;
    if( !context->doesGlobalDataExist("refraction_index" ) ){
        helios_runtime_error("Refraction index data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("refraction_index", data);
    if( data.size() != nw ){
        helios_runtime_error("Size of refraction index data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    refractiveindex.resize(nw);
    wave_length.resize(nw);
    for(int i=0; i<nw; i++) {
        refractiveindex.at(i) = data.at(i).y;
        wave_length.at(i) = data.at(i).x;
    }
    // Load specific absorption coefficient (per elementary layer depth)  for total chlorophyll  -  absorption_chlorophyll (cm^2/micro_g)
    if( !context->doesGlobalDataExist("absorption_chlorophyll" ) ){
        helios_runtime_error("Chlorophyll absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_chlorophyll", data);
    if( data.size() != nw ){
        helios_runtime_error("Size of chlorophyll absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_chlorophyll.resize(nw);
    for(int i=0; i<nw; i++) {
        absorption_chlorophyll.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for total carotenoids  -  absorption_carotenoid (cm^2/micro_g)
    if( !context->doesGlobalDataExist("absorption_carotenoid" ) ){
        helios_runtime_error("Carotenoid absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_carotenoid", data);
    if( data.size() != nw ){
        helios_runtime_error("Size of carotenoid absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_carotenoid.resize(nw);
    for(int i=0; i<nw; i++) {
        absorption_carotenoid.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for anthocyanins  -  sac_an (cm^2/micro_g)
    if( !context->doesGlobalDataExist("absorption_anthocyanin" ) ){
        helios_runtime_error("Anothocyanin absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_anthocyanin", data);
    if( data.size() != nw ){
        helios_runtime_error("Size of anthocyanin absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_anthocyanin.resize(nw);
    for(int i=0; i<nw; i++) {
        absorption_anthocyanin.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for specific brown pigments (phenols during leaf death)   -  absorption_brown (arbitrary unit)
    if( !context->doesGlobalDataExist("absorption_brown" ) ){
        helios_runtime_error("Brown pigment absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_brown", data);
    if( data.size() != nw ){
        helios_runtime_error("Size of brown pigment absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_brown.resize(nw);
    for(int i=0; i<nw; i++) {
        absorption_brown.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for mass of water per leaf area (EWT)-  absorption_water (1/cm or cm^2/g)
    if( !context->doesGlobalDataExist("absorption_water" ) ){
        helios_runtime_error("Water absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_water", data);
    if( data.size() != nw ){
        helios_runtime_error("Size of water absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_water.resize(nw);
    for(int i=0; i<nw; i++) {
        absorption_water.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for dry mass per leaf area (LMA)-  absorption_drymass (cm^2/g)
    if( !context->doesGlobalDataExist("absorption_drymass" ) ){
        helios_runtime_error("Dry mass absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_drymass", data);
    if( data.size() != nw ){
        helios_runtime_error("Size of dry mass absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_drymass.resize(nw);
    for(int i=0; i<nw; i++) {
        absorption_drymass.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for proteins- absorption_proteins (cm2.g-1)
    if( !context->doesGlobalDataExist("absorption_proteins" ) ){
        helios_runtime_error("Protein absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_proteins", data);
    if( data.size() != nw ){
        helios_runtime_error("Size of protein absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_protein.resize(nw);
    for(int i=0; i<nw; i++) {
        absorption_protein.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for carbon based constituents-  absorption_carbonconstituents (cm^2/g)
    if( !context->doesGlobalDataExist("absorption_carbonconstituents" ) ){
        helios_runtime_error("Carbon constituent absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_carbonconstituents", data);
    if( data.size() != nw ){
        helios_runtime_error("Size of carbon constituent absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_carbonconstituents.resize(nw);
    for(int i=0; i<nw; i++) {
        absorption_carbonconstituents.at(i) = data.at(i).y;
    }

    LeafOptics::surface( 40.0, R_spec_normal);
    // get surface (i.e. fresnel)  reflectances for usage within Prospect function, relation from Stern
    // 0..40° degrees range from the vertical = normal incidence on an average rough leaf

    LeafOptics::surface( 90.0, R_spec_diffuse);
    // 0..90° degrees range from the vertical = diffuse incidence on a perfectly smooth leaf

    if( message_flag ) {
        std::cout << "done." << std::endl;
    }
}

int LeafOptics::selfTest(){

    Context context_test;

    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    LeafOpticsProperties leafproperties;

    leafoptics.run(leafproperties, "test");

    return 0;

}

void LeafOptics::run(const std::vector<uint> &UUIDs , const LeafOpticsProperties &leafproperties, const std::string &label) {
    std::vector<vec2> reflectivities_fit;
    std::vector<vec2> transmissivities_fit;
    getLeafSpectra(leafproperties, reflectivities_fit, transmissivities_fit);

    std::string leaf_reflectivity_label = "leaf_reflectivity_"+label;
    std::string leaf_transmissivity_label = "leaf_transmissivity_"+label;
    context->setGlobalData(leaf_reflectivity_label.c_str(),HELIOS_TYPE_VEC2,reflectivities_fit.size(),&reflectivities_fit[0]);
    context->setGlobalData(leaf_transmissivity_label.c_str(),HELIOS_TYPE_VEC2,transmissivities_fit.size(),&transmissivities_fit[0]);

    context->setPrimitiveData( UUIDs, "reflectivity_spectrum", leaf_reflectivity_label);
    context->setPrimitiveData( UUIDs, "transmissivity_spectrum", leaf_transmissivity_label);
    setProperties(UUIDs, leafproperties);
}

void LeafOptics::run(const LeafOpticsProperties &leafproperties, const std::string &label) {
    std::vector<vec2> reflectivities_fit;
    std::vector<vec2> transmissivities_fit;
    getLeafSpectra(leafproperties, reflectivities_fit, transmissivities_fit);

    std::string leaf_reflectivity_label = "leaf_reflectivity_"+label;
    std::string leaf_transmissivity_label = "leaf_transmissivity_"+label;
    context->setGlobalData(leaf_reflectivity_label.c_str(),HELIOS_TYPE_VEC2,reflectivities_fit.size(),&reflectivities_fit[0]);
    context->setGlobalData(leaf_transmissivity_label.c_str(),HELIOS_TYPE_VEC2,transmissivities_fit.size(),&transmissivities_fit[0]);

}


void LeafOptics::PROSPECT(float numberlayers, float Chlorophyllcontent, float carotenoidcontent, float anthocyancontent, float brownpigments,
                          float watermass, float drymass, float protein, float carbonconstituents, std::vector<float> &reflectivities_fit, std::vector<float> &transmissivities_fit)
// Implementation of Prospect-PRO, port of public available matlab code
{
    double k;
    double tau,ralf,r12,talf, t12, t21, r21, denom, Ta, Ra, t, r;
    // Loop over wavelength, might be a way to be vectorized at least partly
    for(int i=0; i<LeafOptics::nw; i++) {
        // k: the mean absorption coefficient of each elementary layer.
        k = (Chlorophyllcontent * absorption_chlorophyll.at(i) + carotenoidcontent * absorption_carotenoid.at(i) + anthocyancontent * absorption_anthocyanin.at(i)
             + brownpigments * absorption_brown.at(i) + watermass * absorption_water.at(i) + drymass * absorption_drymass.at(i)
             +  protein * absorption_protein.at(i) +carbonconstituents * absorption_carbonconstituents.at(i)) / numberlayers;

        // diffuse transmittance through elementary layer, this integral needs more effort in C++
        tau = transmittance(k);
        // surface reflectance at radiated leaf side for near normal incident beam radiation,
        ralf = R_spec_normal.at(i);  // calculate 1-tav in surface function
        // surface reflectance at radiated leaf side for diffuse radiation
        r12  = R_spec_diffuse.at(i);


        talf = 1 - ralf;   //tav90
        t12 = 1 - r12;   //tav

        // transmittance and reflectance for leaf internal diffuse light
        t21     = t12/(refractiveindex.at(i)*refractiveindex.at(i)); // tav/n^2
        r21     = 1-t21;

        // top or incident surface side

        denom   = 1-r21*r21*tau*tau;  // (euqation1 in RPOSPECT)
        Ta      = talf*tau*t21/denom; // transmittance of top surface (euqation2 in RPOSPECT)  taua
        Ra      = ralf+r21*tau*Ta; // reflectance of top surface  (euqation1 in RPOSPECT)   rhoa

        // bottom surface side

        t       = t12*tau*t21/denom;  //  (page 78 paragraph 1 in RPOSPECT)   tau90
        r       = r12+r21*tau*t;    //  (page 78 paragraph 1 in RPOSPECT)    rho90

        // reflectance and transmittance of numberlayers layers, Stokes' solution
        double  D, rq,tq,a,b,bNm1,bN2,a2,Rsub,Tsub;
        D       = sqrt((1.+r+t)*(1.+r-t)*(1.-r+t)*(1.-r-t));
        rq      = r*r;
        tq      = t*t;
        a       = (1.+rq-tq+D)/(2*r);
        b       = (1.-rq+tq+D)/(2*t);
        bNm1    = std::pow(b,(numberlayers - 1));
        bN2     = bNm1*bNm1;
        a2      = a*a;
        denom   = a2*bN2-1.;
        Rsub    = a*(bN2-1.)/denom;
        Tsub    = bNm1*(a2-1.)/denom;

        //Case of zero absorption

        if ((r+t) > 1.0) {
            Tsub 	= t/(t+(1.-t)*(numberlayers - 1));
            Rsub	= 1-Tsub;
        }

        // Reflectance and transmittance of the leaf: combine top layer with next numberlayers-1 layers

        denom   = 1-Rsub*r;
        transmissivities_fit.push_back(Ta * Tsub / denom); //(euqation8 in RPOSPEC
        reflectivities_fit.push_back(Ra + Ta * Rsub * t / denom); //(euqation7 in RPOSPECT)
    }
}

void LeafOptics::getLeafSpectra(const LeafOpticsProperties &leafproperties, std::vector<helios::vec2> &reflectivities_fit, std::vector<helios::vec2> &transmissivities_fit) {

    std::vector<float> reflectivities_fit_y, transmissivities_fit_y;

    float numberlayers = leafproperties.numberlayers;
    float chlorophyllcontent= leafproperties.chlorophyllcontent ;
    float carotenoidcontent = leafproperties.carotenoidcontent;
    float anthocyancontent = leafproperties.anthocyancontent;
    float brownpigments = leafproperties.brownpigments;
    float watermass= leafproperties.watermass;
    float drymass = leafproperties.drymass;
    float protein = leafproperties.protein;
    float carbonconstituents = leafproperties.carbonconstituents;

    if (protein == 0 && carbonconstituents == 0) {
        if (drymass == 0 && message_flag ) {
            std::cerr << "Warning: No leaf mass given" << std::endl;
        }
    }
    else{
        drymass = 0;
    }
    PROSPECT(numberlayers, chlorophyllcontent, carotenoidcontent, anthocyancontent, brownpigments, watermass,
             drymass, protein, carbonconstituents, reflectivities_fit_y, transmissivities_fit_y);

    // Convert float to vec2
    for(int iwave=0; iwave<nw; iwave++) {
        reflectivities_fit.push_back(make_vec2(iwave+400,reflectivities_fit_y.at(iwave)));
        transmissivities_fit.push_back(make_vec2(iwave+400,transmissivities_fit_y.at(iwave)));
    }

}

void LeafOptics::surface(float degree, std::vector<float> &reflectivities)
//! Mean fresnel reflectance over incidence angle 0...degree (°)  Sterns procedure
//! Ported from Prospect-D calctav.m
{

    double rad2degree = 57.2958;
    // tav is the transmissivity of a dielectric plane surface, averaged over all directions of incidence and over all polarizations.
    double n2,np,nm,a,k,sinvalue,b1,b2,b,b3,a3,ts,tp1,tp2,tp3,tp4,tp5,tp,tav;
    for(int i=0; i<LeafOptics::nw; i++) {
        double n = refractiveindex.at(i); //refractive index
        n2 = n*n;
        np = n2 + 1;
        nm = n2 - 1;
        a = (n + 1) * (n + 1) / 2;
        k = -(n2 - 1) * (n2 - 1) / 4;
        sinvalue = sin(degree / rad2degree);
        if (degree < 90.0){b1 =   sqrt((sinvalue * sinvalue - np / 2) * (sinvalue * sinvalue - np / 2) + k);}
        else{b1= 0.0;}
        b2 = sinvalue * sinvalue - np / 2;
        b = b1 - b2;
        b3 = b*b*b;
        a3 = a*a*a;
        ts = (k*k / (6 * b3) + k/ b - b / 2) - (k*k / (6 * a3) + k/ a - a / 2);
        tp1 = -2 * n2*(b - a)/ (np*np);
        tp2 = -2 * n2*np*log(b/ a)/ (nm*nm);
        tp3 = n2*(1 / b - 1 / a) / 2;
        tp4 = 16 * n2*n2 * (n2*n2 + 1)*log((2 * np*b - nm*nm)/ (2 * np*a - nm*nm))/ (np*np*np * nm*nm);
        tp5 = 16 * n2*n2*n2  * (1 / (2 * np*b - nm*nm) - 1 / (2 * np*a - nm*nm))/ (np*np*np);
        tp = tp1 + tp2 + tp3 + tp4 + tp5;
        tav = (ts + tp)/ (2 * sinvalue * sinvalue);
        reflectivities.push_back(1.0 - tav);
    }
    return;
}

float LeafOptics::transmittance(double k) {
//! Implementation for diffuse transmittance through an elementary layer
//! Exponential integral: S13AAF routine from the NAG library
//! Ported from public available  Prospect-D Fortran code
    double xx,yy;
    float tau;
    if  (k < 0.0) return 1.0;

    if ((k > 0.0) && (k<4.0)) {
        xx = 0.5 * k - 1.0;
        yy = (((((((((((((((-3.60311230482612224e-13L
                            *xx + 3.46348526554087424e-12L) * xx - 2.99627399604128973e-11L)
                          *xx + 2.57747807106988589e-10L) * xx - 2.09330568435488303e-9L)
                        *xx + 1.59501329936987818e-8L) * xx - 1.13717900285428895e-7L)
                      *xx + 7.55292885309152956e-7L) * xx - 4.64980751480619431e-6L)
                    *xx + 2.63830365675408129e-5L) * xx - 1.37089870978830576e-4L)
                  *xx + 6.47686503728103400e-4L) * xx - 2.76060141343627983e-3L)
                *xx + 1.05306034687449505e-2L) * xx - 3.57191348753631956e-2L)
              *xx + 1.07774527938978692e-1L) * xx - 2.96997075145080963e-1L;
        yy = (yy * xx + 8.64664716763387311e-1L) * xx + 7.42047691268006429e-1L  ;
        yy = yy - log(k);
        tau = (1.0 - k) * exp(-k) + k*k * yy;
        return tau;
    }
    if ((k >4.0) && (k< 85.0)) {
        xx = 14.5 / (k + 3.25) - 1.0;
        yy = (((((((((((((((-1.62806570868460749e-12L
                            *xx - 8.95400579318284288e-13L) * xx - 4.08352702838151578e-12L)
                          *xx - 1.45132988248537498e-11L) * xx - 8.35086918940757852e-11L)
                        *xx - 2.13638678953766289e-10L) * xx - 1.10302431467069770e-9L)
                      *xx - 3.67128915633455484e-9L) * xx - 1.66980544304104726e-8L)
                    *xx - 6.11774386401295125e-8L) * xx - 2.70306163610271497e-7L)
                  *xx - 1.05565006992891261e-6L) * xx - 4.72090467203711484e-6L)
                *xx - 1.95076375089955937e-5L) * xx - 9.16450482931221453e-5L)
              *xx - 4.05892130452128677e-4L) * xx - 2.14213055000334718e-3L;
        yy = ((yy * xx - 1.06374875116569657e-2L) * xx - 8.50699154984571871e-2L) * xx + 9.23755307807784058e-1L;
        yy = exp(-k) * yy / k;
        tau = (1.0 - k) * exp(-k) + k*k * yy;
        return tau;
    }
    return 0.0;
}

void LeafOptics::setProperties(const std::vector<uint> &UUIDs, const LeafOpticsProperties &leafproperties) {

    context->setPrimitiveData(UUIDs, "chlorophyll", leafproperties.chlorophyllcontent);
    context->setPrimitiveData(UUIDs, "carotenoid", leafproperties.carotenoidcontent);
    context->setPrimitiveData(UUIDs, "anthocyanin", leafproperties.anthocyancontent);
    if (leafproperties.brownpigments > 0.0){
        context->setPrimitiveData(UUIDs, "brown", leafproperties.brownpigments);
    }
    context->setPrimitiveData(UUIDs, "water", leafproperties.watermass);
    if (leafproperties.drymass > 0.0) {
        context->setPrimitiveData(UUIDs, "drymass", leafproperties.drymass);
    }
    else {
        context->setPrimitiveData(UUIDs, "protein", leafproperties.protein);
        context->setPrimitiveData(UUIDs, "cellulose", leafproperties.carbonconstituents);
    }
}

void LeafOptics::disableMessages(){
    message_flag = false;
}

void LeafOptics::enableMessages(){
    message_flag = true;
}

