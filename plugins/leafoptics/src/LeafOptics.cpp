/** \file "LeafOptics.cpp" Implementation of PROSPECT-PRO leaf optical model.

    Copyright (C) 2016-2026 Brian Bailey

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

LeafOptics::LeafOptics(helios::Context *a_context) {

    context = a_context; // just copying the pointer to the context

    // Load leaf refraction index and specific absorption coefficients  (400,401,...2500 nm)
    context->loadXML("plugins/leafoptics/spectral_data/prospect_spectral_library.xml", true);

    // Load leaf refraction index -  refractiveindex (n)
    std::vector<helios::vec2> data;
    if (!context->doesGlobalDataExist("refraction_index")) {
        helios_runtime_error("Refraction index data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("refraction_index", data);
    if (data.size() != nw) {
        helios_runtime_error("Size of refraction index data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    refractiveindex.resize(nw);
    wave_length.resize(nw);
    for (int i = 0; i < nw; i++) {
        refractiveindex.at(i) = data.at(i).y;
        wave_length.at(i) = data.at(i).x;
    }
    // Load specific absorption coefficient (per elementary layer depth)  for total chlorophyll  -  absorption_chlorophyll (cm^2/micro_g)
    if (!context->doesGlobalDataExist("absorption_chlorophyll")) {
        helios_runtime_error("Chlorophyll absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_chlorophyll", data);
    if (data.size() != nw) {
        helios_runtime_error("Size of chlorophyll absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_chlorophyll.resize(nw);
    for (int i = 0; i < nw; i++) {
        absorption_chlorophyll.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for total carotenoids  -  absorption_carotenoid (cm^2/micro_g)
    if (!context->doesGlobalDataExist("absorption_carotenoid")) {
        helios_runtime_error("Carotenoid absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_carotenoid", data);
    if (data.size() != nw) {
        helios_runtime_error("Size of carotenoid absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_carotenoid.resize(nw);
    for (int i = 0; i < nw; i++) {
        absorption_carotenoid.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for anthocyanins  -  sac_an (cm^2/micro_g)
    if (!context->doesGlobalDataExist("absorption_anthocyanin")) {
        helios_runtime_error("Anothocyanin absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_anthocyanin", data);
    if (data.size() != nw) {
        helios_runtime_error("Size of anthocyanin absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_anthocyanin.resize(nw);
    for (int i = 0; i < nw; i++) {
        absorption_anthocyanin.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for specific brown pigments (phenols during leaf death)   -  absorption_brown (arbitrary unit)
    if (!context->doesGlobalDataExist("absorption_brown")) {
        helios_runtime_error("Brown pigment absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_brown", data);
    if (data.size() != nw) {
        helios_runtime_error("Size of brown pigment absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_brown.resize(nw);
    for (int i = 0; i < nw; i++) {
        absorption_brown.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for mass of water per leaf area (EWT)-  absorption_water (1/cm or cm^2/g)
    if (!context->doesGlobalDataExist("absorption_water")) {
        helios_runtime_error("Water absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_water", data);
    if (data.size() != nw) {
        helios_runtime_error("Size of water absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_water.resize(nw);
    for (int i = 0; i < nw; i++) {
        absorption_water.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for dry mass per leaf area (LMA)-  absorption_drymass (cm^2/g)
    if (!context->doesGlobalDataExist("absorption_drymass")) {
        helios_runtime_error("Dry mass absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_drymass", data);
    if (data.size() != nw) {
        helios_runtime_error("Size of dry mass absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_drymass.resize(nw);
    for (int i = 0; i < nw; i++) {
        absorption_drymass.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for proteins- absorption_proteins (cm2.g-1)
    if (!context->doesGlobalDataExist("absorption_proteins")) {
        helios_runtime_error("Protein absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_proteins", data);
    if (data.size() != nw) {
        helios_runtime_error("Size of protein absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_protein.resize(nw);
    for (int i = 0; i < nw; i++) {
        absorption_protein.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for carbon based constituents-  absorption_carbonconstituents (cm^2/g)
    if (!context->doesGlobalDataExist("absorption_carbonconstituents")) {
        helios_runtime_error("Carbon constituent absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_carbonconstituents", data);
    if (data.size() != nw) {
        helios_runtime_error("Size of carbon constituent absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_carbonconstituents.resize(nw);
    for (int i = 0; i < nw; i++) {
        absorption_carbonconstituents.at(i) = data.at(i).y;
    }

    LeafOptics::surface(40.0, R_spec_normal);
    // get surface (i.e. fresnel)  reflectances for usage within Prospect function, relation from Stern
    // 0..40° degrees range from the vertical = normal incidence on an average rough leaf

    LeafOptics::surface(90.0, R_spec_diffuse);
    // 0..90° degrees range from the vertical = diffuse incidence on a perfectly smooth leaf

    // Initialize the species library with PROSPECT-D parameters from LOPEX93 dataset
    initializeSpeciesLibrary();
}

void LeafOptics::initializeSpeciesLibrary() {
    // PROSPECT-D parameters fitted to LOPEX93 spectral library samples
    // All species use PROSPECT-D mode (drymass > 0, protein = 0, carbonconstituents = 0)
    // Fitted using fit_prospect_visrobust.py with --no-calib flag

    LeafOpticsProperties props;

    // Default species (original Helios default values)
    props.numberlayers = 1.5f;
    props.chlorophyllcontent = 30.0f;
    props.carotenoidcontent = 7.0f;
    props.anthocyancontent = 1.0f;
    props.brownpigments = 0.0f;
    props.watermass = 0.015f;
    props.drymass = 0.09f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["default"] = props;

    // Garden lettuce (Lactuca sativa L.) - LOPEX93 sample 0021
    // RMSE: 0.008355
    props.numberlayers = 2.00517f;
    props.chlorophyllcontent = 30.2697f;
    props.carotenoidcontent = 6.9869f;
    props.anthocyancontent = 1.35975f;
    props.brownpigments = 0.107067f;
    props.watermass = 0.0281985f;
    props.drymass = 0.0052668f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["garden_lettuce"] = props;

    // Alfalfa (Medicago sativa L.) - LOPEX93 sample 0036
    // RMSE: 0.007743
    props.numberlayers = 2.00758f;
    props.chlorophyllcontent = 43.6375f;
    props.carotenoidcontent = 10.3145f;
    props.anthocyancontent = 1.33894f;
    props.brownpigments = 0.0f;
    props.watermass = 0.0189936f;
    props.drymass = 0.00473702f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["alfalfa"] = props;

    // Corn (Zea mays L.) - LOPEX93 sample 0041
    // RMSE: 0.015966
    props.numberlayers = 1.59203f;
    props.chlorophyllcontent = 22.8664f;
    props.carotenoidcontent = 3.9745f;
    props.anthocyancontent = 0.0f;
    props.brownpigments = 0.72677f;
    props.watermass = 0.0149645f;
    props.drymass = 0.00441283f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["corn"] = props;

    // Sunflower (Helianthus annuus L.) - LOPEX93 sample 0081
    // RMSE: 0.007353
    props.numberlayers = 1.76358f;
    props.chlorophyllcontent = 54.0514f;
    props.carotenoidcontent = 12.9027f;
    props.anthocyancontent = 1.75194f;
    props.brownpigments = 0.0112026f;
    props.watermass = 0.0185557f;
    props.drymass = 0.00644855f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["sunflower"] = props;

    // English walnut (Juglans regia L.) - LOPEX93 sample 0091
    // RMSE: 0.007828
    props.numberlayers = 1.56274f;
    props.chlorophyllcontent = 55.9211f;
    props.carotenoidcontent = 12.4596f;
    props.anthocyancontent = 1.73981f;
    props.brownpigments = 0.0f;
    props.watermass = 0.0127743f;
    props.drymass = 0.00583351f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["english_walnut"] = props;

    // Rice (Oryza sativa L.) - LOPEX93 sample 0106
    // RMSE: 0.004041
    props.numberlayers = 1.67081f;
    props.chlorophyllcontent = 37.233f;
    props.carotenoidcontent = 9.98756f;
    props.anthocyancontent = 0.0f;
    props.brownpigments = 0.0275106f;
    props.watermass = 0.0100962f;
    props.drymass = 0.00484587f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["rice"] = props;

    // Soybean (Glycine max L.) - LOPEX93 sample 0116
    // RMSE: 0.005875
    props.numberlayers = 1.5375f;
    props.chlorophyllcontent = 46.4121f;
    props.carotenoidcontent = 12.1394f;
    props.anthocyancontent = 0.648353f;
    props.brownpigments = 0.0f;
    props.watermass = 0.0101049f;
    props.drymass = 0.00292814f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["soybean"] = props;

    // Wine grape (Vitis vinifera L.) - LOPEX93 sample 0276
    // RMSE: 0.005585
    props.numberlayers = 1.42673f;
    props.chlorophyllcontent = 50.918f;
    props.carotenoidcontent = 12.5466f;
    props.anthocyancontent = 1.43905f;
    props.brownpigments = 0.0798702f;
    props.watermass = 0.010922f;
    props.drymass = 0.00599315f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["wine_grape"] = props;

    // Tomato (Lycopersicum esculentum) - LOPEX93 sample 0316
    // RMSE: 0.005524
    props.numberlayers = 1.40304f;
    props.chlorophyllcontent = 48.3467f;
    props.carotenoidcontent = 11.604f;
    props.anthocyancontent = 1.45113f;
    props.brownpigments = 0.0f;
    props.watermass = 0.0155627f;
    props.drymass = 0.00261571f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["tomato"] = props;

    // Common bean (Phaseolus vulgaris L.) - GEMINI field experiments, day 35
    // RMSE: 0.009479
    props.numberlayers = 1.44041f;
    props.chlorophyllcontent = 42.3619f;
    props.carotenoidcontent = 15.6263f;
    props.anthocyancontent = 0.844536f;
    props.brownpigments = 0.0f;
    props.watermass = 0.0150048f;
    props.drymass = 0.00196285f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["common_bean"] = props;

    // Cowpea (Vigna unguiculata L.) - GEMINI field experiments, day 48
    // RMSE: 0.010680
    props.numberlayers = 1.22669f;
    props.chlorophyllcontent = 61.5204f;
    props.carotenoidcontent = 25.6171f;
    props.anthocyancontent = 2.51899f;
    props.brownpigments = 0.0f;
    props.watermass = 0.0221158f;
    props.drymass = 0.00108066f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["cowpea"] = props;
}

void LeafOptics::run(const std::vector<uint> &UUIDs, const LeafOpticsProperties &leafproperties, const std::string &label) {
    std::vector<vec2> reflectivities_fit;
    std::vector<vec2> transmissivities_fit;
    getLeafSpectra(leafproperties, reflectivities_fit, transmissivities_fit);

    std::string leaf_reflectivity_label = "leaf_reflectivity_" + label;
    std::string leaf_transmissivity_label = "leaf_transmissivity_" + label;
    context->setGlobalData(leaf_reflectivity_label.c_str(), reflectivities_fit);
    context->setGlobalData(leaf_transmissivity_label.c_str(), transmissivities_fit);

    context->setPrimitiveData(UUIDs, "reflectivity_spectrum", leaf_reflectivity_label);
    context->setPrimitiveData(UUIDs, "transmissivity_spectrum", leaf_transmissivity_label);
    setProperties(UUIDs, leafproperties);

    // Store parameters in map for later retrieval
    spectrum_parameters_map[label] = leafproperties;
}

void LeafOptics::run(const LeafOpticsProperties &leafproperties, const std::string &label) {
    std::vector<vec2> reflectivities_fit;
    std::vector<vec2> transmissivities_fit;
    getLeafSpectra(leafproperties, reflectivities_fit, transmissivities_fit);

    std::string leaf_reflectivity_label = "leaf_reflectivity_" + label;
    std::string leaf_transmissivity_label = "leaf_transmissivity_" + label;
    context->setGlobalData(leaf_reflectivity_label.c_str(), reflectivities_fit);
    context->setGlobalData(leaf_transmissivity_label.c_str(), transmissivities_fit);

    // Store parameters in map for later retrieval
    spectrum_parameters_map[label] = leafproperties;
}


void LeafOptics::PROSPECT(float numberlayers, float Chlorophyllcontent, float carotenoidcontent, float anthocyancontent, float brownpigments, float watermass, float drymass, float protein, float carbonconstituents,
                          std::vector<float> &reflectivities_fit, std::vector<float> &transmissivities_fit)
// Implementation of Prospect-PRO, port of public available matlab code
{
    double k;
    double tau, ralf, r12, talf, t12, t21, r21, denom, Ta, Ra, t, r;
    // Loop over wavelength, might be a way to be vectorized at least partly
    for (int i = 0; i < LeafOptics::nw; i++) {
        // k: the mean absorption coefficient of each elementary layer.
        k = (Chlorophyllcontent * absorption_chlorophyll.at(i) + carotenoidcontent * absorption_carotenoid.at(i) + anthocyancontent * absorption_anthocyanin.at(i) + brownpigments * absorption_brown.at(i) + watermass * absorption_water.at(i) +
             drymass * absorption_drymass.at(i) + protein * absorption_protein.at(i) + carbonconstituents * absorption_carbonconstituents.at(i)) /
            numberlayers;

        // diffuse transmittance through elementary layer, this integral needs more effort in C++
        tau = transmittance(k);
        // surface reflectance at radiated leaf side for near normal incident beam radiation,
        ralf = R_spec_normal.at(i); // calculate 1-tav in surface function
        // surface reflectance at radiated leaf side for diffuse radiation
        r12 = R_spec_diffuse.at(i);


        talf = 1 - ralf; // tav90
        t12 = 1 - r12; // tav

        // transmittance and reflectance for leaf internal diffuse light
        t21 = t12 / (refractiveindex.at(i) * refractiveindex.at(i)); // tav/n^2
        r21 = 1 - t21;

        // top or incident surface side

        denom = 1 - r21 * r21 * tau * tau; // (euqation1 in RPOSPECT)
        Ta = talf * tau * t21 / denom; // transmittance of top surface (euqation2 in RPOSPECT)  taua
        Ra = ralf + r21 * tau * Ta; // reflectance of top surface  (euqation1 in RPOSPECT)   rhoa

        // bottom surface side

        t = t12 * tau * t21 / denom; //  (page 78 paragraph 1 in RPOSPECT)   tau90
        r = r12 + r21 * tau * t; //  (page 78 paragraph 1 in RPOSPECT)    rho90

        // reflectance and transmittance of numberlayers layers, Stokes' solution
        double D, rq, tq, a, b, bNm1, bN2, a2, Rsub, Tsub;
        D = sqrt((1. + r + t) * (1. + r - t) * (1. - r + t) * (1. - r - t));
        rq = r * r;
        tq = t * t;
        a = (1. + rq - tq + D) / (2 * r);
        b = (1. - rq + tq + D) / (2 * t);
        bNm1 = std::pow(b, (numberlayers - 1));
        bN2 = bNm1 * bNm1;
        a2 = a * a;
        denom = a2 * bN2 - 1.;
        Rsub = a * (bN2 - 1.) / denom;
        Tsub = bNm1 * (a2 - 1.) / denom;

        // Case of zero absorption

        if ((r + t) > 1.0) {
            Tsub = t / (t + (1. - t) * (numberlayers - 1));
            Rsub = 1 - Tsub;
        }

        // Reflectance and transmittance of the leaf: combine top layer with next numberlayers-1 layers

        denom = 1 - Rsub * r;
        transmissivities_fit.push_back(Ta * Tsub / denom); //(euqation8 in RPOSPEC
        reflectivities_fit.push_back(Ra + Ta * Rsub * t / denom); //(euqation7 in RPOSPECT)
    }
}

void LeafOptics::getLeafSpectra(const LeafOpticsProperties &leafproperties, std::vector<helios::vec2> &reflectivities_fit, std::vector<helios::vec2> &transmissivities_fit) {

    std::vector<float> reflectivities_fit_y, transmissivities_fit_y;

    float numberlayers = leafproperties.numberlayers;
    float chlorophyllcontent = leafproperties.chlorophyllcontent;
    float carotenoidcontent = leafproperties.carotenoidcontent;
    float anthocyancontent = leafproperties.anthocyancontent;
    float brownpigments = leafproperties.brownpigments;
    float watermass = leafproperties.watermass;
    float drymass = leafproperties.drymass;
    float protein = leafproperties.protein;
    float carbonconstituents = leafproperties.carbonconstituents;

    if (protein == 0 && carbonconstituents == 0) {
        if (drymass == 0 && message_flag) {
            std::cerr << "Warning: No leaf mass given" << std::endl;
        }
    } else {
        drymass = 0;
    }
    PROSPECT(numberlayers, chlorophyllcontent, carotenoidcontent, anthocyancontent, brownpigments, watermass, drymass, protein, carbonconstituents, reflectivities_fit_y, transmissivities_fit_y);

    // Convert float to vec2
    for (int iwave = 0; iwave < nw; iwave++) {
        reflectivities_fit.push_back(make_vec2(iwave + 400, reflectivities_fit_y.at(iwave)));
        transmissivities_fit.push_back(make_vec2(iwave + 400, transmissivities_fit_y.at(iwave)));
    }
}

void LeafOptics::surface(float degree, std::vector<float> &reflectivities)
//! Mean fresnel reflectance over incidence angle 0...degree (°)  Sterns procedure
//! Ported from Prospect-D calctav.m
{

    double rad2degree = 57.2958;
    // tav is the transmissivity of a dielectric plane surface, averaged over all directions of incidence and over all polarizations.
    double n2, np, nm, a, k, sinvalue, b1, b2, b, b3, a3, ts, tp1, tp2, tp3, tp4, tp5, tp, tav;
    for (int i = 0; i < LeafOptics::nw; i++) {
        double n = refractiveindex.at(i); // refractive index
        n2 = n * n;
        np = n2 + 1;
        nm = n2 - 1;
        a = (n + 1) * (n + 1) / 2;
        k = -(n2 - 1) * (n2 - 1) / 4;
        sinvalue = sin(degree / rad2degree);
        if (degree < 90.0) {
            b1 = sqrt((sinvalue * sinvalue - np / 2) * (sinvalue * sinvalue - np / 2) + k);
        } else {
            b1 = 0.0;
        }
        b2 = sinvalue * sinvalue - np / 2;
        b = b1 - b2;
        b3 = b * b * b;
        a3 = a * a * a;
        ts = (k * k / (6 * b3) + k / b - b / 2) - (k * k / (6 * a3) + k / a - a / 2);
        tp1 = -2 * n2 * (b - a) / (np * np);
        tp2 = -2 * n2 * np * log(b / a) / (nm * nm);
        tp3 = n2 * (1 / b - 1 / a) / 2;
        tp4 = 16 * n2 * n2 * (n2 * n2 + 1) * log((2 * np * b - nm * nm) / (2 * np * a - nm * nm)) / (np * np * np * nm * nm);
        tp5 = 16 * n2 * n2 * n2 * (1 / (2 * np * b - nm * nm) - 1 / (2 * np * a - nm * nm)) / (np * np * np);
        tp = tp1 + tp2 + tp3 + tp4 + tp5;
        tav = (ts + tp) / (2 * sinvalue * sinvalue);
        reflectivities.push_back(1.0 - tav);
    }
    return;
}

float LeafOptics::transmittance(double k) {
    //! Implementation for diffuse transmittance through an elementary layer
    //! Exponential integral: S13AAF routine from the NAG library
    //! Ported from public available  Prospect-D Fortran code
    double xx, yy;
    float tau;
    if (k < 0.0)
        return 1.0;

    if ((k > 0.0) && (k < 4.0)) {
        xx = 0.5 * k - 1.0;
        yy = (((((((((((((((-3.60311230482612224e-13L * xx + 3.46348526554087424e-12L) * xx - 2.99627399604128973e-11L) * xx + 2.57747807106988589e-10L) * xx - 2.09330568435488303e-9L) * xx + 1.59501329936987818e-8L) * xx - 1.13717900285428895e-7L) *
                              xx +
                      7.55292885309152956e-7L) *
                             xx -
                     4.64980751480619431e-6L) *
                            xx +
                    2.63830365675408129e-5L) *
                           xx -
                   1.37089870978830576e-4L) *
                          xx +
                  6.47686503728103400e-4L) *
                         xx -
                 2.76060141343627983e-3L) *
                        xx +
                1.05306034687449505e-2L) *
                       xx -
               3.57191348753631956e-2L) *
                      xx +
              1.07774527938978692e-1L) *
                     xx -
             2.96997075145080963e-1L;
        yy = (yy * xx + 8.64664716763387311e-1L) * xx + 7.42047691268006429e-1L;
        yy = yy - log(k);
        tau = (1.0 - k) * exp(-k) + k * k * yy;
        return tau;
    }
    if ((k > 4.0) && (k < 85.0)) {
        xx = 14.5 / (k + 3.25) - 1.0;
        yy = (((((((((((((((-1.62806570868460749e-12L * xx - 8.95400579318284288e-13L) * xx - 4.08352702838151578e-12L) * xx - 1.45132988248537498e-11L) * xx - 8.35086918940757852e-11L) * xx - 2.13638678953766289e-10L) * xx -
                       1.10302431467069770e-9L) *
                              xx -
                      3.67128915633455484e-9L) *
                             xx -
                     1.66980544304104726e-8L) *
                            xx -
                    6.11774386401295125e-8L) *
                           xx -
                   2.70306163610271497e-7L) *
                          xx -
                  1.05565006992891261e-6L) *
                         xx -
                 4.72090467203711484e-6L) *
                        xx -
                1.95076375089955937e-5L) *
                       xx -
               9.16450482931221453e-5L) *
                      xx -
              4.05892130452128677e-4L) *
                     xx -
             2.14213055000334718e-3L;
        yy = ((yy * xx - 1.06374875116569657e-2L) * xx - 8.50699154984571871e-2L) * xx + 9.23755307807784058e-1L;
        yy = exp(-k) * yy / k;
        tau = (1.0 - k) * exp(-k) + k * k * yy;
        return tau;
    }
    return 0.0;
}

void LeafOptics::setProperties(const std::vector<uint> &UUIDs, const LeafOpticsProperties &leafproperties) {
    for (const auto &data: output_prim_data) {
        if (data == "chlorophyll") {
            context->setPrimitiveData(UUIDs, "chlorophyll", leafproperties.chlorophyllcontent);
        } else if (data == "carotenoid") {
            context->setPrimitiveData(UUIDs, "carotenoid", leafproperties.carotenoidcontent);
        } else if (data == "anthocyanin") {
            context->setPrimitiveData(UUIDs, "anthocyanin", leafproperties.anthocyancontent);
        } else if (data == "brown" && leafproperties.brownpigments > 0.0) {
            context->setPrimitiveData(UUIDs, "brown", leafproperties.brownpigments);
        } else if (data == "water") {
            context->setPrimitiveData(UUIDs, "water", leafproperties.watermass);
        } else if (data == "drymass" && leafproperties.drymass > 0.0) {
            context->setPrimitiveData(UUIDs, "drymass", leafproperties.drymass);
        } else if (data == "protein" && leafproperties.drymass == 0.0) {
            context->setPrimitiveData(UUIDs, "protein", leafproperties.protein);
        } else if (data == "cellulose" && leafproperties.drymass == 0.0) {
            context->setPrimitiveData(UUIDs, "cellulose", leafproperties.carbonconstituents);
        }
    }
}

void LeafOptics::getPropertiesFromSpectrum(const std::vector<uint> &UUIDs) {
    const std::string prefix = "leaf_reflectivity_";

    for (uint UUID: UUIDs) {
        // Check if primitive has reflectivity_spectrum data
        if (!context->doesPrimitiveDataExist(UUID, "reflectivity_spectrum")) {
            continue; // Skip silently if no spectrum data
        }

        // Get the spectrum label
        std::string spectrum_label;
        context->getPrimitiveData(UUID, "reflectivity_spectrum", spectrum_label);

        // Check if this is a LeafOptics-generated spectrum (starts with prefix)
        if (spectrum_label.find(prefix) != 0) {
            continue; // Not a LeafOptics spectrum, skip silently
        }

        // Extract the user label by removing the prefix
        std::string user_label = spectrum_label.substr(prefix.length());

        // Check if we have parameters stored for this label
        auto it = spectrum_parameters_map.find(user_label);
        if (it == spectrum_parameters_map.end()) {
            continue; // No parameters found, skip silently
        }

        // Retrieve the stored parameters
        const LeafOpticsProperties &props = it->second;

        // Assign primitive data using the same logic as setProperties()
        for (const auto &data: output_prim_data) {
            if (data == "chlorophyll") {
                context->setPrimitiveData(UUID, "chlorophyll", props.chlorophyllcontent);
            } else if (data == "carotenoid") {
                context->setPrimitiveData(UUID, "carotenoid", props.carotenoidcontent);
            } else if (data == "anthocyanin") {
                context->setPrimitiveData(UUID, "anthocyanin", props.anthocyancontent);
            } else if (data == "brown" && props.brownpigments > 0.0) {
                context->setPrimitiveData(UUID, "brown", props.brownpigments);
            } else if (data == "water") {
                context->setPrimitiveData(UUID, "water", props.watermass);
            } else if (data == "drymass" && props.drymass > 0.0) {
                context->setPrimitiveData(UUID, "drymass", props.drymass);
            } else if (data == "protein" && props.drymass == 0.0) {
                context->setPrimitiveData(UUID, "protein", props.protein);
            } else if (data == "cellulose" && props.drymass == 0.0) {
                context->setPrimitiveData(UUID, "cellulose", props.carbonconstituents);
            }
        }
    }
}

void LeafOptics::getPropertiesFromSpectrum(uint UUID) {
    getPropertiesFromSpectrum(std::vector<uint>{UUID});
}

void LeafOptics::getPropertiesFromLibrary(const std::string &species, LeafOpticsProperties &leafproperties) {
    // Convert species name to lowercase for case-insensitive lookup
    std::string species_lower = species;
    std::transform(species_lower.begin(), species_lower.end(), species_lower.begin(), ::tolower);

    // Look up species in library
    auto it = species_library.find(species_lower);
    if (it != species_library.end()) {
        // Species found in library
        leafproperties = it->second;
        if (message_flag) {
            std::cout << "Setting Leaf Optics Properties to species: " << species << std::endl;
        }
    } else {
        // Species not found - use default and issue warning
        if (message_flag) {
            std::cerr << "WARNING (LeafOptics): unknown species \"" << species << "\". Using default properties." << std::endl;
        }
        leafproperties = species_library["default"];
    }
}

void LeafOptics::disableMessages() {
    message_flag = false;
}

void LeafOptics::enableMessages() {
    message_flag = true;
}

void LeafOptics::optionalOutputPrimitiveData(const char *label) {
    if (strcmp(label, "chlorophyll") == 0 || strcmp(label, "carotenoid") == 0 || strcmp(label, "anthocyanin") == 0 || strcmp(label, "brown") == 0 || strcmp(label, "water") == 0 || strcmp(label, "drymass") == 0 || strcmp(label, "protein") == 0 ||
        strcmp(label, "cellulose") == 0) {
        output_prim_data.emplace_back(label);
    } else {
        if (message_flag) {
            std::cout << "WARNING (LeafOptics::optionalOutputPrimitiveData): unknown output primitive data " << label << std::endl;
        }
    }
}

// === Nitrogen mode helper methods ===

LeafOpticsProperties LeafOptics::computePropertiesFromNitrogen(float N_area_gN_m2, const LeafOpticsProperties_Nauto &params) {
    LeafOpticsProperties props;

    // Convert nitrogen concentration to chlorophyll content
    // N (g/m2) -> N (ug/cm2): multiply by 100
    // Then apply photosynthetic fraction and empirical coefficient
    float N_area_ug_cm2 = N_area_gN_m2 * 100.0f;
    props.chlorophyllcontent = N_area_ug_cm2 * params.f_photosynthetic * params.N_to_Cab_coefficient;

    // Clamp chlorophyll to physically reasonable range [5, 80] ug/cm2
    props.chlorophyllcontent = std::max(5.0f, std::min(80.0f, props.chlorophyllcontent));

    // Calculate carotenoids from chlorophyll ratio
    props.carotenoidcontent = props.chlorophyllcontent * params.Car_to_Cab_ratio;

    // Copy fixed parameters from Nauto struct
    props.numberlayers = params.numberlayers;
    props.anthocyancontent = params.anthocyancontent;
    props.brownpigments = params.brownpigments;
    props.watermass = params.watermass;
    props.drymass = params.drymass;
    props.protein = params.protein;
    props.carbonconstituents = params.carbonconstituents;

    return props;
}

std::map<uint, std::vector<uint>> LeafOptics::groupPrimitivesByObject(const std::vector<uint> &UUIDs) {
    std::map<uint, std::vector<uint>> object_groups;

    for (uint UUID: UUIDs) {
        if (!context->doesPrimitiveExist(UUID)) {
            if (message_flag) {
                std::cerr << "WARNING (LeafOptics::groupPrimitivesByObject): Primitive UUID " << UUID << " does not exist, skipping." << std::endl;
            }
            continue;
        }

        uint objID = context->getPrimitiveParentObjectID(UUID);

        // Object ID 0 means no parent object
        if (objID == 0) {
            if (message_flag) {
                std::cerr << "WARNING (LeafOptics::groupPrimitivesByObject): Primitive UUID " << UUID << " has no parent object, skipping." << std::endl;
            }
            continue;
        }

        object_groups[objID].push_back(UUID);
    }

    return object_groups;
}

void LeafOptics::createAdaptiveBins(const std::vector<float> &nitrogen_values) {
    nitrogen_bins.clear();

    if (nitrogen_values.empty()) {
        return;
    }

    // Sort nitrogen values for quantile calculation
    std::vector<float> sorted_N = nitrogen_values;
    std::sort(sorted_N.begin(), sorted_N.end());

    uint count = sorted_N.size();
    uint target_bins = nitrogen_params.num_bins;

    // Adjust number of bins if fewer unique values than requested bins
    if (count < target_bins) {
        target_bins = count;
    }

    // Create quantile-based bin centers
    std::vector<float> bin_centers;
    for (uint i = 0; i < target_bins; i++) {
        // Calculate index for quantile center
        uint idx = (i * count / target_bins) + (count / (2 * target_bins));
        if (idx >= count) {
            idx = count - 1;
        }
        float center = sorted_N[idx];

        // Only add if not a duplicate (within tolerance)
        bool is_duplicate = false;
        for (float existing: bin_centers) {
            if (std::abs(center - existing) < 0.001f) {
                is_duplicate = true;
                break;
            }
        }
        if (!is_duplicate) {
            bin_centers.push_back(center);
        }
    }

    // Generate PROSPECT spectrum for each bin
    for (uint i = 0; i < bin_centers.size(); i++) {
        SpectrumBin bin;
        bin.N_center = bin_centers[i];
        bin.spectrum_label = "Nauto_" + std::to_string(i);

        // Compute PROSPECT properties and generate spectrum
        LeafOpticsProperties props = computePropertiesFromNitrogen(bin.N_center, nitrogen_params);
        run(props, bin.spectrum_label);

        nitrogen_bins.push_back(bin);
    }

    if (message_flag) {
        std::cout << "LeafOptics: Created " << nitrogen_bins.size() << " nitrogen-based spectrum bins." << std::endl;
    }
}

uint LeafOptics::findNearestBin(float N_value) {
    if (nitrogen_bins.empty()) {
        helios_runtime_error("ERROR (LeafOptics::findNearestBin): No nitrogen bins have been created.");
    }

    uint best_bin = 0;
    float best_distance = std::abs(N_value - nitrogen_bins[0].N_center);

    for (uint i = 1; i < nitrogen_bins.size(); i++) {
        float distance = std::abs(N_value - nitrogen_bins[i].N_center);
        if (distance < best_distance) {
            best_distance = distance;
            best_bin = i;
        }
    }

    return best_bin;
}

bool LeafOptics::shouldReassign(float current_N, uint current_bin) {
    if (current_bin >= nitrogen_bins.size()) {
        return false;
    }

    float bin_center = nitrogen_bins[current_bin].N_center;
    float absolute_change = std::abs(current_N - bin_center);

    // Check absolute threshold first
    if (absolute_change < nitrogen_params.min_reassignment_change) {
        return false;
    }

    // Check relative threshold
    float relative_change = (bin_center > 0.0f) ? (absolute_change / bin_center) : 0.0f;
    return relative_change > nitrogen_params.reassignment_threshold;
}

bool LeafOptics::isSignificantImprovement(float current_N, uint old_bin, uint new_bin) {
    if (old_bin >= nitrogen_bins.size() || new_bin >= nitrogen_bins.size()) {
        return false;
    }

    float old_distance = std::abs(current_N - nitrogen_bins[old_bin].N_center);
    float new_distance = std::abs(current_N - nitrogen_bins[new_bin].N_center);

    // Require improvement of at least half the minimum change threshold (hysteresis)
    return (old_distance - new_distance) > nitrogen_params.min_reassignment_change * 0.5f;
}

void LeafOptics::assignSpectrumToPrimitives(const std::vector<uint> &UUIDs, uint bin_index) {
    if (bin_index >= nitrogen_bins.size()) {
        helios_runtime_error("ERROR (LeafOptics::assignSpectrumToPrimitives): Invalid bin index " + std::to_string(bin_index));
    }

    const std::string &label = nitrogen_bins[bin_index].spectrum_label;
    std::string refl_label = "leaf_reflectivity_" + label;
    std::string trans_label = "leaf_transmissivity_" + label;

    context->setPrimitiveData(UUIDs, "reflectivity_spectrum", refl_label);
    context->setPrimitiveData(UUIDs, "transmissivity_spectrum", trans_label);

    // Write optional primitive data if any are enabled
    if (!output_prim_data.empty()) {
        // Compute properties from the bin's nitrogen center value
        LeafOpticsProperties props = computePropertiesFromNitrogen(nitrogen_bins[bin_index].N_center, nitrogen_params);

        for (const auto &data: output_prim_data) {
            if (data == "chlorophyll") {
                context->setPrimitiveData(UUIDs, "chlorophyll", props.chlorophyllcontent);
            } else if (data == "carotenoid") {
                context->setPrimitiveData(UUIDs, "carotenoid", props.carotenoidcontent);
            } else if (data == "anthocyanin") {
                context->setPrimitiveData(UUIDs, "anthocyanin", props.anthocyancontent);
            } else if (data == "brown" && props.brownpigments > 0.0) {
                context->setPrimitiveData(UUIDs, "brown", props.brownpigments);
            } else if (data == "water") {
                context->setPrimitiveData(UUIDs, "water", props.watermass);
            } else if (data == "drymass" && props.drymass > 0.0) {
                context->setPrimitiveData(UUIDs, "drymass", props.drymass);
            } else if (data == "protein" && props.drymass == 0.0) {
                context->setPrimitiveData(UUIDs, "protein", props.protein);
            } else if (data == "cellulose" && props.drymass == 0.0) {
                context->setPrimitiveData(UUIDs, "cellulose", props.carbonconstituents);
            }
        }
    }
}

void LeafOptics::run(const std::vector<uint> &UUIDs, const LeafOpticsProperties_Nauto &params) {
    if (UUIDs.empty()) {
        if (message_flag) {
            std::cout << "LeafOptics: Empty UUID list provided to nitrogen mode, nothing to do." << std::endl;
        }
        return;
    }

    // Group primitives by parent object
    std::map<uint, std::vector<uint>> object_groups = groupPrimitivesByObject(UUIDs);

    if (object_groups.empty()) {
        if (message_flag) {
            std::cerr << "WARNING (LeafOptics::run): No valid objects found for nitrogen-based leaf optics." << std::endl;
        }
        return;
    }

    // Collect nitrogen values from all objects
    std::vector<float> nitrogen_values;
    std::map<uint, float> object_nitrogen;

    for (const auto &pair: object_groups) {
        uint objID = pair.first;

        // Check if nitrogen data exists on this object
        if (!context->doesObjectDataExist(objID, "leaf_nitrogen_gN_m2")) {
            helios_runtime_error("ERROR (LeafOptics::run): Object " + std::to_string(objID) + " does not have 'leaf_nitrogen_gN_m2' data. Enable PlantArchitecture nitrogen model first.");
        }

        float N_area;
        context->getObjectData(objID, "leaf_nitrogen_gN_m2", N_area);
        nitrogen_values.push_back(N_area);
        object_nitrogen[objID] = N_area;
    }

    if (!nitrogen_mode_active) {
        // First call: Initialize nitrogen mode
        nitrogen_params = params;

        // Create adaptive bins based on nitrogen distribution
        createAdaptiveBins(nitrogen_values);

        // Assign each object to nearest bin
        for (const auto &pair: object_groups) {
            uint objID = pair.first;
            const std::vector<uint> &obj_UUIDs = pair.second;
            float N_area = object_nitrogen[objID];

            uint best_bin = findNearestBin(N_area);
            assignSpectrumToPrimitives(obj_UUIDs, best_bin);

            // Track assignment
            ObjectAssignment assignment;
            assignment.bin_index = best_bin;
            assignment.N_at_assignment = N_area;
            assignment.primitive_UUIDs = obj_UUIDs;
            object_assignments[objID] = assignment;

            // Track primitive to object mapping
            for (uint UUID: obj_UUIDs) {
                primitive_to_object[UUID] = objID;
            }
        }

        nitrogen_mode_active = true;

        if (message_flag) {
            std::cout << "LeafOptics: Nitrogen mode initialized with " << object_assignments.size() << " leaf objects assigned to " << nitrogen_bins.size() << " spectrum bins." << std::endl;
        }

    } else {
        // Subsequent call: Update assignments

        // Build set of current object IDs
        std::set<uint> current_objects;
        for (const auto &pair: object_groups) {
            current_objects.insert(pair.first);
        }

        // Build set of previously tracked objects
        std::set<uint> tracked_objects;
        for (const auto &pair: object_assignments) {
            tracked_objects.insert(pair.first);
        }

        // Find removed objects (tracked but not in current)
        std::vector<uint> removed_objects;
        for (uint objID: tracked_objects) {
            if (current_objects.find(objID) == current_objects.end()) {
                removed_objects.push_back(objID);
            }
        }

        // Remove tracking for removed objects
        for (uint objID: removed_objects) {
            const ObjectAssignment &assignment = object_assignments[objID];
            for (uint UUID: assignment.primitive_UUIDs) {
                primitive_to_object.erase(UUID);
            }
            object_assignments.erase(objID);
        }

        // Process current objects
        for (const auto &pair: object_groups) {
            uint objID = pair.first;
            const std::vector<uint> &obj_UUIDs = pair.second;
            float N_area = object_nitrogen[objID];

            if (tracked_objects.find(objID) == tracked_objects.end()) {
                // New object: assign to nearest existing bin
                uint best_bin = findNearestBin(N_area);
                assignSpectrumToPrimitives(obj_UUIDs, best_bin);

                ObjectAssignment assignment;
                assignment.bin_index = best_bin;
                assignment.N_at_assignment = N_area;
                assignment.primitive_UUIDs = obj_UUIDs;
                object_assignments[objID] = assignment;

                for (uint UUID: obj_UUIDs) {
                    primitive_to_object[UUID] = objID;
                }

            } else {
                // Existing object: check for reassignment
                ObjectAssignment &assignment = object_assignments[objID];

                // Update primitive list (in case it changed)
                assignment.primitive_UUIDs = obj_UUIDs;
                for (uint UUID: obj_UUIDs) {
                    primitive_to_object[UUID] = objID;
                }

                if (shouldReassign(N_area, assignment.bin_index)) {
                    uint best_bin = findNearestBin(N_area);

                    if (best_bin != assignment.bin_index && isSignificantImprovement(N_area, assignment.bin_index, best_bin)) {
                        assignSpectrumToPrimitives(obj_UUIDs, best_bin);
                        assignment.bin_index = best_bin;
                        assignment.N_at_assignment = N_area;
                    }
                }
            }
        }
    }
}

void LeafOptics::updateNitrogenBasedSpectra() {
    if (!nitrogen_mode_active) {
        helios_runtime_error("ERROR (LeafOptics::updateNitrogenBasedSpectra): Nitrogen mode is not active. "
                             "Call run() with LeafOpticsProperties_Nauto first to initialize nitrogen mode.");
    }

    for (auto &pair: object_assignments) {
        uint objID = pair.first;
        ObjectAssignment &assignment = pair.second;

        // Read current nitrogen value
        if (!context->doesObjectDataExist(objID, "leaf_nitrogen_gN_m2")) {
            if (message_flag) {
                std::cerr << "WARNING (LeafOptics::updateNitrogenBasedSpectra): Object " << objID << " no longer has 'leaf_nitrogen_gN_m2' data, skipping." << std::endl;
            }
            continue;
        }

        float current_N;
        context->getObjectData(objID, "leaf_nitrogen_gN_m2", current_N);

        // Check for reassignment
        if (shouldReassign(current_N, assignment.bin_index)) {
            uint best_bin = findNearestBin(current_N);

            if (best_bin != assignment.bin_index && isSignificantImprovement(current_N, assignment.bin_index, best_bin)) {
                assignSpectrumToPrimitives(assignment.primitive_UUIDs, best_bin);
                assignment.bin_index = best_bin;
                assignment.N_at_assignment = current_N;
            }
        }
    }
}
