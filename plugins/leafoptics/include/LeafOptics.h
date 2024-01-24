/** \file "LeafOptics.h" Primary header file for PROSPECT leaf optics model.

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "Context.h"

//! LeafOptics model class
/** Prospect-D & PRO based computation of spectral total reflectance and transmittance with dimension reduction */

struct LeafOpticsProperties{

    float numberlayers = 1.5;
    float brownpigments = 0.f;
    float chlorophyllcontent = 30.f; //µg.cm-2
    float carotenoidcontent = 7.f; //µg.cm-2
    float anthocyancontent = 1; //µg.cm-2
    float watermass = 0.015f; //g.cm-2
    float drymass = 0.09f; //g.cm-2
    float protein = 0.f; //g.cm-2
    float carbonconstituents = 0.f; //g.cm-2

    //Default values for Prospect-D
    //float N=1.5;  float CHL= 30.0 ;  float CAR = 10.0;  float ANT = 1.0; float Brown=0.0; float EWT= 0.015; float LMA = 0.009;

    //Default values for Prospect-PRO
    // float N=1.5;  float CHL= 40.0 ;  float CAR = 10.0;  float ANT = 0.5; float Brown=0.0; float EWT= 0.015; float pro = 0.001; float carbon = 0.009;

    LeafOpticsProperties(){};

    LeafOpticsProperties(float chlorophyllcontent, float carotenoidcontent, float anthocyancontent, float watermass, float drymass, float protein,
                         float carbonconstituents): chlorophyllcontent(chlorophyllcontent), carotenoidcontent(carotenoidcontent),
                                                    anthocyancontent(anthocyancontent), watermass(watermass), drymass(drymass), protein(protein), carbonconstituents(carbonconstituents){}
};

class LeafOptics{
public:

    //! Constructor
    LeafOptics( helios::Context* a_context );

    //! Self-test
    /**
     * \return 0 if test was successful, 1 if test failed
     */
    static int selfTest();

    //! Run the LeafOptics model to generate reflectivity and transmissivity spectra, create associated global data, and assign to specified primitives.
    /**
      * \param[in] UUIDs: UUIDs for primitives that will be assigned the generated reflectivity and transmissivity spectra.
      * \param[in] leafproperties: LeafOptics properties.
      * \param[in] label: label of spectra that will be created. This label will be appended to "leaf_reflectivity_" and "leaf_transmissivity_" (e.g., "leaf_reflectivity_bean" if the label is "bean").
     */
    void run(const std::vector<uint> &UUIDs, const LeafOpticsProperties &leafproperties, const std::string &label);

    //! Run the LeafOptics model to generate reflectivity and transmissivity spectra and create associated global data
    /**
      * \param[in] leafproperties: LeafOptics properties.
      * \param[in] label: label of spectra that will be created. This label will be appended to "leaf_reflectivity_" and "leaf_transmissivity_" (e.g., "leaf_reflectivity_bean" if the label is "bean").
     */
    void run(const LeafOpticsProperties &leafproperties, const std::string &label);

    //! LeafOptics model kernel
    /**
     * \param[in] numberlayers: number of layers in the leaf
     * \param[in] Chlorophyllcontent: chlorophyll content in the leaf
     * \param[in] carotenoidcontent: carotenoid content in the leaf
     * \param[in] anthocyancontent: anthocyan content in the leaf
     * \param[in] brownpigments: brown pigment content in the leaf
     * \param[in] watermass: water mass in the leaf
     * \param[in] drymass: dry mass in the leaf
     * \param[in] protein: protein content in the leaf
     * \param[in] carbonconstituents: carbon constituents in the leaf
     */
    void PROSPECT(float numberlayers, float Chlorophyllcontent, float carotenoidcontent, float anthocyancontent,
                  float brownpigments, float watermass, float drymass, float protein, float carbonconstituents, std::vector<float> &reflectivities_fit, std::vector<float> &transmissivities_fit );

    //! Get the leaf spectra
    /**
     * \param[in] leafproperties: LeafOptics properties.
     * \param[out] reflectivities_fit: reflectivities of the leaf
     * \param[out] transmissivities_fit: transmissivities of the leaf
     */
    void getLeafSpectra(const LeafOpticsProperties &leafproperties, std::vector<helios::vec2> &reflectivities_fit, std::vector<helios::vec2> &transmissivities_fit);

    //! Set leaf optical properties for a set of primitives
    /**
     * \param[in] UUIDs: UUIDs for primitives for which optical properties should be set.
     * \param[in] leafproperties: LeafOptics properties.
     */
    void setProperties(const std::vector<uint> &UUIDs, const LeafOpticsProperties &leafproperties);

    //! Disable command-line output messages from this plug-in
    void disableMessages();

    //! Enable command-line output messages from this plug-in
    void enableMessages();

private:

    std::vector<float> R_spec_normal, R_spec_diffuse, wave_length, Rtotal, Ttotal ;

    //!  400...2500 nm fixed wavelength range of input spectra specifying refractive index,...,absorption_drymass
    const uint nw  = 2101;

    //! Copy of a pointer to the context
    helios::Context* context;
    //! Default input values for grasses (*_g) and non-grasses (estimated medians from Lopex and Angers datasets)
    //! These are used for gap-filling for primitives with missing input values
    float chlorophyll_default               = 43.0;    //! Total chlorophyll (micro_g/cm^2)
    float chlorophyll_default_grass         = 57.0;
    float carotenoid2chlorophyll            = 0.23;    //! Carotenoid to Chlorophyll ratio
    float carotenoid2chlorophyll_grass      = 0.21;
    float anthocyanin2carotenoid            = 0.06;    //! Anthocyanin to Carotenoid  ratio
    float anthocyanin2carotenoid_grass      = 0.006;
    float drymass2chlorophyll               = 1.2669e-04; //! Leaf mass per area to Chlorophyll ratio
    float drymass2chlorophyll_grass         = 9.2196e-05;
    float watercontent                      = 0.9; //!  water content

    //
    std::vector<float>  refractiveindex, absorption_chlorophyll,absorption_carotenoid, absorption_anthocyanin, absorption_brown,absorption_water, absorption_drymass;
    std::vector<float>  absorption_protein, absorption_carbonconstituents;
    std::vector<float> Rcof, Tcof;
    //  std::vector<float> R_spec_normal, R_spec_diffuse, wave_length;

    float transmittance(double k);
    // Computes the diffuse transmittance through an elementary layer

    void surface(float degree, std::vector<float> &reflectivities);
    // Computes surface reflectances for normal and diffuse light incidence

    bool message_flag = true;

};
