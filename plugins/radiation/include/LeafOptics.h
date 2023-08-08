/** \file "LeafOptics.h" Primary header file for LeafOptics model.
    \author Jan Graefe, Tong Lei
**/

#include "Context.h"
#include <vector>

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
    LeafOptics( helios::Context* context );
    //! Constructor


    //! Run the LeafOptics model for specified primitives.
    /**
      * \param[in] UUIDs: UUIDs for primitives that should be included in LeafOptics calculations.
      * \param[in] leafproperties: LeafOptics properties.
      * \param[in] label: mark of spectra (e.g., chl_20).
     */
    void run( std::vector<uint> &UUIDs, const LeafOpticsProperties &leafproperties, const std::string &label);

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
    void prospect(float numberlayers, float Chlorophyllcontent, float carotenoidcontent, float anthocyancontent,
                  float brownpigments, float watermass, float drymass, float protein, float carbonconstituents, std::vector<float> &reflectivities_fit, std::vector<float> &transmissivities_fit );

    //! Get the leaf spectra
    /**
     * \param[inout] reflectivities_fit: reflectivities of the leaf
     * \param[inout] transmissivities_fit: transmissivities of the leaf
     * \param[in] leafproperties: LeafOptics properties.
     */
    void getLeafSpectra(std::vector<helios::vec2> &reflectivities_fit, std::vector<helios::vec2> &transmissivities_fit, const LeafOpticsProperties &leafproperties);

    void setProperties(std::vector<uint> UUIDs,  const LeafOpticsProperties &leafproperties);



private:

    std::vector<float> R_spec_normal, R_spec_diffuse, wave_length, Rtotal, Ttotal ;

    //!  400...2500 nm fixed wavelength range of input spectra specifing refractive index,...,absorption_drymass
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


};
