/** \file "LeafOptics.h" Primary header file for PROSPECT leaf optics model.

    Copyright (C) 2016-2026 Brian Bailey

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

struct LeafOpticsProperties {

    float numberlayers = 1.5;
    float brownpigments = 0.f;
    float chlorophyllcontent = 30.f; // µg.cm-2
    float carotenoidcontent = 7.f; // µg.cm-2
    float anthocyancontent = 1; // µg.cm-2
    float watermass = 0.015f; // g.cm-2
    float drymass = 0.09f; // g.cm-2
    float protein = 0.f; // g.cm-2
    float carbonconstituents = 0.f; // g.cm-2

    // Default values for Prospect-D
    // float N=1.5;  float CHL= 30.0 ;  float CAR = 10.0;  float ANT = 1.0; float Brown=0.0; float EWT= 0.015; float LMA = 0.009;

    // Default values for Prospect-PRO
    //  float N=1.5;  float CHL= 40.0 ;  float CAR = 10.0;  float ANT = 0.5; float Brown=0.0; float EWT= 0.015; float pro = 0.001; float carbon = 0.009;

    LeafOpticsProperties() {};

    LeafOpticsProperties(float chlorophyllcontent, float carotenoidcontent, float anthocyancontent, float watermass, float drymass, float protein, float carbonconstituents) :
        chlorophyllcontent(chlorophyllcontent), carotenoidcontent(carotenoidcontent), anthocyancontent(anthocyancontent), watermass(watermass), drymass(drymass), protein(protein), carbonconstituents(carbonconstituents) {
    }
};

//! Parameters for nitrogen-based automatic leaf optics calculation
/**
 * When passed to LeafOptics::run(), chlorophyll and carotenoid content are automatically
 * computed from leaf nitrogen data ("leaf_nitrogen_gN_m2" object data). All other PROSPECT
 * parameters use the values specified in this structure.
 *
 * The conversion from nitrogen to chlorophyll follows:
 * - Cab (ug/cm2) = N_area (g/m2) * 100 * f_photosynthetic * N_to_Cab_coefficient
 * - Car (ug/cm2) = Cab * Car_to_Cab_ratio
 *
 * Adaptive binning groups leaves by nitrogen content to limit the number of unique spectra,
 * improving computational efficiency in radiation calculations.
 */
struct LeafOpticsProperties_Nauto {

    // === Nitrogen-to-pigment conversion coefficients ===

    //! Fraction of leaf nitrogen in photosynthetic machinery [0-1]
    float f_photosynthetic = 0.50f;

    //! Empirical coefficient for chlorophyll calculation
    /** Cab (ug/cm2) = N_area (g/m2) * 100 * f_photosynthetic * N_to_Cab_coefficient */
    float N_to_Cab_coefficient = 0.40f;

    //! Carotenoid to chlorophyll ratio
    /** Car = Cab * Car_to_Cab_ratio */
    float Car_to_Cab_ratio = 0.25f;

    // === Fixed PROSPECT parameters (not derived from nitrogen) ===

    float numberlayers = 1.5f; //!< Leaf structure parameter (N)
    float anthocyancontent = 1.0f; //!< Anthocyanin content (ug/cm2)
    float brownpigments = 0.0f; //!< Brown pigment content (senescence indicator)
    float watermass = 0.015f; //!< Equivalent water thickness (g/cm2)
    float drymass = 0.006f; //!< Leaf mass per area (g/cm2)
    float protein = 0.0f; //!< Protein content (g/cm2) - for PROSPECT-PRO
    float carbonconstituents = 0.0f; //!< Carbon constituents (g/cm2) - for PROSPECT-PRO

    // === Adaptive binning configuration ===

    //! Target number of spectrum bins
    /** Actual number may be fewer if leaf nitrogen values cluster tightly */
    uint num_bins = 20;

    //! Relative nitrogen change threshold for reassignment
    /** Leaf must change by this fraction from bin center to be considered for reassignment */
    float reassignment_threshold = 0.30f;

    //! Minimum absolute nitrogen change for reassignment (g/m2)
    /** Prevents reassignment for small absolute changes even if relative change is large */
    float min_reassignment_change = 0.3f;
};

class LeafOptics {
public:
    //! Constructor
    LeafOptics(helios::Context *a_context);

    //! Self-test
    /**
     * \return 0 if test was successful, 1 if test failed
     */
    static int selfTest(int argc = 0, char **argv = nullptr);

    //! Run the LeafOptics model to generate reflectivity and transmissivity spectra, create associated global data, and assign to specified primitives.
    /**
     * \param[in] UUIDs UUIDs for primitives that will be assigned the generated reflectivity and transmissivity spectra.
     * \param[in] leafproperties LeafOptics properties.
     * \param[in] label label of spectra that will be created. This label will be appended to "leaf_reflectivity_" and "leaf_transmissivity_" (e.g., "leaf_reflectivity_bean" if the label is "bean").
     */
    void run(const std::vector<uint> &UUIDs, const LeafOpticsProperties &leafproperties, const std::string &label);

    //! Run the LeafOptics model to generate reflectivity and transmissivity spectra and create associated global data
    /**
     * \param[in] leafproperties LeafOptics properties.
     * \param[in] label label of spectra that will be created. This label will be appended to "leaf_reflectivity_" and "leaf_transmissivity_" (e.g., "leaf_reflectivity_bean" if the label is "bean").
     */
    void run(const LeafOpticsProperties &leafproperties, const std::string &label);

    //! LeafOptics model kernel
    /**
     * \param[in] numberlayers number of layers in the leaf
     * \param[in] Chlorophyllcontent chlorophyll content in the leaf
     * \param[in] carotenoidcontent carotenoid content in the leaf
     * \param[in] anthocyancontent anthocyan content in the leaf
     * \param[in] brownpigments brown pigment content in the leaf
     * \param[in] watermass water mass in the leaf
     * \param[in] drymass dry mass in the leaf
     * \param[in] protein protein content in the leaf
     * \param[in] carbonconstituents carbon constituents in the leaf
     */
    void PROSPECT(float numberlayers, float Chlorophyllcontent, float carotenoidcontent, float anthocyancontent, float brownpigments, float watermass, float drymass, float protein, float carbonconstituents, std::vector<float> &reflectivities_fit,
                  std::vector<float> &transmissivities_fit);

    //! Get the leaf spectra
    /**
     * \param[in] leafproperties LeafOptics properties.
     * \param[out] reflectivities_fit reflectivities of the leaf
     * \param[out] transmissivities_fit transmissivities of the leaf
     */
    void getLeafSpectra(const LeafOpticsProperties &leafproperties, std::vector<helios::vec2> &reflectivities_fit, std::vector<helios::vec2> &transmissivities_fit);

    //! Set leaf optical properties for a set of primitives
    /**
     * \param[in] UUIDs UUIDs for primitives for which optical properties should be set.
     * \param[in] leafproperties LeafOptics properties.
     */
    void setProperties(const std::vector<uint> &UUIDs, const LeafOpticsProperties &leafproperties);

    //! Get PROSPECT parameters from reflectivity spectrum for a set of primitives
    /**
     * \param[in] UUIDs UUIDs for primitives to query. For each primitive, this method retrieves the "reflectivity_spectrum" primitive data and checks if it matches a spectrum generated by this LeafOptics instance. If a match is found, the
     * corresponding PROSPECT model parameters are assigned as primitive data using the same labels as setProperties() ("chlorophyll", "carotenoid", "anthocyanin", "brown", "water", "drymass", "protein", "cellulose"). Primitives without matching
     * spectra are silently skipped.
     */
    void getPropertiesFromSpectrum(const std::vector<uint> &UUIDs);

    //! Get PROSPECT parameters from reflectivity spectrum for a single primitive
    /**
     * \param[in] UUID UUID for primitive to query. This method retrieves the "reflectivity_spectrum" primitive data and checks if it matches a spectrum generated by this LeafOptics instance. If a match is found, the corresponding PROSPECT model
     * parameters are assigned as primitive data using the same labels as setProperties() ("chlorophyll", "carotenoid", "anthocyanin", "brown", "water", "drymass", "protein", "cellulose"). If no matching spectrum is found, the primitive is silently
     * skipped.
     */
    void getPropertiesFromSpectrum(uint UUID);

    //! Get leaf optical properties from the built-in species library
    /**
     * \param[in] species Name of the species to retrieve properties for (e.g., "default"). Species names are case-insensitive.
     * \param[out] leafproperties LeafOpticsProperties struct to be populated with the species-specific properties. If the species is not found in the library, default properties are used and a warning is issued.
     */
    void getPropertiesFromLibrary(const std::string &species, LeafOpticsProperties &leafproperties);

    //! Disable command-line output messages from this plug-in
    void disableMessages();

    //! Enable command-line output messages from this plug-in
    void enableMessages();

    //! Add optional primitive data output
    /**
     * \param[in] label Label of the primitive data to output. Available labels: "chlorophyll", "carotenoid", "anthocyanin", "brown", "water", "drymass", "protein", "cellulose".
     */
    void optionalOutputPrimitiveData(const char *label);

    //! Run nitrogen-based leaf optics for the specified primitives
    /**
     * Computes leaf optical properties based on nitrogen content. Groups primitives by parent
     * object, reads "leaf_nitrogen_gN_m2" from each object, and assigns spectra accordingly.
     *
     * Behavior depends on call sequence:
     * - First call: Creates adaptive spectrum bins based on nitrogen distribution, generates
     *   PROSPECT spectra for each bin, assigns primitives to bins.
     * - Subsequent calls: New primitives assigned to nearest existing bin, removed primitives
     *   untracked, existing primitives checked for nitrogen changes and reassigned if threshold exceeded.
     *
     * All primitives belonging to the same parent object receive the same spectrum.
     *
     * \param[in] UUIDs Primitive UUIDs of leaf surfaces
     * \param[in] params Nitrogen-auto parameters specifying conversion coefficients, fixed PROSPECT values, and binning configuration
     *
     * \note Requires "leaf_nitrogen_gN_m2" object data to be set on each leaf object (typically by PlantArchitecture::NitrogenModel)
     */
    void run(const std::vector<uint> &UUIDs, const LeafOpticsProperties_Nauto &params);

    //! Update nitrogen-based spectra for currently tracked primitives
    /**
     * Efficient update method for when nitrogen values have changed but the primitive set is
     * unchanged. Re-reads nitrogen from parent objects and reassigns primitives if the change
     * exceeds configured thresholds.
     *
     * This is equivalent to calling run() with the same UUIDs but faster since it skips
     * primitive grouping and add/remove processing.
     *
     * \note Nitrogen mode must be active (run() called with LeafOpticsProperties_Nauto)
     */
    void updateNitrogenBasedSpectra();

private:
    std::vector<float> R_spec_normal, R_spec_diffuse, wave_length, Rtotal, Ttotal;

    //!  400...2500 nm fixed wavelength range of input spectra specifying refractive index,...,absorption_drymass
    const uint nw = 2101;

    //! Copy of a pointer to the context
    helios::Context *context;
    //! Map to track spectrum labels and their corresponding PROSPECT parameters
    std::map<std::string, LeafOpticsProperties> spectrum_parameters_map;
    //! Built-in species library containing PROSPECT-D parameters for common plant species
    std::map<std::string, LeafOpticsProperties> species_library;
    //! Initialize the species library with PROSPECT-D parameters from LOPEX93 dataset
    void initializeSpeciesLibrary();
    //! Default input values for grasses (*_g) and non-grasses (estimated medians from Lopex and Angers datasets)
    //! These are used for gap-filling for primitives with missing input values
    float chlorophyll_default = 43.0; //! Total chlorophyll (micro_g/cm^2)
    float chlorophyll_default_grass = 57.0;
    float carotenoid2chlorophyll = 0.23; //! Carotenoid to Chlorophyll ratio
    float carotenoid2chlorophyll_grass = 0.21;
    float anthocyanin2carotenoid = 0.06; //! Anthocyanin to Carotenoid  ratio
    float anthocyanin2carotenoid_grass = 0.006;
    float drymass2chlorophyll = 1.2669e-04; //! Leaf mass per area to Chlorophyll ratio
    float drymass2chlorophyll_grass = 9.2196e-05;
    float watercontent = 0.9; //!  water content

    //
    std::vector<float> refractiveindex, absorption_chlorophyll, absorption_carotenoid, absorption_anthocyanin, absorption_brown, absorption_water, absorption_drymass;
    std::vector<float> absorption_protein, absorption_carbonconstituents;
    std::vector<float> Rcof, Tcof;
    //  std::vector<float> R_spec_normal, R_spec_diffuse, wave_length;

    float transmittance(double k);
    // Computes the diffuse transmittance through an elementary layer

    void surface(float degree, std::vector<float> &reflectivities);
    // Computes surface reflectances for normal and diffuse light incidence

    bool message_flag = true;

    //! Names of additional primitive data to add to the Context
    std::vector<std::string> output_prim_data;

    // === Nitrogen mode state ===

    //! Flag indicating nitrogen-based automatic mode is active
    bool nitrogen_mode_active = false;

    //! Stored parameters for nitrogen mode
    LeafOpticsProperties_Nauto nitrogen_params;

    //! Structure representing a spectrum bin for nitrogen-based mode
    struct SpectrumBin {
        float N_center; //!< Bin center nitrogen value (g/m2)
        std::string spectrum_label; //!< Label suffix (e.g., "Nauto_0")
    };

    //! Bins for nitrogen-based spectrum assignment
    std::vector<SpectrumBin> nitrogen_bins;

    //! Structure tracking object assignment to spectrum bins
    struct ObjectAssignment {
        uint bin_index; //!< Current bin assignment
        float N_at_assignment; //!< Nitrogen when assigned (for change detection)
        std::vector<uint> primitive_UUIDs; //!< All primitives in this object
    };

    //! Map from object ID to its bin assignment (objID -> assignment)
    std::map<uint, ObjectAssignment> object_assignments;

    //! Reverse lookup from primitive UUID to parent object ID
    std::map<uint, uint> primitive_to_object;

    // === Nitrogen mode helper methods ===

    //! Convert nitrogen concentration to PROSPECT parameters
    LeafOpticsProperties computePropertiesFromNitrogen(float N_area_gN_m2, const LeafOpticsProperties_Nauto &params);

    //! Group primitive UUIDs by their parent object ID
    std::map<uint, std::vector<uint>> groupPrimitivesByObject(const std::vector<uint> &UUIDs);

    //! Create adaptive bins based on nitrogen value distribution
    void createAdaptiveBins(const std::vector<float> &nitrogen_values);

    //! Find the nearest bin index for a given nitrogen value
    uint findNearestBin(float N_value);

    //! Check if an object should be reassigned based on nitrogen change
    bool shouldReassign(float current_N, uint current_bin);

    //! Check if moving to a new bin represents significant improvement (hysteresis)
    bool isSignificantImprovement(float current_N, uint old_bin, uint new_bin);

    //! Assign spectrum labels to primitives based on bin index
    void assignSpectrumToPrimitives(const std::vector<uint> &UUIDs, uint bin_index);
};
